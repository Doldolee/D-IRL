"""Observed-state trajectory diagnostics: TA-Adv and TA-dSNIPS.

The expected-Q baseline has no explicit entropy term. The normalized DQI
contrast uses per-step action ratios and is separate from sequential OPE.
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch
from . import data
from .agent import ReferencePolicy
from .data import ROOT
from .utils import write_json, log_kernel_density_adaptive


@dataclass(frozen=True)
class TAConfig:
    action_draws: int = 100
    seed: int = 42
    batch_size: int = 256
    gamma_eval: float = 1.0
    bandwidth: float = 0.2
    kernel_nodes: int = 32
    kernel_batch_size: int = 512
    log_ratio_clip: float = 10.0
    bootstrap_draws: int = 1000


TA_CONFIG = TAConfig()
K = TA_CONFIG.action_draws
SEED = TA_CONFIG.seed
BATCH = TA_CONFIG.batch_size
GROUPS = {"senior": "ODA", "ordinary": "YDA"}
REFERENCE_LABELS = ("ODA", "YDA")
SCENARIOS = ("ALL", "0", "1", "2", "3", "4")


# Normalized trajectory arithmetic.
def _vector(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric vector.") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    return array


def _normalize_logs(log_values):
    with np.errstate(under="ignore", over="ignore"):
        values = np.exp(log_values - np.max(log_values))
    return values / values.sum(dtype=np.float64)


def normalized_ta(
    rewards, weights=None, gamma_eval=TA_CONFIG.gamma_eval, *, log_weights=None
):
    """Return a normalized mean-reward difference and weight diagnostics.

    For ``q_t = gamma_eval**t / sum_t gamma_eval**t`` and
    ``p_t = q_t * w_t / sum_t q_t * w_t``, the score is
    ``sum_t q_t * r_t - sum_t p_t * r_t``. By default every time step has
    equal temporal weight. A gamma below one emphasizes the start of a drive.

    Parameters
    ----------
    rewards : one-dimensional array-like
        Finite per-step rewards for one trajectory, in chronological order.
    weights : one-dimensional array-like, optional
        Nonnegative, finite action density ratios, with at least one positive
        entry. Its length must match rewards. Multiplying all ratios by the
        same positive constant does not affect the result.
    gamma_eval : float, default 1.0
        Evaluation discount, satisfying 0 < gamma_eval <= 1.
    log_weights : one-dimensional array-like, keyword-only, optional
        Natural logarithms of the ratios. Supply exactly one of weights and
        log_weights. Negative infinity denotes zero support; NaN and positive
        infinity are invalid. Log-space input avoids ratio overflow. No
        clipping or smoothing is performed in this function.

    Returns
    -------
    dict[str, float]
        ``score``, ``observed_mean``, ``reference_mean``,
        ``effective_sample_size`` (1 / sum(p**2)), ``relative_ess`` (ESS / T),
        ``max_normalized_weight`` (max(p)), and ``temporal_ess``
        (1 / sum(q**2)). ESS is a weight-concentration diagnostic; it does not
        account for temporal dependence between samples.
    """
    reward = _vector(rewards, "rewards")
    if not np.isfinite(reward).all():
        raise ValueError("rewards must contain only finite values.")
    if np.ndim(gamma_eval) != 0 or np.iscomplexobj(gamma_eval):
        raise ValueError("gamma_eval must be a scalar in (0, 1].")
    try:
        gamma = float(gamma_eval)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("gamma_eval must be a scalar in (0, 1].") from exc
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("gamma_eval must be a finite scalar in (0, 1].")
    if (weights is None) == (log_weights is None):
        raise ValueError("Supply exactly one of weights and log_weights.")
    if log_weights is None:
        ratio = _vector(weights, "weights")
        if not np.isfinite(ratio).all() or np.any(ratio < 0):
            raise ValueError("weights must be finite and nonnegative.")
        if not np.any(ratio > 0):
            raise ValueError("weights must have at least one positive value.")
        with np.errstate(divide="ignore"):
            log_ratio = np.log(ratio)
    else:
        log_ratio = _vector(log_weights, "log_weights")
        if np.isnan(log_ratio).any() or np.isposinf(log_ratio).any():
            raise ValueError("log_weights may be finite or negative infinity only.")
        if not np.isfinite(log_ratio).any():
            raise ValueError("log_weights must have at least one finite value.")
    if log_ratio.shape != reward.shape:
        raise ValueError("rewards and weights must have the same length.")
    log_temporal = np.arange(reward.size, dtype=np.float64) * np.log(gamma)
    temporal = _normalize_logs(log_temporal)
    reference = _normalize_logs(log_temporal + log_ratio)
    observed_mean = float(np.dot(temporal, reward))
    reference_mean = float(np.dot(reference, reward))
    ess = float(1.0 / np.dot(reference, reference))
    return {
        "score": observed_mean - reference_mean,
        "observed_mean": observed_mean,
        "reference_mean": reference_mean,
        "effective_sample_size": ess,
        "relative_ess": ess / reward.size,
        "max_normalized_weight": float(reference.max()),
        "temporal_ess": float(1.0 / np.dot(temporal, temporal)),
    }


def critic_difference(observed_q, expected_reference_q):
    observed_q = np.asarray(observed_q, dtype=np.float64)
    reference_q = np.asarray(expected_reference_q, dtype=np.float64)
    if observed_q.shape != reference_q.shape:
        raise ValueError("Observed and expected Q arrays must have the same shape")
    if not np.isfinite(observed_q).all() or not np.isfinite(reference_q).all():
        raise ValueError("Q arrays must be finite")
    return observed_q - reference_q


def kernel_log_weights(log_reference, log_behaviour, log_clip=TA_CONFIG.log_ratio_clip):
    reference = np.asarray(log_reference, dtype=np.float64)
    behaviour = np.asarray(log_behaviour, dtype=np.float64)
    if (
        reference.shape != behaviour.shape
        or not np.isfinite(reference).all()
        or (not np.isfinite(behaviour).all())
    ):
        raise ValueError("Kernel log densities must be finite and have matching shapes")
    if not np.isfinite(log_clip) or log_clip <= 0:
        raise ValueError("log_clip must be positive and finite")
    return np.clip(reference - behaviour, -log_clip, log_clip)


def normalized_dqi_contrast(rewards, log_weights, gamma_eval=TA_CONFIG.gamma_eval):
    return normalized_ta(rewards, log_weights=log_weights, gamma_eval=gamma_eval)


# Reference expectations and fixed-policy inference.
@torch.inference_mode()
def expected_reference_q(model, s, mu, std, z):
    """Independent Gaussian draws per state; minimum taken before averaging."""
    n, k, _ = z.shape
    actions = (
        (mu[:, None, :] + std[:, None, :] * torch.from_numpy(z))
        .tanh()
        .clamp(-1 + 1e-06, 1 - 1e-06)
    )
    states = s[:, None, :].expand(-1, k, -1).reshape(-1, 51)
    q = model.q(states, actions.reshape(-1, 3)).reshape(n, k).numpy().astype(np.float64)
    return (q.mean(axis=1), q.var(axis=1, ddof=1))


@torch.inference_mode()
def trajectory_ta_components(model, states, actions, rewards, log_behavior, seed_parts):
    """Score one drive with the current fixed equations, without fitting anything."""
    states, actions = (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )
    n = len(states)
    if states.shape != (n, 51) or actions.shape != (n, 3) or n == 0:
        raise ValueError("Expected nonempty 51D states and 3D actions")
    if (
        not np.isfinite(states).all()
        or not np.isfinite(actions).all()
        or np.max(np.abs(actions)) > 1
    ):
        raise ValueError("Invalid states or bounded actions")
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    observed, expected, variance, means, scales = [], [], [], [], []
    for start in range(0, n, TA_CONFIG.batch_size):
        s = torch.from_numpy(states[start : start + TA_CONFIG.batch_size])
        a = torch.from_numpy(actions[start : start + TA_CONFIG.batch_size])
        mu, std = model.parameters(s)
        z = rng.standard_normal((len(s), TA_CONFIG.action_draws, 3), dtype=np.float32)
        mean, var = expected_reference_q(model, s, mu, std, z)
        observed.append(model.q(s, a).numpy().astype(np.float64))
        expected.append(mean)
        variance.append(var)
        means.append(mu.numpy())
        scales.append(std.numpy())
    observed, expected, variance = map(np.concatenate, (observed, expected, variance))
    log_reference = log_kernel_density_adaptive(
        np.concatenate(means),
        np.concatenate(scales),
        actions,
        TA_CONFIG.bandwidth,
        nodes=TA_CONFIG.kernel_nodes,
        batch_size=TA_CONFIG.kernel_batch_size,
    )
    log_weights = kernel_log_weights(log_reference, log_behavior)
    contrast = normalized_dqi_contrast(np.asarray(rewards).reshape(-1), log_weights)
    difference = critic_difference(observed, expected)
    row = dict(
        transitions=n,
        ta_adv=float(difference.mean()),
        observed_q=float(observed.mean()),
        expected_reference_q=float(expected.mean()),
        ta_adv_mc_se=float(np.sqrt(variance.sum() / TA_CONFIG.action_draws) / n),
        ta_dsnips=contrast.pop("score"),
        clipped_fraction=float(
            np.mean(np.abs(log_reference - log_behavior) > TA_CONFIG.log_ratio_clip)
        ),
        **contrast,
    )
    components = dict(
        observed_q=observed,
        expected_q=expected,
        action_q_variance=variance,
        ta_adv=difference,
        log_reference=log_reference,
        log_behavior=np.asarray(log_behavior),
        log_weights=log_weights,
    )
    return row, components


@torch.inference_mode()
def recovered_reward_components(model, states, actions, next_states, dones, seed_parts):
    """Compute native-alpha soft Bellman rewards with 64 shared MC draws/state."""
    from torch.nn import functional as F
    from math import log
    rng = np.random.default_rng(np.random.SeedSequence(seed_parts))
    rewards = []
    for start in range(0, len(states), TA_CONFIG.batch_size):
        sl = slice(start, start + TA_CONFIG.batch_size)
        s, a, sn = [torch.as_tensor(v[sl], dtype=torch.float32)
                    for v in (states, actions, next_states)]
        mu, sd = model.parameters(sn)
        z = torch.from_numpy(rng.standard_normal((len(sn), 64, 3), dtype=np.float32))
        latent = mu[:, None] + sd[:, None] * z
        sampled = latent.tanh().clamp(-1 + 1e-6, 1 - 1e-6)
        jac = 2 * log(2) - F.softplus(2 * latent) - F.softplus(-2 * latent)
        logp = (torch.distributions.Normal(mu[:, None], sd[:, None]).log_prob(latent) - jac).sum(-1)
        expanded = sn[:, None].expand(-1, 64, -1).reshape(-1, 51)
        q = model.q(expanded, sampled.reshape(-1, 3)).reshape(len(sn), 64)
        value = (q - model.alpha * logp).numpy().astype(float).mean(1)
        observed = model.q(s, a).numpy().astype(float)
        rewards.append(observed - model.gamma * (1 - np.asarray(dones[sl]).reshape(-1)) * value)
    return np.concatenate(rewards)


def compute_public_ta(output_dir, limit=None, recover=False):
    """Re-evaluate the released synthetic trajectories using bundled frozen models."""
    from .utils import new_output
    output = new_output(output_dir)
    (output / "components").mkdir()
    torch.set_num_threads(2)
    selected = data.records()
    if limit is not None:
        if not 1 <= limit <= len(selected):
            raise ValueError("limit must be between 1 and 24")
        selected = selected[:limit]
    models = {label: ReferencePolicy(ROOT, group) for group, label in GROUPS.items()}
    result = []
    for record in selected:
        drive = data.read_drive(record)
        behavior = data.behavior_policy(record["behavior"])
        mu, sd = data.all_parameters(behavior, drive["generator_states"])
        logb = log_kernel_density_adaptive(mu.numpy(), sd.numpy(), drive["actions"],
                    TA_CONFIG.bandwidth, nodes=TA_CONFIG.kernel_nodes,
                    batch_size=TA_CONFIG.kernel_batch_size)
        for label, model in models.items():
            row, components = trajectory_ta_components(model, drive["states"], drive["actions"],
                    drive["reward_total"], logb, record["ta_seed_parts"])
            if recover:
                components["recovered_reward"] = recovered_reward_components(model,
                    drive["states"], drive["actions"], drive["next_states"], drive["dones"],
                    record["recovery_seed_parts"])
            np.savez_compressed(output / "components" / f"drive{record['id']:03d}_{label}.npz", **components)
            result.append(dict(id=record["id"], source_group=record["source_group"], reference=label, **row))
        write_json(output / "scores.json", result)
        print(f"Scored {record['id'] + 1}/{len(selected)} synthetic trajectories", flush=True)
    write_json(output / "protocol.json", dict(synthetic=True, n_trajectories=len(selected),
                config=asdict(TA_CONFIG), reference_generation="fixed", recovered_reward_draws=64 if recover else None,
                behavior_density="Bundled frozen synthetic behavior models; no refitting in this command"))
    return result

