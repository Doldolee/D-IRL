"""Off-policy estimators and independent FQE for the released synthetic inputs."""

from dataclasses import dataclass
import copy
import math

import numpy as np
import torch
from torch import nn
from scipy.special import ndtri
from scipy.stats import qmc

from .data import load_fqe_data, load_fqe_parameters
from .utils import new_output, write_json


@dataclass(frozen=True)
class OPEConfig:
    gamma: float = 0.99
    dr_cap: float = 5.0
    evaluator_seeds: tuple = (101, 202, 303)
    bootstrap_draws: int = 20000
    bootstrap_seed: int = 42
    fqe_iterations: int = 600
    fqe_updates: int = 50
    fqe_batch_size: int = 256
    fqe_lr: float = 3e-4
    fqe_action_samples: int = 4
    reward_scale: float = 0.01


OPE_CONFIG = OPEConfig()
GAMMA = OPE_CONFIG.gamma
CAP = OPE_CONFIG.dr_cap
B = OPE_CONFIG.bootstrap_draws
EVALUATOR_SEEDS = OPE_CONFIG.evaluator_seeds


def bootstrap_ci(
    per_drive,
    indices=None,
    *,
    repeats=OPE_CONFIG.bootstrap_draws,
    seed=OPE_CONFIG.bootstrap_seed,
):
    """Whole-drive percentile CI, optionally using paired frozen draw indices."""
    values = np.asarray(per_drive, dtype=np.float64)
    if values.ndim not in (1, 2) or not len(values) or not np.isfinite(values).all():
        raise ValueError("per_drive must be a nonempty finite vector or matrix")
    if indices is None:
        indices = np.random.default_rng(seed).integers(
            len(values), size=(repeats, len(values))
        )
    draws = values[indices].mean(axis=1)
    return draws, np.quantile(draws, [0.025, 0.975], axis=0)


def evaluate_dr(
    rewards,
    dones,
    log_ratios,
    q_observed,
    next_v,
    initial_value,
    *,
    gamma=GAMMA,
    cap=CAP,
):
    """One trajectory's DR heads with a cap on raw cumulative action ratios."""
    rewards, q_observed, next_v = [
        np.asarray(x, dtype=float) for x in (rewards, q_observed, next_v)
    ]
    dones = np.asarray(dones, dtype=float).reshape(-1)
    logs = np.asarray(log_ratios, dtype=float).reshape(-1)
    if (
        rewards.ndim != 2
        or q_observed.shape != rewards.shape
        or next_v.shape != rewards.shape
    ):
        raise ValueError("Reward and Q arrays must have matching (T,heads) shapes")
    if len(dones) != len(rewards) or len(logs) != len(rewards):
        raise ValueError("Every transition requires one done flag and log ratio")
    weights = np.exp(np.minimum(np.cumsum(logs, dtype=float), math.log(cap)))
    discount = gamma ** np.arange(len(rewards))
    delta = rewards - q_observed + gamma * (1 - dones[:, None]) * next_v
    terms = discount[:, None] * weights[:, None] * delta
    return np.asarray(initial_value, dtype=float) + np.asarray(
        [math.fsum(terms[:, head].tolist()) for head in range(rewards.shape[1])]
    )


def _normalize_importance_weights(logs):
    shifted = logs - np.max(logs, axis=-1, keepdims=True)
    terms = np.exp(shifted)
    return terms / terms.sum(axis=-1, keepdims=True)


def evaluate_kwis(log_weights, returns):
    """Self-normalized mean of trajectory returns and their normalized weights."""
    weights = _normalize_importance_weights(np.asarray(log_weights, dtype=float))
    return weights @ np.asarray(returns, dtype=float), weights


# Independent FQE fitting and policy expectations.
def _fqe_tensors(d):
    return {
        k: torch.from_numpy(d[k].reshape(-1, 1) if k == "dones" else d[k])
        for k in ("states", "next_states", "actions", "dones", "rewards")
    }


def _fqe_noise(k, seed):
    u = qmc.Sobol(5, scramble=True, seed=seed).random_base2(int(np.log2(k)))
    return torch.from_numpy(
        ndtri(np.clip(u[:, :3], np.finfo(float).eps, 1 - np.finfo(float).eps)).astype(
            np.float32
        )
    )


class FQE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(58, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], -1))


def expected_fqe_value(q, s, mu, sd, k=OPE_CONFIG.fqe_action_samples, z=None):
    if z is None:
        z = torch.randn(k, len(mu), 3)
    else:
        z = z[:, None, :]
        k = len(z)
    actions = (mu[None] + sd[None] * z).tanh().clamp(-1 + 1e-06, 1 - 1e-06)
    return (
        q(s[None].expand(k, -1, -1).reshape(-1, 55), actions.reshape(-1, 3))
        .reshape(k, len(s), 4)
        .mean(0)
    )


def evaluate_fqe(q, d, p):
    ids, pos = np.unique(d["traj_ids"], return_index=True)
    values = (
        expected_fqe_value(
            q,
            torch.from_numpy(d["states"][pos]),
            p["initial_mu"],
            p["initial_std"],
            z=_fqe_noise(1024, 5001),
        )
        .detach()
        .numpy()
    )
    return (ids, np.concatenate([values, values.sum(-1, keepdims=True)], axis=-1))


@torch.no_grad()
def fqe_components(q, d, p):
    ids, iv = evaluate_fqe(q, d, p)
    obs = []
    vn = []
    z = _fqe_noise(16, 4821)
    for i in range(0, len(d["states"]), 1024):
        sl = slice(i, i + 1024)
        s, a, sn = [
            torch.from_numpy(d[k][sl]) for k in ("states", "actions", "next_states")
        ]
        obs.append(q(s, a).numpy())
        vn.append(expected_fqe_value(q, sn, p["mu"][sl], p["std"][sl], z=z).numpy())
    return dict(
        initial_values=iv,
        trajectory_ids=ids,
        q_observed=np.concatenate(obs),
        next_v=np.concatenate(vn),
    )


def fit_fqe(
    key,
    output_dir,
    seed=OPE_CONFIG.evaluator_seeds[0],
    iterations=OPE_CONFIG.fqe_iterations,
    updates=OPE_CONFIG.fqe_updates,
):
    if iterations <= 0 or updates <= 0:
        raise ValueError("FQE iteration and update counts must be positive")
    output = new_output(output_dir)
    torch.set_num_threads(1)
    d = load_fqe_data("pooled", "train")
    tr = _fqe_tensors(d)
    pt = load_fqe_parameters(key, "train")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    q = FQE()
    target = copy.deepcopy(q)
    optimizer = torch.optim.Adam(q.parameters(), lr=OPE_CONFIG.fqe_lr)
    for iteration in range(1, iterations + 1):
        target.load_state_dict(q.state_dict())
        for _ in range(updates):
            ix = torch.from_numpy(
                rng.integers(len(tr["states"]), size=OPE_CONFIG.fqe_batch_size)
            )
            with torch.no_grad():
                vn = expected_fqe_value(
                    target, tr["next_states"][ix], pt["mu"][ix], pt["std"][ix]
                )
                y = (
                    OPE_CONFIG.reward_scale * tr["rewards"][ix]
                    + OPE_CONFIG.gamma * (1 - tr["dones"][ix]) * vn
                )
            loss = (q(tr["states"][ix], tr["actions"][ix]) - y).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite FQE loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if iteration in (400, 600) or iteration == iterations:
            torch.save(q.state_dict(), output / f"evaluator_iteration{iteration}.pth")
            print(key, seed, iteration, float(loss.detach()), flush=True)
    write_json(
        output / "protocol.json",
        dict(
            policy_key=key,
            seed=seed,
            iterations=iterations,
            updates_per_iteration=updates,
            fit_split="synthetic pooled train only",
            next_action_samples=4,
            gamma=0.99,
            reward_scale=0.01,
            independent_from_policy_critic=True,
        ),
    )
    return q

