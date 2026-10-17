"""Shared file handling, input verification and conditional kernel integration."""

from functools import lru_cache
from pathlib import Path
import csv
import hashlib
import json
import math
import operator

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm, roots_legendre


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def json_values(value):
    """Convert scientific scalar/array values to finite JSON-compatible values."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {key: json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_values(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_values(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_output(path):
    """Create a fresh output without overwriting bundled inputs or source files."""
    root = Path(__file__).resolve().parent
    destination = Path(path).expanduser().resolve()
    protected = (root, root / "reproduction_inputs", root / "docs", root / "synthetic_generation")
    if any(destination == item or (item != root and item in destination.parents) for item in protected):
        raise ValueError("Choose an output outside bundled inputs and source folders")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _positive_integer(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if isinstance(value, (bool, np.bool_)) or result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


@lru_cache(maxsize=16)
def _normal_quadrature(nodes):
    """Nodes and log weights for expectations under N(0, 1)."""
    x, weights = roots_hermitenorm(nodes)
    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive]) - 0.5 * math.log(2 * math.pi)
    log_weights -= logsumexp(log_weights)
    x.setflags(write=False)
    log_weights.setflags(write=False)
    return (x, log_weights)


def log_kernel_density(
    mu, std, observed, bandwidth, nodes=64, batch_size=2048, clamp_eps=None
):
    """Return log E[Normal(observed; tanh(U), bandwidth**2 I)].

    ``U`` has independent Normal coordinates with means ``mu`` and standard
    deviations ``std``. All three arrays must have identical shape ``(N, A)``
    or ``(A,)``; outputs have shape ``(N,)`` or are a scalar, respectively.
    Zero standard deviations are allowed and give deterministic coordinates.
    Observations at -1 or +1 are valid; observations are never clipped.

    ``clamp_eps=None`` leaves sampled tanh actions unchanged (the BC sampler).
    A positive epsilon clips only sampled actions to ``[-1+eps, 1-eps]`` to
    reproduce the IQ actor's sampler. This clipping is part of the evaluated
    distribution, not a change to the Gaussian kernel or observed action.

    ``bandwidth`` is a positive scalar in the policy's action coordinates.
    The full Gaussian normalization, ``-A*log(h)-A/2*log(2*pi)``, is included,
    making the result suitable for likelihood comparisons across bandwidths.
    ``nodes`` is the quadrature order per coordinate, not an exponential
    joint-grid size. Working arrays use float64 and are bounded by batch size.
    """
    nodes = _positive_integer(nodes, "nodes")
    batch_size = _positive_integer(batch_size, "batch_size")
    h = float(bandwidth)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("bandwidth must be finite and positive.")
    if clamp_eps is not None:
        clamp_eps = float(clamp_eps)
        if not np.isfinite(clamp_eps) or not 0 <= clamp_eps < 1:
            raise ValueError("clamp_eps must be finite and in [0, 1).")
    arrays = [np.asarray(value, dtype=np.float64) for value in (mu, std, observed)]
    mu, std, observed = arrays
    if mu.ndim not in (1, 2) or any((a.shape != mu.shape for a in arrays)):
        raise ValueError(
            "mu, std and observed must have identical (N, A) or (A,) shapes."
        )
    if mu.shape[-1] < 1:
        raise ValueError("At least one action coordinate is required.")
    if any((not np.isfinite(a).all() for a in arrays)) or np.any(std < 0):
        raise ValueError("Inputs must be finite and standard deviations nonnegative.")
    single = mu.ndim == 1
    if single:
        mu, std, observed = [a[None, :] for a in arrays]
    abscissae, log_weights = _normal_quadrature(nodes)
    result = np.empty(mu.shape[0], dtype=np.float64)
    log_normalizer = math.log(h) + 0.5 * math.log(2 * math.pi)
    for start in range(0, len(result), batch_size):
        stop = start + batch_size
        actions = np.tanh(
            mu[start:stop, :, None] + std[start:stop, :, None] * abscissae
        )
        if clamp_eps is not None:
            np.clip(actions, -1 + clamp_eps, 1 - clamp_eps, out=actions)
        actions -= observed[start:stop, :, None]
        actions /= h
        np.square(actions, out=actions)
        actions *= -0.5
        actions += log_weights
        result[start:stop] = (logsumexp(actions, axis=-1) - log_normalizer).sum(axis=-1)
    return float(result[0]) if single else result


def quadrature_convergence(
    mu,
    std,
    observed,
    bandwidth,
    orders=(64, 128, 256, 512),
    batch_size=2048,
    clamp_eps=None,
):
    """Report order-to-order log-density changes; not a certified error bound.

    Returns JSON-compatible metadata, without choosing an order or claiming
    exact integration. Successive orders can agree even when both miss a
    narrow feature; numerical integration on selected observations provides
    an independent accuracy check when that risk matters.
    """
    orders = tuple((_positive_integer(n, "order") for n in orders))
    if not orders or any((b <= a for a, b in zip(orders, orders[1:]))):
        raise ValueError("orders must be a nonempty strictly increasing sequence.")
    previous = None
    rows = []
    for order in orders:
        current = np.atleast_1d(
            log_kernel_density(
                mu,
                std,
                observed,
                bandwidth,
                nodes=order,
                batch_size=batch_size,
                clamp_eps=clamp_eps,
            )
        )
        row = {"nodes": order, "observations": int(current.size)}
        if previous is not None and current.size:
            change = np.abs(current - previous)
            row.update(
                max_abs_log_density_change=float(change.max()),
                p95_abs_log_density_change=float(np.quantile(change, 0.95)),
                rms_log_density_change=float(np.sqrt(np.mean(change**2))),
            )
        rows.append(row)
        previous = current
    return {
        "bandwidth": float(bandwidth),
        "clamp_eps": clamp_eps,
        "orders": rows,
        "interpretation": "Successive-order sensitivity, not a certified quadrature error bound.",
    }


# Adaptive integration for narrow kernels.
_NORMAL_KNOTS = np.array([-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12.0])

_LATENT_KNOTS = np.array([-20, -12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12, 20.0])

_KERNEL_KNOTS = np.array([-12, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 12.0])


@lru_cache(maxsize=12)
def _legendre_rule(nodes):
    x, weights = roots_legendre(nodes)
    return (x, np.log(weights))


def _integration_knots(mu, std, observed, h, clamp_eps):
    """Per-coordinate standardized-Normal knots, including tail bounds.

    At z=0, let d=(tanh(mu)-observed)/h (including sampler clipping).
    Outside +/-sqrt(d*d+144), the log integrand is at least 72 below
    its value at z=0 because the kernel exponential is at most one.
    This is an exponent bound, NOT a certified relative integral bound.
    It intentionally expands beyond +/-12 for rare observed actions.
    """
    center = np.tanh(mu)
    if clamp_eps is not None:
        center = np.clip(center, -1 + clamp_eps, 1 - clamp_eps)
    bound = np.hypot((center - observed) / h, 12.0)
    normal = np.broadcast_to(_NORMAL_KNOTS, (len(mu), len(_NORMAL_KNOTS)))
    latent = (_LATENT_KNOTS[None, :] - mu[:, None]) / std[:, None]
    action = observed[:, None] + h * _KERNEL_KNOTS
    valid = np.abs(action) < 1
    safe = np.where(valid, action, 0.0)
    near_observed = (np.arctanh(safe) - mu[:, None]) / std[:, None]
    near_observed[~valid] = 0.0
    raw_center = np.tanh(mu)
    slope = std * (1 - raw_center * raw_center)
    variance = h * h + slope * slope
    product_center = slope * (observed - raw_center) / variance
    product_width = h / np.sqrt(variance)
    product = product_center[:, None] + product_width[:, None] * _NORMAL_KNOTS
    refined = product_center.copy()
    for _ in range(8):
        a = np.tanh(mu + std * refined)
        derivative = std * (1 - a * a)
        gradient = refined + (a - observed) * derivative / (h * h)
        curvature = 1 + (derivative**2 - 2 * std * a * derivative * (a - observed)) / (
            h * h
        )
        step = np.clip(gradient / np.maximum(curvature, 0.05), -10, 10)
        proposed = np.clip(refined - step, -bound, bound)
        old_cost = refined**2 + ((a - observed) / h) ** 2
        new_cost = proposed**2 + ((np.tanh(mu + std * proposed) - observed) / h) ** 2
        refined = np.where(new_cost < old_cost, proposed, refined)
    a = np.tanh(mu + std * refined)
    derivative = std * (1 - a * a)
    curvature = 1 + (derivative**2 - 2 * std * a * derivative * (a - observed)) / (
        h * h
    )
    refined_width = 1 / np.sqrt(np.maximum(curvature, 0.05))
    refined_product = refined[:, None] + refined_width[:, None] * _NORMAL_KNOTS
    pieces = [
        normal,
        latent,
        near_observed,
        product,
        refined_product,
        -bound[:, None],
        bound[:, None],
    ]
    if clamp_eps is not None and clamp_eps > 0:
        threshold = np.arctanh(1 - clamp_eps)
        pieces.append(
            (np.array([-threshold, threshold])[None, :] - mu[:, None]) / std[:, None]
        )
    knots = np.concatenate(pieces, axis=1)
    np.clip(knots, -bound[:, None], bound[:, None], out=knots)
    knots.sort(axis=1)
    return knots


def log_kernel_density_adaptive(
    mu, std, observed, bandwidth, nodes=16, batch_size=256, clamp_eps=None
):
    """Log conditional Gaussian-convolution density using adaptive intervals.

    Inputs have matching shapes (N,A) or (A,), yielding (N,) or a scalar.
    For independent U_j~Normal(mu_j,std_j), computes the log of the product
    of E[Normal(observed_j; tanh(U_j), bandwidth**2)]. The Gaussian kernel's
    complete normalization is included. ``nodes`` is the Gauss-Legendre
    order PER interval; interval locations adapt to each input coordinate.
    ``batch_size`` bounds the number of scalar coordinates integrated at once.

    ``clamp_eps`` reproduces clipping of sampled actions only, while observed
    actions (including +/-1) are untouched. std=0 is evaluated analytically.
    The expanded integration bound places tail integrand exponents at least
    72 below the z=0 value; it is not a certified relative-error guarantee.
    Compare orders and independent adaptive integration before interpreting
    exceptionally small densities. This function removes sampling noise,
    not kernel bias or behavior-model error.
    """
    nodes = _positive_integer(nodes, "nodes")
    batch_size = _positive_integer(batch_size, "batch_size")
    h = float(bandwidth)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("bandwidth must be finite and positive.")
    if clamp_eps is not None:
        clamp_eps = float(clamp_eps)
        if not np.isfinite(clamp_eps) or not 0 <= clamp_eps < 1:
            raise ValueError("clamp_eps must be finite and in [0, 1).")
    arrays = [np.asarray(a, dtype=np.float64) for a in (mu, std, observed)]
    mu, std, observed = arrays
    if mu.ndim not in (1, 2) or any((a.shape != mu.shape for a in arrays)):
        raise ValueError("Inputs must have matching (N,A) or (A,) shapes.")
    if mu.shape[-1] < 1:
        raise ValueError("At least one action coordinate is required.")
    if any((not np.isfinite(a).all() for a in arrays)) or np.any(std < 0):
        raise ValueError("Inputs must be finite; std must be nonnegative.")
    original_shape = mu.shape
    single = mu.ndim == 1
    mu, std, observed = [a.ravel() for a in arrays]
    result = np.empty_like(mu)
    deterministic = std == 0
    if deterministic.any():
        center = np.tanh(mu[deterministic])
        if clamp_eps is not None:
            center = np.clip(center, -1 + clamp_eps, 1 - clamp_eps)
        result[deterministic] = (
            -0.5 * ((observed[deterministic] - center) / h) ** 2
            - math.log(h)
            - 0.5 * math.log(2 * math.pi)
        )
    selected = np.flatnonzero(~deterministic)
    legendre_x, legendre_logw = _legendre_rule(nodes)
    for start in range(0, len(selected), batch_size):
        ids = selected[start : start + batch_size]
        m, s, obs = (mu[ids], std[ids], observed[ids])
        knots = _integration_knots(m, s, obs, h, clamp_eps)
        middle = (knots[:, 1:] + knots[:, :-1]) * 0.5
        halfwidth = (knots[:, 1:] - knots[:, :-1]) * 0.5
        z = middle[:, :, None] + halfwidth[:, :, None] * legendre_x
        action = np.tanh(m[:, None, None] + s[:, None, None] * z)
        if clamp_eps is not None:
            np.clip(action, -1 + clamp_eps, 1 - clamp_eps, out=action)
        action -= obs[:, None, None]
        action /= h
        np.square(action, out=action)
        action += z * z
        action *= -0.5
        log_halfwidth = np.full_like(halfwidth, -np.inf)
        positive = halfwidth > 0
        log_halfwidth[positive] = np.log(halfwidth[positive])
        action += log_halfwidth[:, :, None] + legendre_logw
        result[ids] = (
            logsumexp(action, axis=(1, 2)) - math.log(2 * math.pi) - math.log(h)
        )
    shaped = result.reshape(original_shape)
    return float(shaped.sum()) if single else shaped.sum(axis=-1)


def verify_inputs():
    """Validate bundled trajectories, coordinate conversion and frozen models."""
    from . import data
    transitions = 0
    selected = data.records()
    for record in selected:
        drive = data.read_drive(record)
        np.testing.assert_array_equal(
            data.generator_to_model(drive["generator_states"], record["source_group"]),
            drive["states"],
        )
        transitions += len(drive["states"])
    for relative in data.MODEL_FILES:
        data.model_path(relative)
    for key in data.POLICY_GROUPS:
        data.load_policy(key)
    for alias in {record["behavior"] for record in selected}:
        data.behavior_policy(alias)
    return dict(passed=True, synthetic_trajectories=len(selected),
                transitions=transitions, models=len(data.MODEL_FILES),
                original_data_required=False)
