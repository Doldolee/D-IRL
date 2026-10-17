"""Ordered duration plans and connected-segment training/export data.

Phase identity is a conditioning variable. Zero-labeled road sections before,
between and after scenarios are distinct phases, not missing observations.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .data import REWARD_COMPONENTS, TOTAL_COMPONENTS, Tuples, new_directory, write_json

PHASE_SCENARIOS = (0, 1, 0, 2, 0, 3, 0, 4, 0)
CONDITION_DIM = 21  # 9 phase indicators, 2 progress values, 9 durations, final-state flag.


def phase_runs(scenarios):
    x = np.asarray(scenarios).reshape(-1)
    if not len(x) or not np.isfinite(x).all() or not np.equal(x, np.round(x)).all():
        raise ValueError("Scenario labels must be a nonempty finite integer sequence.")
    cuts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
    return tuple(int(v) for v in x[cuts[:-1]]), np.diff(cuts)


def complete_spans(data, spans):
    selected, durations, excluded = [], [], Counter()
    for a, b in spans:
        pattern, sizes = phase_runs(data.arrays["scenarios"][a:b])
        if pattern != PHASE_SCENARIOS:
            excluded[str(pattern)] += 1
        elif not data.arrays["dones"][b - 1]:
            excluded["missing_terminal"] += 1
        else:
            selected.append((a, b))
            durations.append(sizes)
    return selected, np.asarray(durations, dtype=np.int64).reshape(-1, 9), dict(excluded)


@dataclass
class DurationPlan:
    """Joint log-duration distribution, fitted only to complete training drives.

    This is an empirical parametric planner, not a neural termination head or a
    privacy guarantee. No individual source duration vector is sampled/copied.
    """
    mean: list
    covariance: list
    minimum: list
    maximum: list
    total_min: int
    total_max: int

    @classmethod
    def fit(cls, durations):
        d = np.asarray(durations)
        if d.ndim != 2 or d.shape[1] != 9 or len(d) < 2 or (d < 1).any():
            raise ValueError("At least two complete training trajectories are required.")
        logs = np.log(d)
        covariance = np.cov(logs, rowvar=False) + np.eye(9) * 1e-6
        return cls(logs.mean(0).tolist(), covariance.tolist(), d.min(0).tolist(),
                   d.max(0).tolist(), int(d.sum(1).min()), int(d.sum(1).max()))

    def sample(self, count, rng):
        if count < 1:
            raise ValueError("count must be positive.")
        result = []
        for _ in range(count):
            for _ in range(1000):
                log_d = rng.multivariate_normal(self.mean, self.covariance)
                log_d = np.clip(log_d, np.log(self.minimum), np.log(self.maximum))
                d = np.clip(np.rint(np.exp(log_d)), self.minimum, self.maximum).astype(np.int64)
                if self.total_min <= d.sum() <= self.total_max:
                    result.append(d)
                    break
            else:
                raise ValueError("Could not sample a duration plan within fitted total-length bounds.")
        return np.stack(result)

    def conditions(self, durations):
        d = np.asarray(durations, dtype=np.int64)
        if d.shape != (9,) or (d < 1).any():
            raise ValueError("A complete plan needs nine positive phase durations.")
        total = int(d.sum())
        # An extra row represents the final next_state; its other fields are not exported.
        t = np.arange(total + 1)
        ends = np.cumsum(d)
        phase = np.minimum(np.searchsorted(ends, t, side="right"), 8)
        starts = np.r_[0, ends[:-1]]
        local = np.minimum((t - starts[phase]) / np.maximum(d[phase] - 1, 1), 1)
        scale = np.maximum(np.sqrt(np.diag(self.covariance)), 0.1)
        duration_condition = np.clip((np.log(d) - self.mean) / scale, -5, 5)
        return np.column_stack([
            np.eye(9)[phase], 2 * local - 1, 2 * t / total - 1,
            np.broadcast_to(duration_condition, (total + 1, 9)), t == total,
        ]).astype(np.float32)


class ConnectedWindows:
    """Context + two adjacent segments; pad with an explicit validity mask.

    Starts include the beginning, every phase boundary and the terminal region.
    The second segment is generated from the first during training. A sample
    never crosses a source trajectory. Encoded rows can use WindowBatches' cache.
    """
    def __init__(self, data, raw, schema, spans, planner, segment_length, context_length, stride,
                 continuous_transform='input'):
        self.length = context_length + 2 * segment_length
        values, starts, offset = [], [], 0
        for a, b in spans:
            pattern, durations = phase_runs(data.arrays["scenarios"][a:b])
            if pattern != PHASE_SCENARIOS:
                raise ValueError("Connected windows require complete ordered trajectories.")
            rows = np.concatenate([raw[a:b], raw[b - 1:b].copy()])
            rows[-1, :data.obs_dim] = data.arrays["next_states"][b - 1]
            encoded = schema.encode(rows, strict=False, continuous_transform=continuous_transform)
            packed = np.column_stack([encoded, planner.conditions(durations), np.ones(len(rows))])
            padded = np.pad(packed, ((context_length, 2 * segment_length - 1), (0, 0)))
            # Include all phases even when a short phase falls between regular starts.
            positions = set(range(0, len(rows), stride))
            positions.update(int(v) for v in np.r_[0, np.cumsum(durations)[:-1]])
            positions.add(max(0, len(rows) - 2 * segment_length))
            starts.extend(offset + t for t in sorted(positions))
            values.append(padded.astype(np.float32))
            offset += len(padded)
        if not starts:
            raise ValueError("No complete trajectories in this split.")
        self.starts = np.asarray(starts, dtype=np.int64)
        self.values = np.concatenate(values)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        a = self.starts[index]
        return self.values[a:a + self.length]


def trajectory_arrays(sequences, obs_dim, act_dim):
    if not 1 <= len(sequences) <= 32767:
        raise ValueError("Require 1..32767 trajectories for ReplayBuffer compatibility.")
    width = obs_dim + act_dim + 1 + len(REWARD_COMPONENTS)
    for seq in sequences:
        if seq.ndim != 2 or seq.shape[1] != width or len(seq) < 10:
            raise ValueError("Invalid complete-trajectory records.")
        pattern, _ = phase_runs(seq[:-1, obs_dim + act_dim])
        if pattern != PHASE_SCENARIOS:
            raise ValueError("Generated trajectory does not follow the complete scenario plan.")
    lengths = [len(seq) - 1 for seq in sequences]
    rows = np.concatenate([seq[:-1] for seq in sequences])
    arrays = {
        "states": rows[:, :obs_dim].astype(np.float32),
        "next_states": np.concatenate([seq[1:, :obs_dim] for seq in sequences]).astype(np.float32),
        "actions": rows[:, obs_dim:obs_dim + act_dim].astype(np.float32),
        "scenarios": rows[:, obs_dim + act_dim].astype(np.int16),
        "traj_ids": np.repeat(np.arange(len(sequences), dtype=np.int16), lengths),
        "dones": np.zeros(len(rows), dtype=bool),
    }
    arrays["dones"][np.cumsum(lengths) - 1] = True
    for i, name in enumerate(REWARD_COMPONENTS):
        arrays[f"reward_{name}"] = rows[:, obs_dim + act_dim + 1 + i].astype(np.float32)
    arrays["reward_total"] = sum(arrays[f"reward_{name}"] for name in TOTAL_COMPONENTS)
    return Tuples(arrays)


def export_trajectories(sequences, schema, output, metadata):
    data = trajectory_arrays(sequences, schema.obs_dim, schema.act_dim)
    output = new_directory(output)
    for name, values in data.arrays.items():
        # Preserve the original compact metadata types after Tuples validation.
        dtype = bool if name == "dones" else np.int16 if name in ("scenarios", "traj_ids") else np.float32
        np.save(output / f"{name}.npy", values.astype(dtype), allow_pickle=False)
    write_json(output / "manifest.json", {
        **metadata, "synthetic": True, "unit": "complete_trajectory",
        "trajectory_count": len(sequences), "trajectory_lengths": [len(x) - 1 for x in sequences],
        "phase_scenarios": list(PHASE_SCENARIOS),
        "end_semantics": "one terminal at the end of the sampled final road phase; segment boundaries are not terminals",
        "rewards": "13 conditional components; total sums the 9 driving components",
        "privacy_certified": False, "scientific_utility_validated": False,
    })
    return data
