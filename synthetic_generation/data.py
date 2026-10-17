"""Read and validate numerical synthetic transition tuples."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TOTAL_COMPONENTS = (
    "collision", "ttc", "stopline_violation", "signal_violation",
    "fail_in_keeping_lane", "accel_jerk", "steer_jerk", "brake_jerk", "velocity",
)
REWARD_COMPONENTS = TOTAL_COMPONENTS + (
    "eye_angle", "eye_angle_velo", "head_angle", "head_angle_velo",
)
ARRAY_NAMES = ("states", "actions", "next_states", "scenarios", "dones", "traj_ids")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path):
    """Never mix new artifacts with an earlier run or overwrite a dataset."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Tuples:
    def __init__(self, arrays, path=None):
        self.arrays = arrays
        self.path = Path(path) if path is not None else None
        self.validate()

    @classmethod
    def load(cls, path):
        path = Path(path)
        names = ARRAY_NAMES + tuple(f"reward_{r}" for r in REWARD_COMPONENTS + ("total",))
        arrays = {}
        for name in names:
            x = np.load(path / f"{name}.npy", allow_pickle=False)
            # Validate before integer conversion: non-integer scenario/ID values must fail.
            arrays[name] = np.asarray(x, dtype=np.float64)
        return cls(arrays, path)

    def validate(self):
        a = self.arrays
        n = len(a["states"])
        if n == 0:
            raise ValueError("The dataset is empty.")
        for name, x in a.items():
            if len(x) != n or not np.isfinite(x).all():
                raise ValueError(f"Invalid length or nonfinite values in {name}.")
            if name in ("states", "actions", "next_states"):
                if x.ndim != 2 or x.shape[1] == 0:
                    raise ValueError(f"{name} must have shape [N, D].")
            else:
                if x.ndim not in (1, 2) or x.size != n:
                    raise ValueError(f"{name} must have shape [N] or [N, 1].")
                a[name] = x.reshape(-1)
        if a["states"].shape != a["next_states"].shape:
            raise ValueError("states and next_states have different shapes.")
        for name in ("scenarios", "traj_ids", "dones"):
            if not np.equal(a[name], np.round(a[name])).all():
                raise ValueError(f"{name} must contain integers.")
            a[name] = a[name].astype(np.int64)
        if not np.isin(a["dones"], [0, 1]).all():
            raise ValueError("dones must be binary.")
        if np.abs(a["actions"]).max() > 1.00001:
            raise ValueError("The existing squashed policies require actions in [-1, 1].")
        ids = a["traj_ids"]
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        ends = np.r_[starts[1:], n]
        if len(np.unique(ids[starts])) != len(starts):
            raise ValueError("Each trajectory must occupy one contiguous block, in time order.")
        self.spans = list(zip(starts.tolist(), ends.tolist()))
        self.ids = ids[starts]
        for start, end in self.spans:
            if a["dones"][start:end - 1].any():
                raise ValueError("Terminal transition inside a trajectory; split its ID first.")
            if not np.allclose(a["next_states"][start:end - 1], a["states"][start + 1:end],
                               rtol=1e-5, atol=1e-6):
                raise ValueError("next_states[t] != states[t+1] inside a trajectory.")
        expected = sum(a[f"reward_{r}"] for r in TOTAL_COMPONENTS)
        if not np.allclose(expected, a["reward_total"], rtol=1e-5, atol=1e-5):
            raise ValueError("reward_total must sum the nine driving components (excluding gaze).")

    @property
    def obs_dim(self):
        return self.arrays["states"].shape[1]

    @property
    def act_dim(self):
        return self.arrays["actions"].shape[1]

    def matrix(self):
        """Only modeled channels; IDs, next states and totals are derived, never learned."""
        a = self.arrays
        return np.column_stack([
            a["states"], a["actions"], a["scenarios"],
            *[a[f"reward_{r}"] for r in REWARD_COMPONENTS],
        ])

    def summary(self):
        lengths = [end - start for start, end in self.spans]
        return {"transitions": len(self.arrays["states"]), "trajectories": len(self.spans),
                "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                "min_length": min(lengths), "max_length": max(lengths)}


def split_spans(data, validation_fraction, seed, partition_ids=None):
    """Split before windowing; optionally keep multiple trajectories of one person together."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie between 0 and 1.")
    labels = np.asarray(partition_ids if partition_ids is not None else data.ids)
    if labels.ndim != 1 or len(labels) != len(data.spans):
        raise ValueError("Split-group IDs must have one entry per trajectory in appearance order.")
    unique = np.unique(labels)
    if len(unique) < 2:
        raise ValueError("At least two independent split groups are required.")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_val = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    val_labels = set(shuffled[:n_val].tolist())
    train, val = [], []
    for label, span in zip(labels.tolist(), data.spans):
        (val if label in val_labels else train).append(span)
    return train, val




