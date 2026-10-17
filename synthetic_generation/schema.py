"""Fit type-aware transforms on training trajectories only."""
from __future__ import annotations

import numpy as np

from .data import REWARD_COMPONENTS


class Schema:
    def __init__(self, features, obs_dim, act_dim):
        self.features = features
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        offset = 0
        self.blocks = []
        for f in features:
            width = len(f["values"]) if f["kind"] == "categorical" else 1
            self.blocks.append((offset, offset + width))
            offset += width
        self.encoded_dim = offset

    @classmethod
    def fit(cls, matrix, obs_dim, act_dim, spec=None):
        spec = spec or {}
        names = spec.get("state_names", [f"state_{i:02d}" for i in range(obs_dim)])
        if len(names) != obs_dim:
            raise ValueError("Schema state_names length does not match states.npy.")
        names = names + [f"action_{i}" for i in range(act_dim)] + ["scenario"]
        names += [f"reward_{r}" for r in REWARD_COMPONENTS]
        if matrix.shape[1] != len(names):
            raise ValueError("Unexpected matrix width.")
        overrides = spec.get("types", {})
        unknown = set(overrides) - set(names)
        if unknown:
            raise ValueError(f"Unknown feature type overrides: {sorted(unknown)}")
        max_categories = int(spec.get("max_auto_categories", 16))
        features = []
        for i, name in enumerate(names):
            col = matrix[:, i]
            unique = np.unique(col)
            kind = overrides.get(name, "auto")
            if name == "scenario":
                kind = "categorical"
            elif name.startswith("action_") and kind == "auto":
                kind = "continuous"
            if kind not in ("auto", "continuous", "categorical"):
                raise ValueError(f"Unsupported feature type: {kind}")
            if kind == "auto":
                kind = "categorical" if len(unique) <= max_categories else "continuous"
            f = {"name": name, "kind": kind}
            if kind == "categorical":
                if len(unique) > 256:
                    raise ValueError(f"{name} has too many categories; check its schema.")
                # Standardized binary values are retained exactly, not rounded to 0/1.
                f["values"] = unique.tolist()
            else:
                f.update(low=float(col.min()), high=float(col.max()))
            features.append(f)
        return cls(features, obs_dim, act_dim)

    def to_dict(self):
        return {"version": 1, "obs_dim": self.obs_dim, "act_dim": self.act_dim,
                "features": self.features}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1:
            raise ValueError("Unsupported schema version.")
        return cls(data["features"], data["obs_dim"], data["act_dim"])

    def encode(self, matrix, strict=True, continuous_transform='input'):
        if continuous_transform != 'input':
            raise ValueError('The final model uses input numerical coordinates.')
        x = np.asarray(matrix)
        result = np.zeros((*x.shape[:-1], self.encoded_dim), dtype=np.float32)
        for i, (f, (start, end)) in enumerate(zip(self.features, self.blocks)):
            col = x[..., i]
            if f["kind"] == "continuous":
                result[..., start] = col
            else:
                values = np.asarray(f["values"])
                index = np.abs(col[..., None] - values).argmin(axis=-1)
                known = np.isclose(col, values[index], rtol=1e-6, atol=1e-7)
                if strict and not known.all():
                    raise ValueError(f"Unseen category for {f['name']}; inspect the training split.")
                block = np.eye(len(values), dtype=np.float32)[index]
                result[..., start:end] = block * known[..., None]
        return result

    def decode(self, encoded, rng, categorical_mode="argmax", continuous_transform='input'):
        if continuous_transform != 'input':
            raise ValueError('The final model uses input numerical coordinates.')
        if categorical_mode not in ("argmax", "sample"):
            raise ValueError("categorical_mode must be argmax or sample.")
        x = np.asarray(encoded)
        result = np.empty((*x.shape[:-1], len(self.features)), dtype=np.float64)
        for i, (f, (start, end)) in enumerate(zip(self.features, self.blocks)):
            if f["kind"] == "continuous":
                result[..., i] = np.clip(x[..., start], f['low'], f['high'])
            else:
                p = np.maximum(x[..., start:end], 0)
                p = p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)
                if categorical_mode == "argmax":
                    index = p.argmax(axis=-1)
                else:
                    index = (rng.random(p.shape[:-1])[..., None] > p.cumsum(axis=-1)).sum(axis=-1)
                    index = np.minimum(index, end - start - 1)
                result[..., i] = np.asarray(f["values"])[index]
        return result
