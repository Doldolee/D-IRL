"""Seeded transition sampling for the final IQ training recipe."""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, d, seed):
        self.rng = np.random.default_rng(seed)
        self.states = np.asarray(d["states"], dtype=np.float32)
        self.actions = np.asarray(d["actions"], dtype=np.float32)
        self.d = {
            k: torch.from_numpy(np.asarray(d[k], np.float32))
            for k in ("states", "actions", "next_states", "dones")
        }
        self.d["dones"] = self.d["dones"].reshape(-1, 1)
        self.n = len(self.d["states"])
        if not self.n or any(len(value) != self.n for value in self.d.values()):
            raise ValueError(
                "Transition arrays must be nonempty and have matching lengths"
            )

    def sample(self, n):
        ix = torch.from_numpy(self.rng.choice(self.n, n, replace=False))
        return (
            self.d["states"][ix],
            self.d["actions"][ix],
            self.d["next_states"][ix],
            self.d["dones"][ix],
        )
