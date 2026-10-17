"""BC policy architecture and squashed action-density utilities."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import Normal


class BCDataset(Dataset):
    def __init__(self, buffer):
        self.states = torch.from_numpy(buffer.states).float()
        self.actions = torch.from_numpy(buffer.actions).float()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx])


LOG_STD_MIN = -5

LOG_STD_MAX = 2


def squashed_log_prob(dist, actions, eps=1e-06):
    """Density of the interior-clamped action under a tanh Normal policy."""
    actions = actions.clamp(-1 + eps, 1 - eps)
    u = torch.atanh(actions)
    log_det = 2 * math.log(2) - F.softplus(2 * u) - F.softplus(-2 * u)
    return (dist.log_prob(u) - log_det).sum(dim=-1)


class SquashedGaussianPolicy(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, log_std_min=-5, log_std_max=2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        h = self.net(x)
        mu = self.mean_head(h)
        raw_logstd = self.log_std_head(h)
        log_std_tanh = torch.tanh(raw_logstd)
        logstd = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std_tanh + 1
        )
        std = torch.exp(logstd)
        dist = Normal(mu, std)
        return (dist, logstd)


