"""IQ-Learn networks, the final update rule and frozen reference-policy loading."""

import math
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
import torch.nn.functional as F
from .utils import sha256

LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-6
NETWORK_NAMES = (
    "critic1",
    "critic2",
    "critic1_target",
    "critic2_target",
    "actor_net",
    "actor_mean",
    "actor_logstd",
)


class ContinuousIQLearn:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 0.0003,
        hidden_dim: int = 256,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        chi2_alpha: float = 0.2,
        policy_freq: int = 2,
        autotune: bool = True,
        target_entropy: float | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.chi2_alpha = chi2_alpha
        self.policy_freq = policy_freq
        self.total_it = 0
        self.autotune = autotune
        self.target_entropy = -act_dim if target_entropy is None else target_entropy
        self.batch_size = batch_size
        self.hidden_dims = [hidden_dim, hidden_dim]
        if self.autotune:
            self.log_alpha = torch.tensor(
                np.log(self.alpha), requires_grad=True, device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        def make_critic():
            layers = []
            inp = obs_dim + act_dim
            for h in self.hidden_dims:
                layers += [nn.Linear(inp, h), nn.ReLU()]
                inp = h
            layers += [nn.Linear(inp, 1)]
            return nn.Sequential(*layers).to(self.device)

        self.critic1 = make_critic()
        self.critic2 = make_critic()
        self.critic1_target = make_critic()
        self.critic2_target = make_critic()
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_target.eval()
        self.critic2_target.eval()
        self.critic1_target.requires_grad_(False)
        self.critic2_target.requires_grad_(False)
        self.q_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        layers = []
        inp = obs_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        self.actor_net = nn.Sequential(*layers).to(self.device)
        self.actor_mean = nn.Linear(inp, act_dim).to(self.device)
        self.actor_logstd = nn.Linear(inp, act_dim).to(self.device)
        self.pi_opt = optim.Adam(
            list(self.actor_net.parameters())
            + list(self.actor_mean.parameters())
            + list(self.actor_logstd.parameters()),
            lr=lr,
        )

    def _sample_action(self, s: torch.Tensor, stochastic: bool = True):
        h = self.actor_net(s)
        mu = self.actor_mean(h)
        raw_logstd = self.actor_logstd(h)
        log_std_tanh = torch.tanh(raw_logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_tanh + 1)
        std = torch.exp(logstd)
        if stochastic:
            base = Normal(mu, std)
            x = base.rsample()
        else:
            x = mu
            base = Normal(mu, std)
        a = torch.tanh(x)
        a_clamped = a.clamp(-1 + EPS, 1 - EPS)
        logp_x = base.log_prob(x)
        log_detJ = 2 * math.log(2) - F.softplus(2 * x) - F.softplus(-2 * x)
        logp = (logp_x - log_detJ).sum(dim=-1, keepdim=True)
        return (a_clamped, logp)

    def _log_prob(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = self.actor_net(s)
        mu = self.actor_mean(h)
        raw_logstd = self.actor_logstd(h)
        log_std_tanh = torch.tanh(raw_logstd)
        logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std_tanh + 1)
        std = torch.exp(logstd)
        a_clamped = a.clamp(-1 + EPS, 1 - EPS)
        x = torch.atanh(a_clamped)
        dist = Normal(mu, std)
        logp_x = dist.log_prob(x)
        log_detJ = 2 * math.log(2) - F.softplus(2 * x) - F.softplus(-2 * x)
        logp = (logp_x - log_detJ).sum(dim=-1, keepdim=True)
        return (logp, logstd)

    def _compute_q(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        qa = torch.cat([s, a], dim=-1)
        q1 = self.critic1(qa)
        q2 = self.critic2(qa)
        return torch.min(q1, q2)

    def _compute_v(self, s: torch.Tensor) -> torch.Tensor:
        a_pi, logp_pi = self._sample_action(s)
        qa_pi = torch.cat([s, a_pi], dim=-1)
        q1p = self.critic1(qa_pi)
        q2p = self.critic2(qa_pi)
        return torch.min(q1p, q2p) - self.alpha * logp_pi

    def recover_reward(
        self, s: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            q_min = self._compute_q(s, a)
            v_next = self._compute_v(s_next)
        done = done.to(device=q_min.device, dtype=q_min.dtype).reshape(-1, 1)
        if done.shape != q_min.shape:
            raise ValueError("done must contain one terminal flag per transition")
        r = q_min - self.gamma * (1.0 - done) * v_next
        return r

    def update(self, buf):
        """Perform one IQ update with the final pure-SAC actor objective."""
        self.total_it += 1
        self.pi_opt.zero_grad(set_to_none=True)
        s, a, sn, done = buf.sample(self.batch_size)
        s, a, sn, done = [x.to(self.device) for x in (s, a, sn, done)]
        done = done.float().reshape(-1, 1)
        if done.shape != (len(s), 1):
            raise ValueError("done must contain one terminal flag per transition")
        with torch.no_grad():
            ap, lp = self._sample_action(s)
        vc = (
            torch.minimum(
                self.critic1(torch.cat([s, ap], -1)),
                self.critic2(torch.cat([s, ap], -1)),
            )
            - self.alpha * lp
        )
        with torch.no_grad():
            an, ln = self._sample_action(sn)
            vn = (
                torch.minimum(
                    self.critic1_target(torch.cat([sn, an], -1)),
                    self.critic2_target(torch.cat([sn, an], -1)),
                )
                - self.alpha * ln
            )
            y = self.gamma * (1 - done) * vn
        q1 = self.critic1(torch.cat([s, a], -1))
        q2 = self.critic2(torch.cat([s, a], -1))
        loss = 0.5 * (
            (vc - q1).mean()
            + (q1 - y).square().mean() / (4 * self.chi2_alpha)
            + (vc - q2).mean()
            + (q2 - y).square().mean() / (4 * self.chi2_alpha)
        )
        assert torch.isfinite(loss)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        self.q_opt.zero_grad(set_to_none=True)
        logs = dict(
            critic_loss=float(loss.detach()),
            alpha=float(self.alpha),
            critic_chi2_1=float(
                ((q1 - y).square().mean() / (4 * self.chi2_alpha)).detach()
            ),
            critic_chi2_2=float(
                ((q2 - y).square().mean() / (4 * self.chi2_alpha)).detach()
            ),
        )
        if self.total_it % self.policy_freq == 0:
            ap, lp = self._sample_action(s)
            critic_parameters = list(self.critic1.parameters()) + list(
                self.critic2.parameters()
            )
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            qp = self._compute_q(s, ap)
            sac = (self.alpha * lp - qp).mean()
            actor = sac
            assert torch.isfinite(actor)
            self.pi_opt.zero_grad()
            actor.backward()
            self.pi_opt.step()
            for parameter in critic_parameters:
                parameter.requires_grad_(True)
            if self.autotune:
                aloss = -(
                    self.log_alpha.exp() * (lp + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                aloss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            logs.update(
                actor_loss=float(actor.detach()),
                sac_loss=float(sac.detach()),
                entropy=float(-lp.detach().mean()),
                alpha=float(self.alpha),
            )
        with torch.no_grad():
            for src, dest in [
                ("critic1", "critic1_target"),
                ("critic2", "critic2_target"),
            ]:
                for p, pt in zip(
                    getattr(self, src).parameters(), getattr(self, dest).parameters()
                ):
                    pt.mul_(1 - self.tau)
                    pt.add_(self.tau * p)
        return logs

    def save_weights(self, filepath: str) -> None:
        """
        actor 및 critic 네트워크의 가중치를 한 파일에 저장합니다.
        예: agent.save_weights("iq_learn_weights.pth")
        """
        state = {
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_net": self.actor_net.state_dict(),
            "actor_mean": self.actor_mean.state_dict(),
            "actor_logstd": self.actor_logstd.state_dict(),
            "checkpoint_version": 2,
            "alpha": float(self.alpha),
            "total_it": self.total_it,
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "lr": self.lr,
                "hidden_dim": self.hidden_dims[0],
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "tau": self.tau,
                "alpha": float(self.alpha),
                "chi2_alpha": self.chi2_alpha,
                "policy_freq": self.policy_freq,
                "autotune": self.autotune,
                "target_entropy": self.target_entropy,
                "device": str(self.device),
            },
        }
        torch.save(state, filepath)
        print(f"Saved model weights to {filepath}")

    def load_weights(self, filepath: str) -> None:
        """
        Load weights and saved hyperparameters on the requested device.
        Optimizer and RNG states are not saved: this is not an exact training resume.
        예: agent.load_weights("iq_learn_weights.pth")
        """
        state = torch.load(filepath, map_location=self.device, weights_only=True)
        if "config" not in state:
            raise ValueError("Use a checkpoint with saved model configuration.")
        options = {
            key: value
            for key, value in state["config"].items()
            if key
            in (
                "obs_dim",
                "act_dim",
                "lr",
                "hidden_dim",
                "batch_size",
                "gamma",
                "tau",
                "alpha",
                "chi2_alpha",
                "policy_freq",
                "autotune",
                "target_entropy",
                "device",
            )
        }
        options.setdefault("obs_dim", self.obs_dim)
        options.setdefault("act_dim", self.act_dim)
        options["device"] = str(self.device)
        self.__init__(**options)
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        self.actor_net.load_state_dict(state["actor_net"])
        self.actor_mean.load_state_dict(state["actor_mean"])
        self.actor_logstd.load_state_dict(state["actor_logstd"])
        if "alpha" in state:
            self.alpha = float(state["alpha"])
            if self.autotune:
                with torch.no_grad():
                    self.log_alpha.fill_(math.log(self.alpha))
        self.total_it = int(state.get("total_it", 0))
        self.critic1.eval()
        self.critic2.eval()
        self.critic1_target.eval()
        self.critic2_target.eval()
        self.actor_net.eval()
        self.actor_mean.eval()
        self.actor_logstd.eval()
        print(f"Loaded model weights from {filepath}")


class ReferencePolicy:
    """Frozen inference using the original actor/twin-critic network layout.

    ``ReferencePolicy(root, group)`` loads the released fixed seed2/20k policy.
    ``from_checkpoint`` accepts an explicitly supplied compatible checkpoint.
    Loading never changes input coordinates; prepared inputs use model frame.
    """

    def __init__(self, root, group, *, generation="fixed", device="cpu"):
        from .data import reference_policy_spec

        spec = reference_policy_spec(group, generation)
        self._load_checkpoint(
            Path(root) / spec["path"],
            device=device,
            expected_sha256=spec["sha256"],
            alpha=spec["alpha"],
            gamma=spec["gamma"],
        )
        if generation == "fixed":
            config = self.metadata["training_config"]
            if (self.metadata["seed"], self.metadata["step"]) != (2, 20000):
                raise ValueError("Expected the fixed seed2/20k checkpoint")
            if config["warm_start"] != (group == "ordinary"):
                raise ValueError("Checkpoint warm-start setting does not match the cohort")
        self.metadata.update(
            path=spec["path"],
            group=group,
            generation=generation,
            input_coordinates=spec["input_coordinates"],
        )

    @classmethod
    def from_checkpoint(
        cls, filepath, *, device="cpu", expected_sha256=None, alpha=None, gamma=None
    ):
        """Load the released network layout with saved alpha and gamma.

        Dimensions come from tensors. Optional values check saved settings.
        No optimizer or training state is constructed.
        """
        model = cls.__new__(cls)
        model._load_checkpoint(
            Path(filepath),
            device=device,
            expected_sha256=expected_sha256,
            alpha=alpha,
            gamma=gamma,
        )
        return model

    def _load_checkpoint(self, path, *, device, expected_sha256, alpha, gamma):
        digest = sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
        w = torch.load(path, map_location="cpu", weights_only=True)
        required = ("critic1", "critic2", "actor_net", "actor_mean", "actor_logstd")
        if not isinstance(w, dict) or any(key not in w for key in required):
            raise ValueError(
                "Expected an original-layout IQ checkpoint with actor_net and twin critics"
            )
        config = w.get("config", {})

        def setting(name, fallback):
            saved = w.get(name, config.get(name))
            if saved is None:
                raise ValueError(f"Checkpoint must save {name} in its metadata")
            if fallback is not None and not math.isclose(
                float(saved), float(fallback), rel_tol=1e-12
            ):
                raise ValueError(f"Requested {name} conflicts with the saved checkpoint value")
            return float(saved)

        self.alpha, self.gamma = setting("alpha", alpha), setting("gamma", gamma)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        self.device = torch.device(device)
        self.obs_dim = int(w["actor_net"]["0.weight"].shape[1])
        self.hidden_dim = int(w["actor_net"]["0.weight"].shape[0])
        self.act_dim = int(w["actor_mean"]["weight"].shape[0])
        for key, dimension in (
            ("obs_dim", self.obs_dim),
            ("act_dim", self.act_dim),
            ("hidden_dim", self.hidden_dim),
        ):
            if key in config and config[key] != dimension:
                raise ValueError(f"Saved {key} conflicts with the network tensors")
        hidden = self.hidden_dim

        def critic():
            return nn.Sequential(
                nn.Linear(self.obs_dim + self.act_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1, self.q2 = (critic(), critic())
        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head, self.logstd_head = (
            nn.Linear(hidden, self.act_dim), nn.Linear(hidden, self.act_dim)
        )
        for net, key in [
            (self.q1, "critic1"),
            (self.q2, "critic2"),
            (self.actor, "actor_net"),
            (self.mu_head, "actor_mean"),
            (self.logstd_head, "actor_logstd"),
        ]:
            net.load_state_dict(w[key], strict=True)
            net.to(self.device)
            net.eval()
            net.requires_grad_(False)
        self.metadata = dict(
            path=str(path),
            sha256=digest,
            seed=w.get("seed"),
            step=w.get("total_it"),
            alpha=self.alpha,
            gamma=self.gamma,
            training_config=config,
            alpha_source="checkpoint" if "alpha" in w else (
                "config" if "alpha" in config else "explicit"
            ),
            gamma_source="checkpoint" if "gamma" in w else (
                "config" if "gamma" in config else "explicit"
            ),
        )


    def parameters(self, s):
        h = self.actor(s)
        return (self.mu_head(h), (-5 + 3.5 * (self.logstd_head(h).tanh() + 1)).exp())

    def q(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return torch.minimum(self.q1(x), self.q2(x)).flatten()

    def sample_action(self, s, stochastic=True):
        """Return squashed actions and column log densities, using the shared policy."""
        mu, std = self.parameters(s)
        normal = Normal(mu, std)
        latent = normal.rsample() if stochastic else mu
        actions = latent.tanh().clamp(-1 + EPS, 1 - EPS)
        jacobian = 2 * math.log(2) - F.softplus(2 * latent) - F.softplus(-2 * latent)
        return actions, (normal.log_prob(latent) - jacobian).sum(-1, keepdim=True)

    def log_prob(self, s, a):
        mu, std = self.parameters(s)
        latent = torch.atanh(a.clamp(-1 + EPS, 1 - EPS))
        jacobian = 2 * math.log(2) - F.softplus(2 * latent) - F.softplus(-2 * latent)
        return (Normal(mu, std).log_prob(latent) - jacobian).sum(-1, keepdim=True)

    @torch.no_grad()
    def soft_value(self, s, *, samples=1):
        """MC mean of minQ - alpha*logpi; distinct from the TA-Adv baseline."""
        from .utils import _positive_integer

        samples = _positive_integer(samples, "samples")
        total = torch.zeros((len(s), 1), dtype=s.dtype, device=s.device)
        for _ in range(samples):
            action, logp = self.sample_action(s)
            total += self.q(s, action).reshape(-1, 1) - self.alpha * logp
        return total / samples

    @torch.no_grad()
    def recover_reward(self, s, a, s_next, done, *, samples=1):
        """Return N x 1 soft Bellman rewards; no stored DQI reward is used."""
        q = self.q(s, a).reshape(-1, 1)
        done = done.to(device=q.device, dtype=q.dtype).reshape(-1, 1)
        if done.shape != q.shape or len(s_next) != len(s):
            raise ValueError("Expected one next state and terminal flag per transition")
        return q - self.gamma * (1 - done) * self.soft_value(s_next, samples=samples)
