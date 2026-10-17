"""Training configuration, BC fitting and final cohort IQ training."""

from dataclasses import dataclass, asdict, replace
import torch
from torch import optim
from torch.utils.data import DataLoader

from . import data
from .agent import ContinuousIQLearn, NETWORK_NAMES
from .bc import BCDataset, SquashedGaussianPolicy, squashed_log_prob
from .buffer import ReplayBuffer
from .utils import new_output, write_csv, write_json


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 2
    steps: int = 20000
    obs_dim: int = 51
    act_dim: int = 3
    hidden_dim: int = 256
    batch_size: int = 256
    lr: float = 3e-4
    actor_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    chi2_alpha: float = 0.2
    policy_freq: int = 2
    autotune: bool = True
    target_entropy: float = -3
    warm_start: bool = False
    bc_init: str | None = None
    device: str = "cpu"
    log_interval: int = 100
    save_interval: int = 1000
    threads: int = 1


def cohort_config(group, **overrides):
    if group not in data.GROUPS:
        raise ValueError(group)
    return replace(TrainConfig(warm_start=group == "ordinary"), **overrides)


def build_agent(config, group):
    """Initialize the architecture and optionally load an explicit BC checkpoint."""
    if config.warm_start and config.bc_init is None:
        raise ValueError("BC warm start requires --bc-init. Run train-bc on the released synthetic data first.")
    torch.manual_seed(config.seed)
    fields = (
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
    model = ContinuousIQLearn(**{key: getattr(config, key) for key in fields})
    for parameters in model.pi_opt.param_groups:
        parameters["lr"] = config.actor_lr
    if config.warm_start:
        with torch.random.fork_rng(devices=[]):
            initializer = SquashedGaussianPolicy(config.obs_dim, config.act_dim, config.hidden_dim)
            weights = torch.load(config.bc_init, map_location="cpu", weights_only=True)
            initializer.load_state_dict(weights.get("state_dict", weights))
        for destination, source in (
            ("actor_net", "net"),
            ("actor_mean", "mean_head"),
            ("actor_logstd", "log_std_head"),
        ):
            getattr(model, destination).load_state_dict(
                getattr(initializer, source).state_dict()
            )
    return model


def train_iq(group, output_dir, config=None):
    config = cohort_config(group) if config is None else config
    if (
        min(config.steps, config.log_interval, config.save_interval, config.batch_size)
        <= 0
    ):
        raise ValueError("Training counts and intervals must be positive")
    output = new_output(output_dir)
    torch.set_num_threads(config.threads)
    model = build_agent(config, group)
    buffer = ReplayBuffer(data.load_data(group, "train"), config.seed)
    history = []
    for step in range(1, config.steps + 1):
        metrics = model.update(buffer)
        if step % config.log_interval == 0 or step == config.steps:
            history.append(dict(step=step, **metrics))
        if step % config.save_interval == 0 or step == config.steps:
            payload = {key: getattr(model, key).state_dict() for key in NETWORK_NAMES}
            payload.update(
                config=asdict(config),
                seed=config.seed,
                total_it=step,
                alpha=model.alpha,
                group=group,
                normalization="bundled fixed-model coordinates",
            )
            torch.save(payload, output / f"step{step:06d}.pth")
            if history:
                write_csv(output / "history.csv", history)
            print(group, step, metrics, flush=True)
    write_json(
        output / "protocol.json",
        dict(
            group=group,
            seed=config.seed,
            steps=config.steps,
            config=asdict(config),
            training_split="synthetic train only",
            test_used=False,
            checkpoint_selection=False,
        ),
    )
    return model


def train_squashed_bc(
    buffer,
    state_dim,
    action_dim,
    hidden_dim=256,
    epochs=10,
    batch_size=256,
    lr=0.001,
    device="cpu",
):
    """Fit a BC policy that can also serve as an explicit actor initializer."""
    dataset = BCDataset(buffer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0 or epochs <= 0:
        raise ValueError(
            "BC fitting needs positive epochs and at least one full minibatch"
        )
    model = SquashedGaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_log_std = 0.0
        for states, actions in loader:
            states, actions = (states.to(device), actions.to(device))
            EPS = 1e-06
            actions = torch.clamp(actions, -1 + EPS, 1 - EPS)
            dist, log_std = model(states)
            log_probs = squashed_log_prob(dist, actions, EPS)
            loss = -log_probs.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_log_std += log_std.mean().item()
        avg_loss = total_loss / len(loader)
        avg_log_std = total_log_std / len(loader)
        print(
            f"[Epoch {epoch}/{epochs}] Loss = {avg_loss:.4f}, log_std = {avg_log_std:.4f}"
        )
    return model
