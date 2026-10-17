"""Portable synthetic inputs and frozen-model adapters for this release."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from .utils import read_json, sha256
from .bc import SquashedGaussianPolicy

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "reproduction_inputs"
GROUPS = ("senior", "ordinary")
CATEGORIES = ("safety", "traffic_rules", "comfort", "efficiency", "total")
COMPONENTS = (("collision", "ttc"),
              ("stopline_violation", "signal_violation", "fail_in_keeping_lane"),
              ("accel_jerk", "steer_jerk", "brake_jerk"), ("velocity",))
POLICY_GROUPS = {"senior_auto_p2": "senior", "senior_BC_p0": "senior",
                 "ordinary_warm_p2": "ordinary", "ordinary_bc_gaussian_p2": "ordinary"}


def manifest():
    return read_json(INPUTS / "manifest.json")


def records(group=None, split="all"):
    if group is not None and group not in GROUPS:
        raise ValueError("group must be ordinary or senior")
    if split not in ("all", "train", "valid"):
        raise ValueError("The release provides synthetic train/valid/all; no original test split")
    return [r for r in manifest()["records"] if (group is None or r["source_group"] == group)
            and (split == "all" or r["split"] == split)]


def read_drive(record, verify=True):
    path = INPUTS / record["file"]
    if verify and sha256(path) != record["sha256"]:
        raise ValueError(f"Synthetic input checksum mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    n = record["transitions"]
    for key in ("states", "next_states", "generator_states", "generator_next_states"):
        if data[key].shape != (n, 51):
            raise ValueError(f"Invalid shape: {key}")
    if data["actions"].shape != (n, 3) or np.abs(data["actions"]).max() > 1:
        raise ValueError("Expected bounded three-dimensional actions")
    if any(len(v) != n or not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Invalid transition arrays")
    dones = data["dones"].reshape(-1)
    if dones[:-1].any() or dones[-1] != 1:
        raise ValueError("Expected one terminal at the end of the trajectory")
    data["dones"] = dones.astype(np.float32)
    np.testing.assert_array_equal(data["next_states"][:-1], data["states"][1:])
    data["rewards"] = np.stack([sum(data[f"reward_{k}"] for k in c) for c in COMPONENTS], 1).astype(np.float32)
    np.testing.assert_allclose(data["rewards"].sum(1), data["reward_total"].reshape(-1), atol=2e-5)
    data["traj_ids"] = np.full(n, record["id"], dtype=np.int64)
    return data


def load_data(group, split="train"):
    selected = records(group, split)
    if not selected:
        raise ValueError("No trajectories in the requested synthetic split")
    drives = [read_drive(r) for r in selected]
    return {k: np.concatenate([d[k] for d in drives]) for k in drives[0]}


def model_path(relative):
    entry = manifest()["models"][relative]
    path = INPUTS / relative
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Model checksum mismatch: {relative}")
    return path


def reference_policy_spec(group, generation="fixed"):
    if group not in GROUPS or generation != "fixed":
        raise ValueError("Choose ordinary/senior and the fixed reference generation")
    relative = f"policies/irl_{group}.pth"
    entry = manifest()["models"][relative]
    return dict(path=f"reproduction_inputs/{relative}", sha256=entry["sha256"],
                alpha=None, gamma=.99, input_coordinates="fixed_model")




def generator_to_model(states, group):
    transforms = read_json(INPUTS / "transforms.json")
    c = transforms["cohorts"][group]
    # Preserve the saved affine-transform sequence used by the fixed models.
    intermediate = np.asarray(states, float) * c["generator_to_legacy_state_scale"] + c["generator_to_legacy_state_mean"]
    physical = intermediate * c["original_legacy_scaler_scale"] + c["original_legacy_scaler_mean"]
    common, inner = transforms["common"], transforms["inner"]
    corrected = ((physical - common["mean"]) / common["scale"]).astype(np.float32)
    return ((corrected.astype(float) - inner["mean"]) / inner["scale"]).astype(np.float32)


@lru_cache(maxsize=4)
def load_policy(key):
    from .agent import ReferencePolicy
    group = POLICY_GROUPS[key]
    if key.endswith(("auto_p2", "warm_p2")):
        return ReferencePolicy(ROOT, group)
    relative = f"policies/bc_{group}.pth"
    entry = manifest()["models"][relative]
    if entry["architecture"] == "squashed_gaussian":
        model = SquashedGaussianPolicy(51, 3, 256)
        model.load_state_dict(torch.load(model_path(relative), map_location="cpu", weights_only=True))
        return model.eval().requires_grad_(False)
    return ReferencePolicy.from_checkpoint(model_path(relative))


@torch.inference_mode()
def parameters(model, states):
    if isinstance(model, SquashedGaussianPolicy):
        dist, _ = model(states)
        return dist.loc, dist.scale
    return model.parameters(states)


@torch.inference_mode()
def all_parameters(model, states):
    batches = [parameters(model, torch.as_tensor(states[i:i + 8192], dtype=torch.float32))
               for i in range(0, len(states), 8192)]
    return torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])


@lru_cache(maxsize=8)
def behavior_policy(alias):
    w = torch.load(model_path(f"behavior/{alias}.pth"), map_location="cpu", weights_only=True)
    model = SquashedGaussianPolicy(51, 3, w["hidden_dim"])
    model.load_state_dict(w["state_dict"])
    return model.eval().requires_grad_(False)


def load_fqe_data(group="pooled", split="train"):
    selected = records(None if group == "pooled" else group, split)
    drives = []
    for record in selected:
        d = read_drive(record)
        n = len(d["states"])
        previous = np.zeros((n, 3), np.float32)
        previous[1:] = d["actions"][:-1]
        flag = np.ones((n, 1), np.float32)
        flag[0] = 0
        d["states"] = np.column_stack([d["states"], previous, flag])
        d["next_states"] = np.column_stack([d["next_states"], d["actions"], np.ones((n, 1), np.float32)])
        drives.append(d)
    return {k: np.concatenate([d[k] for d in drives]) for k in drives[0]}


def load_fqe_parameters(key, split="train"):
    d = load_fqe_data("pooled", split)
    model = load_policy(key)
    mu, sd = all_parameters(model, d["next_states"][:, :51])
    starts = np.r_[0, np.flatnonzero(d["traj_ids"][1:] != d["traj_ids"][:-1]) + 1]
    initial_mu, initial_std = all_parameters(model, d["states"][starts, :51])
    return dict(mu=mu, std=sd, initial_mu=initial_mu, initial_std=initial_std)
