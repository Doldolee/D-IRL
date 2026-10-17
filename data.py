"""Portable synthetic inputs and frozen-model adapters for this release."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch

from .utils import read_json
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


# Per-drive length, behavior-density model, validation flag and random identity.
# These preserve the accepted 10/2 split per cohort and stochastic TA/recovery draws.
_TRAJECTORIES = (
    (8477, 1, True, (2846584443, 2719173962, 4029140874, 577504367)),
    (8813, 1, False, (1867681820, 2957258742, 1230645887, 566823175)),
    (8332, 1, False, (3981182034, 3916813220, 1778258131, 2469201168)),
    (9237, 1, False, (503973952, 3772416723, 549521275, 2797479765)),
    (8740, 0, False, (3398640017, 3351914680, 2823533387, 2367158373)),
    (7576, 0, False, (2919356777, 746587698, 1089319163, 1784596847)),
    (7868, 0, False, (1942573681, 3756854427, 1102131964, 654314833)),
    (8504, 0, True, (3740428575, 261027323, 1855953331, 708544130)),
    (9780, 0, False, (3572392051, 3525227177, 3417066688, 519909717)),
    (8515, 3, False, (3504780146, 928050194, 682431193, 1102531228)),
    (9224, 2, False, (1396639862, 2338938140, 638757165, 810844290)),
    (8787, 2, False, (2680317331, 2162656685, 3364404568, 2573571653)),
    (9729, 4, True, (1705822798, 1529590460, 4009744879, 3507366184)),
    (10888, 4, False, (2673800409, 3593873855, 1549044727, 1286014667)),
    (10253, 5, False, (3686930621, 3667370465, 1887556692, 3774539603)),
    (11387, 5, False, (2361993381, 2760301691, 2309642527, 2722133522)),
    (11109, 7, False, (4140664110, 78523853, 1367331314, 1152516680)),
    (10463, 7, False, (3382541089, 2689848976, 2468595594, 3882079266)),
    (10767, 7, False, (791293448, 2759156529, 3409953322, 1596356990)),
    (11876, 7, True, (3162163402, 1396583656, 3514201506, 71509635)),
    (9403, 7, False, (1200911912, 455644563, 849107445, 2600687111)),
    (10618, 7, False, (206492449, 2830227064, 2038705860, 2086677067)),
    (11203, 6, False, (4140758176, 3951136507, 1429287114, 1234164565)),
    (10976, 6, False, (1737942301, 1495784606, 85482761, 1336289806)),
)
MODEL_FILES = (
    tuple(f"policies/{kind}_{group}.pth" for group in GROUPS for kind in ("irl", "bc"))
    + tuple(f"behavior/behavior{i:02d}.pth" for i in range(8))
    + tuple(f"diffusion/{group}_s{seed}.pt" for group in GROUPS for seed in (101, 202))
)


def records(group=None, split="all"):
    if group is not None and group not in GROUPS:
        raise ValueError("group must be ordinary or senior")
    if split not in ("all", "train", "valid"):
        raise ValueError("The release provides synthetic train/valid/all; no original test split")
    available = [dict(id=i, file=f"synthetic24/drive{i:03d}.npz",
                      source_group="ordinary" if i < 12 else "senior",
                      split="valid" if valid else "train", transitions=n,
                      behavior=f"behavior{behavior:02d}",
                      ta_seed_parts=(42, 0, *identity),
                      recovery_seed_parts=(42, 1, *identity))
                 for i, (n, behavior, valid, identity) in enumerate(_TRAJECTORIES)]
    return [r for r in available if (group is None or r["source_group"] == group)
            and (split == "all" or r["split"] == split)]


def read_drive(record):
    path = INPUTS / record["file"]
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
    if relative not in MODEL_FILES:
        raise ValueError(f"Unknown bundled model: {relative}")
    path = INPUTS / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def reference_policy_spec(group, generation="fixed"):
    if group not in GROUPS or generation != "fixed":
        raise ValueError("Choose ordinary/senior and the fixed reference generation")
    relative = f"policies/irl_{group}.pth"
    return dict(path=f"reproduction_inputs/{relative}", sha256=None,
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
    if group == "senior":
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
