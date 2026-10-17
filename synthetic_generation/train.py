"""Checkpoint loading and cohort-labelled raw temporal TabDiff generation.

The network and rollout kernels implement the final format-v3 generator.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import random

import numpy as np
import torch

from .data import write_json
from .schema import Schema


def seed_all(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint(data):
    digest = hashlib.sha256()
    for name in sorted(data.arrays):
        x = np.ascontiguousarray(data.arrays[name])
        digest.update(name.encode())
        digest.update(str(x.shape).encode())
        digest.update(str(x.dtype).encode())
        digest.update(memoryview(x).cast("B"))
    return digest.hexdigest()


def source_group_name(value):
    if value not in ("ordinary", "senior"):
        raise ValueError("source_group must be ordinary or senior (generator training cohort).")
    return value


def load_generator(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format_version") != 3 or state.get("model_type") != "temporal_tabdiff":
        raise ValueError("The active package supports format-v3 temporal TabDiff checkpoints only.")
    if any(state.get(name) is None for name in ("source_group", "road_coordinates", "road_increments", "export_schema")):
        raise ValueError("Checkpoint is missing final generator metadata.")
    from .diffusion import TemporalTabDiff
    schema = Schema.from_dict(state["schema"])
    generator = TemporalTabDiff(schema, state["quantile_knots"], state["config"]).to(device)
    generator.load_state_dict(state["generator"], strict=True)
    export_schema = Schema.from_dict(state["export_schema"])
    return generator.eval(), export_schema, state


def generate(checkpoint, output, count=32, seed=2026, batch_size=8, device="cpu",
             categorical_mode=None, source_group=None, sampling_steps=None,
             save_decoded_sequences=True):
    """Export unchanged decoded tuples and explicit generator cohort metadata.

    Source states remain in the checkpoint's declared generator input frame.
    Fixed-model coordinate conversion is provided by driving.data.generator_to_model.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if not 1 <= count <= 32767:
        raise ValueError("count must be in 1..32767 for ReplayBuffer compatibility.")
    generator, schema, state = load_generator(checkpoint, device)
    declared = source_group_name(state["source_group"])
    if source_group is None:
        source_group = declared
    source_group_name(source_group)
    if declared is not None and declared != source_group:
        raise ValueError(f"Checkpoint source_group={declared}, requested {source_group}.")
    categorical_mode = categorical_mode or "sample"
    from .trajectory import export_trajectories
    from .diffusion_train import draw_diffusion_trajectories
    sequences, plans = draw_diffusion_trajectories(generator, state, count, seed, batch_size,
                                        categorical_mode, sampling_steps=sampling_steps)
    checkpoint_sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    generator_id = f"{source_group}-{checkpoint_sha[:16]}"
    metadata = {
        "source_group": source_group, "source_cohort": source_group,
        "cohort_semantics": "generator training cohort; not real participant identity",
        "source_group_evidence": "checkpoint",
        "generator_id": generator_id, "generator_step": state["step"],
        "generation_seed": seed, "generation_batch_size": batch_size,
        "categorical_mode": categorical_mode, "checkpoint_sha256": checkpoint_sha,
        "input_fingerprint": state["input_fingerprint"],
        "input_manifest": state.get("input_manifest"),
        "input_manifest_sha256": state.get("input_manifest_sha256"),
        "artificial_fixture": state.get("artificial_fixture", False),
        "model_type": "temporal_tabdiff",
        "sampling_steps": sampling_steps or state["config"]["sampling_steps"],
        "category_sampling": "absorbing reverse chain; argmax is an optional diagnostic variant",
        "segment_length": state["config"]["sequence_length"],
        "context_length": state["config"]["context_length"],
        "phase_durations": plans.tolist(),
        "scenario_semantics": "conditioned on ordered phase plan, not sampled independently",
        "post_generation_action_reward_correction": False,
        "state_coordinate_frame": (state.get("input_manifest") or {}).get("state_coordinate_frame", "supplied input tuple frame"),
    }
    if state.get("road_coordinates") is not None:
        metadata.update(position_representation=state["config"]["position_representation"],
                        exported_position_coordinates="original generator-input tuple XY",
                        reference_geometry=state["road_coordinates"]["source"],
                        posthoc_road_snapping=False, real_history_at_inference=False)
    if state.get("road_increments") is not None:
        metadata.update(progress_generation=state["road_increments"], endpoint_correction=False,
                        accumulated_progress_clipping=False)
    if state["config"].get("progress_conditioning"):
        metadata.update(progress_conditioning=state["config"]["progress_spec"],
                        individual_future_positions_at_inference=False)
    result = export_trajectories(sequences, schema, output, metadata)
    for name in ("source_group", "source_cohort"):
        np.save(output / f"{name}.npy", np.full(len(result.arrays["states"]), source_group), allow_pickle=False)
    index = [
        {"trajectory_id": i, "candidate_id": f"{source_group}:{generator_id}:{seed}:{i}",
         "source_group": source_group, "source_cohort": source_group,
         "generator_id": generator_id, "seed": seed, "transitions": len(rows) - 1}
        for i, rows in enumerate(sequences)
    ]
    write_json(output / "trajectory_index.json", index)
    if save_decoded_sequences:
        lengths = np.asarray([len(rows) for rows in sequences], dtype=np.int64)
        np.savez_compressed(output / "decoded_sequences.npz", rows=np.concatenate(sequences),
                            offsets=np.r_[0, np.cumsum(lengths)], phase_durations=plans)
    return result
