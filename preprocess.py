"""Validate synthetic transitions and export model/generator tuple inputs."""
import numpy as np
from . import data
from .utils import new_output, write_json


def prepare_tuples(output_dir):
    output = new_output(output_dir)
    report = []
    for group in data.GROUPS:
        d = data.load_data(group, "all")
        np.testing.assert_array_equal(data.generator_to_model(d["generator_states"], group), d["states"])
        for frame in ("model", "generator"):
            folder = output / frame / group
            folder.mkdir(parents=True)
            for key, value in d.items():
                if key in ("generator_states", "generator_next_states", "rewards"):
                    continue
                if key in ("states", "next_states") and frame == "generator":
                    value = d[f"generator_{key}"]
                np.save(folder / f"{key}.npy", value, allow_pickle=False)
        report.append(dict(group=group, trajectories=len(data.records(group)), transitions=len(d["states"])))
    write_json(output / "manifest.json", dict(synthetic=True, cohorts=report,
                source_state_frame="fixed_model", generator_state_frame="generator_train_standardized"))
    return report
