"""Run from the repository root: python -m synthetic_generation train-diffusion|generate."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path


def main(argv=None):
    from .diffusion_train import DiffusionConfig, fit_diffusion
    from .data import Tuples
    from .train import generate, seed_all
    parser = argparse.ArgumentParser(description="Cohort-specific temporal TabDiff driving synthesis")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Validate tuple continuity and reward structure")
    inspect.add_argument("--data-dir", type=Path, required=True)
    train = sub.add_parser("train-diffusion", help="Fit complete-drive temporal TabDiff")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--source-group", choices=("ordinary", "senior"), required=True)
    train.add_argument("--input-manifest", type=Path)
    train.add_argument("--schema", type=Path)
    train.add_argument("--split-groups", type=Path)
    defaults = DiffusionConfig()
    for item in fields(defaults):
        default = getattr(defaults, item.name)
        parse = (lambda value: json.loads(Path(value).read_text())) if isinstance(default, dict) else type(default)
        train.add_argument("--" + item.name.replace("_", "-"), type=parse, default=default)
    gen = sub.add_parser("generate", help="Generate raw decoded trajectories without source data")
    gen.add_argument("--checkpoint", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--source-group", choices=("ordinary", "senior"))
    gen.add_argument("--count", type=int, default=32)
    gen.add_argument("--seed", type=int, default=2026)
    gen.add_argument("--batch-size", type=int, default=8)
    gen.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    gen.add_argument("--threads", type=int, default=2)
    gen.add_argument("--categorical-mode", choices=("sample", "argmax"))
    gen.add_argument("--sampling-steps", type=int)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    seed_all(args.get("seed", 42), args.get("threads", 2))
    if command == "inspect":
        print(json.dumps(Tuples.load(**{("path" if k == "data_dir" else k): v for k, v in args.items()}).summary(), indent=2))
    elif command == "train-diffusion":
        config = DiffusionConfig(**{item.name: args.pop(item.name) for item in fields(defaults)})
        schema_path, groups_path = args.pop("schema"), args.pop("split_groups")
        import torch
        checkpoint = Path(__file__).resolve().parents[1] / "reproduction_inputs" / "diffusion" / f"{args['source_group']}_s101.pt"
        coordinate_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not config.road_spec:
            config.road_spec = coordinate_state["config"]["road_spec"]
        print(fit_diffusion(**args, config=config,
                            coordinate_state=coordinate_state,
                            spec=json.loads(schema_path.read_text()) if schema_path else None,
                            split_groups=json.loads(groups_path.read_text()) if groups_path else None))
    else:
        args.pop("threads")
        print(json.dumps(generate(**args).summary(), indent=2))


if __name__ == "__main__":
    main()
