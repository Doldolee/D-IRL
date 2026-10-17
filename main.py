"""Portable synthetic-data workflow. Run: python main.py --help."""
import argparse
import json

if not __package__:
    from _bootstrap import activate
    __package__ = activate()


def main(argv=None):
    from . import data
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Verify bundled files and synthetic/model identities")
    sub.add_parser("ope-results", help="Print the fixed OPE result table in JSON")
    prep = sub.add_parser("preprocess", help="Validate and export model/generator synthetic tuples")
    prep.add_argument("--output", required=True)
    train = sub.add_parser("train", help="Demonstrate the final IQ training recipe on synthetic data")
    train.add_argument("--group", choices=data.GROUPS, required=True)
    train.add_argument("--output", required=True)
    for flag, default in (("steps", 20000), ("seed", 2), ("batch-size", 256),
                          ("log-interval", 100), ("save-interval", 1000)):
        train.add_argument("--" + flag, type=int, default=default)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--bc-init", help="BC state_dict for the ordinary-group actor warm start")
    bc = sub.add_parser("train-bc", help="Fit BC on the released synthetic training trajectories")
    bc.add_argument("--group", choices=data.GROUPS, required=True)
    bc.add_argument("--epochs", type=int, default=10)
    bc.add_argument("--output", required=True)
    score = sub.add_parser("ta", help="Recompute TA-Adv and TA-dSNIPS from states/actions")
    score.add_argument("--output", required=True)
    score.add_argument("--limit", type=int)
    score.add_argument("--recover", action="store_true", help="Also recompute recovered rewards")
    fit = sub.add_parser("fit-fqe", help="Fit a new independent FQE using synthetic training inputs")
    fit.add_argument("--policy-key", choices=data.POLICY_GROUPS, required=True)
    fit.add_argument("--iterations", type=int, default=600)
    fit.add_argument("--updates", type=int, default=50)
    fit.add_argument("--seed", type=int, default=101)
    fit.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-fqe", help="Evaluate a supplied FQE on synthetic validation trajectories")
    evaluate.add_argument("--policy-key", choices=data.POLICY_GROUPS, required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    plot = sub.add_parser("figures", help="Render the four saved public comparison figures")
    plot.add_argument("--panel", choices=("all", "tsne", "position", "ta"), default="all")
    plot.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "check":
        from .utils import verify_inputs
        print(json.dumps(verify_inputs(), indent=2))
    elif args.command == "ope-results":
        print(json.dumps(data.read_json(data.INPUTS / "ope_results.json"), indent=2))
    elif args.command == "preprocess":
        from .preprocess import prepare_tuples
        print(json.dumps(prepare_tuples(args.output), indent=2))
    elif args.command == "train":
        from .train import cohort_config, train_iq
        keys = ("steps", "seed", "batch_size", "log_interval", "save_interval", "device", "bc_init")
        train_iq(args.group, args.output, cohort_config(args.group, **{k:getattr(args,k) for k in keys}))
    elif args.command == "train-bc":
        import torch
        from .buffer import ReplayBuffer
        from .train import train_squashed_bc
        from .utils import new_output, write_json
        torch.set_num_threads(2)
        torch.manual_seed(42)
        out = new_output(args.output)
        buffer = ReplayBuffer(data.load_data(args.group, "train"), 42)
        model = train_squashed_bc(buffer, 51, 3, epochs=args.epochs)
        torch.save(model.state_dict(), out / "policy.pth")
        write_json(out / "protocol.json", dict(synthetic=True, group=args.group, split="train", epochs=args.epochs))
    elif args.command == "ta":
        from .ta import compute_public_ta
        compute_public_ta(args.output, args.limit, args.recover)
    elif args.command == "fit-fqe":
        from .ope import fit_fqe
        fit_fqe(args.policy_key, args.output, args.seed, args.iterations, args.updates)
    elif args.command == "evaluate-fqe":
        import torch
        from .ope import FQE, evaluate_fqe, bootstrap_ci
        from .utils import new_output, write_json
        torch.set_num_threads(2)
        q = FQE().eval().requires_grad_(False)
        q.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
        ids, values = evaluate_fqe(q, data.load_fqe_data("pooled", "valid"),
                                  data.load_fqe_parameters(args.policy_key, "valid"))
        values = values * 100
        _, ci = bootstrap_ci(values)
        write_json(new_output(args.output) / "estimates.json", dict(synthetic=True, split="valid",
                    trajectory_ids=ids, values=values, mean=values.mean(0), ci=ci))
    else:
        from .plot import render_figures
        render_figures(args.output, args.panel)


if __name__ == "__main__":
    main()
