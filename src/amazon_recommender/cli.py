"""Local pipeline commands. Only etl, sample, split, train, evaluate, select-popularity-window and select-grid
start Spark; none contacts MongoDB."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tomllib

from amazon_recommender.config import load_config
from amazon_recommender.config.experiment import parse_experiment, parse_model
from amazon_recommender.environment import inspect_environment
from amazon_recommender.artifacts import create_run
from amazon_recommender.models import MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="Inspect versions without starting Spark or MongoDB")
    validate = subcommands.add_parser("validate-config", help="Validate a profile without starting services")
    validate.add_argument("--config", type=Path, required=True)
    initialize = subcommands.add_parser("init-run", help="Snapshot config and hash local inputs; do not run jobs")
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--reviews", type=Path, required=True)
    initialize.add_argument("--metadata", type=Path, required=True)
    etl = subcommands.add_parser("etl", help="Run the ETL stage of a prepared run; starts Spark")
    etl.add_argument("--config", type=Path, required=True)
    etl.add_argument("--run", type=Path, required=True, help="Run directory created by init-run")
    sample = subcommands.add_parser("sample", help="Sample users of a run with a completed ETL stage; starts Spark")
    sample.add_argument("--config", type=Path, required=True)
    sample.add_argument("--run", type=Path, required=True, help="Run directory with a completed ETL stage")
    sample.add_argument("--user-fraction", type=float, help="Override experiment.user_fraction, e.g. for cohorts")
    split = subcommands.add_parser("split", help="Temporal split, k-core and mappings of a sample; starts Spark")
    split.add_argument("--config", type=Path, required=True, help="Spark resources; experiment settings come from the sample")
    split.add_argument("--run", type=Path, required=True, help="Run directory with a completed sample")
    split.add_argument("--sample", required=True, help="Sample label, e.g. fraction-0.2-seed-2221")
    split.add_argument("--cutoffs-from", type=Path, help="Reuse the cutoffs of another split's stage.json")
    split.add_argument("--refit-from", type=Path,
                       help="Train on another split's train + validation windows; its test window is kept")
    train = subcommands.add_parser("train", help="Fit one model on a split's training window; starts Spark")
    train.add_argument("--config", type=Path, required=True, help="Spark resources and [model] hyperparameters")
    train.add_argument("--run", type=Path, required=True, help="Run directory with a completed split")
    train.add_argument("--sample", required=True, help="Sample label, e.g. fraction-0.2-seed-2221")
    train.add_argument("--split", required=True, help="Split label, e.g. p0.8-0.1-0.1-k5-5")
    train.add_argument("--model", required=True, choices=sorted(MODELS))
    train.add_argument("--popularity-window-days", help='Override model.popularity_window_days: "all" or days')
    train.add_argument("--rank", type=int, help="Override model.rank (ALS)")
    train.add_argument("--regularization", type=float, help="Override model.regularization (ALS)")
    train.add_argument("--iterations", type=int, help="Override model.iterations (ALS)")
    evaluate = subcommands.add_parser("evaluate", help="Metrics of a trained model on a held-out window; starts Spark")
    evaluate.add_argument("--config", type=Path, required=True, help="Spark resources and bootstrap seed")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--sample", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--model-label", required=True,
                          help="Directory name under <split>/models/, or oracle-popularity (reference only)")
    evaluate.add_argument("--window", required=True, choices=("validation", "test"))
    evaluate.add_argument("--horizon-days", type=int, help="Only held-out rows within this many days of the fit end")
    evaluate.add_argument("--coverage", action="store_true",
                          help="Serving view: the popularity fallback on all held-out users, including unknown ones")
    select = subcommands.add_parser("select-popularity-window",
                                    help="Choose the popularity window from validation evaluations; starts Spark")
    select.add_argument("--config", type=Path, required=True)
    select.add_argument("--run", type=Path, required=True)
    select.add_argument("--sample", required=True)
    select.add_argument("--split", required=True)
    grid = subcommands.add_parser("select-grid", help="Choose ALS settings from validation evaluations; starts Spark")
    grid.add_argument("--config", type=Path, required=True)
    grid.add_argument("--run", type=Path, required=True)
    grid.add_argument("--sample", required=True)
    grid.add_argument("--split", required=True)
    grid.add_argument("--model", required=True, choices=("als", "bias-als"))
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = inspect_environment()
        print(json.dumps(report, indent=2))
        return int(bool(report["errors"]))
    try:
        config = load_config(args.config)
        if args.command == "init-run":
            directory = create_run(config, reviews=args.reviews, metadata=args.metadata)
            print(directory / "run.json")
            return 0
        if args.command == "etl":
            from amazon_recommender.etl import run_etl

            print(run_etl(config, args.run) / "stage.json")
            return 0
        if args.command == "sample":
            from amazon_recommender.sample import run_sample

            if args.user_fraction is not None:
                experiment = {**asdict(config.experiment), "split": list(config.experiment.split)}
                config = replace(config, experiment=parse_experiment({**experiment, "user_fraction": args.user_fraction}))
            print(run_sample(config, args.run) / "stage.json")
            return 0
        if args.command == "split":
            from amazon_recommender.split import run_split

            print(run_split(config, args.run, args.sample, cutoffs_from=args.cutoffs_from,
                            refit_from=args.refit_from) / "stage.json")
            return 0
        if args.command == "train":
            from amazon_recommender.train import run_train

            overrides = {key: getattr(args, key) for key in ("rank", "regularization", "iterations")
                         if getattr(args, key) is not None}
            if args.popularity_window_days is not None:
                window = args.popularity_window_days
                overrides["popularity_window_days"] = window if window == "all" else int(window)
            if overrides:
                config = replace(config, model=parse_model({**asdict(config.model), **overrides}))
            print(run_train(config, args.run, args.sample, args.split, args.model) / "stage.json")
            return 0
        if args.command == "evaluate":
            from amazon_recommender.evaluate import run_evaluate

            print(run_evaluate(config, args.run, args.sample, args.split, args.model_label, args.window,
                               horizon_days=args.horizon_days, coverage=args.coverage) / "stage.json")
            return 0
        if args.command == "select-grid":
            from amazon_recommender.evaluate import run_select_grid

            print(run_select_grid(config, args.run, args.sample, args.split, args.model))
            return 0
        if args.command == "select-popularity-window":
            from amazon_recommender.evaluate import run_select_popularity_window

            print(run_select_popularity_window(config, args.run, args.sample, args.split))
            return 0
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        print(f"Invalid configuration or input: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # e.g. the environment check before Spark starts
        print(f"Cannot run {args.command}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"config": config.as_dict(), "config_digest": config.digest}, indent=2))
    return 0
