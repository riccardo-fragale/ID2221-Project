"""Local pipeline commands. Only etl, sample, split, train, recommend, materialize, evaluate,
select-popularity-window, select-grid, benchmark and publish start Spark; only publish and dashboard
contact MongoDB, and dashboard never starts Spark."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
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
    recommend = subcommands.add_parser("recommend", help="Top-k lists for all training users of a model; starts Spark")
    recommend.add_argument("--config", type=Path, required=True, help="Spark resources and model.recommendation_count")
    recommend.add_argument("--run", type=Path, required=True)
    recommend.add_argument("--sample", required=True)
    recommend.add_argument("--split", required=True)
    recommend.add_argument("--model-label", required=True, help="Directory name under <split>/models/")
    materialize = subcommands.add_parser(
        "materialize", help="Top-k serving lists in original IDs for every sample user, with fallbacks; starts Spark")
    materialize.add_argument("--config", type=Path, required=True, help="Spark resources and model.recommendation_count")
    materialize.add_argument("--run", type=Path, required=True)
    materialize.add_argument("--sample", required=True)
    materialize.add_argument("--split", required=True)
    materialize.add_argument("--model-label", required=True, help="Served model, a directory under <split>/models/")
    materialize.add_argument("--fallback-label", required=True,
                             help="Popularity model under <split>/models/ for users without a model list")
    materialize.add_argument("--batch-users", type=int, default=500_000, help="Users per restartable batch")
    publish = subcommands.add_parser("publish", help="Publish a materialization to MongoDB ($MONGODB_URI); starts Spark")
    publish.add_argument("--config", type=Path, required=True, help="Spark resources and the [mongo] database")
    publish.add_argument("--serving", type=Path, required=True, help="<split>/serving/<model-label>")
    publish.add_argument("--metadata", type=Path, required=True, help="Item titles, e.g. <run>/etl/metadata")
    dashboard = subcommands.add_parser("dashboard", help="Streamlit lookup and results of the active publication "
                                       "($MONGODB_URI); never starts Spark")
    dashboard.add_argument("--config", type=Path, required=True, help="The [mongo] database")
    dashboard.add_argument("--port", type=int, default=8501)
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
    benchmark = subcommands.add_parser("benchmark",
                                       help="Fixed-resource input-scaling benchmark; starts Spark once per stage")
    benchmark.add_argument("--config", type=Path, required=True, help="Spark resources, identical for every stage")
    benchmark.add_argument("--run", type=Path, required=True, help="Run directory with a completed ETL stage")
    benchmark.add_argument("--label", required=True, help="Output name under <run>/benchmarks/")
    benchmark.add_argument("--reference-split", type=Path, required=True,
                           help="stage.json of the split whose cutoffs every fraction uses")
    benchmark.add_argument("--model-stage", type=Path, action="append", required=True,
                           help="stage.json of a trained model with frozen settings; repeat for each model")
    benchmark.add_argument("--fractions", type=float, nargs="+", required=True)
    benchmark.add_argument("--repetitions", type=int, default=1)
    benchmark.add_argument("--cache", choices=("warm", "cold"), default="warm",
                           help="Declared page-cache state; the runner never drops caches")
    benchmark.add_argument("--stage-timeout", type=float, default=3600, help="Seconds before a stage counts as timed out")
    benchmark.add_argument("--delete-outputs", action="store_true",
                           help="Delete each fraction's data after measuring it; stage.json files are kept")
    report = subcommands.add_parser("benchmark-report", help="Scaling tables and figure from a benchmark's saved files")
    report.add_argument("--benchmark", type=Path, required=True, help="<run>/benchmarks/<label>")
    args = parser.parse_args(argv)
    if args.command == "benchmark-report":
        from amazon_recommender.benchmark import run_benchmark_report

        try:
            print(run_benchmark_report(args.benchmark))
        except (OSError, ValueError) as exc:
            print(f"Invalid benchmark: {exc}", file=sys.stderr)
            return 2
        return 0
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
        if args.command == "recommend":
            from amazon_recommender.recommend import run_recommend

            print(run_recommend(config, args.run, args.sample, args.split, args.model_label) / "stage.json")
            return 0
        if args.command == "materialize":
            from amazon_recommender.materialize import run_materialize

            print(run_materialize(config, args.run, args.sample, args.split, args.model_label, args.fallback_label,
                                  batch_users=args.batch_users) / "stage.json")
            return 0
        if args.command == "publish":
            from amazon_recommender.publish import publish

            print(publish(config, args.serving, args.metadata))
            return 0
        if args.command == "dashboard":
            from amazon_recommender.publish import mongo_uri

            mongo_uri(config)  # fail here, not in the browser, if the URI is not set
            app = Path(__file__).resolve().parent / "dashboard" / "app.py"
            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port",
                                    str(args.port), "--", "--config", str(args.config.resolve())])
        if args.command == "benchmark":
            from amazon_recommender.benchmark import run_benchmark

            print(run_benchmark(args.config, args.run, label=args.label, reference_split=args.reference_split,
                                model_stages=args.model_stage, fractions=args.fractions,
                                repetitions=args.repetitions, cache=args.cache, stage_timeout=args.stage_timeout,
                                keep_outputs=not args.delete_outputs) / "manifest.json")
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
