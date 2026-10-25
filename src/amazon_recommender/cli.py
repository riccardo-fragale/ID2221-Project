"""Local pipeline commands. Only etl and sample start Spark; none contacts MongoDB."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tomllib

from amazon_recommender.config import load_config
from amazon_recommender.config.experiment import parse_experiment
from amazon_recommender.environment import inspect_environment
from amazon_recommender.artifacts import create_run


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
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        print(f"Invalid configuration or input: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:  # e.g. the environment check before Spark starts
        print(f"Cannot run {args.command}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"config": config.as_dict(), "config_digest": config.digest}, indent=2))
    return 0
