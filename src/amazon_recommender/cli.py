"""Local development setup. No command starts Spark or MongoDB."""

import argparse
import json
from pathlib import Path
import sys
import tomllib

from amazon_recommender.config import load_config
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
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        print(f"Invalid configuration or input: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"config": config.as_dict(), "config_digest": config.digest}, indent=2))
    return 0
