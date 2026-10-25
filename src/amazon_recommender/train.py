"""Train stage: fit one model on a completed split's training window and save it.

Input is `<run>/samples/<sample>/splits/<split>/`. The model hyperparameters and Spark
resources come from the CLI config ([model], [spark]); the split's experiment settings are
not changed. Output: `<split>/models/<label>/` with the saved model (`model/`) and
`stage.json`, completed by the same `.partial` rename as the other stages. The label names
the model and its parameters, so different settings never overwrite each other.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import timed
from amazon_recommender.models import MODELS, create_model


def model_label(name: str, params: dict) -> str:
    return "-".join([name, *(f"{key}{value}" if isinstance(value, str) else f"{key}{value:g}"
                             for key, value in params.items())])


def run_train(config: ProjectConfig, run_directory: Path, sample: str, split: str, model_name: str) -> Path:
    """Fit and save one model on one completed split of a run. Starts Spark."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}; choose one of: {', '.join(MODELS)}")
    run_directory = Path(run_directory).resolve(strict=True)
    split_directory = run_directory / "samples" / sample / "splits" / split
    split_stage_path = split_directory / "stage.json"
    if not split_stage_path.is_file():
        raise FileNotFoundError(f"Split is not complete: {split_directory}")
    split_bytes = split_stage_path.read_bytes()
    split_stage = json.loads(split_bytes)
    model = create_model(model_name, config.model, seed=config.seed)
    label = model_label(model_name, model.params)
    final, partial = split_directory / "models" / label, split_directory / "models" / f"{label}.partial"
    if final.exists():
        raise FileExistsError(f"Model already trained for this split: {final}")
    if partial.exists():
        shutil.rmtree(partial)  # scratch from an interrupted attempt; never marked complete
    partial.mkdir(parents=True)
    model.provenance = {
        "run_id": split_stage["run_id"],
        "sample": sample,
        "split": split,
        "split_stage_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "mappings": split_stage["mappings"],
    }

    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    spark = create_spark_session(config, app_name=f"amazon-recommender-train-{model_name}")
    # Fitting checkpoints its tables; remove this session's checkpoint directory at the end.
    checkpoints = Path(urlparse(spark.sparkContext.getCheckpointDir()).path)
    try:
        train = spark.read.parquet(str(split_directory / split_stage["outputs"]["train"]))
        with timed(timings, "fit"):
            model.fit(train)
        with timed(timings, "save"):
            model.save(partial / "model")
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "train",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **model.provenance,
            "purpose": split_stage["purpose"],
            "model": model.metadata(),
            "config": config.as_dict(),
            "config_digest": config.digest,
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            # Relative to the model directory.
            "outputs": {"model": "model"},
        }
    finally:
        spark.stop()
        shutil.rmtree(checkpoints, ignore_errors=True)
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final
