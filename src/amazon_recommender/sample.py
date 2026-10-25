"""Sampling stage: a reproducible user-level subset of the ETL interactions.

A user is kept when a hash of (seed, original user id) falls below
`experiment.user_fraction`, and every kept user retains the full history. The
choice depends only on the id and seed, so it needs no shuffle, does not change
when other users are added, and is identical across runs with the same seed.
Samples with the same seed are nested: a smaller fraction keeps a subset of the
users of a larger one. With fraction 1 nothing is copied: `stage.json` points at
the ETL interactions.

One completed ETL serves several samples: each fraction and seed gets its own
`<run>/samples/<label>/`, completed by the same rename as the ETL stage. The
sample's own configuration may differ from the run snapshot; it is recorded,
with the SHA-256 of the ETL `stage.json` it consumed.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import timed

METHOD = "pmod(xxhash64(cast(seed as bigint), user_id as string), 2^32) / 2^32 < user_fraction"
HASH_RANGE = 2**32


def sample_label(fraction: float, seed: int) -> str:
    return f"fraction-{fraction:g}-seed-{seed}"


def select_users(df, fraction: float, seed: int):
    """Keep all rows of users whose hash bucket falls below `fraction`."""
    from pyspark.sql import functions as F

    # The hash depends on the literal's type: pin both, so METHOD fully specifies membership.
    key = F.xxhash64(F.lit(seed).cast("long"), F.col("user_id").cast("string"))
    bucket = F.pmod(key, F.lit(HASH_RANGE)) / HASH_RANGE
    return df.filter(bucket < fraction)


def run_sample(config: ProjectConfig, run_directory: Path) -> Path:
    """Sample users from the completed ETL stage of a run. Starts Spark."""
    run_directory = Path(run_directory).resolve(strict=True)
    manifest = json.loads((run_directory / "run.json").read_text())
    etl_stage = run_directory / "etl/stage.json"
    if not etl_stage.is_file():
        raise FileNotFoundError(f"ETL stage is not complete for this run: {etl_stage.parent}")
    fraction, seed = config.experiment.user_fraction, config.seed
    label = sample_label(fraction, seed)
    final, partial = run_directory / "samples" / label, run_directory / "samples" / f"{label}.partial"
    if final.exists():
        raise FileExistsError(f"Sample already completed for this run: {final}")
    if partial.exists():
        shutil.rmtree(partial)  # scratch from an interrupted attempt; never marked complete
    partial.parent.mkdir(exist_ok=True)

    from pyspark.sql import functions as F

    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    spark = create_spark_session(config, app_name="amazon-recommender-sample")
    try:
        spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        source = spark.read.parquet(str(run_directory / "etl/interactions"))
        with timed(timings, "count_before"):
            before = source.agg(F.count("*").alias("interactions"), F.countDistinct("user_id").alias("users")).first()
        if fraction == 1:
            interactions_path, after = "etl/interactions", before
            partial.mkdir()
        else:
            with timed(timings, "select_write"):
                select_users(source, fraction, seed).repartition("year").write.mode("error").partitionBy(
                    "year"
                ).parquet(str(partial / "interactions"))
            interactions_path = f"samples/{label}/interactions"
            with timed(timings, "count_after"):
                after = (
                    spark.read.parquet(str(partial / "interactions"))
                    .agg(F.count("*").alias("interactions"), F.countDistinct("user_id").alias("users"))
                    .first()
                )
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "sample",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": manifest["run_id"],
            "config_digest": config.digest,
            "config": config.as_dict(),
            "etl_stage_sha256": hashlib.sha256(etl_stage.read_bytes()).hexdigest(),
            "purpose": config.experiment.purpose,
            "method": METHOD,
            "seed": seed,
            "user_fraction": fraction,
            "dataset": manifest["inputs"],  # input paths, sizes and SHA-256 identify the dataset
            "users": {"before": before["users"], "after": after["users"],
                      "achieved_fraction": after["users"] / before["users"] if before["users"] else None},
            "interactions": {"before": before["interactions"], "after": after["interactions"]},
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            # Relative to the run directory.
            "outputs": {"interactions": interactions_path, "metadata": "etl/metadata"},
        }
    finally:
        spark.stop()
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final
