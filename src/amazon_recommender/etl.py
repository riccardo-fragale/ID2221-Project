"""ETL stage: raw review/metadata JSONL to typed, deduplicated Parquet.

Outputs live in `<run>/etl/` and appear only once complete: everything is
written to `<run>/etl.partial/`, and the directory is renamed after
`stage.json` is saved. A missing `etl/stage.json` therefore means no usable ETL.

Dense integer mappings are not built here: the approved protocol derives them
from the training window after k-core filtering, so they belong to the split stage.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from amazon_recommender.config import ProjectConfig
from amazon_recommender.schemas import metadata_schema, review_schema

CORRUPT = "_corrupt_record"
# Review timestamps are Unix milliseconds. The data spans 1996-2023, so this window
# also rejects values in seconds (which would land in 1970).
MIN_TIMESTAMP_MS = 631_152_000_000  # 1990-01-01 UTC
MAX_TIMESTAMP_MS = 1_893_456_000_000  # 2030-01-01 UTC
REJECT_REASONS = ("malformed_record", "missing_field", "invalid_rating", "invalid_timestamp")


@contextmanager
def timed(timings: dict, step: str):
    """Record wall-clock seconds of a step; wrap completed Spark actions, not lazy plans."""
    start = time.monotonic()
    yield
    timings[step] = round(time.monotonic() - start, 3)


def _read_json(spark, path: Path, schema):
    from pyspark.sql.types import StringType, StructField

    return (
        spark.read.schema(schema.add(StructField(CORRUPT, StringType(), True)))
        .option("mode", "PERMISSIVE")
        .option("columnNameOfCorruptRecord", CORRUPT)
        .json(str(path))
    )


def classify_reviews(df):
    """Add `reject_reason`: the first failed check, or null for a valid review."""
    from pyspark.sql import functions as F

    def blank(column):
        return F.col(column).isNull() | (F.trim(F.col(column)) == "")

    return df.withColumn(
        "reject_reason",
        F.when(F.col(CORRUPT).isNotNull(), "malformed_record")
        .when(blank("user_id") | blank("parent_asin") | F.col("rating").isNull() | F.col("timestamp").isNull(),
              "missing_field")
        .when(~F.col("rating").between(1.0, 5.0), "invalid_rating")  # also rejects NaN
        .when(~F.col("timestamp").between(MIN_TIMESTAMP_MS, MAX_TIMESTAMP_MS), "invalid_timestamp"),
    )


def deduplicate_interactions(valid):
    """Keep the latest review per (user, item); on equal timestamps, the highest rating.

    Struct maximum orders by timestamp, then rating, so exact duplicates collapse too.
    Event time and year are in UTC (the session factory fixes the SQL time zone).
    """
    from pyspark.sql import functions as F

    latest = valid.groupBy("user_id", "parent_asin").agg(
        F.max(F.struct("timestamp", "rating")).alias("latest")
    )
    return latest.select(
        "user_id",
        "parent_asin",
        F.col("latest.rating").alias("rating"),
        F.col("latest.timestamp").alias("timestamp"),
    ).withColumn("event_time", F.expr("timestamp_millis(timestamp)")).withColumn("year", F.year("event_time"))


def clean_metadata(raw):
    """One title per item. Duplicated items take the smallest nonblank title, deterministically.

    Blank titles become null, so they never win over a real title and the item falls back to its ID.
    """
    from pyspark.sql import functions as F

    usable = raw.filter(F.col(CORRUPT).isNull() & F.col("parent_asin").isNotNull() & (F.trim("parent_asin") != ""))
    title = F.when(F.trim("title") != "", F.col("title"))
    return usable.groupBy("parent_asin").agg(F.min(title).alias("title"))


def run_etl(config: ProjectConfig, run_directory: Path) -> Path:
    """Run the ETL stage for a prepared run. Starts Spark."""
    run_directory = Path(run_directory).resolve(strict=True)
    manifest = json.loads((run_directory / "run.json").read_text())
    if manifest["config_digest"] != config.digest:
        raise ValueError("Configuration differs from the one snapshotted by init-run; start a new run")
    final, partial = run_directory / "etl", run_directory / "etl.partial"
    if final.exists():
        raise FileExistsError(f"ETL already completed for this run: {final}")
    inputs = {}
    for name in ("reviews", "metadata"):
        recorded = manifest["inputs"][name]
        path = Path(recorded["path"])
        # Size only: rehashing multi-GB inputs on every stage would be slow.
        if path.stat().st_size != recorded["bytes"]:
            raise ValueError(f"{name} input changed since init-run: {path}")
        inputs[name] = path
    if partial.exists():
        shutil.rmtree(partial)  # scratch from an interrupted attempt; never marked complete

    from pyspark.sql import functions as F

    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    spark = create_spark_session(config, app_name="amazon-recommender-etl")
    try:
        spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        parsed = classify_reviews(_read_json(spark, inputs["reviews"], review_schema()))
        # Two scans of the raw input (counts, then the write) avoid caching it in memory.
        with timed(timings, "reviews_classify_count"):
            counted = {row["reject_reason"]: row["count"] for row in parsed.groupBy("reject_reason").count().collect()}
        valid_count = counted.pop(None, 0)
        rejected = {reason: counted.pop(reason, 0) for reason in REJECT_REASONS}
        if counted:
            raise RuntimeError(f"Unexpected reject reasons: {counted}")

        interactions = deduplicate_interactions(parsed.filter(F.col("reject_reason").isNull()))
        # One shuffle by year gives one file per year instead of one per shuffle partition and year.
        with timed(timings, "reviews_deduplicate_write"):
            interactions.repartition("year").write.mode("error").partitionBy("year").parquet(
                str(partial / "interactions")
            )

        raw_metadata = _read_json(spark, inputs["metadata"], metadata_schema())
        with timed(timings, "metadata_count_write"):
            metadata_read = raw_metadata.count()
            clean_metadata(raw_metadata).write.mode("error").parquet(str(partial / "metadata"))

        written = spark.read.parquet(str(partial / "interactions"))
        titled = spark.read.parquet(str(partial / "metadata"))
        with timed(timings, "verification_counts"):
            interaction_count = written.count()
            items = written.select("parent_asin").distinct()
            years = written.agg(F.min("year"), F.max("year")).first()
            users = written.select("user_id").distinct().count()
            item_count = items.count()
            metadata_items = titled.count()
            titled_items = titled.filter(F.col("title").isNotNull())
            without_title = items.join(titled_items, "parent_asin", "left_anti").count()
            without_row = items.join(titled, "parent_asin", "left_anti").count()
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "etl",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": manifest["run_id"],
            "config_digest": manifest["config_digest"],
            "inputs": manifest["inputs"],
            "timestamp_bounds_ms": [MIN_TIMESTAMP_MS, MAX_TIMESTAMP_MS],
            "reviews": {
                "rows_read": valid_count + sum(rejected.values()),
                "rejected": rejected,
                "valid": valid_count,
                "duplicates_removed": valid_count - interaction_count,
                "interactions": interaction_count,
                "users": users,
                "items": item_count,
                "years": [years[0], years[1]],
            },
            "metadata": {
                "rows_read": metadata_read,
                "items_with_metadata": metadata_items,
                "interaction_items_without_metadata_row": without_row,
                "interaction_items_without_title": without_title,
            },
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            "outputs": {"interactions": "interactions", "metadata": "metadata"},
        }
    finally:
        spark.stop()
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final

