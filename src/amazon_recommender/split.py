"""Split stage: global temporal split, training-only k-core, and dense integer mappings.

Input is one completed sample (`<run>/samples/<label>/`). Its `stage.json` config supplies
the experiment settings (split proportions, k-core thresholds); the CLI config supplies only
the Spark resources. Output goes to `<run>/samples/<label>/splits/<split-label>/`, completed
by the same `.partial` rename as the other stages.

Split: one global chronological split by interaction count. With N interactions,
`train_cutoff` is the smallest timestamp whose cumulative count reaches `split[0] * N`, and
`validation_cutoff` the smallest reaching `(split[0] + split[1]) * N`. Train holds
timestamps <= train_cutoff, validation (train_cutoff, validation_cutoff], test the rest, so
a group of identical timestamps always falls in one window (the earlier one). Instead of
proportions, the two cutoffs can be copied from another split's `stage.json`, so that
several cohorts share frozen boundaries.

K-core: users and items with fewer than `user_min_reviews` / `item_min_reviews` training
interactions are removed, and degrees are recomputed until nothing is below threshold.
Validation and test keep only rows whose user and item both survive in training.

Mappings: retained training users and items get dense 0-based indices in ascending order of
their original string ID, so they do not depend on partitioning.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from urllib.parse import urlparse

from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import timed

MAX_INDEX = 2**31 - 2  # Spark ALS ids are signed 32-bit integers
WINDOWS = ("train", "validation", "test")


def split_label(split, user_min: int, item_min: int, cutoffs_source: str | None = None) -> str:
    policy = f"cutoffs-{cutoffs_source[:12]}" if cutoffs_source else "p" + "-".join(f"{v:g}" for v in split)
    return f"{policy}-k{user_min}-{item_min}"


def temporal_cutoffs(df, split) -> tuple[int, int]:
    """Exact count-based cutoffs; ties on a timestamp stay together."""
    from pyspark.sql import Window, functions as F

    total = df.count()
    if total == 0:
        raise ValueError("Cannot split an empty sample")
    per_timestamp = df.groupBy("timestamp").count()
    # One ordered pass over distinct timestamps (small rows); Spark moves them to one partition.
    cumulative = per_timestamp.withColumn(
        "cumulative", F.sum("count").over(Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, 0))
    )

    def first_reaching(share: float):
        target = max(1, math.ceil(share * total - 1e-9))  # tolerance: 0.8 + 0.1 == 0.9000000000000001
        return F.min(F.when(F.col("cumulative") >= target, F.col("timestamp")))

    # Both cutoffs in one pass over the cumulative counts.
    row = cumulative.agg(first_reaching(split[0]), first_reaching(split[0] + split[1])).first()
    train_cutoff = row[0]
    validation_cutoff = train_cutoff if split[1] == 0 else row[1]
    return train_cutoff, validation_cutoff


def assign_windows(df, train_cutoff: int, validation_cutoff: int) -> dict:
    from pyspark.sql import functions as F

    ts = F.col("timestamp")
    return {
        "train": df.filter(ts <= train_cutoff),
        "validation": df.filter((ts > train_cutoff) & (ts <= validation_cutoff)),
        "test": df.filter(ts > validation_cutoff),
    }


def k_core(train, user_min: int, item_min: int):
    """Iterate until every user and item has enough interactions. Returns (core, iterations)."""
    from pyspark.sql import functions as F

    current = train.select("user_id", "parent_asin", "rating", "timestamp").checkpoint()
    size = current.count()
    iterations = []  # removal passes only; the final check that removes nothing is not counted
    while True:
        users = current.groupBy("user_id").count()
        items = current.groupBy("parent_asin").count()
        low_users = users.filter(F.col("count") < user_min).count()
        low_items = items.filter(F.col("count") < item_min).count()
        if low_users == 0 and low_items == 0:
            return current, iterations
        # Reliable checkpoint every pass: cuts the lineage, which otherwise grows each iteration.
        current = (
            current.join(users.filter(F.col("count") >= user_min).select("user_id"), "user_id")
            .join(items.filter(F.col("count") >= item_min).select("parent_asin"), "parent_asin")
            .checkpoint()
        )
        before, size = size, current.count()
        iterations.append({
            "users_removed": low_users,
            "items_removed": low_items,
            "interactions_removed": before - size,
        })


def dense_mapping(df, column: str, index: str):
    """0-based indices in ascending order of the original ID; independent of partitioning."""
    from pyspark.sql import Window, functions as F

    return df.select(column).distinct().withColumn(index, F.row_number().over(Window.orderBy(column)) - 1)


def mapping_summary(mapping, column: str, index: str) -> dict:
    """Count, max index and an order-independent digest, computed without collecting the mapping."""
    from pyspark.sql import functions as F

    row = mapping.agg(
        F.count("*").alias("count"),
        F.max(index).alias("max_index"),
        F.countDistinct(index).alias("distinct_indices"),
        F.sum(F.xxhash64(F.col(column), F.col(index)).cast("decimal(38,0)")).alias("hash_sum"),
    ).first()
    if row["count"] and (row["max_index"] != row["count"] - 1 or row["distinct_indices"] != row["count"]):
        raise RuntimeError(f"{column} mapping is not dense and unique")
    if row["count"] and row["max_index"] > MAX_INDEX:
        raise RuntimeError(f"{column} mapping exceeds the signed 32-bit range")
    return {"count": row["count"], "max_index": row["max_index"], "digest": f"{row['count']}:{row['hash_sum']}"}


def project(window, users, items, train) -> tuple:
    """Keep held-out rows whose user and item are in the training vocabulary; count the drops.

    `train` is the training window before k-core, so missing users and items are split into
    those never seen in training (cold starts) and those removed by the k-core.
    """
    from pyspark.sql import functions as F

    def flag(frame, column, name):
        return frame.select(column).distinct().withColumn(name, F.lit(True))

    flagged = (
        window.join(flag(users, "user_id", "_known_user"), "user_id", "left")
        .join(flag(items, "parent_asin", "_known_item"), "parent_asin", "left")
        .join(flag(train, "user_id", "_seen_user"), "user_id", "left")
        .join(flag(train, "parent_asin", "_seen_item"), "parent_asin", "left")
    )
    known_user, known_item = F.col("_known_user").isNotNull(), F.col("_known_item").isNotNull()
    seen_user, seen_item = F.col("_seen_user").isNotNull(), F.col("_seen_item").isNotNull()
    counts = flagged.agg(
        F.count("*").alias("rows"),
        F.sum((known_user & known_item).cast("long")).alias("kept"),
        F.sum((~known_user & known_item).cast("long")).alias("unknown_user_only"),
        F.sum((known_user & ~known_item).cast("long")).alias("unknown_item_only"),
        F.sum((~known_user & ~known_item).cast("long")).alias("unknown_user_and_item"),
        F.countDistinct(F.when(~seen_user, F.col("user_id"))).alias("users_without_training_history"),
        F.countDistinct(F.when(seen_user & ~known_user, F.col("user_id"))).alias("users_removed_by_k_core"),
        F.countDistinct(F.when(~seen_item, F.col("parent_asin"))).alias("items_without_training_history"),
        F.countDistinct(F.when(seen_item & ~known_item, F.col("parent_asin"))).alias("items_removed_by_k_core"),
    ).first().asDict()
    kept = flagged.filter(known_user & known_item).select(window.columns)
    return kept, {key: value or 0 for key, value in counts.items()}


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def run_split(config: ProjectConfig, run_directory: Path, sample: str, *, cutoffs_from: Path | None = None) -> Path:
    """Split one completed sample of a run. Starts Spark."""
    run_directory = Path(run_directory).resolve(strict=True)
    sample_directory = run_directory / "samples" / sample
    sample_stage_path = sample_directory / "stage.json"
    if not sample_stage_path.is_file():
        raise FileNotFoundError(f"Sample is not complete: {sample_directory}")
    sample_stage = json.loads(sample_stage_path.read_text())
    experiment = sample_stage["config"]["experiment"]
    split, user_min, item_min = experiment["split"], experiment["user_min_reviews"], experiment["item_min_reviews"]
    source = None
    if cutoffs_from is not None:
        cutoffs_from = Path(cutoffs_from).resolve(strict=True)
        source_bytes = cutoffs_from.read_bytes()
        frozen = json.loads(source_bytes)["cutoffs_ms"]
        source = {"path": str(cutoffs_from), "sha256": hashlib.sha256(source_bytes).hexdigest()}
    label = split_label(split, user_min, item_min, source and source["sha256"])
    final, partial = sample_directory / "splits" / label, sample_directory / "splits" / f"{label}.partial"
    if final.exists():
        raise FileExistsError(f"Split already completed for this sample: {final}")
    if partial.exists():
        shutil.rmtree(partial)  # scratch from an interrupted attempt; never marked complete
    partial.parent.mkdir(exist_ok=True)

    from pyspark.sql import functions as F

    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    spark = create_spark_session(config, app_name="amazon-recommender-split")
    # Spark does not delete reliable checkpoints; remove this session's directory at the end.
    checkpoints = Path(urlparse(spark.sparkContext.getCheckpointDir()).path)
    try:
        spark.conf.set("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        interactions = spark.read.parquet(str(run_directory / sample_stage["outputs"]["interactions"])).select(
            "user_id", "parent_asin", "rating", "timestamp"
        )
        with timed(timings, "cutoffs"):
            if source:
                train_cutoff, validation_cutoff = frozen["train"], frozen["validation"]
            else:
                train_cutoff, validation_cutoff = temporal_cutoffs(interactions, split)
        windows = assign_windows(interactions, train_cutoff, validation_cutoff)
        with timed(timings, "window_counts"):
            before = {name: frame.count() for name, frame in windows.items()}
        total = sum(before.values())

        with timed(timings, "k_core"):
            core, iterations = k_core(windows["train"], user_min, item_min)

        with timed(timings, "mappings_write"):
            users = dense_mapping(core, "user_id", "user_index")
            items = dense_mapping(core, "parent_asin", "item_index")
            users.write.mode("error").parquet(str(partial / "users"))
            items.write.mode("error").parquet(str(partial / "items"))
            users = spark.read.parquet(str(partial / "users"))
            items = spark.read.parquet(str(partial / "items"))
            mappings = {
                "users": mapping_summary(users, "user_id", "user_index"),
                "items": mapping_summary(items, "parent_asin", "item_index"),
            }

        with timed(timings, "windows_write"):
            core.join(users, "user_id").join(items, "parent_asin").write.mode("error").parquet(str(partial / "train"))
            projection = {}
            for name in ("validation", "test"):
                kept, projection[name] = project(windows[name], users, items, windows["train"])
                kept.join(users, "user_id").join(items, "parent_asin").write.mode("error").parquet(
                    str(partial / name)
                )

        with timed(timings, "verification_counts"):
            after = {name: spark.read.parquet(str(partial / name)).count() for name in WINDOWS}
        timings["total"] = round(time.monotonic() - started, 3)
        resources = describe_resources(spark, config)
        stage = {
            "stage": "split",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": sample_stage["run_id"],
            "sample": sample,
            "sample_stage_sha256": hashlib.sha256(sample_stage_path.read_bytes()).hexdigest(),
            "purpose": experiment["purpose"],
            "policy": {
                "method": "global chronological split by interaction count; equal timestamps share a window",
                "split": split if source is None else None,
                "cutoffs_from": source,
                "user_min_reviews": user_min,
                "item_min_reviews": item_min,
            },
            "cutoffs_ms": {"train": train_cutoff, "validation": validation_cutoff},
            "cutoffs_utc": {"train": _iso(train_cutoff), "validation": _iso(validation_cutoff)},
            "before_filtering": {
                "interactions": before,
                "proportions": {name: count / total for name, count in before.items()},
            },
            "k_core": {
                "iterations": iterations,
                "converged_after": len(iterations),
                "retained_interactions": after["train"],
                "retained_fraction": after["train"] / before["train"] if before["train"] else None,
            },
            "projection": projection,
            "mappings": mappings,
            "interactions": after,
            "proportions": {name: count / sum(after.values()) if sum(after.values()) else None
                            for name, count in after.items()},
            "resources": resources,
            "timings_seconds": timings,
            # Relative to the split directory.
            "outputs": {name: name for name in (*WINDOWS, "users", "items")},
        }
    finally:
        spark.stop()
        shutil.rmtree(checkpoints, ignore_errors=True)
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final
