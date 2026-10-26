"""Materialize stage (P7.1): top-k serving lists for every user of a sample, in original IDs.

Population: every user of the sample (all windows), typed as in the coverage view:
- `known`: in the split's user mapping; ranked by the served model (`rec_type` "model");
- `removed_by_k_core`: has training-window history, removed by the k-core;
- `new`: no interaction up to the train cutoff.
Removed and new users, and any known user the model returns no list for (no factors), get the
popularity fallback (`rec_type` "fallback", P0.5). Seen items are the user's interactions up to
the split's train cutoff, before the k-core (for a known user these are exactly its training
rows); they are never recommended. Both models rank the split's training catalog exactly, so
every list has min(k, unseen catalog items) entries; a shorter list means the catalog ran out
(`catalog_exhausted`), and the audit refuses anything else.

Users get serving indices (known users keep their split index; the others follow in ascending
user_id order) and are processed in batches of `batch_users` consecutive indices. Each batch is
written to `pending/`, moved into the partitioned outputs and then marked `batches/<n>.json`
(rows, timings); an interrupted run resumes from the completed batches if the arguments match
`plan.json`. Nothing is collected on the driver except the ALS item matrix (see
`rank_by_factors`) and per-batch counts.

Output: `<split>/serving/<model-label>/`, completed by rename, with `users/` (one row per user:
user_id, user_type, rec_type, approximate, unseen_items, length), `recommendations/` (user_id,
rank, parent_asin, score; partitioned by batch), `batches/` and `stage.json` with the coverage
audit. The publisher builds one document per `users/` row, so empty lists are kept.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import timed

USER_TYPES = ("known", "removed_by_k_core", "new")
REC_TYPES = ("model", "fallback")
DEFAULT_BATCH_USERS = 500_000


def seen_items(raw, items, users, train_cutoff: int):
    """(user_index, item_index) of catalog items each user rated up to the train cutoff, before the k-core."""
    from pyspark.sql import functions as F

    return (raw.filter(F.col("timestamp") <= train_cutoff).join(items, "parent_asin")
            .join(users.select("user_id", "user_index"), "user_id").select("user_index", "item_index").distinct())


def serving_users(raw, known, items, train_cutoff: int, batch_users: int):
    """One row per sample user: user_id, user_index, user_type, unseen_items, batch."""
    from pyspark.sql import Window, functions as F

    pre_cutoff = raw.filter(F.col("timestamp") <= train_cutoff)
    offset = known.agg(F.max("user_index")).first()[0] + 1
    others = raw.select("user_id").distinct().join(known.select("user_id"), "user_id", "left_anti")
    # A global row_number moves the other users (about 5.9M on the full category) to one partition; fine
    # at this size and deterministic, unlike indices that depend on partitioning.
    index = (F.row_number().over(Window.orderBy("user_id")) - 1 + offset).cast("int")
    others = others.withColumn("user_index", index).cache()  # used twice below: number them once
    trained = pre_cutoff.select("user_id").distinct().withColumn("_trained", F.lit(True))
    users = (
        known.select("user_id", "user_index", F.lit("known").alias("user_type"))
        .unionByName(others.join(trained, "user_id", "left").select(
            "user_id", "user_index",
            F.when(F.col("_trained"), "removed_by_k_core").otherwise("new").alias("user_type")))
    )
    catalog = items.count()
    seen_counts = seen_items(raw, items, users, train_cutoff).groupBy("user_index").agg(F.count("*").alias("_seen"))
    users = users.join(seen_counts, "user_index", "left").select(
        "user_id", "user_index", "user_type",
        (F.lit(catalog) - F.coalesce("_seen", F.lit(0))).alias("unseen_items"),
        (F.col("user_index") / batch_users).cast("int").alias("batch"),
    )
    return users


def recommend_batch(model, fallback, users, seen, k: int):
    """(per-user lists, index rows) of one batch; model for known users, fallback for the rest."""
    from pyspark.sql import functions as F

    known = users.filter(F.col("user_type") == "known").select("user_index")
    model_rows = model.recommend(known, seen, k).persist()
    covered = model_rows.select("user_index").distinct().withColumn("rec_type", F.lit("model"))
    rest = users.select("user_index").join(covered, "user_index", "left_anti")
    fallback_rows = fallback.recommend(rest, seen, k)
    rows = model_rows.unionByName(fallback_rows)
    lengths = rows.groupBy("user_index").agg(F.count("*").alias("length"))
    approximate = F.when(F.col("rec_type") == "model", F.lit(bool(model.contract["approximate"]))) \
        .otherwise(F.lit(bool(fallback.contract["approximate"])))
    lists = (
        users.join(covered, "user_index", "left").join(lengths, "user_index", "left")
        .select("user_id", "user_index", "user_type", F.coalesce("rec_type", F.lit("fallback")).alias("rec_type"),
                "unseen_items", F.coalesce("length", F.lit(0)).alias("length"))
        .withColumn("approximate", approximate)
    )
    return lists, rows, model_rows


def audit(spark, users, rows, seen_pairs, k: int) -> dict:
    """Coverage and list checks on the written outputs; raises if a rule is broken."""
    from pyspark.sql import Window, functions as F

    by_type = {(r.user_type, r.rec_type): r["count"] for r in users.groupBy("user_type", "rec_type").count().collect()}
    expected_length = F.least(F.lit(k), F.col("unseen_items"))
    totals = users.agg(
        F.count("*").alias("users"), F.countDistinct("user_id").alias("distinct_users"),
        F.sum("length").alias("length_sum"),
        F.sum((F.col("length") != expected_length).cast("long")).alias("wrong_length"),
        F.sum((F.col("length") < k).cast("long")).alias("short_lists"),
        F.sum((F.col("length") == 0).cast("long")).alias("empty_lists"),
    ).first()
    per_user = rows.groupBy("user_id").agg(
        F.count("*").alias("n"), F.countDistinct("parent_asin").alias("items"),
        F.countDistinct("rank").alias("ranks"), F.min("rank").alias("first"), F.max("rank").alias("last"),
        F.sum((F.col("score").isNull() | F.isnan("score")).cast("long")).alias("bad_scores"),
    )
    lists = per_user.agg(
        F.count("*").alias("users"), F.sum("n").alias("rows"),
        F.sum(((F.col("items") != F.col("n")) | (F.col("ranks") != F.col("n")) | (F.col("first") != 1)
               | (F.col("last") != F.col("n"))).cast("long")).alias("malformed_lists"),
        F.sum("bad_scores").alias("bad_scores"),
    ).first()
    # Scores must not increase with rank (the full-model score decides the order).
    previous = F.lag("score").over(Window.partitionBy("user_id").orderBy("rank"))
    unordered = rows.withColumn("_previous", previous).filter(F.col("score") > F.col("_previous") + 1e-9).count()
    seen_recommended = rows.join(seen_pairs, ["user_id", "parent_asin"]).count()
    orphan_rows = rows.select("user_id").distinct().join(users, "user_id", "left_anti").count()
    problems = {
        "duplicate_users": totals["users"] - totals["distinct_users"],
        "lengths_not_min_k_unseen": totals["wrong_length"] or 0,
        "rows_not_matching_lengths": (totals["length_sum"] or 0) - (lists["rows"] or 0),
        "malformed_lists": lists["malformed_lists"] or 0,
        "missing_or_nan_scores": lists["bad_scores"] or 0,
        "scores_increasing_with_rank": unordered,
        "seen_items_recommended": seen_recommended,
        "rows_of_unlisted_users": orphan_rows,
    }
    broken = {name: value for name, value in problems.items() if value}
    if broken:
        raise RuntimeError(f"Materialization audit failed: {broken}")
    return {
        "users": totals["users"],
        "rows": lists["rows"] or 0,
        "users_by_type": {t: sum(v for (u, _), v in by_type.items() if u == t) for t in USER_TYPES},
        "users_by_rec_type": {r: sum(v for (_, s), v in by_type.items() if s == r) for r in REC_TYPES},
        "users_by_type_and_rec_type": {f"{u}/{r}": v for (u, r), v in sorted(by_type.items())},
        "short_lists": {"count": totals["short_lists"] or 0, "empty": totals["empty_lists"] or 0,
                        "reason": "catalog_exhausted: every list has min(k, unseen catalog items) entries"},
        "checks": problems,
    }


def _move(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def run_materialize(config: ProjectConfig, run_directory: Path, sample: str, split: str, model_label: str,
                    fallback_label: str, *, batch_users: int = DEFAULT_BATCH_USERS) -> Path:
    """Write serving lists for every sample user from one trained model plus the popularity fallback. Starts Spark."""
    if batch_users < 1:
        raise ValueError("batch_users must be positive")
    run_directory = Path(run_directory).resolve(strict=True)
    sample_directory = run_directory / "samples" / sample
    split_directory = sample_directory / "splits" / split
    split_stage_path = split_directory / "stage.json"
    if not split_stage_path.is_file():
        raise FileNotFoundError(f"Split is not complete: {split_directory}")
    split_stage = json.loads(split_stage_path.read_text())
    sample_stage = json.loads((sample_directory / "stage.json").read_text())
    stages = {}
    for role, label in (("model", model_label), ("fallback", fallback_label)):
        path = split_directory / "models" / label / "stage.json"
        if not path.is_file():
            raise FileNotFoundError(f"Model is not trained: {path.parent}")
        stages[role] = json.loads(path.read_text())
    if stages["fallback"]["model"]["model"] != "popularity":
        raise ValueError(f"The fallback must be a popularity model (P0.5), got {stages['fallback']['model']['model']}")
    k = config.model.recommendation_count
    final = split_directory / "serving" / model_label
    partial = final.with_name(f"{model_label}.partial")
    if final.exists():
        raise FileExistsError(f"Serving lists already written for this model: {final}")
    plan = {
        "model_label": model_label,
        "fallback_label": fallback_label,
        "k": k,
        "batch_users": batch_users,
        "split_stage_sha256": hashlib.sha256(split_stage_path.read_bytes()).hexdigest(),
        "sample_stage_sha256": hashlib.sha256((sample_directory / "stage.json").read_bytes()).hexdigest(),
        "model_stage_sha256": {role: hashlib.sha256(
            (split_directory / "models" / label / "stage.json").read_bytes()).hexdigest()
            for role, label in (("model", model_label), ("fallback", fallback_label))},
    }
    resumed = (partial / "plan.json").is_file() and json.loads((partial / "plan.json").read_text()) == plan
    if partial.exists() and not resumed:
        if (partial / "plan.json").is_file():
            raise FileExistsError(f"{partial} was started with other arguments ({partial / 'plan.json'}); "
                                  "rerun with them or delete it")
        shutil.rmtree(partial)  # no plan yet: nothing reusable
    partial.mkdir(parents=True, exist_ok=True)

    from pyspark.sql import functions as F

    from amazon_recommender.models import load_model
    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    with timed(timings, "spark_start"):  # JVM launch and session; not data work
        spark = create_spark_session(config, app_name="amazon-recommender-materialize")
    try:
        raw = spark.read.parquet(str(run_directory / sample_stage["outputs"]["interactions"])).select(
            "user_id", "parent_asin", "timestamp")
        items = spark.read.parquet(str(split_directory / split_stage["outputs"]["items"]))
        known = spark.read.parquet(str(split_directory / split_stage["outputs"]["users"]))
        train_cutoff = split_stage["cutoffs_ms"]["train"]
        if not resumed:
            with timed(timings, "serving_users"):
                serving_users(raw, known, items, train_cutoff, batch_users) \
                    .write.mode("error").parquet(str(partial / "serving_users"))
                users = spark.read.parquet(str(partial / "serving_users"))
                # Written once with the batch, so each batch reads only its own seen pairs.
                seen_items(raw, items, users, train_cutoff).join(users.select("user_index", "batch"), "user_index") \
                    .write.mode("error").partitionBy("batch").parquet(str(partial / "seen"))
            (partial / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        users = spark.read.parquet(str(partial / "serving_users"))
        seen = spark.read.parquet(str(partial / "seen"))
        batches = users.agg(F.max("batch")).first()[0] + 1
        model = load_model(spark, split_directory / "models" / model_label / stages["model"]["outputs"]["model"],
                           mappings=split_stage["mappings"])
        fallback = load_model(spark, split_directory / "models" / fallback_label
                              / stages["fallback"]["outputs"]["model"], mappings=split_stage["mappings"])

        reused = 0
        for batch in range(batches):
            marker = partial / "batches" / f"{batch:05d}.json"
            if marker.is_file():
                reused += 1
                continue
            for leftover in (partial / "pending" / f"{batch:05d}", partial / "users" / f"batch={batch}",
                             partial / "recommendations" / f"batch={batch}"):
                shutil.rmtree(leftover, ignore_errors=True)  # an interrupted, unmarked batch
            batch_timings = {}
            batch_started = time.monotonic()
            batch_users_frame = users.filter(F.col("batch") == batch).drop("batch")
            batch_seen = seen.filter(F.col("batch") == batch).drop("batch")
            lists, rows, model_rows = recommend_batch(model, fallback, batch_users_frame, batch_seen, k)
            pending = partial / "pending" / f"{batch:05d}"
            with timed(batch_timings, "recommend_write"):
                rows.join(items, "item_index").join(batch_users_frame.select("user_index", "user_id"), "user_index") \
                    .select("user_id", "rank", "parent_asin", "score") \
                    .write.mode("error").parquet(str(pending / "recommendations"))
                lists.drop("user_index").write.mode("error").parquet(str(pending / "users"))
            model_rows.unpersist()
            counts = spark.read.parquet(str(pending / "users")).agg(
                F.count("*").alias("users"), F.sum("length").alias("rows"),
                F.sum((F.col("rec_type") == "fallback").cast("long")).alias("fallback_users")).first()
            _move(pending / "recommendations", partial / "recommendations" / f"batch={batch}")
            _move(pending / "users", partial / "users" / f"batch={batch}")
            batch_timings["total"] = round(time.monotonic() - batch_started, 3)
            marker.parent.mkdir(exist_ok=True)
            marker.write_text(json.dumps({"batch": batch, "users": counts["users"], "rows": counts["rows"] or 0,
                                          "fallback_users": counts["fallback_users"] or 0,
                                          "completed_at": datetime.now(timezone.utc).isoformat(),
                                          "timings_seconds": batch_timings}, indent=2) + "\n")
        shutil.rmtree(partial / "pending", ignore_errors=True)

        with timed(timings, "audit"):
            written_users = spark.read.parquet(str(partial / "users"))
            written_rows = spark.read.parquet(str(partial / "recommendations"))
            seen_pairs = raw.filter(F.col("timestamp") <= train_cutoff).select("user_id", "parent_asin")
            coverage = audit(spark, written_users, written_rows, seen_pairs, k)
            sample_users = raw.select("user_id").distinct().count()
        if coverage["users"] != sample_users or coverage["users_by_type"]["known"] != \
                split_stage["mappings"]["users"]["count"]:
            raise RuntimeError(f"Coverage does not reconcile: {coverage['users']} listed users for {sample_users} "
                               f"sample users, {coverage['users_by_type']['known']} known for "
                               f"{split_stage['mappings']['users']['count']} mapped")
        batch_records = [json.loads((partial / "batches" / f"{b:05d}.json").read_text()) for b in range(batches)]
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "materialize",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": split_stage["run_id"],
            "sample": sample,
            "split": split,
            "split_stage_sha256": plan["split_stage_sha256"],
            "model_label": model_label,
            "model": stages["model"]["model"]["model"],
            "fallback_label": fallback_label,
            "contracts": {"model": model.contract, "fallback": fallback.contract},
            "k": k,
            "train_cutoff_utc": split_stage["cutoffs_utc"]["train"],
            "policy": {
                "population": "every user of the sample (all windows)",
                "seen": "interactions up to the train cutoff, before the k-core",
                "model_users": "known users (split mapping) with a model list",
                "fallback_users": "removed_by_k_core and new users, and known users without a model list",
            },
            "counts": {"users": coverage["users"], "sample_users": sample_users},
            "coverage": coverage,
            "batches": {"batch_users": batch_users, "count": batches, "reused_on_resume": reused,
                        "records": batch_records},
            "config": config.as_dict(),
            "config_digest": config.digest,
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            # Relative to the serving directory.
            "outputs": {"users": "users", "recommendations": "recommendations"},
        }
    finally:
        spark.stop()
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final
