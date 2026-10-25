"""Recommend stage: top-k lists for every training user of a split, from one trained model.

Input is a trained model (`<split>/models/<label>/`). Every user of the split's mapping gets
`model.recommendation_count` items, ranked by the model's own `recommend` (the same code
evaluation uses), with the user's training items excluded. Output:
`<split>/recommendations/<label>/` with `recommendations/` (Parquet, index-based:
user_index, item_index, rank, score) and `stage.json`, completed by rename.

This is the computation of P7.1 for the users the model knows. Mapping back to original IDs,
fallbacks for users without factors and the MongoDB publication belong to P7; the benchmark
(P6) times this stage as "recommendation generation".
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import timed


def run_recommend(config: ProjectConfig, run_directory: Path, sample: str, split: str, model_label: str) -> Path:
    """Write top-k recommendations of one trained model for all its split's training users. Starts Spark."""
    run_directory = Path(run_directory).resolve(strict=True)
    split_directory = run_directory / "samples" / sample / "splits" / split
    split_stage_path = split_directory / "stage.json"
    if not split_stage_path.is_file():
        raise FileNotFoundError(f"Split is not complete: {split_directory}")
    model_directory = split_directory / "models" / model_label
    if not (model_directory / "stage.json").is_file():
        raise FileNotFoundError(f"Model is not trained: {model_directory}")
    split_stage = json.loads(split_stage_path.read_text())
    model_stage = json.loads((model_directory / "stage.json").read_text())
    k = config.model.recommendation_count
    final = split_directory / "recommendations" / model_label
    partial = final.with_name(f"{model_label}.partial")
    if final.exists():
        raise FileExistsError(f"Recommendations already written for this model: {final}")
    if partial.exists():
        shutil.rmtree(partial)  # scratch from an interrupted attempt; never marked complete
    partial.mkdir(parents=True)

    from pyspark.sql import functions as F

    from amazon_recommender.models import load_model
    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    with timed(timings, "spark_start"):  # JVM launch and session; not data work
        spark = create_spark_session(config, app_name="amazon-recommender-recommend")
    try:
        users = spark.read.parquet(str(split_directory / split_stage["outputs"]["users"])).select("user_index")
        seen = spark.read.parquet(str(split_directory / split_stage["outputs"]["train"]))
        model = load_model(spark, model_directory / model_stage["outputs"]["model"], mappings=split_stage["mappings"])
        with timed(timings, "recommend_write"):
            model.recommend(users, seen, k).write.mode("error").parquet(str(partial / "recommendations"))
        with timed(timings, "verification_counts"):
            per_user = spark.read.parquet(str(partial / "recommendations")).groupBy("user_index").count()
            row = per_user.agg(F.sum("count").alias("rows"), F.count("*").alias("users"),
                               F.sum((F.col("count") < k).cast("long")).alias("short_lists")).first()
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "recommend",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": split_stage["run_id"],
            "sample": sample,
            "split": split,
            "model_label": model_label,
            "model": model_stage["model"]["model"],
            "k": k,
            "users": split_stage["mappings"]["users"]["count"],
            "users_with_recommendations": row["users"],
            "rows": row["rows"] or 0,
            "short_lists": row["short_lists"] or 0,
            "seen_policy": "training items of the user are excluded",
            "config": config.as_dict(),
            "config_digest": config.digest,
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            # Relative to the recommendations directory.
            "outputs": {"recommendations": "recommendations"},
        }
    finally:
        spark.stop()
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final
