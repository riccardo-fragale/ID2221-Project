"""Session factory for the pinned labs-compatible Spark environment."""

import sys

from amazon_recommender.config import ProjectConfig
from amazon_recommender.environment import inspect_environment


def create_spark_session(config: ProjectConfig, *, app_name: str = "amazon-recommender"):
    """Start Spark only when explicitly called by a pipeline stage."""
    report = inspect_environment()
    if report["errors"]:
        raise RuntimeError("; ".join(report["errors"]))
    from pyspark import SparkContext
    from pyspark.sql import SparkSession

    if SparkContext._active_spark_context is not None:
        raise RuntimeError("An existing Spark context could override the configured resources")
    root = config.storage_root
    event_logs = root / "spark-events"
    checkpoints = root / "checkpoints"
    for directory in (event_logs, checkpoints, root / "spark-local"):
        directory.mkdir(parents=True, exist_ok=True)
    builder = (
        SparkSession.builder.appName(app_name)
        .master(config.spark.master)
        .config("spark.driver.memory", config.spark.driver_memory)
        .config("spark.sql.shuffle.partitions", config.spark.shuffle_partitions)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.local.dir", str(root / "spark-local"))
        .config("spark.sql.warehouse.dir", (root / "warehouse").as_uri())
        .config("spark.eventLog.enabled", str(config.spark.event_log_enabled).lower())
        .config("spark.eventLog.dir", event_logs.as_uri())
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setCheckpointDir(str(checkpoints))
    return spark
