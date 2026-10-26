"""Publication stage (P7.2): materialized serving lists and item titles to MongoDB.

Each publication writes into collections named after its source (run, sample, split and model
label, hashed into `version`), validates them against the source Parquet, and only then
records the publication and switches the `active_run` pointer document, so readers never see a
partially written or mixed run. Documents are keyed on the original user id and written with
idempotent upserts, so a rerun after an interruption completes the same collections, and a
repeated publication rewrites identical documents. The publication time and the model
description live in the pointer and in `publications`, not in the user documents.

Input contract (from the materialize stage): a serving directory with `stage.json`
(`stage` "materialize", `run_id`, `sample`, `split`, `model_label`, `model`, `k`, `counts.users`),
`users/` (user_id, user_type, rec_type, approximate, unseen_items, length) and
`recommendations/` (user_id, rank, parent_asin, score). Every `users/` row becomes one document,
including users with an empty list.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from amazon_recommender.config import ProjectConfig

ACTIVE_COLLECTION = "active_run"
PUBLICATIONS_COLLECTION = "publications"
ACTIVE_ID = "current"
BATCH_SIZE = 1000
RETRIES = 3
VALIDATION_SAMPLE = 20  # documents per user_type


def mongo_uri(config: ProjectConfig) -> str:
    """The connection string comes from the environment, never from the config file."""
    uri = os.environ.get(config.mongo.uri_env)
    if not uri:
        raise ValueError(f"Set {config.mongo.uri_env} to the MongoDB connection string (see .env.example)")
    return uri


def publication_version(stage: dict) -> str:
    """Short key of one materialization: republishing it reuses its collections, any other source
    (including a rematerialization with the same labels) gets new ones."""
    source = "/".join(str(stage[key]) for key in ("run_id", "sample", "split", "model_label", "fallback_label", "k",
                                                   "completed_at"))
    return f"{stage['model'].replace('-', '_')}_{hashlib.sha256(source.encode()).hexdigest()[:12]}"


def collection_names(version: str) -> dict:
    return {"recommendations": f"recommendations_{version}", "items": f"items_{version}"}


def user_document(row, version: str, k: int) -> dict:
    """One document per user; `items` is ordered by rank, and a short list states why."""
    items = sorted(row["items"] or [], key=lambda item: item["rank"])
    return {
        "_id": row["user_id"],
        "version": version,
        "user_type": row["user_type"],
        "rec_type": row["rec_type"],
        "approximate": bool(row["approximate"]),
        "short_list_reason": "catalog_exhausted" if len(items) < k else None,
        "items": [
            {"rank": int(item["rank"]), "parent_asin": item["parent_asin"], "score": float(item["score"])}
            for item in items
        ],
    }


def write_batches(collection, documents, *, batch_size: int = BATCH_SIZE, retries: int = RETRIES) -> int:
    """Upsert documents by `_id` in bounded batches; retry a failed batch, then raise."""
    from pymongo import ReplaceOne
    from pymongo.errors import PyMongoError

    written, batch = 0, []

    def flush():
        for attempt in range(retries + 1):
            try:
                collection.bulk_write([ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in batch], ordered=False)
                return
            except PyMongoError:
                if attempt == retries:
                    raise
                time.sleep(2**attempt)  # replacing the same _ids again is harmless

    for document in documents:
        batch.append(document)
        if len(batch) == batch_size:
            flush()
            written, batch = written + len(batch), []
    if batch:
        flush()
        written += len(batch)
    return written


def _partition_writer(uri: str, database: str, collection: str, to_document):
    """Build a mapPartitions function: one client per partition, yields the written count."""

    def write(rows):
        from pymongo import MongoClient

        with MongoClient(uri) as client:
            yield write_batches(client[database][collection], (to_document(row) for row in rows))

    return write


def user_lists(users, recommendations):
    """One row per user (empty lists included) with the ranked items collected into a list."""
    from pyspark.sql import functions as F

    items = recommendations.groupBy("user_id").agg(
        F.collect_list(F.struct("rank", "parent_asin", "score")).alias("items"))
    return users.select("user_id", "user_type", "rec_type", "approximate", "length").join(items, "user_id", "left")


def validate(collection, lists, expected_users: int, version: str, k: int, seed: int) -> dict:
    """Compare the document count, and a seeded sample of documents of every user_type, with the source rows."""
    from pyspark.sql import Window, functions as F

    published = collection.count_documents({})
    if published != expected_users:
        raise RuntimeError(f"Published {published} user documents, expected {expected_users}")
    order = Window.partitionBy("user_type").orderBy(F.xxhash64(F.lit(seed), "user_id"), "user_id")
    sample = lists.withColumn("_n", F.row_number().over(order)).filter(F.col("_n") <= VALIDATION_SAMPLE).collect()
    mismatches = sum(collection.find_one({"_id": row["user_id"]}) != user_document(row, version, k) for row in sample)
    if mismatches:
        raise RuntimeError(f"{mismatches} of {len(sample)} sampled documents differ from the source rows")
    return {"user_documents": published, "sampled_documents_checked": len(sample),
            "sampled_by_user_type": {t: sum(row["user_type"] == t for row in sample)
                                     for t in sorted({row["user_type"] for row in sample})}}


def publish(config: ProjectConfig, serving_directory: Path, metadata_directory: Path) -> Path:
    """Publish one materialization and switch the active-run pointer. Starts Spark."""
    serving_directory = Path(serving_directory).resolve(strict=True)
    stage = json.loads((serving_directory / "stage.json").read_text())
    if stage.get("stage") != "materialize" or stage.get("status") != "completed":
        raise ValueError(f"{serving_directory} is not a completed materialize stage")
    uri = mongo_uri(config)
    version = publication_version(stage)
    names = collection_names(version)
    k, users_expected = stage["k"], stage["counts"]["users"]

    from pymongo import MongoClient

    from amazon_recommender.etl import timed
    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    with MongoClient(uri, serverSelectionTimeoutMS=5000) as client:
        database = client[config.mongo.database]
        database.command("ping")  # fail before starting Spark if MongoDB is unreachable
        spark = create_spark_session(config, app_name="amazon-recommender-publish")
        try:
            users = spark.read.parquet(str(serving_directory / stage["outputs"]["users"]))
            recommendations = spark.read.parquet(str(serving_directory / stage["outputs"]["recommendations"]))
            lists = user_lists(users, recommendations)
            with timed(timings, "write_recommendations"):
                written = lists.rdd.mapPartitions(_partition_writer(
                    uri, config.mongo.database, names["recommendations"],
                    lambda row: user_document(row, version, k))).sum()
            items = recommendations.select("parent_asin").distinct().join(
                spark.read.parquet(str(Path(metadata_directory))).select("parent_asin", "title"), "parent_asin", "left")
            with timed(timings, "write_items"):
                items_written = items.rdd.mapPartitions(_partition_writer(
                    uri, config.mongo.database, names["items"],
                    lambda row: {"_id": row["parent_asin"], "title": row["title"]})).sum()
            with timed(timings, "validate"):
                checks = validate(database[names["recommendations"]], lists, users_expected, version, k, config.seed)
                item_documents = database[names["items"]].count_documents({})
                if item_documents != items_written:
                    raise RuntimeError(f"{item_documents} item documents, {items_written} items written")
            resources = describe_resources(spark, config)
        finally:
            spark.stop()

        published_at = datetime.now(timezone.utc).isoformat()
        publication = {
            "_id": version, **names, "published_at": published_at, "users": users_expected,
            "item_documents": item_documents, "k": k,
            **{key: stage[key] for key in ("run_id", "sample", "split", "model", "model_label", "fallback_label")},
            "coverage": stage["coverage"],
            "source": str(serving_directory),
        }
        database[PUBLICATIONS_COLLECTION].replace_one({"_id": version}, publication, upsert=True)
        # The pointer is switched last, after validation: readers see the old run until here.
        database[ACTIVE_COLLECTION].replace_one({"_id": ACTIVE_ID}, {**publication, "_id": ACTIVE_ID,
                                                                     "version": version}, upsert=True)

    record = {
        "stage": "publish",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "database": config.mongo.database,
        "version": version,
        "collections": names,
        "source": str(serving_directory),
        "source_stage_sha256": hashlib.sha256((serving_directory / "stage.json").read_bytes()).hexdigest(),
        "documents_written": written,
        "item_documents": items_written,
        "validation": checks,
        "resources": resources,
        "timings_seconds": timings,
    }
    output = serving_directory / "publications" / f"{version}-{published_at.replace(':', '')}.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return output
