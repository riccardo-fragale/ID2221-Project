"""Publisher logic runs everywhere; the MongoDB round trip runs only with RUN_MONGO_TESTS=1 and MONGODB_URI set."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pymongo.errors import AutoReconnect

from amazon_recommender.config import load_config
from amazon_recommender.publish import (
    ACTIVE_COLLECTION, collection_names, item_details_stage, item_document, mongo_uri, publication_version, publish,
    user_document, write_batches,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeCollection:
    """Records upserts by _id; optionally fails the first `failures` bulk writes."""

    def __init__(self, failures=0):
        self.documents, self.calls, self.failures = {}, [], failures

    def bulk_write(self, operations, ordered):
        self.calls.append(len(operations))
        if self.failures:
            self.failures -= 1
            raise AutoReconnect("simulated outage")
        for operation in operations:
            self.documents[operation._filter["_id"]] = operation._doc


class PublisherLogicTests(unittest.TestCase):
    def test_document_orders_items_by_rank_and_keeps_scores(self):
        row = {
            "user_id": "u1", "user_type": "known", "rec_type": "model", "approximate": False,
            "items": [{"rank": 2, "parent_asin": "b", "score": 3.5}, {"rank": 1, "parent_asin": "a", "score": 4.0}],
        }
        document = user_document(row, "v1", 2)
        self.assertEqual(document["_id"], "u1")
        self.assertEqual([item["parent_asin"] for item in document["items"]], ["a", "b"])
        self.assertEqual(document["items"][0], {"rank": 1, "parent_asin": "a", "score": 4.0})
        self.assertEqual((document["version"], document["rec_type"], document["user_type"]), ("v1", "model", "known"))
        self.assertIsNone(document["short_list_reason"])
        self.assertNotIn("published_at", document)  # repeat publications rewrite identical documents

    def test_empty_and_short_lists_state_the_reason(self):
        row = {"user_id": "u2", "user_type": "new", "rec_type": "fallback", "approximate": False, "items": None}
        document = user_document(row, "v1", 20)
        self.assertEqual((document["items"], document["short_list_reason"]), ([], "catalog_exhausted"))

    def test_batches_are_bounded_and_rerun_is_idempotent(self):
        collection = FakeCollection()
        documents = [{"_id": f"u{n}", "items": [n]} for n in range(2500)]
        self.assertEqual(write_batches(collection, documents, batch_size=1000), 2500)
        self.assertEqual(collection.calls, [1000, 1000, 500])
        self.assertEqual(write_batches(collection, documents, batch_size=1000), 2500)
        self.assertEqual(len(collection.documents), 2500)

    def test_failed_batch_is_retried_then_raises(self):
        with mock.patch("amazon_recommender.publish.time.sleep"):
            collection = FakeCollection(failures=2)
            self.assertEqual(write_batches(collection, [{"_id": "u"}], retries=2), 1)
            self.assertEqual(len(collection.documents), 1)
            with self.assertRaises(AutoReconnect):
                write_batches(FakeCollection(failures=3), [{"_id": "u"}], retries=2)

    def test_collection_names_depend_on_the_whole_source(self):
        stage = {"run_id": "20261025T093611Z-2b4b605ead0f", "sample": "fraction-1-seed-2221",
                 "split": "refit-c0cb58f463fb-k5-5", "model": "bias-als", "model_label": "bias-als-rank50",
                 "fallback_label": "popularity-smoothing10-window_days180", "k": 20,
                 "completed_at": "2026-10-26T10:00:00+00:00"}
        version = publication_version(stage)
        self.assertRegex(version, r"^bias_als_[0-9a-f]{12}$")
        self.assertEqual(version, publication_version(dict(stage)))
        for key, value in (("model_label", "bias-als-rank25"), ("split", "p0.8-0.1-0.1-k5-5"),
                           ("sample", "fraction-0.2-seed-2221"), ("fallback_label", "popularity-x"), ("k", 10),
                           ("completed_at", "2026-10-26T11:00:00+00:00")):
            self.assertNotEqual(version, publication_version({**stage, key: value}), key)
        self.assertEqual(collection_names(version)["recommendations"], f"recommendations_{version}")
        with_details = publication_version(stage, "a" * 64)
        self.assertNotIn(with_details, (version, publication_version(stage, "b" * 64)))
        self.assertEqual(with_details, publication_version(dict(stage), "a" * 64))

    def test_item_documents_keep_the_old_shape_without_details(self):
        row = {"parent_asin": "a", "title": None, "main_category": "Prime Video", "directors": ("D",),
               "starring": [], "studio": None, "image": "u", "rating_number": 5}
        self.assertEqual(item_document(row, False), {"_id": "a", "title": None})
        self.assertEqual(item_document(row, True), {"_id": "a", "title": None, "main_category": "Prime Video",
                                                    "directors": ["D"], "starring": [], "studio": None,
                                                    "image": "u", "rating_number": 5})

    def test_item_details_must_be_a_completed_stage_of_the_same_run(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        (path / "stage.json").write_text(json.dumps({"stage": "item-details", "status": "completed", "run_id": "r1"}))
        self.assertEqual(item_details_stage(path, "r1")["run_id"], "r1")
        with self.assertRaisesRegex(ValueError, "cannot be published with run r2"):
            item_details_stage(path, "r2")
        (path / "stage.json").write_text(json.dumps({"stage": "etl", "status": "completed", "run_id": "r1"}))
        with self.assertRaisesRegex(ValueError, "not a completed item-details stage"):
            item_details_stage(path, "r1")

    def test_missing_uri_fails_before_connecting(self):
        config = load_config(ROOT / "configs/local.toml")
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "MONGODB_URI"):
            mongo_uri(config)


@unittest.skipUnless(os.environ.get("RUN_MONGO_TESTS") == "1", "set RUN_MONGO_TESTS=1 and MONGODB_URI (make check-mongo)")
class MongoRoundTripTests(unittest.TestCase):
    def test_publish_validates_and_switches_pointer(self):
        from pymongo import MongoClient
        from pyspark.sql import SparkSession

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        database_name = f"test_publish_{os.getpid()}"
        config = replace(base, storage_root=root / "storage", mongo=replace(base.mongo, database=database_name))
        client = MongoClient(mongo_uri(config))
        self.addCleanup(client.close)
        self.addCleanup(client.drop_database, database_name)

        serving = root / "serving"
        spark = SparkSession.builder.master("local[1]").getOrCreate()
        try:
            spark.createDataFrame(
                [("u1", 1, "a", 4.5), ("u1", 2, "b", 4.0), ("u2", 1, "c", 3.0)],
                "user_id string, rank int, parent_asin string, score double",
            ).write.parquet(str(serving / "recommendations"))
            spark.createDataFrame(
                [("u1", "known", "model", False, 2, 2), ("u2", "new", "fallback", False, 1, 1),
                 ("u3", "removed_by_k_core", "fallback", False, 0, 0)],
                "user_id string, user_type string, rec_type string, approximate boolean, unseen_items long, length long",
            ).write.parquet(str(serving / "users"))
            spark.createDataFrame([("a", "Title A"), ("b", None)], "parent_asin string, title string").write.parquet(
                str(root / "metadata"))
        finally:
            spark.stop()
        stage = {"stage": "materialize", "status": "completed", "run_id": "run1", "sample": "s", "split": "p",
                 "model": "bias-als", "model_label": "bias-als-x", "fallback_label": "popularity-x", "k": 2,
                 "completed_at": "2026-10-26T10:00:00+00:00",
                 "counts": {"users": 3}, "coverage": {}, "outputs": {"users": "users", "recommendations": "recommendations"}}
        (serving / "stage.json").write_text(json.dumps(stage))

        for _ in range(2):  # the second publication rewrites the same documents
            record = json.loads(publish(config, serving, root / "metadata").read_text())
        database = client[database_name]
        names = collection_names(publication_version(stage))
        self.assertEqual(record["documents_written"], 3)
        self.assertEqual(record["validation"]["sampled_by_user_type"], {"known": 1, "new": 1, "removed_by_k_core": 1})
        self.assertEqual(database[names["recommendations"]].count_documents({}), 3)
        self.assertEqual([i["parent_asin"] for i in database[names["recommendations"]].find_one("u1")["items"]], ["a", "b"])
        self.assertEqual(database[names["recommendations"]].find_one("u3")["items"], [])
        self.assertEqual(database[names["recommendations"]].find_one("u2")["short_list_reason"], "catalog_exhausted")
        self.assertEqual(database[names["items"]].find_one("c"), {"_id": "c", "title": None})
        self.assertEqual(database[ACTIVE_COLLECTION].find_one("current")["recommendations"], names["recommendations"])


if __name__ == "__main__":
    unittest.main()
