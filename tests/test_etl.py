"""ETL preconditions run everywhere; Spark tests run only with RUN_SPARK_TESTS=1 (make check-spark)."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.cli import main
from amazon_recommender.config import load_config
from amazon_recommender.etl import run_etl


ROOT = Path(__file__).resolve().parents[1]
DAY = 86_400_000
T0 = 1_577_836_800_000  # 2020-01-01 UTC

REVIEWS = [
    # Valid and distinct.
    {"user_id": "u1", "parent_asin": "i1", "rating": 5.0, "timestamp": T0},
    {"user_id": "u1", "parent_asin": "i2", "rating": 3.0, "timestamp": T0 + DAY},
    # A later repeat replaces an earlier review, even with a lower rating.
    {"user_id": "u2", "parent_asin": "i1", "rating": 5.0, "timestamp": T0},
    {"user_id": "u2", "parent_asin": "i1", "rating": 2.0, "timestamp": T0 + 400 * DAY},
    # Same timestamp: the highest rating wins.
    {"user_id": "u3", "parent_asin": "i1", "rating": 2.0, "timestamp": T0},
    {"user_id": "u3", "parent_asin": "i1", "rating": 4.0, "timestamp": T0},
    # Exact duplicate collapses.
    {"user_id": "u4", "parent_asin": "i3", "rating": 1.0, "timestamp": T0},
    {"user_id": "u4", "parent_asin": "i3", "rating": 1.0, "timestamp": T0},
    # Rejected rows, one per reason.
    {"user_id": "", "parent_asin": "i1", "rating": 4.0, "timestamp": T0},
    {"user_id": "u5", "parent_asin": "i1", "rating": 6.0, "timestamp": T0},
    {"user_id": "u5", "parent_asin": "i2", "rating": 0.0, "timestamp": T0},
    {"user_id": "u5", "parent_asin": "i3", "rating": 4.0, "timestamp": T0 // 1000},  # seconds, not ms
]
MALFORMED = "{not json"
METADATA = [
    {"parent_asin": "i1", "title": "Movie One"},
    {"parent_asin": "i2", "title": "Movie Two"},
    {"parent_asin": "i2", "title": "Movie Two (duplicate)"},
    # Blank titles never win over a real one; an item with only a blank title has no title.
    {"parent_asin": "i1", "title": ""},
    {"parent_asin": "i1", "title": "   "},
    {"parent_asin": "i3", "title": " "},
    {"parent_asin": None, "title": "No identifier"},
]


def write_lines(path: Path, rows, extra=()):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "".join(line + "\n" for line in extra))


class EtlTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = replace(load_config(ROOT / "configs/local.toml"), storage_root=self.root / "storage")
        self.reviews, self.metadata = self.root / "reviews.ndjson", self.root / "metadata.ndjson"
        write_lines(self.reviews, REVIEWS, extra=[MALFORMED])
        write_lines(self.metadata, METADATA)
        self.run_directory = create_run(self.config, reviews=self.reviews, metadata=self.metadata)


class PreconditionTests(EtlTests):
    def test_changed_configuration_is_rejected_before_spark_starts(self):
        other = replace(self.config, seed=self.config.seed + 1)
        with self.assertRaisesRegex(ValueError, "Configuration differs"):
            run_etl(other, self.run_directory)

    def test_changed_input_is_rejected(self):
        self.reviews.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "reviews input changed"):
            run_etl(self.config, self.run_directory)

    def test_completed_stage_is_not_overwritten(self):
        (self.run_directory / "etl").mkdir()
        with self.assertRaises(FileExistsError):
            run_etl(self.config, self.run_directory)

    def test_cli_reports_missing_run_cleanly(self):
        self.assertEqual(main(["etl", "--config", str(ROOT / "configs/local.toml"), "--run", str(self.root / "nope")]), 2)


@unittest.skipUnless(os.environ.get("RUN_SPARK_TESTS") == "1", "set RUN_SPARK_TESTS=1 to start Spark")
class SparkEtlTests(EtlTests):
    def test_etl_cleans_deduplicates_and_records_counts(self):
        output = run_etl(self.config, self.run_directory)
        self.assertEqual(output, self.run_directory / "etl")
        self.assertFalse((self.run_directory / "etl.partial").exists())
        stage = json.loads((output / "stage.json").read_text())
        reviews = stage["reviews"]
        self.assertEqual(reviews["rows_read"], len(REVIEWS) + 1)
        self.assertEqual(reviews["rejected"], {
            "malformed_record": 1, "missing_field": 1, "invalid_rating": 2, "invalid_timestamp": 1,
        })
        self.assertEqual(reviews["valid"], 8)
        self.assertEqual(reviews["interactions"], 5)
        self.assertEqual(reviews["duplicates_removed"], 3)
        self.assertEqual(reviews["years"], [2020, 2021])
        # Raw-to-clean reconciliation.
        self.assertEqual(reviews["rows_read"], reviews["valid"] + sum(reviews["rejected"].values()))
        self.assertEqual(reviews["valid"] - reviews["duplicates_removed"], reviews["interactions"])
        self.assertEqual(stage["metadata"]["items_with_metadata"], 3)
        self.assertEqual(stage["metadata"]["interaction_items_without_metadata_row"], 0)
        self.assertEqual(stage["metadata"]["interaction_items_without_title"], 1)  # i3: blank title only
        self.assertEqual(stage["resources"]["master"], self.config.spark.master)
        self.assertEqual(set(stage["timings_seconds"]), {
            "spark_start", "reviews_classify_count", "reviews_deduplicate_write", "metadata_count_write",
            "verification_counts", "total",
        })

        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
        try:
            rows = spark.read.parquet(str(output / "interactions")).collect()
            kept = {(r.user_id, r.parent_asin): (r.rating, r.timestamp, r.year) for r in rows}
            self.assertEqual(kept[("u2", "i1")], (2.0, T0 + 400 * DAY, 2021))
            self.assertEqual(kept[("u3", "i1")][0], 4.0)
            self.assertEqual(kept[("u4", "i3")][0], 1.0)
            titles = {r.parent_asin: r.title for r in spark.read.parquet(str(output / "metadata")).collect()}
            self.assertEqual(titles, {"i1": "Movie One", "i2": "Movie Two", "i3": None})
        finally:
            spark.stop()


if __name__ == "__main__":
    unittest.main()
