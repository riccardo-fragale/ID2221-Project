"""Sampling preconditions run everywhere; Spark tests run only with RUN_SPARK_TESTS=1 (make check-spark)."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.config import load_config
from amazon_recommender.etl import run_etl
from amazon_recommender.cli import main
from amazon_recommender.sample import run_sample, sample_label


ROOT = Path(__file__).resolve().parents[1]
T0 = 1_577_836_800_000  # 2020-01-01 UTC
USERS = 200


class SampleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        self.config = replace(
            base, storage_root=self.root / "storage", experiment=replace(base.experiment, user_fraction=0.5)
        )
        self.reviews, self.metadata = self.root / "reviews.ndjson", self.root / "metadata.ndjson"
        # User n reviews n % 4 + 1 distinct items, so histories differ in length.
        self.reviews.write_text("".join(
            json.dumps({"user_id": f"u{n}", "parent_asin": f"i{k}", "rating": 4.0, "timestamp": T0 + k})
            + "\n"
            for n in range(USERS) for k in range(n % 4 + 1)
        ))
        self.metadata.write_text(json.dumps({"parent_asin": "i0", "title": "Item"}) + "\n")
        self.run_directory = create_run(self.config, reviews=self.reviews, metadata=self.metadata)


class PreconditionTests(SampleTests):
    def test_requires_completed_etl(self):
        with self.assertRaises(FileNotFoundError):
            run_sample(self.config, self.run_directory)

    def test_completed_sample_is_not_overwritten(self):
        (self.run_directory / "etl").mkdir()
        (self.run_directory / "etl/stage.json").write_text("{}")
        (self.run_directory / "samples" / sample_label(0.5, self.config.seed)).mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            run_sample(self.config, self.run_directory)

    def test_cli_rejects_invalid_fraction_override(self):
        (self.run_directory / "etl").mkdir()
        (self.run_directory / "etl/stage.json").write_text("{}")
        arguments = ["sample", "--config", str(ROOT / "configs/local.toml"), "--run", str(self.run_directory)]
        self.assertEqual(main([*arguments, "--user-fraction", "1.5"]), 2)


@unittest.skipUnless(os.environ.get("RUN_SPARK_TESTS") == "1", "set RUN_SPARK_TESTS=1 to start Spark")
class SparkSampleTests(SampleTests):
    def sampled(self, config, run_directory, *, etl=True):
        if etl:
            run_etl(config, run_directory)
        output = run_sample(config, run_directory)
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
        try:
            rows = spark.read.parquet(str(run_directory / json.loads((output / "stage.json").read_text())["outputs"]["interactions"])).collect()
        finally:
            spark.stop()
        counts = {}
        for row in rows:
            counts[row.user_id] = counts.get(row.user_id, 0) + 1
        return json.loads((output / "stage.json").read_text()), counts

    def test_sample_keeps_whole_histories_and_is_reproducible(self):
        stage, kept = self.sampled(self.config, self.run_directory)
        self.assertEqual(stage["users"]["before"], USERS)
        self.assertEqual(stage["users"]["after"], len(kept))
        self.assertTrue(0.3 < stage["users"]["achieved_fraction"] < 0.7, stage["users"])
        for user, count in kept.items():
            self.assertEqual(count, int(user[1:]) % 4 + 1)  # full history retained
        self.assertEqual(stage["interactions"]["after"], sum(kept.values()))
        self.assertIn("select_write", stage["timings_seconds"])

        again = create_run(self.config, reviews=self.reviews, metadata=self.metadata)
        _, repeated = self.sampled(self.config, again)
        self.assertEqual(set(kept), set(repeated))

        other = replace(self.config, seed=self.config.seed + 1)
        different = create_run(other, reviews=self.reviews, metadata=self.metadata)
        _, changed = self.sampled(other, different)
        self.assertNotEqual(set(kept), set(changed))

    def test_full_fraction_reuses_etl_output(self):
        config = replace(self.config, experiment=replace(self.config.experiment, user_fraction=1.0))
        directory = create_run(config, reviews=self.reviews, metadata=self.metadata)
        stage, kept = self.sampled(config, directory)
        self.assertEqual(stage["outputs"]["interactions"], "etl/interactions")
        self.assertEqual(len(kept), USERS)
        self.assertEqual(stage["users"]["after"], USERS)

    def test_one_etl_serves_nested_samples(self):
        _, half = self.sampled(self.config, self.run_directory)
        quarter_config = replace(self.config, experiment=replace(self.config.experiment, user_fraction=0.25))
        stage, quarter = self.sampled(quarter_config, self.run_directory, etl=False)
        self.assertEqual(stage["config_digest"], quarter_config.digest)
        self.assertTrue(set(quarter) < set(half))  # same seed: smaller fraction is a subset


if __name__ == "__main__":
    unittest.main()
