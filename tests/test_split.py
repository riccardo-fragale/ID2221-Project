"""Split preconditions run everywhere; Spark tests run only with RUN_SPARK_TESTS=1 (make check-spark)."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.config import load_config
from amazon_recommender.etl import run_etl
from amazon_recommender.sample import run_sample, sample_label
from amazon_recommender.split import run_split, split_label


ROOT = Path(__file__).resolve().parents[1]
T0 = 1_577_836_800_000  # 2020-01-01 UTC
SPARK = os.environ.get("RUN_SPARK_TESTS") == "1"


def rows(*triples):
    """(user, item, timestamp offset) triples as interaction rows with rating 4."""
    return [(user, item, 4.0, T0 + offset) for user, item, offset in triples]


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class SplitFunctionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession

        cls.directory = tempfile.TemporaryDirectory()
        cls.spark = (
            SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", 3).config("spark.sql.session.timeZone", "UTC").getOrCreate()
        )
        cls.spark.sparkContext.setCheckpointDir(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        cls.directory.cleanup()

    def frame(self, data):
        return self.spark.createDataFrame(data, "user_id string, parent_asin string, rating double, timestamp long")

    def windows(self, df, split):
        from amazon_recommender.split import assign_windows, temporal_cutoffs

        cutoffs = temporal_cutoffs(df, split)
        return cutoffs, {name: frame.count() for name, frame in assign_windows(df, *cutoffs).items()}

    def test_distinct_timestamps_split_by_count(self):
        df = self.frame(rows(*[(f"u{n}", "a", n) for n in range(10)]))
        cutoffs, counts = self.windows(df, [0.8, 0.1, 0.1])
        self.assertEqual(cutoffs, (T0 + 7, T0 + 8))
        self.assertEqual(counts, {"train": 8, "validation": 1, "test": 1})

    def test_tie_group_on_the_boundary_stays_in_one_window(self):
        # Six distinct timestamps, then three rows sharing offset 6 straddle the 80% mark (rows 7-9).
        df = self.frame(rows(*[(f"u{n}", "a", n) for n in range(6)], ("x", "a", 6), ("y", "a", 6), ("z", "a", 6),
                             ("w", "a", 7)))
        cutoffs, counts = self.windows(df, [0.8, 0.1, 0.1])
        self.assertEqual(cutoffs, (T0 + 6, T0 + 6))
        self.assertEqual(counts, {"train": 9, "validation": 0, "test": 1})

    def test_training_only_split_has_empty_held_out_windows(self):
        df = self.frame(rows(*[(f"u{n}", "a", n) for n in range(5)]))
        _, counts = self.windows(df, [1.0, 0.0, 0.0])
        self.assertEqual(counts, {"train": 5, "validation": 0, "test": 0})

    def test_k_core_cascades_until_converged(self):
        from amazon_recommender.split import k_core

        # u4 (one item) goes first; then c has one user, then u3 has one item.
        df = self.frame(rows(("u1", "a", 0), ("u1", "b", 1), ("u2", "a", 2), ("u2", "b", 3),
                             ("u3", "b", 4), ("u3", "c", 5), ("u4", "c", 6)))
        core, iterations = k_core(df, 2, 2)
        self.assertEqual({(r.user_id, r.parent_asin) for r in core.collect()},
                         {("u1", "a"), ("u1", "b"), ("u2", "a"), ("u2", "b")})
        self.assertEqual(iterations, [
            {"users_removed": 1, "items_removed": 0, "interactions_removed": 1},
            {"users_removed": 0, "items_removed": 1, "interactions_removed": 1},
            {"users_removed": 1, "items_removed": 0, "interactions_removed": 1},
        ])

    def test_projection_counts_each_kind_of_unknown(self):
        from amazon_recommender.split import project

        # Training saw u1, u2, a and b; the k-core kept only u1 and a.
        train = self.frame(rows(("u1", "a", 0), ("u2", "b", 1)))
        users = self.spark.createDataFrame([("u1",)], "user_id string")
        items = self.spark.createDataFrame([("a",)], "parent_asin string")
        held_out = self.frame(rows(("u1", "a", 2), ("u2", "a", 3), ("u1", "b", 4), ("u2", "b", 5), ("u3", "b", 6),
                                   ("u3", "c", 7)))
        kept, counts = project(held_out, users, items, train)
        self.assertEqual([(r.user_id, r.parent_asin) for r in kept.collect()], [("u1", "a")])
        self.assertEqual(kept.columns, held_out.columns)
        self.assertEqual(counts, {
            "rows": 6, "kept": 1, "unknown_user_only": 1, "unknown_item_only": 1, "unknown_user_and_item": 3,
            "users_without_training_history": 1, "users_removed_by_k_core": 1,  # u3; u2
            "items_without_training_history": 1, "items_removed_by_k_core": 1,  # c; b
        })

    def test_mapping_is_dense_ordered_and_independent_of_partitioning(self):
        from amazon_recommender.split import dense_mapping, mapping_summary

        ids = [f"i{n:03d}" for n in range(57)][::-1]
        df = self.frame(rows(*[("u", item, n) for n, item in enumerate(ids)]))
        results = []
        for partitions in (1, 7):
            mapping = dense_mapping(df.repartition(partitions), "parent_asin", "item_index")
            results.append((sorted(mapping.collect()), mapping_summary(mapping, "parent_asin", "item_index")))
        self.assertEqual(results[0], results[1])
        pairs, summary = results[0]
        self.assertEqual([(r.parent_asin, r.item_index) for r in pairs], [(item, n) for n, item in enumerate(sorted(ids))])
        self.assertEqual((summary["count"], summary["max_index"]), (57, 56))


class SplitRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        self.config = replace(base, storage_root=self.root / "storage", experiment=replace(
            base.experiment, user_fraction=1.0, user_min_reviews=2, item_min_reviews=2
        ))
        self.reviews, self.metadata = self.root / "reviews.ndjson", self.root / "metadata.ndjson"
        # 40 users; user n rates n % 5 + 1 of 12 items, at increasing times.
        self.reviews.write_text("".join(
            json.dumps({"user_id": f"u{n:02d}", "parent_asin": f"i{(n + k) % 12:02d}", "rating": 1.0 + (n + k) % 5,
                        "timestamp": T0 + 1000 * n + k}) + "\n"
            for n in range(40) for k in range(n % 5 + 1)
        ))
        self.metadata.write_text(json.dumps({"parent_asin": "i00", "title": "Item"}) + "\n")
        self.run_directory = create_run(self.config, reviews=self.reviews, metadata=self.metadata)
        self.sample = sample_label(1.0, self.config.seed)


class PreconditionTests(SplitRunTests):
    def test_requires_completed_sample(self):
        with self.assertRaises(FileNotFoundError):
            run_split(self.config, self.run_directory, self.sample)

    def test_cutoffs_and_refit_sources_are_exclusive(self):
        with self.assertRaises(ValueError):
            run_split(self.config, self.run_directory, self.sample, cutoffs_from=Path("a"), refit_from=Path("b"))

    def test_completed_split_is_not_overwritten(self):
        sample = self.run_directory / "samples" / self.sample
        (sample / "splits" / split_label([0.8, 0.1, 0.1], 2, 2)).mkdir(parents=True)
        (sample / "stage.json").write_text(json.dumps({"config": self.config.as_dict()}))
        with self.assertRaises(FileExistsError):
            run_split(self.config, self.run_directory, self.sample)


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class SparkSplitTests(SplitRunTests):
    def read(self, directory):
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
        try:
            return {name: spark.read.parquet(str(directory / name)).collect()
                    for name in ("train", "validation", "test", "users", "items")}
        finally:
            spark.stop()

    def test_split_end_to_end_and_frozen_cutoffs(self):
        run_etl(self.config, self.run_directory)
        run_sample(self.config, self.run_directory)
        output = run_split(self.config, self.run_directory, self.sample)
        stage = json.loads((output / "stage.json").read_text())
        data = self.read(output)

        before = stage["before_filtering"]["interactions"]
        self.assertEqual(sum(before.values()), sum(n % 5 + 1 for n in range(40)))
        self.assertEqual(stage["interactions"]["train"], len(data["train"]))
        for name in ("validation", "test"):
            self.assertEqual(stage["projection"][name]["rows"], before[name])
            self.assertEqual(stage["projection"][name]["kept"], len(data[name]))

        # Windows are disjoint in time and in (user, item) pairs.
        cutoffs = stage["cutoffs_ms"]
        self.assertTrue(all(r.timestamp <= cutoffs["train"] for r in data["train"]))
        self.assertTrue(all(cutoffs["train"] < r.timestamp <= cutoffs["validation"] for r in data["validation"]))
        self.assertTrue(all(r.timestamp > cutoffs["validation"] for r in data["test"]))
        pairs = [{(r.user_id, r.parent_asin) for r in data[name]} for name in ("train", "validation", "test")]
        self.assertFalse(pairs[0] & pairs[1] or pairs[0] & pairs[2] or pairs[1] & pairs[2])

        # The retained training graph satisfies both thresholds.
        degrees = {}
        for r in data["train"]:
            degrees[("u", r.user_id)] = degrees.get(("u", r.user_id), 0) + 1
            degrees[("i", r.parent_asin)] = degrees.get(("i", r.parent_asin), 0) + 1
        self.assertTrue(degrees and min(degrees.values()) >= 2)

        # Held-out rows use training vocabularies; indices round-trip to the original IDs.
        users = {r.user_id: r.user_index for r in data["users"]}
        items = {r.parent_asin: r.item_index for r in data["items"]}
        self.assertEqual(sorted(users.values()), list(range(len(users))))
        for name in ("train", "validation", "test"):
            for r in data[name]:
                self.assertEqual((users[r.user_id], items[r.parent_asin]), (r.user_index, r.item_index))
        self.assertEqual(stage["mappings"]["users"]["count"], len(users))
        self.assertAlmostEqual(sum(stage["proportions"].values()), 1.0)
        checkpoints = self.config.storage_root / "checkpoints"
        self.assertFalse([path for path in checkpoints.rglob("*") if path.is_file()])  # split checkpoints removed

        # Copying the cutoffs into a new split reproduces the same windows.
        copy = run_split(self.config, self.run_directory, self.sample, cutoffs_from=output / "stage.json")
        copied = json.loads((copy / "stage.json").read_text())
        self.assertNotEqual(copy, output)
        self.assertEqual(copied["cutoffs_ms"], stage["cutoffs_ms"])
        self.assertEqual(copied["before_filtering"], stage["before_filtering"])
        self.assertEqual(copied["mappings"], stage["mappings"])
        self.assertEqual(copied["policy"]["cutoffs_from"]["path"], str(output / "stage.json"))

        # Refit: train is the source's train + validation window, validation is empty, test unchanged.
        refit = run_split(self.config, self.run_directory, self.sample, refit_from=output / "stage.json")
        refitted = json.loads((refit / "stage.json").read_text())
        self.assertTrue(refit.name.startswith("refit-"))
        self.assertEqual(refitted["cutoffs_ms"], {"train": cutoffs["validation"], "validation": cutoffs["validation"]})
        self.assertEqual(refitted["before_filtering"]["interactions"], {
            "train": before["train"] + before["validation"], "validation": 0, "test": before["test"],
        })
        self.assertEqual(refitted["policy"]["refit_from"]["path"], str(output / "stage.json"))
        self.assertIsNone(refitted["policy"]["cutoffs_from"])
        self.assertEqual(refitted["interactions"]["validation"], 0)


if __name__ == "__main__":
    unittest.main()
