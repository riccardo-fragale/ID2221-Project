"""Materialize stage. Spark tests run only with RUN_SPARK_TESTS=1 (make check-spark).

Expected lists are rebuilt in plain Python from the split's Parquet, not with the model code.
"""

from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from amazon_recommender.artifacts import create_run
from amazon_recommender.config import load_config
from amazon_recommender.materialize import audit, run_materialize


ROOT = Path(__file__).resolve().parents[1]
T0 = 1_577_836_800_000  # 2020-01-01 UTC
SPARK = os.environ.get("RUN_SPARK_TESTS") == "1"
SAMPLE, SPLIT = "fraction-1-seed-2221", "p0.8-0.1-0.1-k2-2"
POPULARITY = "popularity-smoothing10-window_daysall"


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        self.config = replace(
            base, storage_root=self.root / "storage",
            experiment=replace(base.experiment, user_fraction=1.0, user_min_reviews=2, item_min_reviews=2),
            model=replace(base.model, recommendation_count=5, rank=2, iterations=3),
        )
        # 40 users; user n rates n % 5 + 1 of 12 items at increasing times, so late users fall in the
        # held-out windows only (new) and one-review users are removed by the k-core. "heavy" rates
        # 10 items first, leaving fewer than 5 unseen catalog items: a short list.
        rows = [{"user_id": f"u{n:02d}", "parent_asin": f"i{(n + k) % 12:02d}", "rating": 1.0 + (n + k) % 5,
                 "timestamp": T0 + 1000 * n + k} for n in range(40) for k in range(n % 5 + 1)]
        rows += [{"user_id": "heavy", "parent_asin": f"i{k:02d}", "rating": 4.0, "timestamp": T0 - 1000 + k}
                 for k in range(10)]
        reviews, metadata = self.root / "reviews.ndjson", self.root / "metadata.ndjson"
        reviews.write_text("".join(json.dumps(row) + "\n" for row in rows))
        metadata.write_text(json.dumps({"parent_asin": "i00", "title": "Item"}) + "\n")
        self.run_directory = create_run(self.config, reviews=reviews, metadata=metadata)
        self.split = self.run_directory / "samples" / SAMPLE / "splits" / SPLIT

    def test_fallback_must_be_popularity(self):
        for label in (POPULARITY, "bias"):
            path = self.split / "models" / label
            path.mkdir(parents=True)
            (path / "stage.json").write_text(json.dumps({"model": {"model": label.split("-")[0]}}))
        (self.split / "stage.json").write_text(json.dumps({"run_id": "r"}))
        (self.split.parents[1] / "stage.json").write_text(json.dumps({"outputs": {}}))
        with self.assertRaisesRegex(ValueError, "popularity"):
            run_materialize(self.config, self.run_directory, SAMPLE, SPLIT, POPULARITY, "bias")


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class SparkMaterializeTests(MaterializeTests):
    def prepare(self, models=("popularity", "als")):
        from amazon_recommender.etl import run_etl
        from amazon_recommender.sample import run_sample
        from amazon_recommender.split import run_split
        from amazon_recommender.train import run_train

        run_etl(self.config, self.run_directory)
        run_sample(self.config, self.run_directory)
        run_split(self.config, self.run_directory, SAMPLE)
        return {name: run_train(self.config, self.run_directory, SAMPLE, SPLIT, name).name for name in models}

    def read(self, serving):
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
        try:
            users = {r.user_id: r.asDict() for r in spark.read.parquet(str(serving / "users")).collect()}
            lists = {}
            for r in spark.read.parquet(str(serving / "recommendations")).orderBy("user_id", "rank").collect():
                lists.setdefault(r.user_id, []).append((r.rank, r.parent_asin, r.score))
            train = spark.read.parquet(str(self.split / "train")).collect()
            items = {r.parent_asin for r in spark.read.parquet(str(self.split / "items")).collect()}
            interactions = spark.read.parquet(str(self.run_directory / "etl/interactions")).collect()
        finally:
            spark.stop()
        return users, lists, train, items, interactions

    def test_every_user_gets_an_exact_list_and_fallbacks_are_labeled(self):
        labels = self.prepare()
        serving = run_materialize(self.config, self.run_directory, SAMPLE, SPLIT, labels["als"], labels["popularity"],
                                  batch_users=7)
        stage = json.loads((serving / "stage.json").read_text())
        split_stage = json.loads((self.split / "stage.json").read_text())
        users, lists, train, catalog, interactions = self.read(serving)
        cutoff = split_stage["cutoffs_ms"]["train"]
        known = {r.user_id for r in train}

        # Every sample user once; types follow the split.
        self.assertEqual(set(users), {r.user_id for r in interactions})
        pre_cutoff = {r.user_id for r in interactions if r.timestamp <= cutoff}
        for user, row in users.items():
            expected = "known" if user in known else "removed_by_k_core" if user in pre_cutoff else "new"
            self.assertEqual(row["user_type"], expected, user)
            self.assertEqual(row["rec_type"], "model" if user in known else "fallback", user)
            self.assertFalse(row["approximate"])
        types = Counter(row["user_type"] for row in users.values())
        self.assertTrue(all(types[t] for t in ("known", "removed_by_k_core", "new")), types)
        self.assertEqual(stage["coverage"]["users_by_type"], dict(types))
        self.assertEqual(stage["counts"]["users"], len(users))

        # Lists: unseen catalog items only, unique, ranks 1..n, min(k, unseen) long.
        seen = {}
        for r in interactions:
            if r.timestamp <= cutoff and r.parent_asin in catalog:
                seen.setdefault(r.user_id, set()).add(r.parent_asin)
        for user, row in users.items():
            items = lists.get(user, [])
            unseen = len(catalog) - len(seen.get(user, ()))
            self.assertEqual(len(items), min(5, unseen), user)
            self.assertEqual(row["length"], len(items))
            self.assertEqual([rank for rank, _, _ in items], list(range(1, len(items) + 1)))
            self.assertFalse({asin for _, asin, _ in items} & seen.get(user, set()), user)
            self.assertEqual(len({asin for _, asin, _ in items}), len(items))
        self.assertEqual(users["heavy"]["length"], 2)  # 12 catalog items, 10 seen
        self.assertEqual(stage["coverage"]["short_lists"]["count"], 1)

        # The fallback is the training popularity order (count desc, then original ID), minus seen items.
        counts = Counter(r.parent_asin for r in train)
        order = sorted(catalog, key=lambda asin: (-counts[asin], asin))
        for user, row in users.items():
            if row["rec_type"] == "fallback":
                expected = [a for a in order if a not in seen.get(user, set())][:5]
                self.assertEqual([asin for _, asin, _ in lists.get(user, [])], expected, user)
                self.assertEqual([score for _, _, score in lists.get(user, [])],
                                 [float(counts[a]) for a in expected])

        # Batches: consecutive serving indices, all recorded.
        self.assertEqual(stage["batches"]["count"], -(-len(users) // 7))
        self.assertEqual(sum(b["users"] for b in stage["batches"]["records"]), len(users))
        with self.assertRaises(FileExistsError):
            run_materialize(self.config, self.run_directory, SAMPLE, SPLIT, labels["als"], labels["popularity"])

    def test_interrupted_run_resumes_from_completed_batches(self):
        import amazon_recommender.materialize as materialize

        labels = self.prepare(models=("popularity",))
        original, calls = materialize.recommend_batch, []

        def fail_on_third(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("simulated interruption")
            return original(*args, **kwargs)

        with mock.patch.object(materialize, "recommend_batch", fail_on_third), \
                self.assertRaisesRegex(RuntimeError, "simulated"):
            run_materialize(self.config, self.run_directory, SAMPLE, SPLIT, POPULARITY, POPULARITY, batch_users=10)
        partial = self.split / "serving" / f"{POPULARITY}.partial"
        self.assertEqual(len(list((partial / "batches").glob("*.json"))), 2)
        with self.assertRaisesRegex(FileExistsError, "other arguments"):
            run_materialize(self.config, self.run_directory, SAMPLE, SPLIT, POPULARITY, POPULARITY, batch_users=5)
        serving = run_materialize(self.config, self.run_directory, SAMPLE, SPLIT, POPULARITY, POPULARITY,
                                  batch_users=10)
        stage = json.loads((serving / "stage.json").read_text())
        self.assertEqual(stage["batches"]["reused_on_resume"], 2)
        users, lists, *_ = self.read(serving)
        self.assertEqual(stage["counts"]["users"], len(users))
        self.assertEqual(sum(len(items) for items in lists.values()), stage["coverage"]["rows"])
        self.assertEqual(labels["popularity"], POPULARITY)


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class AuditTests(unittest.TestCase):
    """Each case breaks one rule of a valid k=2 materialization; the named check must fire."""

    USERS = [("u1", "known", "model", False, 5, 2), ("u2", "new", "fallback", False, 1, 1)]
    ROWS = [("u1", 1, "a", 2.0), ("u1", 2, "b", 1.0), ("u2", 1, "c", 1.0)]
    SEEN = [("u1", "z")]

    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession

        cls.spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def check(self, users=USERS, rows=ROWS, seen=SEEN):
        frames = (
            self.spark.createDataFrame(users, "user_id string, user_type string, rec_type string, approximate boolean, "
                                              "unseen_items long, length long"),
            self.spark.createDataFrame(rows, "user_id string, rank int, parent_asin string, score double"),
            self.spark.createDataFrame(seen, "user_id string, parent_asin string"),
        )
        return audit(self.spark, *frames, 2)

    def test_valid_lists_pass(self):
        coverage = self.check()
        self.assertEqual((coverage["users"], coverage["rows"]), (2, 3))
        self.assertEqual(coverage["users_by_rec_type"], {"model": 1, "fallback": 1})
        self.assertEqual(coverage["short_lists"]["count"], 1)

    def test_each_defect_is_caught(self):
        users, rows = self.USERS, self.ROWS
        cases = [  # (check that must fire, one defect)
            ("malformed_lists", dict(rows=[("u1", 1, "a", 2.0), ("u1", 2, "a", 1.0), rows[2]])),  # duplicate item
            ("malformed_lists", dict(rows=[rows[0], ("u1", 3, "b", 1.0), rows[2]])),  # rank gap
            ("seen_items_recommended", dict(seen=[("u1", "a")])),
            ("lengths_not_min_k_unseen", dict(users=[users[0], ("u2", "new", "fallback", False, 5, 1)])),
            ("scores_increasing_with_rank", dict(rows=[("u1", 1, "a", 1.0), ("u1", 2, "b", 2.0), rows[2]])),
            ("missing_or_nan_scores", dict(rows=[*rows[:2], ("u2", 1, "c", float("nan"))])),
            ("rows_of_unlisted_users", dict(rows=[*rows, ("u9", 1, "a", 1.0)])),
            ("duplicate_users", dict(users=[*users, users[1]])),
        ]
        for check, arguments in cases:
            with self.subTest(check=check), self.assertRaisesRegex(RuntimeError, check):
                self.check(**arguments)


if __name__ == "__main__":
    unittest.main()
