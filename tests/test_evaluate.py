"""Evaluation metrics and the popularity window rule. Spark tests run only with RUN_SPARK_TESTS=1.

Expected values are worked out by hand in the comments.
"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.config import load_config
from amazon_recommender.evaluate import choose_window


ROOT = Path(__file__).resolve().parents[1]

T0 = 1_577_836_800_000  # 2020-01-01 UTC
SPARK = os.environ.get("RUN_SPARK_TESTS") == "1"


class PoissonTailTests(unittest.TestCase):
    def test_hand_values(self):
        from amazon_recommender.evaluate import poisson_upper_tail

        self.assertEqual(poisson_upper_tail(0, 3.0), 1.0)
        self.assertAlmostEqual(poisson_upper_tail(1, 3.0), 1 - math.exp(-3))
        self.assertAlmostEqual(poisson_upper_tail(2, 3.0), 1 - math.exp(-3) * (1 + 3))
        self.assertEqual(poisson_upper_tail(3, 0.0), 0.0)

    def test_far_tail_stays_finite_in_log_space(self):
        from amazon_recommender.evaluate import poisson_upper_tail_log10

        # P(X >= 20 | 1) = sum_{n>=20} e^-1 / n!; the sum is 10^-18.80.
        self.assertAlmostEqual(poisson_upper_tail_log10(20, 1.0), -18.80, places=2)
        self.assertLess(poisson_upper_tail_log10(6000, 30.0), -10000)  # 10**that underflows; the log does not


class WindowRuleTests(unittest.TestCase):
    def test_longest_window_not_detectably_worse_than_the_best(self):
        candidates = {
            30: [1, 0, 0, 0, 0, 0, 0, 0],      # mean 1/8
            90: [1, 1, 1, 1, 0, 0, 0, 0],      # mean 1/2: best
            365: [1, 1, 1, 0, 0, 0, 0, 0],     # loses one user: a resample without that user (p = (7/8)^8 = 0.34)
                                               # gives difference 0, so the 95% interval reaches 0
            "all": [0, 0, 0, 0, 0, 0, 0, 0],   # loses four users: difference 0 only with p = (1/2)^8 < 2.5%
        }
        choice = choose_window(candidates, seed=2221)
        self.assertEqual((choice["best"], choice["chosen"]), (90, 365))
        self.assertEqual(choice["difference_vs_best_ci95"][90], [0.0, 0.0])
        self.assertLess(choice["difference_vs_best_ci95"]["all"][1], 0)

    def test_equal_means_prefer_the_longer_window(self):
        choice = choose_window({30: [1, 0], 90: [0, 1]}, seed=1)
        self.assertEqual(choice["best"], 90)

    def test_shorter_windows_are_never_chosen_over_the_best(self):
        choice = choose_window({30: [1, 1, 0, 0], 90: [1, 1, 1, 0]}, seed=1)
        self.assertEqual((choice["best"], choice["chosen"]), (90, 90))


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession

        cls.directory = tempfile.TemporaryDirectory()
        cls.spark = (
            SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", 3).config("spark.sql.session.timeZone", "UTC").getOrCreate()
        )
        cls.spark.sparkContext.setCheckpointDir(str(Path(cls.directory.name) / "checkpoints"))

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        cls.directory.cleanup()

    def test_ranking_per_user_hand_values(self):
        from amazon_recommender.evaluate import ranking_per_user

        recommendations = self.spark.createDataFrame(
            [(1, 10, 1, 0.0), (1, 11, 2, 0.0), (1, 12, 3, 0.0), (2, 10, 1, 0.0)],
            "user_index int, item_index int, rank int, score double",
        )
        relevant = self.spark.createDataFrame([(1, 11), (1, 20), (2, 10), (3, 5)], "user_index int, item_index int")
        rows = {r.user_index: r for r in ranking_per_user(recommendations, relevant, 3).collect()}
        # u1: one hit at rank 2, two relevant items: DCG 1/log2(3), IDCG 1 + 1/log2(3).
        self.assertEqual((rows[1].relevant, rows[1].returned, rows[1].hits), (2, 3, 1))
        self.assertAlmostEqual(rows[1].ndcg, (1 / math.log2(3)) / (1 + 1 / math.log2(3)), places=12)
        self.assertAlmostEqual(rows[1].precision, 1 / 3, places=12)
        # u2: a short list (one item) with a hit at rank 1: NDCG 1, precision still divided by k = 3.
        self.assertEqual((rows[2].returned, rows[2].hits, rows[2].ndcg), (1, 1, 1.0))
        self.assertAlmostEqual(rows[2].precision, 1 / 3, places=12)
        # u3: nothing returned still counts, with zeros.
        self.assertEqual((rows[3].returned, rows[3].hits, rows[3].ndcg, rows[3].precision), (0, 0, 0.0, 0.0))

    def test_evaluate_popularity_cohort_ranking_and_rmse(self):
        from amazon_recommender.evaluate import evaluate
        from amazon_recommender.models import PopularityModel

        train = self.spark.createDataFrame(
            [(0, 0, 5.0, T0), (0, 1, 3.0, T0), (1, 0, 4.0, T0), (1, 2, 2.0, T0), (2, 1, 4.0, T0)],
            "user_index int, item_index int, rating double, timestamp long",
        )
        held_out = self.spark.createDataFrame([(0, 2, 5.0), (1, 1, 2.0), (2, 0, 4.0)],
                                              "user_index int, item_index int, rating double")
        model = PopularityModel({"smoothing": 1.0, "window_days": "all"}).fit(train)
        summary, per_user = evaluate(model, train, held_out, k=10, relevant_rating=4.0, seed=1, catalog_size=3)
        per_user.unpersist()
        # u1's only held-out rating is 2: not in the cohort. u0 gets [i2] (hit at rank 1) and u2 gets
        # [i0, i2] (hit at rank 1): NDCG 1 each, precision 1/10 each, both short lists.
        self.assertEqual(summary["cohort"], {"held_out_rows": 3, "held_out_users": 3, "eligible_users": 2,
                                             "users_without_relevant": 1, "relevant_pairs": 2})
        ranking = summary["ranking"]
        self.assertEqual((ranking["hits"], ranking["users_with_hit"], ranking["short_lists"]), (2, 2, 2))
        self.assertAlmostEqual(ranking["ndcg_at_k"], 1.0)
        self.assertAlmostEqual(ranking["precision_at_k"], 0.1)
        # u0 has 2 training items and u2 one: both in the "<5" segment.
        self.assertEqual(summary["segments"]["<5"]["users"], 2)
        self.assertEqual(summary["segments"]["5-9"], {"users": 0})
        # Random over the unseen catalog: u0 has one candidate (i2), which is relevant, so a random
        # list hits surely; u2 has two candidates, one relevant, both shown: a sure hit too.
        self.assertAlmostEqual(summary["random_reference"]["expected_users_with_hit"], 2.0)
        self.assertAlmostEqual(summary["signal_vs_random"]["p_value_one_sided"], 1 - math.exp(-2) * (1 + 2))
        # Recall@50/100: each user's single relevant item is returned. Both lists are short of 100.
        self.assertAlmostEqual(ranking["recall_at_50"], 1.0)
        self.assertEqual(ranking["short_lists_at_100"], 2)
        self.assertEqual(summary["segments"]["<5"]["short_lists_at_100"], 2)
        # Predictions 2.8, 10.6/3 and 4.2 against 5, 2 and 4 (all inside [1,5], so no clipping).
        rating = summary["rating"]
        self.assertEqual((rating["rows"], rating["predicted"], rating["clipped_rows"]), (3, 3, 0))
        self.assertAlmostEqual(rating["rmse"], math.sqrt((2.2 ** 2 + (10.6 / 3 - 2) ** 2 + 0.2 ** 2) / 3), places=12)
        self.assertAlmostEqual(rating["rmse_raw"], rating["rmse"], places=12)

    def test_random_expectation_hand_values(self):
        from amazon_recommender.evaluate import with_random_expectation

        # C' = 4 unseen items, 1 relevant, k = 2: hit = 1 - (3/4)(2/3) = 1/2; precision = (1/4) * 2 / 2;
        # NDCG = (1/4)(1 + 1/log2(3)) / 1; recall@3 = 3/4. A user with C' = 0 gets zeros.
        frame = self.spark.createDataFrame([(1, 1, 4, 1.0), (2, 1, 0, 1.0)],
                                           "user_index int, relevant int, candidates int, idcg double")
        rows = {r.user_index: r for r in with_random_expectation(frame, 2, (3,)).collect()}
        self.assertAlmostEqual(rows[1].random_hit, 0.5, places=12)
        self.assertAlmostEqual(rows[1].random_precision, 0.25, places=12)
        self.assertAlmostEqual(rows[1].random_ndcg, 0.25 * (1 + 1 / math.log2(3)), places=12)
        self.assertAlmostEqual(rows[1].random_recall_at_3, 0.75, places=12)
        self.assertEqual((rows[2].random_hit, rows[2].random_precision, rows[2].random_ndcg), (0.0, 0.0, 0.0))

    def test_recall_counts_deeper_ranks_of_the_same_lists(self):
        from amazon_recommender.evaluate import ranking_per_user

        recommendations = self.spark.createDataFrame(
            [(1, 10, 1, 0.0), (1, 11, 2, 0.0), (1, 12, 3, 0.0)], "user_index int, item_index int, rank int, score double")
        relevant = self.spark.createDataFrame([(1, 12), (1, 20)], "user_index int, item_index int")
        row = ranking_per_user(recommendations, relevant, 2, (3,)).first()
        # Rank 3 is outside k = 2 (no hit, two returned) but inside recall@3: 1 of 2 relevant items.
        self.assertEqual((row.returned, row.returned_all, row.hits, row.hits_at_3), (2, 3, 0, 1))
        self.assertAlmostEqual(row.recall_at_3, 0.5)

    def test_oracle_ranks_by_relevant_held_out_ratings(self):
        from amazon_recommender.evaluate import OraclePopularity, oracle_item_scores

        items = self.spark.createDataFrame([(0,), (1,), (2,)], "item_index int")
        held_out = self.spark.createDataFrame([(5, 2, 5.0), (6, 2, 4.0), (7, 1, 4.0), (8, 0, 2.0)],
                                              "user_index int, item_index int, rating double")
        # Relevant counts: i2 2, i1 1, i0 0 (its only rating is 2).
        scores = {r.item_index: r.item_score for r in oracle_item_scores(items, held_out, 4.0).collect()}
        self.assertEqual(scores, {0: 0, 1: 1, 2: 2})
        oracle = OraclePopularity(oracle_item_scores(items, held_out, 4.0))
        seen = self.spark.createDataFrame([(9, 2)], "user_index int, item_index int")
        users = self.spark.createDataFrame([(9,)], "user_index int")
        self.assertEqual([(r.item_index, r.rank) for r in oracle.recommend(users, seen, 2).orderBy("rank").collect()],
                         [(1, 1), (0, 2)])
        self.assertFalse(oracle.predicts_ratings)

    def test_rmse_clips_but_reports_raw(self):
        from amazon_recommender.evaluate import rating_metrics

        class Fixed:
            predicts_ratings = True

            def __init__(self, spark):
                self.spark = spark

            def predict(self, pairs):
                from pyspark.sql import functions as F

                # 6.0 for item 0 (clipped to 5), NaN for item 1 (counted as missing).
                return pairs.withColumn("prediction", F.when(F.col("item_index") == 0, 6.0).otherwise(float("nan")))

        held_out = self.spark.createDataFrame([(0, 0, 4.0), (0, 1, 3.0)], "user_index int, item_index int, rating double")
        rating = rating_metrics(Fixed(self.spark), held_out)
        self.assertEqual((rating["rows"], rating["predicted"], rating["missing_or_nonfinite"], rating["clipped_rows"]),
                         (2, 1, 1, 1))
        self.assertAlmostEqual(rating["rmse"], 1.0)      # |5 - 4|
        self.assertAlmostEqual(rating["rmse_raw"], 2.0)  # |6 - 4|


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class EvaluateRunTests(unittest.TestCase):
    def test_evaluate_and_select_end_to_end(self):
        from amazon_recommender.etl import run_etl
        from amazon_recommender.evaluate import run_evaluate, run_select_popularity_window
        from amazon_recommender.sample import run_sample
        from amazon_recommender.split import run_split
        from amazon_recommender.train import run_train

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        config = replace(base, storage_root=root / "storage", experiment=replace(
            base.experiment, user_fraction=1.0, user_min_reviews=2, item_min_reviews=2
        ))
        # 40 users; user n rates n % 5 + 1 items. A user's k-th rating happens at k seconds, so later
        # windows hold the later ratings of users already seen in training. Ratings 1 + (n + 2k) % 5
        # make some third ratings (validation) and some later ones (test) relevant.
        reviews, metadata = root / "reviews.ndjson", root / "metadata.ndjson"
        reviews.write_text("".join(
            json.dumps({"user_id": f"u{n:02d}", "parent_asin": f"i{(n + k) % 12:02d}", "rating": 1.0 + (n + 2 * k) % 5,
                        "timestamp": T0 + 1000 * k + n}) + "\n"
            for n in range(40) for k in range(n % 5 + 1)
        ))
        metadata.write_text(json.dumps({"parent_asin": "i00", "title": "Item"}) + "\n")
        run_directory = create_run(config, reviews=reviews, metadata=metadata)
        run_etl(config, run_directory)
        run_sample(config, run_directory)
        sample = "fraction-1-seed-2221"
        split = run_split(config, run_directory, sample)
        labels = [run_train(replace(config, model=replace(config.model, popularity_window_days=window)),
                            run_directory, sample, split.name, "popularity").name for window in (1, "all")]

        stages = [json.loads((run_evaluate(config, run_directory, sample, split.name, label, "validation")
                              / "stage.json").read_text()) for label in labels]
        cohort = stages[0]["metrics"]["cohort"]
        self.assertGreater(cohort["eligible_users"], 0)
        self.assertEqual(stages[1]["metrics"]["cohort"], cohort)  # same held-out rows, same cohort
        self.assertEqual(sum(segment["users"] for segment in stages[0]["metrics"]["segments"].values()),
                         cohort["eligible_users"])
        with self.assertRaises(FileExistsError):
            run_evaluate(config, run_directory, sample, split.name, labels[0], "validation")
        with self.assertRaises(ValueError):  # test only on a refit split
            run_evaluate(config, run_directory, sample, split.name, labels[0], "test")

        selection = json.loads(run_select_popularity_window(config, run_directory, sample, split.name).read_text())
        self.assertEqual(selection["users"], cohort["eligible_users"])
        self.assertIn(selection["chosen"], (1, "all"))
        with self.assertRaises(FileExistsError):
            run_select_popularity_window(config, run_directory, sample, split.name)

        # Additional views: a short horizon keeps a subset; coverage adds users unknown to training.
        horizon = json.loads((run_evaluate(config, run_directory, sample, split.name, labels[1], "validation",
                                           horizon_days=1) / "stage.json").read_text())
        self.assertEqual(horizon["view"]["name"], "validation-h1d")
        self.assertLessEqual(horizon["metrics"]["cohort"]["held_out_rows"], cohort["held_out_rows"])
        covered = json.loads((run_evaluate(config, run_directory, sample, split.name, labels[1], "validation",
                                           coverage=True) / "stage.json").read_text())
        types = covered["metrics"]["user_types"]
        self.assertEqual(sum(types[label]["users"] for label in types), covered["metrics"]["cohort"]["eligible_users"])
        self.assertGreaterEqual(covered["metrics"]["cohort"]["eligible_users"], cohort["eligible_users"])
        self.assertEqual(types["known"]["users"], cohort["eligible_users"])
        oracle = json.loads((run_evaluate(config, run_directory, sample, split.name, "oracle-popularity", "validation")
                             / "stage.json").read_text())
        self.assertTrue(oracle["reference_only"])
        self.assertIsNone(oracle["metrics"]["rating"])
        self.assertEqual(oracle["metrics"]["cohort"], cohort)  # same view, same cohort

        refit = run_split(config, run_directory, sample, refit_from=split / "stage.json")
        label = run_train(config, run_directory, sample, refit.name, "popularity").name
        stage = json.loads((run_evaluate(config, run_directory, sample, refit.name, label, "test")
                            / "stage.json").read_text())
        self.assertEqual(stage["window"], "test")
        self.assertGreater(stage["metrics"]["cohort"]["held_out_rows"], 0)


if __name__ == "__main__":
    unittest.main()
