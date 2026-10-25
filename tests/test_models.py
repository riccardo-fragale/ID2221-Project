"""Model contract and baselines. Spark tests run only with RUN_SPARK_TESTS=1 (make check-spark).

Expected values are worked out by hand in the comments, not recomputed with the model formulas.
"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.cli import main
from amazon_recommender.config import load_config
from amazon_recommender.models import ALSModel, BiasALSModel, BiasModel, PopularityModel, create_model
from amazon_recommender.train import model_label, run_train


ROOT = Path(__file__).resolve().parents[1]
T0 = 1_577_836_800_000  # 2020-01-01 UTC
SPARK = os.environ.get("RUN_SPARK_TESTS") == "1"

# Popularity fixture: mean 18/5 = 3.6; counts i0 2 (sum 9), i1 2 (sum 7), i2 1 (sum 2).
POPULARITY_TRAIN = [(0, 0, 5.0), (0, 1, 3.0), (1, 0, 4.0), (1, 2, 2.0), (2, 1, 4.0)]
# Bias fixture: 3 users x 3 items, mean 21/6 = 3.5, residuals 1.5, 1.5, -0.5, -2.5, 0.5, -0.5.
BIAS_TRAIN = [(0, 0, 5.0), (0, 1, 5.0), (1, 0, 3.0), (1, 2, 1.0), (2, 1, 4.0), (2, 2, 3.0)]
# With regularization 1, by hand:
# pass 1 users (item biases 0): b0 = 3/3 = 1, b1 = -3/3 = -1, b2 = 0/3 = 0
# pass 1 items: i0 = ((1.5-1) + (-0.5+1))/3 = 1/3, i1 = ((1.5-1) + (0.5-0))/3 = 1/3,
#               i2 = ((-2.5+1) + (-0.5-0))/3 = -2/3; largest change 1
# pass 2 users: b0 = 2(1.5-1/3)/3 = 7/9, b1 = ((-0.5-1/3) + (-2.5+2/3))/3 = -8/9,
#               b2 = ((0.5-1/3) + (-0.5+2/3))/3 = 1/9
# pass 2 items: i0 = ((1.5-7/9) + (-0.5+8/9))/3 = 10/27, i1 = ((1.5-7/9) + (0.5-1/9))/3 = 10/27,
#               i2 = ((-2.5+8/9) + (-0.5-1/9))/3 = -20/27; largest change |7/9 - 1| = 2/9
TWO_PASS_USERS = {0: 7 / 9, 1: -8 / 9, 2: 1 / 9}
TWO_PASS_ITEMS = {0: 10 / 27, 1: 10 / 27, 2: -20 / 27}


class ModelContractTests(unittest.TestCase):
    def test_parameters_come_from_the_model_config(self):
        config = load_config(ROOT / "configs/local.toml").model
        self.assertEqual(create_model("popularity", config).params, {"smoothing": 10.0, "window_days": "all"})
        self.assertEqual(create_model("bias", config).params,
                         {"regularization": 10.0, "max_passes": 10, "tolerance": 0.0001})
        with self.assertRaises(ValueError):
            create_model("svd", config)

    def test_contracts_declare_every_policy(self):
        for model in (PopularityModel, BiasModel, ALSModel, BiasALSModel):
            self.assertEqual(set(model.contract), {
                "fitting_window", "candidate_universe", "seen_item_policy", "cold_start", "ranking_score",
                "approximate", "rating_prediction",
            })
            self.assertTrue(model.predicts_ratings)
            self.assertFalse(model.contract["approximate"])

    def test_als_parameters_include_the_seed_and_the_bias_settings(self):
        config = load_config(ROOT / "configs/local.toml").model
        self.assertEqual(create_model("als", config, seed=7).params,
                         {"rank": 10, "regularization": 0.1, "iterations": 10, "seed": 7,
                          "user_blocks": 10, "item_blocks": 10})
        self.assertEqual(create_model("bias-als", config, seed=7).params, {
            "rank": 10, "regularization": 0.1, "iterations": 10, "seed": 7, "user_blocks": 10, "item_blocks": 10,
            "bias_regularization": 10.0, "bias_max_passes": 10, "bias_tolerance": 0.0001,
        })

    def test_unfitted_model_refuses_to_save(self):
        with self.assertRaises(RuntimeError):
            PopularityModel({"smoothing": 10.0, "window_days": "all"}).save(Path("unused"))

    def test_label_names_model_and_parameters(self):
        self.assertEqual(model_label("popularity", {"smoothing": 10.0, "window_days": "all"}),
                         "popularity-smoothing10-window_daysall")
        self.assertEqual(model_label("popularity", {"smoothing": 10.0, "window_days": 90}),
                         "popularity-smoothing10-window_days90")
        self.assertEqual(model_label("bias", {"regularization": 10.0, "max_passes": 10, "tolerance": 0.0001}),
                         "bias-regularization10-max_passes10-tolerance0.0001")


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class ModelFunctionTests(unittest.TestCase):
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

    def ratings(self, rows):
        """(user, item, rating[, timestamp]) rows; the timestamp defaults to T0."""
        rows = [(*row, T0) if len(row) == 3 else row for row in rows]
        return self.spark.createDataFrame(rows, "user_index int, item_index int, rating double, timestamp long")

    def pairs(self, rows):
        return self.spark.createDataFrame(rows, "user_index int, item_index int")

    def users(self, indices):
        return self.spark.createDataFrame([(n,) for n in indices], "user_index int")

    def recommendations(self, model, train, users, k):
        rows = model.recommend(self.users(users), train, k).collect()
        lists = {}
        for row in sorted(rows, key=lambda r: (r.user_index, r.rank)):
            lists.setdefault(row.user_index, []).append((row.item_index, row.rank, row.score))
        return lists

    def predictions(self, model, pairs):
        return {(r.user_index, r.item_index): r.prediction for r in model.predict(self.pairs(pairs)).collect()}

    def assertLists(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for user, items in expected.items():
            self.assertEqual([(i, rank) for i, rank, _ in actual[user]], [(i, rank) for i, rank, _ in items])
            for (_, _, score), (_, _, wanted) in zip(actual[user], items):
                self.assertAlmostEqual(score, wanted, places=12)

    def test_popularity_counts_smoothed_predictions_and_ranking(self):
        train = self.ratings(POPULARITY_TRAIN)
        model = PopularityModel({"smoothing": 1.0, "window_days": "all"}).fit(train)
        self.assertEqual({r.item_index: (r["count"], r.rating_sum, r.window_count)
                          for r in model.tables["items"].collect()},
                         {0: (2, 9.0, 2), 1: (2, 7.0, 2), 2: (1, 2.0, 1)})
        self.assertAlmostEqual(model.fit_summary["global_mean"], 3.6)
        self.assertEqual((model.fit_summary["interactions"], model.fit_summary["users"],
                          model.fit_summary["items"]), (5, 3, 3))
        predictions = self.predictions(model, [(0, 0), (1, 1), (0, 2), (5, 9)])
        # (9 + 3.6)/3, (7 + 3.6)/3, (2 + 3.6)/2; unknown item: (0 + 3.6)/1.
        for pair, wanted in {(0, 0): 4.2, (1, 1): 10.6 / 3, (0, 2): 2.8, (5, 9): 3.6}.items():
            self.assertAlmostEqual(predictions[pair], wanted, places=12)
        # Order i0 (2), i1 (2, tie broken by index), i2 (1). u0 has one unseen item left: a short list.
        # Repeated users (as when taken from held-out rows) still get one list each.
        self.assertLists(self.recommendations(model, train, [0, 1, 2, 7, 7, 2], 2), {
            0: [(2, 1, 1.0)],
            1: [(1, 1, 2.0)],
            2: [(0, 1, 2.0), (2, 2, 1.0)],
            7: [(0, 1, 2.0), (1, 2, 2.0)],  # unknown user: global order
        })

    def test_popularity_window_counts_only_recent_training_rows(self):
        day = 86_400_000
        # i0: three ratings on day 0; i1: one on day 10; i2: two, on days 9.5 and 10 (the last training day).
        train = self.ratings([(0, 0, 5.0, T0), (1, 0, 5.0, T0), (2, 0, 5.0, T0), (3, 1, 1.0, T0 + 10 * day),
                              (4, 2, 3.0, T0 + 9 * day + day // 2), (5, 2, 3.0, T0 + 10 * day)])
        recent = PopularityModel({"smoothing": 1.0, "window_days": 1}).fit(train)
        # Window (day 9, day 10]: i0 0, i1 1, i2 2; the rating adapter still counts all rows.
        self.assertEqual({r.item_index: (r["count"], r.window_count) for r in recent.tables["items"].collect()},
                         {0: (3, 0), 1: (1, 1), 2: (2, 2)})
        self.assertEqual((recent.fit_summary["window_end_ms"], recent.fit_summary["window_start_ms_exclusive"],
                          recent.fit_summary["window_interactions"], recent.fit_summary["window_items"]),
                         (T0 + 10 * day, T0 + 9 * day, 3, 2))
        # Mean 22/6; i0 predicts (15 + 22/6) / 4 from all its rows although it is outside the window.
        self.assertAlmostEqual(self.predictions(recent, [(9, 0)])[(9, 0)], (15 + 22 / 6) / 4, places=12)
        self.assertLists(self.recommendations(recent, train, [9], 3),
                         {9: [(2, 1, 2.0), (1, 2, 1.0), (0, 3, 0.0)]})  # items outside the window stay last
        whole = PopularityModel({"smoothing": 1.0, "window_days": "all"}).fit(train)
        self.assertLists(self.recommendations(whole, train, [9], 3),
                         {9: [(0, 1, 3.0), (2, 2, 2.0), (1, 3, 1.0)]})

    def test_bias_two_passes_match_hand_computation(self):
        model = BiasModel({"regularization": 1.0, "max_passes": 2, "tolerance": 0.0001}).fit(self.ratings(BIAS_TRAIN))
        users = {r.user_index: r.user_bias for r in model.tables["users"].collect()}
        items = {r.item_index: r.item_bias for r in model.tables["items"].collect()}
        for actual, expected in ((users, TWO_PASS_USERS), (items, TWO_PASS_ITEMS)):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                self.assertAlmostEqual(actual[key], value, places=12)
        summary = model.fit_summary
        self.assertAlmostEqual(summary["global_mean"], 3.5)
        self.assertEqual((summary["passes"], summary["converged"]), (2, False))  # stopped by max_passes
        self.assertAlmostEqual(summary["max_change_per_pass"][0], 1.0, places=12)
        self.assertAlmostEqual(summary["max_change_per_pass"][1], 2 / 9, places=12)

    def test_bias_stops_after_the_first_pass_below_tolerance(self):
        model = BiasModel({"regularization": 1.0, "max_passes": 10, "tolerance": 1.5}).fit(self.ratings(BIAS_TRAIN))
        self.assertEqual((model.fit_summary["passes"], model.fit_summary["converged"]), (1, True))
        items = {r.item_index: r.item_bias for r in model.tables["items"].collect()}
        self.assertAlmostEqual(items[2], -2 / 3, places=12)

    def test_bias_predictions_default_missing_biases_to_zero_and_ranking(self):
        train = self.ratings(BIAS_TRAIN)
        model = BiasModel({"regularization": 1.0, "max_passes": 2, "tolerance": 0.0001}).fit(train)
        predictions = self.predictions(model, [(0, 2), (9, 0), (1, 9), (9, 9)])
        for pair, wanted in {(0, 2): 3.5 + 7 / 9 - 20 / 27, (9, 0): 3.5 + 10 / 27, (1, 9): 3.5 - 8 / 9,
                             (9, 9): 3.5}.items():
            self.assertAlmostEqual(predictions[pair], wanted, places=12)
        # Item order: i0 and i1 tie at 10/27 (index breaks it), then i2. Each user has seen two items.
        self.assertLists(self.recommendations(model, train, [0, 1, 2, 9, 9, 0], 2), {
            0: [(2, 1, 3.5 + 7 / 9 - 20 / 27)],
            1: [(1, 1, 3.5 - 8 / 9 + 10 / 27)],
            2: [(0, 1, 3.5 + 1 / 9 + 10 / 27)],
            9: [(0, 1, 3.5 + 10 / 27), (1, 2, 3.5 + 10 / 27)],
        })

    def test_shared_order_ranking_matches_exhaustive_ranking(self):
        from amazon_recommender.models import rank_global_order

        rng = random.Random(2221)
        scores = {item: float(rng.randint(0, 5)) for item in range(25)}  # many ties
        seen = sorted({(user, item) for user in range(40) for item in rng.sample(range(30), rng.randint(0, 24))})
        offsets = {user: rng.random() for user in range(42)}  # users 40, 41 have seen nothing
        item_scores = self.spark.createDataFrame(list(scores.items()), "item_index int, item_score double")
        users = self.spark.createDataFrame([*offsets.items(), *list(offsets.items())[:5]],  # 5 users repeated
                                           "user_index int, user_score double")
        for k in (1, 5, 30):
            actual = {(r.user_index, r.rank): (r.item_index, r.score)
                      for r in rank_global_order(item_scores, users, self.pairs(seen), k).collect()}
            expected = {}
            for user, offset in offsets.items():
                unseen = [item for item in sorted(scores, key=lambda i: (-scores[i], i)) if (user, item) not in seen]
                for rank, item in enumerate(unseen[:k], 1):
                    expected[(user, rank)] = (item, scores[item] + offset)
            self.assertEqual(actual, expected)

    def test_factor_ranking_hand_values(self):
        from amazon_recommender.models import rank_by_factors

        items = self.spark.createDataFrame([(0, [1.0, 0.0]), (1, [0.0, 1.0]), (2, [1.0, 1.0]), (3, [0.0, 0.0])],
                                           "item_index int, vector array<double>")
        vectors = self.spark.createDataFrame([(0, [2.0, 1.0], 0.5), (1, [1.0, 1.0], 0.0), (2, [1.0, 0.0], 0.0)],
                                             "user_index int, vector array<double>, offset double")
        # u0: dots 2, 1, 3, 0; i2 is seen (and item 99 is outside the catalog): i0 (2.5), i1 (1.5).
        # u1: dots 1, 1, 2, 0: i2, then the i0/i1 tie by index. u2 has seen three of four items: a
        # short list. u5 has no vector: no rows.
        seen = self.pairs([(0, 2), (0, 99), (2, 0), (2, 1), (2, 2)])
        rows = rank_by_factors(self.users([0, 1, 2, 5]), vectors, items, seen, 3).collect()
        lists = {}
        for row in sorted(rows, key=lambda r: (r.user_index, r.rank)):
            lists.setdefault(row.user_index, []).append((row.item_index, row.rank, row.score))
        self.assertEqual(lists, {
            0: [(0, 1, 2.5), (1, 2, 1.5), (3, 3, 0.5)],
            1: [(2, 1, 2.0), (0, 2, 1.0), (1, 3, 1.0)],
            2: [(3, 1, 0.0)],
        })

    def test_bias_als_ranks_by_the_full_score_not_the_residual(self):
        model = BiasALSModel({})
        model.fit_summary = {"global_mean": 3.0}
        model.tables = {
            "users": self.spark.createDataFrame([(0, 0.25, [1.0])], "user_index int, user_bias double, "
                                                                     "user_factors array<double>"),
            "items": self.spark.createDataFrame([(0, 2.0, [0.0]), (1, 0.0, [1.5])],
                                                "item_index int, item_bias double, item_factors array<double>"),
        }
        # Residual dots: i0 0, i1 1.5; full scores: i0 3 + 0.25 + 2 = 5.25, i1 3 + 0.25 + 1.5 = 4.75.
        self.assertLists(self.recommendations(model, self.pairs([]).limit(0), [0], 2),
                         {0: [(0, 1, 5.25), (1, 2, 4.75)]})
        predictions = self.predictions(model, [(0, 0), (0, 1), (0, 9)])
        self.assertEqual((predictions[(0, 0)], predictions[(0, 1)], predictions[(0, 9)]), (5.25, 4.75, None))

    def als_fixture(self):
        rng = random.Random(7)
        return self.ratings([(user, item, float(rng.randint(1, 5))) for user in range(30)
                             for item in rng.sample(range(15), 6)])

    def test_als_fits_explicit_ratings_and_round_trips(self):
        from amazon_recommender.models import load_model

        train = self.als_fixture()
        rows = [(r.user_index, r.item_index, r.rating) for r in train.collect()]
        mean = sum(r for _, _, r in rows) / len(rows)
        for model in (ALSModel({"rank": 4, "regularization": 0.05, "iterations": 10, "seed": 1}),
                      BiasALSModel({"rank": 4, "regularization": 0.05, "iterations": 10, "seed": 1,
                                    "bias_regularization": 1.0, "bias_max_passes": 5, "bias_tolerance": 1e-4})):
            with self.subTest(model=model.name):
                model.fit(train)
                self.assertFalse(model.fit_summary["implicit_prefs"])
                self.assertLess(model.fit_summary["train_rmse"], 5)
                predictions = self.predictions(model, [(u, i) for u, i, _ in rows])
                self.assertTrue(all(value is not None and math.isfinite(value) for value in predictions.values()))
                fitted = math.sqrt(sum((predictions[(u, i)] - r) ** 2 for u, i, r in rows) / len(rows))
                baseline = math.sqrt(sum((mean - r) ** 2 for _, _, r in rows) / len(rows))
                self.assertLess(fitted, baseline)  # training fit beats the global mean
                # Ranking scores equal the predictions of the same pairs, and seen items never appear.
                recommendations = self.recommendations(model, train, list(range(30)), 5)
                seen = {(u, i) for u, i, _ in rows}
                for user, items in recommendations.items():
                    unseen = self.predictions(model, [(user, item) for item in range(15) if (user, item) not in seen])
                    expected = sorted(unseen, key=lambda pair: (-unseen[pair], pair[1]))[:5]
                    self.assertEqual([item for item, _, _ in items], [item for _, item in expected])
                    for item, _, score in items:
                        self.assertAlmostEqual(score, unseen[(user, item)], places=9)
                path = Path(self.directory.name) / f"saved-{model.name}"
                model.provenance = {"mappings": {}}
                model.save(path)
                loaded = load_model(self.spark, path, mappings={})
                self.assertEqual(self.predictions(loaded, [(0, 0), (3, 7)]), self.predictions(model, [(0, 0), (3, 7)]))
                self.assertEqual(self.recommendations(loaded, train, [0, 1], 5), self.recommendations(model, train, [0, 1], 5))

    def test_bias_als_reuses_the_bias_fit_and_reconstructs_predictions(self):
        train = self.als_fixture()
        params = {"rank": 3, "regularization": 0.1, "iterations": 5, "seed": 1,
                  "bias_regularization": 1.0, "bias_max_passes": 5, "bias_tolerance": 1e-4}
        model = BiasALSModel(params).fit(train)
        bias = BiasModel({"regularization": 1.0, "max_passes": 5, "tolerance": 1e-4}).fit(train)
        users = {r.user_index: r for r in model.tables["users"].collect()}
        expected = {r.user_index: r.user_bias for r in bias.tables["users"].collect()}
        for user, value in expected.items():
            self.assertAlmostEqual(users[user].user_bias, value, places=9)
        items = {r.item_index: r for r in model.tables["items"].collect()}
        mean = model.fit_summary["global_mean"]
        prediction = self.predictions(model, [(2, 4)])[(2, 4)]
        dot = sum(a * b for a, b in zip(users[2].user_factors, items[4].item_factors))
        self.assertAlmostEqual(prediction, mean + users[2].user_bias + items[4].item_bias + dot, places=9)

    def test_save_and_load_round_trip_and_mapping_check(self):
        from amazon_recommender.models import load_model

        train = self.ratings(BIAS_TRAIN)
        mappings = {"users": {"count": 3, "max_index": 2, "digest": "3:1"},
                    "items": {"count": 3, "max_index": 2, "digest": "3:2"}}
        pairs = [(0, 2), (9, 0), (2, 1)]
        for model in (PopularityModel({"smoothing": 1.0, "window_days": "all"}),
                      BiasModel({"regularization": 1.0, "max_passes": 2, "tolerance": 0.0001})):
            with self.subTest(model=model.name):
                model.fit(train)
                model.provenance = {"mappings": mappings}
                path = Path(self.directory.name) / model.name
                model.save(path)
                with self.assertRaises(FileExistsError):
                    model.save(path)
                loaded = load_model(self.spark, path, mappings=mappings)
                self.assertEqual(loaded.metadata(), json.loads(json.dumps(model.metadata())))
                self.assertEqual(self.predictions(loaded, pairs), self.predictions(model, pairs))
                self.assertEqual(self.recommendations(loaded, train, [0, 1, 2, 9], 2),
                                 self.recommendations(model, train, [0, 1, 2, 9], 2))
                other = {**mappings, "items": {**mappings["items"], "digest": "3:9"}}
                with self.assertRaises(ValueError):
                    load_model(self.spark, path, mappings=other)


class TrainRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        self.config = replace(base, storage_root=self.root / "storage", experiment=replace(
            base.experiment, user_fraction=1.0, user_min_reviews=2, item_min_reviews=2
        ))
        reviews, metadata = self.root / "reviews.ndjson", self.root / "metadata.ndjson"
        # 40 users; user n rates n % 5 + 1 of 12 items, at increasing times (as in test_split).
        reviews.write_text("".join(
            json.dumps({"user_id": f"u{n:02d}", "parent_asin": f"i{(n + k) % 12:02d}", "rating": 1.0 + (n + k) % 5,
                        "timestamp": T0 + 1000 * n + k}) + "\n"
            for n in range(40) for k in range(n % 5 + 1)
        ))
        metadata.write_text(json.dumps({"parent_asin": "i00", "title": "Item"}) + "\n")
        self.run_directory = create_run(self.config, reviews=reviews, metadata=metadata)
        self.sample, self.split = "fraction-1-seed-2221", "p0.8-0.1-0.1-k2-2"


class TrainPreconditionTests(TrainRunTests):
    def test_requires_completed_split(self):
        with self.assertRaises(FileNotFoundError):
            run_train(self.config, self.run_directory, self.sample, self.split, "bias")

    def test_completed_model_is_not_overwritten(self):
        split = self.run_directory / "samples" / self.sample / "splits" / self.split
        (split / "models" / "popularity-smoothing10-window_daysall").mkdir(parents=True)
        (split / "stage.json").write_text(json.dumps({"run_id": "r", "mappings": {}, "outputs": {}}))
        with self.assertRaises(FileExistsError):
            run_train(self.config, self.run_directory, self.sample, self.split, "popularity")

    def test_cli_rejects_unknown_model(self):
        arguments = ["train", "--config", str(ROOT / "configs/local.toml"), "--run", str(self.run_directory),
                     "--sample", self.sample, "--split", self.split, "--model", "svd"]
        with self.assertRaises(SystemExit):
            main(arguments)


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class SparkTrainTests(TrainRunTests):
    def test_train_end_to_end(self):
        from pyspark.sql import SparkSession

        from amazon_recommender.etl import run_etl
        from amazon_recommender.models import load_model
        from amazon_recommender.sample import run_sample
        from amazon_recommender.split import run_split

        run_etl(self.config, self.run_directory)
        run_sample(self.config, self.run_directory)
        split = run_split(self.config, self.run_directory, self.sample)
        split_stage = json.loads((split / "stage.json").read_text())
        outputs = {name: run_train(self.config, self.run_directory, self.sample, split.name, name)
                   for name in ("popularity", "bias")}
        checkpoints = self.config.storage_root / "checkpoints"
        self.assertFalse([path for path in checkpoints.rglob("*") if path.is_file()])

        spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
        try:
            train = spark.read.parquet(str(split / "train"))
            for name, output in outputs.items():
                stage = json.loads((output / "stage.json").read_text())
                self.assertEqual(output.name, model_label(name, stage["model"]["params"]))
                self.assertEqual((stage["status"], stage["sample"], stage["split"]), ("completed", self.sample,
                                                                                       split.name))
                self.assertEqual(stage["model"]["fit"]["interactions"], split_stage["interactions"]["train"])
                self.assertEqual(stage["model"]["provenance"]["mappings"], split_stage["mappings"])
                model = load_model(spark, output / stage["outputs"]["model"], mappings=split_stage["mappings"])
                users = train.select("user_index").distinct()
                recommendations = model.recommend(users, train, 3)
                self.assertEqual(recommendations.join(train, ["user_index", "item_index"]).count(), 0)
                self.assertEqual(model.predict(train).filter("prediction is null").count(), 0)
            bias = json.loads((outputs["bias"] / "stage.json").read_text())["model"]["fit"]
            self.assertEqual(bias["converged"], bias["max_change_per_pass"][-1] < 0.0001)
        finally:
            spark.stop()


if __name__ == "__main__":
    unittest.main()
