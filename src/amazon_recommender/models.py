"""Recommendation models behind one contract (P4.1), and the popularity and bias-only baselines (P4.2).

Models work on the split stage's dense indices. `fit` takes training rows with user_index,
item_index and rating. `predict` adds a raw `prediction` column to rows with user_index and
item_index; predictions are unclipped, since clipping to [1,5] belongs to evaluation.
`recommend` returns the top-k items per user as (user_index, item_index, rank, score), rank
from 1, score ties broken by ascending item_index, never returning an item of the `seen`
pairs. `seen` is an argument because evaluation and serving decide which history to exclude.

Each model declares its fitting window, candidate universe, seen-item policy, cold-start
behavior and scores in `contract`, and whether it predicts ratings in `predicts_ratings`.
Outputs stay index-based: mapping indices back to original IDs and labeling the
recommendation type (model or fallback) belong to recommendation materialization.
Fitting checkpoints its tables, so the Spark session needs a checkpoint directory (the
session factory sets one). A saved model is a directory with one Parquet table per fitted
table and `model.json` (name, parameters, contract, fit summary, and the provenance set by
the train stage: split stage SHA-256 and mapping digests), written last.
"""

import json
from pathlib import Path


class Recommender:
    """Common contract; subclasses set the class attributes and implement fit/predict/recommend."""

    name = ""
    table_names: tuple[str, ...] = ()
    predicts_ratings = True
    contract: dict = {}

    def __init__(self, params: dict):
        self.params = dict(params)
        self.fit_summary: dict | None = None  # None until fitted or loaded
        self.tables: dict = {}
        self.provenance: dict = {}  # split identity the indices belong to; set by the train stage

    @classmethod
    def params_from(cls, model_config, *, seed: int = 0) -> dict:
        raise NotImplementedError

    def fit(self, train) -> "Recommender":
        raise NotImplementedError

    def predict(self, pairs):
        raise NotImplementedError(f"{self.name} does not predict ratings")

    def recommend(self, users, seen, k: int):
        raise NotImplementedError

    def fitted(self) -> dict:
        if self.fit_summary is None:
            raise RuntimeError(f"The {self.name} model is not fitted")
        return self.fit_summary

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "params": self.params,
            "predicts_ratings": self.predicts_ratings,
            "contract": self.contract,
            "fit": self.fit_summary,
            "provenance": self.provenance,
        }

    def save(self, path: Path) -> None:
        self.fitted()
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Model directory already exists: {path}")
        for name in self.table_names:
            self.tables[name].write.mode("error").parquet(str(path / name))
        (path / "model.json").write_text(json.dumps(self.metadata(), indent=2, allow_nan=False) + "\n")


def _training_summary(ratings) -> dict:
    from pyspark.sql import functions as F

    row = ratings.agg(
        F.count("*").alias("interactions"),
        F.avg("rating").alias("global_mean"),
        F.countDistinct("user_index").alias("users"),
        F.countDistinct("item_index").alias("items"),
    ).first().asDict()
    if not row["interactions"]:
        raise ValueError("Cannot fit a model on an empty training window")
    return row


def rank_global_order(item_scores, users, seen, k: int):
    """Top-k unseen items per user when all users share one item order.

    `item_scores` has item_index and item_score (higher first, ties by ascending item_index).
    `users` has user_index (duplicates are dropped) and a user_score added to each of the user's
    item scores; it is constant within a user, so it does not change the order. A user who has seen
    n items needs at most the first k + n positions of the shared order, so candidates come
    from an equi-join on position: k + n rows per user instead of the whole catalog.
    Duplicate or out-of-catalog seen pairs only add candidates, so the result stays exact.
    """
    from pyspark.sql import Window, functions as F

    # Ordering the catalog moves it to one partition and is recomputed by the count and the join;
    # cheap at catalog size (tens to hundreds of thousands of items), so it is not cached.
    ordered = item_scores.select(
        "item_index",
        F.col("item_score").cast("double").alias("item_score"),
        (F.row_number().over(Window.orderBy(F.desc("item_score"), F.asc("item_index"))) - 1).alias("position"),
    )
    catalog = ordered.count()
    seen = seen.select("user_index", "item_index")
    seen_counts = seen.groupBy("user_index").agg(F.count("*").alias("_seen"))
    last = F.least(F.lit(k) + F.coalesce(F.col("_seen"), F.lit(0)), F.lit(catalog)) - 1
    candidates = (
        users.select("user_index", "user_score").dropDuplicates(["user_index"])
        .join(seen_counts, "user_index", "left")
        .withColumn("position", F.explode(F.sequence(F.lit(0), last.cast("int"))))
        .join(F.broadcast(ordered), "position")  # catalog-sized: at most a few hundred thousand rows
        .join(seen, ["user_index", "item_index"], "left_anti")
    )
    rank = F.row_number().over(Window.partitionBy("user_index").orderBy("position"))
    return candidates.withColumn("rank", rank).filter(F.col("rank") <= k).select(
        "user_index", "item_index", "rank", (F.col("item_score") + F.col("user_score")).alias("score")
    )


DAY_MS = 86_400_000


class PopularityModel(Recommender):
    """Ranks by recent training interaction count; predicts ratings with the smoothed item mean (P0.3).

    `window_days` counts only interactions in the last W days of the training data (timestamp >
    latest fitted timestamp - W days); "all" counts the whole training history, the original
    P0.3 rule. The window ends at the latest fitted timestamp, which equals the split's train
    cutoff (the validation cutoff on a refit split) unless the k-core removed every row at it. The window affects only the ranking: the rating adapter uses all training rows,
    and items with no interaction in the window stay in the catalog with count 0.
    """

    name = "popularity"
    table_names = ("items",)
    contract = {
        "fitting_window": "training interactions only; the ranking counts only the last window_days of them",
        "candidate_universe": "items of the fitted training window",
        "seen_item_policy": "items in the caller's seen pairs are never recommended",
        "cold_start": "unknown users get the global popularity order; unknown items are never ranked and "
                      "predict the training global mean",
        "ranking_score": "interaction count in the last window_days of training; ties by ascending item_index",
        "approximate": False,
        "rating_prediction": "(item_rating_sum + smoothing * global_mean) / (item_count + smoothing) over all "
                             "training rows, unclipped",
    }

    @classmethod
    def params_from(cls, model_config, *, seed: int = 0) -> dict:
        return {"smoothing": model_config.popularity_smoothing, "window_days": model_config.popularity_window_days}

    def fit(self, train) -> "PopularityModel":
        from pyspark.sql import functions as F

        ratings = train.select("user_index", "item_index", "rating", "timestamp")
        summary = _training_summary(ratings)
        window = self.params["window_days"]
        end = ratings.agg(F.max("timestamp")).first()[0]
        start = None if window == "all" else end - window * DAY_MS  # exclusive
        in_window = F.lit(True) if start is None else F.col("timestamp") > start
        items = ratings.groupBy("item_index").agg(
            F.count("*").alias("count"),
            F.sum("rating").alias("rating_sum"),
            F.sum(in_window.cast("long")).alias("window_count"),
        ).checkpoint()
        window_row = items.agg(F.sum("window_count"), F.sum((F.col("window_count") > 0).cast("long"))).first()
        self.tables = {"items": items}
        self.fit_summary = {
            **summary,
            "window_end_ms": end,
            "window_start_ms_exclusive": start,
            "window_interactions": window_row[0],
            "window_items": window_row[1],
        }
        return self

    def predict(self, pairs):
        from pyspark.sql import functions as F

        mean, alpha = self.fitted()["global_mean"], self.params["smoothing"]
        items = self.tables["items"]
        return pairs.join(items, "item_index", "left").select(
            *pairs.columns,
            ((F.coalesce("rating_sum", F.lit(0.0)) + alpha * mean) / (F.coalesce("count", F.lit(0)) + alpha))
            .alias("prediction"),
        )

    def recommend(self, users, seen, k: int):
        from pyspark.sql import functions as F

        self.fitted()
        item_scores = self.tables["items"].select("item_index", F.col("window_count").alias("item_score"))
        return rank_global_order(item_scores, users.select("user_index").withColumn("user_score", F.lit(0.0)), seen, k)


def _bias_update(residuals, other, other_key: str, other_bias: str, key: str, bias: str, regularization: float):
    """b_key = sum(residual - b_other) / (count + regularization) over the key's ratings."""
    from pyspark.sql import functions as F

    return residuals.join(other, other_key).groupBy(key).agg(
        (F.sum(F.col("residual") - F.col(other_bias)) / (F.count("*") + regularization)).alias(bias)
    )


def _max_change(old, new, key: str, bias: str) -> float:
    from pyspark.sql import functions as F

    return new.join(old.withColumnRenamed(bias, "_old"), key).agg(
        F.max(F.abs(F.col(bias) - F.col("_old")))
    ).first()[0]


class BiasModel(Recommender):
    """Regularized global mean + user bias + item bias, fitted by alternating updates (P0.4)."""

    name = "bias"
    table_names = ("users", "items")
    contract = {
        "fitting_window": "training interactions only",
        "candidate_universe": "items of the fitted training window",
        "seen_item_policy": "items in the caller's seen pairs are never recommended",
        "cold_start": "missing user or item biases are zero: unknown users are ranked by item bias alone; "
                      "unknown items are never ranked",
        "ranking_score": "global_mean + user_bias + item_bias, unclipped; ties by ascending item_index",
        "approximate": False,
        "rating_prediction": "global_mean + user_bias + item_bias, unclipped",
    }

    @classmethod
    def params_from(cls, model_config, *, seed: int = 0) -> dict:
        return {
            "regularization": model_config.bias_regularization,
            "max_passes": model_config.bias_max_passes,
            "tolerance": model_config.bias_tolerance,
        }

    def fit(self, train) -> "BiasModel":
        """Biases start at zero; each pass updates users, then items, from the training mean.

        Stops after a complete pass whose largest absolute bias change is below the tolerance,
        or after max_passes. Each pass's tables are checkpointed, which cuts the lineage.
        """
        from pyspark.sql import functions as F

        regularization = self.params["regularization"]
        ratings = train.select("user_index", "item_index", "rating").persist()
        try:
            summary = _training_summary(ratings)
            residuals = ratings.select(
                "user_index", "item_index", (F.col("rating") - F.lit(summary["global_mean"])).alias("residual")
            )
            users = ratings.select("user_index").distinct().withColumn("user_bias", F.lit(0.0))
            items = ratings.select("item_index").distinct().withColumn("item_bias", F.lit(0.0))
            changes = []
            for _ in range(self.params["max_passes"]):
                new_users = _bias_update(
                    residuals, items, "item_index", "item_bias", "user_index", "user_bias", regularization
                ).checkpoint()
                new_items = _bias_update(
                    residuals, new_users, "user_index", "user_bias", "item_index", "item_bias", regularization
                ).checkpoint()
                changes.append(max(_max_change(users, new_users, "user_index", "user_bias"),
                                   _max_change(items, new_items, "item_index", "item_bias")))
                users, items = new_users, new_items
                if changes[-1] < self.params["tolerance"]:
                    break
        finally:
            ratings.unpersist()
        self.tables = {"users": users, "items": items}
        self.fit_summary = {
            **summary,
            "passes": len(changes),
            "max_change_per_pass": changes,
            "converged": changes[-1] < self.params["tolerance"],
        }
        return self

    def _user_scores(self, users):
        """global_mean + user_bias per user; a missing user bias is zero."""
        from pyspark.sql import functions as F

        mean = self.fitted()["global_mean"]
        return users.join(self.tables["users"], "user_index", "left").select(
            "user_index", (F.lit(mean) + F.coalesce("user_bias", F.lit(0.0))).alias("user_score")
        )

    def predict(self, pairs):
        from pyspark.sql import functions as F

        mean = self.fitted()["global_mean"]
        return (
            pairs.join(self.tables["users"], "user_index", "left").join(self.tables["items"], "item_index", "left")
            .select(*pairs.columns, (F.lit(mean) + F.coalesce("user_bias", F.lit(0.0))
                                     + F.coalesce("item_bias", F.lit(0.0))).alias("prediction"))
        )

    def recommend(self, users, seen, k: int):
        from pyspark.sql import functions as F

        item_scores = self.tables["items"].select("item_index", F.col("item_bias").alias("item_score"))
        return rank_global_order(item_scores, self._user_scores(users.select("user_index")), seen, k)


def rank_by_factors(users, user_vectors, item_vectors, seen, k: int):
    """Exact top-k unseen items per user for score = user offset + dot(user vector, item vector).

    `user_vectors` has user_index, vector (array<double>) and offset; `item_vectors` has
    item_index and vector. The item matrix (catalog x rank) goes to the driver and is
    broadcast; each partition scores its users against the whole catalog with numpy, in
    blocks, masks the user's seen items, and keeps the k best (ties by ascending item_index).
    Users without a vector get no rows. This is the exact full-catalog ranking of P0.4.

    Memory: the driver collects the item matrix (catalog x rank doubles, plus Python Row objects
    while collecting), and every Python worker holds its own copy of the broadcast plus one
    block of block x catalog scores (64 x 175,550 doubles = 90 MB). The result is checkpointed
    locally and the broadcast released before returning.
    """
    from pyspark.sql import functions as F

    import numpy as np

    spark = users.sparkSession
    rows = item_vectors.select("item_index", "vector").orderBy("item_index").collect()
    ids = np.array([row.item_index for row in rows], dtype=np.int64)
    matrix = np.array([row.vector for row in rows], dtype=np.float64).reshape(len(rows), -1)
    shared = spark.sparkContext.broadcast((ids, matrix))
    seen_lists = seen.groupBy("user_index").agg(F.collect_list("item_index").alias("seen"))
    scored_users = (
        users.select("user_index").distinct().join(user_vectors.select("user_index", "vector", "offset"), "user_index")
        .join(seen_lists, "user_index", "left")
    )

    def top(partition, block=64):
        ids, matrix = shared.value
        batch = []
        for row in partition:
            batch.append(row)
            if len(batch) == block:
                yield from score(batch, ids, matrix)
                batch = []
        if batch:
            yield from score(batch, ids, matrix)

    def score(batch, ids, matrix):
        scores = np.array([row.vector for row in batch], dtype=np.float64) @ matrix.T
        for row, user_scores in zip(batch, scores):
            if row.seen:
                seen_items = np.asarray(row.seen, dtype=np.int64)
                positions = np.searchsorted(ids, seen_items)
                inside = positions < len(ids)
                positions, seen_items = positions[inside], seen_items[inside]
                user_scores[positions[ids[positions] == seen_items]] = -np.inf  # out-of-catalog seen items: ignored
            available = int(np.isfinite(user_scores).sum())
            count = min(k, available)
            if count == 0:
                continue
            threshold = user_scores[np.argpartition(-user_scores, count - 1)[:count]].min()
            pool = np.nonzero(user_scores >= threshold)[0]  # all ties at the threshold
            order = pool[np.lexsort((ids[pool], -user_scores[pool]))][:count]
            for rank, j in enumerate(order, 1):
                yield (int(row.user_index), int(ids[j]), rank, float(user_scores[j] + row.offset))

    ranked = spark.createDataFrame(scored_users.rdd.mapPartitions(top),
                                   "user_index int, item_index int, rank int, score double").localCheckpoint()
    shared.unpersist()
    return ranked


def _fit_als(ratings, rating_column: str, params: dict):
    """Explicit-feedback ALS (signed values allowed); returns checkpointed (users, items) factor tables."""
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import functions as F

    als = ALS(
        rank=params["rank"], maxIter=params["iterations"], regParam=params["regularization"], seed=params["seed"],
        userCol="user_index", itemCol="item_index", ratingCol=rating_column,
        implicitPrefs=False, nonnegative=False, coldStartStrategy="nan", checkpointInterval=5,
        # Models saved before blocks were configurable used Spark's default of 10.
        numUserBlocks=params.get("user_blocks", 10), numItemBlocks=params.get("item_blocks", 10),
    )
    fitted = als.fit(ratings)
    users = fitted.userFactors.select(F.col("id").alias("user_index"),
                                      F.col("features").cast("array<double>").alias("user_factors")).checkpoint()
    items = fitted.itemFactors.select(F.col("id").alias("item_index"),
                                      F.col("features").cast("array<double>").alias("item_factors")).checkpoint()
    return users, items


def _train_rmse(model, train) -> float:
    """RMSE of the fitted model on its own training ratings (unclipped), to expose overfitting."""
    from pyspark.sql import functions as F

    return model.predict(train.select("user_index", "item_index", "rating")).agg(
        F.sqrt(F.avg((F.col("prediction") - F.col("rating")) ** 2))
    ).first()[0]


DOT = "aggregate(zip_with(user_factors, item_factors, (a, b) -> a * b), 0D, (acc, x) -> acc + x)"


def _als_params(model_config, seed: int) -> dict:
    return {"rank": model_config.rank, "regularization": model_config.regularization,
            "iterations": model_config.iterations, "seed": seed,
            "user_blocks": model_config.num_user_blocks, "item_blocks": model_config.num_item_blocks}


class ALSModel(Recommender):
    """Plain explicit ALS on the original ratings (P4.3); score = dot(user factors, item factors)."""

    name = "als"
    table_names = ("users", "items")
    contract = {
        "fitting_window": "training interactions only",
        "candidate_universe": "items of the fitted training window",
        "seen_item_policy": "items in the caller's seen pairs are never recommended",
        "cold_start": "users or items without factors get no recommendations and no rating prediction (null)",
        "ranking_score": "dot(user factors, item factors), exact over the whole catalog; ties by ascending item_index",
        "approximate": False,
        "rating_prediction": "dot(user factors, item factors), unclipped; explicit feedback, nonnegative=False",
    }

    @classmethod
    def params_from(cls, model_config, *, seed: int = 0) -> dict:
        return _als_params(model_config, seed)

    def fit(self, train) -> "ALSModel":
        ratings = train.select("user_index", "item_index", "rating").persist()
        try:
            summary = _training_summary(ratings)
            users, items = _fit_als(ratings, "rating", self.params)
        finally:
            ratings.unpersist()
        self.tables = {"users": users, "items": items}
        self.fit_summary = {**summary, "implicit_prefs": False, "nonnegative": False,
                            "factor_users": users.count(), "factor_items": items.count(),
                            "input_partitions": train.rdd.getNumPartitions()}
        self.fit_summary["train_rmse"] = _train_rmse(self, train)
        return self

    def predict(self, pairs):
        from pyspark.sql import functions as F

        self.fitted()
        return (
            pairs.join(self.tables["users"], "user_index", "left").join(self.tables["items"], "item_index", "left")
            .select(*pairs.columns, F.expr(DOT).alias("prediction"))
        )

    def recommend(self, users, seen, k: int):
        from pyspark.sql import functions as F

        self.fitted()
        return rank_by_factors(
            users,
            self.tables["users"].select("user_index", F.col("user_factors").alias("vector"), F.lit(0.0).alias("offset")),
            self.tables["items"].select("item_index", F.col("item_factors").alias("vector")),
            seen, k,
        )


class BiasALSModel(Recommender):
    """Biases (as in BiasModel) plus explicit ALS on the signed residuals (P4.4).

    score = global_mean + user_bias + item_bias + dot(user factors, item factors). Ranking uses the
    full score exactly: the item bias is appended to the item vector and a 1 to the user vector,
    so the residual-only latent score never decides the order alone.
    """

    name = "bias-als"
    table_names = ("users", "items")
    contract = {
        "fitting_window": "training interactions only",
        "candidate_universe": "items of the fitted training window",
        "seen_item_policy": "items in the caller's seen pairs are never recommended",
        "cold_start": "users or items without factors get no recommendations and no rating prediction (null)",
        "ranking_score": "global_mean + user_bias + item_bias + dot(factors), unclipped, exact over the whole "
                         "catalog; ties by ascending item_index",
        "approximate": False,
        "rating_prediction": "global_mean + user_bias + item_bias + dot(factors), unclipped",
    }

    @classmethod
    def params_from(cls, model_config, *, seed: int = 0) -> dict:
        return {**_als_params(model_config, seed), **{f"bias_{key}": value for key, value in
                                                      BiasModel.params_from(model_config).items()}}

    def fit(self, train) -> "BiasALSModel":
        from pyspark.sql import functions as F

        bias = BiasModel({key[len("bias_"):]: value for key, value in self.params.items() if key.startswith("bias_")})
        bias.fit(train)
        mean = bias.fit_summary["global_mean"]
        residuals = (
            train.select("user_index", "item_index", "rating")
            .join(bias.tables["users"], "user_index").join(bias.tables["items"], "item_index")
            .select("user_index", "item_index",
                    (F.col("rating") - mean - F.col("user_bias") - F.col("item_bias")).alias("residual"))
            .persist()
        )
        try:
            users, items = _fit_als(residuals, "residual", self.params)
            residual_row = residuals.agg(F.avg("residual"), F.stddev("residual")).first()
        finally:
            residuals.unpersist()
        self.tables = {
            "users": bias.tables["users"].join(users, "user_index", "left").checkpoint(),
            "items": bias.tables["items"].join(items, "item_index", "left").checkpoint(),
        }
        self.fit_summary = {
            **bias.fit_summary,  # global mean, counts and the bias passes
            "residual_mean": residual_row[0],
            "residual_stddev": residual_row[1],
            "implicit_prefs": False,
            "nonnegative": False,
            "input_partitions": train.rdd.getNumPartitions(),
        }
        self.fit_summary["train_rmse"] = _train_rmse(self, train)
        return self

    def predict(self, pairs):
        from pyspark.sql import functions as F

        mean = self.fitted()["global_mean"]
        return (
            pairs.join(self.tables["users"], "user_index", "left").join(self.tables["items"], "item_index", "left")
            .select(*pairs.columns, (F.lit(mean) + F.col("user_bias") + F.col("item_bias") + F.expr(DOT))
                    .alias("prediction"))
        )

    def recommend(self, users, seen, k: int):
        from pyspark.sql import functions as F

        mean = self.fitted()["global_mean"]
        return rank_by_factors(
            users,
            self.tables["users"].filter(F.col("user_factors").isNotNull()).select(
                "user_index", F.concat("user_factors", F.array(F.lit(1.0))).alias("vector"),
                (F.lit(mean) + F.col("user_bias")).alias("offset")),
            self.tables["items"].filter(F.col("item_factors").isNotNull()).select(
                "item_index", F.concat("item_factors", F.array("item_bias")).alias("vector")),
            seen, k,
        )


MODELS = {model.name: model for model in (PopularityModel, BiasModel, ALSModel, BiasALSModel)}


def create_model(name: str, model_config, *, seed: int = 0) -> Recommender:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose one of: {', '.join(MODELS)}")
    return MODELS[name](MODELS[name].params_from(model_config, seed=seed))


def load_model(spark, path: Path, *, mappings: dict | None) -> Recommender:
    """Rebuild a saved model; its tables are read back from Parquet, not refitted.

    `mappings` (a split stage.json's `mappings`) is required so callers cannot forget the check:
    a model fitted on other indices is refused. Pass None only to skip it deliberately.
    """
    path = Path(path)
    metadata = json.loads((path / "model.json").read_text())
    if metadata["model"] not in MODELS:
        raise ValueError(f"Unknown saved model {metadata['model']!r} in {path}")
    if mappings is not None and metadata["provenance"].get("mappings") != mappings:
        raise ValueError(f"Model {path} was fitted on different user/item mappings")
    model = MODELS[metadata["model"]](metadata["params"])
    model.fit_summary = metadata["fit"]
    model.provenance = metadata["provenance"]
    model.tables = {name: spark.read.parquet(str(path / name)) for name in model.table_names}
    return model
