"""Evaluation stage: ranking and rating metrics of one saved model on a held-out window.

Rules (P0.3): an item is relevant when its held-out rating is >= relevant_rating. The cohort is
the held-out users (all known in training after the split's projection) with at least one
relevant held-out item; users without one are counted, not scored. Each cohort user gets the
model's top-k over the training catalog, excluding the items they rated in the model's training
window. Precision@k = relevant returned items in the first k ranks / k, short lists included.
NDCG@k uses binary gains `1 / log2(rank + 1)` and the ideal DCG of min(k, relevant) items. Both
are macro-averaged over the cohort; missing ranks count zero. RMSE uses every held-out rating,
including those below the threshold, with predictions clipped to [1,5]; the raw RMSE, clipped
rows and missing or nonfinite predictions are reported too. Ranking metrics are also broken down
by the user's number of training items (`segments`), a diagnostic that leaves the headline
numbers unchanged. Recall@50 and @100 (relevant items in the top K / the user's relevant items,
macro-averaged) are secondary metrics from the same top-100 lists; k = 10 stays primary.

Random reference (not a system): every evaluation stores the exact expectation of a uniformly
random top-k over each user's unseen catalog (C' = catalog - seen items, r relevant items: each
rank is relevant with probability r/C', a user has a hit with probability
1 - prod_j (C'-r-j)/(C'-j)), and tests the model's users-with-a-hit count against it, one-sided,
with a Poisson approximation of the Poisson-binomial null.

Oracle popularity (reference, not a system): model label `oracle-popularity` ranks the catalog
by the number of relevant held-out ratings in the evaluated view itself, the best single global
order for that view. It uses the labels, so it is never used for decisions, and it is not a
bound for personalized models.

Views: `--horizon-days H` keeps only held-out rows within H days after the end of the fitted
window (the split's train cutoff), an additional view next to the full window. `--coverage`
evaluates a fallback (popularity or the oracle) on every held-out user with a relevant
in-catalog item, including users unknown to training: new users, and users removed by the
k-core, whose seen items come from the training window before the k-core. Coverage results are
a separately labeled serving view, never part of the model comparison (P0.5).

The test window is evaluated only on a refit split (`split --refit-from`), whose training
window is train + validation: on an ordinary split, items rated in validation would be neither
seen nor excluded, which is not the approved protocol (P0.5 option (b)).

Aggregation is distributed; only per-user metric values (one row per cohort user) reach the
driver, for user-level bootstrap intervals, so driver memory bounds the cohort size (359,147
users in the full test coverage view, a few hundred MB). Output: `<split>/evaluations/<view>/<model-label>/`
with `stage.json` and `per_user/` Parquet, completed by rename; `<view>` is the window plus
`-h<H>d` and/or `-coverage`.

Popularity window selection (approved 2026-10-25). The rule was written after exploratory
validation numbers for some windows had been seen, and before any test result: among the
evaluated popularity models of one split, `best` has the highest mean validation NDCG@k
(ties: the longer window). The chosen window is the longest window, at least as long as best's,
whose paired user-level bootstrap 95% interval of (NDCG_W - NDCG_best) contains 0, i.e. the
longest window not detectably worse than the best. "all" is the longest window.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import timed
from amazon_recommender.models import DAY_MS, Recommender, rank_global_order

BOOTSTRAP_RESAMPLES = 2000
K = 10
RECALL_AT = (50, 100)
WINDOWS = ("validation", "test")
SEGMENTS = (("<5", 0, 4), ("5-9", 5, 9), ("10-49", 10, 49), ("50+", 50, None))  # training items per user
USER_TYPES = ("known", "removed_by_k_core", "new")
ORACLE_LABEL = "oracle-popularity"


class OraclePopularity(Recommender):
    """Reference only: ranks by relevant held-out ratings of the evaluated view (it sees the labels)."""

    name = ORACLE_LABEL
    predicts_ratings = False
    contract = {
        "fitting_window": "the evaluated held-out view itself (uses the labels; reference only)",
        "candidate_universe": "items of the training catalog",
        "seen_item_policy": "items in the caller's seen pairs are never recommended",
        "cold_start": "every user gets the same global order",
        "ranking_score": "relevant held-out ratings per item; ties by ascending item_index",
        "approximate": False,
        "rating_prediction": "none",
    }

    def __init__(self, item_scores):
        super().__init__({})
        self.item_scores = item_scores
        self.fit_summary = {}

    def recommend(self, users, seen, k: int):
        from pyspark.sql import functions as F

        return rank_global_order(self.item_scores, users.select("user_index").withColumn("user_score", F.lit(0.0)),
                                 seen, k)


def oracle_item_scores(items, held_out, relevant_rating: float):
    from pyspark.sql import functions as F

    counts = held_out.filter(F.col("rating") >= relevant_rating).groupBy("item_index").agg(
        F.countDistinct("user_index").alias("item_score")
    )
    return items.select("item_index").join(counts, "item_index", "left").fillna(0, ["item_score"])


def ranking_per_user(recommendations, relevant, k: int, recall_at=()):
    """One row per cohort user: relevant, returned, hits, dcg, idcg, precision, ndcg, recall_at_K.

    `relevant` holds the cohort's relevant held-out (user_index, item_index) pairs; users with
    no recommendation row still appear, with zero hits. Only ranks <= k count for the k metrics;
    recall_at_K counts ranks <= K of the same lists.
    """
    from pyspark.sql import functions as F

    counts = relevant.groupBy("user_index").agg(F.count("*").alias("relevant"))
    gain = F.coalesce(F.col("_relevant"), F.lit(0))
    top = F.col("rank") <= k
    per_user = (
        recommendations.join(relevant.withColumn("_relevant", F.lit(1)), ["user_index", "item_index"], "left")
        .groupBy("user_index").agg(
            F.count("*").alias("returned_all"),
            F.count(F.when(top, 1)).alias("returned"),
            F.sum(F.when(top, gain).otherwise(0)).alias("hits"),
            F.sum(F.when(top, gain / F.log2(F.col("rank") + 1)).otherwise(0.0)).alias("dcg"),
            *(F.sum(F.when(F.col("rank") <= cut, gain).otherwise(0)).alias(f"hits_at_{cut}") for cut in recall_at),
        )
    )
    ideal = F.expr(f"aggregate(sequence(1, least(relevant, {int(k)})), 0D, (acc, r) -> acc + 1 / log2(r + 1))")
    frame = counts.join(per_user, "user_index", "left").select(
        "user_index",
        "relevant",
        F.coalesce("returned", F.lit(0)).alias("returned"),
        F.coalesce("returned_all", F.lit(0)).alias("returned_all"),
        F.coalesce("hits", F.lit(0)).alias("hits"),
        F.coalesce("dcg", F.lit(0.0)).alias("dcg"),
        ideal.alias("idcg"),
        *(F.coalesce(f"hits_at_{cut}", F.lit(0)).alias(f"hits_at_{cut}") for cut in recall_at),
    ).withColumn("precision", F.col("hits") / k).withColumn("ndcg", F.col("dcg") / F.col("idcg"))
    for cut in recall_at:
        frame = frame.withColumn(f"recall_at_{cut}", F.col(f"hits_at_{cut}") / F.col("relevant"))
    return frame


def with_random_expectation(per_user, k: int, recall_at=()):
    """Add the exact expected metrics of a uniformly random top-k over the user's `candidates` (C')."""
    from pyspark.sql import functions as F

    c, r = "candidates", "relevant"
    shown = f"least({int(k)}, {c})"
    per_user = per_user.withColumn("random_hit", F.expr(
        f"CASE WHEN {c} <= 0 THEN 0D ELSE 1 - aggregate(sequence(0, {shown} - 1), 1D, "
        f"(acc, j) -> acc * greatest({c} - {r} - j, 0) / ({c} - j)) END"
    )).withColumn("random_precision", F.expr(
        f"CASE WHEN {c} <= 0 THEN 0D ELSE {r} / {c} * {shown} / {int(k)} END"
    )).withColumn("random_ndcg", F.expr(
        f"CASE WHEN {c} <= 0 THEN 0D ELSE {r} / {c} * aggregate(sequence(1, {shown}), 0D, "
        f"(acc, x) -> acc + 1 / log2(x + 1)) / idcg END"
    ))
    for cut in recall_at:
        per_user = per_user.withColumn(f"random_recall_at_{cut}", F.expr(
            f"CASE WHEN {c} <= 0 THEN 0D ELSE least({int(cut)}, {c}) / {c} END"
        ))
    return per_user


def poisson_upper_tail_log10(observed: int, mean: float) -> float:
    """log10 P(X >= observed) for X ~ Poisson(mean); finite even where the probability underflows."""
    if observed <= 0:
        return 0.0
    if mean <= 0:
        return -math.inf
    log_pmf = lambda n: n * math.log(mean) - mean - math.lgamma(n + 1)  # noqa: E731
    if observed <= mean:
        below = sum(math.exp(log_pmf(n)) for n in range(observed))
        return math.log10(max(1.0 - below, 1e-300))
    # Upper terms decrease from `observed` on: sum them in log space until they stop mattering.
    total, n = log_pmf(observed), observed + 1
    while True:
        term = log_pmf(n)
        if term < total - 40:
            return total / math.log(10)
        total = max(total, term) + math.log1p(math.exp(-abs(total - term)))
        n += 1


def poisson_upper_tail(observed: int, mean: float) -> float:
    """P(X >= observed) for X ~ Poisson(mean)."""
    return 10 ** poisson_upper_tail_log10(observed, mean)


def rating_metrics(model, held_out) -> dict | None:
    from pyspark.sql import functions as F

    if not model.predicts_ratings:
        return None
    prediction = F.col("prediction")
    valid = prediction.isNotNull() & ~F.isnan(prediction)
    clipped = F.least(F.greatest(prediction, F.lit(1.0)), F.lit(5.0))
    row = model.predict(held_out.select("user_index", "item_index", "rating")).agg(
        F.count("*").alias("rows"),
        F.sum(valid.cast("long")).alias("predicted"),
        F.sum(F.when(valid, (clipped - F.col("rating")) ** 2)).alias("clipped_sse"),
        F.sum(F.when(valid, (prediction - F.col("rating")) ** 2)).alias("raw_sse"),
        F.sum((valid & ((prediction < 1) | (prediction > 5))).cast("long")).alias("clipped_rows"),
    ).first()
    predicted = row["predicted"] or 0
    return {
        "rows": row["rows"],
        "predicted": predicted,
        "missing_or_nonfinite": row["rows"] - predicted,
        "clipped_rows": row["clipped_rows"] or 0,
        "rmse": (row["clipped_sse"] / predicted) ** 0.5 if predicted else None,
        "rmse_raw": (row["raw_sse"] / predicted) ** 0.5 if predicted else None,
    }


def bootstrap_interval(values, *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES) -> list[float]:
    """Percentile 95% interval of the mean over users resampled with replacement."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(resamples)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _ranking_summary(rows, k: int, recall_at, *, seed: int | None = None) -> dict:
    """Macro averages of per-user rows; bootstrap intervals only when a seed is given."""
    users = len(rows)
    if not users:
        return {"users": 0}
    mean = lambda field: sum(getattr(row, field) for row in rows) / users  # noqa: E731
    summary = {
        "users": users,
        "precision_at_k": mean("precision"),
        "ndcg_at_k": mean("ndcg"),
        **{f"recall_at_{cut}": mean(f"recall_at_{cut}") for cut in recall_at},
        "hit_rate": sum(row.hits > 0 for row in rows) / users,
        "users_with_hit": sum(row.hits > 0 for row in rows),
        "expected_users_with_hit_if_random": sum(row.random_hit for row in rows),
        "short_lists": sum(row.returned < k for row in rows),
        **({f"short_lists_at_{max(recall_at)}": sum(row.returned_all < max(recall_at) for row in rows)}
           if recall_at else {}),
    }
    expected = summary["expected_users_with_hit_if_random"]
    summary["ratio_to_random"] = summary["users_with_hit"] / expected if expected else None
    if seed is not None:
        summary["ndcg_ci95"] = bootstrap_interval([row.ndcg for row in rows], seed=seed)
    return summary


def evaluate(model, seen, held_out, *, k: int, relevant_rating: float, seed: int, catalog_size: int,
             recall_at=RECALL_AT, user_types=None) -> tuple[dict, object]:
    """Metrics of one model on one view; returns (summary, per-user DataFrame).

    `seen` must hold only catalog items (the user's training items); `user_types`, if given,
    maps user_index to a `user_type` label for a breakdown.
    """
    from pyspark.sql import functions as F

    relevant = held_out.filter(F.col("rating") >= relevant_rating).select("user_index", "item_index").distinct()
    cohort = relevant.select("user_index").distinct()
    degrees = seen.groupBy("user_index").agg(F.count("*").alias("training_items"))
    recommendations = model.recommend(cohort, seen, max(k, *recall_at))
    per_user = (
        ranking_per_user(recommendations, relevant, k, recall_at)
        .join(degrees, "user_index", "left").fillna(0, ["training_items"])
        .withColumn("candidates", F.lit(catalog_size) - F.col("training_items"))
    )
    per_user = with_random_expectation(per_user, k, recall_at)
    if user_types is not None:
        per_user = per_user.join(user_types, "user_index", "left")
    per_user = per_user.cache()
    try:
        values = per_user.collect()
        held = held_out.agg(F.count("*").alias("rows"), F.countDistinct("user_index").alias("users")).first()
        users = len(values)
        if not users:
            raise ValueError("No held-out user has a relevant item; nothing to rank")
        ranking = _ranking_summary(values, k, recall_at)
        observed, expected = ranking["users_with_hit"], ranking["expected_users_with_hit_if_random"]
        summary = {
            "k": k,
            "relevant_rating": relevant_rating,
            "catalog_items": catalog_size,
            "cohort": {
                "held_out_rows": held["rows"],
                "held_out_users": held["users"],
                "eligible_users": users,
                "users_without_relevant": held["users"] - users,
                "relevant_pairs": relevant.count(),
            },
            "ranking": {
                "precision_at_k": ranking["precision_at_k"],
                "ndcg_at_k": ranking["ndcg_at_k"],
                **{f"recall_at_{cut}": ranking[f"recall_at_{cut}"] for cut in recall_at},
                "hit_rate": ranking["hit_rate"],
                "users_with_hit": observed,
                "hits": sum(row.hits for row in values),
                "short_lists": ranking["short_lists"],
                **({f"short_lists_at_{max(recall_at)}": ranking[f"short_lists_at_{max(recall_at)}"]}
                   if recall_at else {}),
                "precision_ci95": bootstrap_interval([row.precision for row in values], seed=seed),
                "ndcg_ci95": bootstrap_interval([row.ndcg for row in values], seed=seed),
                "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": seed, "unit": "user"},
            },
            "random_reference": {
                "note": "exact expectation of a uniformly random top-k over each user's unseen catalog; not a system",
                "precision_at_k": sum(row.random_precision for row in values) / users,
                "ndcg_at_k": sum(row.random_ndcg for row in values) / users,
                **{f"recall_at_{cut}": sum(getattr(row, f"random_recall_at_{cut}") for row in values) / users
                   for cut in recall_at},
                "expected_users_with_hit": expected,
            },
            "signal_vs_random": {
                "observed_users_with_hit": observed,
                "expected_users_with_hit": expected,
                "ratio": observed / expected if expected else None,
                "p_value_one_sided": poisson_upper_tail(observed, expected),
                "log10_p_value_one_sided": (lambda v: v if math.isfinite(v) else None)(
                    poisson_upper_tail_log10(observed, expected)),
                "test": "users with a hit vs the random null, Poisson approximation of the Poisson-binomial "
                        "(conservative for the upper tail: its variance sum p exceeds sum p(1-p))",
            },
            "segments": {label: _ranking_summary(
                [row for row in values if row.training_items >= low and (high is None or row.training_items <= high)],
                k, recall_at) for label, low, high in SEGMENTS},
            "rating": rating_metrics(model, held_out),
        }
        if user_types is not None:
            summary["user_types"] = {label: _ranking_summary(
                [row for row in values if row.user_type == label], k, recall_at, seed=seed) for label in USER_TYPES}
    except BaseException:
        per_user.unpersist()
        raise
    return summary, per_user


def view_name(window: str, horizon_days: int | None, coverage: bool) -> str:
    return window + (f"-h{horizon_days}d" if horizon_days else "") + ("-coverage" if coverage else "")


def coverage_frames(spark, run_directory: Path, split_stage: dict, split_directory: Path, window: str):
    """Held-out rows of every user (including users unknown to training) on catalog items.

    Users get a local index for this view, so unknown users can be ranked. Seen pairs come from
    the training window before the k-core. Returns (held_out, seen, user_types, counts).
    """
    from pyspark.sql import functions as F

    from amazon_recommender.split import dense_mapping

    sample_stage = json.loads((split_directory.parents[1] / "stage.json").read_text())
    raw = spark.read.parquet(str(run_directory / sample_stage["outputs"]["interactions"])).select(
        "user_id", "parent_asin", "rating", "timestamp"
    )
    cutoffs = split_stage["cutoffs_ms"]
    ts = F.col("timestamp")
    window_rows = raw.filter((ts > cutoffs["train"]) & (ts <= cutoffs["validation"])) if window == "validation" \
        else raw.filter(ts > cutoffs["validation"])
    items = spark.read.parquet(str(split_directory / split_stage["outputs"]["items"]))
    known = spark.read.parquet(str(split_directory / split_stage["outputs"]["users"]))
    pre_k_core = raw.filter(ts <= cutoffs["train"])
    in_catalog = window_rows.join(items, "parent_asin")
    users = dense_mapping(in_catalog, "user_id", "user_index")
    held_out = in_catalog.join(users, "user_id").select("user_id", "user_index", "item_index", "rating", "timestamp")
    seen = pre_k_core.join(users, "user_id").join(items, "parent_asin").select("user_index", "item_index").distinct()
    trained = pre_k_core.select("user_id").distinct().withColumn("_trained", F.lit(True))
    user_types = (
        users.join(known.select("user_id").withColumn("_known", F.lit(True)), "user_id", "left")
        .join(trained, "user_id", "left")
        .select("user_index", F.when(F.col("_known"), "known").when(F.col("_trained"), "removed_by_k_core")
                .otherwise("new").alias("user_type"))
    )
    counts = {
        "window_rows": window_rows.count(),
        "rows_on_catalog_items": held_out.count(),
    }
    return held_out.drop("user_id"), seen, user_types, counts


def run_evaluate(config: ProjectConfig, run_directory: Path, sample: str, split: str, model_label: str,
                 window: str, *, horizon_days: int | None = None, coverage: bool = False) -> Path:
    """Evaluate one trained model (or the oracle reference) of a split on one view. Starts Spark."""
    if window not in WINDOWS:
        raise ValueError(f"window must be one of: {', '.join(WINDOWS)}")
    if horizon_days is not None and (type(horizon_days) is not int or horizon_days <= 0):
        raise ValueError("horizon_days must be a positive integer")
    run_directory = Path(run_directory).resolve(strict=True)
    split_directory = run_directory / "samples" / sample / "splits" / split
    split_stage_path = split_directory / "stage.json"
    if not split_stage_path.is_file():
        raise FileNotFoundError(f"Split is not complete: {split_directory}")
    oracle = model_label == ORACLE_LABEL
    model_directory = split_directory / "models" / model_label
    if not oracle and not (model_directory / "stage.json").is_file():
        raise FileNotFoundError(f"Model is not trained: {model_directory}")
    split_stage = json.loads(split_stage_path.read_text())
    if window == "test" and not split_stage["policy"].get("refit_from"):
        raise ValueError("The test window is evaluated only on a refit split (split --refit-from), "
                         "so that validation items count as seen")
    if not split_stage["before_filtering"]["interactions"][window]:
        raise ValueError(f"The {window} window of {split} is empty")
    model_stage = None if oracle else json.loads((model_directory / "stage.json").read_text())
    if coverage and not oracle and model_stage["model"]["model"] != "popularity":
        raise ValueError("The coverage view evaluates the popularity fallback (or the oracle) only")
    view = view_name(window, horizon_days, coverage)
    final = split_directory / "evaluations" / view / model_label
    partial = final.with_name(f"{model_label}.partial")
    if final.exists():
        raise FileExistsError(f"Model already evaluated on this view: {final}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    experiment = split_stage_experiment(split_directory)

    from pyspark.sql import functions as F

    from amazon_recommender.models import load_model
    from amazon_recommender.spark.session import create_spark_session, describe_resources

    timings = {}
    started = time.monotonic()
    with timed(timings, "spark_start"):  # JVM launch and session; not data work
        spark = create_spark_session(config, app_name="amazon-recommender-evaluate")
    try:
        items = spark.read.parquet(str(split_directory / split_stage["outputs"]["items"]))
        coverage_counts, user_types = None, None
        if coverage:
            held_out, seen, user_types, coverage_counts = coverage_frames(
                spark, run_directory, split_stage, split_directory, window)
        else:
            seen = spark.read.parquet(str(split_directory / split_stage["outputs"]["train"]))
            held_out = spark.read.parquet(str(split_directory / split_stage["outputs"][window]))
        if horizon_days:
            held_out = held_out.filter(F.col("timestamp") <= split_stage["cutoffs_ms"]["train"] + horizon_days * DAY_MS)
        model = OraclePopularity(oracle_item_scores(items, held_out, experiment["relevant_rating"])) if oracle else \
            load_model(spark, model_directory / model_stage["outputs"]["model"], mappings=split_stage["mappings"])
        with timed(timings, "evaluate"):
            summary, per_user = evaluate(model, seen, held_out, k=K, relevant_rating=experiment["relevant_rating"],
                                         seed=config.seed, catalog_size=split_stage["mappings"]["items"]["count"],
                                         user_types=user_types)
        if coverage_counts is not None:
            summary["coverage"] = coverage_counts
        with timed(timings, "per_user_write"):
            per_user.write.mode("error").parquet(str(partial / "per_user"))
        per_user.unpersist()
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "evaluate",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": split_stage["run_id"],
            "sample": sample,
            "split": split,
            "window": window,
            "view": {"name": view, "horizon_days": horizon_days, "coverage": coverage,
                     "label": "serving coverage view, not a model comparison" if coverage else None},
            "purpose": split_stage["purpose"],
            "model_label": model_label,
            "model": model.name,
            "reference_only": oracle,
            "params": model.params,
            "model_stage_sha256": None if oracle else
            hashlib.sha256((model_directory / "stage.json").read_bytes()).hexdigest(),
            "approximate": model.contract["approximate"],
            "metrics": summary,
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            "outputs": {"per_user": "per_user"},
        }
    finally:
        spark.stop()
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final


def split_stage_experiment(split_directory: Path) -> dict:
    """The experiment settings of the sample a split was made from."""
    sample_stage = json.loads((split_directory.parents[1] / "stage.json").read_text())
    return sample_stage["config"]["experiment"]


def _window_order(window) -> float:
    return float("inf") if window == "all" else float(window)


def choose_window(candidates: dict, *, seed: int) -> dict:
    """Apply the predeclared rule. `candidates` maps window -> list of per-user NDCG, same user order."""
    import numpy as np

    means = {window: float(np.mean(values)) for window, values in candidates.items()}
    best = max(candidates, key=lambda w: (means[w], _window_order(w)))
    rng = np.random.default_rng(seed)
    n = len(candidates[best])
    draws = [rng.integers(0, n, n) for _ in range(BOOTSTRAP_RESAMPLES)]
    differences = {}
    for window, values in candidates.items():
        diff = np.asarray(values, dtype=float) - np.asarray(candidates[best], dtype=float)
        means_diff = [diff[d].mean() for d in draws]
        differences[window] = [float(np.percentile(means_diff, 2.5)), float(np.percentile(means_diff, 97.5))]
    eligible = [w for w in candidates if _window_order(w) >= _window_order(best) and differences[w][1] >= 0]
    chosen = max(eligible, key=_window_order)
    return {"best": best, "chosen": chosen, "mean_ndcg": means, "difference_vs_best_ci95": differences}


def _grid_key(params: dict) -> tuple:
    """Tie order, fixed before any ALS result: smaller rank, then larger regularization, then fewer iterations."""
    return (-params["rank"], params["regularization"], -params["iterations"])


def choose_grid(candidates: dict, params: dict, *, seed: int) -> dict:
    """Highest mean validation NDCG@k wins; ties by _grid_key. Paired bootstrap intervals vs the best are reported."""
    import numpy as np

    means = {label: float(np.mean(values)) for label, values in candidates.items()}
    best = max(candidates, key=lambda label: (means[label], _grid_key(params[label])))
    rng = np.random.default_rng(seed)
    n = len(candidates[best])
    draws = [rng.integers(0, n, n) for _ in range(BOOTSTRAP_RESAMPLES)]
    differences = {}
    for label, values in candidates.items():
        diff = np.asarray(values, dtype=float) - np.asarray(candidates[best], dtype=float)
        means_diff = [diff[d].mean() for d in draws]
        differences[label] = [float(np.percentile(means_diff, 2.5)), float(np.percentile(means_diff, 97.5))]
    return {"chosen": best, "mean_ndcg": means, "difference_vs_chosen_ci95": differences}


def run_select_grid(config: ProjectConfig, run_directory: Path, sample: str, split: str, model: str) -> Path:
    """Choose the ALS (or bias+ALS) settings from the split's validation evaluations. Starts Spark."""
    split_directory = Path(run_directory).resolve(strict=True) / "samples" / sample / "splits" / split
    evaluations = split_directory / "evaluations" / "validation"
    stages = {}
    for directory in sorted(evaluations.glob(f"{model}-rank*")):
        if (directory / "stage.json").is_file():
            stage = json.loads((directory / "stage.json").read_text())
            if stage["model"] == model:
                stages[stage["model_label"]] = (directory, stage)
    if not stages:
        raise FileNotFoundError(f"No evaluated {model} model in {evaluations}")
    output = split_directory / "selections" / f"{model}.json"
    if output.exists():
        raise FileExistsError(f"Selection already made: {output}")

    from amazon_recommender.spark.session import create_spark_session

    spark = create_spark_session(config, app_name="amazon-recommender-select")
    try:
        candidates, users = {}, None
        for label, (directory, _) in stages.items():
            rows = sorted(spark.read.parquet(str(directory / "per_user")).select("user_index", "ndcg").collect(),
                          key=lambda row: row.user_index)
            order = [row.user_index for row in rows]
            if users is not None and order != users:
                raise RuntimeError("Evaluations of the grid do not share the same user cohort")
            users = order
            candidates[label] = [row.ndcg for row in rows]
    finally:
        spark.stop()
    choice = choose_grid(candidates, {label: stage["params"] for label, (_, stage) in stages.items()}, seed=config.seed)
    result = {
        "stage": "select-grid",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "sample": sample,
        "model": model,
        "rule": "highest mean validation NDCG@10; ties: smaller rank, larger regularization, fewer iterations",
        "rule_written": "before any ALS result was seen",
        "users": len(users),
        **choice,
        "params": {label: stage["params"] for label, (_, stage) in stages.items()},
        "evaluations": {label: {
            "users_with_hit": stage["metrics"]["ranking"]["users_with_hit"],
            "ndcg_ci95": stage["metrics"]["ranking"]["ndcg_ci95"],
            "rmse": (stage["metrics"]["rating"] or {}).get("rmse"),
        } for label, (_, stage) in stages.items()},
    }
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return output


def run_select_popularity_window(config: ProjectConfig, run_directory: Path, sample: str, split: str) -> Path:
    """Choose the popularity window from the split's validation evaluations. Starts Spark."""
    split_directory = Path(run_directory).resolve(strict=True) / "samples" / sample / "splits" / split
    evaluations = split_directory / "evaluations" / "validation"
    stages = {}
    for directory in sorted(evaluations.glob("popularity-*")):
        if (directory / "stage.json").is_file():
            stage = json.loads((directory / "stage.json").read_text())
            stages[stage["params"]["window_days"]] = (directory, stage)
    if len(stages) < 2:
        raise FileNotFoundError(f"Need at least two evaluated popularity windows in {evaluations}")
    output = split_directory / "selections" / "popularity-window.json"
    if output.exists():
        raise FileExistsError(f"Selection already made: {output}")

    from amazon_recommender.spark.session import create_spark_session

    spark = create_spark_session(config, app_name="amazon-recommender-select")
    try:
        candidates, users = {}, None
        for window, (directory, stage) in stages.items():
            rows = spark.read.parquet(str(directory / "per_user")).select("user_index", "ndcg").collect()
            rows.sort(key=lambda row: row.user_index)
            order = [row.user_index for row in rows]
            if users is not None and order != users:
                raise RuntimeError("Evaluations of different windows do not share the same user cohort")
            users = order
            candidates[window] = [row.ndcg for row in rows]
    finally:
        spark.stop()
    choice = choose_window(candidates, seed=config.seed)
    result = {
        "stage": "select-popularity-window",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "sample": sample,
        "rule": "best = highest mean validation NDCG@10 (ties: longer window); chosen = longest window at least as "
                "long as best whose paired user bootstrap 95% CI of NDCG difference vs best contains 0",
        "rule_written": "after exploratory validation numbers for some windows were seen, before any test result",
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": config.seed, "unit": "user"},
        "users": len(users),
        **choice,
        "evaluations": {str(window): {
            "model_label": stage["model_label"],
            "users_with_hit": stage["metrics"]["ranking"]["users_with_hit"],
            "ndcg_ci95": stage["metrics"]["ranking"]["ndcg_ci95"],
        } for window, (_, stage) in stages.items()},
    }
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return output
