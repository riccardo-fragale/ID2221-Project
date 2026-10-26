# ID2221-Project

Distributed explicit-feedback ALS on Amazon Reviews 2023 (Movies and TV), for
ID2221, Autumn 2026. The proposal specifies yearly Parquet storage, PySpark/Spark
SQL, popularity/bias-only/plain ALS/bias+ALS comparisons, temporal evaluation,
fixed-resource input scaling, MongoDB top-20 serving, and a Streamlit dashboard.

The development foundation provides an installed Python package and lockfile,
validated experiment settings, input/output schemas, run configuration/input
snapshots, a non-starting environment check, an explicit Spark-session factory,
a MongoDB Compose setup, and foundation tests. ETL, sampling, the split, the four systems
(popularity, bias-only, ALS, bias+ALS), evaluation and the scalability benchmark runner are
implemented; the quality and scalability results are below. Serving lists for every user and
their MongoDB publication (P7) are implemented; the dashboard is still to be implemented.

## ETL stage

`amazon-recommender etl --config configs/local.toml --run storage/runs/<run-id>`
starts Spark and processes the inputs recorded by `init-run`. It refuses to run
if the configuration digest differs from the run snapshot, if an input's size
changed (checksums are not recomputed), or if `etl/` already exists.

- Reviews are read with the four-column schema; malformed JSON lines and rows with
  missing user/item/rating/timestamp, rating outside [1,5], or a timestamp outside
  1990-2030 (Unix milliseconds; seconds are rejected) are dropped and counted by reason.
- Per original (user, item) pair the latest review is kept; ties on timestamp keep the
  highest rating. Output: `<run>/etl/interactions/year=YYYY/` (UTC year, Parquet).
- Metadata is reduced to one title per `parent_asin` (blank titles count as missing, so the
  item falls back to its ID): `<run>/etl/metadata/`.
- `<run>/etl/stage.json` records read/rejected/duplicate/output counts, users, items,
  year range, items without a title, the Spark resources used, and wall-clock seconds per
  step (around completed Spark actions). The stage is written to `etl.partial/` and
  renamed when complete, so an existing `etl/` is a finished stage.
- `malformed_record` also counts lines whose fields have the wrong JSON type (for example
  a rating given as a string), since Spark cannot parse them with the schema.
- Identifiers are stored as read, without trimming.
- Dense integer mappings are not created here: the approved protocol derives them from
  the training window after k-core filtering (split stage).

## Sampling stage

`amazon-recommender sample --config <config> --run storage/runs/<run-id> [--user-fraction F]`
requires a completed ETL stage and starts Spark. It keeps a user when
`pmod(xxhash64(cast(seed as bigint), user_id), 2^32) / 2^32 < user_fraction`, with the user's
full history. The choice depends only on the user ID and seed: the same seed gives the same
users on the same data, no shuffle is needed to select, and with one seed a smaller fraction
keeps a subset of a larger one's users (nested 25/50/100% cohorts). The fraction is approximate
(about `user_fraction` of users, not exactly), and a different seed gives a different sample.

One ETL serves several samples: each goes to `<run>/samples/fraction-<F>-seed-<seed>/`, and
the sampling configuration (with `--user-fraction` overriding `experiment.user_fraction`) may
differ from the run snapshot. Its `stage.json` records that configuration and digest, the
SHA-256 of the consumed `etl/stage.json`, the method, the input identities, user and
interaction counts before/after, the achieved fraction, resources, and timings. With
fraction 1 nothing is copied and `stage.json` points at `etl/interactions`. Later stages must
read the paths listed in `outputs` (relative to the run directory). Sampling happens after
ETL, so ingestion is timed on the full input only; cohort timings start at the sample.
Sampling happens before the split and k-core, as in the proposal. The sample is a
development choice; results from different fractions are exploratory and do not show
model improvements by themselves.

## Split stage (temporal split, k-core, mappings)

`amazon-recommender split --config <config> --run storage/runs/<run-id> --sample <label> [--cutoffs-from <stage.json>]`
requires a completed sample and starts Spark. The experiment settings (split proportions,
`user_min_reviews`, `item_min_reviews`) come from the sample's `stage.json` config; the
`--config` file only supplies Spark resources. Output: `<run>/samples/<label>/splits/<split-label>/`
with `train/`, `validation/`, `test/`, `users/`, `items/` and `stage.json`, completed by rename.

- One global chronological split by interaction count. With N interactions, the train cutoff
  is the smallest timestamp whose cumulative count reaches `split[0] * N` (validation:
  `(split[0] + split[1]) * N`). Train is `timestamp <= train cutoff`, so a group of equal
  timestamps stays in one window, the earlier one; actual counts and proportions are recorded.
- `--cutoffs-from` copies both cutoffs from another split's `stage.json` (label
  `cutoffs-<sha>-k<u>-<i>`), to freeze common boundaries across cohorts.
- Iterative k-core on training only, until every user and item reaches its threshold; each
  pass is checkpointed and its removed users/items/interactions are recorded.
  `converged_after` counts removal passes; the final check, which removes nothing, is not
  counted. The session's checkpoint directory is deleted when the stage ends.
- Validation and test keep rows whose user and item both survive in training. Drops are
  counted as unknown user only, unknown item only, or both. Distinct held-out users and items
  are also split into those never seen in training (cold starts) and those removed by the
  k-core. `proportions` gives the train/validation/test shares after filtering.
- Dense 0-based `user_index`/`item_index` in ascending order of the original ID, independent
  of partitioning; `stage.json` records count, maximum index and a digest per mapping. All
  three windows carry both indices.

`make check` skips the Spark tests; `make check-spark` runs the ETL, sampling, split, model and evaluation ones and needs Java 17.
They passed on 2026-10-25 on synthetic data on the Linux machine (Ubuntu 24.04), which uses
OpenJDK 17.0.20.1 selected through `JAVA_HOME`.

## Train stage (model contract, popularity and bias-only baselines)

`amazon-recommender train --config <config> --run storage/runs/<run-id> --sample <label> --split <label> --model popularity|bias`
requires a completed split and starts Spark. Hyperparameters come from the `--config` file's
`[model]` section; the split is not changed. Output: `<split>/models/<model-label>/` with the
saved model (`model/`) and `stage.json`, completed by rename. The label names the model and its
parameters (e.g. `bias-regularization10-max_passes10-tolerance0.0001`).

All models in `models.py` share one contract on the split's dense indices: `fit(train)`,
`predict(pairs)` (adds a raw, unclipped `prediction`; clipping to [1,5] belongs to evaluation),
`recommend(users, seen, k)` returning `(user_index, item_index, rank, score)`, `save`, and
`load_model`. Each model declares its fitting window, candidate universe, seen-item policy,
cold-start behavior, ranking score, rating predictor and whether it is approximate in `contract`.
The seen pairs are an argument, so evaluation and serving choose the history to exclude. Outputs
stay index-based: mapping back to original IDs and labeling model or fallback lists belong to
recommendation materialization. A saved model records the split `stage.json` SHA-256 and the
mapping digests; `load_model(..., mappings=...)` refuses a model fitted on other mappings.

- Popularity ranks by the interaction count in the last `popularity_window_days` of the fitted
  data (after the latest fitted timestamp minus W days; `"all"` is the original whole-history
  rule) and predicts ratings separately with `(item_rating_sum + alpha * mean) / (item_count + alpha)`
  over all training rows; an unknown item predicts the mean. Items without interactions in the
  window stay in the catalog with count 0. `train --popularity-window-days` overrides the config.
- Bias-only follows P0.4: training mean, zero start, user update then item update per pass,
  regularization 10 for both, stop after the first complete pass whose largest absolute change
  is below 0.0001, or after 10 passes. Each pass is checkpointed. Missing biases are zero.
  `fit` in `stage.json` records the passes, the largest change per pass and `converged`.
- For both, every user shares one item order (score descending, ties by ascending
  `item_index`, i.e. ascending original ID). A user who has seen n items needs at most the first
  k + n positions, so ranking joins on position instead of crossing users with the catalog, and
  is exact. Lists are shorter than k only when the catalog runs out of unseen items. Bias-only
  scores are the full prediction `mean + user_bias + item_bias`.

Measured on 2026-10-25 on the 20% development split (`p0.8-0.1-0.1-k5-5`, `configs/real.toml`,
warm cache): both models fitted on 912,329 interactions (mean rating 4.296). Popularity took
5.6 s in total, bias-only 14.1 s and converged after 9 passes (last change 5.2e-5). Top-20 lists
for all 88,251 training users (1,765,020 rows, none already seen) took 1.8 s (popularity) and
5.0 s (bias-only) in a separate timing script. These are single development measurements, not
quality or scalability results.

## Popularity window, test-time refit and evaluation (approved 2026-10-25)

The user approved two protocol changes after the first validation results, in which
whole-history popularity reached NDCG@10 of about 0.00005 (2 of 6,123 users with a hit): the
all-time top items were reviewed in 2015-2016, while 94% of validation rows are on items beyond
popularity rank 1000.

- P0.3 change: popularity has a window hyperparameter W in {30, 90, 180, 365, all} days, chosen
  on the development validation window like the ALS settings. `all` stays as the reported
  whole-history row. Seen-item exclusion and the rating adapter use the full training history.
- P0.5 change, option (b): test-time models are fitted on train + validation, with every
  hyperparameter frozen from development. `split --refit-from <split stage.json>` makes that
  split: both cutoffs become the source's validation cutoff, k-core and mappings are recomputed on
  train + validation, validation is empty and the test window is unchanged (label
  `refit-<sha>-k<u>-<i>`). Fitting then ends where the evaluated window starts, for validation
  and test alike.

`evaluate --run R --sample S --split P --model-label M --window validation|test` writes
`<split>/evaluations/<window>/<model-label>/` with `stage.json` and per-user Parquet, following
P0.3: relevance rating >= 4, cohort = held-out users with at least one relevant item (the others
are counted), top-10 over the training catalog excluding the user's training items, Precision@10
over 10 including short lists, binary NDCG@10, macro averages, user-level bootstrap 95% intervals
(2000 resamples, config seed), and RMSE on all held-out ratings with predictions clipped to [1,5]
plus raw RMSE, clipped rows and missing or nonfinite predictions. Ranking metrics are also broken
down by the user's number of training items (<5, 5-9, 10-49, 50+) as a diagnostic. The test window
is evaluated only on a refit split, so items rated in validation count as seen.

Reference rows and additional views (approved 2026-10-25, after the baseline test results above
had been seen):

- Random reference: every evaluation stores the exact expectation of a uniformly random top-10
  over each user's unseen catalog (P@10, NDCG@10, Recall@50/100, expected users with a hit) and
  tests the observed users-with-a-hit count against it (one-sided, Poisson approximation, with
  `log10_p_value_one_sided` where the p-value underflows). It is not a system.
- Oracle popularity: `--model-label oracle-popularity` ranks the catalog by the relevant ratings
  of the evaluated view itself. It is the best single global order for that view, uses the labels
  and is never used for decisions; personalized models can exceed it.
- Recall@50 and Recall@100 (relevant items in the top K / the user's relevant items) are
  secondary metrics from the same lists; P@10 and NDCG@10 stay primary.
- `--horizon-days 180`: only held-out rows within 180 days of the end of the fitted window, as an
  additional view next to the full window, motivated by periodic retraining. It changes the
  cohort (users active in those days), so its counts are reported separately.
- `--coverage`: the popularity fallback (or the oracle) on every held-out user with a relevant
  item in the catalog, including users unknown to training. Users removed by the k-core keep
  their pre-k-core training items as seen items; new users have none. `user_types` breaks the
  results down into known, removed_by_k_core and new users. It is a serving view, never part of
  the model comparison (P0.5).

Views are stored as `<split>/evaluations/<window>[-h<H>d][-coverage]/<model-label>/`.

Structural ceiling first: of the 1,249,679 relevant test ratings before any projection, 53.1%
are on items outside the k-core catalog (41.2% on items never rated before the refit cutoff),
so at most about 47% are reachable by any model fitted at the cutoff.

Measured on 2026-10-25 (same splits as above; hits = users with at least one relevant item in
their top 10; expected = random reference; p is the one-sided test of the hit count against
random, Poisson approximation, conservative for the upper tail; exact values are in each
`stage.json`, including `log10_p_value_one_sided`):

| View | System | Users | NDCG@10 [95% CI] | Hit rate | Recall@100 | Hits / expected | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dev validation | Popularity W=180 | 6,123 | 0.0078 [0.0061, 0.0095] | 2.04% | 0.057 | 125 / 3.2 | < 1e-10 |
| Dev validation | Popularity W=all | 6,123 | 0.00005 [0, 0.0001] | 0.03% | 0.005 | 2 / 3.2 | 0.83 |
| Dev validation | Bias-only | 6,123 | 0.0002 [0, 0.0003] | 0.07% | 0.004 | 4 / 3.2 | 0.40 |
| Dev validation | Random (reference) | 6,123 | 0.00012 | | 0.002 | | |
| Dev validation | Oracle popularity (reference) | 6,123 | 0.0122 [0.0104, 0.0140] | 3.77% | 0.100 | 231 / 3.2 | |
| Test (refit) | Popularity W=180 | 44,612 | 0.0034 [0.0030, 0.0038] | 0.93% | 0.028 | 415 / 7.1 | < 1e-10 |
| Test (refit) | Popularity W=all | 44,612 | 0.00004 [0, 0.0001] | 0.02% | 0.002 | 7 / 7.1 | 0.56 |
| Test (refit) | Bias-only | 44,612 | 0.00007 [0, 0.0001] | 0.04% | 0.002 | 17 / 7.1 | 0.001 |
| Test (refit) | Random (reference) | 44,612 | 0.000034 | | 0.0006 | | |
| Test (refit) | Oracle popularity (reference) | 44,612 | 0.0090 [0.0084, 0.0096] | 2.53% | 0.055 | 1,128 / 7.1 | |
| Test, 180-day horizon | Popularity W=180 | 21,613 | 0.0054 [0.0048, 0.0061] | 1.40% | 0.042 | 302 / 2.6 | < 1e-10 |
| Test, 180-day horizon | Bias-only | 21,613 | 0 | 0% | 0.002 | 0 / 2.6 | 1 |
| Test, 180-day horizon | Oracle popularity (reference) | 21,613 | 0.0158 [0.0146, 0.0171] | 3.55% | 0.084 | 767 / 2.6 | |
| Test coverage (serving) | Popularity W=180 | 359,147 | 0.0073 [0.0071, 0.0075] | 1.68% | 0.042 | 6,031 / 33.4 | < 1e-10 |
| Test coverage (serving) | Popularity W=all | 359,147 | 0.00004 [0, 0.0001] | 0.01% | 0.002 | 47 / 33.4 | 0.015 |

About 20 system/view p-values are reported, so they are descriptive: signal is claimed only
where it survives a Bonferroni threshold of about 0.0025. Bias-only with 0 hits for 2.6
expected on the 180-day horizon is not above random (P(X = 0) = 0.07), not below it, and
whole-history popularity at p = 0.015 on the coverage view is not evidence of signal.
Recent popularity is far above random in every view (39 to 180 times the random hit count);
whole-history popularity is at random level, and bias-only ranking is barely above it on the
test window (17 hits for 7.1 expected, p = 0.001) and not on validation. In absolute terms 98-99%
of users still get no relevant item in their top 10. The test coverage view covers 359,147 users:
44,612 known (NDCG@10 0.0034, 415 hits, matching the model comparison), 87,178 removed by the
k-core (0.0063, 1,298 hits) and 227,357 new users (0.0085, 4,318 hits). It keeps 750,953 of
1,715,851 test rows; the rest are on items outside the catalog.

Why the absolute numbers are small, measured on the full test window (relevant ratings before
projection, 1,249,679): 41.2% are on items never rated before the refit cutoff, which no model
fitted before that date can recommend, and 53.1% are on items outside the k-core catalog. Among
the ratings on catalog items, 81.3% are on items first rated more than two years before the
cutoff, so within the catalog the relevant items are mostly not new releases. The development
validation window is similar (40.5% new items, 64.5% outside the catalog). The horizon decay
(dev validation NDCG@10 0.020 in the first month, 0.008 over the full window) mixes time and
cohort composition, since the users active in the first month differ from the full cohort. As in
every offline evaluation, an item counts as relevant only if the user chose to review it, so the
absolute metrics are conservative for every system alike.


`select-popularity-window` applies a rule written after exploratory validation numbers for some
windows had been seen (30-365 days, 0.0061-0.0078), and before any test result: best = highest mean
validation NDCG@10 (ties: longer window); chosen = the longest window, at least as long as the
best, whose paired user-level bootstrap 95% interval of the NDCG difference to the best contains
0. It writes `<split>/selections/popularity-window.json`.

Measured on 2026-10-25 (`configs/real.toml`, warm cache). Development validation, 20% sample,
6,123 cohort users:

| Model | NDCG@10 [95% CI] | P@10 | Users with a hit | RMSE |
| --- | --- | --- | --- | --- |
| Popularity W=30 | 0.0072 [0.0058, 0.0087] | 0.0022 | 126 | 1.159 |
| Popularity W=90 | 0.0075 [0.0059, 0.0091] | 0.0020 | 117 | 1.159 |
| Popularity W=180 | 0.0078 [0.0061, 0.0095] | 0.0021 | 125 | 1.159 |
| Popularity W=365 | 0.0061 [0.0046, 0.0078] | 0.0014 | 82 | 1.159 |
| Popularity W=all | 0.00005 [0, 0.00013] | 0.00003 | 2 | 1.159 |
| Bias-only | 0.00015 [0.00001, 0.00034] | 0.00007 | 4 | 1.071 |

The rule chose W = 180 days (30 and 90 are within noise but shorter; 365 and all are worse).

Full category, refit split `refit-c0cb58f463fb-k5-5` (train = everything up to 2020-11-17,
6,685,212 interactions after k=5, 598,310 users, 175,550 items; split 105 s). The test window
keeps 162,922 of 1,715,851 rows (9.5%); the cohort is 44,612 of 53,753 test users:

| Model | NDCG@10 [95% CI] | P@10 | Users with a hit | RMSE |
| --- | --- | --- | --- | --- |
| Popularity W=180 | 0.0034 [0.0030, 0.0038] | 0.00093 | 415 | 1.263 |
| Popularity W=all | 0.00004 [0.00001, 0.00008] | 0.00002 | 7 | 1.263 |
| Bias-only | 0.00007 [0.00003, 0.00011] | 0.00004 | 17 | 1.151 |

Bias fitting on this split stopped at the 10-pass limit without converging (`converged: false`,
last change 2.3e-4), which the approved rule allows. The test window spans about 2.5 years after
the refit cutoff. These baseline test evaluations were run on 2026-10-25, after W was frozen and
before any decision on evaluation horizons or ALS, so every later protocol change is made after
this look at the baseline test results. ALS had not been run at that point.

## Test reporting plan (frozen 2026-10-25, before any ALS result)

Fixed before ALS is trained or touches the test window, so that views are not chosen after
seeing ALS results (approved by the user with the critic's review):

- Headline: NDCG@10 on the standard test view of the refit split, for the four systems
  (popularity with the selected window, bias-only, ALS, bias+ALS), with bootstrap intervals,
  users with a hit against the random expectation, and the random and oracle-popularity
  reference rows. The ALS comparison cohort is this standard cohort of known users.
- Secondary metrics: P@10, hit rate, Recall@50 and @100 (with lists shorter than 100 reported),
  and RMSE (clipped, with raw RMSE).
- Additional views: the 180-day horizon (same systems) and the coverage view (popularity
  fallback and oracle only, reported separately as serving coverage).
- Diagnostics: the training-item segments (<5, 5-9, 10-49, 50+) and, in the coverage view, the
  user types, always with users-with-a-hit counts next to the NDCG values.
- ALS hyperparameters are chosen on the development validation window only, compared with
  popularity W=180 on the same development cohort, and refitted on train + validation for test.
  Already seen before freezing: the baseline test results above.

## ALS and bias+ALS (P4.3, P4.4)

`train --model als|bias-als [--rank R --regularization L --iterations N]` fits explicit-feedback
Spark ALS (`implicitPrefs=False`, `nonnegative=False`, seed from the config, checkpoint every 5
iterations). Plain ALS fits the original ratings; bias+ALS first fits the biases exactly as the
bias-only model and then fits ALS on the signed residuals `rating - mean - user_bias - item_bias`.
Predictions are `dot(factors)` and `mean + user_bias + item_bias + dot(factors)`, unclipped; a
user or item without factors gets no prediction (counted as missing) and no recommendations.

Ranking is exact over the whole catalog: the item factor matrix is broadcast and each partition
scores its users with numpy in blocks, masks seen items and keeps the top k (ties by ascending
item_index). For bias+ALS the item bias is appended to the item vector and a 1 to the user vector,
so the full score, including the item bias, decides the order. This is the exact option of P0.4;
the approximate shortlist approved in P0.4 is not needed (about 44,612 users x 175,550 items x
rank on the full test), so `approximate` is false. The user approved this change to P0.4 on
2026-10-25, and the `shortlist_size`/`shortlist_cap` settings were removed. Spark ALS block counts
are set by `num_user_blocks`/`num_item_blocks` (10, Spark's default, also used by the development
grid before they were configurable); factors depend on blocks and partitioning, so a seed makes
runs reproducible within numerical tolerance, not bit for bit. `fit` records the training RMSE.
Spark's `regParam` (ALS-WR, scaled by each user's and item's rating count) is not comparable to
the absolute bias regularization of 10; do not align them.

Development grid and selection rule, fixed on 2026-10-25 before any ALS result: rank in
{10, 25, 50} x regularization in {0.03, 0.1, 0.3}, 10 iterations, seed 2221, for both ALS models,
fitted on the 20% development training window and evaluated on its validation window.
`select-grid --model als|bias-als` picks the highest mean validation NDCG@10; ties go to the
smaller rank, then the larger regularization, then fewer iterations. The selected settings are
then refitted on the refit split (train + validation) for the test window, without retuning.
Spark's `regParam` is scaled by each user's and item's number of ratings (ALS-WR).

Development grid results, 2026-10-25 (all 18 trials; 20% validation window, 6,123 users; random
expectation 3.2 users with a hit; no list shorter than 100; Spark default 10/10 blocks). Columns:
validation NDCG@10, users with a hit, Recall@100, validation RMSE, training RMSE.

| Model | Rank | Reg. | NDCG@10 | Hits | R@100 | Val. RMSE | Train RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALS | 10 | 0.03 | 0.0001 | 2 | 0.0021 | 2.089 | 0.363 |
| ALS | 10 | 0.1 | 0.0001 | 4 | 0.0026 | 1.299 | 0.373 |
| ALS | 10 | 0.3 | 0.0001 | 4 | 0.0037 | 1.185 | 0.686 |
| ALS | 25 | 0.03 | 0.0003 | 8 | 0.0041 | 1.618 | 0.131 |
| ALS | 25 | 0.1 | 0.0002 | 6 | 0.0034 | 1.226 | 0.284 |
| ALS | 25 | 0.3 | 0.0001 | 3 | 0.0025 | 1.178 | 0.660 |
| ALS (selected) | 50 | 0.03 | 0.0003 | 12 | 0.0057 | 1.408 | 0.089 |
| ALS | 50 | 0.1 | 0.0002 | 5 | 0.0064 | 1.209 | 0.261 |
| ALS | 50 | 0.3 | 0.0001 | 2 | 0.0032 | 1.180 | 0.652 |
| Bias+ALS | 10 | 0.03 | 0.0001 | 5 | 0.0025 | 1.219 | 0.200 |
| Bias+ALS | 10 | 0.1 | 0.0001 | 2 | 0.0034 | 1.146 | 0.347 |
| Bias+ALS | 10 | 0.3 | 0.00004 | 2 | 0.0035 | 1.082 | 0.664 |
| Bias+ALS | 25 | 0.03 | 0.0002 | 5 | 0.0054 | 1.130 | 0.103 |
| Bias+ALS | 25 | 0.1 | 0.0003 | 7 | 0.0048 | 1.102 | 0.279 |
| Bias+ALS | 25 | 0.3 | 0.0002 | 6 | 0.0035 | 1.077 | 0.644 |
| Bias+ALS | 50 | 0.03 | 0.0001 | 4 | 0.0043 | 1.097 | 0.088 |
| Bias+ALS (selected) | 50 | 0.1 | 0.0004 | 7 | 0.0042 | 1.086 | 0.262 |
| Bias+ALS | 50 | 0.3 | 0.0002 | 3 | 0.0044 | 1.073 | 0.637 |

For reference on the same cohort: bias-only 0.0002 (4 hits, validation RMSE 1.071, training RMSE
0.874), popularity W=180 0.0078 (125 hits), oracle popularity 0.0122 (231 hits).

Reading: explicit ALS ranks at about random level. The best trial (12 hits for 3.2 expected,
p = 1.3e-4, about 0.002 after Bonferroni over 18 trials) is marginally above random and an order
of magnitude below popularity W=180; the others are at random. The paired bootstrap intervals
between the selected configuration and most others contain 0, so both selections are within noise.
No configuration beats bias-only on validation RMSE (best 1.073 vs 1.071), training RMSE shows
strong overfitting at low regularization, and the RMSE optimum lies on the upper edge of the grid,
so the grid does not show whether stronger regularization would help RMSE.

Checks: the selected ALS ranks the users' own training ratings >= 4 (seen items not excluded) at
NDCG@10 0.092, about 290 times random, which rules out an index misalignment between factors and
items. Its seen-excluded top 10 favors niche, highly rated items (median 12 training ratings, 34%
of recommended items with fewer than 10, mean item rating 4.84, 7,131 distinct items), while
popularity W=180 recommends 14 distinct items with a median of 62 ratings and a lower mean rating
(3.69) and still hits ten times more often. Which item a user rates next follows exposure and
recency, not how highly items are rated, and a model fitted only to observed ratings cannot see
that. This matches Cremonesi, Koren and Turrin (RecSys 2010): rating-prediction models do poorly on
top-N ranking. The user chose to report this as the result of the proposed experiment (option a);
a wider regularization grid and implicit-feedback ALS were not run.

## Final test results, four systems (frozen plan, 2026-10-25)

Refit split `refit-c0cb58f463fb-k5-5` (train + validation up to 2020-11-17, 6,685,212 ratings,
598,310 users, 175,550 items); settings frozen on the development split: popularity W=180,
bias-only (regularization 10), ALS rank 50 / regularization 0.03, bias+ALS rank 50 /
regularization 0.1, 10 iterations, 10/10 blocks. Refit times: ALS 68 s, bias+ALS 90 s; training
RMSE 0.148 and 0.329. Standard test view, 44,612 known users; random expectation 7.1 users with
a hit. No list is shorter than 100; every system predicts all 162,922 test ratings.

| System | NDCG@10 [95% CI] | P@10 | Hit rate | Recall@100 | Hits / expected | p | RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Popularity W=180 | 0.0034 [0.0030, 0.0038] | 0.00093 | 0.93% | 0.028 | 415 / 7.1 | < 1e-10 | 1.263 |
| Bias-only | 0.00007 [0.00003, 0.00011] | 0.00004 | 0.04% | 0.0018 | 17 / 7.1 | 0.001 | **1.151** |
| ALS | 0.00015 [0.00009, 0.00022] | 0.00006 | 0.06% | 0.0021 | 27 / 7.1 | 1e-8 | 1.448 |
| Bias+ALS | 0.00003 [0.00001, 0.00006] | 0.00002 | 0.02% | 0.0016 | 7 / 7.1 | 0.56 | 1.173 |
| Popularity W=all (row) | 0.00004 [0.00001, 0.00008] | 0.00002 | 0.02% | 0.0017 | 7 / 7.1 | 0.56 | 1.263 |
| Random (reference) | 0.000034 | | | 0.0006 | | | |
| Oracle popularity (reference) | 0.0090 [0.0084, 0.0096] | 0.00257 | 2.53% | 0.055 | 1,128 / 7.1 | | |
| Training global mean 4.270 (reference) | | | | | | | 1.355 |

180-day horizon (21,613 users, 2.6 expected): popularity W=180 0.0054 (302 hits), ALS 0.00019
(16 hits, p = 2e-8), bias+ALS 0.00007 (6 hits, p = 0.047, not significant after correction),
bias-only 0 (0 hits), oracle 0.0158. ALS hits by training-item segment on the standard view:
5-9 items 9 (2.4 expected), 10-49 items 14 (3.4), 50+ items 4 (1.3); popularity W=180: 203, 179
and 33.

Result: recent popularity is the best ranker by a wide margin. Explicit ALS, the best
personalized model, is above random (p = 1e-8) but has 15 times fewer users with a hit than
popularity W=180 (27 vs 415) and a 23 times lower NDCG@10; bias+ALS is at random level. About 30
test p-values are reported: ALS and popularity W=180 survive a Bonferroni correction easily,
bias-only's 17 hits (p = 0.0011) only barely (threshold about 0.0017). The small ALS lift is flat
across training-item segments (about 3.7, 4.1 and 3.1 times random for 5-9, 10-49 and 50+ items),
so the earlier hypothesis that it would concentrate on heavy users is not supported.

Bias-only is the best rating predictor (RMSE 1.151 against 1.355 for the training global mean);
bias+ALS (1.173) does not improve on it and popularity's smoothed item mean reaches 1.263. ALS's
1.448, worse than the global mean, reflects the selection metric: the rule selects by NDCG@10,
which picked the least regularized, most overfitted configuration; on development the most
accurate explicit ALS (rank 50, regularization 0.3) reached 1.180, still above bias-only's 1.071,
so no ALS trial improved on the biases there either. Test results agree qualitatively with the
development grid; since the development selection was within noise, this shows the pattern is
stable, not that the chosen settings are optimal. Explicit rating models do not beat recent
popularity at top-N on this data, and they do not beat biases at rating prediction.

## Scalability benchmark (P6)

The runner, the event-log parser and the report are implemented and tested (`make check-spark`).
The 25/50/100% benchmark `fixed-resource-v1` ran on 2026-10-25; its results are at the end of this
section.

Contract approved by the user on 2026-10-25:
- Nested user samples 0.25 / 0.5 / 1 with seed 2221. All three are split with the cutoffs of the
  100% split `p0.8-0.1-0.1-k5-5` (`--cutoffs-from`), so the 100% point is also re-split.
- The four frozen test settings: popularity W=180, bias λ=10, ALS rank 50 / reg 0.03, bias+ALS
  rank 50 / reg 0.1. They are passed as the `stage.json` of the trained test models. The runner
  refuses before starting if the config cannot reproduce them, and checks them again after each fit.
  There is no hyperparameter search.
- `configs/real.toml` resources (`local[8]`, 12 GB driver, 64 shuffle partitions, event log on).
- One repetition, labelled "single run". Warm cache, declared. MongoDB writes not measured (P7 pending).

Measurement type: stage from materialized input.
- **ETL** is not re-run. It is timed once at full input (`etl/stage.json`) and is not a scaling point.
- **Isolation:** for each fraction the runner builds a fresh mini-run under
  `<run>/benchmarks/<label>/rep-1/fraction-<f>/`, with a copy of `run.json` and a symlink to `etl/`.
  It runs `sample`, `split`, then `train` and `recommend` (top-20 for every training user) for
  each model, each as a new CLI process with its own JVM.
- **Warm cache:** before each fraction it reads the ETL Parquet once, so every scale starts with
  the same warm cache.
- **Order:** fractions run in ascending order; they are not randomized.
- **Host state:** each row records the host load and free memory before the stage.
- **Failures and reruns:** a failed or timed-out stage is a visible row, and the stages that depend
  on it are `skipped`. A completed benchmark is never overwritten; an interrupted one is renamed
  `<label>.aborted-<time>`.

Each stage records its own timings:
- `spark_start` (JVM launch and session) is timed separately.
- The headline ratio uses work seconds: stage total minus `spark_start` minus `verification_counts`.
- The whole stage total and the process wall time are kept as secondary columns.

Sizes come from each stage's actual input rows:
- sample: the ETL rows, constant across scales, so the sample is not input scaling;
- split: the sample rows;
- train and recommend: the training rows after k-core.

Each fraction also records k-core passes and retention (`scales.jsonl`). The event log of each
stage, linked by application id, gives:
- shuffle read/write;
- spill;
- GC share;
- task skew (slowest task / median per Spark stage), as a maximum over the Spark stages whose
  slowest task takes at least 1 s;
- the sampled peak JVM heap.

Run it with nothing else heavy on the host (no other Spark job, no browser load). Before the run it
was estimated at 25-40 min and about 3 GB:
- **Time basis:** split and train scale about linearly in rows (×4-5 from the probe below). The
  exact ALS and bias+ALS top-20 scale with users × items, about 12.5× (536,945/114,908 ×
  153,860/57,407), so about 5.5 min each at 100%.
- **Disk basis:** 0.4 GB for the probe, so about 0.4 + 0.8 + 1.6 GB.
- **Memory:** the heap at 100% is not extrapolated. Watch ALS training on about 5× the rows; that is
  the main out-of-memory risk.

```bash
R=storage/runs/20261025T093611Z-2b4b605ead0f
M=$R/samples/fraction-1-seed-2221/splits/refit-c0cb58f463fb-k5-5/models
.venv/bin/amazon-recommender benchmark --config configs/real.toml --run $R --label fixed-resource-v1 \
  --reference-split $R/samples/fraction-1-seed-2221/splits/p0.8-0.1-0.1-k5-5/stage.json \
  --model-stage $M/popularity-smoothing10-window_days180/stage.json \
  --model-stage $M/bias-regularization10-max_passes10-tolerance0.0001/stage.json \
  --model-stage $M/als-rank50-regularization0.03-iterations10-seed2221-user_blocks10-item_blocks10/stage.json \
  --model-stage $M/bias-als-rank50-regularization0.1-iterations10-seed2221-user_blocks10-item_blocks10-bias_regularization10-bias_max_passes10-bias_tolerance0.0001/stage.json \
  --fractions 0.25 0.5 1
.venv/bin/amazon-recommender benchmark-report --benchmark $R/benchmarks/fixed-resource-v1
```

Outputs are in `<run>/benchmarks/<label>/`:
- `manifest.json`: the contract, resources, host, and git revision plus a hash of uncommitted changes;
- `measurements.jsonl` / `.csv`: one row per stage run, including failed ones;
- `scales.jsonl`;
- after `benchmark-report`: `report.json`, `report.csv` and `scaling.svg`. The figure plots the time
  ratio against the input ratio relative to 25%; the diagonal is linear scaling.

Limitations:
- one host in local mode (`local[8]`), so this is fixed-resource input scaling, not worker speedup;
- three points, one run each;
- ascending order;
- warm cache;
- k-core retention changes with the sample, so it is reported beside the times.

25% probe (2026-10-25, label `probe-0.25`, an estimate only, not the 25% point): 1,624,497 users and
4,288,107 interactions. After the frozen cutoffs and 13 k-core passes, 1,204,219 training rows
remained (114,908 users, 57,407 items). All ten stages completed in 3 min 42 s of wall time, with
identical resources:

| Stage | Work seconds |
| --- | --- |
| split | 35.1 |
| train popularity | 4.3 |
| train bias | 14.1 |
| train ALS | 21.9 |
| train bias+ALS | 32.7 |
| exact top-20 ALS | 26.8 |
| exact top-20 bias+ALS | 27.1 |
| sample (constant input: full ETL scan) | 9.8 |
| top-20 popularity | 6.0 |
| top-20 bias | 10.0 |

Other probe observations:
- **Spark start-up** took 1.6-3.5 s per stage.
- **Memory:** the sampled peak heap was 3.8 GB of 12 GB, and there was no spill. Short stages
  report 0 because the heap is sampled.
- **Split skew:** the split's largest task skew (29) was in a sub-second stage, the projection
  counts, whose slowest task took 0.46 s. The longest Spark stages had skew ≤ 3.1.
- **Host load:** the load was already 3.4 before the probe (desktop applications); this is the only
  statement about external load. The load recorded before each later stage includes the previous
  stage's own `local[8]` work, because the 1-minute average decays slowly. It rose from 3.4 to 7.9.

Results of `fixed-resource-v1` (2026-10-25):
- **Run:** one run per fraction; commit e13b455 with a dirty tree (uncommitted P7 files and a
  test-only Makefile change), identified by hash in `manifest.json`.
- **Status:** 30 of 30 stages completed, with identical Spark resources; MongoDB is not measured.
  The whole run took 26 min 45 s of wall time.
- **Host load:** 1.0 before the start.

Work seconds exclude Spark start-up (1.6-2.0 s per stage) and verification counts. The ratios are
relative to 25%.

| Stage | 25% | 50% | 100% | Size ratio 50% / 100% | Time ratio 50% / 100% |
| --- | --- | --- | --- | --- | --- |
| sample (input constant: full ETL scan) | 9.7 | 12.4 | 6.6 (no copy) | output 2.0 / - | 1.28 / - |
| split | 34.3 | 56.0 | 120.6 | sample rows 2.0 / 4.0 | 1.63 / 3.51 |
| train popularity | 4.4 | 5.1 | 6.0 | train rows 2.3 / 4.9 | 1.17 / 1.37 |
| train bias | 13.9 | 19.1 | 29.5 | 2.3 / 4.9 | 1.37 / 2.12 |
| train ALS | 22.6 | 38.7 | 72.3 | 2.3 / 4.9 | 1.71 / 3.20 |
| train bias+ALS | 34.0 | 53.3 | 97.4 | 2.3 / 4.9 | 1.56 / 2.86 |
| top-20 popularity | 5.5 | 10.0 | 13.4 | 2.3 / 4.9 | 1.84 / 2.45 |
| top-20 bias | 5.9 | 10.1 | 13.8 | 2.3 / 4.9 | 1.71 / 2.35 |
| exact top-20 ALS | 27.8 | 70.7 | 277.8 | users x items 3.7 / 12.5 | 2.54 / 9.99 |
| exact top-20 bias+ALS | 27.6 | 74.5 | 335.2 | 3.7 / 12.5 | 2.70 / 12.16 |
| total work | 185.6 | 349.9 | 972.6 | | 1.89 / 5.24 |

Retention per scale:

| Fraction | Sample users / rows | Train rows after k-core (retained) | k-core passes | Train users / items |
| --- | --- | --- | --- | --- |
| 25% | 1,624,497 / 4,288,107 | 1,204,219 (35.1%) | 13 | 114,908 / 57,407 |
| 50% | 3,251,353 / 8,571,510 | 2,719,619 (39.7%) | 10 | 253,361 / 97,040 |
| 100% | 6,503,429 / 17,158,519 | 5,895,362 (42.9%) | 13 | 536,945 / 153,860 |

Findings, from `report.json` and the event logs:
- **Exact ALS ranking dominates.** At 100% the exact ALS and bias+ALS top-20 take 613 s of 973 s
  of work (63%).
  - Their time follows users x items (every user is scored against the whole catalogue), not the
    training rows: x10.0 and x12.2 for x12.5 users x items, against x4.9 rows. Retention adds to
    this, since k-core keeps a larger share of rows and items as the sample grows.
  - This is the cost of the approved exact ranking (P0.4 change).
- **The split is about linear in its input:** x3.5 for x4.0 sample rows.
  - k-core is 78% of it at 100% (94 s).
  - The number of passes is a confounder: 13, 10 and 13 passes, at about 1.8, 4.0 and 7.2 s per
    pass. The 50% split therefore did less k-core work per row than the other two.
- **Training grows less than linearly in rows** (exponents 0.2-0.7). This is consistent with
  fixed per-job costs (ALS iterations, bias passes, checkpoints) weighing more at small inputs, but
  nothing isolates that overhead. ALS is the steepest (x3.2 for x4.9 rows).
- **The sample is not input scaling.** It reads all 17.2 M ETL rows at every fraction, and at 100%
  it copies nothing.
- **No memory pressure:** no spill in any stage, and GC at most 2.7% of executor time. Shuffle
  writes reach 12.9 GB (bias+ALS training at 100%).
  - The peak heap, 5.4 GB of 12 GB, was sampled during the popularity top-20 at 100%.
  - A sample includes garbage not yet collected, so it is not the working set of a stage: it does
    not mean popularity is the stage that needs the most memory.
- **Parallelism is bounded by partitions.**
  - Most of the split's Spark-stage time (153 of 177 s at 100%) is in stages with 9 tasks. On 8
    cores they need a second wave of a single task.
  - The likely cause, not verified, is the 9 input partitions carried through the k-core
    checkpoints. Repartitioning to a multiple of 8 might help; it was not tested.
  - The longest stages have skew 1.8-2.6. The largest skew over 1-s stages (27.1) is again the
    projection counts, a 1.0 s stage.
- **Host load** rose to 8-9 during the run, from our own `local[8]` work.

These are single-run, warm-cache measurements on one 16-core host in local mode. They show how
stage time grows with input under fixed resources, not distributed speedup. With three points and
one run, the ratios carry run-to-run noise that is not measured. As a rough indication, the 25%
probe, run at a higher host load (3.4), and the 25% point agree within 1.4 s per stage, except
top-20 bias (10.0 s against 5.9 s, about 70%). Short stages (under 15 s) can therefore vary a lot
between runs, and their ratios are the least reliable. The long stages (split, ALS training and the
exact top-20) are the informative ones.

## Serving lists and MongoDB publication (P7)

`materialize --run R --sample S --split P --model-label M --fallback-label F [--batch-users N]` writes
top-k serving lists (`model.recommendation_count`, 20) for every user of the sample, in original
IDs, to `<split>/serving/<M>/`. It starts Spark and never contacts MongoDB.
- Users are typed like the coverage view. `known` users are in the split mapping and get model M
  (`rec_type` "model"). `removed_by_k_core` users have history up to the train cutoff but were
  removed by the k-core. `new` users have none. Removed and new users, and any known user M returns
  no list for, get the popularity fallback F (`rec_type` "fallback", P0.5). F must be a popularity model.
- Seen items are the user's interactions up to the train cutoff, before the k-core, and are never
  recommended. Both models rank the training catalog exactly, so every list has min(k, unseen
  catalog items) entries. A shorter list can only mean the catalog ran out (`catalog_exhausted`).
- Users are processed in batches of N consecutive serving indices (default 500,000). Each batch
  is written, moved into place and then marked in `batches/`. After an interruption, rerunning
  the same command reuses the completed batches. It refuses if the labels, k, N, or the
  sample/split/model `stage.json` files changed (`plan.json`).
- Outputs: `users/` (one row per user, empty lists included: user_type, rec_type, approximate,
  unseen_items, length), `recommendations/` (user_id, rank, parent_asin, score) and `stage.json`.
  `stage.json` holds the coverage audit. The stage fails if a user is duplicated, a length is not
  min(k, unseen), ranks are not 1..n, an item repeats in a list, a score is missing or increases
  with rank, a seen item is recommended, or the listed users differ from the sample's users.

`publish --config C --serving <split>/serving/<M> --metadata <run>/etl/metadata` (needs `MONGODB_URI`)
writes one document per `users/` row into `recommendations_<version>`, and the titles of the
recommended items into `items_<version>`. `<version>` hashes the run, sample, split, labels, k
and the materialization time, so republishing the same materialization rewrites identical
documents in the same collections, and anything else gets new collections. A user document has
`_id` (original user ID), `version`, `user_type`, `rec_type`, `approximate`, `short_list_reason`
and the ordered `items` (rank, parent_asin, score). The publisher writes with bounded bulk
upserts (1,000 per batch, 3 retries with backoff) and one client per Spark partition. It then
checks the document count and 20 documents per user type (chosen by a seeded hash) against the Parquet. Only after that does it
record the publication in `publications` and switch the `active_run` pointer (`_id`
"current"), so readers never see a partial run. The model, labels, coverage and publication
time live in the pointer, not in the user documents. A missing title is stored as null, and the
item falls back to its ID.

Real-category materialization, 2026-10-26 (`configs/real.toml`, refit split
`refit-c0cb58f463fb-k5-5`, model and fallback popularity W=180, the user's choice for serving; one run):
6,503,429 users, equal to the sample's users. 598,310 are known (rec_type model), 5,257,666 were
removed by the k-core and 647,453 are new (both fallback). That gives 130,068,580 rows, no short
or empty lists, and every audit check at 0. It took 308 s: 33 s for the user table, 14 batches
of 9.3-14.8 s, and a 128 s audit. Output: 708 MB of Parquet. Serving popularity to known users
means their "model" lists are not personalized; the dashboard must show the publication's model.

Tests: `make check-spark` covers materialization on a fixture with all three user types, a
short list, the exact fallback order, an interrupted run that resumes, and one failing case per
audit rule. `make check-mongo` (`RUN_MONGO_TESTS=1`, `MONGODB_URI` set) round-trips a publication
twice. On this Linux host (kernel 7.0) the Compose image `mongo:8.0` refuses to start
("Linux kernel versions 6.19 and newer has a known incompatibility", SERVER-121912). The tests
passed against a throwaway `mongo:8.2.12` container. `compose.yaml` is unchanged.

The throwaway container has no authentication and listens only on localhost. It keeps its data in
anonymous volumes (the original container used the floating tag `mongo:8.2`, then 8.2.12), so `docker stop`/`docker start` preserve the data and `docker rm -v` deletes it:

```bash
docker run -d --name id2221-p7-test -p 127.0.0.1:27117:27017 --memory 1g --cpus 1 mongo:8.2.12
export MONGODB_URI=mongodb://127.0.0.1:27117
make check-mongo   # the Makefile sets RUN_MONGO_TESTS=1
```

Real-category publication, 2026-10-26 (the materialization above, metadata
`<run>/etl/metadata`, the throwaway container on port 27117, database `amazon_recommender`; one run):
version `popularity_8918f91d1974`, collections `recommendations_popularity_8918f91d1974` and
`items_popularity_8918f91d1974`. It wrote 6,503,429 user documents, equal to the materialized
users, and 30 item documents. Every list draws from the same W=180 popularity ranking minus the
user's seen items, so only 30 distinct items occur. Grouped by (user_type, rec_type), the documents
match the materialization audit (598,310 known/model, 5,257,666 removed_by_k_core/fallback,
647,453 new/fallback). The built-in check compared 60 seeded documents (20 per user type) with the
Parquet and found no difference. Then `active_run` ("current") was switched to this version
(published_at 12:42:47 UTC). It took 197 s including Spark start: 166 s for the user documents,
1.6 s for the items and 24 s for the validation. 29 of the 30 items have no title in the
category metadata (the ETL counts 313,873 interaction items without a title), so the dashboard
will show their `parent_asin`. The record is in `<serving>/publications/` and in the `publications` collection, and `source` in the
pointer is an absolute local path. The publication exists only in the throwaway container, not
in the Compose MongoDB. If the container is removed, rerun `publish`.

## First run on the real category

`configs/real.toml` is a per-machine profile and is gitignored. Create it from the committed
local profile and adjust the resources to your host, then validate it:

```bash
cp configs/local.toml configs/real.toml
# Edit [spark] for your machine and set experiment.user_fraction = 0.2.
.venv/bin/amazon-recommender validate-config --config configs/real.toml
```

In local mode the driver JVM does all the work. As a starting point, use about half the logical
cores for `local[N]` (`nproc`), about 40% of RAM for `driver_memory` (`free -g`), and 4-8 times N
shuffle partitions. These are rules of thumb, not measured minimums.

The first real run used, on a Linux host with 16 cores and 30 GB RAM (approved by the user on
2026-10-25): `master = "local[8]"`, `driver_memory = "12g"`, `shuffle_partitions = 64`,
`event_log_enabled = true`, `user_fraction = 0.2`, with the other values as in `configs/local.toml`.
Expected disk use is ~14 GB at peak and ~10.5 GB after the run, mostly the decompressed inputs.
Each run's `run.json` and `stage.json` files keep the full configuration used. Timings are only
comparable between runs on the same machine with the same resources, so run all scaling
experiments on one host.

Gzip files cannot be split, so each raw scan of a `.gz` runs as a single task. Decompress
the official files once and run on the `.jsonl` copies; record the `.gz` checksums too:

```bash
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64   # or an SDKMAN Temurin 17
export PATH="$JAVA_HOME/bin:$PATH"
make doctor && make check && make check-spark
sha256sum data/raw/*.jsonl.gz > data/raw/SHA256SUMS.gz.txt
zcat data/raw/Movies_and_TV.jsonl.gz > data/raw/Movies_and_TV.jsonl
zcat data/raw/meta_Movies_and_TV.jsonl.gz > data/raw/meta_Movies_and_TV.jsonl
.venv/bin/amazon-recommender init-run --config configs/real.toml \
  --reviews data/raw/Movies_and_TV.jsonl --metadata data/raw/meta_Movies_and_TV.jsonl
.venv/bin/amazon-recommender etl --config configs/real.toml --run storage/runs/<run-id>
# Check etl/stage.json before sampling.
.venv/bin/amazon-recommender sample --config configs/real.toml --run storage/runs/<run-id>
```

Measured on 2026-10-25 (run `20261025T093611Z-2b4b605ead0f`, `configs/real.toml`, Java 17.0.20.1):
ETL read 17,328,314 reviews with no rejections, removed 169,795 duplicate (user, item) pairs,
and kept 17,158,519 interactions from 6,503,429 users and 747,764 items, 1997-2023, in 27 yearly
partitions (756 MB of Parquet). It took 37.5 s in total, including 7.3 s of verification counts.
This is a warm-cache measurement: `zcat` had just written the input, so the 8.4 GB were read from the
page cache (about 1.2 GB/s per scan), not from disk. Scaling runs must state cold or warm cache
(cold: `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` first) and report processing time with and
without `verification_counts`. All reviewed items have a metadata row (checked
separately; later runs record it in `interaction_items_without_metadata_row`), but 313,873 have no title: the raw metadata has 313,988 records without `title` and 14 blank ones,
so the dashboard must fall back to the item ID. The 20% development sample (`sample`, seed 2221)
kept 1,300,010 users (achieved fraction 0.19990) and 3,431,067 interactions (20.0%) in 12.8 s,
also warm cache, in `samples/fraction-0.2-seed-2221/` (143 MB). These figures come from one run on one machine;
they are not a scalability result.

Split `p0.8-0.1-0.1-k5-5`, measured on 2026-10-25 (warm cache). Each sample derives its own
cutoffs (P0.2 allows this), so the two validation windows are not the same time period:

| Sample | Cutoffs (train / validation) | Train after k=5 | Users / items | Validation kept | Test kept | Test users (with rating >= 4) | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20% | 2018-12-21 / 2020-11-13 | 912,329 (33% of train window) | 88,251 / 47,491 | 18,524 (5.4%) | 8,575 (2.5%) | 3,924 (3,245) | 31 s |
| 100% | 2018-12-25 / 2020-11-17 | 5,895,362 (43%) | 536,945 / 153,860 | 173,119 (10.1%) | 85,192 (5.0%) | 32,265 (26,713) | 111 s |

Most held-out rows are lost because their user or item is unknown in the filtered training
window: users who start reviewing after the cutoff, and users and items removed by the k-core.
User sampling lowers every item's degree, so the 20% sample loses far more items than the full
category. Development tuning uses only the 20% validation window (6,123 users with a relevant
item; 7,239 in total), with a coarse grid and paired bootstrap intervals; the headline test is the
full category. The development train graph is sparser than the full one, so tuned settings do
not transfer exactly, especially the fixed bias regularization.

## Environment

Read-only inspection of the adjacent `ID2221-labs` directory on 2026-10-23 found:

| Component | Labs environment | Project choice |
| --- | --- | --- |
| PySpark | 3.5.6, installed and pinned | 3.5.6 |
| Python | 3.11.14, installed; labs requires 3.11 | 3.11.14 / Python 3.11 |
| Java | Temurin 17.0.13, SDKMAN installation and pin | Java 17 |
| NumPy | 1.26.4, installed | 1.26.4 |

Spark 3.5.6 supports Java 17 and Python 3.8+
([official documentation](https://spark.apache.org/docs/3.5.6/)). The default
system Java was 25 during inspection; select Java 17 explicitly. Delta Lake is
used in the labs but is not required by this project's Parquet design.

From this project's root, prepare an isolated environment:

```bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
sdk env
export JAVA_HOME="$HOME/.sdkman/candidates/java/17.0.13-tem"
export PATH="$JAVA_HOME/bin:$PATH"
uv sync --locked
uv run amazon-recommender doctor
uv run amazon-recommender validate-config --config configs/local.toml
uv run python -m unittest discover -s tests -v
```

The package was installed successfully into this project's `.venv` on
2026-10-23, independently of the labs environment. `uv.lock` records the
resolved dependencies; `uv sync --locked` reproduces them without updating the
lock. Installing dependencies does not start Spark. Select this project's
`.venv/bin/python` interpreter in VS Code.

## Daily development without services

After installation, these commands use the existing environment directly:

```bash
make check             # Run the Python-only tests; no Spark JVM or database
make config            # Validate and display current experiment settings
make prepare-fixture   # Create a run manifest for small synthetic inputs
```

`make doctor` also inspects Java, so first set `JAVA_HOME` to Java 17 as above.
Only `etl`, `sample`, `split`, `train`, `recommend`, `materialize`, `evaluate`, `select-popularity-window`,
`select-grid`, `benchmark` and `publish` start Spark; only `publish` contacts MongoDB. Use
`.venv/bin/amazon-recommender --help` to list them.

Edit `configs/local.toml` for a development run. Its `[experiment]` section
controls the user fraction, split `[train, validation, test]`, k-core thresholds,
and relevance threshold. There is no mandatory development percentage or split;
`[1.0, 0.0, 0.0]` is valid for a training-only fixture. `purpose = "final"`
requires all users, 80/10/10, k=5, and relevance=4. This validates settings only;
it does not approve full-data execution. Lower the development k-core thresholds
when exercising very small synthetic data later.

The `[model]` section stores the popularity, bias and ALS settings, including the ALS block
counts. Ranking is exact, so there are no retrieval budget settings.

To prepare a run for your own existing local review/metadata files:

```bash
.venv/bin/amazon-recommender init-run --config configs/local.toml \
  --reviews /path/to/reviews.jsonl.gz --metadata /path/to/metadata.jsonl.gz
```

This streams each input once to compute its SHA-256, then saves a new
`storage/runs/<run-id>/run.json` with input paths/sizes/checksums, configuration,
seed, dependency versions, and Git revision/dirty state. It does not parse,
copy, or download data. Large files take time to hash. Input paths are relative
to the current working directory; configuration storage paths are relative to
the configuration file. The JSON is marked `prepared`, not completed. Existing
runs are never overwritten; an unfinished directory is not proof of a successful
pipeline stage. Stage completion/reuse will be added with actual ETL stages.

`schemas.py` defines raw review, clean interaction, metadata, integer mapping,
and flat recommendation row schemas. Schema-only tests verify types without
starting a Spark context or Java gateway. The committed NDJSON fixtures are
tiny invented records, not downloaded Amazon reviews.

Development stays simple: no generic pipeline scheduler, retry framework, or
automatic timing of every operation. Add real stage commands alongside their
implementations; add coarse timings later where needed for the project report.

To inspect the labs installation directly without creating a Spark session:

```bash
../ID2221-labs/.venv/bin/python -c 'import sys; from importlib.metadata import version; print(sys.version); print("pyspark", version("pyspark"))'
"$HOME/.sdkman/candidates/java/17.0.13-tem/bin/java" -version
```

The foundation commands can also run without installing the project, using the
existing labs interpreter for read-only verification:

```bash
export JAVA_HOME="$HOME/.sdkman/candidates/java/17.0.13-tem"
PYTHONPATH=src ../ID2221-labs/.venv/bin/python -m amazon_recommender doctor
PYTHONPATH=src ../ID2221-labs/.venv/bin/python -m unittest discover -s tests -v
```

The `doctor` command reads package metadata and runs `java -version`; it never
imports PySpark or contacts MongoDB. A mismatch produces a nonzero exit code.

## Approved local profile

The user approved development on this Mac with Spark `local[2]`, a 2 GB driver,
and local MongoDB in Docker with persistent storage bound to localhost on
2026-10-23. `configs/local.toml` records these settings. Four SQL shuffle
partitions and UTC SQL timestamps are initial fixture settings.

Paths resolve relative to the configuration file, independent of the current
working directory. Validation creates no storage. The session factory is an
explicit Python API, `amazon_recommender.spark.session.create_spark_session`;
calling it starts Spark. No CLI command currently calls it. It configures
separate event logs, Spark lineage checkpoints, spill space, and warehouse
directories under `storage/`. Event logging is disabled by default; enable it
when investigating Spark bottlenecks.

The machine reports 10 logical CPUs and approximately 220 GiB free disk.
RAM inspection was restricted by the sandbox; this is not a full-data resource
approval. `configs/full.example.toml` is deliberately rejected by validation
until a full-data host, shared storage, executor configuration, and budget are
approved and implemented. A local fixture run will not demonstrate distributed
scalability.

## MongoDB: recommended development setup

Use the supplied Docker Compose service for initial development. MongoDB stores
the precomputed recommendation lists, titles, and experiment summaries; Spark
will continue to use Parquet for processing. A MongoDB document is a JSON-like
record, a collection groups documents, and a database groups collections. The
future dashboard will fetch a user's document by its original user ID.

The Compose service uses the Docker Official `mongo:8.0` image with a named
volume, localhost-only port, authentication, a health check, and limits of one
CPU and 1 GB RAM. The image supports ARM64. Named volumes are recommended for
MongoDB persistence on macOS
([image documentation](https://hub.docker.com/_/mongo)). The `8.0` tag tracks
patch updates; freeze the resolved image digest before measured experiments.
These MongoDB limits are initial fixture settings, subject to measurement.

To start MongoDB when ready (this does not start Spark):

1. Open Docker Desktop and wait for its engine to be ready.
2. Copy the environment template and edit `.env` to set a nonempty password:

   ```bash
   cp .env.example .env
   ```

3. Validate the Compose configuration, start the service, and inspect its status:

   ```bash
   docker compose config --quiet
   docker compose up -d mongodb
   docker compose ps
   ```

4. Connect using the shell already inside the container. Enter the password at
   the prompt; the example assumes the template's `localadmin` username:

   ```bash
   docker compose exec mongodb mongosh --username localadmin --authenticationDatabase admin
   ```

   Run `db.runCommand({ ping: 1 })` and then `exit` in that shell.

5. Stop the service with `docker compose down`. The named volume persists;
   `down -v` deletes its data. Initialization credentials apply only when the
   volume is first created; editing `.env` does not change an existing password.

The publisher (and later the dashboard) reads the URI from `MONGODB_URI`.
Compose reads `.env` automatically; Python does not, so export this variable
in the shell that runs those applications. Do not put credentials in TOML,
logs, or Git. Separate application users and reader permissions will be added
with the publisher/dashboard; the bootstrap administrator is for setup.

MongoDB Atlas is an option if teammates need a shared remote database later.
Its free tier has roughly 0.5 GB of storage including indexes, so it is not a
sensible default for millions of users with top-20 lists
([Atlas limits](https://www.mongodb.com/docs/atlas/reference/free-shared-limitations/)).
We must measure document sizes and available disk before full publication.
No MongoDB container has been started as part of this increment.

## Approved preprocessing decisions (P0.2)

On 2026-10-23 the user approved a final evaluation protocol with one global
chronological split by interaction count, targeting 80/10/10. All interactions
with an identical timestamp stay in one window; actual counts, proportions,
and boundary timestamps must be saved.

Before splitting, keep the latest review for each original (user, item) pair.
For reviews with the same pair and timestamp, keep the highest rating; exact
duplicates collapse. This follows the proposal's deduplication order: a later
repeat can remove an earlier interaction from the training window. The approved
highest-rating tie-break may favor positive ratings among tied duplicates.

The user replaced the proposal's fixed 20% development sample with a data
portion chosen case by case while testing. No fixed development percentage is
required. Record the portion, selection method, seed, and dataset identity for
each run so results are reproducible. The precise sampling contract will be
resolved with P3.1; changing data portions does not establish model improvements.

Intermediate implementation tests may choose both their data portion and their
split proportions or date cutoffs case by case; they are not required to use
80/10/10 or share the final dataset's cutoffs. Record each run's split policy
and actual boundaries, and use the same split for models compared within a run.
Label these runs exploratory, separately from final evaluation. P0.2 is complete
as a protocol decision; the split implementation remains pending.

Before measured 25/50/100% scalability experiments, freeze a common set of
temporal cutoffs for those comparisons. Their specific cutoffs and cohort
construction will be resolved with the benchmark contract.

## Approved filtering and evaluation decisions (P0.3)

On 2026-10-23 the user approved these shared rules for all four systems:

- Final training-only iterative k-core: at least five interactions per user
  and item after convergence. Intermediate tests may use lower thresholds.
- Ranking relevance: held-out rating >=4. Rank the full retained training-item
  catalog, excluding training-seen items. Use the same eligible training-known
  users with at least one relevant held-out item in that catalog for all models.
  Report exclusions, users without relevant items, and cold-start coverage.
- Precision@10: relevant returned items in the first ten ranks divided by 10,
  including short lists. Binary DCG@10 sums `relevance / log2(rank + 1)`;
  normalize by the ideal DCG for up to ten relevant held-out items. Macro-average
  Precision and NDCG equally across eligible users. Missing ranks contribute
  zero; break score ties by ascending item ID with a shared persisted mapping.
- RMSE: the same held-out ratings with training-known users/items for all
  models, including ratings below 4. Clip reconstructed predictions to [1,5],
  retain raw diagnostics, and leave ranking scores unclipped. Report missing
  or nonfinite predictions and coverage; do not silently drop a different
  evaluation cohort for each model.
- Popularity: rank by training interaction counts. For RMSE, separately predict
  `(item_rating_sum + alpha * training_global_mean) / (item_count + alpha)`.
  Fit these statistics on training only. The user approved `alpha=10` with
  the bias-fitting settings; record this value in each run.

P0.3 is complete as a protocol decision. Filtering, prediction, evaluation,
and their fixture tests remain to be implemented. The later P0.4 approval below
permits approximate bias+ALS retrieval while retaining these relevance sets,
eligible users, and metric denominators.

## Approved bias fitting and approximate retrieval (P0.4)

Use the training global mean, initialize biases to zero, and alternate user
updates followed by item updates. Regularize both with 10; stop after a complete
pass when the maximum absolute bias change is below 0.0001, or after 10 passes.
Missing user/item biases default to zero. Fit explicit ALS on signed residuals;
restore the global mean and both bias terms for rating predictions.

The user approved approximate ranking for now to avoid high full-catalog
scoring costs. Retrieve a bounded latent ALS shortlist, exclude seen items,
and rerank using full model scores including item bias. This can miss strong
items outside the shortlist. Record candidate/expansion caps and counts, label
recommendations and ranking results as approximate, and report any short lists
caused by those caps. Ground truth and metric denominators remain unchanged.
Verify retrieval limitations with exhaustive scores only on tiny fixtures.
Superseded on 2026-10-25: the user approved exact full-catalog ranking instead (see the ALS
section), so no recommendation is approximate.

## Approved evaluation/serving policy and deferred budget (P0.5)

Freeze the chosen development dataset and split before final model selection.
Select by validation NDCG@10, freeze hyperparameters, and fit all four systems on
full-category training data before held-out test evaluation without retuning.
Initially serve from the training-only model. Users without factors receive
labeled popularity fallbacks with seen items excluded; report fallback coverage
separately from model-comparison metrics.

Publish into run-specific collections, validate completeness, then switch an
active-run pointer after successful publication. This policy is approved;
the publisher is implemented (see the serving section); the dashboard is not.

The user approved fixture work only and explicitly deferred full-data hosting
and experiment budgets. Tuning grids, benchmark repetitions, full-data memory,
disk, and runtime limits still need approval before large jobs. Suggested grid
values were not approved as executable defaults. This does not revoke the
existing instruction not to start Spark.

## Remaining experiment decisions

Local Phase 0 algorithm and serving choices are settled. Before full-data
execution, resolve the host, resource limits, tuning grid, benchmark cohorts,
cutoffs, and repetitions. Development sampling details remain for P3.1.
P0.1/P0.5 retain open full-data budget gates; P0.2/P0.3/P0.4 record completed
protocol decisions. No algorithm implementation is implied by these approvals.

Raw data, processed data, models, checkpoints, logs, credentials, and local
Markdown handoff files are ignored. Keep project documentation in this README;
do not stage the local plan or `AGENTS.md`.
