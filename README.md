# ID2221-Project

Distributed explicit-feedback ALS on Amazon Reviews 2023 (Movies and TV), for
ID2221, Autumn 2026. The proposal specifies yearly Parquet storage, PySpark/Spark
SQL, popularity/bias-only/plain ALS/bias+ALS comparisons, temporal evaluation,
fixed-resource input scaling, MongoDB top-20 serving, and a Streamlit dashboard.

The development foundation provides an installed Python package and lockfile,
validated experiment settings, input/output schemas, run configuration/input
snapshots, a non-starting environment check, an explicit Spark-session factory,
a MongoDB Compose setup, and foundation tests. Data ingestion, models,
metrics, publication, and the dashboard are still to be implemented. The ETL stage
described below is the first implemented stage. No quality
or scalability results have been measured. Spark has not been started for this
increment, so the Spark-session factory is not yet runtime verified.

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

`make check` skips the Spark tests; `make check-spark` runs the ETL, sampling and split ones and needs Java 17.
They passed on 2026-10-25 on synthetic data on the Linux machine (Ubuntu 24.04), which uses
OpenJDK 17.0.20.1 selected through `JAVA_HOME`.

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
Only `etl` and `sample` start Spark; no command contacts MongoDB. Use
`.venv/bin/amazon-recommender --help` to list them.

Edit `configs/local.toml` for a development run. Its `[experiment]` section
controls the user fraction, split `[train, validation, test]`, k-core thresholds,
and relevance threshold. There is no mandatory development percentage or split;
`[1.0, 0.0, 0.0]` is valid for a training-only fixture. `purpose = "final"`
requires all users, 80/10/10, k=5, and relevance=4. This validates settings only;
it does not approve full-data execution. Lower the development k-core thresholds
when exercising very small synthetic data later.

The `[model]` section stores ALS/bias settings and bounded approximate retrieval
settings. Initial shortlist size 200 and cap 1000 are editable development
starting points, not measured recommendations or approved full-data budgets.
Training and retrieval themselves are not implemented yet.

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

The eventual Python publisher/dashboard will read the URI from `MONGODB_URI`.
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

## Approved evaluation/serving policy and deferred budget (P0.5)

Freeze the chosen development dataset and split before final model selection.
Select by validation NDCG@10, freeze hyperparameters, and fit all four systems on
full-category training data before held-out test evaluation without retuning.
Initially serve from the training-only model. Users without factors receive
labeled popularity fallbacks with seen items excluded; report fallback coverage
separately from model-comparison metrics.

Publish into run-specific collections, validate completeness, then switch an
active-run pointer after successful publication. This policy is approved;
the publisher and serving components are still unimplemented.

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
