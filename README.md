# ID2221-Project

Distributed explicit-feedback ALS on Amazon Reviews 2023 (Movies and TV), for
ID2221, Autumn 2026. The proposal specifies yearly Parquet storage, PySpark/Spark
SQL, popularity/bias-only/plain ALS/bias+ALS comparisons, temporal evaluation,
fixed-resource input scaling, MongoDB top-20 serving, and a Streamlit dashboard.

The development foundation provides an installed Python package and lockfile,
validated experiment settings, input/output schemas, run configuration/input
snapshots, a non-starting environment check, an explicit Spark-session factory,
a MongoDB Compose setup, and foundation tests. Data ingestion, models,
metrics, publication, and the dashboard are still to be implemented. No quality
or scalability results have been measured. Spark has not been started for this
increment, so the Spark-session factory is not yet runtime verified.

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
All current CLI commands operate without starting Spark or MongoDB. Use
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
Sampling, splitting, training, and retrieval themselves are not implemented yet.

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
