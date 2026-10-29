# Scalable Recommendation with Spark ALS

ID2221 project, Group 0: Riccardo Fragale, Tommaso Tron, Edoardo F.A. de Cal.

A batch recommendation pipeline on the Amazon Reviews 2023 *Movies and TV* category, built with
PySpark/Spark SQL, Spark MLlib ALS, Parquet, MongoDB and Streamlit. It compares popularity,
bias-only, plain ALS and bias+ALS models with a temporal split, measures input scaling, and serves
top-20 lists from MongoDB through a Streamlit dashboard.

The report (`report/report.pdf`) covers what we did, the dataset, the methodology and the results.
This file only explains how to run the code.

## Repository layout

| Path | Content |
| --- | --- |
| `src/amazon_recommender/` | the package and its `amazon-recommender` CLI (one subcommand per stage) |
| `configs/` | Spark/experiment profiles: `local.toml` (tests, small data), `real.toml` (the profile behind every reported result) |
| `tests/` | unit, Spark, MongoDB and end-to-end tests, with small fixtures |
| `compose.yaml`, `.env.example` | the MongoDB service and its credentials template |
| `report/` | the 2-page report |

The pipeline writes all its output under `storage/runs/<run-id>/`, which is not included in the
repository.

## Requirements

- Linux (macOS untested)
- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- Java 17 (Spark 3.5.6 does not run on newer default JDKs)
- Docker with Compose v2, for MongoDB (only needed to publish and for the dashboard)
- About 15 GB of free disk for the full category. `configs/real.toml` uses `local[8]` and a 12 GB
  driver: lower `[spark]` for smaller machines.

## Setup

```bash
uv sync --frozen                                     # creates .venv from uv.lock
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64  # any Java 17
export PATH="$JAVA_HOME/bin:$PATH"
make doctor                                          # prints versions; starts nothing
make check                                           # Python-only tests
make check-spark                                     # Spark tests (about 4 min)
```

MongoDB runs as a local Compose service (authenticated, bound to 127.0.0.1:27017). Set a password
in `.env`, then export `MONGODB_URI`, which is what the publisher and the dashboard read:

```bash
cp .env.example .env            # set MONGO_INITDB_ROOT_PASSWORD
docker compose up -d mongodb
set -a; . ./.env; set +a
export MONGODB_URI="mongodb://$MONGO_INITDB_ROOT_USERNAME:$(.venv/bin/python -c "import os, urllib.parse; print(urllib.parse.quote_plus(os.environ['MONGO_INITDB_ROOT_PASSWORD']))")@127.0.0.1:27017/?authSource=admin"
make check-mongo                # publisher and dashboard tests
make check-e2e                  # whole pipeline on a synthetic dataset (about 2.5 min)
```

`make check-e2e` runs every stage end to end without any download, so it is the quickest way to
check that the setup works.

## Dataset

[Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab), category Movies and
TV: 17.3M reviews, about 8.4 GB decompressed. It is not included. Download and decompress it
(gzip input cannot be split by Spark):

```bash
mkdir -p data/raw && cd data/raw
curl -LO https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Movies_and_TV.jsonl.gz
curl -LO https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Movies_and_TV.jsonl.gz
zcat Movies_and_TV.jsonl.gz > Movies_and_TV.jsonl
zcat meta_Movies_and_TV.jsonl.gz > meta_Movies_and_TV.jsonl
cd ../..
```

## Running the pipeline

Each command starts its own Spark session, writes a `stage.json` record, and refuses to overwrite
a completed stage. Labels such as `fraction-0.2-seed-2221` are printed by the stage that creates
them.

```bash
A=.venv/bin/amazon-recommender; C=configs/real.toml

# 1. Ingest: cleaned interactions as yearly Parquet partitions
$A init-run --config $C --reviews data/raw/Movies_and_TV.jsonl --metadata data/raw/meta_Movies_and_TV.jsonl
R=storage/runs/<run-id>                                   # printed by init-run
$A etl --config $C --run $R
$A item-details --config $C --run $R

# 2. Model selection on a 20% user sample (validation window)
$A sample --config $C --run $R                            # fraction-0.2-seed-2221
$A split  --config $C --run $R --sample fraction-0.2-seed-2221
D="--config $C --run $R --sample fraction-0.2-seed-2221 --split p0.8-0.1-0.1-k5-5"
for W in 30 90 180 365 all; do $A train $D --model popularity --popularity-window-days $W; done
$A train $D --model bias
for M in als bias-als; do for K in 10 25 50; do for L in 0.03 0.1 0.3; do
  $A train $D --model $M --rank $K --regularization $L --iterations 10; done; done; done
for L in $(ls $R/samples/fraction-0.2-seed-2221/splits/p0.8-0.1-0.1-k5-5/models); do
  $A evaluate $D --model-label $L --window validation; done
$A select-popularity-window $D
$A select-grid $D --model als
$A select-grid $D --model bias-als

# 3. Final test: full category, refit on train+validation with the selected settings
$A sample --config $C --run $R --user-fraction 1           # fraction-1-seed-2221
F="--config $C --run $R --sample fraction-1-seed-2221"
$A split $F
$A split $F --refit-from $R/samples/fraction-1-seed-2221/splits/p0.8-0.1-0.1-k5-5/stage.json
T="$F --split <refit label>"                               # printed by the previous command
$A train $T --model popularity --popularity-window-days 180
$A train $T --model bias
$A train $T --model als      --rank 50 --regularization 0.03 --iterations 10
$A train $T --model bias-als --rank 50 --regularization 0.1  --iterations 10
for L in $(ls $R/samples/fraction-1-seed-2221/splits/<refit label>/models); do
  $A evaluate $T --model-label $L --window test; done

# 4. Scaling benchmark: nested 25/50/100% samples with the frozen settings
M=$R/samples/fraction-1-seed-2221/splits/<refit label>/models
$A benchmark --config $C --run $R --label fixed-resource-v1 \
  --reference-split $R/samples/fraction-1-seed-2221/splits/p0.8-0.1-0.1-k5-5/stage.json \
  --model-stage $M/<model label>/stage.json \
  --fractions 0.25 0.5 1                                   # repeat --model-stage for each model
$A benchmark-report --benchmark $R/benchmarks/fixed-resource-v1

# 5. Serving: top-20 lists for every user, published to MongoDB, then the dashboard
$A materialize $T --model-label popularity-smoothing10-window_days180 \
  --fallback-label popularity-smoothing10-window_days180
$A publish --config $C \
  --serving $R/samples/fraction-1-seed-2221/splits/<refit label>/serving/popularity-smoothing10-window_days180 \
  --metadata $R/etl/metadata --item-details $R/item_details
make dashboard CONFIG=configs/real.toml                    # http://localhost:8501
```

Use `$A <command> --help` to see every option. On another machine, copy `configs/real.toml`,
adapt `[spark]`, and check it with `$A validate-config --config <file>`. Timings are only
comparable between runs on the same machine with the same resources.

## Dataset citation

Hou et al., *Bridging Language and Items for Retrieval and Recommendation*, arXiv:2403.03952, 2024.
