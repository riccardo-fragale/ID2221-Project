PYTHON := .venv/bin/python
CLI := .venv/bin/amazon-recommender

.PHONY: check check-spark config doctor prepare-fixture

check:
	$(PYTHON) -m unittest discover -s tests -v

check-spark:
	RUN_SPARK_TESTS=1 $(PYTHON) -m unittest tests.test_etl tests.test_sample tests.test_split tests.test_models tests.test_evaluate tests.test_benchmark -v

config:
	$(CLI) validate-config --config configs/local.toml

doctor:
	$(CLI) doctor

prepare-fixture:
	$(CLI) init-run --config configs/local.toml --reviews tests/fixtures/reviews.ndjson --metadata tests/fixtures/metadata.ndjson
