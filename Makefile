PYTHON := .venv/bin/python
CLI := .venv/bin/amazon-recommender

.PHONY: check check-spark check-mongo check-e2e config dashboard doctor prepare-fixture

check:
	$(PYTHON) -m unittest discover -s tests -v

check-spark:
	RUN_SPARK_TESTS=1 PYSPARK_PYTHON=$(abspath $(PYTHON)) $(PYTHON) -m unittest tests.test_etl tests.test_sample tests.test_split tests.test_models tests.test_evaluate tests.test_benchmark tests.test_materialize tests.test_item_details -v

check-mongo:
	RUN_MONGO_TESTS=1 PYSPARK_PYTHON=$(abspath $(PYTHON)) $(PYTHON) -m unittest tests.test_publish tests.test_dashboard -v

check-e2e:
	RUN_SPARK_TESTS=1 RUN_MONGO_TESTS=1 PYSPARK_PYTHON=$(abspath $(PYTHON)) $(PYTHON) -m unittest tests.test_e2e -v

config:
	$(CLI) validate-config --config configs/local.toml

dashboard:
	$(CLI) dashboard --config $(or $(CONFIG),configs/local.toml)

doctor:
	$(CLI) doctor

prepare-fixture:
	$(CLI) init-run --config configs/local.toml --reviews tests/fixtures/reviews.ndjson --metadata tests/fixtures/metadata.ndjson
