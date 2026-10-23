PYTHON := .venv/bin/python
CLI := .venv/bin/amazon-recommender

.PHONY: check config doctor prepare-fixture

check:
	$(PYTHON) -m unittest discover -s tests -v

config:
	$(CLI) validate-config --config configs/local.toml

doctor:
	$(CLI) doctor

prepare-fixture:
	$(CLI) init-run --config configs/local.toml --reviews tests/fixtures/reviews.ndjson --metadata tests/fixtures/metadata.ndjson
