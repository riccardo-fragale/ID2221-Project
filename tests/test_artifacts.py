"""Local run preparation must preserve inputs and never imply stage completion."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class RunTests(unittest.TestCase):
    def test_run_snapshots_are_distinct_and_input_changes_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(load_config(ROOT / "configs/local.toml"), storage_root=root / "storage")
            reviews, metadata = root / "reviews.ndjson", root / "metadata.ndjson"
            reviews.write_text("first input\n")
            metadata.write_text("metadata\n")
            first = create_run(config, reviews=reviews, metadata=metadata)
            initial = (first / "run.json").read_bytes()
            manifest = json.loads(initial)
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(manifest["config_digest"], config.digest)
            self.assertEqual(manifest["inputs"]["reviews"]["sha256"], hashlib.sha256(reviews.read_bytes()).hexdigest())
            reviews.write_text("changed input\n")
            second = create_run(config, reviews=reviews, metadata=metadata)
            self.assertNotEqual(first, second)
            self.assertEqual((first / "run.json").read_bytes(), initial)
            self.assertNotEqual(json.loads((second / "run.json").read_text())["inputs"], manifest["inputs"])

    def test_missing_input_leaves_no_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(load_config(ROOT / "configs/local.toml"), storage_root=Path(directory) / "storage")
            with self.assertRaises(FileNotFoundError):
                create_run(config, reviews=Path(directory) / "missing", metadata=Path(directory) / "missing")
            self.assertFalse(config.storage_root.exists())

    def test_schemas_build_without_starting_spark(self):
        script = """
from amazon_recommender.schemas import review_schema, interaction_schema, metadata_schema, mapping_schema, recommendation_schema
from pyspark import SparkContext
assert review_schema()['timestamp'].dataType.simpleString() == 'bigint'
assert interaction_schema()['event_time'].dataType.simpleString() == 'timestamp'
assert mapping_schema('item')['item_index'].dataType.simpleString() == 'int'
assert metadata_schema()['title'].nullable
assert recommendation_schema()['rank'].dataType.simpleString() == 'int'
assert SparkContext._gateway is None
assert SparkContext._active_spark_context is None
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
