"""Item-details normalization runs everywhere; the stage itself only with RUN_SPARK_TESTS=1 (make check-spark)."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from amazon_recommender.artifacts import create_run
from amazon_recommender.config import load_config
from amazon_recommender.item_details import extract, main_image, run_item_details, split_names, unique_names

ROOT = Path(__file__).resolve().parents[1]
SPARK = os.environ.get("RUN_SPARK_TESTS") == "1"
LIST = json.dumps  # list-family values reach extract() as JSON text, as the Spark reader keeps them


class NormalizationTests(unittest.TestCase):
    def test_strings_are_split_and_suffixes_rejoined(self):
        self.assertEqual(split_names("Robert Downey, Jr., Gwyneth Paltrow"), ["Robert Downey, Jr.", "Gwyneth Paltrow"])
        self.assertEqual(split_names("A Smith, sr, B Jones,III,, C"), ["A Smith, sr", "B Jones, III", "C"])
        self.assertEqual(split_names("Jr., Someone"), ["Jr.", "Someone"])  # nothing to attach to

    def test_lists_are_never_split_but_split_suffixes_are_rejoined(self):
        details = {"Starring": LIST(["Robert Downey, Jr.", "Gwyneth Paltrow"]), "Actors": "X, Y"}
        self.assertEqual(extract(details, None)["starring"], ["Robert Downey, Jr.", "Gwyneth Paltrow"])
        upstream = {"Starring": LIST(["Robert Downey", "Jr.", "Antonio Banderas", "Michael Sheen"])}  # B083L8HVBX
        self.assertEqual(extract(upstream, None)["starring"], ["Robert Downey, Jr.", "Antonio Banderas", "Michael Sheen"])

    def test_names_are_trimmed_deduplicated_in_order_and_capped(self):
        self.assertEqual(unique_names([" Janelle  Monáe", "janelle monáe", "B", "C", "D"]), ["Janelle Monáe", "B", "C"])

    def test_both_detail_families(self):
        prime = extract({"Directors": LIST(["Rian Johnson"]), "Starring": LIST(["Daniel Craig", "Chris Evans"]),
                         "Studio": LIST(["Lionsgate"])}, None)
        self.assertEqual(prime, {"directors": ["Rian Johnson"], "starring": ["Daniel Craig", "Chris Evans"],
                                 "studio": "Lionsgate", "image": None})
        physical = extract({"Director": "Ann Lee, Bo Ma", "Actors": "C, Jr., D", "Studio": "Criterion"}, None)
        self.assertEqual((physical["directors"], physical["starring"], physical["studio"]),
                         (["Ann Lee", "Bo Ma"], ["C, Jr.", "D"], "Criterion"))
        self.assertEqual(extract(None, None), {"directors": [], "starring": [], "studio": None, "image": None})
        self.assertEqual(extract({"Directors": LIST([]), "Director": "Ann Lee"}, None)["directors"], ["Ann Lee"])

    def test_main_image_prefers_main_variant_and_mid_size(self):
        images = [{"variant": "PT01", "thumb": "t1"}, {"variant": "MAIN", "480w": "m480", "large": "ml"}]
        self.assertEqual(main_image(images), "m480")
        self.assertEqual(main_image([{"variant": "PT01", "large": "l", "thumb": "t"}]), "l")
        self.assertIsNone(main_image([]))


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to start Spark")
class SparkItemDetailsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        base = load_config(ROOT / "configs/local.toml")
        self.config = replace(base, storage_root=root / "storage")
        rows = [
            {"parent_asin": "prime", "title": None, "main_category": "Prime Video", "rating_number": 10,
             "details": {"Directors": ["Rian Johnson"], "Starring": ["A", "B", "C", "D", "a"],
                         "Studio": ["Lionsgate"], "Subtitles": ["English"]},
             "images": [{"variant": "PT01", "thumb": "t"},
                        {"variant": "MAIN", "360w": "u360", "480w": "u480", "1920w": "big"}]},
            {"parent_asin": "disc", "title": "A Disc", "main_category": "Movies & TV", "rating_number": 3,
             "details": {"Director": "Ann Lee", "Actors": "Robert Downey, Jr., Gwyneth Paltrow", "Studio": "Criterion",
                         "Number of discs": 2},
             "images": [{"variant": "MAIN", "thumb": "t", "large": "l", "hi_res": "h"}]},
            {"parent_asin": "bare", "title": None},
            {"parent_asin": "twice", "main_category": "Prime Video", "rating_number": 1,
             "details": {"Directors": ["Low"]}},
            {"parent_asin": "twice", "main_category": "Prime Video", "rating_number": 7,
             "details": {"Directors": ["High"]}},
        ]
        metadata = root / "metadata.jsonl"
        metadata.write_text("".join(json.dumps(row) + "\n" for row in rows) + "{not json\n")
        reviews = root / "reviews.jsonl"
        reviews.write_text(json.dumps({"user_id": "u", "parent_asin": "prime", "rating": 5.0,
                                       "timestamp": 1_600_000_000_000}) + "\n")
        self.run_directory = create_run(self.config, reviews=reviews, metadata=metadata)
        self.metadata = metadata

    def test_stage_extracts_both_families_and_resolves_duplicates(self):
        from pyspark.sql import SparkSession

        output = run_item_details(self.config, self.run_directory)
        stage = json.loads((output / "stage.json").read_text())
        spark = SparkSession.builder.master("local[1]").config("spark.ui.enabled", "false").getOrCreate()
        try:
            items = {row.parent_asin: row.asDict() for row in spark.read.parquet(str(output / "items")).collect()}
        finally:
            spark.stop()
        self.assertEqual(sorted(items), ["bare", "disc", "prime", "twice"])
        self.assertEqual(items["prime"], {"parent_asin": "prime", "main_category": "Prime Video",
                                          "directors": ["Rian Johnson"], "starring": ["A", "B", "C"],
                                          "studio": "Lionsgate", "image": "u360", "rating_number": 10})
        self.assertEqual((items["disc"]["directors"], items["disc"]["starring"], items["disc"]["studio"],
                          items["disc"]["image"]), (["Ann Lee"], ["Robert Downey, Jr.", "Gwyneth Paltrow"],
                                                    "Criterion", "l"))
        self.assertEqual((items["bare"]["directors"], items["bare"]["image"]), ([], None))
        self.assertEqual(items["twice"]["directors"], ["High"])  # larger rating_number wins
        self.assertEqual(stage["counts"], {"items": 4, "with_directors": 3, "with_starring": 2,
                                           "with_main_category": 3, "with_studio": 2, "with_image": 2,
                                           "with_rating_number": 3, "rows_read": 6})
        with self.assertRaisesRegex(FileExistsError, "already written"):
            run_item_details(self.config, self.run_directory)

    def test_changed_input_is_refused(self):
        with self.metadata.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "changed since init-run"):
            run_item_details(self.config, self.run_directory)
