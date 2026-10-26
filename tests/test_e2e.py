"""End-to-end fixture workflow (P9.2): the real CLI, stage by stage, from raw JSONL to a dashboard lookup.

Runs only with RUN_SPARK_TESTS=1, RUN_MONGO_TESTS=1 and MONGODB_URI set (make check-e2e). Each
command is a new `python -m amazon_recommender` process, in the order of the README section
"Fixture workflow (end to end)"; keep both lists in step. The category is synthetic and seeded
(tests/e2e_fixture.py); storage is a temporary directory and MongoDB a temporary
`test_e2e_<uuid>` database, dropped afterwards, so the real active pointer is never touched.

Also checked: a completed stage refuses to run again, a repeated publication rewrites identical
documents, and materialize recovers from a simulated interruption (its last completed batch is
deleted, and the stage marked incomplete again; nothing is killed).
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

from tests.e2e_fixture import generate

ROOT = Path(__file__).resolve().parents[1]
ENABLED = (os.environ.get("RUN_SPARK_TESTS") == "1" and os.environ.get("RUN_MONGO_TESTS") == "1"
           and bool(os.environ.get("MONGODB_URI")))
K = 20
BATCH_USERS = 25
ALS_OPTIONS = ["--rank", "4", "--iterations", "5"]
MODELS = {"popularity": [], "bias": [], "als": ALS_OPTIONS, "bias-als": ALS_OPTIONS}


def parquet_rows(directory: Path, order: tuple) -> list[dict]:
    """All rows of a (batch-partitioned) Parquet directory, without the partition column, sorted."""
    import pyarrow.parquet as pq

    rows = [{key: value for key, value in row.items() if key != "batch"}
            for row in pq.read_table(directory).to_pylist()]
    return sorted(rows, key=lambda row: tuple(row[key] for key in order))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def serving_digest(serving: Path) -> str:
    return digest([parquet_rows(serving / "users", ("user_id",)),
                   parquet_rows(serving / "recommendations", ("user_id", "rank"))])


@unittest.skipUnless(ENABLED, "set RUN_SPARK_TESTS=1, RUN_MONGO_TESTS=1 and MONGODB_URI (make check-e2e)")
class EndToEndTests(unittest.TestCase):
    timings: dict = {}

    @classmethod
    def cli(cls, *args, expect: int = 0) -> subprocess.CompletedProcess:
        started = time.monotonic()
        result = subprocess.run([sys.executable, "-m", "amazon_recommender", *map(str, args)],
                                capture_output=True, text=True, cwd=ROOT, timeout=900)
        step = args[0] + ("-refit" if "--refit-from" in args else "")
        cls.timings[step] = round(cls.timings.get(step, 0) + time.monotonic() - started, 1)
        if result.returncode != expect:
            raise AssertionError(f"{' '.join(map(str, args))} exited {result.returncode}:\n{result.stderr[-3000:]}")
        return result

    @classmethod
    def stage(cls, *args) -> Path:
        """Run a stage; its last stdout line is the path it wrote."""
        return Path(cls.cli(*args).stdout.strip().splitlines()[-1])

    @classmethod
    def setUpClass(cls):
        from pymongo import MongoClient

        started = time.monotonic()
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.fixture = generate(root / "raw")
        cls.database_name = f"test_e2e_{uuid.uuid4().hex}"
        profile = (ROOT / "configs/local.toml").read_text()
        for old, new in (('storage_root = "../storage"', f'storage_root = "{root / "storage"}"'),
                         ('database = "amazon_recommender"', f'database = "{cls.database_name}"')):
            if old not in profile:
                raise AssertionError(f"configs/local.toml no longer contains {old!r}")
            profile = profile.replace(old, new)
        config = cls.config = root / "e2e.toml"
        config.write_text(profile)
        cls.client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=5000)
        try:
            cls.run_chain(config)
        except BaseException:
            cls.tearDownClass()
            raise
        cls.timings["chain_total"] = round(time.monotonic() - started, 1)

    @classmethod
    def run_chain(cls, config: Path):
        run = cls.run_directory = cls.stage("init-run", "--config", config, "--reviews", cls.fixture["reviews"],
                                  "--metadata", cls.fixture["metadata"]).parent
        cls.stage("etl", "--config", config, "--run", run)
        cls.details = cls.stage("item-details", "--config", config, "--run", run).parent
        sample = cls.stage("sample", "--config", config, "--run", run).parent.name
        split_stage = cls.stage("split", "--config", config, "--run", run, "--sample", sample)
        refit = cls.stage("split", "--config", config, "--run", run, "--sample", sample,
                          "--refit-from", split_stage).parent.name
        common = ("--config", config, "--run", run, "--sample", sample, "--split", refit)
        cls.labels = {model: cls.stage("train", *common, "--model", model, *options).parent.name
                      for model, options in MODELS.items()}
        for label in cls.labels.values():
            cls.stage("evaluate", *common, "--model-label", label, "--window", "test")
        materialize = (*common, "--model-label", cls.labels["als"], "--fallback-label", cls.labels["popularity"],
                       "--batch-users", BATCH_USERS)
        serving = cls.serving = cls.stage("materialize", *materialize).parent
        cls.first = {"stage": json.loads((serving / "stage.json").read_text()), "digest": serving_digest(serving)}

        # Simulated interruption: the last completed batch is removed and the stage made incomplete again.
        partial = serving.with_name(serving.name + ".partial")
        serving.rename(partial)
        (partial / "stage.json").unlink()
        last = cls.first["stage"]["batches"]["count"] - 1
        (partial / "batches" / f"{last:05d}.json").unlink()
        for output in ("users", "recommendations"):
            shutil.rmtree(partial / output / f"batch={last}")
        if cls.stage("materialize", *materialize).parent != serving:
            raise AssertionError("the resumed materialization wrote another directory")
        cls.resumed = {"stage": json.loads((serving / "stage.json").read_text()), "digest": serving_digest(serving)}

        cls.publication_record = cls.stage("publish", "--config", config, "--serving", serving,
                                           "--metadata", run / "etl" / "metadata", "--item-details", cls.details)
        cls.database = cls.client[cls.database_name]

    @classmethod
    def tearDownClass(cls):
        print(f"\nend-to-end wall seconds: {json.dumps(cls.timings)}", file=sys.stderr)
        if hasattr(cls, "client"):
            cls.client.drop_database(cls.database_name)
            cls.client.close()
        cls.directory.cleanup()

    def published(self) -> dict:
        pointer = self.database["active_run"].find_one({"_id": "current"})
        documents = {name: sorted(self.database[pointer[name]].find(), key=lambda d: d["_id"])
                     for name in ("recommendations", "items")}
        return {"pointer": {key: value for key, value in pointer.items() if key != "published_at"},
                "digest": digest(documents), "counts": {name: len(docs) for name, docs in documents.items()}}

    def test_fixture_is_not_degenerate(self):
        stage = self.resumed["stage"]
        users = parquet_rows(self.serving / "users", ("user_id",))
        known = [user for user in users if user["user_type"] == "known"]
        intended = {"known": "core", "removed_by_k_core": "sparse", "new": "late"}
        for user_type, group in intended.items():  # exact, so the README's counts hold
            self.assertEqual(stage["coverage"]["users_by_type"][user_type], len(self.fixture["users"][group]))
        self.assertGreaterEqual(min(user["unseen_items"] for user in known), K)  # catalog >= k + max seen
        self.assertGreaterEqual(sum(user["length"] == K for user in known), 1)
        self.assertEqual(stage["batches"]["count"], -(-stage["counts"]["users"] // BATCH_USERS))  # 3 for 70 users
        titles = {item["_id"]: item["title"] for item in self.database[self.published()["pointer"]["items"]].find()}
        self.assertTrue(set(self.fixture["untitled"]) & set(titles))  # an untitled item is recommended
        self.assertEqual(stage["counts"]["users"], sum(len(ids) for ids in self.fixture["users"].values()))

    def test_serving_lists(self):
        users = parquet_rows(self.serving / "users", ("user_id",))
        rows = parquet_rows(self.serving / "recommendations", ("user_id", "rank"))
        lists = {}
        for row in rows:
            lists.setdefault(row["user_id"], []).append(row)
        for user in users:
            items = lists.get(user["user_id"], [])
            self.assertEqual(len(items), min(K, user["unseen_items"]), user["user_id"])
            self.assertEqual([item["rank"] for item in items], list(range(1, len(items) + 1)))
            self.assertEqual(len({item["parent_asin"] for item in items}), len(items))
            self.assertEqual(user["rec_type"], "model" if user["user_type"] == "known" else "fallback")
        checks = self.resumed["stage"]["coverage"]["checks"]
        self.assertTrue(all(value == 0 for value in checks.values()), checks)

    def test_simulated_interruption_recomputes_only_the_removed_batch(self):
        first, resumed = self.first["stage"]["batches"], self.resumed["stage"]["batches"]
        self.assertEqual(resumed["reused_on_resume"], first["count"] - 1)
        self.assertEqual(resumed["records"][:-1], first["records"][:-1])  # same completion records
        self.assertNotEqual(resumed["records"][-1]["completed_at"], first["records"][-1]["completed_at"])
        self.assertEqual(self.resumed["digest"], self.first["digest"])
        self.assertEqual(self.resumed["stage"]["coverage"], self.first["stage"]["coverage"])

    def test_completed_stages_refuse_to_run_again(self):
        model = self.serving.parents[1] / "models" / self.labels["popularity"] / "stage.json"
        before = model.read_bytes()
        refused = self.cli("train", "--config", self.config, "--run", self.run_directory, "--sample",
                           self.serving.parents[3].name, "--split", self.serving.parents[1].name,
                           "--model", "popularity", expect=2)
        self.assertIn("already trained", refused.stderr)
        self.assertEqual(model.read_bytes(), before)

    def test_publication_lookup_and_evidence(self):
        from amazon_recommender.dashboard.evidence import load_evidence
        from amazon_recommender.dashboard.lookup import OK, UNKNOWN_USER, lookup_user

        published = self.published()
        coverage = self.resumed["stage"]["coverage"]
        self.assertEqual(published["counts"]["recommendations"], coverage["users"])
        record = json.loads(self.publication_record.read_text())
        self.assertEqual(record["validation"]["user_documents"], coverage["users"])

        recommendations = self.database[published["pointer"]["recommendations"]]
        with_untitled = recommendations.find_one({"user_type": "known",
                                                  "items.parent_asin": {"$in": self.fixture["untitled"]}})
        self.assertIsNotNone(with_untitled)
        result = lookup_user(self.database, f" {with_untitled['_id']} ")
        self.assertEqual(result.status, OK)
        self.assertEqual([row["rank"] for row in result.rows], list(range(1, K + 1)))
        for row in result.rows:
            asin = row["parent_asin"]
            untitled = asin in self.fixture["untitled"]
            label = self.fixture["labels"].get(asin)
            self.assertEqual((row["title_missing"], row["derived"]), (untitled, label is not None))
            self.assertEqual(row["title"], (label or asin) if untitled else f"Synthetic Movie {asin[-2:]}")
            self.assertEqual(row["image"], self.fixture["images"].get(asin))
            self.assertEqual(row["link"], f"https://www.amazon.com/dp/{asin}")
        items = {item["_id"]: item for item in self.database[published["pointer"]["items"]].find()}
        shown = {asin for asin in self.fixture["untitled"] if asin in items}
        self.assertTrue(any(self.fixture["labels"][asin] for asin in shown), shown)  # a derived label is published
        self.assertTrue(any(self.fixture["labels"][asin] is None for asin in shown), shown)  # and a bare one
        self.assertEqual(published["pointer"]["item_details"]["items_without_details"], 0)
        for user_id in (self.fixture["users"]["sparse"][0], self.fixture["users"]["late"][0]):
            self.assertEqual(lookup_user(self.database, user_id).document["rec_type"], "fallback")
        self.assertEqual(lookup_user(self.database, "e2e-nobody").status, UNKNOWN_USER)

        evidence = load_evidence(published["pointer"]["source"])
        self.assertEqual(list(evidence["views"]), ["test"])
        rows = evidence["views"]["test"]["rows"]
        self.assertEqual(sorted(row["model"] for row in rows), sorted(MODELS))
        self.assertEqual(evidence["selected"], {model: {"status": "no selection record"} for model in MODELS})
        self.assertEqual((evidence["selections"], evidence["benchmarks"], evidence["problems"]), ([], [], []))
        self.assertTrue(evidence["headline_label"].startswith("full category (all users)"))

    def test_repeated_publication_rewrites_identical_documents(self):
        before = self.published()
        self.stage("publish", "--config", self.config, "--serving", self.serving,
                   "--metadata", self.run_directory / "etl" / "metadata", "--item-details", self.details)
        self.assertEqual(self.published(), before)  # published_at is the only field allowed to change
