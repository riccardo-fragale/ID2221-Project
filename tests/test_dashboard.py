"""Dashboard logic runs everywhere, against a fake database and a temporary record tree; the MongoDB
checks run only with RUN_MONGO_TESTS=1 and MONGODB_URI set."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import uuid

from amazon_recommender.dashboard import tables
from amazon_recommender.dashboard.evidence import EvidenceError, load_evidence
from amazon_recommender.dashboard.lookup import EMPTY_ID, NO_ACTIVE_RUN, OK, UNKNOWN_USER, lookup_user
from amazon_recommender.publish import ACTIVE_COLLECTION, ACTIVE_ID

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/amazon_recommender/dashboard/app.py"
RUN_ID = "20260101T000000Z-aaaaaaaaaaaa"


class FakeCollection:
    def __init__(self, documents=()):
        self.documents = {document["_id"]: document for document in documents}
        self.queries = []

    def find_one(self, query):
        self.queries.append(query)
        return self.documents.get(query["_id"])

    def find(self, query):
        self.queries.append(query)
        return [self.documents[key] for key in query["_id"]["$in"] if key in self.documents]


class FakeDatabase(dict):
    def __missing__(self, name):
        self[name] = FakeCollection()
        return self[name]


def publication(version, k=3):
    return {"_id": ACTIVE_ID, "version": version, "recommendations": f"recommendations_{version}",
            "items": f"items_{version}", "k": k, "model_label": "m", "fallback_label": "f", "run_id": RUN_ID,
            "split": "s", "published_at": "2026-01-01T00:00:00+00:00", "source": "/nowhere"}


def user(user_id, items, version="v1", reason=None):
    return {"_id": user_id, "version": version, "user_type": "known", "rec_type": "model", "approximate": False,
            "short_list_reason": reason,
            "items": [{"rank": rank, "parent_asin": asin, "score": score} for rank, asin, score in items]}


def database_with(version="v1", users=(), items=(), k=3):
    database = FakeDatabase()
    database[ACTIVE_COLLECTION] = FakeCollection([publication(version, k)])
    database[f"recommendations_{version}"] = FakeCollection(users)
    database[f"items_{version}"] = FakeCollection(items)
    return database


class LookupTests(unittest.TestCase):
    def test_rows_follow_rank_with_titles_from_one_batched_query(self):
        database = database_with(users=[user("U1", [(2, "b", 2.0), (1, "a", 3.0), (3, "c", 1.0)])],
                                 items=[{"_id": "a", "title": "Alpha"}, {"_id": "b", "title": "Beta"},
                                        {"_id": "c", "title": "Gamma"}])
        result = lookup_user(database, "  U1 ")
        self.assertEqual(result.status, OK)
        self.assertEqual([(row["rank"], row["title"], row["score"]) for row in result.rows],
                         [(1, "Alpha", 3.0), (2, "Beta", 2.0), (3, "Gamma", 1.0)])
        self.assertEqual(len(database["items_v1"].queries), 1)
        self.assertFalse(result.short_list)
        self.assertEqual(result.missing_titles, 0)

    def test_missing_titles_fall_back_to_parent_asin(self):
        database = database_with(users=[user("U1", [(1, "a", 3.0), (2, "b", 2.0), (3, "c", 1.0)])],
                                 items=[{"_id": "a", "title": None}, {"_id": "b", "title": "  "}])
        rows = lookup_user(database, "U1").rows
        self.assertEqual([(row["title"], row["title_missing"]) for row in rows],
                         [("a", True), ("b", True), ("c", True)])
        self.assertEqual(tables.recommendation_table(rows)[0]["title"], "a (no title in metadata)")

    def test_untitled_items_with_details_get_a_marked_derived_label(self):
        database = database_with(users=[user("U1", [(1, "a", 3.0), (2, "b", 2.0), (3, "c", 1.0)])], items=[
            {"_id": "a", "title": None, "main_category": "Prime Video", "directors": ["Rian Johnson"],
             "starring": ["Daniel Craig", "Chris Evans"], "image": "https://img/a.jpg"},
            {"_id": "b", "title": None, "main_category": "Prime Video", "directors": [], "starring": []},
            {"_id": "c", "title": "Gamma", "directors": ["Ignored"]}])
        result = lookup_user(database, "U1")
        a, b, c = result.rows
        self.assertEqual((a["title"], a["derived"], a["title_missing"], a["image"]),
                         ("Prime Video · dir. Rian Johnson · Daniel Craig, Chris Evans", True, True, "https://img/a.jpg"))
        self.assertEqual((b["title"], b["derived"], b["title_missing"]), ("b", False, True))  # category alone is not enough
        self.assertEqual((c["title"], c["derived"]), ("Gamma", False))
        self.assertEqual((result.missing_titles, result.derived_labels), (2, 1))
        table = tables.recommendation_table(result.rows)
        self.assertEqual([row["title"] for row in table],
                         ["Prime Video · dir. Rian Johnson · Daniel Craig, Chris Evans (derived: no title in metadata)",
                          "b (no title in metadata)", "Gamma"])
        self.assertEqual(table[0]["link"], "https://www.amazon.com/dp/a")

    def test_derived_label_parts_are_optional(self):
        from amazon_recommender.dashboard.lookup import derived_label

        self.assertEqual(derived_label({"directors": ["D"]}), "dir. D")
        self.assertEqual(derived_label({"main_category": "Prime Video", "starring": ["S1", "S2"]}),
                         "Prime Video · S1, S2")
        self.assertIsNone(derived_label({"_id": "old", "title": None}))  # publication without details

    def test_short_and_empty_lists_keep_their_reason(self):
        database = database_with(users=[user("U1", [(1, "a", 1.0)], reason="catalog_exhausted"),
                                        user("U2", [], reason="catalog_exhausted")])
        short = lookup_user(database, "U1")
        self.assertTrue(short.short_list)
        self.assertEqual(short.document["short_list_reason"], "catalog_exhausted")
        empty = lookup_user(database, "U2")
        self.assertEqual((empty.status, empty.rows), (OK, []))
        self.assertEqual(database["items_v1"].queries, [{"_id": {"$in": ["a"]}}])  # no query for an empty list

    def test_unknown_empty_and_case_sensitive_ids(self):
        database = database_with(users=[user("U1", [(1, "a", 1.0)])])
        self.assertEqual(lookup_user(database, "u1").status, UNKNOWN_USER)
        self.assertEqual(lookup_user(database, "   ").status, EMPTY_ID)
        self.assertEqual(lookup_user(database, None).status, EMPTY_ID)

    def test_no_active_run(self):
        self.assertEqual(lookup_user(FakeDatabase(), "U1").status, NO_ACTIVE_RUN)

    def test_pointer_switch_is_seen_by_the_next_lookup(self):
        database = database_with(users=[user("U1", [(1, "a", 1.0)])], items=[{"_id": "a", "title": "Old"}])
        database["recommendations_v2"] = FakeCollection([user("U1", [(1, "z", 9.0)], version="v2")])
        database["items_v2"] = FakeCollection([{"_id": "z", "title": "New"}])
        self.assertEqual(lookup_user(database, "U1").rows[0]["title"], "Old")
        database[ACTIVE_COLLECTION] = FakeCollection([publication("v2")])
        result = lookup_user(database, "U1")
        self.assertEqual((result.publication["version"], result.document["version"], result.rows[0]["title"]),
                         ("v2", "v2", "New"))


def write(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def evaluation(model, label, params, ndcg, reference_only=False, view="test", run_id=RUN_ID):
    return {
        "stage": "evaluate", "status": "completed", "run_id": run_id, "sample": "fraction-1-seed-1",
        "split": "refit-x", "window": "test", "model": model, "model_label": label, "params": params,
        "reference_only": reference_only, "approximate": False, "purpose": "exploratory",
        "view": {"name": view, "horizon_days": None, "coverage": False, "label": None},
        "metrics": {
            "k": 10, "relevant_rating": 4.0, "catalog_items": 50,
            "cohort": {"eligible_users": 7, "held_out_users": 9, "users_without_relevant": 2},
            "ranking": {"ndcg_at_k": ndcg, "ndcg_ci95": [0.0, ndcg], "precision_at_k": 0.1, "hit_rate": 0.2,
                        "recall_at_50": 0.3, "recall_at_100": 0.4, "users_with_hit": 3, "short_lists_at_100": 0,
                        "bootstrap": {"resamples": 20, "seed": 1, "unit": "user"}},
            "random_reference": {"ndcg_at_k": 0.001, "note": "random"},
            "signal_vs_random": {"expected_users_with_hit": 0.5, "p_value_one_sided": 0.01, "test": "poisson"},
            "rating": None if reference_only else {"rmse": 1.1, "rmse_raw": 1.2},
        },
    }


def selection(path: Path, stage, model, chosen, params=None, sample="fraction-0.2-seed-1", split="p-k5-5"):
    record = {"stage": stage, "sample": sample, "split": split, "rule": "max", "rule_written": "before",
              "users": 5, "chosen": chosen, "mean_ndcg": {str(chosen): 0.5, "other": 0.1},
              "difference_vs_chosen_ci95": {str(chosen): [0, 0]}}
    if model:
        record["model"] = model
    if params is not None:
        record["params"] = {chosen: params}
    return write(path, record)


def record_tree(root: Path) -> Path:
    """A run with a published refit split, a 20% development split and two benchmarks."""
    run = root / "runs" / RUN_ID
    write(run / "run.json", {"run_id": RUN_ID})
    sample = run / "samples/fraction-1-seed-1"
    write(sample / "stage.json", {"stage": "sample", "purpose": "exploratory", "user_fraction": 1.0,
                                  "users": {"after": 9}, "interactions": {"after": 30}})
    split = sample / "splits/refit-x"
    write(split / "stage.json", {"stage": "split", "split": "refit-x", "policy": {"refit_from": {"path": "p"}},
                                 "before_filtering": {"interactions": {"train": 25, "test": 5}},
                                 "k_core": {"converged_after": 2, "retained_interactions": 20,
                                            "retained_fraction": 0.8},
                                 "mappings": {"users": {"count": 7}, "items": {"count": 50}}})
    serving = split / "serving/pop"
    write(serving / "stage.json", {"stage": "materialize", "status": "completed", "run_id": RUN_ID,
                                   "sample": "fraction-1-seed-1", "split": "refit-x",
                                   "coverage": {"users": 9, "rows": 27}})
    rows = [("popularity", "pop-180", {"smoothing": 10.0, "window_days": 180}, 0.03),
            ("popularity", "pop-all", {"smoothing": 10.0, "window_days": "all"}, 0.001),
            ("bias", "bias-10", {"regularization": 10.0}, 0.002),
            ("als", "als-r5-u2", {"rank": 5, "regularization": 0.1, "seed": 1, "user_blocks": 2}, 0.004),
            ("oracle-popularity", "oracle-popularity", {}, 0.09)]
    for model, label, params, ndcg in rows:
        write(split / f"evaluations/test/{label}/stage.json",
              evaluation(model, label, params, ndcg, reference_only=model == "oracle-popularity"))
    write(split / "evaluations/test-h180d/pop-180/stage.json",
          evaluation("popularity", "pop-180", rows[0][2], 0.05, view="test-h180d"))
    write(split / "evaluations/test/foreign/stage.json",
          evaluation("als", "foreign", {}, 0.5, run_id="20250101T000000Z-bbbbbbbbbbbb"))
    development = run / "samples/fraction-0.2-seed-1"
    write(development / "stage.json", {"stage": "sample", "purpose": "exploratory", "user_fraction": 0.2})
    selections = development / "splits/p-k5-5/selections"
    selection(selections / "popularity-window.json", "select-popularity-window", None, 180)
    selection(selections / "als.json", "select-grid", "als", "als-r5", {"rank": 5, "regularization": 0.1, "seed": 1})
    selection(selections / "misfiled.json", "select-grid", "bias-als", "x", {}, split="elsewhere")
    benchmark = run / "benchmarks/b1"
    write(benchmark / "manifest.json", {"run_id": RUN_ID, "status": "completed", "repetitions": 1,
                                        "host": {"logical_cpus": 4, "mem_total_bytes": 2**33}})
    write(benchmark / "report.json", {
        "benchmark": "b1", "measurement_type": "stage", "scaling": "fixed", "cache": {"state": "warm"},
        "spark": {"master": "local[2]"}, "spark_resources_identical": True,
        "steps": {"split": {"points": [
            {"fraction": 0.5, "completed": 1, "runs": 1, "statuses": ["completed"], "label": "single run",
             "work_seconds_median": 2.0, "input_rows": 10, "input_ratio": 1.0, "time_ratio": 1.0,
             "scaling_exponent": None},
            {"fraction": 1.0, "completed": 0, "runs": 1, "statuses": ["failed"], "label": None,
             "work_seconds_median": None, "input_rows": None, "input_ratio": None, "time_ratio": None,
             "scaling_exponent": None}]}}})
    write(run / "benchmarks/probe/manifest.json", {"run_id": RUN_ID})
    write(run / "benchmarks/copied/manifest.json", {"run_id": "20250101T000000Z-bbbbbbbbbbbb"})
    write(run / "benchmarks/copied/report.json", {"benchmark": "copied", "steps": {}})
    other = root / "runs/20250101T000000Z-bbbbbbbbbbbb/samples/fraction-0.2-seed-1/splits/p-k5-5/selections"
    selection(other / "als.json", "select-grid", "als", "als-r9", {"rank": 9})
    return serving


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.serving = record_tree(self.root)

    def test_headline_is_the_standard_test_view_labelled_from_the_records(self):
        evidence = load_evidence(self.serving)
        self.assertEqual(evidence["headline_label"],
                         "full category (all users), purpose: exploratory, test window, split refit-x")
        self.assertEqual(sorted(evidence["views"]), ["test", "test-h180d"])
        labels = [row["model_label"] for row in evidence["views"]["test"]["rows"]]
        self.assertNotIn("foreign", labels)
        self.assertTrue(any("foreign" in problem for problem in evidence["problems"]))
        self.assertEqual(evidence["definitions"]["k"], 10)
        self.assertEqual(evidence["retention"]["split"]["retained_fraction"], 0.8)
        self.assertEqual(evidence["retention"]["serving_coverage"], {"users": 9, "rows": 27})

    def test_development_selection_is_matched_on_params_and_kept_apart(self):
        evidence = load_evidence(self.serving)
        marks = evidence["selected"]
        self.assertEqual(marks["popularity"]["matches"], ["pop-180"])
        self.assertEqual(marks["als"]["matches"], ["als-r5-u2"])  # label differs, params are a subset
        self.assertEqual(marks["bias"], {"status": "no selection record"})
        self.assertNotIn("oracle-popularity", marks)
        self.assertEqual(sorted(s["model"] for s in evidence["selections"]), ["als", "popularity"])
        self.assertTrue(all(s["label"].startswith("development sample (20% of users)") for s in evidence["selections"]))
        self.assertTrue(any("misfiled" in problem for problem in evidence["problems"]))
        self.assertFalse(any("als-r9" == s["chosen"] for s in evidence["selections"]))  # the other run

        table = tables.quality_table(evidence["views"]["test"]["rows"], marks)
        self.assertEqual(table[-1]["selection"], "reference, not a system")
        by_label = {row["model_label"]: row["selection"] for row in table}
        self.assertEqual((by_label["pop-180"], by_label["pop-all"], by_label["bias-10"]),
                         ("selected on development", "not selected", "no selection record"))

    def test_ambiguous_or_unmatched_selection_is_stated(self):
        selections = self.serving.parents[4] / "fraction-0.2-seed-1/splits/p-k5-5/selections"
        selection(selections / "als.json", "select-grid", "als", "als-r7", {"rank": 7})
        self.assertEqual(load_evidence(self.serving)["selected"]["als"]["status"],
                         "no test record matches the selection")
        selection(selections / "als-again.json", "select-grid", "als", "als-r5", {"rank": 5})
        self.assertEqual(load_evidence(self.serving)["selected"]["als"]["status"],
                         "ambiguous: several selection records")

    def test_benchmarks_of_the_run_only_with_unmeasured_points_listed(self):
        evidence = load_evidence(self.serving)
        self.assertEqual([b["label"] for b in evidence["benchmarks"]], ["b1"])
        benchmark = evidence["benchmarks"][0]
        self.assertEqual([p["fraction"] for p in tables.scaling_points(benchmark)], [0.5])
        self.assertEqual(tables.unmeasured_points(benchmark), [{"step": "split", "fraction": 1.0, "statuses": "failed"}])
        self.assertIn("Benchmark probe: no report.json, not shown", evidence["problems"])
        self.assertTrue(any("copied" in p for p in evidence["problems"]))

    def test_unrecorded_user_fraction_is_labelled(self):
        from amazon_recommender.dashboard.evidence import population_label

        self.assertEqual(population_label({"purpose": "exploratory"}),
                         "user fraction unrecorded, purpose: exploratory")

    def test_untraceable_sources_are_refused(self):
        with self.assertRaisesRegex(EvidenceError, "not found"):
            load_evidence(self.root / "missing")
        write(self.serving.parents[5] / "run.json", {"run_id": "20250101T000000Z-bbbbbbbbbbbb"})
        with self.assertRaisesRegex(EvidenceError, "does not belong"):
            load_evidence(self.serving)


SMOKE = """
import sys
from streamlit.testing.v1 import AppTest
sys.argv = ["app.py", "--config", sys.argv[1]]
app = AppTest.from_file({app!r}, default_timeout=60)
app.run()
assert not app.exception, app.exception
print("ERRORS", [element.value for element in app.error])
print("PYSPARK", any(name.startswith("pyspark") for name in sys.modules))
"""


def run_app(uri: str | None) -> str:
    env = {key: value for key, value in os.environ.items() if key != "MONGODB_URI"}
    if uri:
        env["MONGODB_URI"] = uri
    result = subprocess.run([sys.executable, "-c", SMOKE.format(app=str(APP)), str(ROOT / "configs/local.toml")],
                            capture_output=True, text=True, env=env, timeout=120)
    if result.returncode:
        raise AssertionError(result.stderr[-2000:])
    return result.stdout


class AppSmokeTests(unittest.TestCase):
    def test_unreachable_database_is_reported_without_spark(self):
        output = run_app("mongodb://127.0.0.1:9/?connectTimeoutMS=500")
        self.assertIn("MongoDB is unavailable", output)
        self.assertIn("PYSPARK False", output)

    def test_missing_uri_is_reported(self):
        self.assertIn("Cannot configure the dashboard", run_app(None))

    def test_cli_and_dashboard_modules_do_not_import_spark(self):
        code = ("import sys, amazon_recommender.cli, amazon_recommender.dashboard.lookup, "
                "amazon_recommender.dashboard.evidence, amazon_recommender.dashboard.tables; "
                "print(any(name.startswith('pyspark') for name in sys.modules))")
        self.assertEqual(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                        check=True).stdout.strip(), "False")


@unittest.skipUnless(os.environ.get("RUN_MONGO_TESTS") == "1", "set RUN_MONGO_TESTS=1 and MONGODB_URI (make check-mongo)")
class MongoLookupTests(unittest.TestCase):
    def setUp(self):
        from pymongo import MongoClient

        self.client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=5000)
        self.addCleanup(self.client.close)

    def test_lookup_on_a_temporary_publication(self):
        name = f"test_dashboard_{uuid.uuid4().hex}"
        try:
            database = self.client[name]
            database[ACTIVE_COLLECTION].insert_one(publication("t1", k=2))
            database["recommendations_t1"].insert_many([user("U1", [(2, "b", 1.0), (1, "a", 2.0)]),
                                                        user("U2", [(1, "a", 2.0)], reason="catalog_exhausted")])
            database["items_t1"].insert_many([
                {"_id": "a", "title": "Alpha"},
                {"_id": "b", "title": None, "main_category": "Prime Video", "directors": ["Rian Johnson"],
                 "starring": ["Daniel Craig"], "studio": None, "image": "https://img/b.jpg", "rating_number": 5}])
            result = lookup_user(database, "U1")
            self.assertEqual([(row["title"], row["title_missing"], row["derived"]) for row in result.rows],
                             [("Alpha", False, False), ("Prime Video · dir. Rian Johnson · Daniel Craig", True, True)])
            self.assertEqual(result.rows[1]["image"], "https://img/b.jpg")
            self.assertTrue(lookup_user(database, "U2").short_list)
            self.assertEqual(lookup_user(database, "nobody").status, UNKNOWN_USER)
        finally:
            self.client.drop_database(name)

    def test_lookup_on_the_active_publication_is_read_only(self):
        database = self.client[os.environ.get("MONGO_TEST_DATABASE", "amazon_recommender")]
        active = database[ACTIVE_COLLECTION].find_one({"_id": ACTIVE_ID})
        if active is None:
            self.skipTest("no active publication in this database")
        for user_type in ("known", "new"):
            document = database[active["recommendations"]].find_one({"user_type": user_type})
            if document is None:
                continue
            result = lookup_user(database, document["_id"])
            self.assertEqual(result.status, OK)
            self.assertEqual([row["rank"] for row in result.rows], list(range(1, len(document["items"]) + 1)))
            self.assertEqual([row["parent_asin"] for row in result.rows],
                             [item["parent_asin"] for item in sorted(document["items"], key=lambda i: i["rank"])])
            self.assertTrue(all(row["title"] for row in result.rows))
        self.assertEqual(lookup_user(database, "no-such-user-" + uuid.uuid4().hex).status, UNKNOWN_USER)
        self.assertEqual(database[ACTIVE_COLLECTION].find_one({"_id": ACTIVE_ID}), active)
