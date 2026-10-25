"""Benchmark runner, event-log summaries and the scaling report. Spark tests run only with RUN_SPARK_TESTS=1."""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from amazon_recommender.benchmark import scaling_report, scaling_svg
from amazon_recommender.eventlog import find_event_log, summarize_event_log


ROOT = Path(__file__).resolve().parents[1]
T0 = 1_577_836_800_000  # 2020-01-01 UTC
SPARK = os.environ.get("RUN_SPARK_TESTS") == "1"


def task(stage, launch, finish, *, gc=0, run=0, shuffle_read=0, shuffle_write=0, spill=0, peak=0, failed=False):
    return {"Event": "SparkListenerTaskEnd", "Stage ID": stage, "Stage Attempt ID": 0,
            "Task Info": {"Launch Time": launch, "Finish Time": finish, "Failed": failed},
            "Task Metrics": {"Executor Run Time": run, "Executor CPU Time": run * 1_000_000, "JVM GC Time": gc,
                             "Peak Execution Memory": peak, "Memory Bytes Spilled": spill, "Disk Bytes Spilled": 0,
                             "Input Metrics": {"Bytes Read": 100}, "Output Metrics": {"Bytes Written": 10},
                             "Shuffle Read Metrics": {"Local Bytes Read": shuffle_read, "Remote Bytes Read": 0},
                             "Shuffle Write Metrics": {"Shuffle Bytes Written": shuffle_write}}}


def stage_completed(stage, name, submitted, completed, tasks):
    return {"Event": "SparkListenerStageCompleted",
            "Stage Info": {"Stage ID": stage, "Stage Attempt ID": 0, "Stage Name": name, "Number of Tasks": tasks,
                           "Submission Time": submitted, "Completion Time": completed}}


class EventLogTests(unittest.TestCase):
    def write(self, directory, name, events, *, cut=False):
        path = Path(directory) / name
        text = "".join(json.dumps(event) + "\n" for event in events)
        path.write_text(text + ('{"Event": "SparkListenerTaskE' if cut else ""))
        return path

    def test_summary_hand_values(self):
        events = [
            {"Event": "SparkListenerApplicationStart", "Timestamp": 1_000},
            {"Event": "SparkListenerExecutorAdded", "Executor Info": {"Total Cores": 8}},
            {"Event": "SparkListenerBlockManagerAdded", "Maximum Memory": 5_000},
            {"Event": "SparkListenerJobStart"},
            # Stage 0: task durations 100, 100, 400 -> median 100, max 400, skew 4.
            task(0, 0, 100, run=90, gc=9, shuffle_write=50),
            task(0, 0, 100, run=90, gc=9, shuffle_write=50),
            task(0, 0, 400, run=380, gc=2, spill=7, peak=300),
            stage_completed(0, "groupBy at x.py:1", 1_000, 1_500, 3),
            # Stage 1: one task, skew undefined.
            task(1, 0, 50, run=40, shuffle_read=100),
            stage_completed(1, "parquet at y.py:2", 1_500, 3_500, 1),
            {"Event": "SparkListenerStageExecutorMetrics",
             "Executor Metrics": {"JVMHeapMemory": 2_000, "OnHeapExecutionMemory": 600}},
            {"Event": "SparkListenerApplicationEnd", "Timestamp": 4_000},
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = summarize_event_log(self.write(directory, "local-1", events))
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["application_seconds"], 3.0)
        self.assertEqual((summary["tasks"], summary["jobs"], summary["spark_stages"], summary["cores"]), (4, 1, 2, 8))
        self.assertEqual(summary["executor_run_ms"], 600)
        self.assertEqual(summary["gc_ms"], 20)
        self.assertAlmostEqual(summary["gc_fraction_of_run"], 20 / 600)
        self.assertEqual((summary["input_bytes"], summary["output_bytes"]), (400, 40))
        self.assertEqual((summary["shuffle_read_bytes"], summary["shuffle_write_bytes"]), (100, 100))
        self.assertEqual(summary["memory_spilled_bytes"], 7)
        self.assertEqual(summary["peaks"], {"jvm_heap_bytes": 2_000, "execution_memory_bytes": 600,
                                            "task_peak_execution_memory_bytes": 300})
        # Stage 0's slowest task takes 0.4 s, so its skew of 4 is not a costly one.
        self.assertIsNone(summary["max_task_skew"])
        self.assertEqual(summary["max_task_skew_all_stages"], 4.0)
        # The longest Spark stage comes first.
        self.assertEqual([stage["name"] for stage in summary["top_stages"]], ["parquet at y.py:2", "groupBy at x.py:1"])
        self.assertIsNone(summary["top_stages"][0]["skew"])

    def test_in_progress_log_with_a_cut_last_line(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write(directory, "local-2.inprogress", [task(0, 0, 10, run=10)], cut=True)
            path = find_event_log(Path(directory), "local-2")
            self.assertEqual(path.name, "local-2.inprogress")
            summary = summarize_event_log(path)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["tasks"], 1)
            # A completed log of the same application wins; another application's log is never used.
            self.write(directory, "local-2", [task(0, 0, 10, run=10)])
            self.assertEqual(find_event_log(Path(directory), "local-2").name, "local-2")
            self.assertIsNone(find_event_log(Path(directory), "local-3"))

    def test_compressed_codecs_are_refused_not_misread(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local-4.lz4"
            path.write_bytes(b"\x04\x22\x4d\x18")
            self.assertEqual(find_event_log(Path(directory), "local-4"), path)
            with self.assertRaises(ValueError):
                summarize_event_log(path)


def row(fraction, stage, seconds, rows, *, model=None, status="completed", materialized=True, output=None):
    """`seconds` of work; the stage total adds 3 s of Spark start-up and the process 2 s more."""
    return {"fraction": fraction, "stage": stage, "model": model, "status": status, "work_seconds": seconds,
            "spark_start_seconds": seconds and 3.0, "internal_seconds": seconds and seconds + 3,
            "internal_seconds_without_verification": seconds and seconds + 3, "wall_seconds": seconds and seconds + 5,
            "input_rows": rows, "output_rows": output, "materialized": materialized}


class ScalingReportTests(unittest.TestCase):
    def test_ratios_use_actual_rows_and_work_seconds(self):
        rows = [
            row(0.25, "split", 10.0, 1_000), row(0.5, "split", 20.0, 2_000), row(1, "split", 80.0, 4_000),
            # Sample at fraction 1 copies nothing: listed but excluded from the ratios.
            row(0.25, "sample", 4.0, 9_000, output=2_000), row(0.5, "sample", 5.0, 9_000, output=4_000),
            row(1, "sample", 0.5, 9_000, materialized=False, output=9_000),
            row(0.25, "train", None, 500, model="als", status="failed"), row(0.5, "train", 6.0, 1_000, model="als"),
        ]
        report = scaling_report(rows, [])["steps"]
        split = {point["fraction"]: point for point in report["split"]["points"]}
        self.assertEqual(split[1]["input_ratio"], 4.0)
        self.assertEqual(split[1]["time_ratio"], 8.0)
        self.assertAlmostEqual(split[1]["scaling_exponent"], 1.5)  # 4^1.5 = 8
        self.assertEqual(split[1]["internal_ratio"], 83 / 13)  # start-up flattens the stage-total ratio
        self.assertEqual(split[1]["wall_ratio"], 85 / 15)
        self.assertEqual(split[0.25]["throughput_rows_per_second"], 100.0)
        self.assertEqual(split[0.25]["label"], "single run")
        sample = {point["fraction"]: point for point in report["sample"]["points"]}
        self.assertIsNone(sample[1]["time_ratio"])
        self.assertEqual(sample[0.5]["time_ratio"], 1.25)
        self.assertEqual((sample[0.5]["input_ratio"], sample[0.5]["output_ratio"]), (1.0, 2.0))
        self.assertIsNone(sample[0.5]["scaling_exponent"])  # constant input: not input scaling
        self.assertIsNotNone(report["sample"]["note"])
        # A failed smallest point moves the base to the smallest completed one, and stays listed.
        self.assertEqual(report["train:als"]["base_fraction"], 0.5)
        self.assertEqual(report["train:als"]["points"][0]["statuses"], ["failed"])
        self.assertIn("<polyline", scaling_svg({"steps": report}))


@unittest.skipUnless(SPARK, "set RUN_SPARK_TESTS=1 to run Spark tests")
class BenchmarkRunTests(unittest.TestCase):
    def test_runner_end_to_end_with_a_failed_scale(self):
        from amazon_recommender.artifacts import create_run
        from amazon_recommender.benchmark import run_benchmark, run_benchmark_report
        from amazon_recommender.config import load_config
        from amazon_recommender.etl import run_etl
        from amazon_recommender.sample import run_sample
        from amazon_recommender.split import run_split
        from amazon_recommender.train import run_train

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        # The stages run as CLI processes, so the profile is a file: the local one with fixture
        # thresholds, event logs on and storage in the temporary directory.
        text = (ROOT / "configs/local.toml").read_text()
        for old, new in (('storage_root = "../storage"', 'storage_root = "storage"'),
                         ("event_log_enabled = false", "event_log_enabled = true"),
                         ("user_min_reviews = 5", "user_min_reviews = 2"),
                         ("item_min_reviews = 5", "item_min_reviews = 2")):
            self.assertIn(old, text)
            text = text.replace(old, new)
        config_path = root / "bench.toml"
        config_path.write_text(text)
        config = load_config(config_path)
        reviews, metadata = root / "reviews.ndjson", root / "metadata.ndjson"
        reviews.write_text("".join(
            json.dumps({"user_id": f"u{n:02d}", "parent_asin": f"i{(n + k) % 12:02d}", "rating": 1.0 + (n + 2 * k) % 5,
                        "timestamp": T0 + 1000 * k + n}) + "\n"
            for n in range(40) for k in range(n % 5 + 1)
        ))
        metadata.write_text(json.dumps({"parent_asin": "i00", "title": "Item"}) + "\n")
        run_directory = create_run(config, reviews=reviews, metadata=metadata)
        run_etl(config, run_directory)
        run_sample(config, run_directory)
        sample = "fraction-1-seed-2221"
        reference = run_split(config, run_directory, sample)
        windowed = replace(config, model=replace(config.model, popularity_window_days=1))
        models = [run_train(windowed, run_directory, sample, reference.name, "popularity") / "stage.json",
                  run_train(config, run_directory, sample, reference.name, "bias") / "stage.json"]

        # Fraction 0.01 keeps no fixture user: the sample writes no file and fails to read it back.
        benchmark = run_benchmark(config_path, run_directory, label="fixture", reference_split=reference / "stage.json",
                                  model_stages=models, fractions=[1, 0.01])
        manifest = json.loads((benchmark / "manifest.json").read_text())
        rows = [json.loads(line) for line in (benchmark / "measurements.jsonl").read_text().splitlines()]
        self.assertEqual(manifest["status"], "completed_with_failures")
        self.assertEqual([entry["fraction"] for entry in manifest["order"]], [0.01, 1])
        self.assertTrue(manifest["spark_resources_identical"])
        self.assertIn("uncommitted_sha256", manifest["code"])

        failed = [r for r in rows if r["fraction"] == 0.01]
        self.assertEqual([(r["stage"], r["status"]) for r in failed], [
            ("sample", "failed"), ("split", "skipped"),
            ("train", "skipped"), ("recommend", "skipped"), ("train", "skipped"), ("recommend", "skipped"),
            ("publish", "not_measured")])
        self.assertIn("run_sample", failed[0]["error"])  # the process's traceback is kept

        # The next scale still ran, with fresh processes and its own event logs.
        full = {(r["stage"], r["model"]): r for r in rows if r["fraction"] == 1}
        self.assertEqual({r["status"] for r in full.values()}, {"completed", "not_measured"})
        self.assertFalse(full[("sample", None)]["materialized"])
        for r in full.values():
            if r["status"] == "completed":
                self.assertGreater(r["spark_start_seconds"], 0)
                self.assertAlmostEqual(r["work_seconds"], r["internal_seconds_without_verification"]
                                       - r["spark_start_seconds"], places=2)
        self.assertEqual(full[("split", None)]["input_rows"], 120)  # 40 users with n % 5 + 1 ratings: 8 * (1+...+5)
        for key in (("split", None), ("train", "bias"), ("recommend", "popularity")):
            self.assertGreater(full[key]["event_log"]["tasks"], 0)
            self.assertTrue(full[key]["event_log"]["complete"])
        applications = [r["application_id"] for r in rows if r["application_id"]]
        self.assertEqual(len(applications), len(set(applications)))
        recommend = json.loads(next((benchmark / "rep-1/fraction-1/samples").rglob(
            "recommendations/bias*/stage.json")).read_text())
        self.assertEqual(recommend["users_with_recommendations"], recommend["users"])
        self.assertEqual(full[("recommend", "bias")]["output_rows"], recommend["rows"])
        # Every fraction used the reference cutoffs.
        scales = [json.loads(line) for line in (benchmark / "scales.jsonl").read_text().splitlines()]
        self.assertEqual(scales[1]["cutoffs_ms"], json.loads((reference / "stage.json").read_text())["cutoffs_ms"])

        with self.assertRaises(FileExistsError):
            run_benchmark(config_path, run_directory, label="fixture", reference_split=reference / "stage.json",
                          model_stages=models, fractions=[1])
        # An interrupted attempt is set aside, not deleted, by the next one.
        (benchmark.parent / "again.partial").mkdir()
        (benchmark.parent / "again.partial/measurements.jsonl").write_text("{}\n")
        run_benchmark(config_path, run_directory, label="again", reference_split=reference / "stage.json",
                      model_stages=models, fractions=[0.01])
        aborted = list(benchmark.parent.glob("again.aborted-*"))
        self.assertEqual(len(aborted), 1)
        self.assertEqual((aborted[0] / "measurements.jsonl").read_text(), "{}\n")

        report = json.loads(run_benchmark_report(benchmark).read_text())
        self.assertEqual(report["steps"]["sample"]["points"][0]["statuses"], ["failed"])
        self.assertEqual(report["steps"]["split"]["points"][0]["statuses"], ["skipped"])
        self.assertTrue((benchmark / "scaling.svg").is_file())


class TimeoutTests(unittest.TestCase):
    def test_a_timed_out_stage_is_killed_with_its_child_processes(self):
        import time

        from amazon_recommender.benchmark import Runner
        from amazon_recommender.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Stands in for the CLI: it starts a long-lived child (as PySpark starts its JVM) and hangs.
            stage = root / "stage.sh"
            stage.write_text(f"#!/bin/sh\nsleep 300 &\necho $! > {root / 'child.pid'}\nsleep 300\n")
            stage.chmod(0o755)
            runner = Runner(root, root / "unused.toml", load_config(ROOT / "configs/local.toml"), timeout=2, python=str(stage))
            status, wall, path, error, host = runner.run(["sample"])
            self.assertEqual(status, "timeout")
            self.assertIsNone(path)
            self.assertIn("killed", error)
            self.assertGreaterEqual(wall, 2)
            self.assertIn("loadavg_1m", host)
            child = int((root / "child.pid").read_text())
            for _ in range(50):  # the kill is asynchronous; init reaps the orphaned child
                if not Path(f"/proc/{child}").exists() or "Z" in Path(f"/proc/{child}/stat").read_text().split()[2]:
                    break
                time.sleep(0.1)
            else:
                self.fail("the stage's child process survived the timeout")


class FrozenSettingsTests(unittest.TestCase):
    def test_frozen_settings_the_config_cannot_reproduce_are_refused_before_running(self):
        from amazon_recommender.benchmark import check_frozen
        from amazon_recommender.config import load_config

        config = load_config(ROOT / "configs/local.toml")
        frozen = {"model": "bias", "overrides": {},
                  "params": {"regularization": 3.0, "max_passes": 10, "tolerance": 0.0001}}
        with self.assertRaises(ValueError):
            check_frozen(config, [frozen])


if __name__ == "__main__":
    unittest.main()
