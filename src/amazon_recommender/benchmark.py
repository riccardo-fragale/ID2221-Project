"""Fixed-resource input-scaling benchmark (P6.1) and its report (P6.2).

Contract (approved 2026-10-25): nested user samples (one seed, fractions such as 0.25/0.5/1), all
split with the cutoffs of one reference split, the four frozen model settings (given as the
`stage.json` of already trained models; no hyperparameter search here), one Spark profile.

Measurement type: "stage from materialized input". ETL is not re-run; its timing comes once from
the run's `etl/stage.json`, at full input only, and is not a scaling point. For every repetition
and fraction the runner builds a fresh mini-run `rep-<r>/fraction-<f>/` (a copy of `run.json` and a
symlink to the run's `etl/`) and runs sample, split, then train and recommend for each model, each
as a separate CLI process, so every stage starts its own JVM and nothing cached is reused. A
completed benchmark directory is never overwritten, and an interrupted one (`<label>.partial`) is
renamed to `<label>.aborted-<UTC time>` by the next attempt, so its rows stay visible.

Before each fraction the ETL Parquet files are read once (timed), so every scale starts with the
same warm page cache; the runner does not drop caches. Before each stage it records the host load
and free memory. A failed or timed-out stage is a visible row; the stages that depend on it are
rows with status `skipped`; the next fraction still runs. A timed-out stage's process group (the
CLI process and its Spark JVM) is terminated, then killed after a grace period. MongoDB writes are a `not_measured` row
until publication exists (P7).

Output `<run>/benchmarks/<label>/`: `manifest.json` (contract, resources, host, code identity),
`measurements.jsonl` (one row per stage run, appended as it happens), `scales.jsonl` (sizes and
k-core retention per fraction) and `measurements.csv`. `benchmark-report` derives `report.json`,
`report.csv` and `scaling.svg` from those files only.
"""

from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time

from amazon_recommender.artifacts import code_identity
from amazon_recommender.config import load_config
from amazon_recommender.eventlog import find_event_log, summarize_event_log
from amazon_recommender.sample import sample_label

MODEL_OVERRIDES = {  # model param -> (train CLI option, ModelConfig field)
    "rank": ("--rank", "rank"),
    "regularization": ("--regularization", "regularization"),
    "iterations": ("--iterations", "iterations"),
    "window_days": ("--popularity-window-days", "popularity_window_days"),
}
CSV_COLUMNS = ("repetition", "fraction", "order", "stage", "model", "status", "work_seconds", "spark_start_seconds",
               "internal_seconds", "internal_seconds_without_verification", "wall_seconds", "input_rows", "output_rows",
               "output_bytes", "materialized", "shuffle_read_bytes", "shuffle_write_bytes", "spilled_bytes",
               "gc_ms", "executor_run_ms", "max_task_skew", "peak_jvm_heap_bytes", "loadavg_1m_before",
               "mem_available_bytes_before", "application_id", "error")
STDERR_TAIL = 4000
KILL_GRACE_SECONDS = 30


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _du(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file() and not file.is_symlink())


def _host_state() -> dict:
    available = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {"loadavg_1m": os.getloadavg()[0], "mem_available_bytes": available}


def _host() -> dict:
    model, total = None, None
    try:
        model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), None)
        total = next((int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                      if line.startswith("MemTotal:")), None)
    except OSError:
        pass
    return {"logical_cpus": os.cpu_count(), "cpu_model": model, "mem_total_bytes": total,
            "platform": platform.platform(), "python": sys.version.split()[0]}


def _code() -> dict:
    """Revision plus a hash of the uncommitted changes, so a dirty tree is identifiable."""
    identity = code_identity()
    root = Path(__file__).resolve().parents[2]
    try:
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD"], capture_output=True, check=True,
                              timeout=30).stdout
        untracked = subprocess.run(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
                                   capture_output=True, text=True, check=True, timeout=30).stdout.split()
        digest = hashlib.sha256(diff)
        for name in untracked:
            digest.update(name.encode() + b"\0" + (root / name).read_bytes())
        identity |= {"uncommitted_sha256": digest.hexdigest(), "untracked_files": untracked}
    except (OSError, subprocess.SubprocessError):
        identity |= {"uncommitted_sha256": None, "untracked_files": None}
    return identity


def _spark_resources(resources: dict) -> dict:
    """The resources that must match across scales; the application id names a log, not a resource."""
    return {key: value for key, value in resources.items() if key != "application_id"}


def frozen_model(stage_path: Path) -> dict:
    """Model name, parameters and train CLI options of an already trained model (frozen settings)."""
    stage = json.loads(Path(stage_path).read_text())
    model = stage["model"]
    options, overrides = [], {}
    for param, (option, field) in MODEL_OVERRIDES.items():
        if param in model["params"]:
            options += [option, str(model["params"][param])]
            overrides[field] = model["params"][param]
    return {"stage_path": str(Path(stage_path).resolve()), "stage_sha256": _sha256(stage_path),
            "model": model["model"], "params": model["params"], "train_options": options, "overrides": overrides}


def check_frozen(config, models: list[dict]) -> None:
    """Refuse before running anything if the config cannot reproduce a frozen model's parameters."""
    from dataclasses import asdict, replace

    from amazon_recommender.config.experiment import parse_model
    from amazon_recommender.models import create_model

    for frozen in models:
        model_config = replace(config, model=parse_model({**asdict(config.model), **frozen["overrides"]})).model
        params = create_model(frozen["model"], model_config, seed=config.seed).params
        if params != frozen["params"]:
            raise ValueError(f"The config does not reproduce the frozen {frozen['model']} settings: "
                             f"{params} != {frozen['params']}")


class Runner:
    """Runs the stages of one benchmark and appends one row per stage run."""

    def __init__(self, directory: Path, config_path: Path, config, *, timeout: float, python: str):
        self.directory, self.config_path, self.config = directory, config_path, config
        self.timeout, self.python = timeout, python
        self.events = config.storage_root / "spark-events"
        self.order = 0
        self.resources = None
        self.resources_identical = True

    def row(self, repetition: int, fraction: float, stage: str, model: str | None = None, **fields) -> dict:
        self.order += 1
        row = {"repetition": repetition, "fraction": fraction, "order": self.order, "stage": stage, "model": model,
               "status": None, "wall_seconds": None, "work_seconds": None, "spark_start_seconds": None,
               "internal_seconds": None, "internal_seconds_without_verification": None, "timings": None, "input_rows": None,
               "output_rows": None, "output_bytes": None, "materialized": None, "application_id": None,
               "resources_match": None, "event_log": None, "host_before": None, "error": None, "note": None,
               **fields}
        with (self.directory / "measurements.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def run(self, arguments: list[str]) -> tuple[str, float, Path | None, str | None, dict]:
        """One CLI stage in a new process: (status, wall seconds, stage.json path, error, host state)."""
        host = _host_state()
        command = [self.python, "-m", "amazon_recommender", *arguments]
        started = time.monotonic()
        # A new session makes the CLI process and its Spark JVM one process group, killed together.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            return ("timeout", round(time.monotonic() - started, 3), None,
                    f"no result after {self.timeout} s; process group killed", host)
        wall = round(time.monotonic() - started, 3)
        if process.returncode != 0:
            return "failed", wall, None, (stderr or stdout)[-STDERR_TAIL:], host
        return "completed", wall, Path(stdout.strip().splitlines()[-1]), None, host

    def measured(self, stage_path: Path) -> dict:
        """Internal timings, Spark resources check and event-log summary of a completed stage."""
        stage = json.loads(stage_path.read_text())
        timings = stage["timings_seconds"]
        resources = _spark_resources(stage["resources"])
        if self.resources is None:
            self.resources = resources
        match = resources == self.resources
        self.resources_identical &= match
        application = stage["resources"].get("application_id")
        log = find_event_log(self.events, application) if application else None
        try:
            summary = summarize_event_log(log) if log else {"error": "event log not found"}
        except (OSError, ValueError) as exc:
            summary = {"error": str(exc)}
        without_verification = round(timings["total"] - timings.get("verification_counts", 0), 3)
        start = timings.get("spark_start")  # absent in stages written before it was timed
        return {"internal_seconds": timings["total"], "internal_seconds_without_verification": without_verification,
                "spark_start_seconds": start,
                "work_seconds": round(without_verification - start, 3) if start is not None else None,
                "timings": timings, "application_id": application, "resources_match": match, "event_log": summary}


def _kill_group(process: subprocess.Popen) -> None:
    """SIGTERM the stage's process group, then SIGKILL whatever is left after a grace period."""
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_SECONDS), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.communicate(timeout=wait)
            if sig == signal.SIGTERM:
                os.killpg(process.pid, 0)  # raises if no process of the group survived
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            break


def _prewarm(directory: Path) -> dict:
    """Read every file once so each scale starts with the same warm page cache."""
    started, size = time.monotonic(), 0
    for file in sorted(Path(directory).resolve().rglob("*")):
        if file.is_file():
            with file.open("rb") as handle:
                while chunk := handle.read(1 << 24):
                    size += len(chunk)
    return {"seconds": round(time.monotonic() - started, 3), "bytes": size}


def _prune(directory: Path) -> None:
    """Delete data files, keeping every stage.json, after a scale's sizes have been recorded."""
    (Path(directory) / "etl").unlink()  # the symlink to the run's ETL, never its target
    for path in sorted(Path(directory).rglob("*"), reverse=True):
        if path.is_file() and not path.is_symlink() and path.name != "stage.json" and path.name != "run.json":
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def run_scale(runner: Runner, run_directory: Path, repetition: int, fraction: float, reference: Path,
              models: list[dict], *, keep_outputs: bool) -> dict:
    """All stages of one fraction in one repetition; returns the scale record."""
    mini = runner.directory / f"rep-{repetition}" / f"fraction-{fraction:g}"
    mini.mkdir(parents=True)
    shutil.copy2(run_directory / "run.json", mini / "run.json")
    (mini / "etl").symlink_to(run_directory / "etl", target_is_directory=True)
    scale = {"repetition": repetition, "fraction": fraction, "prewarm": _prewarm(run_directory / "etl/interactions")}
    common = ["--config", str(runner.config_path), "--run", str(mini)]
    sample = sample_label(fraction, runner.config.seed)

    def skip_rest(stage: str, reason: str) -> None:
        for name in ([] if stage == "split" else ["split"]):
            runner.row(repetition, fraction, name, status="skipped", note=reason)
        for frozen in models:
            runner.row(repetition, fraction, "train", frozen["model"], status="skipped", note=reason)
            runner.row(repetition, fraction, "recommend", frozen["model"], status="skipped", note=reason)
        publish()

    def publish() -> None:
        runner.row(repetition, fraction, "publish", status="not_measured",
                   note="MongoDB publication (P7) not implemented yet")

    def unlinked(status: str) -> str | None:
        return "event log not linked: no stage.json" if status == "timeout" else None

    status, wall, path, error, host = runner.run(["sample", *common, "--user-fraction", f"{fraction:g}"])
    if status != "completed":
        runner.row(repetition, fraction, "sample", status=status, wall_seconds=wall, error=error, host_before=host,
                   note=unlinked(status))
        skip_rest("sample", "sample did not complete")
        return scale
    stage = json.loads(path.read_text())
    materialized = fraction != 1
    runner.row(repetition, fraction, "sample", status=status, wall_seconds=wall, host_before=host,
               input_rows=stage["interactions"]["before"], output_rows=stage["interactions"]["after"],
               output_bytes=_du(path.parent / "interactions") if materialized else 0, materialized=materialized,
               note=None if materialized else "fraction 1 copies nothing (points at the ETL output); not a scaling point",
               **runner.measured(path))
    scale |= {"sample_users": stage["users"]["after"], "sample_interactions": stage["interactions"]["after"],
              "achieved_fraction": stage["users"]["achieved_fraction"]}

    status, wall, path, error, host = runner.run(["split", *common, "--sample", sample, "--cutoffs-from", str(reference)])
    if status != "completed":
        runner.row(repetition, fraction, "split", status=status, wall_seconds=wall, error=error, host_before=host,
                   note=unlinked(status))
        skip_rest("split", "split did not complete")
        return scale
    stage = json.loads(path.read_text())
    split, split_directory = path.parent.name, path.parent
    train_rows = stage["interactions"]["train"]
    runner.row(repetition, fraction, "split", status=status, wall_seconds=wall, host_before=host,
               input_rows=sum(stage["before_filtering"]["interactions"].values()),
               output_rows=sum(stage["interactions"].values()), output_bytes=_du(split_directory), materialized=True,
               **runner.measured(path))
    scale |= {"split": split, "cutoffs_ms": stage["cutoffs_ms"],
              "before_filtering": stage["before_filtering"]["interactions"],
              "after_filtering": stage["interactions"],
              "k_core_passes": stage["k_core"]["converged_after"],
              "k_core_retained_fraction": stage["k_core"]["retained_fraction"],
              "train_users": stage["mappings"]["users"]["count"], "train_items": stage["mappings"]["items"]["count"]}

    for frozen in models:
        name = frozen["model"]
        status, wall, path, error, host = runner.run(
            ["train", *common, "--sample", sample, "--split", split, "--model", name, *frozen["train_options"]])
        if status == "completed":
            trained = json.loads(path.read_text())["model"]["params"]
            if trained != frozen["params"]:
                status, error = "failed", f"trained parameters differ from the frozen ones: {trained}"
        if status != "completed":
            runner.row(repetition, fraction, "train", name, status=status, wall_seconds=wall, error=error,
                       host_before=host, note=unlinked(status), **(runner.measured(path) if path else {}))
            runner.row(repetition, fraction, "recommend", name, status="skipped", note="train did not complete")
            continue
        label = path.parent.name
        runner.row(repetition, fraction, "train", name, status=status, wall_seconds=wall, host_before=host,
                   input_rows=train_rows, output_bytes=_du(path.parent / "model"), materialized=True,
                   **runner.measured(path))
        status, wall, path, error, host = runner.run(
            ["recommend", *common, "--sample", sample, "--split", split, "--model-label", label])
        if status != "completed":
            runner.row(repetition, fraction, "recommend", name, status=status, wall_seconds=wall, error=error,
                       host_before=host, note=unlinked(status))
            continue
        stage = json.loads(path.read_text())
        runner.row(repetition, fraction, "recommend", name, status=status, wall_seconds=wall, host_before=host,
                   input_rows=train_rows, output_rows=stage["rows"], output_bytes=_du(path.parent / "recommendations"),
                   materialized=True, note=f"{stage['users']} users, top-{stage['k']}, {stage['short_lists']} short lists",
                   **runner.measured(path))
    publish()
    if not keep_outputs:
        _prune(mini)
    return scale


def run_benchmark(config_path: Path, run_directory: Path, *, label: str, reference_split: Path,
                  model_stages: list[Path], fractions: list[float], repetitions: int = 1, cache: str = "warm",
                  stage_timeout: float = 3600, keep_outputs: bool = True, python: str = sys.executable) -> Path:
    """Run the fixed-resource scaling benchmark of one run. Starts one Spark process per stage."""
    config_path = Path(config_path).resolve(strict=True)
    config = load_config(config_path)
    if not config.spark.event_log_enabled:
        raise ValueError("The benchmark needs spark.event_log_enabled = true for its event-log evidence")
    if cache not in ("warm", "cold"):
        raise ValueError("cache must be warm or cold")
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions) or len(set(fractions)) != len(fractions):
        raise ValueError("fractions must be distinct values in (0, 1]")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    run_directory = Path(run_directory).resolve(strict=True)
    if not (run_directory / "etl/stage.json").is_file():
        raise FileNotFoundError(f"ETL stage is not complete for this run: {run_directory}")
    reference = Path(reference_split).resolve(strict=True)
    reference_stage = json.loads(reference.read_text())
    models = [frozen_model(path) for path in model_stages]
    if not models:
        raise ValueError("Give at least one frozen model stage.json")
    check_frozen(config, models)
    final = run_directory / "benchmarks" / label
    partial = final.with_name(f"{label}.partial")
    if final.exists():
        raise FileExistsError(f"Benchmark already completed: {final}")
    if partial.exists():
        # An interrupted attempt: kept aside, neither deleted nor mixed into the new measurements.
        partial.rename(final.with_name(f"{label}.aborted-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"))
    partial.mkdir(parents=True)

    etl_stage = json.loads((run_directory / "etl/stage.json").read_text())
    order = [(repetition, fraction) for repetition in range(1, repetitions + 1) for fraction in sorted(fractions)]
    manifest = {
        "stage": "benchmark",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "run_id": json.loads((run_directory / "run.json").read_text())["run_id"],
        "measurement_type": "stage from materialized input; one new process (JVM) per stage; ETL not re-run",
        "scaling": "fixed-resource input scaling on one host, not worker-count speedup",
        "cache": {"state": cache, "prewarm": "ETL Parquet read once before each fraction" if cache == "warm" else None},
        "order": [{"repetition": repetition, "fraction": fraction} for repetition, fraction in order],
        "order_note": "fractions run in ascending order within a repetition; not randomized",
        "repetitions": repetitions,
        "fractions": sorted(fractions),
        "seed": config.seed,
        "reference_split": {"path": str(reference), "sha256": _sha256(reference),
                            "cutoffs_ms": reference_stage["cutoffs_ms"], "cutoffs_utc": reference_stage["cutoffs_utc"]},
        "models": models,
        "etl_reference": {"note": "full input only, measured once; not a scaling point",
                          "timings_seconds": etl_stage["timings_seconds"], "resources": etl_stage["resources"]},
        "config_path": str(config_path),
        "config": config.as_dict(),
        "spark": config.as_dict()["spark"],
        "stage_timeout_seconds": stage_timeout,
        "keep_outputs": keep_outputs,
        "host": _host(),
        "code": _code(),
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    runner = Runner(partial, config_path, config, timeout=stage_timeout, python=python)
    scales = []
    for repetition, fraction in order:
        scale = run_scale(runner, run_directory, repetition, fraction, reference, models, keep_outputs=keep_outputs)
        scales.append(scale)
        with (partial / "scales.jsonl").open("a") as handle:
            handle.write(json.dumps(scale, allow_nan=False) + "\n")

    rows = [json.loads(line) for line in (partial / "measurements.jsonl").read_text().splitlines()]
    write_csv(partial / "measurements.csv", rows)
    statuses = {row["status"] for row in rows} - {"not_measured"}
    manifest |= {
        "status": "completed" if statuses == {"completed"} else "completed_with_failures",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "rows": {status: sum(row["status"] == status for row in rows) for status in sorted({r["status"] for r in rows})},
        "spark_resources_of_stages": runner.resources,
        "spark_resources_identical": runner.resources_identical,
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final


def write_csv(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            log = row.get("event_log") or {}
            peaks = log.get("peaks") or {}
            flat = {**row, "loadavg_1m_before": (row.get("host_before") or {}).get("loadavg_1m"),
                    "mem_available_bytes_before": (row.get("host_before") or {}).get("mem_available_bytes"),
                    "shuffle_read_bytes": log.get("shuffle_read_bytes"),
                    "shuffle_write_bytes": log.get("shuffle_write_bytes"),
                    "spilled_bytes": (log["memory_spilled_bytes"] + log["disk_spilled_bytes"])
                    if "memory_spilled_bytes" in log else None,
                    "gc_ms": log.get("gc_ms"), "executor_run_ms": log.get("executor_run_ms"),
                    "max_task_skew": log.get("max_task_skew"), "peak_jvm_heap_bytes": peaks.get("jvm_heap_bytes"),
                    "error": (row.get("error") or "").splitlines()[-1] if row.get("error") else None}
            writer.writerow([flat.get(column) for column in CSV_COLUMNS])


# P6.2 ---------------------------------------------------------------------------------------------

def _step(row: dict) -> str:
    return row["stage"] if row["model"] is None else f"{row['stage']}:{row['model']}"


def scaling_report(rows: list[dict], scales: list[dict]) -> dict:
    """Per stage and fraction: measurements, throughput, and time/input ratios to the smallest fraction.

    The headline `time_ratio` uses work seconds: the stage's own total minus the Spark start-up (JVM
    launch and session) and minus the verification counts. `internal_ratio` (the whole stage total)
    and `wall_ratio` (the process, from launch to exit) are secondary. `input_ratio` uses the stage's
    actual input rows (sample: ETL rows; split: sample rows; train and recommend: training rows after
    k-core), not the nominal fraction. Sample reads the whole ETL output at every fraction, so its
    input is constant and it is not input scaling: its output (`output_ratio`) is what grows. A stage
    that did not materialize output (sample at fraction 1) is listed but not used for ratios.
    """
    steps = {}
    for row in rows:
        steps.setdefault(_step(row), {}).setdefault(row["fraction"], []).append(row)
    report = {}
    for step, by_fraction in steps.items():
        entries = []
        for fraction in sorted(by_fraction):
            done = [row for row in by_fraction[fraction] if row["status"] == "completed"]
            work = [row["work_seconds"] for row in done if row["work_seconds"] is not None]
            internal = [row["internal_seconds"] for row in done]
            entry = {
                "fraction": fraction,
                "runs": len(by_fraction[fraction]),
                "completed": len(done),
                "statuses": sorted({row["status"] for row in by_fraction[fraction]}),
                "label": None if not done else ("single run" if len(done) == 1 else f"{len(done)} runs"),
                "work_seconds": work,
                "work_seconds_median": statistics.median(work) if work and len(work) == len(done) else None,
                "work_seconds_range": [min(work), max(work)] if work else None,
                "spark_start_seconds_median": statistics.median(
                    [row["spark_start_seconds"] for row in done]) if work and len(work) == len(done) else None,
                "internal_seconds_median": statistics.median(internal) if internal else None,
                "internal_seconds_without_verification_median": statistics.median(
                    [row["internal_seconds_without_verification"] for row in done]) if done else None,
                "wall_seconds_median": statistics.median([row["wall_seconds"] for row in done]) if done else None,
                "input_rows": done[0]["input_rows"] if done else None,
                "output_rows": done[0]["output_rows"] if done else None,
                "materialized": all(row["materialized"] for row in done) if done else None,
            }
            entry["throughput_rows_per_second"] = (entry["input_rows"] / entry["work_seconds_median"]
                                                   if done and entry["work_seconds_median"] else None)
            entries.append(entry)
        usable = [entry for entry in entries if entry["completed"] and entry["materialized"]]
        base = usable[0] if usable else None
        for entry in entries:
            ok = base is not None and entry in usable and base["input_rows"] and base["work_seconds_median"]
            entry["input_ratio"] = entry["input_rows"] / base["input_rows"] if ok else None
            entry["output_ratio"] = (entry["output_rows"] / base["output_rows"]
                                     if ok and entry["output_rows"] is not None and base["output_rows"] else None)
            entry["time_ratio"] = entry["work_seconds_median"] / base["work_seconds_median"] if ok else None
            entry["internal_ratio"] = entry["internal_seconds_median"] / base["internal_seconds_median"] if ok else None
            entry["wall_ratio"] = entry["wall_seconds_median"] / base["wall_seconds_median"] if ok else None
            # Exponent b of time ~ input^b between the base and this point; 1 is linear.
            entry["scaling_exponent"] = (math.log(entry["time_ratio"]) / math.log(entry["input_ratio"])
                                         if ok and entry["input_ratio"] not in (None, 1) and entry["time_ratio"] else None)
        report[step] = {"base_fraction": base["fraction"] if base else None, "points": entries,
                        "note": "constant input (full ETL scan); output grows with the fraction"
                        if step == "sample" else None}
    retention = [{key: scale.get(key) for key in ("repetition", "fraction", "sample_interactions", "sample_users",
                                                   "before_filtering", "after_filtering", "k_core_passes",
                                                   "k_core_retained_fraction", "train_users", "train_items")}
                 for scale in scales]
    return {"steps": report, "retention": retention}


def scaling_svg(report: dict) -> str:
    """Time ratio against input ratio (both relative to the smallest fraction); the diagonal is linear."""
    series = [(step, [(p["input_ratio"], p["time_ratio"]) for p in data["points"]
                      if p["input_ratio"] is not None and p["time_ratio"] is not None])
              for step, data in report["steps"].items()]
    series = [(step, points) for step, points in series if len(points) > 1]
    top = max([1.0, *(value for _, points in series for point in points for value in point)]) * 1.05
    width, height, margin = 640, 440, 56
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4b5563", "#ca8a04", "#db2777",
              "#65a30d", "#7c3aed"]

    def x(value):
        return margin + (value / top) * (width - 2 * margin)

    def y(value):
        return height - margin - (value / top) * (height - 2 * margin)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 20 * len(series)}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="100%" height="100%" fill="white"/>',
             f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(top)}" y2="{y(0)}" stroke="black"/>',
             f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(0)}" y2="{y(top)}" stroke="black"/>',
             f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(top)}" y2="{y(top)}" stroke="#9ca3af" stroke-dasharray="4 4"/>',
             f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle">input rows / input rows at the smallest '
             f'fraction</text>',
             f'<text x="16" y="{height / 2}" transform="rotate(-90 16 {height / 2})" text-anchor="middle">'
             f'stage time / time at the smallest fraction</text>']
    for tick in range(0, math.floor(top) + 1):
        parts.append(f'<text x="{x(tick)}" y="{y(0) + 16}" text-anchor="middle">{tick}</text>')
        parts.append(f'<text x="{x(0) - 8}" y="{y(tick) + 4}" text-anchor="end">{tick}</text>')
    for index, (step, points) in enumerate(series):
        color = colors[index % len(colors)]
        path = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in sorted(points))
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        parts += [f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="3" fill="{color}"/>' for a, b in points]
        parts.append(f'<text x="{margin}" y="{height + 20 * index + 4}" fill="{color}">{step}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def run_benchmark_report(benchmark_directory: Path) -> Path:
    """Derive report.json, report.csv and scaling.svg from a completed benchmark's saved files only."""
    directory = Path(benchmark_directory).resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] == "running":
        raise ValueError(f"Benchmark is not finished: {directory}")
    rows = [json.loads(line) for line in (directory / "measurements.jsonl").read_text().splitlines()]
    scales = [json.loads(line) for line in (directory / "scales.jsonl").read_text().splitlines()]
    report = scaling_report(rows, scales)
    report = {"benchmark": manifest["label"], "measurement_type": manifest["measurement_type"],
              "scaling": manifest["scaling"], "cache": manifest["cache"], "spark": manifest["spark"],
              "spark_resources_identical": manifest["spark_resources_identical"], **report}
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (directory / "report.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        columns = ("fraction", "completed", "label", "input_rows", "output_rows", "work_seconds_median",
                   "spark_start_seconds_median", "internal_seconds_median",
                   "internal_seconds_without_verification_median", "wall_seconds_median",
                   "throughput_rows_per_second", "input_ratio", "output_ratio", "time_ratio", "internal_ratio",
                   "wall_ratio", "scaling_exponent")
        writer.writerow(("step", *columns))
        for step, data in report["steps"].items():
            for point in data["points"]:
                writer.writerow((step, *(point[column] for column in columns)))
    (directory / "scaling.svg").write_text(scaling_svg(report))
    return directory / "report.json"
