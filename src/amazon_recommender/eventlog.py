"""Summaries of Spark event logs, the evidence for the scalability analysis (P6).

Every stage records its Spark `application_id` in `stage.json` (`resources`), and Spark writes
that application's event log to `<storage>/spark-events/<application_id>`, or with a codec
suffix when `spark.eventLog.compress` is on, or `.inprogress` while running or after a crash.
`find_event_log` resolves a log by application id only, never by modification time.

The summary is per application (one pipeline stage): task totals (run time, CPU, GC, input and
output bytes, shuffle read/write, memory and disk spill), the peak JVM heap and execution memory
reported per stage, and per Spark stage the task-duration skew (slowest task / median task).
"""

import gzip
import json
from pathlib import Path
import statistics

CODECS = ("", ".lz4", ".zstd", ".snappy", ".lzf", ".gz")
TOP_STAGES = 5
SKEW_MIN_TASK_MS = 1000  # skew of stages whose slowest task is shorter cannot cost a second


def find_event_log(directory: Path, application_id: str) -> Path | None:
    """The event log of one application; a completed log wins over an `.inprogress` one."""
    for suffix in (*CODECS, *(codec + ".inprogress" for codec in CODECS)):
        path = Path(directory) / f"{application_id}{suffix}"
        if path.is_file():
            return path
    return None


def _lines(path: Path):
    name = path.name.removesuffix(".inprogress")
    if name.endswith(".gz"):
        opener = gzip.open
    elif name.endswith((".lz4", ".zstd", ".snappy", ".lzf")):
        # Spark's own framing of these codecs needs extra libraries; keep the default (uncompressed).
        raise ValueError(f"Compressed event log not supported: {path}; set spark.eventLog.compress=false")
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:  # the last line of an .inprogress log may be cut
                    if not path.name.endswith(".inprogress"):
                        raise


def summarize_event_log(path: Path) -> dict:
    """Totals, memory peaks and per-stage skew of one application's event log."""
    path = Path(path)
    totals = dict.fromkeys(("tasks", "failed_tasks", "executor_run_ms", "executor_cpu_ms", "gc_ms",
                            "input_bytes", "output_bytes", "shuffle_read_bytes", "shuffle_write_bytes",
                            "memory_spilled_bytes", "disk_spilled_bytes"), 0)
    stages, durations = {}, {}
    peaks = {"jvm_heap_bytes": 0, "execution_memory_bytes": 0, "task_peak_execution_memory_bytes": 0}
    app = {"start_ms": None, "end_ms": None, "jobs": 0, "block_manager_max_memory_bytes": None, "cores": None}
    for event in _lines(path):
        kind = event.get("Event")
        if kind == "SparkListenerTaskEnd":
            info, metrics = event["Task Info"], event.get("Task Metrics") or {}
            totals["tasks"] += 1
            totals["failed_tasks"] += int(bool(info.get("Failed")))
            key = (event["Stage ID"], event["Stage Attempt ID"])
            durations.setdefault(key, []).append(info["Finish Time"] - info["Launch Time"])
            if not metrics:
                continue
            totals["executor_run_ms"] += metrics.get("Executor Run Time", 0)
            totals["executor_cpu_ms"] += metrics.get("Executor CPU Time", 0) / 1e6
            totals["gc_ms"] += metrics.get("JVM GC Time", 0)
            totals["input_bytes"] += metrics.get("Input Metrics", {}).get("Bytes Read", 0)
            totals["output_bytes"] += metrics.get("Output Metrics", {}).get("Bytes Written", 0)
            read = metrics.get("Shuffle Read Metrics", {})
            totals["shuffle_read_bytes"] += read.get("Local Bytes Read", 0) + read.get("Remote Bytes Read", 0)
            totals["shuffle_write_bytes"] += metrics.get("Shuffle Write Metrics", {}).get("Shuffle Bytes Written", 0)
            totals["memory_spilled_bytes"] += metrics.get("Memory Bytes Spilled", 0)
            totals["disk_spilled_bytes"] += metrics.get("Disk Bytes Spilled", 0)
            peaks["task_peak_execution_memory_bytes"] = max(peaks["task_peak_execution_memory_bytes"],
                                                            metrics.get("Peak Execution Memory", 0))
        elif kind == "SparkListenerStageCompleted":
            info = event["Stage Info"]
            stages[(info["Stage ID"], info["Stage Attempt ID"])] = {
                "stage_id": info["Stage ID"],
                "name": info.get("Stage Name"),
                "tasks": info.get("Number of Tasks"),
                "seconds": (info["Completion Time"] - info["Submission Time"]) / 1000
                if info.get("Completion Time") and info.get("Submission Time") else None,
                "failed": "Failure Reason" in info,
            }
        elif kind == "SparkListenerStageExecutorMetrics":
            metrics = event.get("Executor Metrics", {})
            peaks["jvm_heap_bytes"] = max(peaks["jvm_heap_bytes"], metrics.get("JVMHeapMemory", 0))
            peaks["execution_memory_bytes"] = max(peaks["execution_memory_bytes"],
                                                  metrics.get("OnHeapExecutionMemory", 0))
        elif kind == "SparkListenerJobStart":
            app["jobs"] += 1
        elif kind == "SparkListenerApplicationStart":
            app["start_ms"] = event["Timestamp"]
        elif kind == "SparkListenerApplicationEnd":
            app["end_ms"] = event["Timestamp"]
        elif kind == "SparkListenerBlockManagerAdded":
            app["block_manager_max_memory_bytes"] = event.get("Maximum Memory")
        elif kind == "SparkListenerExecutorAdded":
            app["cores"] = event.get("Executor Info", {}).get("Total Cores")

    for key, values in durations.items():
        stage = stages.setdefault(key, {"stage_id": key[0], "name": None, "tasks": len(values), "seconds": None,
                                        "failed": None})
        median = statistics.median(values)
        stage["task_ms_median"] = median
        stage["task_ms_max"] = max(values)
        # Slowest task over the median task; 1 means balanced. Undefined for one task or zero median.
        stage["skew"] = max(values) / median if len(values) > 1 and median > 0 else None
    ranked = sorted(stages.values(), key=lambda stage: stage["seconds"] or 0, reverse=True)
    skews = [stage["skew"] for stage in stages.values() if stage.get("skew") is not None]
    costly = [stage["skew"] for stage in stages.values()
              if stage.get("skew") is not None and stage["task_ms_max"] >= SKEW_MIN_TASK_MS]
    return {
        "path": str(path),
        "complete": app["end_ms"] is not None,
        "application_seconds": (app["end_ms"] - app["start_ms"]) / 1000 if app["end_ms"] and app["start_ms"] else None,
        "jobs": app["jobs"],
        "spark_stages": len(stages),
        "cores": app["cores"],
        "block_manager_max_memory_bytes": app["block_manager_max_memory_bytes"],
        **{key: round(value, 3) if isinstance(value, float) else value for key, value in totals.items()},
        "gc_fraction_of_run": totals["gc_ms"] / totals["executor_run_ms"] if totals["executor_run_ms"] else None,
        "peaks": peaks,
        # Over Spark stages whose slowest task takes at least a second; sub-second stages are noise.
        "max_task_skew": max(costly) if costly else None,
        "max_task_skew_all_stages": max(skews) if skews else None,
        "top_stages": ranked[:TOP_STAGES],
    }
