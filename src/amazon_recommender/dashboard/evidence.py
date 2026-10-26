"""Experiment evidence (P8.2) read from the saved records behind the active publication.

Everything starts from the publication's `source` (a serving directory), whose stage.json names
the run, sample and split. The run directory is accepted only when its run.json carries the same
run_id, and every record below is used only when its own run/sample/split fields match the
directory it was found in. Labels (purpose, user fraction, window, view) come from the records;
no number is written here. Missing records become entries in `problems`, never exceptions.
Standard library only: no Spark, no Streamlit.
"""

import json
from pathlib import Path

HEADLINE_VIEW = "test"
SELECTION_MODELS = {"select-popularity-window": "popularity"}  # select-grid records name their model


class EvidenceError(ValueError):
    """The publication's source cannot be traced back to a consistent run directory."""


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def locate(source) -> dict:
    """Serving, split, sample and run directories of a publication, checked against each other."""
    serving = Path(source)
    stage_path = serving / "stage.json"
    if not stage_path.is_file():
        raise EvidenceError(f"Publication source not found on this machine: {stage_path}")
    stage = _read(stage_path)
    split = serving.parent.parent
    sample = split.parent.parent
    run = sample.parent.parent
    if (split.name, sample.name) != (stage.get("split"), stage.get("sample")):
        raise EvidenceError(f"{serving} is not under samples/{stage.get('sample')}/splits/{stage.get('split')}")
    run_json = run / "run.json"
    if not run_json.is_file() or _read(run_json).get("run_id") != stage.get("run_id"):
        raise EvidenceError(f"{run_json} does not belong to run {stage.get('run_id')}")
    return {"serving": serving, "split": split, "sample": sample, "run": run, "stage": stage}


def population_label(sample_stage: dict) -> str:
    fraction = sample_stage.get("user_fraction")
    if not isinstance(fraction, (int, float)):
        scope = "user fraction unrecorded"
    else:
        scope = "full category (all users)" if fraction == 1 else f"development sample ({fraction:.0%} of users)"
    return f"{scope}, purpose: {sample_stage.get('purpose', 'unrecorded')}"


def metric_row(record: dict) -> dict:
    metrics = record["metrics"]
    ranking = metrics.get("ranking") or {}
    signal = metrics.get("signal_vs_random") or {}
    rating = metrics.get("rating") or {}  # null for the oracle reference
    return {
        "model": record["model"], "model_label": record["model_label"], "params": record.get("params", {}),
        "reference_only": bool(record.get("reference_only")), "approximate": bool(record.get("approximate")),
        "eligible_users": metrics.get("cohort", {}).get("eligible_users"),
        "ndcg_at_k": ranking.get("ndcg_at_k"), "ndcg_ci95": ranking.get("ndcg_ci95"),
        "precision_at_k": ranking.get("precision_at_k"), "hit_rate": ranking.get("hit_rate"),
        "recall_at_50": ranking.get("recall_at_50"), "recall_at_100": ranking.get("recall_at_100"),
        "users_with_hit": ranking.get("users_with_hit"),
        "expected_users_with_hit": signal.get("expected_users_with_hit"),
        "p_value": signal.get("p_value_one_sided"),
        "short_lists_at_100": ranking.get("short_lists_at_100"),
        "rmse": rating.get("rmse"), "rmse_raw": rating.get("rmse_raw"),
    }


def evaluation_views(split_dir: Path, expected: dict, problems: list) -> dict:
    """Test-window evaluation records of the published split, grouped by view name."""
    views = {}
    for path in sorted(split_dir.glob("evaluations/*/*/stage.json")):
        record = _read(path)
        if record.get("status") != "completed" or record.get("window") != "test":
            continue
        if any(record.get(key) != expected[key] for key in ("run_id", "sample", "split")):
            problems.append(f"Skipped {path}: run/sample/split differ from the publication")
            continue
        view = record["view"]
        entry = views.setdefault(view["name"], {"view": view, "records": [], "paths": []})
        entry["records"].append(record)
        entry["paths"].append(str(path))
    for entry in views.values():
        entry["rows"] = [metric_row(record) for record in entry["records"]]
        system = next((r for r in entry["records"] if not r.get("reference_only")), entry["records"][0])
        entry["random_reference"] = system["metrics"].get("random_reference")
        entry["user_types"] = {r["model_label"]: r["metrics"]["user_types"]
                               for r in entry["records"] if "user_types" in r["metrics"]}
    return views


def definitions(records: list[dict], problems: list) -> dict:
    """Metric contract as recorded; disagreements between records of one view are reported."""
    if not records:
        return {}
    keys = {"k": lambda m: m.get("k"), "relevant_rating": lambda m: m.get("relevant_rating"),
            "catalog_items": lambda m: m.get("catalog_items"),
            "bootstrap": lambda m: m.get("ranking", {}).get("bootstrap")}
    result = {}
    for name, get in keys.items():
        values = {json.dumps(get(r["metrics"]), sort_keys=True) for r in records}
        if len(values) > 1:
            problems.append(f"Evaluation records disagree on {name}: {sorted(values)}")
        result[name] = get(records[0]["metrics"])
    first = records[0]["metrics"]
    result["cohort"] = first.get("cohort")
    result["random_reference_note"] = first.get("random_reference", {}).get("note")
    result["signal_test"] = first.get("signal_vs_random", {}).get("test")
    return result


def selection_params(record: dict) -> dict:
    if record["stage"] == "select-popularity-window":
        return {"window_days": record["chosen"]}
    return record.get("params", {}).get(record["chosen"], {})


def development_selections(run_dir: Path, problems: list) -> list[dict]:
    """Selection records of every split of the verified run, each checked against its directory."""
    selections = []
    for path in sorted(run_dir.glob("samples/*/splits/*/selections/*.json")):
        record = _read(path)
        split_dir = path.parent.parent
        if (record.get("split"), record.get("sample")) != (split_dir.name, split_dir.parent.parent.name):
            problems.append(f"Skipped {path}: its sample/split fields differ from its directory")
            continue
        sample_stage = _read(split_dir.parent.parent / "stage.json")
        selections.append({
            "path": str(path), "model": record.get("model") or SELECTION_MODELS.get(record["stage"]),
            "label": f"{population_label(sample_stage)}, validation window, split {record['split']}",
            "rule": record.get("rule"), "rule_written": record.get("rule_written"), "users": record.get("users"),
            "chosen": record["chosen"], "chosen_params": selection_params(record),
            "mean_ndcg": record.get("mean_ndcg", {}),
            "difference_ci95": record.get("difference_vs_chosen_ci95") or record.get("difference_vs_best_ci95") or {},
        })
    return selections


def mark_selected(rows: list[dict], selections: list[dict]) -> dict:
    """Per model family: which test row the development selection points to, matched on params."""
    marks = {}
    for model in sorted({row["model"] for row in rows if not row["reference_only"]}):
        chosen = [s for s in selections if s["model"] == model]
        if not chosen:
            marks[model] = {"status": "no selection record"}
            continue
        if len(chosen) > 1:
            marks[model] = {"status": "ambiguous: several selection records", "selections": [s["path"] for s in chosen]}
            continue
        wanted = chosen[0]["chosen_params"]
        matches = [row["model_label"] for row in rows if row["model"] == model
                   and wanted and all(row["params"].get(key) == value for key, value in wanted.items())]
        status = "selected on development" if len(matches) == 1 else (
            "no test record matches the selection" if not matches else "ambiguous: several test records match")
        marks[model] = {"status": status, "matches": matches, "selection": chosen[0]["path"]}
    return marks


def retention(locations: dict) -> dict:
    sample_stage = _read(locations["sample"] / "stage.json")
    split_stage = _read(locations["split"] / "stage.json")
    k_core = split_stage.get("k_core", {})
    return {
        "label": population_label(sample_stage),
        "sample": {"user_fraction": sample_stage.get("user_fraction"), "users": sample_stage.get("users"),
                   "interactions": sample_stage.get("interactions")},
        "split": {"label": split_stage.get("split"), "cutoffs_utc": split_stage.get("cutoffs_utc"),
                  "refit_from": (split_stage.get("policy") or {}).get("refit_from"),
                  "before_filtering": split_stage.get("before_filtering", {}).get("interactions"),
                  "k_core_passes": k_core.get("converged_after"),
                  "retained_interactions": k_core.get("retained_interactions"),
                  "retained_fraction": k_core.get("retained_fraction"),
                  "mapped_users": split_stage.get("mappings", {}).get("users", {}).get("count"),
                  "mapped_items": split_stage.get("mappings", {}).get("items", {}).get("count"),
                  "test_projection": split_stage.get("projection", {}).get("test")},
        "serving_coverage": locations["stage"].get("coverage"),
    }


def benchmarks(run_dir: Path, run_id: str, problems: list) -> list[dict]:
    """Benchmark reports of the verified run; a benchmark without report.json is noted, not shown."""
    results = []
    for directory in sorted(path for path in (run_dir / "benchmarks").glob("*") if path.is_dir()):
        manifest_path, report_path = directory / "manifest.json", directory / "report.json"
        missing = [path.name for path in (manifest_path, report_path) if not path.is_file()]
        if missing:
            problems.append(f"Benchmark {directory.name}: no {' or '.join(missing)}, not shown")
            continue
        manifest = _read(manifest_path)
        if manifest.get("run_id") != run_id:
            problems.append(f"Skipped benchmark {directory.name}: manifest run_id {manifest.get('run_id')}")
            continue
        report = _read(report_path)
        points = [{"step": step, **point} for step, entry in report["steps"].items() for point in entry["points"]]
        figure = directory / "scaling.svg"
        results.append({
            "label": report["benchmark"], "path": str(report_path), "status": manifest.get("status"),
            "measurement_type": report.get("measurement_type"), "scaling": report.get("scaling"),
            "cache": report.get("cache"), "spark": report.get("spark"),
            "spark_resources_identical": report.get("spark_resources_identical"),
            "repetitions": manifest.get("repetitions"), "host": manifest.get("host"),
            "notes": {step: entry["note"] for step, entry in report["steps"].items() if entry.get("note")},
            "points": points, "retention": report.get("retention", []),
            "figure": str(figure) if figure.is_file() else None,
        })
    return results


def load_evidence(source) -> dict:
    """All P8.2 evidence for one publication source; raises EvidenceError only if it cannot be traced."""
    locations = locate(source)
    stage, problems = locations["stage"], []
    views = evaluation_views(locations["split"], stage, problems)
    headline = views.get(HEADLINE_VIEW)
    if headline is None:
        problems.append(f"No '{HEADLINE_VIEW}' evaluation records under {locations['split'] / 'evaluations'}")
    selections = development_selections(locations["run"], problems)
    sample_stage = _read(locations["sample"] / "stage.json")
    return {
        "run_id": stage["run_id"], "source": str(locations["serving"]),
        "headline_label": f"{population_label(sample_stage)}, test window, split {stage['split']}",
        "views": views,
        "definitions": definitions(headline["records"] if headline else [], problems),
        "selected": mark_selected(headline["rows"], selections) if headline else {},
        "selections": selections,
        "retention": retention(locations),
        "benchmarks": benchmarks(locations["run"], stage["run_id"], problems),
        "problems": problems,
    }
