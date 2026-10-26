"""Display rows for the dashboard, built from lookup results and loaded evidence. No Streamlit."""


def _interval(values) -> str:
    return f"[{values[0]:.5f}, {values[1]:.5f}]" if values else ""


def _number(value, digits: int = 5) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _p_value(value) -> str:
    return "" if value is None else (f"{value:.1e}" if value < 1e-3 else f"{value:.3f}")


def system_name(row: dict) -> str:
    params = ", ".join(f"{key}={value}" for key, value in row["params"].items())
    return f"{row['model']} ({params})" if params else row["model"]


def quality_table(rows: list[dict], marks: dict) -> list[dict]:
    """Systems first in descending NDCG@k, then the reference rows; the selection mark comes from P8.2 matching."""
    ordered = sorted(rows, key=lambda row: (row["reference_only"], -(row["ndcg_at_k"] or 0)))
    table = []
    for row in ordered:
        mark = marks.get(row["model"], {})
        if row["reference_only"]:
            selection = "reference, not a system"
        elif mark.get("status") == "selected on development":
            selection = mark["status"] if row["model_label"] in mark["matches"] else "not selected"
        else:
            selection = mark.get("status", "")
        table.append({
            "system": system_name(row), "selection": selection, "users": row["eligible_users"],
            "NDCG@10": _number(row["ndcg_at_k"]), "NDCG 95% CI": _interval(row["ndcg_ci95"]),
            "P@10": _number(row["precision_at_k"]), "hit rate": _number(row["hit_rate"], 4),
            "Recall@100": _number(row["recall_at_100"], 4),
            "users with hit": row["users_with_hit"],
            "expected if random": _number(row["expected_users_with_hit"], 1),
            "p (vs random)": _p_value(row["p_value"]),
            "RMSE (clipped)": _number(row["rmse"], 3), "RMSE (raw)": _number(row["rmse_raw"], 3),
            "approximate": row["approximate"], "model_label": row["model_label"],
        })
    return table


def random_row(reference: dict | None) -> list[dict]:
    if not reference:
        return []
    return [{"system": "random top-k (expectation)", "NDCG@10": _number(reference.get("ndcg_at_k")),
             "P@10": _number(reference.get("precision_at_k")),
             "Recall@100": _number(reference.get("recall_at_100"), 4),
             "expected users with hit": _number(reference.get("expected_users_with_hit"), 1),
             "note": reference.get("note", "")}]


def user_type_table(user_types: dict) -> list[dict]:
    return [{"model_label": label, "user type": user_type, "users": values.get("users"),
             "NDCG@10": _number(values.get("ndcg_at_k")), "users with hit": values.get("users_with_hit"),
             "expected if random": _number(values.get("expected_users_with_hit_if_random"), 1)}
            for label, types in user_types.items() for user_type, values in types.items()]


def selection_table(selection: dict) -> list[dict]:
    return [{"trial": label, "mean validation NDCG@10": _number(value),
             "difference vs reference 95% CI": _interval(selection["difference_ci95"].get(label)),
             "chosen": str(label) == str(selection["chosen"])}  # popularity trials are keyed by window
            for label, value in sorted(selection["mean_ndcg"].items(), key=lambda item: -item[1])]


def scaling_points(benchmark: dict) -> list[dict]:
    """Completed measurements only; other statuses are listed separately by `unmeasured_points`."""
    return [{"step": point["step"], "fraction": point["fraction"], "work seconds (median)": point["work_seconds_median"],
             "input rows": point["input_rows"], "input ratio": point["input_ratio"], "time ratio": point["time_ratio"],
             "scaling exponent": point["scaling_exponent"], "runs": point["runs"], "label": point["label"]}
            for point in benchmark["points"] if point["completed"]]


def unmeasured_points(benchmark: dict) -> list[dict]:
    return [{"step": point["step"], "fraction": point["fraction"], "statuses": ", ".join(point["statuses"])}
            for point in benchmark["points"] if not point["completed"]]


def recommendation_table(rows: list[dict]) -> list[dict]:
    def title(row):
        if not row["title_missing"]:
            return row["title"]
        return row["title"] + (" (derived: no title in metadata)" if row["derived"] else " (no title in metadata)")

    return [{"rank": row["rank"], "image": row["image"], "title": title(row), "parent_asin": row["parent_asin"],
             "score": row["score"], "link": row["link"]} for row in rows]
