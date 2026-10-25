"""Small, explicit experiment settings; these do not execute a pipeline."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ExperimentConfig:
    purpose: str
    user_fraction: float
    split: tuple[float, float, float]
    user_min_reviews: int
    item_min_reviews: int
    relevant_rating: float


@dataclass(frozen=True)
class ModelConfig:
    rank: int
    regularization: float
    iterations: int
    bias_regularization: float
    bias_max_passes: int
    bias_tolerance: float
    popularity_smoothing: float
    popularity_window_days: int | str  # "all" or a positive number of days
    recommendation_count: int
    num_user_blocks: int  # Spark ALS blocks; the factors depend on them (reproducible within tolerance)
    num_item_blocks: int


def _positive(value, name: str, *, integer: bool = False) -> None:
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")


def parse_experiment(raw: dict) -> ExperimentConfig:
    if raw["purpose"] not in ("exploratory", "final"):
        raise ValueError("experiment.purpose must be exploratory or final")
    _positive(raw["user_fraction"], "experiment.user_fraction")
    if raw["user_fraction"] > 1:
        raise ValueError("experiment.user_fraction must be in (0,1]")
    split = raw["split"]
    if (
        not isinstance(split, list)
        or len(split) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in split)
        or not math.isclose(sum(split), 1.0, abs_tol=1e-9, rel_tol=0)
        or split[0] <= 0
    ):
        raise ValueError("experiment.split must be [train, validation, test], sum to 1, and have train > 0")
    for key in ("user_min_reviews", "item_min_reviews"):
        _positive(raw[key], f"experiment.{key}", integer=True)
    _positive(raw["relevant_rating"], "experiment.relevant_rating")
    if not 1 <= raw["relevant_rating"] <= 5:
        raise ValueError("experiment.relevant_rating must be in [1,5]")
    if raw["purpose"] == "final" and (
        tuple(split) != (0.8, 0.1, 0.1)
        or raw["user_fraction"] != 1
        or raw["user_min_reviews"] != 5
        or raw["item_min_reviews"] != 5
        or raw["relevant_rating"] != 4
    ):
        raise ValueError("final evaluation requires all users, split [0.8,0.1,0.1], k=5, relevance=4")
    return ExperimentConfig(**{**raw, "split": tuple(split)})


def parse_model(raw: dict) -> ModelConfig:
    integers = {"rank", "iterations", "bias_max_passes", "recommendation_count", "num_user_blocks", "num_item_blocks"}
    for key, value in raw.items():
        if key == "popularity_window_days" and value == "all":
            continue
        _positive(value, f"model.{key}", integer=key in integers | {"popularity_window_days"})
    return ModelConfig(**raw)
