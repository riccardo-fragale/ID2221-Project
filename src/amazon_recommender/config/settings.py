"""Read non-secret settings without importing Spark or connecting to MongoDB."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import tomllib

from .experiment import ExperimentConfig, ModelConfig, parse_experiment, parse_model


@dataclass(frozen=True)
class SparkConfig:
    master: str
    driver_memory: str
    shuffle_partitions: int
    event_log_enabled: bool


@dataclass(frozen=True)
class MongoConfig:
    database: str
    uri_env: str


@dataclass(frozen=True)
class ProjectConfig:
    profile: str
    seed: int
    storage_root: Path
    spark: SparkConfig
    mongo: MongoConfig
    experiment: ExperimentConfig
    model: ModelConfig

    def as_dict(self) -> dict:
        values = asdict(self)
        values["storage_root"] = str(self.storage_root)
        return values

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _exact_keys(values: dict, expected: set[str], section: str) -> None:
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError(f"{section} must contain exactly: {', '.join(sorted(expected))}")


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).resolve()
    with path.open("rb") as source:
        raw = tomllib.load(source)
    _exact_keys(raw, {"profile", "seed", "storage_root", "spark", "mongo", "experiment", "model"}, "config")
    _exact_keys(raw["spark"], set(SparkConfig.__dataclass_fields__), "spark")
    _exact_keys(raw["mongo"], set(MongoConfig.__dataclass_fields__), "mongo")
    _exact_keys(raw["experiment"], set(ExperimentConfig.__dataclass_fields__), "experiment")
    _exact_keys(raw["model"], set(ModelConfig.__dataclass_fields__), "model")
    if raw["profile"] != "local":
        raise ValueError("Only the approved local profile is supported; full-data resources are pending")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**31:
        raise ValueError("seed must be an integer in [0, 2^31)")
    root = raw["storage_root"]
    if not isinstance(root, str) or not root.strip() or "://" in root:
        raise ValueError("local storage_root must be a nonempty filesystem path")
    spark = SparkConfig(**raw["spark"])
    if not isinstance(spark.master, str) or not re.fullmatch(r"local\[[1-9][0-9]*\]", spark.master):
        raise ValueError("local master must use an explicit positive thread count, e.g. local[2]")
    if not isinstance(spark.driver_memory, str) or not re.fullmatch(r"[1-9][0-9]*[mg]", spark.driver_memory):
        raise ValueError("driver_memory must be positive with m/g units")
    if type(spark.shuffle_partitions) is not int or spark.shuffle_partitions < 1:
        raise ValueError("shuffle_partitions must be a positive integer")
    if type(spark.event_log_enabled) is not bool:
        raise ValueError("event_log_enabled must be a boolean")
    mongo = MongoConfig(**raw["mongo"])
    if not isinstance(mongo.database, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", mongo.database):
        raise ValueError("mongo.database must start with a letter and contain letters, digits or underscores")
    if not isinstance(mongo.uri_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", mongo.uri_env):
        raise ValueError("mongo.uri_env must be an uppercase environment variable name")
    return ProjectConfig(
        raw["profile"], raw["seed"], (path.parent / root).resolve(), spark, mongo,
        parse_experiment(raw["experiment"]), parse_model(raw["model"]),
    )
