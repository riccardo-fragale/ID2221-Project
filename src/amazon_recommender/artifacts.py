"""Create local run folders with enough context to reproduce an experiment."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from uuid import uuid4

from amazon_recommender.config import ProjectConfig


def file_identity(path: Path) -> dict:
    """Hash a local input in bounded memory; never download or parse its data."""
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def code_identity() -> dict:
    """Record this package checkout rather than the caller's working directory."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout
        return {"revision": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}


def create_run(config: ProjectConfig, *, reviews: Path, metadata: Path) -> Path:
    """Snapshot settings and input identities. This is preparation, not execution.

    Every invocation gets a new directory; existing runs are never overwritten.
    Pipeline stages will write their own outputs here when implemented.
    """
    inputs = {"reviews": file_identity(reviews), "metadata": file_identity(metadata)}
    created = datetime.now(timezone.utc)
    run_id = f"{created:%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}"
    directory = config.storage_root / "runs" / run_id
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "created_at": created.isoformat(),
        "config_digest": config.digest,
        "config": config.as_dict(),
        "inputs": inputs,
        "code": code_identity(),
        "dependencies": {name: version(name) for name in ("pyspark", "numpy")},
    }
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "run.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "run.json")
    return directory
