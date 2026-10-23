"""Read installed versions and execute java -version, never start Spark."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def inspect_environment() -> dict:
    errors = []
    packages = {}
    for package, expected in {"pyspark": "3.5.6", "numpy": "1.26.4"}.items():
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
        if packages[package] != expected:
            errors.append(f"{package}=={expected} required; found {packages[package]}")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        errors.append("Python 3.11 required")
    java_home = os.environ.get("JAVA_HOME")
    java = str(Path(java_home) / "bin/java") if java_home else shutil.which("java")
    java_version = None
    try:
        if not java:
            raise FileNotFoundError("java executable not found")
        result = subprocess.run([java, "-version"], capture_output=True, text=True, timeout=10, check=True)
        match = re.search(r'version "([^"]+)"', result.stderr + result.stdout)
        java_version = match.group(1) if match else None
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"Could not inspect Java: {exc}")
    if not java_version or java_version.split(".")[0] != "17":
        errors.append("Select Java 17 with JAVA_HOME before running this project")
    return {
        "python": platform.python_version(),
        "packages": packages,
        "java_executable": java,
        "java_version": java_version,
        "spark_started": False,
        "errors": errors,
    }
