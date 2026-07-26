"""Evidence writers.

Every stage writes machine-readable raw output under `.openresearch/artifacts/`
so a run is inspectable after the fact, and prints the same numbers to stdout
because in local orx mode the run log is the only evidence channel.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

ARTIFACTS = Path(".openresearch/artifacts")


def artifact_dir(*parts: str) -> Path:
    d = ARTIFACTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n")
    return path


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return path


def _default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def git_sha() -> str:
    for cmd in (["git", "rev-parse", "HEAD"],):
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            pass
    return os.environ.get("GIT_COMMIT", "unknown")


def cpu_info() -> dict:
    """CPU allocation actually granted to this run (recorded in every artifact)."""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "os_cpu_count": os.cpu_count(),
    }
    try:
        info["sched_getaffinity"] = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        info["sched_getaffinity"] = None
    try:
        with open("/proc/cpuinfo") as fh:
            models = [l.split(":", 1)[1].strip() for l in fh if l.startswith("model name")]
        if models:
            info["model_name"] = models[0]
            info["n_logical_from_cpuinfo"] = len(models)
    except OSError:
        pass
    for path, key in (
        ("/sys/fs/cgroup/cpu.max", "cgroup_cpu_max"),
        ("/sys/fs/cgroup/memory.max", "cgroup_memory_max"),
    ):
        try:
            info[key] = Path(path).read_text().strip()
        except OSError:
            pass
    return info


class Section:
    """Times a stage, prints a banner, and returns runtime for the artifact."""

    def __init__(self, title: str) -> None:
        self.title = title

    def __enter__(self) -> "Section":
        self.t0 = time.time()
        print(f"\n{'=' * 78}\n== {self.title}\n{'=' * 78}", flush=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.time() - self.t0
        print(f"-- {self.title}: {self.seconds:.2f}s", flush=True)


def banner(msg: str) -> None:
    print(f"\n--- {msg}", flush=True)


def kv(label: str, value: Any, width: int = 46) -> None:
    print(f"    {label:<{width}} {value}", flush=True)
