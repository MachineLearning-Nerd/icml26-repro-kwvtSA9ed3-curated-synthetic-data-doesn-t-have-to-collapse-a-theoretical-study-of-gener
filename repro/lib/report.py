"""Evidence writers.

Every stage writes machine-readable raw output under `.openresearch/artifacts/`
so a run is inspectable after the fact, and prints the same numbers to stdout
because in local orx mode the run log is the only evidence channel.
"""

from __future__ import annotations

import csv
import hashlib
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
    _emit(path)
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
    _emit(path)
    return path


# --------------------------------------------------------------------------- #
# Lossless artifact channel.
#
# In orx local mode `orx artifacts` does not exist: files written under
# .openresearch/artifacts/ inside a compute job are unreachable once the job ends,
# and the run log is the only wire out. So every artifact is also emitted to stdout
# between framed markers carrying its exact byte count and SHA-256. A consumer can
# reconstruct the file byte-for-byte and verify it, which means a truncated or
# interleaved log fails the hash check loudly instead of yielding plausible-looking
# wrong numbers.
#
# Emission is base64 of the raw bytes: artifacts contain newlines, and a payload that
# cannot collide with the marker lines is what makes the frame unambiguous.

EMIT_BEGIN = "<<<ORX-ARTIFACT"
EMIT_END = ">>>ORX-ARTIFACT-END"
EMIT_LIMIT = int(os.environ.get("ORX_ARTIFACT_EMIT_LIMIT", 4_000_000))


def _emit(path: Path) -> None:
    import base64

    raw = path.read_bytes()
    rel = path.as_posix()
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) > EMIT_LIMIT:
        # Never truncate silently: say so, and say exactly how much was withheld.
        print(f"{EMIT_BEGIN} path={rel} bytes={len(raw)} sha256={digest} "
              f"encoding=none status=OMITTED-TOO-LARGE limit={EMIT_LIMIT}")
        print(f"{EMIT_END} path={rel}")
        return
    payload = base64.b64encode(raw).decode("ascii")
    print(f"{EMIT_BEGIN} path={rel} bytes={len(raw)} sha256={digest} "
          f"encoding=base64 status=OK")
    for i in range(0, len(payload), 76):
        print(payload[i:i + 76])
    print(f"{EMIT_END} path={rel}")


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
