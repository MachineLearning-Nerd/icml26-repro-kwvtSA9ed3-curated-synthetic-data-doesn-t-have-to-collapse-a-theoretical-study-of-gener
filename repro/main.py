"""Entry point for every experiment node.

The run command is a fixed contract -- `bash run.sh` on every node, which calls
`python -m repro.main`.  What a node *does* is decided entirely by the committed
file `repro/config/active.json`, never by the command line or the environment.

Exit code is 0 only if every enabled stage's verdict gate passes.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

from repro.lib import report
from repro.lib.verdict import Verdict

CONFIG_PATH = Path("repro/config/active.json")

# stage name -> module implementing `run(cfg) -> Verdict | list[Verdict]`
STAGES = {
    "judged_reference": "repro.stages.judged_reference",
    "claim1_outside_decay": "repro.stages.claim1_outside_decay",
    "claim2_noncollapse": "repro.stages.claim2_noncollapse",
    "claim3_variance": "repro.stages.claim3_variance",
    "claim4_nash": "repro.stages.claim4_nash",
    "claim5_cifar_flow": "repro.stages.claim5_cifar_flow",
    "claim6_text_gpt2": "repro.stages.claim6_text_gpt2",
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"missing {CONFIG_PATH} -- every node must commit its stage config")
    cfg = json.loads(CONFIG_PATH.read_text())
    unknown = [s for s in cfg.get("stages", []) if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages in {CONFIG_PATH}: {unknown}")
    return cfg


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    sha = report.git_sha()
    cpu = report.cpu_info()

    print("=" * 78)
    print("Reproduction: 'Curated Synthetic Data Doesn't Have to Collapse'")
    print("             (arXiv:2605.07724, OpenReview kwvtSA9ed3)")
    print("=" * 78)
    report.kv("node", cfg.get("node", "?"))
    report.kv("git sha", sha)
    report.kv("stages", ", ".join(cfg.get("stages", [])))
    report.kv("cpu: logical cores visible", cpu.get("os_cpu_count"))
    report.kv("cpu: cores in affinity mask", cpu.get("sched_getaffinity"))
    report.kv("cpu: model", cpu.get("model_name", "n/a"))
    report.kv("cpu: cgroup cpu.max", cpu.get("cgroup_cpu_max", "n/a"))
    report.kv("estimated cores required (declared)", cfg.get("estimated_cores", "?"))
    report.kv("compute target (declared)", cfg.get("compute_target", "?"))

    verdicts: list[Verdict] = []
    for name in cfg.get("stages", []):
        mod = importlib.import_module(STAGES[name])
        with report.Section(f"stage: {name}") as sec:
            out = mod.run(cfg.get("params", {}).get(name, {}))
        got = out if isinstance(out, list) else [out]
        for v in got:
            v.runtime_s = sec.seconds
            v.print_summary()
        verdicts.extend(got)

    summary = {
        "paper": "arXiv:2605.07724",
        "openreview": "kwvtSA9ed3",
        "node": cfg.get("node", "?"),
        "git_sha": sha,
        "stages": cfg.get("stages", []),
        "cpu": cpu,
        "declared_estimated_cores": cfg.get("estimated_cores"),
        "declared_compute_target": cfg.get("compute_target"),
        "total_runtime_s": time.time() - t0,
        "verdicts": [v.to_dict() for v in verdicts],
    }
    gates = {v.claim_id: v.gate()[0] for v in verdicts}
    summary["all_gates_pass"] = all(gates.values())
    out_path = report.write_json(report.ARTIFACTS / "verdicts.json", summary)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for v in verdicts:
        ok = "GATE-PASS" if v.gate()[0] else "GATE-FAIL"
        print(f"  {v.claim_id:<28} {v.status:<10} {ok}")
    print(f"\n  artifacts: {out_path}")
    print(f"  total runtime: {summary['total_runtime_s']:.1f}s")

    if not summary["all_gates_pass"]:
        print("\n  FAILED: at least one verdict is not supported by its own evidence.")
        return 1
    print("\n  OK: every recorded verdict is supported by its own evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
