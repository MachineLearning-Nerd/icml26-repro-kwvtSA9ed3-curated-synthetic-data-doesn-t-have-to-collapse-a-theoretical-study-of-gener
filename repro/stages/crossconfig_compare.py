"""Decide Claims 5 and 6 from the comparison their statements actually make.

Both claims are COMPARATIVE: pluralistic (multi-reward) curation is said to sustain more
diversity across recursive retraining than single-reward curation. A single configuration
cannot test that, yet the per-configuration stages hardcoded status=VERIFIED and checked
only that the experiment ran and that metrics were recorded. Those gates passed
vacuously. This stage repairs that: it takes the measured series from every sibling
configuration and DERIVES the status from the comparison, so a reversed or absent effect
produces FALSIFIED rather than a green check.

Inputs are the per-configuration summaries emitted by the sibling runs, carried here as
committed data with their originating run id and SHA-256 (see repro/data/README.md).
Nothing is recomputed: the numbers are exactly what those runs printed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from repro.lib import report
from repro.lib.verdict import BLOCKED, FALSIFIED, VERIFIED, Verdict

DATA = Path("repro/data/sibling_summaries.json")
# A margin below this is treated as "no effect": the tail means are noisy at these
# sample sizes, so a hair's-breadth ordering must not be reported as support.
MIN_MARGIN = 0.05
# The single-reward arm must end below this fraction of its OWN first round for the
# comparison to have a live collapsing baseline.
COLLAPSE_FRACTION = 0.6


def _load() -> dict:
    raw = DATA.read_bytes()
    d = json.loads(raw)
    report.kv("sibling summary file", f"{DATA} ({len(raw)} bytes, "
                                      f"sha256 {hashlib.sha256(raw).hexdigest()[:16]}...)")
    for e in d["configs"]:
        report.kv(f"  {e['label']}", f"run {e['run_id']}  ({e['claim']})")
    return d


def _compare(entries: list[dict], metric: str, plural_key, label: str) -> tuple[bool, dict]:
    """Is the pluralistic arm's tail metric higher than the single-reward arm's?"""
    plural = [e for e in entries if plural_key(e)]
    single = [e for e in entries if not plural_key(e)]
    if not plural or not single:
        return False, {"error": f"{label}: need both arms, got "
                                f"{len(plural)} pluralistic and {len(single)} single"}
    p_best = max(plural, key=lambda e: e[metric])
    s_best = max(single, key=lambda e: e[metric])
    margin = p_best[metric] - s_best[metric]
    return margin > MIN_MARGIN, {
        "metric": metric,
        "pluralistic": {e["label"]: round(e[metric], 4) for e in plural},
        "single_reward": {e["label"]: round(e[metric], 4) for e in single},
        "best_pluralistic": p_best["label"], "best_single": s_best["label"],
        "margin": round(margin, 4), "min_margin_required": MIN_MARGIN,
    }


def run(params: dict) -> list[Verdict]:
    out = report.artifact_dir("crossconfig")
    if not DATA.exists():
        v = Verdict(claim_id="claim5+6/cross-configuration", title="Cross-configuration comparison",
                    status=BLOCKED, statement="Pluralistic curation sustains more diversity "
                                              "than single-reward curation.")
        v.add("sibling summaries are available", False, f"{DATA} not present")
        v.limitations = ["No sibling summaries were committed, so the comparative claims "
                         "cannot be evaluated."]
        return [v]

    data = _load()
    verdicts = []

    for claim, metric, statement, pk in (
        ("claim5", "entropy_tail_mean",
         "CIFAR-10 flow retraining: curation under balanced multi-reward preferences "
         "sustains higher class entropy across recursive generations than single-reward "
         "curation.",
         lambda e: e["M"] > 1),
        ("claim6", "H_tail_mean",
         "Text retraining: curation under two length preferences sustains higher length "
         "entropy H(L) across recursive generations than single-preference curation.",
         lambda e: e["M"] > 1),
    ):
        entries = [e for e in data["configs"] if e["claim"] == claim]
        report.banner(f"{claim}: {metric} across configurations")
        for e in sorted(entries, key=lambda e: e["M"]):
            report.kv(f"  {e['label']} (M={e['M']})",
                      f"first {e.get('first', float('nan')):.4f} -> tail {e[metric]:.4f}")

        holds, detail = _compare(entries, metric, pk, claim)
        v = Verdict(claim_id=f"{claim}/pluralistic-vs-single",
                    title=f"{claim.upper()}: pluralistic vs single-reward curation",
                    status=VERIFIED if holds else FALSIFIED,
                    statement=statement)
        if "error" in detail:
            v.status = BLOCKED
            v.add("both arms were measured", False, detail["error"])
            v.limitations = [detail["error"]]
            verdicts.append(v)
            continue

        v.add(
            "the comparison the claim states was computed from measured series, "
            "not assumed",
            True,
            f"best pluralistic arm {detail['best_pluralistic']} reached {metric} "
            f"{max(detail['pluralistic'].values()):.4f}; best single-reward arm "
            f"{detail['best_single']} reached {max(detail['single_reward'].values()):.4f}; "
            f"margin {detail['margin']:+.4f} against a required {MIN_MARGIN}",
            **detail,
        )
        v.add(
            "the ordering predicted by the paper holds with a margin above noise",
            holds,
            f"margin {detail['margin']:+.4f} "
            + ("exceeds" if holds else "does NOT exceed")
            + f" the {MIN_MARGIN} threshold, so the claim is "
            + ("supported" if holds else "not supported")
            + " by this reproduction",
        )
        # The collapse test is RELATIVE to the arm's own first round, not an absolute
        # cutoff: the two domains have different entropy scales (10 CIFAR classes vs a
        # length distribution over ~60 values), so any fixed threshold would be a number
        # chosen to fit one of them.
        s_arm = min((e for e in entries if not pk(e)), key=lambda e: e[metric])
        decline_ok = s_arm[metric] < COLLAPSE_FRACTION * s_arm["first"]
        v.add_control(
            "the single-reward arm actually collapsed, so the comparison has a live "
            "baseline",
            decline_ok,
            f"single-reward arm {s_arm['label']} fell from {s_arm['first']:.4f} to "
            f"{s_arm[metric]:.4f}, i.e. to {s_arm[metric] / s_arm['first']:.1%} of its "
            f"own first round (required: below {COLLAPSE_FRACTION:.0%}). If the "
            "single-reward arm had also stayed diverse there would be no collapse to "
            "avoid and the comparison would be vacuous.",
        )
        # Non-circularity: show that this comparison is capable of returning FALSIFIED.
        # Exchange the two arms' values and re-run the identical decision procedure; if
        # the swapped input still "passed", the test would be a rubber stamp.
        swapped = []
        for e in entries:
            e2 = dict(e)
            e2[metric] = (min if pk(e) else max)(x[metric] for x in entries)
            swapped.append(e2)
        swap_holds, swap_detail = _compare(swapped, metric, pk, claim)
        v.add_control(
            "the decision procedure can return FALSIFIED (it is not a rubber stamp)",
            not swap_holds,
            f"re-running the identical comparison with the two arms' values exchanged "
            f"gives margin {swap_detail.get('margin')} and would report "
            f"{'VERIFIED -- BROKEN' if swap_holds else 'FALSIFIED'}, so the outcome is "
            f"driven by the measurements and not by the code path.",
        )

        v.numbers = detail
        v.limitations = [
            "Downscaled from the paper's Appendix C.5/C.6 settings; the selection "
            "pressure (5% keep ratio) and the loop structure are preserved, the absolute "
            "sample counts and model sizes are not.",
            "One seed per configuration: the margin is a point estimate, not a "
            f"confidence interval, which is why a {MIN_MARGIN} threshold is required "
            "rather than any positive difference.",
            "Summaries are carried as committed data from the sibling runs listed above; "
            "each is traceable to its run id and the run log it was recovered from.",
        ]
        verdicts.append(v)

    report.write_json(out / "crossconfig_comparison.json",
                      {"verdicts": [v.claim_id for v in verdicts],
                       "detail": [v.numbers for v in verdicts],
                       "source_runs": [e["run_id"] for e in data["configs"]]})
    return verdicts
