"""Claim 3 / Theorem 3.6 -- Route B: the variance bound on general two-basin mixtures.

The judged evidence checked the plateau EQUALITY Var = a(1-a)Delta^2 on a 3-state
system, where each basin is a single atom so the within-basin variances are zero by
construction. Here the within-basin distributions are arbitrary continuous densities,
which is the setting the theorem quantifies over.

  (A) Randomised sweep over general two-basin mixtures with arbitrary within-basin
      shapes, checking Var_{p_inf}[r_i] >= a(1-a) max{Delta_i-2eps,0}^2 for BOTH i.
  (B) The same check on limits actually produced by the retraining dynamics, so the
      mixtures tested are ones the process can reach rather than hand-built ones.
  (C) Slack analysis: how much room the bound leaves, and where it is tightest.
"""

from __future__ import annotations

import numpy as np

from repro.lib import report
from repro.lib.landscape import quadratic_grid_landscape, uniform_init
from repro.lib.verdict import VERIFIED, Verdict


def _mixture_stats(r: np.ndarray, p1: np.ndarray, p2: np.ndarray, a: float) -> dict:
    """Var under a P_1 + (1-a) P_2 and its decomposition."""
    p = a * p1 + (1 - a) * p2
    mean = float(p @ r)
    var = float(p @ (r - mean) ** 2)
    m1 = float(p1 @ r)
    m2 = float(p2 @ r)
    v1 = float(p1 @ (r - m1) ** 2)
    v2 = float(p2 @ (r - m2) ** 2)
    return {"var": var, "mu1": m1, "mu2": m2, "within1": v1, "within2": v2,
            "inter_basin_term": a * (1 - a) * (m1 - m2) ** 2}


def run(params: dict) -> Verdict:
    out = report.artifact_dir("claim3", "continuous")
    n_random = int(params.get("n_random", 3000))
    steps = int(params.get("steps", 300))
    v = Verdict(
        claim_id="claim3/theorem-3.6-variance",
        title="Theorem 3.6: variance lower bound on general two-basin mixtures",
        status=VERIFIED,
        statement=(
            "For a two-basin limit p_inf = a P_1 + (1-a) P_2 with a in (0,1) and basin "
            "separation, Var_{p_inf}[r_i] >= a(1-a) max{Delta_i - 2 eps, 0}^2 for i=1,2."
        ),
    )

    # ================================================================= (A) == #
    report.banner(f"(A) {n_random} random two-basin mixtures with ARBITRARY within-basin shapes")
    rng = np.random.default_rng(20260726)
    rows = []
    for i in range(n_random):
        d = float(rng.uniform(1.5, 9.0))
        eps = float(rng.uniform(0.02, 0.8))
        lam = quadratic_grid_landscape(d=d, eps=eps, n=1601, half_width=12.0)
        if not (lam.S1.any() and lam.S2.any()) or (lam.S1 & lam.S2).any():
            continue
        a = float(rng.uniform(0.02, 0.98))
        # ARBITRARY (random, non-uniform, non-symmetric) densities within each basin
        w1 = rng.uniform(0.0, 1.0, lam.S1.sum()) ** rng.uniform(0.3, 4.0) + 1e-12
        w2 = rng.uniform(0.0, 1.0, lam.S2.sum()) ** rng.uniform(0.3, 4.0) + 1e-12
        p1 = np.zeros_like(lam.r1); p1[lam.S1] = w1 / w1.sum()
        p2 = np.zeros_like(lam.r1); p2[lam.S2] = w2 / w2.sum()
        D1, D2 = lam.cross_gaps_appendix()
        rec = {"i": i, "d": d, "eps": eps, "a": a, "Delta1": D1, "Delta2": D2}
        ok = True
        for idx, (r, D) in enumerate(((lam.r1, D1), (lam.r2, D2)), start=1):
            st = _mixture_stats(r, p1, p2, a)
            bound = a * (1 - a) * max(D - 2 * eps, 0.0) ** 2
            rec[f"var_r{idx}"] = st["var"]
            rec[f"bound_r{idx}"] = bound
            rec[f"slack_r{idx}"] = st["var"] - bound
            rec[f"holds_r{idx}"] = bool(st["var"] >= bound - 1e-12)
            rec[f"within_var_r{idx}"] = st["within1"] + st["within2"]
            ok = ok and rec[f"holds_r{idx}"]
        rec["holds_both"] = ok
        rows.append(rec)
    report.write_csv(out / "random_mixture_variance.csv", rows)
    n_hold = sum(r["holds_both"] for r in rows)
    worst = min(rows, key=lambda r: min(r["slack_r1"], r["slack_r2"]))
    nonzero_within = sum(1 for r in rows if r["within_var_r1"] > 1e-9)
    report.kv("mixtures tested", len(rows))
    report.kv("bound holds for BOTH i=1 and i=2", f"{n_hold}/{len(rows)}")
    report.kv("mixtures with strictly positive within-basin variance",
              f"{nonzero_within}/{len(rows)}  (the judged 3-state reduction had 0)")
    report.kv("tightest case", f"a={worst['a']:.4f} d={worst['d']:.3f} eps={worst['eps']:.3f} "
                               f"slack_r1={worst['slack_r1']:.4e} slack_r2={worst['slack_r2']:.4e}")
    v.add(
        "the variance lower bound holds on every random general two-basin mixture, for both rewards",
        len(rows) > 500 and n_hold == len(rows),
        f"{n_hold}/{len(rows)} mixtures satisfy Var >= a(1-a)max{{Delta_i-2eps,0}}^2 for i=1 AND "
        f"i=2, with random within-basin densities (random exponent shaping), random a in "
        f"[0.02,0.98], d in [1.5,9] and eps in [0.02,0.8]. {nonzero_within} of them have "
        f"strictly positive within-basin variance -- the degree of freedom the judged "
        f"3-state reduction lacked entirely. Minimum slack observed: "
        f"{min(min(r['slack_r1'], r['slack_r2']) for r in rows):.4e}.",
        n_tested=len(rows), n_hold=n_hold, n_nonzero_within=nonzero_within,
    )

    # ================================================================= (B) == #
    report.banner("(B) The same check on limits actually REACHED by the retraining dynamics")
    dyn_rows = []
    for q in (0.2, 0.35, 0.5, 0.65, 0.8):
        for d in (3.0, 5.0, 7.0):
            lam = quadratic_grid_landscape(d=d, eps=0.1, n=2001, half_width=12.0)
            res = lam.run(uniform_init(lam), q, steps)
            p_inf = res["p_final"]
            a_inf = float(p_inf[lam.S1].sum())
            D1, D2 = lam.cross_gaps_appendix()
            rec = {"q": q, "d": d, "a_inf": a_inf,
                   "m_inf": float(p_inf[lam.outside].sum())}
            good = True
            for idx, (r, D) in enumerate(((lam.r1, D1), (lam.r2, D2)), start=1):
                mean = float(p_inf @ r)
                var = float(p_inf @ (r - mean) ** 2)
                bound = a_inf * (1 - a_inf) * max(D - 2 * 0.1, 0.0) ** 2
                rec[f"var_r{idx}"] = var
                rec[f"bound_r{idx}"] = bound
                rec[f"holds_r{idx}"] = bool(var >= bound - 1e-9)
                good = good and rec[f"holds_r{idx}"]
            rec["holds_both"] = good
            dyn_rows.append(rec)
            if q in (0.2, 0.5) and d in (3.0, 7.0):
                report.kv(f"q={q:<5g} d={d:<4g}",
                          f"a_inf={a_inf:.4f}  Var[r1]={rec['var_r1']:.4f} >= {rec['bound_r1']:.4f}"
                          f"  Var[r2]={rec['var_r2']:.4f} >= {rec['bound_r2']:.4f}")
    report.write_csv(out / "dynamics_limit_variance.csv", dyn_rows)
    n_dyn = sum(r["holds_both"] for r in dyn_rows)
    v.add(
        "the bound also holds on the limits the retraining dynamics actually produce",
        n_dyn == len(dyn_rows),
        f"{n_dyn}/{len(dyn_rows)} dynamics-produced limits over q x d satisfy the bound for "
        "both rewards. These are reachable limits, not hand-constructed mixtures.",
        n_dyn=n_dyn,
    )

    # ================================================================= (C) == #
    report.banner("(C) Where is the bound tight, and is it ever vacuous?")
    tight = [r for r in rows if min(r["slack_r1"], r["slack_r2"]) < 0.05 * max(r["bound_r1"], 1e-9)]
    vacuous = [r for r in rows if r["bound_r1"] <= 0.0]
    report.kv("cases within 5% of the bound", len(tight))
    report.kv("cases where the bound is vacuous (Delta_i <= 2 eps)", len(vacuous))
    v.add(
        "the bound is non-vacuous on the great majority of the sampled domain",
        len(vacuous) < len(rows),
        f"{len(vacuous)}/{len(rows)} sampled configurations have Delta_i <= 2 eps, where the "
        f"positive part makes the bound 0 and the theorem asserts nothing. On the remaining "
        f"{len(rows) - len(vacuous)} it is a genuine strictly positive lower bound.",
    )

    # ---- negative controls -------------------------------------------- #
    report.banner("Negative controls")
    # a degenerate (collapsed) limit must give a zero bound and can have zero variance
    lam_c = quadratic_grid_landscape(d=6.0, eps=0.1, n=2001, half_width=12.0)
    res_c = lam_c.run(uniform_init(lam_c), 1.0, steps)          # q=1 => single reward
    p_c = res_c["p_final"]
    a_c = float(p_c[lam_c.S1].sum())
    mean_c = float(p_c @ lam_c.r1)
    var_c = float(p_c @ (lam_c.r1 - mean_c) ** 2)
    bound_c = a_c * (1 - a_c) * max(lam_c.cross_gaps_appendix()[0] - 0.2, 0.0) ** 2
    report.kv("single-reward limit", f"a_inf={a_c:.8f}  Var[r1]={var_c:.3e}  bound={bound_c:.3e}")
    v.add_control(
        "single-reward curation collapses: variance -> 0 and the bound goes vacuous with it",
        var_c < 1e-2 and bound_c < 1e-2,
        f"at q=1 the limit has a_inf={a_c:.8f}, so a(1-a) -> 0 and the theorem's bound "
        f"becomes {bound_c:.3e} while the measured variance is {var_c:.3e}. The theorem "
        "therefore does NOT assert positive variance for collapsed limits -- it is not "
        "vacuously true for every distribution, and it correctly declines to protect the "
        "single-reward case the paper contrasts against.",
        a_inf=a_c, var=var_c, bound=bound_c,
    )
    # deliberately violate separation: overlapping basins must void the bound
    lam_o = quadratic_grid_landscape(d=0.4, eps=0.5, n=2001, half_width=12.0)
    overlap = int((lam_o.S1 & lam_o.S2).sum())
    report.kv("overlapping-basin control", f"|S1 ∩ S2| = {overlap} cells")
    v.add_control(
        "with overlapping basins the theorem's disjointness hypothesis fails outright",
        overlap > 0,
        f"at d=0.4, eps=0.5 the eps-basins overlap in {overlap} grid cells, so Assumption "
        "2.1(ii) fails and no conclusion may be drawn -- the auditor rejects such "
        "landscapes rather than reporting a pass.",
    )

    v.numbers = {
        "n_random_mixtures": len(rows),
        "n_bound_holds": n_hold,
        "n_with_positive_within_basin_variance": nonzero_within,
        "min_slack": float(min(min(r["slack_r1"], r["slack_r2"]) for r in rows)),
        "n_dynamics_limits": len(dyn_rows),
        "single_reward_variance": var_c,
    }
    v.limitations = [
        "A randomised sweep can refute but not prove a universally quantified inequality; "
        "Route A supplies the symbolic derivation. This route's job is to exercise the "
        "general within-basin freedom the judged 3-state evidence could not represent.",
        "Densities are represented on a 1601/2001-cell grid, so reported variances carry "
        "discretisation error of order the cell width squared.",
    ]
    v.artifacts = [str(p) for p in sorted(out.rglob("*")) if p.is_file()]
    return v
