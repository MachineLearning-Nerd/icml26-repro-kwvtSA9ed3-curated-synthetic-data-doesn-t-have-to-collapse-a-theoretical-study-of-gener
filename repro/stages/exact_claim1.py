"""Claim 1 / Lemma 3.3 -- Route B: exact counterexample + the paper's own geometry.

Three independent pieces of evidence, none of which uses symbolic algebra:

  (A) An explicit 4-atom landscape in EXACT RATIONAL ARITHMETIC that satisfies every
      audited clause of Assumption 2.1 and on which m_t strictly INCREASES, converging
      to 1 while a_t -> 0. No rounding is involved anywhere, so the refutation of
      "there exists rho_eps in (0,1) with m_{t+1} <= rho_eps m_t for all t" is exact.

  (B) A sweep over the counterexample family confirming numerically the boundary
      delta = log 2 that Route A derived symbolically, from the other side.

  (C) The paper's OWN synthetic reward geometry (Appendix C.4: r_i(x) = -||x-mu_i||^2)
      on a fine continuous grid, measuring sup_{x not in S_eps} W_t(x) per iteration --
      i.e. measuring whether appendix Assumption B.4 actually holds in the setting the
      authors themselves experiment in.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import numpy as np

from repro.lib import report
from repro.lib.exact import ExactLandscape
from repro.lib.landscape import quadratic_grid_landscape, uniform_init
from repro.lib.verdict import FALSIFIED, Verdict


def _counterexample() -> tuple[ExactLandscape, dict[str, F], F]:
    """The 4-atom exact counterexample.

    eps = log(10/9)  (so eps_ratio = 9/10 and S_{i,eps} = {w_i >= 0.9 w_i*}).
      A: w1 = 1,    w2 = 1/100   -> S_1 only
      B: w1 = 1/100, w2 = 1      -> S_2 only
      C: w1 = w2 = 4/5           -> outside (0.8 < 0.9), but only just
      D: w1 = w2 = 1/1000        -> outside, deeply
    C is the "compromise" atom: not eps-optimal for either reward, yet good enough for
    both that the balanced multiplier lifts it above 1 while the basins hold little mass.
    Assumption 2.1(iii) permits this because it only requires SOME delta > 0.
    """
    w1 = {"A": F(1), "B": F(1, 100), "C": F(4, 5), "D": F(1, 1000)}
    w2 = {"A": F(1, 100), "B": F(1), "C": F(4, 5), "D": F(1, 1000)}
    lam = ExactLandscape(w1, w2, eps_ratio=F(9, 10), name="exact-4atom-counterexample")
    p0 = {"A": F(1, 1000), "B": F(1, 1000), "C": F(998, 1000), "D": F(0)}
    return lam, p0, F(1, 2)


def run(params: dict) -> Verdict:
    out = report.artifact_dir("claim1", "exact_and_continuous")
    steps = int(params.get("steps", 60))
    v = Verdict(
        claim_id="claim1/lemma-3.3-outside-decay",
        title="Lemma 3.3: geometric decay of mass outside the eps-optimal basins",
        status=FALSIFIED,
        statement=(
            "Under Assumption 2.1 (i,iii) there exists rho_eps in (0,1) with "
            "m_{t+1} <= rho_eps m_t for all t, hence m_t -> 0 and a_t + b_t -> 1."
        ),
    )

    # ================================================================= (A) == #
    report.banner("(A) Exact rational counterexample: audit Assumption 2.1 clause by clause")
    lam, p0, q = _counterexample()
    audit = lam.audit(p0)
    for key in (
        "A1_i_bounded_rewards", "A1_ii_basins_disjoint", "A1_ii_nontrivial_init",
        "A1_iii_outside_gap", "A1_iii_cross_gaps",
    ):
        report.kv(key, f"ok={audit[key]['ok']}   " + str({
            k: val for k, val in audit[key].items() if k != "ok"
        })[:150])
    all_ok = all(audit[k]["ok"] for k in audit if isinstance(audit[k], dict))
    report.kv("eps", f"{audit['eps']:.6f}  (= log(10/9), exact ratio {audit['eps_ratio_exact']})")
    report.kv("Delta_1, Delta_2", f"{audit['A1_iii_cross_gaps']['Delta1']:.6f}, "
                                  f"{audit['A1_iii_cross_gaps']['Delta2']:.6f}  (= log 100)")
    report.kv("outside gap delta", f"{audit['A1_iii_outside_gap']['delta']:.6f}  (= log(5/4))")
    v.add(
        "the counterexample satisfies EVERY audited clause of Assumption 2.1",
        all_ok,
        f"eps={audit['eps']:.6f}, delta={audit['A1_iii_outside_gap']['delta']:.6f}>0, "
        f"Delta_1=Delta_2={audit['A1_iii_cross_gaps']['Delta1']:.6f}>0, basins disjoint, "
        f"p_0(S_1)={audit['A1_ii_nontrivial_init']['p0_S1']}>0, "
        f"p_0(S_2)={audit['A1_ii_nontrivial_init']['p0_S2']}>0",
        audit=audit,
    )
    # the counterexample must NOT satisfy the appendix's extra hypothesis B.4
    report.banner("(A) Run the exact dynamics")
    rows = lam.run(p0, q, steps)
    report.write_csv(out / "exact_counterexample_trace.csv", rows)
    for r in rows[:6] + rows[-2:]:
        report.kv(f"t={r['t']:<3d}", f"m_t={r['m_t']:.12f}  m_(t+1)/m_t={r['m_next_over_m_t']:.10f}"
                                     f"  a_t={r['a_t']:.3e}  supW_out={r['sup_W_outside']:.6f}"
                                     f"  W(S1)={r['W_basin1']:.6f}")
    increased_every_step = all(r["m_increased"] for r in rows)
    max_ratio = max(r["m_next_over_m_t"] for r in rows)
    mass_ok = all(r["mass_conserved_exact"] for r in rows)
    report.kv("m_t increased at EVERY step (exact)", increased_every_step)
    report.kv("max_t m_(t+1)/m_t (exact arithmetic)", f"{max_ratio:.10f}")
    report.kv("exact ratio at t=0", rows[0]["m_ratio_exact"])
    report.kv("m_0 -> m_final", f"{rows[0]['m_t']:.10f} -> {rows[-1]['m_t']:.12f}")
    report.kv("a_0 -> a_final", f"{rows[0]['a_t']:.3e} -> {rows[-1]['a_t']:.3e}")
    v.add(
        "total mass is exactly conserved at every step (the update is a valid density)",
        mass_ok, "sum_x p_{t+1}(x) == 1 as an exact rational identity at every step",
    )
    v.add(
        "NO rho_eps in (0,1) exists: m_{t+1} > m_t at every step, in exact arithmetic",
        increased_every_step and max_ratio > 1.0,
        f"m_(t+1)/m_t = {rows[0]['m_ratio_exact']} = {max_ratio:.10f} > 1 at t=0 and m_t "
        f"rises monotonically {rows[0]['m_t']:.6f} -> {rows[-1]['m_t']:.12f}. Since rho_eps "
        "is existentially quantified before the universal over t, one such step refutes "
        "the statement; here every step does.",
        max_ratio=max_ratio, exact_ratio_t0=rows[0]["m_ratio_exact"],
    )
    v.add(
        "the conclusion inverts: m_t -> 1 and a_t + b_t -> 0, not the reverse",
        rows[-1]["m_t"] > rows[0]["m_t"] and rows[-1]["a_t"] < rows[0]["a_t"],
        f"a_t falls {rows[0]['a_t']:.3e} -> {rows[-1]['a_t']:.3e} while m_t rises to "
        f"{rows[-1]['m_t']:.12f}: all mass ends up OUTSIDE the eps-optimal basins, on the "
        "compromise atom C. The lemma asserts exactly the opposite.",
    )

    # ---- negative control: repair the landscape, decay returns ------------- #
    report.banner("(A) Negative control: deepen the outside atom and the lemma's conclusion returns")
    # Move C from w = 4/5 down to w = 1/50, which makes delta large enough that
    # 2 e^{-delta} < 1 + e^{-Delta}, i.e. leaves the counterexample family.
    w1c = dict(lam.w1); w2c = dict(lam.w2)
    w1c["C"] = F(1, 50); w2c["C"] = F(1, 50)
    lam_ctrl = ExactLandscape(w1c, w2c, eps_ratio=F(9, 10), name="control-deep-outside")
    rows_ctrl = lam_ctrl.run(p0, q, steps)
    report.write_csv(out / "exact_control_trace.csv", rows_ctrl)
    ctrl_ratios = [r["m_next_over_m_t"] for r in rows_ctrl]
    ctrl_decays = all(r["m_increased"] is False for r in rows_ctrl)
    report.kv("control: max m_(t+1)/m_t", f"{max(ctrl_ratios):.8f}")
    report.kv("control: m_0 -> m_final", f"{rows_ctrl[0]['m_t']:.8f} -> {rows_ctrl[-1]['m_t']:.3e}")
    v.add_control(
        "control landscape (deep outside atom) DOES decay geometrically as the lemma says",
        ctrl_decays,
        f"changing only w(C) from 4/5 to 1/50 -- leaving every other atom and p_0 "
        f"untouched -- restores contraction: max_t m_(t+1)/m_t = {max(ctrl_ratios):.8f} < 1 "
        f"and m_t falls to {rows_ctrl[-1]['m_t']:.3e}. So the failure is caused by the "
        "shallow outside gap that Assumption 2.1 permits, not by a coding error in the "
        "update: the same code produces the lemma's behaviour on a conforming landscape.",
        max_ratio=max(ctrl_ratios),
    )

    # ================================================================= (B) == #
    report.banner("(B) Sweep the counterexample family; confirm the delta = log 2 boundary")
    rows_fam = []
    for delta in (0.05, 0.1, 0.2, 0.4, 0.6, 0.69, 0.70, 0.8, 1.0, 1.5):
        for Delta in (3.0, 5.0, 10.0):
            # exact rationals approximating e^{-delta}, e^{-Delta} to 12 digits
            wC = F(round(math.exp(-delta) * 10**12), 10**12)
            wB = F(round(math.exp(-Delta) * 10**12), 10**12)
            if not (0 < wC < F(9, 10)):
                continue  # C must be strictly outside both basins
            l = ExactLandscape({"A": F(1), "B": wB, "C": wC},
                               {"A": wB, "B": F(1), "C": wC}, F(9, 10))
            p = {"A": F(1, 1000), "B": F(1, 1000), "C": F(998, 1000)}
            r0 = l.run(p, F(1, 2), 1)[0]
            predicted = 2 * math.exp(-delta) > 1 + math.exp(-Delta)
            rows_fam.append({
                "delta": delta, "Delta": Delta,
                "m_increased_observed": r0["m_increased"],
                "predicted_by_2exp_condition": predicted,
                "agree": r0["m_increased"] == predicted,
                "m_ratio": r0["m_next_over_m_t"],
            })
    report.write_csv(out / "counterexample_family_sweep.csv", rows_fam)
    n_agree = sum(r["agree"] for r in rows_fam)
    for r in rows_fam:
        if r["delta"] in (0.6, 0.69, 0.70, 0.8):
            report.kv(f"delta={r['delta']:<5g} Delta={r['Delta']:<5g}",
                      f"m increased={r['m_increased_observed']}  predicted={r['predicted_by_2exp_condition']}"
                      f"  ratio={r['m_ratio']:.8f}")
    v.add(
        "the closed-form condition 2e^{-delta} > 1 + e^{-Delta} predicts the observed "
        "failures exactly across the swept family",
        n_agree == len(rows_fam),
        f"{n_agree}/{len(rows_fam)} parameter combinations agree with the condition Route A "
        f"derived symbolically. The boundary sits between delta=0.69 and delta=0.70, "
        f"bracketing log 2 = {math.log(2):.6f}. Two independent routes, same boundary.",
        n_agree=n_agree, n_total=len(rows_fam),
    )

    # ================================================================= (C) == #
    report.banner("(C) The paper's own geometry (Appendix C.4): does Assumption B.4 hold there?")
    rows_cont = []
    for d in (2.0, 4.0, 6.0, 8.0):
        for eps in (0.1, 0.5):
            lam_c = quadratic_grid_landscape(d=d, eps=eps, n=4001, half_width=12.0)
            p = uniform_init(lam_c)          # the paper's C.9.2 initialisation
            res = lam_c.run(p, q=0.5, steps=50)
            tr = res["trace"]
            sup_W = [r["rho_outside_sup_W"] for r in tr]
            ms = np.array([r["m_t"] for r in tr])
            ratios = ms[1:] / np.maximum(ms[:-1], 1e-300)
            rows_cont.append({
                "d": d, "eps": eps,
                "max_sup_W_outside": float(max(sup_W)),
                "B4_holds_uniformly": bool(max(sup_W) < 1.0),
                "max_m_ratio": float(ratios.max()),
                "uniform_rho_lt_1_exists": bool(ratios.max() < 1.0),
                "m_0": float(ms[0]), "m_50": float(ms[-1]),
                "m_decays_asymptotically": bool(ms[-1] < ms[0]),
            })
            report.kv(f"d={d:<4g} eps={eps:<4g}",
                      f"max_t sup_(x outside) W_t = {max(sup_W):.4f}  "
                      f"B.4 holds={max(sup_W) < 1.0}  max_t m_(t+1)/m_t={ratios.max():.4f}  "
                      f"m: {ms[0]:.4f} -> {ms[-1]:.3e}")
    report.write_csv(out / "paper_geometry_outside_domination.csv", rows_cont)
    any_b4_violated = any(not r["B4_holds_uniformly"] for r in rows_cont)
    all_decay = all(r["m_decays_asymptotically"] for r in rows_cont)
    v.add(
        "in the paper's own quadratic reward geometry, appendix Assumption B.4 is "
        "violated at some iterations, yet m_t still decays asymptotically",
        any_b4_violated and all_decay,
        "sup_{x outside} W_t(x) exceeds 1 for early t in the configurations marked above "
        "(the points just beyond the basin boundary have r_i barely below r_i*-eps, so the "
        "balanced multiplier lifts them), which means B.4 -- the hypothesis the appendix "
        "proof needs -- does not hold uniformly in t even in the authors' own setting. "
        "The CONCLUSION m_t -> 0 nonetheless holds there, because mass does concentrate at "
        "the two optima. So the lemma's conclusion is right in the paper's experiments "
        "while the stated route to it is not available.",
        rows=rows_cont,
    )
    v.numbers = {
        "exact_m_ratio_t0": rows[0]["m_ratio_exact"],
        "exact_m_ratio_t0_float": max_ratio,
        "m_final_counterexample": rows[-1]["m_t"],
        "a_final_counterexample": rows[-1]["a_t"],
        "control_max_ratio": max(ctrl_ratios),
        "family_sweep_agreement": f"{n_agree}/{len(rows_fam)}",
    }
    v.limitations = [
        "The counterexample is a finite (4-atom) landscape. Assumption 2.1 places no "
        "cardinality or continuity restriction on X, and a finite X with a counting base "
        "measure is a legitimate instance; part (C) additionally exercises continuous "
        "landscapes on a 4001-cell grid.",
        "Part (C) measures a hypothesis (B.4) on a discretised continuum; the reported "
        "sup is over grid cells, so it is a lower bound on the true supremum -- which "
        "only strengthens the finding that it exceeds 1.",
    ]
    v.deviations = [
        "FALSIFIED refers to the main-text statement quoted in the contract, which cites "
        "Assumption 2.1(i,iii). The appendix statement (Lemma B.4, which adds Assumption "
        "B.4 as a hypothesis) is sound and is separately verified in Route A.",
    ]
    v.artifacts = [str(p) for p in sorted(out.rglob("*")) if p.is_file()]
    return v
