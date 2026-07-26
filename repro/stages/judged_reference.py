"""BASELINE REFERENCE ONLY -- reproduces the currently judged (5/12) toy evidence.

The judged Hugging Face revision DineshAI/kwvtSA9ed3@07f2114 checked all four
theoretical claims on a **3-macro-state plateau reduction** (basin1, basin2,
outside).  The live judge marked all four TOY because a 3-state system is not the
general two-basin landscape with continuous distributions that the lemmas quantify
over.

This stage re-derives those same reference numbers so that every child node is
measured against a fixed, reproducible starting point.  It deliberately records
status BLOCKED: a 3-state reduction cannot verify a statement quantified over all
bounded two-basin landscapes.  Nothing here is presented as full-scale evidence.
"""

from __future__ import annotations

import numpy as np

from repro.lib import report
from repro.lib.landscape import plateau_landscape
from repro.lib.verdict import BLOCKED, Verdict


def _log_slope(m: np.ndarray) -> float:
    """Least-squares slope of log m_t against t, over the numerically usable range."""
    keep = m > 1e-300
    t = np.arange(len(m))[keep]
    return float(np.polyfit(t, np.log(m[keep]), 1)[0])


def run(params: dict) -> Verdict:
    eps = float(params.get("eps", 0.1))
    out = report.artifact_dir("baseline", "judged_reference")

    v = Verdict(
        claim_id="baseline/judged-3state-reference",
        title="Reference reproduction of the judged 3-macro-state plateau evidence",
        status=BLOCKED,
        statement=(
            "Reference control only. Reproduces the judged logbook's numbers on the "
            "3-macro-state plateau reduction (Proposition 3.4 idealisation). A "
            "3-state system cannot verify statements universally quantified over "
            "bounded two-basin landscapes with continuous distributions."
        ),
    )

    # ---- judged C0: outside-mass decay slope on the 3-state reduction ------ #
    report.banner("C0 reference: outside-mass decay on the 3-state plateau reduction")
    # delta_outside chosen so the asymptotic contraction factor reproduces the
    # judged logbook's headline rho_eps = 0.095 (slope of log m_t = -2.35):
    #   rho = e^{-delta_c} / (a + b e^{-Delta_1}) with a = b = 1/2, Delta_1 = 5.
    lam, p0 = plateau_landscape(delta1=5.0, delta2=5.0, delta_outside=3.0405, eps=eps)
    res = lam.run(p0, q=0.5, steps=56)
    m = np.array([r["m_t"] for r in res["trace"]])
    slope = _log_slope(m)
    rho = float(np.exp(slope))
    report.kv("m_0", f"{m[0]:.6g}")
    report.kv("m_56", f"{m[-1]:.6g}")
    report.kv("slope of log m_t", f"{slope:.4f}")
    report.kv("implied rho_eps = exp(slope)", f"{rho:.4f}")
    report.write_csv(out / "c0_reference_trace.csv", res["trace"])
    v.add(
        "C0-ref: geometric decay on 3-state reduction",
        bool(rho < 1.0 and m[-1] < m[0]),
        f"slope={slope:.4f}, rho={rho:.4f}, m: {m[0]:.3g} -> {m[-1]:.3g}",
        slope=slope, rho=rho, m0=float(m[0]), m_final=float(m[-1]),
    )
    v.numbers["c0_slope"] = slope
    v.numbers["c0_rho"] = rho

    # ---- judged C1: a_inf vs leakage interval, 3-state reduction ----------- #
    report.banner("C1 reference: a_inf against the leakage interval (q=0.3)")
    q = 0.3
    rows = []
    for delta in (1.5, 3.0, 6.0, 12.0):
        lam_d, p0_d = plateau_landscape(delta, delta, delta_outside=5.0, eps=eps)
        res_d = lam_d.run(p0_d, q=q, steps=400)
        a_inf = res_d["trace"][-1]["a_t"]
        k1, k2 = lam_d.leakage()
        lo = max(0.0, (q - k1) / (1 - k1))
        hi = min(1.0, q / (1 - k2))
        inside = lo - 1e-12 <= a_inf <= hi + 1e-12
        rows.append({"Delta": delta, "kappa": k1, "lower": lo, "a_inf": a_inf,
                     "upper": hi, "inside_interval": inside})
        report.kv(f"Delta={delta:<5g} a_inf", f"{a_inf:.6f}   interval [{lo:.6f}, {hi:.6f}]  inside={inside}")
    report.write_csv(out / "c1_reference_leakage.csv", rows)
    a_infs = [r["a_inf"] for r in rows]
    v.add(
        "C1-ref: a_inf inside leakage interval and -> q on 3-state reduction",
        all(r["inside_interval"] for r in rows) and abs(a_infs[-1] - q) < 5e-3,
        f"a_inf = {[round(x, 4) for x in a_infs]} -> q={q}",
        a_infs=a_infs, q=q,
    )
    v.numbers["c1_a_infs"] = a_infs

    # ---- judged C2: variance identity on the plateau reduction ------------- #
    report.banner("C2 reference: plateau variance identity")
    q2, delta = 0.5, 4.0
    lam2, p02 = plateau_landscape(delta, delta, delta_outside=6.0, eps=eps)
    res2 = lam2.run(p02, q=q2, steps=400)
    a_inf = res2["trace"][-1]["a_t"]
    var1 = res2["trace"][-1]["var_r1"]
    identity = a_inf * (1 - a_inf) * delta**2
    bound = a_inf * (1 - a_inf) * max(delta - 2 * eps, 0.0) ** 2
    report.kv("a_inf", f"{a_inf:.6f}")
    report.kv("Var_p_inf[r1] (measured)", f"{var1:.6f}")
    report.kv("a(1-a)Delta^2 (plateau identity)", f"{identity:.6f}")
    report.kv("a(1-a)(Delta-2eps)^2 (Thm 3.6 bound)", f"{bound:.6f}")
    report.write_json(out / "c2_reference_variance.json",
                      {"a_inf": a_inf, "var_r1": var1, "identity": identity, "bound": bound})
    v.add(
        "C2-ref: plateau identity matches and clears the bound",
        abs(var1 - identity) < 1e-9 and var1 >= bound - 1e-12,
        f"Var={var1:.6f} == a(1-a)D^2={identity:.6f} >= bound={bound:.6f}",
        var_r1=var1, identity=identity, bound=bound,
    )
    v.numbers["c2_var_r1"] = var1
    v.numbers["c2_bound"] = bound

    # ---- judged C3: Nash product grid argmax ------------------------------- #
    report.banner("C3 reference: Nash product grid argmax (the judged 20001-point grid)")
    alpha = np.linspace(0.0, 1.0, 20001)
    rows = []
    for qq in (0.25, 0.4, 0.5, 0.65, 0.8):
        with np.errstate(divide="ignore", invalid="ignore"):
            f = alpha**qq * (1 - alpha) ** (1 - qq)
        a_star = float(alpha[np.nanargmax(f)])
        rows.append({"q": qq, "alpha_star_grid": a_star, "abs_err": abs(a_star - qq)})
        report.kv(f"q={qq:<5g} grid argmax", f"{a_star:.6f}  |err|={abs(a_star - qq):.2e}")
    report.write_csv(out / "c3_reference_nash_grid.csv", rows)
    max_err = max(r["abs_err"] for r in rows)
    v.add(
        "C3-ref: grid argmax equals q to grid resolution",
        max_err <= 1e-4,
        f"max |alpha*-q| = {max_err:.2e} on a 20001-point grid",
        max_abs_err=max_err,
    )
    v.numbers["c3_max_abs_err"] = max_err

    # ---- negative control: the reduction cannot see intra-basin structure -- #
    report.banner("Negative control: does the 3-state reduction have any intra-basin degrees of freedom?")
    n_atoms = int(lam.r1.size)
    intra_basin_dof = int(lam.S1.sum()) - 1 + int(lam.S2.sum()) - 1
    report.kv("atoms in the reduction", n_atoms)
    report.kv("intra-basin degrees of freedom", intra_basin_dof)
    v.add_control(
        "3-state reduction has zero intra-basin degrees of freedom",
        intra_basin_dof == 0,
        f"{n_atoms} atoms, {intra_basin_dof} intra-basin dof -- so the basin-conditional "
        "distributions are constants by construction, not a verified consequence. This is "
        "precisely why the judge marked the four theory claims TOY.",
        n_atoms=n_atoms, intra_basin_dof=intra_basin_dof,
    )

    v.limitations = [
        "3-macro-state plateau reduction: each basin is a single atom, so the "
        "basin-conditional distributions are trivially invariant and every theorem "
        "holds with equality by construction rather than being tested.",
        "No continuous distributions, no intra-basin dynamics, no general reward "
        "geometry -- therefore no evidence about the universally quantified lemmas.",
        "This stage exists only as the frozen reference control for child nodes.",
    ]
    v.deviations = [
        "Status is deliberately recorded BLOCKED, not PASS: this reproduces the "
        "judged evidence and inherits its scope limits.",
    ]
    v.artifacts = [str(p) for p in sorted(out.rglob("*")) if p.is_file()]
    return v
