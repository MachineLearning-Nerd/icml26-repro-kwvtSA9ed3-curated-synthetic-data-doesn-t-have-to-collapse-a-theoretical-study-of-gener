"""Interactive walkthrough of the kwvtSA9ed3 reproduction.

Run with:  uv run marimo edit notebooks/reproduction.py

Everything here reads evidence RECOVERED from the compute runs -- it never recomputes a
verdict. The one exception is the Claim 1 counterexample, which is re-derived live in exact
rational arithmetic because that is cheap and it is the reproduction's central result:
you can watch the outside mass grow rather than take a table's word for it.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Curated Synthetic Data Doesn't Have to Collapse — reproduction

        **arXiv:2605.07724** · OpenReview `kwvtSA9ed3`

        Six claims, six derived verdicts, all gates passing on one CPU-only run
        (`3bf9f951-47af-4f1e-921d-b27606ed8522`).

        | # | Claim | Verdict |
        |---|---|---|
        | 1 | Lemma 3.3 — outside decay | **FALSIFIED** (main text) / **VERIFIED** (appendix) |
        | 2 | Lemma 3.5 — non-collapse | **VERIFIED** |
        | 3 | Theorem 3.6 — variance bound | **VERIFIED** |
        | 4 | Theorem 3.7 — Nash bargaining | **VERIFIED** |
        | 5 | E3 — CIFAR-10 flow retraining | **VERIFIED** (margin +1.1427) |
        | 6 | E4 — text retraining | **VERIFIED** (margin +2.4241) |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1 · Claim 1, re-derived live in exact arithmetic

        Main-text Assumption 2.1 does **not** include appendix Assumption B.4 (outside
        domination). Below, a 4-atom landscape that satisfies every audited clause of
        Assumption 2.1 is stepped forward in `fractions.Fraction`, so nothing is rounded and
        the growth cannot be a floating-point artifact.
        """
    )
    return


@app.cell
def _():
    import sys
    from fractions import Fraction as F

    sys.path.insert(0, "..")
    from repro.lib.exact import ExactLandscape

    land = ExactLandscape(
        w1={"A": F(1), "B": F(1, 100), "C": F(4, 5), "D": F(1, 1000)},
        w2={"A": F(1, 100), "B": F(1), "C": F(4, 5), "D": F(1, 1000)},
        eps_ratio=F(9, 10),
    )
    p0 = {"A": F(1, 1000), "B": F(1, 1000), "C": F(998, 1000), "D": F(0)}
    trace = land.run(p0, q=F(1, 2), steps=60)
    return F, land, p0, trace


@app.cell
def _(mo, trace):
    first, last = trace[0], trace[-1]
    mo.md(
        f"""
        | quantity | value |
        |---|---|
        | `m₁/m₀` (exact rational) | **`{first['m_ratio_exact']}`** |
        | is that > 1? | **{first['m_increased']}** — the lemma requires ≤ ρ < 1 |
        | `m_∞` after 60 steps | {float(last['m_t']):.10f} |
        | `a_∞` | {float(last['a_t']):.3e} |
        | mass exactly conserved every step | {all(r['mass_conserved_exact'] for r in trace)} |

        The conclusion does not merely lose its rate — it **inverts**: `m_t → 1` instead of
        `m_t → 0`.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Why this is a falsification and not a bug

        Two things make it stick:

        1. **A control.** Deepen the outside atom and the same code decays geometrically
           exactly as the lemma predicts (max ratio 0.95374, `m_∞ ≈ 9.2e-81`). The machinery
           is not simply broken.
        2. **An independent symbolic route.** sympy proves the sign-equivalence
           `W_t(C) > 1 ⟺ 2e^(−δ) > 1 + e^(−Δ)`, i.e. `δ < log 2` as `Δ → ∞`. That closed form
           predicts the exact-arithmetic outcome at all 24 sweep points, and the two routes
           share no code.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 2 · The recovered evidence

        orx local mode has no artifact store, so each run emits its artifacts into stdout,
        base64-framed with a declared byte count and SHA-256. The extractor rebuilds them and
        **rejects** any block whose hash does not match — a truncated log fails loudly rather
        than yielding plausible-looking wrong numbers.
        """
    )
    return


@app.cell
def _():
    from pathlib import Path

    import pandas as pd

    EV = Path("../.openresearch/evidence")
    ART = EV / "artifacts/final/.openresearch/artifacts"
    available = EV.exists()
    return ART, EV, Path, available, pd


@app.cell
def _(ART, available, mo, pd):
    if available:
        c5 = {k: pd.read_csv(f"../.openresearch/evidence/rounds_c5_{k}.csv")
              for k in ("single", "m2", "m5")}
        table = "\n".join(
            f"| {lab} | {d.class_entropy.iloc[0]:.4f} | "
            f"{d.class_entropy.tail(5).mean():.4f} | {int(d.n_classes_present.iloc[-1])} |"
            for lab, d in (("single reward (M=1)", c5["single"]),
                           ("pluralistic M=2", c5["m2"]),
                           ("pluralistic M=5", c5["m5"])))
        out = mo.md(f"""
        ### Claim 5 — CIFAR-10, 25 recursive generations

        | arm | round 1 | tail (last 5) | classes left |
        |---|---|---|---|
        {table}

        Margin (best pluralistic vs single): **+1.1427**. Note that **M=2 collapsed as
        completely as M=1** — pluralism per se is not sufficient at this scale, and that
        caveat is reported rather than dropped.
        """)
    else:
        out = mo.md(
            "_Run `python tools/extract_evidence.py` first to populate "
            "`.openresearch/evidence`._"
        )
    out
    return


@app.cell
def _(ART, available, mo, pd):
    if available:
        sw = pd.read_csv(ART / "claim2/continuous/continuous_q_d_sweep.csv")
        rm = pd.read_csv(ART / "claim3/continuous/random_mixture_variance.csv")
        dv = pd.read_csv(ART / "claim4/continuous/dynamics_vs_nash.csv")
        res = mo.md(f"""
        ### Claims 2, 3, 4 — on general continuous landscapes

        | claim | result |
        |---|---|
        | 2 · non-collapse | {int(sw.inside_interval.sum())}/{len(sw)} inside the predicted interval; max excursion {sw.excursion.max():.2e}; min margin η = {sw.eta_observed.min():.6f} |
        | 3 · variance bound | {int(rm.holds_both.sum())}/{len(rm)} random mixtures hold for **both** rewards; min within-basin variance {rm.within_var_r1.min():.4f} (strictly positive — not a plateau) |
        | 4 · Nash bargaining | max \\|α*−q\\| = {dv.abs_diff_nash_vs_q.max():.1e}; dynamics vs independent 50-digit bisection {dv[dv.d >= 5].abs_diff_dynamics_vs_nash.max():.2e} at d ≥ 5 |

        These run on continuous landscapes with free within-basin structure — not the 3-state
        reduction, which has zero intra-basin degrees of freedom and is why the original
        theory evidence was judged toy.
        """)
    else:
        res = mo.md("_Evidence directory not populated._")
    res
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 3 · How a verdict is decided

        No status is written by hand. Each stage records machine-checkable checks **and at
        least one negative control**, and a gate re-derives whether the recorded status is
        supported — failing if any check failed, if the verdict rests only on controls, or if
        no control was recorded. `repro.main` exits non-zero when any gate fails.

        The comparative claims add a **self-test**: the identical decision procedure is re-run
        on arm-swapped input and must return FALSIFIED. A test that cannot fail is not
        evidence.
        """
    )
    return


@app.cell
def _(available, mo):
    import json

    if available:
        V = json.load(open("../.openresearch/evidence/artifacts/final/"
                           ".openresearch/artifacts/verdicts.json"))
        rows = "\n".join(
            f"| `{v['claim_id']}` | **{v['status']}** | "
            f"{sum(not c.get('is_negative_control') for c in v['checks'])} | "
            f"{sum(bool(c.get('is_negative_control')) for c in v['checks'])} |"
            for v in V["verdicts"])
        g = mo.md(f"""
        | claim | verdict | checks | negative controls |
        |---|---|---|---|
        {rows}

        `all_gates_pass = {V['all_gates_pass']}` · git SHA `{V['git_sha'][:12]}` ·
        {V['cpu']['model_name']}, cgroup `cpu.max` = `{V['cpu']['cgroup_cpu_max']}`
        """)
    else:
        g = mo.md("_Evidence directory not populated._")
    g
    return


if __name__ == "__main__":
    app.run()
