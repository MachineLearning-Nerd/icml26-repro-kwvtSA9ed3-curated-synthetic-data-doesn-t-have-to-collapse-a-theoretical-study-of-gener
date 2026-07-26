"""Render the report figures from RECOVERED run artifacts.

Every figure is drawn from data that was reconstructed out of a compute run's log and
verified (theory figures from SHA-256-framed artifact blocks, empirical figures from the
per-round tables parsed out of the E3/E4 logs -- see tools/extract_evidence.py). Nothing
here recomputes a result: if a number appears in a figure, a run produced it.

Usage:
    python tools/make_figures.py <artifact_root> <images_dir> [<evidence_dir>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False, "figure.autolayout": True,
})
FALSE_C, TRUE_C = "#c1121f", "#1d3557"


def fig1_claim1_falsification(art: Path, out: Path) -> str:
    """The mass that main-text Lemma 3.3 says must vanish instead goes to 1."""
    ce = pd.read_csv(art / "claim1/exact_and_continuous/exact_counterexample_trace.csv")
    ct = pd.read_csv(art / "claim1/exact_and_continuous/exact_control_trace.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))

    ax1.plot(ce.t, ce.m_t, color=FALSE_C, lw=2,
             label=f"counterexample  ($m_\\infty$={ce.m_t.iloc[-1]:.6f})")
    ax1.plot(ct.t, ct.m_t, color=TRUE_C, lw=2, ls="--",
             label=f"control  ($m_\\infty$={ct.m_t.iloc[-1]:.2e})")
    ax1.set_yscale("log")
    ax1.set_xlabel("retraining round $t$")
    ax1.set_ylabel("outside mass $m_t$")
    ax1.set_title("Lemma 3.3 predicts $m_t\\to 0$ in both cases")
    ax1.legend(fontsize=7.5, loc="center right")

    ax2.plot(ce.t, ce.m_next_over_m_t, color=FALSE_C, lw=2, label="counterexample")
    ax2.plot(ct.t, ct.m_next_over_m_t, color=TRUE_C, lw=2, ls="--", label="control")
    ax2.axhline(1.0, color="k", lw=0.9)
    ax2.set_xlabel("retraining round $t$")
    ax2.set_ylabel("$m_{t+1}/m_t$")
    ax2.set_title("contraction factor: $>1$ means the mass grows")
    ax2.legend(fontsize=7.5)

    p = out / "fig1_claim1_falsification.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def fig2_claim1_boundary(art: Path, out: Path) -> str:
    """The failure region is exactly the one the symbolic route predicts."""
    sw = pd.read_csv(art / "claim1/exact_and_continuous/counterexample_family_sweep.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for grew, marker, c, lab in ((True, "o", FALSE_C, "$m_t$ grew (lemma fails)"),
                                 (False, "x", TRUE_C, "$m_t$ decayed")):
        s = sw[sw.m_increased_observed == grew]
        ax.scatter(s.Delta, s.delta, marker=marker, s=46, c=c, label=lab,
                   edgecolors="none" if grew else None, zorder=3)
    D = np.linspace(sw.Delta.min(), sw.Delta.max(), 400)
    ax.plot(D, np.log(2.0 / (1.0 + np.exp(-D))), color="k", lw=1.4,
            label=r"symbolic boundary  $2e^{-\delta}=1+e^{-\Delta}$")
    ax.axhline(np.log(2), color="gray", ls=":", lw=1.1)
    ax.text(sw.Delta.max(), np.log(2), r" $\log 2$", va="center", fontsize=8, color="gray")
    ax.set_xlabel(r"cross-reward gap $\Delta$")
    ax.set_ylabel(r"outside-basin depth $\delta$")
    ax.set_title(f"Independent routes agree on every point ({int(sw.agree.sum())}/{len(sw)})")
    ax.legend(fontsize=7.5, loc="center", framealpha=0.95)
    p = out / "fig2_claim1_boundary.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def fig3_claim2_interval(art: Path, out: Path) -> str:
    """Every measured limit lands inside the paper's predicted interval."""
    sw = pd.read_csv(art / "claim2/continuous/continuous_q_d_sweep.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    # The interval is drawn as a band rather than as error bars: at large d it degenerates
    # to the single point {q}, and a_inf then sits outside it by float noise (quantified in
    # the title), which error bars cannot express.
    for d, mk in zip(sorted(sw.d.unique()), "os^Dv<>"):
        s = sw[sw.d == d].sort_values("q")
        ax1.fill_between(s.q, s.lower_L, s.upper_U, alpha=0.18, lw=0)
        ax1.plot(s.q, s.a_inf, mk, ms=4.5, label=f"$d$={d:g}")
    ax1.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
    ax1.set_xlabel("preference weight $q$")
    ax1.set_ylabel(r"limiting basin mass $a_\infty$")
    ax1.set_title(f"$a_\\infty$ inside $[L,U]$: {int(sw.inside_interval.sum())}/{len(sw)}"
                  f"\nmax excursion {sw.excursion.max():.2e} (float noise)")
    ax1.legend(fontsize=7, ncol=2)

    ax2.semilogy(sw.d, sw.eta_observed, "o", ms=5, color=TRUE_C)
    ax2.axhline(sw.eta_observed.min(), color=FALSE_C, ls="--", lw=1.2,
                label=f"min $\\eta$ = {sw.eta_observed.min():.4f} > 0")
    ax2.set_xlabel("landscape separation $d$")
    ax2.set_ylabel(r"non-collapse margin $\eta$")
    ax2.set_title("No configuration collapses")
    ax2.legend(fontsize=7.5)
    p = out / "fig3_claim2_interval.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def fig4_claim3_variance(art: Path, out: Path) -> str:
    """The variance lower bound holds on random mixtures with real within-basin spread."""
    rm = pd.read_csv(art / "claim3/continuous/random_mixture_variance.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax1.scatter(rm.bound_r1, rm.var_r1, s=5, alpha=0.35, c=TRUE_C, edgecolors="none")
    lim = [0, max(rm.var_r1.max(), rm.bound_r1.max()) * 1.02]
    ax1.plot(lim, lim, color=FALSE_C, lw=1.4, label="bound = variance (tight)")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel(r"predicted lower bound  $a(1-a)(\Delta_1-2\epsilon)^2$")
    ax1.set_ylabel(r"measured $\mathrm{Var}_{p}[r_1]$")
    ax1.set_title(f"{int(rm.holds_both.sum())}/{len(rm)} mixtures satisfy the bound "
                  f"(both rewards)")
    ax1.legend(fontsize=7.5, loc="upper left")

    ax2.hist(rm.within_var_r1, bins=60, color=TRUE_C, alpha=0.85)
    ax2.set_xlabel(r"within-basin variance of $r_1$")
    ax2.set_ylabel("count")
    ax2.set_title(f"Within-basin variance is strictly positive\n"
                  f"(min {rm.within_var_r1.min():.3f}) - not a plateau reduction")
    p = out / "fig4_claim3_variance.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def fig5_claim4_nash(art: Path, out: Path) -> str:
    """The dynamics land on the independently located Nash bargaining solution."""
    dv = pd.read_csv(art / "claim4/continuous/dynamics_vs_nash.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for q, mk in zip(sorted(dv.q.unique()), "os^Dv<>p"):
        s = dv[dv.q == q].sort_values("d")
        ax1.semilogy(s.d, np.maximum(s.abs_diff_dynamics_vs_nash, 1e-18), mk + "-",
                     ms=4.5, lw=1, label=f"$q$={q:g}")
    ax1.set_xlabel("landscape separation $d$")
    ax1.set_ylabel(r"$|a_\infty - \arg\max_\alpha N(\alpha)|$")
    far = dv[dv.d >= 5].abs_diff_dynamics_vs_nash.max()
    ax1.set_title(f"Dynamics vs Nash argmax  (max {far:.2e} at $d\\geq5$)")
    ax1.legend(fontsize=7, ncol=2)

    ax2.scatter(dv.q, dv.nash_argmax, s=42, c=FALSE_C, zorder=3,
                label=r"$\arg\max_\alpha N(\alpha)$ (50-digit bisection)")
    ax2.plot([0, 1], [0, 1], color=TRUE_C, lw=1.4, ls="--", label="$\\alpha^*=q$")
    ax2.set_xlabel("preference weight $q$")
    ax2.set_ylabel(r"Nash argmax $\alpha^*$")
    ax2.set_title(f"max $|\\alpha^*-q|$ = {dv.abs_diff_nash_vs_q.max():.1e}")
    ax2.legend(fontsize=7.5, loc="upper left")
    p = out / "fig5_claim4_nash.png"
    fig.savefig(p); plt.close(fig)
    return p.name


# --------------------------------------------------------------------------- #
# Empirical figures, from the per-round tables recovered out of the E3/E4 run logs.

def fig6_claim5_cifar(ev: Path, out: Path) -> str:
    """Wider pluralism sustains class diversity; single-reward and M=2 collapse."""
    arms = [("c5_single", "single reward (M=1)", FALSE_C, "-"),
            ("c5_m2", "pluralistic M=2", "#e07a5f", "--"),
            ("c5_m5", "pluralistic M=5", TRUE_C, "-")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for key, lab, c, ls in arms:
        d = pd.read_csv(ev / f"rounds_{key}.csv")
        ax1.plot(d["round"], d.class_entropy, ls, color=c, lw=2,
                 label=f"{lab}  (tail {d.class_entropy.tail(5).mean():.3f})")
        ax2.plot(d["round"], d.n_classes_present, ls, color=c, lw=2, label=lab)
    ax1.axhline(np.log(10), color="gray", ls=":", lw=1)
    ax1.text(1, np.log(10), " log 10 = max", fontsize=7, va="bottom", color="gray")
    ax1.set_xlabel("retraining round"); ax1.set_ylabel("class entropy of generations")
    ax1.set_title("CIFAR-10 OT-CFM retraining (E3)"); ax1.legend(fontsize=7)
    ax2.set_xlabel("retraining round"); ax2.set_ylabel("distinct classes generated")
    ax2.set_title("Modes surviving 25 recursive generations"); ax2.legend(fontsize=7.5)
    p = out / "fig6_claim5_cifar_entropy.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def fig7_claim6_text(ev: Path, out: Path) -> str:
    """Two length preferences keep both basins populated; one preference collapses."""
    single = pd.read_csv(ev / "rounds_c6_single.csv")
    duo = pd.read_csv(ev / "rounds_c6_d5.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax1.plot(single["round"], single.H_L, color=FALSE_C, lw=2,
             label=f"one preference  (tail {single.H_L.tail(5).mean():.3f})")
    ax1.plot(duo["round"], duo.H_L, color=TRUE_C, lw=2,
             label=f"two preferences  (tail {duo.H_L.tail(5).mean():.3f})")
    ax1.set_xlabel("retraining round"); ax1.set_ylabel("length entropy $H(L)$")
    ax1.set_title("GPT-2 text retraining (E4)"); ax1.legend(fontsize=7.5)

    ax2.plot(duo["round"], duo.frac_near_T_A, "o-", ms=3.5, color=TRUE_C,
             label="near $T_A$=10 (two prefs)")
    ax2.plot(duo["round"], duo.frac_near_T_B, "s-", ms=3.5, color="#457b9d",
             label="near $T_B$=15 (two prefs)")
    ax2.plot(single["round"], single.frac_near_T_A, "^--", ms=3.5, color=FALSE_C,
             label="near $T_A$=10 (one pref)")
    ax2.set_xlabel("retraining round"); ax2.set_ylabel("fraction of generations")
    ax2.set_title("Both preference basins stay populated"); ax2.legend(fontsize=7)
    p = out / "fig7_claim6_text_entropy.png"
    fig.savefig(p); plt.close(fig)
    return p.name


def main() -> int:
    art, out = Path(sys.argv[1]), Path(sys.argv[2])
    ev = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    out.mkdir(parents=True, exist_ok=True)
    for fn in (fig1_claim1_falsification, fig2_claim1_boundary, fig3_claim2_interval,
               fig4_claim3_variance, fig5_claim4_nash):
        print("wrote", fn(art, out))
    if ev:
        for fn in (fig6_claim5_cifar, fig7_claim6_text):
            print("wrote", fn(ev, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
