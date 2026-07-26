"""Claim 3 / Theorem 3.6 (appendix Theorem B.6) -- symbolic certificate route.

Contract (id=claim3): for any two-basin limit p_inf = a P_1 + (1-a) P_2 with
a in (0,1), P_i supported on S_{i,eps}, and separation as in B.5,

    Var_{p_inf}[r_i]  >=  a(1-a) * max{Delta_i - 2 eps, 0}^2      for i = 1, 2.

The judged evidence checked the plateau EQUALITY Var = a(1-a)Delta^2, which is a
single point of the domain: on a 3-state reduction each basin is one atom, so the
within-basin variances are identically zero and the basin means are forced. Here the
decomposition is derived for ARBITRARY within-basin distributions, carrying symbolic
within-basin variances that the plateau case sets to zero.
"""

from __future__ import annotations

import sympy as sp

from repro.lib import report
from repro.lib.symbolic import prove_identity, prove_nonneg
from repro.lib.verdict import VERIFIED, Verdict


def run(params: dict) -> Verdict:
    out = report.artifact_dir("claim3", "symbolic")
    v = Verdict(
        claim_id="claim3/theorem-3.6-variance",
        title="Theorem 3.6: strictly positive reward variance in any nontrivial two-basin limit",
        status=VERIFIED,
        statement=(
            "For a two-basin limit p_inf = a P_1 + (1-a) P_2 with a in (0,1) and "
            "basin separation, Var_{p_inf}[r_i] >= a(1-a) max{Delta_i-2eps,0}^2 for i=1,2."
        ),
    )

    a = sp.Symbol("a", positive=True)
    eps = sp.Symbol("epsilon", positive=True)
    # ARBITRARY within-basin distributions: carry their first two moments as symbols.
    mu1, mu2 = sp.symbols("mu_1 mu_2", real=True)          # basin-conditional means of r_i
    s1sq, s2sq = sp.symbols("sigma_1^2 sigma_2^2", nonnegative=True)  # within-basin variances

    # ------------------------------------------------------------------ (1) -- #
    report.banner("(1) Law of total variance for a two-component mixture (arbitrary components)")
    mean = a * mu1 + (1 - a) * mu2
    # E[r^2] under the mixture = a(sigma_1^2 + mu_1^2) + (1-a)(sigma_2^2 + mu_2^2)
    second = a * (s1sq + mu1**2) + (1 - a) * (s2sq + mu2**2)
    var_mix = sp.expand(second - mean**2)
    decomposition = a * s1sq + (1 - a) * s2sq + a * (1 - a) * (mu1 - mu2) ** 2
    ok_dec, det_dec = prove_identity(
        var_mix, decomposition,
        "Var = a s_1^2 + (1-a) s_2^2 + a(1-a)(mu_1-mu_2)^2",
    )
    report.kv("Var_{p_inf}[r_i]", sp.factor(var_mix))
    report.kv("identity check", det_dec)
    v.add(
        "law of total variance holds for ARBITRARY within-basin components",
        ok_dec, det_dec,
    )
    v.notes.append(
        "The within-basin variances sigma_i^2 are carried as free symbols. The judged "
        "3-state reduction is the single point sigma_1^2 = sigma_2^2 = 0, which is why "
        "it saw an equality rather than the inequality the theorem asserts."
    )

    # ------------------------------------------------------------------ (2) -- #
    report.banner("(2) Dropping the nonnegative within-basin terms")
    drop = sp.simplify(var_mix - a * (1 - a) * (mu1 - mu2) ** 2)
    # a is declared positive but sympy also needs a < 1 to sign (1-a). Reparameterise
    # a = z/(1+z) with z > 0, which sweeps exactly the hypothesis a_inf in (0,1).
    z = sp.Symbol("z", positive=True)
    drop_param = sp.simplify(drop.subs(a, z / (1 + z)))
    ok_drop, det_drop = prove_nonneg(drop_param, "a s_1^2 + (1-a) s_2^2 >= 0")
    report.kv("dropped part", sp.simplify(drop))
    report.kv("dropped part with a = z/(1+z)", drop_param)
    v.add(
        "Var >= a(1-a)(mu_{i,1}-mu_{i,2})^2 after dropping within-basin variance",
        ok_drop,
        det_drop + " (requires a in (0,1), which is the theorem's hypothesis a_inf in (0,1))",
    )

    # ------------------------------------------------------------------ (3) -- #
    report.banner("(3) The separation bound on the basin-mean gap")
    D = sp.Symbol("Delta_i", positive=True)
    rstar = sp.Symbol("r_star", real=True)
    # P_1 supported on S_{1,eps}  => mu_{1,1} >= r_1* - eps
    # separation (B.5)            => mu_{1,2} <= r_1* - Delta_1 + eps
    slack_hi = sp.Symbol("t_hi", nonnegative=True)   # mu_{i,1} = r* - eps + t_hi
    slack_lo = sp.Symbol("t_lo", nonnegative=True)   # mu_{i,2} = r* - Delta + eps - t_lo
    m_hi = rstar - eps + slack_hi
    m_lo = rstar - D + eps - slack_lo
    gap = sp.simplify(m_hi - m_lo)
    report.kv("mu_{i,1} - mu_{i,2}", gap)
    excess = sp.simplify(gap - (D - 2 * eps))
    ok_gap, det_gap = prove_nonneg(excess, "gap - (Delta_i - 2 eps) >= 0")
    v.add(
        "basin-mean gap satisfies mu_{i,1} - mu_{i,2} >= Delta_i - 2 eps",
        ok_gap,
        f"mu_{{i,1}} - mu_{{i,2}} = {gap}; subtracting (Delta_i - 2eps) leaves {excess} >= 0. "
        "Uses ONLY support in S_{i,eps} and the B.5 separation inequality, for arbitrary "
        "within-basin shape.",
    )

    # squaring is monotone on the nonnegative reals, and the positive part handles Delta<=2eps
    g = sp.Symbol("g", nonnegative=True)      # g := Delta_i - 2 eps, in the regime Delta>2eps
    extra = sp.Symbol("extra", nonnegative=True)   # gap = g + extra
    sq_excess = sp.expand((g + extra) ** 2 - g**2)
    ok_sq, det_sq = prove_nonneg(sq_excess, "(g+extra)^2 - g^2 >= 0")
    report.kv("squaring step", det_sq)
    v.add(
        "squaring preserves the inequality, giving Var >= a(1-a)(Delta_i-2eps)_+^2",
        ok_sq,
        det_sq + "; for Delta_i <= 2 eps the positive part makes the bound 0, which any "
        "variance satisfies trivially",
    )

    # ------------------------------------------------------------------ (4) -- #
    report.banner("(4) Both indices i = 1, 2 (the argument is symmetric)")
    # For i=2 the roles of the basins swap: mu_{2,2} >= r_2*-eps, mu_{2,1} <= r_2*-Delta_2+eps.
    # The prefactor a(1-a) is symmetric under a -> 1-a, so the same bound results.
    pref = a * (1 - a)
    ok_sym, det_sym = prove_identity(pref, pref.subs(a, 1 - a), "a(1-a) symmetric under a -> 1-a")
    report.kv("prefactor symmetry", det_sym)
    v.add(
        "the i=2 bound follows by the same argument with the basins swapped",
        ok_sym,
        det_sym + "; the prefactor a(1-a) is invariant, so Var[r_2] >= a(1-a)(Delta_2-2eps)_+^2",
    )

    # ---- negative controls -------------------------------------------- #
    report.banner("Negative controls")
    at_zero = sp.simplify(pref.subs(a, 0))
    v.add_control(
        "the bound is vacuous at a = 0 or a = 1 (degenerate limit)",
        sp.simplify(at_zero) == 0,
        f"a(1-a) = {at_zero} at a=0, so the theorem's hypothesis a_inf in (0,1) is "
        "genuinely required: a collapsed limit gets no positive lower bound",
    )
    vacuous = sp.Max(sp.Rational(-1, 2), 0)
    v.add_control(
        "the bound is vacuous when Delta_i <= 2 eps",
        sp.simplify(vacuous) == 0,
        "max{Delta_i - 2eps, 0} = 0 there, so the theorem asserts nothing about "
        "positive variance without genuine separation -- it cannot be used to claim "
        "diversity for overlapping basins",
    )

    v.numbers = {
        "decomposition": "Var = a*sigma_1^2 + (1-a)*sigma_2^2 + a(1-a)(mu_1-mu_2)^2",
        "bound": "a(1-a)*max(Delta_i - 2*eps, 0)^2",
    }
    v.limitations = [
        "This certifies the algebraic core of Theorem B.6 for arbitrary within-basin "
        "components. The weak-convergence step Var_{p_tk} -> Var_{p_inf} (Portmanteau "
        "with bounded continuous r_i) is not formalised symbolically; Route B "
        "corroborates it numerically on continuous landscapes.",
    ]
    report.write_json(out / "symbolic_certificate.json", {
        "claim": v.claim_id,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in v.checks],
        "numbers": v.numbers,
    })
    v.artifacts = [str(p) for p in sorted(out.rglob("*")) if p.is_file()]
    return v
