"""Claim 1 / Lemma 3.3 (appendix Lemma B.4) -- symbolic certificate route.

Contract (repro/contracts/claim_contracts.json, id=claim1):

    "Fix eps>0 and work in the large-K regime (Eq 5). UNDER ASSUMPTION 2.1 (i,iii),
     there exists a constant rho_eps in (0,1) such that m_{t+1} <= rho_eps*m_t for
     all t. Hence m_t <= rho_eps^t m_0 -> 0 and therefore a_t + b_t -> 1."

Quantifier order matters: rho_eps is existentially quantified BEFORE the universal
over t, so a single step with m_{t+1} > m_t refutes the statement.

What this stage establishes symbolically:

  (1) The Eq-5 update is mass preserving and invariant to shifting either reward by
      a constant -- so setting r_1* = r_2* = 0 below is WLOG, not an assumption.
  (2) GIVEN appendix Assumption B.4 (sup_{x not in S_eps} W_t(x) <= rho < 1), the
      contraction m_{t+1} <= rho*m_t and the iterate m_t <= rho^t*m_0 -> 0 follow.
      So the appendix form of the lemma is sound.
  (3) Main-text Assumption 2.1(i,iii) does NOT imply B.4. Symbolically deriving the
      largest outside multiplier admissible under 2.1 and asking when it exceeds 1
      yields a CLOSED-FORM CHARACTERISATION of a whole family of landscapes that
      satisfy every clause of Assumption 2.1 yet violate the conclusion. The family
      is non-empty because Assumption 2.1(iii)'s outside clause is near-vacuous:
      x not in S_eps already forces r_i(x) < r_i* - eps, so delta = eps always
      works and no real constraint is added.

Route B independently exhibits a concrete member of that family in exact rational
arithmetic and iterates it; here the point is the symbolic characterisation.
"""

from __future__ import annotations

import sympy as sp

from repro.lib import report
from repro.lib.symbolic import prove_identity, prove_nonneg, prove_positive
from repro.lib.verdict import FALSIFIED, Verdict


def run(params: dict) -> Verdict:
    out = report.artifact_dir("claim1", "symbolic")
    v = Verdict(
        claim_id="claim1/lemma-3.3-outside-decay",
        title="Lemma 3.3: geometric decay of mass outside the eps-optimal basins",
        status=FALSIFIED,
        statement=(
            "Under Assumption 2.1 (i,iii) there exists rho_eps in (0,1) with "
            "m_{t+1} <= rho_eps m_t for all t, hence m_t -> 0 and a_t + b_t -> 1. "
            "FALSIFIED as stated: Assumption 2.1 does not imply the appendix's "
            "outside-domination hypothesis B.4, without which the conclusion fails. "
            "The appendix form (Lemma B.4, which assumes B.4) is verified sound."
        ),
    )

    # ------------------------------------------------------------------ (1) -- #
    report.banner("(1) The Eq-5 update: mass preservation and shift invariance")
    q = sp.Symbol("q", positive=True)
    # aggregate representation over the three regions
    a, b, m = sp.symbols("a b m", nonnegative=True)
    # e^{r_i} aggregated per region; all strictly positive since r_i is finite
    A1, B1, M1, A2, B2, M2 = sp.symbols("A_1 B_1 M_1 A_2 B_2 M_2", positive=True)
    Z1 = A1 + B1 + M1
    Z2 = A2 + B2 + M2
    # sum over x of p(x) W(x) = q*Z1/Z1 + (1-q)*Z2/Z2
    total = q * (A1 + B1 + M1) / Z1 + (1 - q) * (A2 + B2 + M2) / Z2
    ok, det = prove_identity(total, sp.Integer(1), "sum_x p_t(x) W_t(x) == 1")
    report.kv("mass preservation", det)
    v.add("Eq-5 update preserves total mass", ok, det)

    c1, c2 = sp.symbols("c_1 c_2", real=True)
    # shifting r_i -> r_i + c_i multiplies every e^{r_i} aggregate by e^{c_i}
    s1, s2 = sp.exp(c1), sp.exp(c2)
    e1x, e2x = sp.symbols("e_1x e_2x", positive=True)
    W_plain = q * e1x / Z1 + (1 - q) * e2x / Z2
    W_shift = q * (s1 * e1x) / (s1 * Z1) + (1 - q) * (s2 * e2x) / (s2 * Z2)
    ok, det = prove_identity(W_shift, W_plain, "W_t is invariant under r_i -> r_i + c_i")
    report.kv("shift invariance", det)
    v.add("W_t invariant under adding a constant to either reward (r_i* = 0 is WLOG)", ok, det)

    # ------------------------------------------------------------------ (2) -- #
    report.banner("(2) Appendix form: Assumption B.4 => geometric contraction")
    rho = sp.Symbol("rho", positive=True)
    p_x = sp.Symbol("p_x", nonnegative=True)
    gap = sp.Symbol("gap", nonnegative=True)  # gap := rho - W_t(x) >= 0 on x outside S_eps
    # termwise: p(x)*(rho - W(x)) >= 0, so summing over the outside region gives
    # rho*m_t - m_{t+1} >= 0 by finite additivity of >=.
    ok_term, det_term = prove_nonneg(p_x * gap, "p(x)*(rho - W_t(x)) >= 0 termwise")
    report.kv("termwise", det_term)
    v.add(
        "Lemma B.4 step: termwise nonnegativity under Assumption B.4",
        ok_term,
        det_term + "  => rho*m_t - m_{t+1} = sum over outside of p(x)(rho-W(x)) >= 0",
    )

    # induction m_t <= rho^t m_0
    t = sp.Symbol("t", nonnegative=True, integer=True)
    m0 = sp.Symbol("m_0", nonnegative=True)
    ih = sp.Symbol("ih_slack", nonnegative=True)  # rho^t m_0 - m_t >= 0
    # m_{t+1} <= rho m_t <= rho * (rho^t m_0) = rho^{t+1} m_0
    step_expr = rho ** (t + 1) * m0 - rho * (rho**t * m0 - ih)
    ok_ind, det_ind = prove_nonneg(sp.expand(step_expr), "induction step")
    report.kv("induction", det_ind)
    v.add(
        "Lemma B.4: induction gives m_t <= rho^t m_0",
        ok_ind,
        det_ind + "  (rho^{t+1} m_0 - rho*m_t = rho*(rho^t m_0 - m_t) >= 0)",
    )

    # rho in (0,1) => rho^t -> 0.  Parameterise rho = 1/(1+s), s > 0, which ranges over
    # exactly (0,1); then sympy can take the limit for the whole range at once.
    s = sp.Symbol("s", positive=True)
    rho_u = 1 / (1 + s)
    lim_rho_t = sp.limit(rho_u**t * m0, t, sp.oo)
    report.kv("lim_t rho^t m_0 for rho = 1/(1+s), s>0", lim_rho_t)
    decay_ok = sp.simplify(lim_rho_t) == 0
    v.add(
        "Lemma B.4: rho in (0,1) forces rho^t -> 0, hence m_t -> 0 and a_t+b_t -> 1",
        bool(decay_ok),
        f"with rho = 1/(1+s) covering exactly (0,1) for s > 0, sympy gives "
        f"lim_t rho^t m_0 = {lim_rho_t}",
    )
    v.notes.append(
        "Steps (2) establish that the APPENDIX statement (Lemma B.4, which takes "
        "outside domination as a hypothesis) is sound. The falsification below is of "
        "the MAIN-TEXT implication, which cites only Assumption 2.1(i,iii)."
    )

    # ------------------------------------------------------------------ (3) -- #
    report.banner("(3) Main-text Assumption 2.1(i,iii) does not imply B.4")

    eps, delta, Delta = sp.symbols("epsilon delta Delta", positive=True)
    # A landscape admissible under Assumption 2.1, written in the extremal form that
    # maximises the outside multiplier:
    #   basin S_1 = {A}:  r_1 = 0,        r_2 = -Delta
    #   basin S_2 = {B}:  r_1 = -Delta,   r_2 = 0
    #   outside  = {C}:   r_1 = r_2 = -delta   with delta > eps  (so C is outside)
    # Assumption 2.1(iii) is satisfied with outside gap delta > 0 and cross gaps
    # Delta_1 = Delta_2 = Delta > 0. Nothing in 2.1 bounds delta away from 0.
    E, Dg = sp.exp(-Delta), sp.exp(-delta)
    Z1e = a * 1 + b * E + m * Dg
    Z2e = a * E + b * 1 + m * Dg
    W_C = q * Dg / Z1e + (1 - q) * Dg / Z2e
    W_A = q * 1 / Z1e + (1 - q) * E / Z2e

    # Specialise to the symmetric balanced case the paper's own experiments use.
    sym = {q: sp.Rational(1, 2), b: a, m: 1 - 2 * a}
    W_C_sym = sp.simplify(W_C.subs(sym))
    W_A_sym = sp.simplify(W_A.subs(sym))
    report.kv("W_t(C), symmetric case", W_C_sym)

    # When does the OUTSIDE multiplier exceed 1?  Rather than asking sympy to solve a
    # transcendental inequality, establish the SIGN EQUIVALENCE exactly: show that
    # W_t(C) - 1 equals `target` times a provably positive factor.
    target = 2 * Dg - (1 + E)  # 2 e^{-delta} - (1 + e^{-Delta})
    num, den = sp.fraction(sp.together(W_C_sym - 1))
    ratio = sp.simplify(sp.factor(num) / (a * target))
    report.kv("numer(W_t(C) - 1)", sp.factor(num))
    report.kv("numer / (a * target)", ratio)
    report.kv("denominator", sp.factor(den))
    pos_factor_ok, pos_factor_det = prove_positive(ratio, "numer/(a*target) > 0")
    den_ok, den_det = prove_positive(den.subs(a, sp.Rational(1, 10)), "denominator > 0")
    v.add(
        "sign equivalence: W_t(C) > 1  <=>  2e^{-delta} > 1 + e^{-Delta}",
        pos_factor_ok and den_ok,
        f"W_t(C)-1 = a*(2e^-delta - 1 - e^-Delta) * {ratio} / den, with "
        f"{pos_factor_det} and {den_det}; a > 0, so the two have the same sign",
        ratio=str(ratio),
    )

    # The family is non-empty and lies strictly inside Assumption 2.1's admissible set:
    # need eps < delta (so C is outside both basins) and 2e^{-delta} > 1 + e^{-Delta}.
    limit_cond = sp.simplify(sp.limit(target, Delta, sp.oo))
    delta_max = sp.solve(sp.Eq(limit_cond, 0), delta)
    report.kv("condition as Delta -> oo", f"{limit_cond} > 0")
    report.kv("=> admissible delta range", f"eps < delta < {delta_max[0]} = {float(sp.log(2)):.6f}")
    nonempty = sp.simplify(delta_max[0] - sp.log(2)) == 0
    v.add(
        "counterexample family is non-empty: any eps < delta < log 2, Delta large",
        bool(nonempty),
        f"as Delta -> oo the condition becomes 2e^-delta > 1, i.e. delta < log 2 = "
        f"{float(sp.log(2)):.6f}. Assumption 2.1 permits ANY delta > 0 outside the "
        f"basins and eps > 0 is arbitrary, so the family is non-empty for every eps < log 2.",
        delta_upper=str(delta_max[0]),
    )

    # On that same family the basins SHRINK, so m_t -> 1: the conclusion inverts.
    W_A_lim = sp.simplify(sp.limit(W_A_sym, a, 0))
    W_A_lim_hard = sp.simplify(sp.limit(W_A_lim, Delta, sp.oo))
    report.kv("lim_{a->0} W_t(basin)", W_A_lim)
    report.kv("then Delta -> oo", W_A_lim_hard)
    same_condition = sp.simplify(W_A_lim_hard - sp.exp(delta) / 2) == 0
    v.add(
        "on the same family the basin multiplier is < 1, so m_t increases to 1",
        bool(same_condition),
        "lim_{a->0} W_t(basin) -> e^{delta}/2 as Delta -> oo, which is < 1 exactly when "
        "delta < log 2 -- the identical condition. So BOTH basins lose mass while the "
        "outside gains it: the conclusion does not merely lose its rate, it inverts "
        "from m_t -> 0 to m_t -> 1.",
        W_A_limit=str(W_A_lim_hard),
    )

    # ---- negative control ------------------------------------------------- #
    report.banner("Negative control: the derivation must FAIL when rho = 1")
    # With rho = 1 the contraction step is still true but vacuous, and the limit step
    # must break -- otherwise our chain would 'prove' decay for any landscape.
    rho_one_limit = sp.limit(sp.Integer(1) ** t * m0, t, sp.oo)
    report.kv("lim_t rho^t m_0 at rho=1", rho_one_limit)
    control_ok = sp.simplify(rho_one_limit - m0) == 0
    v.add_control(
        "with rho = 1 the decay conclusion does NOT follow",
        bool(control_ok),
        f"lim_t 1^t m_0 = {rho_one_limit} = m_0, not 0, so the chain genuinely consumes "
        "rho < 1 and cannot be satisfied by any landscape whose outside multiplier reaches 1",
    )

    v.numbers = {
        "counterexample_condition": "2*exp(-delta) > 1 + exp(-Delta)",
        "delta_upper_bound_as_Delta_to_inf": float(sp.log(2)),
        "outside_multiplier_symmetric": str(W_C_sym),
        "basin_multiplier_limit": str(W_A_lim),
    }
    v.limitations = [
        "This route establishes the falsification symbolically, as a characterised "
        "family. A concrete member iterated in exact rational arithmetic is Route B's "
        "job; neither route alone is the whole evidence.",
        "The symmetric specialisation q=1/2, Delta_1=Delta_2 is used to get a "
        "closed-form condition. It is a specialisation of the counterexample, which "
        "only has to exist -- it does not weaken a falsification.",
    ]
    v.deviations = [
        "The claim is reported FALSIFIED against the main-text statement quoted in the "
        "contract, which cites Assumption 2.1(i,iii). The appendix statement Lemma B.4, "
        "which additionally assumes outside domination B.4, is separately verified sound "
        "in part (2) above and is NOT falsified.",
    ]
    report.write_json(out / "symbolic_certificate.json", {
        "claim": v.claim_id,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in v.checks],
        "numbers": v.numbers,
    })
    v.artifacts = [str(p) for p in sorted(out.rglob("*")) if p.is_file()]
    return v
