"""Shared machinery for the symbolic proof certificates (Route A).

The proofs in Appendix B.3 never touch individual atoms: every step is a statement
about the *aggregates*

    A_i := integral over S_{1,eps} of e^{r_i} dp_t
    B_i := integral over S_{2,eps} of e^{r_i} dp_t
    M_i := integral over X \\ S_eps  of e^{r_i} dp_t          (so Z_i = A_i+B_i+M_i)

together with the masses (a_t, b_t, m_t).  Reformulating a proof over these
finitely many symbols is therefore *exact*, not a discretisation, and it lets
sympy verify each step for ALL admissible parameter values at once -- which is
what a universally quantified lemma needs and a grid search can never give.

Convention throughout: r_1* = r_2* = 0 (WLOG, since the update is invariant under
adding a constant to r_i -- Z_i absorbs it; that invariance is itself verified in
`symbolic_claim1.check_shift_invariance`).
"""

from __future__ import annotations

import sympy as sp


def prove_nonneg(expr: sp.Expr, name: str) -> tuple[bool, str]:
    """Verify expr >= 0 for every value consistent with its symbols' assumptions."""
    simplified = sp.simplify(expr)
    verdict = simplified.is_nonnegative
    if verdict is True:
        return True, f"{name}: sympy proves {simplified} >= 0 from symbol assumptions"
    # Fall back to a factorisation whose factors are individually signed.
    factored = sp.factor(simplified)
    if factored.is_nonnegative is True:
        return True, f"{name}: factored as {factored} >= 0"
    return False, f"{name}: could NOT establish {simplified} >= 0 (is_nonnegative={verdict})"


def prove_identity(lhs: sp.Expr, rhs: sp.Expr, name: str) -> tuple[bool, str]:
    """Verify lhs == rhs as an algebraic identity."""
    diff = sp.simplify(sp.together(sp.expand(lhs - rhs)))
    ok = sp.simplify(diff) == 0
    return ok, f"{name}: simplify(lhs - rhs) = {diff}" + ("" if ok else "  <-- NOT zero")


def prove_positive(expr: sp.Expr, name: str) -> tuple[bool, str]:
    simplified = sp.simplify(expr)
    if simplified.is_positive is True:
        return True, f"{name}: sympy proves {simplified} > 0"
    factored = sp.factor(simplified)
    if factored.is_positive is True:
        return True, f"{name}: factored as {factored} > 0"
    return False, f"{name}: could NOT establish {simplified} > 0"


def implies_under_constraints(
    hypothesis: list[sp.Expr], conclusion: sp.Expr, syms: list[sp.Symbol], name: str
) -> tuple[bool, str]:
    """Check that `conclusion >= 0` follows from `h >= 0 for h in hypothesis`.

    Strategy: substitute each hypothesis by a fresh non-negative slack symbol and
    ask sympy to establish the conclusion's sign.  Used where a bare `simplify`
    cannot see the sign because it depends on the hypotheses.
    """
    subs = {}
    slacks = []
    for k, h in enumerate(hypothesis):
        s = sp.Symbol(f"slack_{k}", nonnegative=True)
        slacks.append((h, s))
        subs[h] = s
    expr = conclusion
    for h, s in slacks:
        expr = expr.subs(h, s)
    return prove_nonneg(expr, name)
