"""Exact rational arithmetic for the pluralistic-curation update.

Floating point cannot settle a falsification: a reviewer is entitled to ask whether
m_{t+1} > m_t was rounding.  The trick that removes all doubt is to choose the
EXPONENTIATED rewards as exact rationals and define the rewards as their logarithms,

    w_i(x) := e^{r_i(x)} in Q,      r_i(x) := log w_i(x),

because the update (Eq. 5) and every quantity in Assumption 2.1 depend on the rewards
only through ratios of the w_i:

    x in S_{i,eps}   <=>   r_i(x) >= r_i* - eps    <=>   w_i(x) >= w_i* e^{-eps}
    Delta_i, delta   are logs of ratios of w's

So choosing eps = log(1/E) for a rational E makes the basin membership test a rational
comparison, and the whole trajectory stays in Q.  Nothing is rounded anywhere.
"""

from __future__ import annotations

import math
from fractions import Fraction as F


class ExactLandscape:
    """A finite two-reward landscape with exact rational e^{r_i} values."""

    def __init__(
        self,
        w1: dict[str, F],
        w2: dict[str, F],
        eps_ratio: F,
        name: str = "exact",
    ) -> None:
        """`eps_ratio` = e^{-eps} in Q, so eps = -log(eps_ratio) > 0 requires 0 < ratio < 1."""
        if not (0 < eps_ratio < 1):
            raise ValueError("eps_ratio must be in (0,1) so that eps > 0")
        self.atoms = list(w1)
        self.w1 = {k: F(v) for k, v in w1.items()}
        self.w2 = {k: F(v) for k, v in w2.items()}
        self.eps_ratio = F(eps_ratio)
        self.name = name
        self.w1_star = max(self.w1.values())
        self.w2_star = max(self.w2.values())
        # x in S_{i,eps}  <=>  w_i(x) >= w_i* * eps_ratio   -- an exact rational test
        self.S1 = [x for x in self.atoms if self.w1[x] >= self.w1_star * self.eps_ratio]
        self.S2 = [x for x in self.atoms if self.w2[x] >= self.w2_star * self.eps_ratio]
        self.outside = [x for x in self.atoms if x not in self.S1 and x not in self.S2]

    @property
    def eps(self) -> float:
        return -math.log(float(self.eps_ratio))

    # ---- Assumption 2.1, checked exactly ---------------------------------- #
    def audit(self, p0: dict[str, F]) -> dict:
        disjoint = not (set(self.S1) & set(self.S2))
        a0 = sum(p0[x] for x in self.S1)
        b0 = sum(p0[x] for x in self.S2)
        total = sum(p0.values())
        # outside gap: min over outside atoms of r_i* - r_i(x) = min log(w_i*/w_i(x))
        if self.outside:
            ratio1 = min(self.w1_star / self.w1[x] for x in self.outside)
            ratio2 = min(self.w2_star / self.w2[x] for x in self.outside)
            delta_ratio = min(ratio1, ratio2)
            delta_ok = delta_ratio > 1  # exact: strictly below optimal
        else:
            delta_ratio, delta_ok = F(0), False
        # cross gaps: x in S_2 => r_1(x) <= r_1* - Delta_1
        d1_ratio = min((self.w1_star / self.w1[x] for x in self.S2), default=F(0))
        d2_ratio = min((self.w2_star / self.w2[x] for x in self.S1), default=F(0))
        return {
            "A1_i_bounded_rewards": {
                "ok": all(v > 0 for v in list(self.w1.values()) + list(self.w2.values())),
                "detail": "every e^{r_i(x)} is a strictly positive rational, so r_i is finite and bounded on a finite X",
            },
            "A1_ii_basins_disjoint": {
                "ok": disjoint, "S1": self.S1, "S2": self.S2, "outside": self.outside,
            },
            "A1_ii_nontrivial_init": {
                "ok": a0 > 0 and b0 > 0,
                "p0_S1": str(a0), "p0_S2": str(b0),
                "p0_sums_to_one": total == 1, "p0_total": str(total),
            },
            "A1_iii_outside_gap": {
                "ok": delta_ok,
                "delta": math.log(float(delta_ratio)) if delta_ok else 0.0,
                "delta_ratio_exact": str(delta_ratio),
            },
            "A1_iii_cross_gaps": {
                "ok": d1_ratio > 1 and d2_ratio > 1,
                "Delta1": math.log(float(d1_ratio)) if d1_ratio > 1 else 0.0,
                "Delta2": math.log(float(d2_ratio)) if d2_ratio > 1 else 0.0,
                "Delta1_ratio_exact": str(d1_ratio), "Delta2_ratio_exact": str(d2_ratio),
            },
            "eps": self.eps,
            "eps_ratio_exact": str(self.eps_ratio),
        }

    # ---- exact dynamics --------------------------------------------------- #
    def multiplier(self, p: dict[str, F], q: F) -> dict[str, F]:
        z1 = sum(p[x] * self.w1[x] for x in self.atoms)
        z2 = sum(p[x] * self.w2[x] for x in self.atoms)
        return {x: q * self.w1[x] / z1 + (1 - q) * self.w2[x] / z2 for x in self.atoms}

    def step(self, p: dict[str, F], q: F) -> tuple[dict[str, F], dict[str, F]]:
        W = self.multiplier(p, q)
        return {x: p[x] * W[x] for x in self.atoms}, W

    def masses(self, p: dict[str, F]) -> tuple[F, F, F]:
        return (
            sum(p[x] for x in self.S1),
            sum(p[x] for x in self.S2),
            sum(p[x] for x in self.outside),
        )

    def run(self, p0: dict[str, F], q: F, steps: int) -> list[dict]:
        p = dict(p0)
        rows = []
        for t in range(steps):
            a, b, m = self.masses(p)
            p_next, W = self.step(p, q)
            a2, b2, m2 = self.masses(p_next)
            rows.append({
                "t": t,
                "a_t": float(a), "b_t": float(b), "m_t": float(m),
                "m_t_exact": str(m),
                "m_next_over_m_t": float(m2 / m) if m > 0 else float("nan"),
                "m_ratio_exact": str(m2 / m) if m > 0 else "nan",
                "m_increased": bool(m2 > m),          # EXACT rational comparison
                "sup_W_outside": float(max((W[x] for x in self.outside), default=0)),
                "W_basin1": float(W[self.S1[0]]) if self.S1 else float("nan"),
                "mass_conserved_exact": bool(sum(p_next.values()) == 1),
            })
            p = p_next
        return rows
