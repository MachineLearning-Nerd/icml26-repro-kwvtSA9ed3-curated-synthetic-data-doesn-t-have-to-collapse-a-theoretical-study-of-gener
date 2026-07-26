"""Two-basin reward landscapes and the pluralistic-curation retraining dynamics.

Everything here follows arXiv:2605.07724 (ar5iv rendering, see docs/source_audit.md):

  * Eq. (5) / Lemma 3.2  large-K pluralistic update
        p_{t+1}(x) = p_t(x) * ( q e^{r1(x)}/Z1(t) + (1-q) e^{r2(x)}/Z2(t) ),
        Z_i(t) = E_{p_t}[ e^{r_i } ]
  * Definition 2.1 / Lemma 3.1  finite-K Bradley-Terry curation weight
  * Assumption 2.1 (main text) and Assumptions B.1-B.6 (appendix)

A distribution is represented as a vector of probability *masses* over atoms.
For a continuous landscape the atoms are grid cells, `coords` holds cell centres
and `cell_volume` the cell measure, so densities are mass / cell_volume.  Because
the update is a pointwise multiply-and-renormalise, a cell-wise representation is
exact for cell-wise constant rewards and converges to the continuum otherwise --
`stages/` always reports a grid-refinement study alongside any continuum claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# landscape
# --------------------------------------------------------------------------- #
@dataclass
class Landscape:
    """A bounded two-reward landscape on a finite set of atoms."""

    r1: np.ndarray
    r2: np.ndarray
    eps: float
    name: str = "landscape"
    coords: np.ndarray | None = None
    cell_volume: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.r1 = np.asarray(self.r1, dtype=np.float64)
        self.r2 = np.asarray(self.r2, dtype=np.float64)
        if self.r1.shape != self.r2.shape:
            raise ValueError("r1 and r2 must have the same shape")
        if self.cell_volume is None:
            self.cell_volume = np.ones_like(self.r1)
        self.cell_volume = np.asarray(self.cell_volume, dtype=np.float64)

    # ---- derived quantities of Section 2.1 -------------------------------- #
    @property
    def r1_star(self) -> float:
        return float(self.r1.max())

    @property
    def r2_star(self) -> float:
        return float(self.r2.max())

    @property
    def S1(self) -> np.ndarray:
        """S_{1,eps} = { x : r1(x) >= r1* - eps }."""
        return self.r1 >= self.r1_star - self.eps

    @property
    def S2(self) -> np.ndarray:
        return self.r2 >= self.r2_star - self.eps

    @property
    def S_union(self) -> np.ndarray:
        return self.S1 | self.S2

    @property
    def outside(self) -> np.ndarray:
        """O_eps = X \\ S_eps."""
        return ~self.S_union

    # ---- separation constants -------------------------------------------- #
    def delta_outside(self) -> float:
        """Largest delta with r_i(x) <= r_i* - delta for every x outside S_eps.

        Assumption 2.1(iii), outside clause.  Returns +inf if O_eps is empty.
        """
        o = self.outside
        if not o.any():
            return float("inf")
        return float(
            min((self.r1_star - self.r1[o]).min(), (self.r2_star - self.r2[o]).min())
        )

    def cross_gaps(self) -> tuple[float, float]:
        """(Delta_1, Delta_2) in the *main-text* form of Assumption 2.1(iii):

            x in S_{2,eps} => r1(x) <= r1* - Delta_1
            x in S_{1,eps} => r2(x) <= r2* - Delta_2
        """
        d1 = (
            float((self.r1_star - self.r1[self.S2]).min())
            if self.S2.any()
            else float("inf")
        )
        d2 = (
            float((self.r2_star - self.r2[self.S1]).min())
            if self.S1.any()
            else float("inf")
        )
        return d1, d2

    def cross_gaps_appendix(self) -> tuple[float, float]:
        """(Delta_1, Delta_2) in the *appendix* form (Assumption B.5), which reads

            x in S_{2,eps} => r1(x) <= r1* - Delta_1 + eps

        i.e. the largest admissible gap is eps larger than the main-text one.
        """
        d1, d2 = self.cross_gaps()
        return d1 + self.eps, d2 + self.eps

    def leakage(self, appendix: bool = False) -> tuple[float, float]:
        """kappa_i = exp(-(Delta_i - 2 eps)) (definition above Lemma 3.4)."""
        d1, d2 = self.cross_gaps_appendix() if appendix else self.cross_gaps()
        return (
            float(np.exp(-(d1 - 2 * self.eps))),
            float(np.exp(-(d2 - 2 * self.eps))),
        )

    # ---- assumption auditor ---------------------------------------------- #
    def audit(self, p0: np.ndarray, q: float, appendix: bool = False) -> dict:
        """Machine-check every clause of Assumption 2.1 / B.1-B.5.

        Returns a dict of clause -> {"ok": bool, ...evidence}.  A stage must
        refuse to draw any conclusion from a landscape whose audit fails.
        """
        d1, d2 = self.cross_gaps_appendix() if appendix else self.cross_gaps()
        k1, k2 = self.leakage(appendix=appendix)
        delta = self.delta_outside()
        a0 = float(p0[self.S1].sum())
        b0 = float(p0[self.S2].sum())
        finite = bool(np.isfinite(self.r1).all() and np.isfinite(self.r2).all())
        out: dict = {
            # Assumption 2.1(i) / B.1 -- bounded, measurable rewards
            "A1_i_bounded_rewards": {
                "ok": finite,
                "r1_range": [float(self.r1.min()), float(self.r1.max())],
                "r2_range": [float(self.r2.min()), float(self.r2.max())],
            },
            # Assumption 2.1(ii) / B.2+B.3 -- disjoint basins, both populated
            "A1_ii_basins_disjoint": {
                "ok": bool(not (self.S1 & self.S2).any()),
                "n_S1": int(self.S1.sum()),
                "n_S2": int(self.S2.sum()),
                "n_overlap": int((self.S1 & self.S2).sum()),
            },
            "A1_ii_nontrivial_init": {
                "ok": bool(a0 > 0.0 and b0 > 0.0),
                "p0_S1": a0,
                "p0_S2": b0,
            },
            # Assumption 2.1(iii) -- outside gap and cross-reward leakage gaps
            "A1_iii_outside_gap": {
                "ok": bool(delta > 0.0),
                "delta": delta,
            },
            "A1_iii_cross_gaps": {
                "ok": bool(d1 > 0.0 and d2 > 0.0),
                "Delta1": d1,
                "Delta2": d2,
            },
            # extra conditions specific quantitative results need
            "leakage_factors": {
                "ok": bool(k1 < 1.0 and k2 < 1.0),
                "kappa1": k1,
                "kappa2": k2,
                "requires_Delta_gt_2eps": bool(
                    d1 > 2 * self.eps and d2 > 2 * self.eps
                ),
            },
            "lemma35_q_window": {
                "ok": bool(k1 < q < 1.0 - k2),
                "q": float(q),
                "window": [k1, 1.0 - k2],
            },
        }
        out["all_ok"] = all(v["ok"] for k, v in out.items() if isinstance(v, dict))
        return out

    def outside_domination_rho(self, p: np.ndarray, q: float) -> float:
        """sup_{x not in S_eps} W_t(x) -- the quantity Assumption B.4 bounds by rho<1.

        This is the hypothesis the *appendix* proof of Lemma B.4 assumes and the
        main text omits, so every stage measures it explicitly.
        """
        W = self.multiplier(p, q)
        o = self.outside
        return float(W[o].max()) if o.any() else 0.0

    # ---- dynamics --------------------------------------------------------- #
    def multiplier(self, p: np.ndarray, q: float) -> np.ndarray:
        """W_t(x) = q e^{r1}/Z1 + (1-q) e^{r2}/Z2  (Eq. 5).

        Shifting r_i by a constant leaves W unchanged because Z_i absorbs it, so
        the max-subtraction below is an exact algebraic identity, not an
        approximation -- it only buys floating-point range.
        """
        e1 = np.exp(self.r1 - self.r1_star)
        e2 = np.exp(self.r2 - self.r2_star)
        z1 = float(p @ e1)
        z2 = float(p @ e2)
        return q * e1 / z1 + (1.0 - q) * e2 / z2

    def step(self, p: np.ndarray, q: float) -> np.ndarray:
        """One exact large-K pluralistic retraining step."""
        return p * self.multiplier(p, q)

    def masses(self, p: np.ndarray) -> tuple[float, float, float]:
        """(a_t, b_t, m_t) = (p(S_1), p(S_2), p(X \\ S_eps))."""
        return (
            float(p[self.S1].sum()),
            float(p[self.S2].sum()),
            float(p[self.outside].sum()),
        )

    def reward_stats(self, p: np.ndarray) -> dict:
        """E_p[r_i] and Var_p[r_i] for i = 1, 2."""
        out = {}
        for i, r in ((1, self.r1), (2, self.r2)):
            mean = float(p @ r)
            out[f"mean_r{i}"] = mean
            out[f"var_r{i}"] = float(p @ (r - mean) ** 2)
        return out

    def run(
        self, p0: np.ndarray, q: float, steps: int, record_every: int = 1
    ) -> dict:
        """Iterate the update, recording the theory's tracked quantities."""
        p = np.array(p0, dtype=np.float64)
        rows: list[dict] = []
        for t in range(steps + 1):
            if t % record_every == 0 or t == steps:
                a, b, m = self.masses(p)
                row = {
                    "t": t,
                    "a_t": a,
                    "b_t": b,
                    "m_t": m,
                    "rho_outside_sup_W": self.outside_domination_rho(p, q),
                }
                row.update(self.reward_stats(p))
                rows.append(row)
            if t < steps:
                p = self.step(p, q)
        return {"trace": rows, "p_final": p}


# --------------------------------------------------------------------------- #
# finite-K Bradley-Terry weight (Definition 2.1) -- used for the finite-K checks
# --------------------------------------------------------------------------- #
def bt_weight_finite_K(
    p: np.ndarray, r: np.ndarray, K: int, n_mc: int, rng: np.random.Generator
) -> np.ndarray:
    """Monte-Carlo estimate of H^{K,r}_p(x) = E[ K e^{r(x)} / (e^{r(x)} + sum e^{r(y_j)}) ].

    y_1..y_{K-1} are i.i.d. from p.  Returned for every atom x simultaneously.
    """
    e = np.exp(r - r.max())
    idx = rng.choice(len(p), size=(n_mc, K - 1), p=p)
    partial = e[idx].sum(axis=1)  # (n_mc,)
    # E over MC draws, for each atom x
    return float(K) * (e[:, None] / (e[:, None] + partial[None, :])).mean(axis=1)


# --------------------------------------------------------------------------- #
# landscape builders
# --------------------------------------------------------------------------- #
def plateau_landscape(
    delta1: float, delta2: float, delta_outside: float, eps: float = 0.1
) -> tuple[Landscape, np.ndarray]:
    """The 3-macro-state plateau reduction used by the CURRENTLY JUDGED logbook.

    basin1: r1=0, r2=-Delta2 ; basin2: r1=-Delta1, r2=0 ; outside: r1=r2=-delta_c.
    Kept only as the documented reference control (Proposition 3.4 idealisation);
    it is a 3-state system, not a continuous landscape.
    """
    r1 = np.array([0.0, -delta1, -delta_outside])
    r2 = np.array([-delta2, 0.0, -delta_outside])
    lam = Landscape(r1, r2, eps, name="plateau-3state", meta={"kind": "plateau3"})
    return lam, np.array([0.3, 0.3, 0.4])


def quadratic_grid_landscape(
    d: float,
    eps: float,
    n: int = 4001,
    half_width: float = 12.0,
    box: float | None = None,
) -> Landscape:
    """The paper's own synthetic reward geometry (Appendix C.4), on a fine 1-D grid.

        r_i(x) = -||x - mu_i||^2,  mu_1 = -d/2, mu_2 = +d/2

    X is the bounded interval [-box, box] so Assumption 2.1(i) (bounded rewards)
    holds.  These basins are genuine continuous regions -- balls of radius
    sqrt(eps) about each optimum -- and the density evolves *within* them, which
    is exactly what a 3-macro-state reduction cannot represent.
    """
    box = half_width if box is None else box
    edges = np.linspace(-box, box, n + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    vol = np.diff(edges)
    mu1, mu2 = -0.5 * d, 0.5 * d
    r1 = -((x - mu1) ** 2)
    r2 = -((x - mu2) ** 2)
    return Landscape(
        r1,
        r2,
        eps,
        name=f"quadratic-1d-d{d:g}-n{n}",
        coords=x,
        cell_volume=vol,
        meta={"kind": "quadratic1d", "d": d, "n": n, "box": box,
              "mu1": mu1, "mu2": mu2},
    )


def gaussian_init(lam: Landscape, mean: float, sd: float) -> np.ndarray:
    """Discretised N(mean, sd^2) initial model, restricted and renormalised to X."""
    x = lam.coords
    if x is None:
        raise ValueError("gaussian_init needs a landscape with coords")
    dens = np.exp(-0.5 * ((x - mean) / sd) ** 2)
    p = dens * lam.cell_volume
    return p / p.sum()


def uniform_init(lam: Landscape) -> np.ndarray:
    p = np.array(lam.cell_volume, dtype=np.float64)
    return p / p.sum()
