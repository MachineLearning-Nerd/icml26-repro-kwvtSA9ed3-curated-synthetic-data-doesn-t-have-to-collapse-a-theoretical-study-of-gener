# Source audit — arXiv:2605.07724

Every claim contract in `repro/contracts/` is anchored to the statements quoted here.
All three sources were retrieved with an explicit browser User-Agent:

```
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36
```

| Source | URL | Retrieved (UTC) | Bytes | SHA-256 |
|---|---|---|---|---|
| Abstract page | `https://arxiv.org/abs/2605.07724` | 2026-07-26 | 42316 | `2a3e87e54e3540c8669119714f532cd40ccce873f42624c383c4f87df58be93c` |
| arXiv LaTeXML HTML (v1, **main text only — appendix absent**) | `https://arxiv.org/html/2605.07724v1` | 2026-07-26 | 241869 | `ebd4297558bbe3cb784906588eb9b1b94cb69669abfb31ebda57a2d25cccfe7d` |
| **ar5iv HTML (main text + full Appendices A–C)** | `https://ar5iv.labs.arxiv.org/html/2605.07724` | 2026-07-26 | 847304 | `13e7591d971a2ff88da73262a77c14b6de1b885403ab4ea850f57da02bec2af4` |

Paper: Ali Falahati, Mohammad Mohammadi Amiri, Kate Larson, Lukasz Golab,
*Curated Synthetic Data Doesn't Have to Collapse: A Theoretical Study of Generative
Retraining with Pluralistic Preferences*, arXiv:2605.07724v1 [cs.LG], 08 May 2026.
OpenReview `kwvtSA9ed3`. License CC BY 4.0.

**The ar5iv rendering is the authoritative source for this reproduction** because the
arXiv HTML omits the appendix, and the appendix contains the proofs (B.3), the full
assumption list (B.2) and every experiment hyperparameter (C.2–C.9). The judged
reproduction appears to have worked from the main text alone; several of the
findings below turn on main-text/appendix differences.

---

## The model (Section 2.1)

Domain `X ⊆ R^d`; model distribution `p_t` with density. Each retraining iteration:

1. **Sampling** — draw `K ≥ 2` i.i.d. candidates `x_1..x_K ~ p_t`.
2. **Multi-preference selection** — independently of the candidates, pick reward
   `r_1` w.p. `q ∈ (0,1)`, else `r_2`; then select one candidate by Bradley–Terry,
   `P(x̂ = x_k | x_{1:K}, r) = e^{r(x_k)} / Σ_j e^{r(x_j)}`  (Eq. 1).
3. **Retraining** — idealised exact MLE over *all* densities, so `p_{t+1} = p̃_t`.

Large-`K` update actually analysed (Eq. 5, from Lemmas 3.1 + 3.2):

```
p_{t+1}(x) = p_t(x) · ( q·e^{r_1(x)}/Z_1(t) + (1-q)·e^{r_2(x)}/Z_2(t) ),
Z_i(t) := E_{p_t}[ e^{r_i(x)} ]
```

Write `W_t(x)` for that multiplier. Tracked masses, with `S_{i,ε} := {x : r_i(x) ≥ r_i* - ε}`,
`S_ε := S_{1,ε} ∪ S_{2,ε}`, `r_i* := sup_x r_i(x)`:

```
a_t := p_t(S_{1,ε}),  b_t := p_t(S_{2,ε}),  m_t := p_t(X \ S_ε),   a_t + b_t + m_t = 1
```

## Assumption 2.1 (main text) — "Two-basin landscape with leakage gaps"

Fix `ε > 0`.

- **(i) Bounded rewards.** `r_i(x) ∈ [L_i, U_i]` for all `x ∈ X`, `i ∈ {1,2}`.
- **(ii) Disjoint near-optimal basins with nontrivial initialization.**
  `S_{1,ε} ∩ S_{2,ε} = ∅`, and `p_0(S_{1,ε}) > 0`, `p_0(S_{2,ε}) > 0`.
- **(iii) Cross-reward leakage gaps.** There exist constants `δ_{i,ε} > 0` and `Δ_i > 0` s.t.
  - for all `x ∉ S_ε`: `r_1(x) ≤ r_1* - δ_{c,ε}` and `r_2(x) ≤ r_2* - δ_{c,ε}`;
  - `x ∈ S_{1,ε} ⇒ r_2(x) ≤ r_2* - Δ_2`;  `x ∈ S_{2,ε} ⇒ r_1(x) ≤ r_1* - Δ_1`.

> *Verbatim typo in the source:* clause (iii) declares `δ_{i,ε}` but then uses
> `δ_{c,ε}`. The auditor in `repro/lib/landscape.py` satisfies the strongest reading
> — a single `δ > 0` working uniformly for both rewards and all outside points —
> so no finding here depends on resolving the subscript.

Leakage factors (defined above Lemma 3.4):
`κ_1 := exp(-(Δ_1 - 2ε))`, `κ_2 := exp(-(Δ_2 - 2ε))`.

## Assumptions B.1–B.6 (appendix) — the set the proofs actually use

- **B.1** bounded measurable rewards (= 2.1(i)).
- **B.2** `p_0(S_{i,ε}) > 0` (= half of 2.1(ii)).
- **B.3** `S_{1,ε} ∩ S_{2,ε} = ∅` (= other half of 2.1(ii)).
- **B.4 Outside domination.** *There exists* `ρ_ε ∈ (0,1)` such that **for all `t ≥ 0`**
  ```
  sup_{x ∉ S_ε} [ q·e^{r_1(x)}/Z_1(t) + (1-q)·e^{r_2(x)}/Z_2(t) ] ≤ ρ_ε
  ```
  **This is an assumption in the appendix and is NOT part of main-text Assumption 2.1.**
  It is the pivot of finding F1 below.
- **B.5 Cross-basin separation.** `x ∈ S_{1,ε} ⇒ r_2(x) ≤ r_2* - Δ_2 + ε`, and
  `x ∈ S_{2,ε} ⇒ r_1(x) ≤ r_1* - Δ_1 + ε`. Note the extra `+ ε` relative to
  main-text 2.1(iii): the appendix admits gaps `ε` *larger* than the main text does
  for the same landscape.
- **B.6 Plateau-basin idealisation** (used *only* for the explicit `q`-mixture and the
  bargaining corollary): `r_1 ≡ U_1`, `r_2 ≡ U_2 - Δ_2` on `S_{1,ε}`; `r_1 ≡ U_1 - Δ_1`,
  `r_2 ≡ U_2` on `S_{2,ε}`.

The appendix's own interpretation paragraph concedes what B.4 excludes: an outside
point violating it "is not genuinely low-reward background mass: it is an additional
compromise basin receiving support from multiple rewards."

---

## The four theoretical statements, with their exact quantifiers

### Claim 1 — Lemma 3.3 (appendix Lemma B.4), "Decay outside ε-optimal basins"

> Fix `ε>0` and work in the large-`K` regime (5). **Under Assumption 2.1 (i,iii)**, there
> exists a constant `ρ_ε ∈ (0,1)` such that `m_{t+1} ≤ ρ_ε m_t` for all `t`. Hence
> `m_t ≤ ρ_ε^t m_0 → 0` and therefore `a_t + b_t = 1 - m_t → 1`.

Quantifier structure: `∀ landscapes satisfying 2.1(i,iii) ∀ p_0 . ∃ρ_ε ∈ (0,1) . ∀t . m_{t+1} ≤ ρ_ε m_t`.
The appendix version (Lemma B.4) instead reads `∀ landscapes satisfying **B.4** . …`,
and B.4 already *is* the conclusion's hypothesis in disguised form.

**Finding F1 (falsification of the main-text implication).** Assumption 2.1(iii)'s
outside clause is nearly vacuous: for any landscape, `x ∉ S_ε` already implies
`r_1(x) < r_1* - ε`, so `δ = ε` always works. It therefore places **no lower bound on
how far below optimal outside points are**, while `Z_i(t)` can be small, so `W_t(x)`
is not forced below 1 outside `S_ε`. `repro/stages/claim1_outside_decay.py` exhibits an
explicit 4-atom landscape satisfying every clause of 2.1 on which `m_t` *increases*
monotonically to 1 and `a_t → 0`, in **exact rational arithmetic**.

### Claim 2 — Lemma 3.5 (appendix Lemma B.6), "Uniform non-collapse under bounded leakage"

> Fix `ε>0`, large-`K` regime. Assume Assumption 2.1 **and the decay conclusion of
> Lemma 3.3**. Suppose `q ∈ (κ_1, 1-κ_2)`. Then there exist `c_ε>0`, `t_0` with
> `max{0,(q-κ_1)/(1-κ_1)} - c_ε m_t ≤ a_t ≤ min{1, q/(1-κ_2)} + c_ε m_t` for all `t ≥ t_0`.
> **In particular, there exists `η>0` such that `η ≤ a_t ≤ 1-η` for all `t ≥ t_0`.**

The claim under test is the emphasised non-collapse conclusion.

**Finding F2 (interval endpoint needs a correction factor).** The appendix proof of
Lemma B.5/B.6 bounds `p_t^{(1)}(S_1) ≥ a_t / (a_t + κ_1 b_t + C_ε m_t)`. Redoing the
step keeping every `ε` gives `p_t^{(1)}(S_1) ≥ a_t e^{-ε} / (a_t + e^{-Δ_1} b_t + m_t)`,
whose fixed point is `(q e^{-ε} - e^{-Δ_1}) / (1 - e^{-Δ_1})`, strictly *below* the paper's
`(q-κ_1)/(1-κ_1)` whenever `ε>0`. The two agree at `ε = 0`. The non-collapse
conclusion survives, because `q > κ_1 = e^{2ε-Δ_1}` implies `q e^{-ε} > e^{-Δ_1}`.

### Claim 3 — Theorem 3.6 (appendix Theorem B.6), "Variance preservation"

> Assume the setting of Theorem 3.3 and `a_∞ ∈ (0,1)`. Then `Var_{p_t}[r_i] → Var_{p_∞}[r_i]`,
> and `Var_{p_∞}[r_i]` contains an explicit inter-basin term `a_∞(1-a_∞)(μ_{i,1}-μ_{i,2})^2`,
> `μ_{i,j} := E_{p_{∞,j}}[r_i]`. Under basin separation this yields
> `Var_{p_∞}[r_i] ≳ a_∞(1-a_∞)·max{Δ_i - 2ε, 0}^2`.

Appendix form is an exact inequality (`≥`, not `≳`) under B.5's separation.

### Claim 4 — Theorem 3.7 (appendix Theorem B.7), "Weighted Nash bargaining"

> Let `p_α := α P_1 + (1-α) P_2`, `u_i(p) := E_p[r_i]`, `d_1 := u_1(P_2)`, `d_2 := u_2(P_1)`.
> Assume `u_1(P_1) > u_1(P_2)` and `u_2(P_2) > u_2(P_1)`. Then the weighted Nash product
> `(u_1(p_α)-d_1)^q (u_2(p_α)-d_2)^{1-q}` is **uniquely maximized over `α ∈ [0,1]` at `α* = q`**.

Corollary 3.8 / B.8 supplies the second half the judge asked for — that the *limiting
distribution of the curation dynamics* coincides with that Nash point — but only in the
plateau (B.6) hard-separation regime, where Proposition 3.4/B.3 identifies `p_∞` with `p_q`.

---

## The two empirical experiments

### Claim 5 — E3, CIFAR-10 flow retraining (Section 4, Table 1, Fig. 4; Appendix C.5)

- Normalizing flow trained with **OT-CFM** (optimal-transport conditional flow matching);
  initial model pretrained on all 50,000 CIFAR-10 training images.
- Per round: generate `5·10^4` samples, retain `2.5·10^3` (5%) by discrete `K`-BT selection.
  `T = 25` rounds.
- Reward `r(x) = γ·π_i(x)` from class probabilities of a pretrained VGG11 (92.39% test acc.).
- Regimes: balanced multi-preference, `M ∈ {1..5}` target classes, reward uniform per draw;
  polarized two-preference with weight `q` vs `1-q`.
- Metrics: FID; class entropy; KL to uniform; feature variance; intra-class variance.
- Paper's numbers (Table 1): entropy 1.098 / 1.357 / 1.670 / 1.687 and FID 77.6 / 64.2 /
  44.5 / 35.9 for `M = 2,3,4,5`; single preference (`q=0`) entropy 0.032, FID 100.0.
- Hardware used by the authors: 4× NVIDIA H200 (Appendix C.2).

### Claim 6 — E4, GPT-2 length preferences (Section 4, Fig. 5; Appendix C.6)

- WikiText-2 seed pool, 1,000 initial sequences.
- Model: GPT-2-style decoder, **6 layers, 6 heads, embedding dim 384, vocab 50,257**.
- Reward `R(y;T) = -|L(y) - T|`, `L(y)` = word count; two targets `T_A`, `T_B`;
  conflict distance `d = |T_A - T_B|`.
- `N = 20` rounds. Each round: score pool; curate **200** samples with balanced mixture
  `q = 0.5` and BT sampling `∝ exp(R/τ)`, `τ = 0.5`; fine-tune AdamW `lr = 5e-5`,
  **2 epochs**, batch size 8; generate 200 new samples by nucleus sampling at
  temperature 0.8; filter with the same rule and add survivors to the pool.
- Metric: discrete entropy `H(L)` of the generated length distribution per round.
- Claimed result: larger `d` ⇒ higher sustained entropy; no collapse to one compromise length.

### Common selection rule (Appendix C.3)

`P{x = x_i | r, {x_j}} = exp(r(x_i)/τ) / Σ_j exp(r(x_j)/τ)` with temperature `τ > 0`,
repeated `n_curated` times with replacement.
