# Curated Synthetic Data Doesn't Have to Collapse — Reproduction Report

**Paper:** arXiv:2605.07724 · OpenReview `kwvtSA9ed3`
**Reproduction:** 6 claims, 6 derived verdicts, all gates passing on one run
**Citable node:** `final/all-six-claims` — run `3bf9f951-47af-4f1e-921d-b27606ed8522`
**Git SHA:** see `raw_evidence/artifacts/final/.openresearch/artifacts/verdicts.json`
**Compute:** Hugging Face `cpu-upgrade`, 8 CPUs (cgroup `cpu.max` = `800000 100000`), AMD EPYC 7R13. No GPU was used at any point.

---

## Headline result

| # | Claim | Verdict | Decided by |
|---|---|---|---|
| 1 | Lemma 3.3 — outside mass decays geometrically | **FALSIFIED** | exact-rational counterexample + symbolic boundary |
| 2 | Lemma 3.5 — non-collapse under pluralistic curation | **VERIFIED** | symbolic certificate + 432 continuous landscapes |
| 3 | Theorem 3.6 — variance lower bound | **VERIFIED** | symbolic proof with free within-basin variance + 2 986 random mixtures |
| 4 | Theorem 3.7 / Cor 3.8 — Nash bargaining limit | **VERIFIED** | continuum certificate + independent 50-digit bisection |
| 5 | E3 — CIFAR-10 flow retraining sustains diversity | **VERIFIED** | 3-arm comparison, margin **+1.1427** |
| 6 | E4 — text retraining sustains length entropy | **VERIFIED** | 2-arm comparison, margin **+2.4241** |

The single substantive disagreement with the paper is Claim 1, and it is a **precision defect in
the main text, not an error in the result**: the appendix proof is correct and this reproduction
verifies it symbolically. Details in §1.

---

## 1 · Claim 1 is falsified as stated in the main text — and verified as stated in the appendix

This is the reproduction's most consequential finding, and it turns on a discrepancy between two
statements of the same lemma.

- **Appendix Lemma B.4** assumes **B.4 (outside domination)**: `sup_{x∉S_ε} W_t(x) ≤ ρ_ε < 1`.
  Given that, `m_t ≤ ρ_ε^t m_0 → 0`. This reproduction **proves that implication symbolically**
  (termwise non-negativity → induction → `ρ^t → 0`), and a negative control confirms the
  conclusion does *not* follow at `ρ = 1`.
- **Main-text Assumption 2.1** never states B.4. Under Assumption 2.1 alone the conclusion is false.

Two independent routes establish the failure and agree exactly.

**Route A (symbolic).** The outside multiplier exceeds 1 precisely when

```
W_t(C) > 1   ⟺   2e^(−δ) > 1 + e^(−Δ)
```

proved as a sign-equivalence rather than by solving a transcendental inequality. As `Δ → ∞` this
becomes `δ < log 2 ≈ 0.693`. Assumption 2.1 permits *any* `δ > 0`, so the counterexample family is
non-empty for every `ε < log 2`.

**Route B (exact rational arithmetic).** A 4-atom landscape with `w₁ = {A:1, B:1/100, C:4/5, D:1/1000}`,
`w₂` mirrored, `ε` chosen so every basin test is a rational comparison. Nothing is rounded:

| quantity | value |
|---|---|
| `m₁/m₀` | **80000/79941** (exact, > 1) |
| `m_∞` | 1.0000000000 |
| `a_∞` | 1.632 × 10⁻¹⁵ |
| every audited clause of Assumption 2.1 | satisfied |
| control (deep outside atom), max `m_{t+1}/m_t` | 0.95374 → `m_∞` = 9.22 × 10⁻⁸¹ |

![Claim 1 falsification](images/fig1_claim1_falsification.png)

*Left: the outside mass rises to 1 in the counterexample while the control decays geometrically as
the lemma predicts — both satisfy Assumption 2.1. Right: the contraction factor, which must be < 1
for the lemma, sits above 1 at every step.*

The conclusion does not merely lose its rate; it **inverts**. On the same family the basin
multiplier tends to `e^δ/2 < 1`, so both basins lose mass while the outside gains it.

![Claim 1 boundary](images/fig2_claim1_boundary.png)

*The closed-form boundary from Route A separates the Route B outcomes with no exceptions — 24/24
sweep points agree. Two methods sharing no code agreeing on a boundary is what makes this a
falsification rather than a numerical artifact.*

**The gap is not hypothetical for this paper.** In the paper's own quadratic reward geometry
(Appendix C.4), B.4 is violated at every setting tested:

| d | ε | max `sup_{x∉S_ε} W_t(x)` | B.4 holds? | does `m_t` still decay? |
|---|---|---|---|---|
| 2 | 0.1 | 6.51 | no | yes |
| 4 | 0.1 | 6.11 | no | yes |
| 8 | 0.5 | 4.10 | no | yes |

So B.4 is **sufficient but not necessary**, and the paper's experiments live outside it while still
behaving well. The honest reading: the theorem is true under its appendix hypothesis, the main-text
statement omits a hypothesis it needs, and the omission is reachable by admissible parameters.

---

## 2 · Claim 2 — non-collapse (VERIFIED)

The paper's interval is `[L, U]` with `L = (q − κ₁)/(1 − κ₁)`. Under **appendix B.5** the fixed
point equals that expression **exactly** (sympy: difference identically 0). Under main-text
Assumption 2.1(iii) the bound is *tighter*, so the paper's interval is safe under either reading.

Non-collapse is proved universally, not sampled: with `κᵢ = 1/(1+uᵢ)` and slack variables
`q = κ₁+s₁ = 1−κ₂−s₂`, `L = s₁ + s₁/u₁ > 0` and `1 − U = s₂ + s₂/u₂ > 0` — positive by
construction for every admissible parameter.

Numerically, on **continuous** landscapes (not the 3-state reduction the original evidence used):

- **32/32** configurations land inside the interval; max excursion **1.044 × 10⁻¹⁴**
- minimum non-collapse margin **η = 0.081606** — bounded away from 0
- grid refinement 501 → 8001 points: spread **5.56 × 10⁻¹⁴**, so this is not a discretisation artifact
- **400** random non-plateau landscapes searched adversarially: no counterexample, max excursion 3.97 × 10⁻⁸

![Claim 2](images/fig3_claim2_interval.png)

Both endpoint controls fire correctly: at `q = κ₁` the lower endpoint collapses to 0 and at
`q = 1−κ₂` the upper reaches 1 — i.e. the guarantee genuinely depends on `κ₁ < q < 1−κ₂`.

> **Paper defect noted:** in Table 3 the `D=2` and `D=4` rows are identical for all six values of
> `q`. That is very unlikely to be correct and looks like a copy error in the table.

---

## 3 · Claim 3 — variance lower bound (VERIFIED)

`Var_p[rᵢ] ≥ a(1−a)(Δᵢ − 2ε)₊²`, proved by the law of total variance with the **within-basin
variances left as free symbols** — so the proof covers arbitrary intra-basin structure, which is
exactly what the judged 3-state reduction could not do (it has zero intra-basin degrees of freedom).

- **2 986/2 986** random general two-basin mixtures satisfy the bound, for **both** rewards
- minimum slack **0.1535**; minimum within-basin variance **0.0400** — strictly positive throughout,
  so no sampled point is secretly a plateau
- the bound also holds on the limits the retraining dynamics actually reach

![Claim 3](images/fig4_claim3_variance.png)

Both vacuity controls are recorded rather than hidden: the bound is vacuous at `a ∈ {0,1}` and when
`Δᵢ ≤ 2ε`.

---

## 4 · Claim 4 — Nash bargaining (VERIFIED)

`d/dα log f = (α−q)/(α(α−1))` has the unique root `α = q` for **symbolic** `q`; strict log-concavity
plus `f(0) = f(1) = 0 < f(q)` makes it the unique maximiser on the closed interval.

The judge's specific criticism of the original evidence was that it asserted the *dynamics*
coincidence rather than testing it. Tested here, against a **50-digit mpmath bisection** on the Nash
derivative that shares no code with the dynamics:

- `max |argmax − q| = 0.0` (exactly)
- `max |a_∞ − argmax| = 1.183 × 10⁻¹¹` for `d ≥ 5`

![Claim 4](images/fig5_claim4_nash.png)

> **Paper defect noted:** our Table 5 analogue keeps falling (1.0 × 10⁻¹, 1.35 × 10⁻⁴, 4.27 × 10⁻⁹,
> 5.83 × 10⁻²³) whereas the paper's values floor at ≈1.25 × 10⁻⁵ — consistent with the paper hitting
> float64 precision. Separately, Table 4's reported slopes (−0.565 … −0.585) contradict the text's
> claim of ≈−13.6 at `d=8` and ≈−1.3 at `d=2`.

---

## 5 · Claims 5 and 6 — the empirical experiments (VERIFIED, with caveats)

### How these verdicts are decided

Both claim sentences are **comparative**. An early version of this reproduction ran each
configuration separately and recorded `VERIFIED` from checks that only confirmed *the experiment
ran* and *metrics were recorded* — a gate that passes vacuously. That was found and repaired: a
single stage now ingests the measured series from every arm and **derives** the verdict, subject to

1. a margin above a noise floor (0.05),
2. a **relative** collapse control — the single-reward arm must end below 60% of its own first
   round, so the baseline cannot be vacuous, and
3. a **self-test** that re-runs the identical decision procedure on arm-swapped input and confirms
   it returns FALSIFIED. A test that cannot fail is not a test.

### Claim 5 — CIFAR-10, real OT-CFM with exact minibatch OT

1.00 M-parameter UNet velocity field, real CIFAR-10, reward from a CNN at **84.5%** held-out
accuracy (paper's VGG11: 92.39%), K-BT curation with `K=256` keeping **75/1500 = 5.0%** per round —
the paper's keep ratio exactly. 25 rounds.

| arm | round 1 | tail (last 5) | classes left |
|---|---|---|---|
| single reward (M=1) | 1.7581 | **0.0000** | 1 |
| pluralistic M=2 | 1.7581 | **0.0000** | 1 |
| pluralistic M=5 | 1.7581 | **1.1427** | 5 |

Margin (best pluralistic vs single): **+1.1427** ≫ 0.05 → VERIFIED.

![Claim 5](images/fig6_claim5_cifar_entropy.png)

**Caveat that matters:** M=2 collapsed just as completely as M=1. In this downscaled setting
pluralism *per se* is not sufficient — the protective effect appears at M=5. The paper's claim is
supported in the M=5 vs M=1 comparison and **not** at M=2.

### Claim 6 — WikiText-2, real 30.0 M-parameter GPT-2

6 layers, 6 heads, `d=384`, vocab 50 257, nucleus sampling, EOS-terminated so lengths are
model-determined rather than a decoding constant. 20 rounds.

| arm | round 1 | tail (last 5) | final mean length | distinct lengths |
|---|---|---|---|---|
| one preference (`T_A=10`) | 2.8944 | **0.6760** | 9.2 | 4 |
| two preferences (`T_A=10, T_B=15`) | 3.1820 | **3.1001** | 27.9 | 58 |

Margin **+2.4241** → VERIFIED. The single-preference arm collapses to 23.4% of its own first round
with all mass at the target (`nearA → 1.00`); the two-preference arm keeps **both** basins populated
in every round (`nearA` 0.28–0.59, `nearB` 0.16–0.32) — it hedges rather than picking one
compromise length.

![Claim 6](images/fig7_claim6_text_entropy.png)

### A bug that invalidated an earlier round of E4 — and how it was caught

`encode()` padded to `MAX_LEN=96` with EOS, and the WikiText prose lines used here are 5–60 words,
so **only ~16% of target positions were real tokens**. Training with `labels=input_ids` scored the
model mostly on predicting padding, and the cheapest way to win that game is to emit EOS
immediately. Symptoms: pretrain loss **0.3** (implausible for prose), generations of mean length
**0.0** words, **1** distinct length from round 1.

The verdict gate refused to record a verdict for that run — it failed on the very checks that
mattered, which is what a gate is for. The fix masks padding with `-100`; loss on real tokens then
starts at **10.84 ≈ ln(50257)**, the correct value for an untrained model. All pre-fix E4 runs are
marked void in the experiment tree and are cited nowhere.

---

## 6 · Limitations, stated plainly

- **Downscaling.** E3 uses 1 500 generated / 75 kept per round against the paper's 50 000 / 2 500;
  the *selection pressure* (5% keep ratio) and loop structure are preserved, the absolute counts and
  model sizes are not. The flow net is 1.00 M parameters against the paper's OT-CFM on 4×H200.
- **One seed per arm.** The empirical margins are point estimates, not confidence intervals. This is
  precisely why a margin threshold is required rather than any positive difference.
- **No FID.** It needs an InceptionV3 pass outside the CPU-only budget, and it is not part of the
  claim sentences under test (entropy and diversity are).
- **Claim 5's M=2 result runs against the trend** and is reported rather than dropped.
- **Baseline node is BLOCKED, deliberately.** The judged 3-state reduction is reproduced for
  reference, and its control records that it has zero intra-basin degrees of freedom — which is
  exactly why the original four theory claims were judged toy. It is not counted as evidence.

## 7 · What is auditable, and where

- `raw_evidence/artifacts/final/.openresearch/artifacts/` — all 22 artifacts from the citable run,
  each recovered from the run log and **SHA-256 verified**
- `raw_evidence/rounds_*.csv` — per-round tables for all five empirical arms
- `raw_evidence/log_*.txt` — complete run logs
- `raw_evidence/artifacts/final/.openresearch/artifacts/verdicts.json` — every verdict, check,
  control, limitation, CPU allocation and git SHA

orx local mode has no artifact store, so the run log is the only channel out of a job. Artifacts are
therefore emitted into stdout base64-framed with a declared byte count and SHA-256, and rebuilt by
`tools/extract_evidence.py`; a truncated or interleaved log fails the hash check instead of yielding
plausible-looking wrong numbers.

## 8 · Reproducing this

One fixed command on every node, no exceptions:

```bash
bash run.sh
```

It installs `uv` if absent, runs `uv sync --frozen`, prints resolved versions, **refuses to run if a
GPU is visible**, then executes `uv run python -m repro.main`. Hyperparameters live in committed
`repro/config/active.json` per node — never in the command, never in environment variables — so
every node runs identical code paths over different committed configuration.
