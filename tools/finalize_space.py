"""Bring the published logbook into the challenge's canonical shape.

Three things this fixes, all of them regressions introduced when the Space was first
updated:

  1. README.md frontmatter had lost its tags. The judge DISCOVERS logbooks by the Space
     tags `icml2026-repro` and `paper-<openreview-id>`; dropping them makes the logbook
     invisible to judging entirely. This is the single most important fix here.
  2. pages/index.md had been overwritten with prose. The index must be title + a Pages
     table only, and the table is what drives the sidebar.
  3. Page slugs did not follow the `executive-summary` / `claim-N-*` / `conclusion`
     convention the challenge guide specifies, so claims were harder to attribute.

Historical pages are still never modified or removed: they keep their original slugs and
files, and simply move to the end of the navigation under their existing labels.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SPACE = "DineshAI/kwvtSA9ed3"
ORID = "kwvtSA9ed3"
PAPER_TITLE = ("Curated Synthetic Data Doesn't Have to Collapse: A Theoretical Study of "
               "Generative Retraining with Pluralistic Preferences")
GH = "https://github.com/MachineLearning-Nerd/icml26-repro-kwvtSA9ed3-curated-synthetic-data-doesn-t-have-to-collapse-a-theoretical-study-of-gener"
RUN = "3bf9f951-47af-4f1e-921d-b27606ed8522"

# Tags the judge's discovery cron matches on. Do not drop any of these.
TAGS = ["trackio", "trackio-logbook", "open-experiment", "icml2026-repro", f"paper-{ORID}"]

# old slug -> (new slug, nav title)
RENAME = {
    "claim1-lemma-3-3": ("claim-1-lemma-3-3-outside-mass-decay",
                         "Claim 1: outside mass decays geometrically (Lemma 3.3)"),
    "claim2-lemma-3-5": ("claim-2-lemma-3-5-non-collapse",
                         "Claim 2: non-collapse under pluralistic curation (Lemma 3.5)"),
    "claim3-theorem-3-6": ("claim-3-theorem-3-6-variance-lower-bound",
                           "Claim 3: reward-variance lower bound (Theorem 3.6)"),
    "claim4-theorem-3-7": ("claim-4-theorem-3-7-nash-bargaining",
                           "Claim 4: limit is the Nash bargaining solution (Theorem 3.7)"),
    "claim5-e3-cifar": ("claim-5-e3-cifar10-flow-retraining",
                        "Claim 5: CIFAR-10 flow retraining sustains diversity (E3)"),
    "claim6-e4-text": ("claim-6-e4-text-length-entropy",
                       "Claim 6: text retraining sustains length entropy (E4)"),
}
HISTORICAL = [("overview", "Overview"), ("claims", "Claims"), ("evidence", "Evidence"),
              ("methods", "Methods & negative controls"), ("conclusion", "Conclusion")]

VERDICT = {
    "claim-1-lemma-3-3-outside-mass-decay": ("FALSIFIED", "as stated in the main text (VERIFIED as stated in the appendix)"),
    "claim-2-lemma-3-5-non-collapse": ("VERIFIED", ""),
    "claim-3-theorem-3-6-variance-lower-bound": ("VERIFIED", ""),
    "claim-4-theorem-3-7-nash-bargaining": ("VERIFIED", ""),
    "claim-5-e3-cifar10-flow-retraining": ("VERIFIED", ""),
    "claim-6-e4-text-length-entropy": ("VERIFIED", ""),
}
FIGURE = {
    "claim-1-lemma-3-3-outside-mass-decay": ["fig1_claim1_falsification.png",
                                             "fig2_claim1_boundary.png"],
    "claim-2-lemma-3-5-non-collapse": ["fig3_claim2_interval.png"],
    "claim-3-theorem-3-6-variance-lower-bound": ["fig4_claim3_variance.png"],
    "claim-4-theorem-3-7-nash-bargaining": ["fig5_claim4_nash.png"],
    "claim-5-e3-cifar10-flow-retraining": ["fig6_claim5_cifar_entropy.png"],
    "claim-6-e4-text-length-entropy": ["fig7_claim6_text_entropy.png"],
}


def cell(meta: dict, body: str) -> str:
    return "\n---\n<!-- trackio-cell\n" + json.dumps(meta) + "\n-->\n" + body + "\n"


def poster_html() -> str:
    return """````html
<!-- poster_embed.html -->
<div style="font-family:Inter,system-ui;background:#0b1220;color:#eaf2ff;padding:34px;border-radius:22px;border:1px solid #2b4a7a">
  <div style="color:#7dd3fc;font-weight:800;letter-spacing:.12em">SIX CLAIMS, CPU-ONLY, ALL DECIDED</div>
  <h1 style="font-size:32px;line-height:1.1;margin:12px 0">Curated Synthetic Data Doesn't Have to Collapse</h1>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:22px">
    <div style="background:#132038;padding:16px;border-radius:14px"><b style="font-size:26px">80000/79941</b><br>exact m&#8321;/m&#8320; &gt; 1 &mdash; Lemma 3.3 falsified</div>
    <div style="background:#132038;padding:16px;border-radius:14px"><b style="font-size:26px">2986/2986</b><br>random mixtures satisfy the variance bound</div>
    <div style="background:#132038;padding:16px;border-radius:14px"><b style="font-size:26px">1.18e-11</b><br>dynamics vs independent Nash argmax</div>
    <div style="background:#132038;padding:16px;border-radius:14px"><b style="font-size:26px">+2.42</b><br>pluralistic vs single-preference entropy margin</div>
  </div>
  <p style="font-size:17px;line-height:1.5;margin:22px 0 0">Appendix Lemma B.4 assumes outside domination and is proved here symbolically. Main-text Assumption 2.1 omits it, and an exact-rational counterexample satisfying every audited clause of 2.1 sends the outside mass to 1 instead of 0.</p>
  <div style="font-size:36px;font-weight:900;color:#7dd3fc;margin-top:18px">5 verified &middot; 1 falsified</div>
</div>
````"""


def executive_summary() -> str:
    head = "# Executive summary\n"
    summary = cell(
        {"type": "markdown", "title": "Executive summary", "pinned": True},
        f"""All six claims of *{PAPER_TITLE}* were decided with reproducible evidence on a single
CPU-only run ([`{RUN}`]({GH})): **five VERIFIED and one FALSIFIED**. The falsification is
narrow and precise rather than sweeping — appendix Lemma B.4 assumes **outside domination**
(`sup_{{x∉S_ε}} W_t(x) ≤ ρ_ε < 1`) and is correct, and this reproduction proves that
implication symbolically; main-text Assumption 2.1 never states it, and without it the
conclusion inverts. An exact-rational counterexample satisfying every audited clause of
Assumption 2.1 gives `m₁/m₀ = 80000/79941 > 1` and `m_t → 1`, while a control landscape
decays geometrically exactly as the lemma predicts.

The four theory claims are established on **general continuous landscapes with free
within-basin structure**, not the 3-state reduction used previously (which has zero
intra-basin degrees of freedom). The two empirical claims are decided by a **derived
cross-configuration comparison** carrying a self-test that proves the decision procedure
returns FALSIFIED on arm-swapped input.

| Scope | This run | Full replication |
|---|---|---|
| Hardware | HF `cpu-upgrade`, 8 CPUs (AMD EPYC 7R13) | 4× H200 |
| Wall clock | ~6 h across 10 jobs | days |
| Cost | CPU-only, no GPU at any point | GPU cluster |
| CIFAR-10 (E3) | 1 500 generated / 75 kept per round, 25 rounds | 50 000 / 2 500 |
| Text (E4) | 30.0M-param GPT-2, WikiText-2, 20 rounds | full-scale LM |
| Outcome | 5 VERIFIED, 1 FALSIFIED, every gate passing | — |

The 5% keep ratio — the selection pressure the theory is about — is preserved exactly;
only absolute counts and model sizes are reduced.

**Honest caveat:** for Claim 5 the M=2 arm collapsed as completely as single-reward
curation; only M=5 sustains diversity. Pluralism per se is not sufficient at this scale,
and that is reported rather than dropped.

Code: [{GH.split('/')[-1][:60]}…]({GH}) · [Space]({'https://huggingface.co/spaces/' + SPACE})""")
    poster = cell({"type": "figure", "title": "Reproduction poster", "pinned": True,
                   "poster": True}, poster_html())
    return head + summary + poster


def conclusion_page() -> str:
    return "# Conclusion\n" + cell(
        {"type": "markdown", "title": "Conclusion", "pinned": True},
        f"""**Five claims verified, one falsified, none left inconclusive.**

| # | Claim | Verdict | Decided by |
|---|---|---|---|
| 1 | Lemma 3.3 — outside decay | **FALSIFIED** (main text) / **VERIFIED** (appendix) | exact-rational counterexample + symbolic boundary |
| 2 | Lemma 3.5 — non-collapse | **VERIFIED** | symbolic certificate + 432 continuous landscapes |
| 3 | Theorem 3.6 — variance bound | **VERIFIED** | symbolic proof, free within-basin variance + 2 986 mixtures |
| 4 | Theorem 3.7 — Nash bargaining | **VERIFIED** | continuum certificate + independent 50-digit bisection |
| 5 | E3 — CIFAR-10 flow retraining | **VERIFIED** | 3-arm comparison, margin +1.1427 |
| 6 | E4 — text retraining | **VERIFIED** | 2-arm comparison, margin +2.4241 |

### What the falsification does and does not say

It does **not** say the theorem is wrong. Appendix Lemma B.4 is correct under its stated
hypothesis, and that implication is proved symbolically here. It says the **main-text
statement omits a hypothesis it needs**, and that the omission is reachable by admissible
parameters — in the paper's own quadratic reward geometry (Appendix C.4),
`sup_{{x∉S_ε}} W_t(x)` reaches 4.09–6.51, so B.4 is violated there while `m_t` still
decays. B.4 is therefore sufficient but not necessary.

### Two defects found in the reproduction's own machinery, and fixed

- **Vacuous verdicts.** The per-configuration E3/E4 stages hardcoded `VERIFIED` and checked
  only that the run happened. Both claim sentences are comparative, so those gates passed
  on nothing. Replaced by a stage that derives the status from the measured arms.
- **A padding bug that invalidated a whole E4 round.** `encode()` padded to 96 tokens with
  EOS while training on `labels=input_ids`, so only ~16% of targets were real tokens and
  the model learned to emit EOS immediately (loss 0.3, generations of 0.0 words). The
  verdict gate refused to record a verdict for that run. After masking with `-100`, loss
  starts at 10.84 ≈ ln(50257) and generations run 12–28 words.

### Limitations

One seed per empirical arm, so margins are point estimates. FID is not reported (it needs
an InceptionV3 pass outside the CPU-only budget and is not part of the claim sentences).
The baseline 3-state reference node is recorded as BLOCKED and is deliberately **not**
counted as evidence.

Full report and figures: [{GH.split('/')[-1][:50]}…]({GH}/blob/main/report/REPORT.md)""")


def main() -> int:
    C = Path("/tmp/space_candidate")
    imgs = Path("images")

    # --- figures, so claim pages can show their evidence ---------------------- #
    (C / "images").mkdir(exist_ok=True)
    for p in imgs.glob("*.png"):
        shutil.copy2(p, C / "images" / p.name)

    # --- rename claim pages to the canonical slugs ---------------------------- #
    for old, (new, _title) in RENAME.items():
        src, dst = C / "pages" / old, C / "pages" / new
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        page = dst / "page.md"
        body = page.read_text()
        verdict, note = VERDICT[new]
        banner = (f"> **Verdict: {verdict}**{(' — ' + note) if note else ''}. "
                  f"Decided by run `{RUN}` on Hugging Face `cpu-upgrade` (8 CPUs, no GPU). "
                  f"Code: [{GH.split('/')[-1][:48]}…]({GH})\n\n")
        figs = "\n".join(f"![{new}](../../images/{f})" for f in FIGURE.get(new, []))
        if figs:
            figs = "\n## Evidence figures\n\n" + figs + "\n"
        page.write_text(banner + body + figs)

    # --- executive summary and conclusion ------------------------------------- #
    (C / "pages/executive-summary").mkdir(parents=True, exist_ok=True)
    (C / "pages/executive-summary/page.md").write_text(executive_summary())
    (C / "pages/conclusion-current").mkdir(parents=True, exist_ok=True)
    (C / "pages/conclusion-current/page.md").write_text(conclusion_page())
    shutil.rmtree(C / "pages/verification-summary", ignore_errors=True)

    # --- navigation ------------------------------------------------------------ #
    order = [("executive-summary", "Executive summary")]
    order += [(new, title) for _o, (new, title) in RENAME.items()]
    order += [("conclusion-current", "Conclusion"),
              ("visibility-matrix", "Evaluator visibility matrix"),
              ("reproduce", "Reproducing this"),
              ("baseline-judged-reference", "Reference: the judged 3-state reduction")]
    order += [(s, f"Historical rejected baseline — {t}") for s, t in HISTORICAL]

    lb = json.loads((C / "logbook.json").read_text())
    lb["title"] = f"Reproduction: {PAPER_TITLE}"
    lb["tags"] = TAGS
    lb["root"]["title"] = lb["title"]
    lb["root"]["children"] = [{"slug": s, "title": t, "file": f"pages/{s}/page.md",
                               "children": []} for s, t in order]
    (C / "logbook.json").write_text(json.dumps(lb, indent=2) + "\n")

    rows = "\n".join(f"| [{t}](#/{s}) |" for s, t in order)
    (C / "pages/index.md").write_text(
        f"# Reproduction: {PAPER_TITLE}\n\n"
        f"[OpenReview](https://openreview.net/forum?id={ORID}) · [Code]({GH})\n\n"
        f"## Pages\n\n| Page |\n| --- |\n{rows}\n")

    # --- README: the tags here are how the judge finds this logbook ----------- #
    (C / "README.md").write_text(
        "---\n"
        f'title: "Reproduction: Curated Synthetic Data Doesn\'t Have to Collapse"\n'
        "emoji: 🎯\ncolorFrom: yellow\ncolorTo: red\nsdk: static\npinned: false\ntags:\n"
        + "".join(f" - {t}\n" for t in TAGS) + "---\n\n"
        f"# Reproduction: {PAPER_TITLE}\n\n"
        "An open experiment logbook, published with "
        "[Trackio](https://github.com/gradio-app/trackio).\n\n"
        "All six claims decided with reproducible evidence: **five VERIFIED, one FALSIFIED**\n"
        "as stated in the main text (and VERIFIED as stated in the appendix). CPU-only\n"
        f"throughout. Citable run `{RUN}`.\n\n"
        f"Code and full report: {GH}\n")

    missing = [s for s, _t in order if not (C / "pages" / s / "page.md").is_file()]
    print("nav entries:", len(order), "| pages missing:", missing or "none")
    print("tags:", lb["tags"])
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
