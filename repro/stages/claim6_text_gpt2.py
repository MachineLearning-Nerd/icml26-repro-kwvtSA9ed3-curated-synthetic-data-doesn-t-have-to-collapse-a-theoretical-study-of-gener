"""Claim 6 / E4 -- text-domain retraining of a real GPT-2-style LM under length preferences.

Follows Appendix C.6 exactly where the paper specifies a value:

  data          WikiText-2 (raw), 1,000 initial seed sequences
  model         GPT-2-style decoder: 6 layers, 6 heads, embedding dim 384, vocab 50,257
  reward        R(y;T) = -|L(y) - T|, L = word count; targets T_A, T_B; d = |T_A - T_B|
  loop          N = 20 rounds; each round curates 200 samples with a balanced mixture
                policy (q = 0.5) using BT sampling proportional to exp(R/tau), tau = 0.5;
                fine-tunes with AdamW, lr 5e-5, 2 epochs, batch size 8; generates 200 new
                samples by nucleus sampling at temperature 0.8; filters with the same
                selection rule and adds survivors to the pool
  metric        H(L), the discrete entropy of the generated length distribution per round

The judged evidence replaced all of this with a closed-form 2-basin entropy calculation
and no language model at all. This stage trains and fine-tunes an actual autoregressive
LM on its own curated generations, which is what the claim is about.

Length has to be something the model can actually control, so sequences are terminated
with the EOS token and generation stops at EOS: the length distribution is then a
learned property of the model rather than a decoding constant.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from repro.lib import report
from repro.lib.verdict import FALSIFIED, VERIFIED, Verdict

MAX_LEN = 96


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_wikitext(tokenizer, n_seed: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    lines = [t.strip() for t in ds["text"]]
    # keep prose lines of a usable length; drop headings ("= Title =") and blanks
    lines = [t for t in lines if t and not t.startswith("=") and 5 <= len(t.split()) <= 60]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lines))
    seed_pool = [lines[i] for i in idx[:n_seed]]
    # WikiText-2 yields ~4k usable prose lines, so pretraining cycles the remainder
    # rather than consuming it linearly.
    pretrain = [lines[i] for i in idx[n_seed:]]
    return seed_pool, pretrain


def encode(tokenizer, texts: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (input_ids, labels) with padding positions masked out of the loss.

    Sequences are padded to MAX_LEN with EOS, and WikiText prose lines here are 5-60
    words, so most positions in a padded row are padding. Training with labels equal to
    the inputs therefore scores the model mostly on predicting padding, and the cheapest
    way to win that game is to emit EOS immediately -- which is exactly what happened:
    pretrain loss collapsed to ~0.3, far below anything plausible for prose, and
    generations came out at ~0.0 words with a single distinct length from round one.
    Masking padding with -100 makes the loss depend only on real tokens, so length stays
    a property the model has to learn rather than a decoding artifact.
    """
    eos = tokenizer.eos_token_id
    ids_out = torch.full((len(texts), MAX_LEN), eos, dtype=torch.long)
    labels = torch.full((len(texts), MAX_LEN), -100, dtype=torch.long)
    for i, t in enumerate(texts):
        ids = tokenizer(t, truncation=True, max_length=MAX_LEN - 1)["input_ids"] + [eos]
        row = torch.tensor(ids, dtype=torch.long)
        ids_out[i, : len(ids)] = row
        labels[i, : len(ids)] = row  # the terminating EOS IS supervised; the padding is not
    return ids_out.to(device), labels.to(device)


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(seed: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    cfg = GPT2Config(
        vocab_size=50257, n_positions=MAX_LEN, n_embd=384, n_layer=6, n_head=6,
        bos_token_id=50256, eos_token_id=50256,
    )
    return GPT2LMHeadModel(cfg)


def train_steps(model, batches, lr: float, label: str, log_every: int = 200) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for i, (x, lab) in enumerate(batches):
        out = model(x, labels=lab)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(out.loss))
        if log_every and (i + 1) % log_every == 0:
            report.kv(f"{label} step {i + 1}", f"loss {np.mean(losses[-log_every:]):.4f}")
    return float(np.mean(losses[-50:])) if losses else float("nan")


@torch.no_grad()
def generate(model, tokenizer, n: int, temperature: float, top_p: float,
             batch: int, gen_seed: int) -> list[str]:
    """Nucleus sampling; each sequence stops at EOS so lengths are model-determined."""
    model.eval()
    torch.manual_seed(gen_seed)
    eos = tokenizer.eos_token_id
    texts: list[str] = []
    while len(texts) < n:
        b = min(batch, n - len(texts))
        ids = torch.full((b, 1), eos, dtype=torch.long)
        done = torch.zeros(b, dtype=torch.bool)
        for _ in range(MAX_LEN - 1):
            logits = model(ids).logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            sp, si = torch.sort(probs, descending=True, dim=-1)
            keep = (torch.cumsum(sp, dim=-1) - sp) < top_p
            sp = torch.where(keep, sp, torch.zeros_like(sp))
            sp = sp / sp.sum(dim=-1, keepdim=True)
            nxt = si.gather(-1, torch.multinomial(sp, 1))
            nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, eos), nxt)
            ids = torch.cat([ids, nxt], dim=1)
            done = done | (nxt.squeeze(1) == eos)
            if bool(done.all()):
                break
        for row in ids:
            toks = row.tolist()[1:]
            if eos in toks:
                toks = toks[: toks.index(eos)]
            texts.append(tokenizer.decode(toks).strip())
    return texts[:n]


# --------------------------------------------------------------------------- #
# curation (Appendix C.3 selection rule, C.6 reward)
# --------------------------------------------------------------------------- #
def bt_curate(pool: list[str], targets: list[float], q: float, n_curated: int,
              tau: float, rng: np.random.Generator) -> tuple[list[str], dict]:
    """Repeat n_curated times: draw the active reward from the mixture, then BT-sample."""
    lengths = np.array([word_count(t) for t in pool], dtype=np.float64)
    rewards = [-np.abs(lengths - T) for T in targets]
    probs = []
    for r in rewards:
        z = (r - r.max()) / tau
        e = np.exp(z)
        probs.append(e / e.sum())
    weights = [q, 1.0 - q] if len(targets) == 2 else [1.0]
    chosen, active = [], []
    for _ in range(n_curated):
        k = int(rng.choice(len(targets), p=weights))
        active.append(k)
        chosen.append(int(rng.choice(len(pool), p=probs[k])))
    sel_len = lengths[chosen]
    # leakage proxy (Appendix C.7): fraction of selections nearer the OTHER target
    leak = 0.0
    if len(targets) == 2:
        near = np.argmin(np.abs(sel_len[:, None] - np.array(targets)[None, :]), axis=1)
        leak = float(np.mean(near != np.array(active)))
    return [pool[i] for i in chosen], {
        "mean_selected_length": float(sel_len.mean()),
        "leakage_proxy": leak,
    }


def length_entropy(texts: list[str]) -> float:
    """Discrete entropy H(L) of the generated length distribution, in nats."""
    counts = Counter(word_count(t) for t in texts)
    n = sum(counts.values())
    return float(-sum((c / n) * math.log(c / n) for c in counts.values() if c))


# --------------------------------------------------------------------------- #
def run(params: dict) -> Verdict:
    out = report.artifact_dir("claim6", "text_gpt2")
    torch.set_num_threads(int(params.get("threads", 8)))

    T_A = float(params.get("T_A", 10))
    T_B = params.get("T_B", 30)
    single_reward = T_B is None
    rounds = int(params.get("rounds", 20))
    n_seed = int(params.get("n_seed", 1000))
    n_curated = int(params.get("n_curated", 200))
    n_generate = int(params.get("n_generate", 200))
    tau = float(params.get("tau", 0.5))
    q = float(params.get("q", 0.5))
    lr = float(params.get("lr", 5e-5))
    epochs = int(params.get("epochs", 2))
    batch = int(params.get("batch", 8))
    temperature = float(params.get("temperature", 0.8))
    top_p = float(params.get("top_p", 0.95))
    pretrain_steps = int(params.get("pretrain_steps", 1200))
    pretrain_batch = int(params.get("pretrain_batch", 16))
    seed = int(params.get("seed", 0))
    d = None if single_reward else abs(T_A - float(T_B))

    label = "single-reward" if single_reward else f"d={d:g}"
    report.kv("configuration", f"{label}  T_A={T_A:g}  T_B={T_B}  q={q}  seed={seed}")
    report.kv("rounds / curated / generated", f"{rounds} / {n_curated} / {n_generate}")
    report.kv("tau / lr / epochs / batch", f"{tau} / {lr} / {epochs} / {batch}")
    report.kv("torch threads", torch.get_num_threads())

    from transformers import GPT2TokenizerFast

    device = torch.device("cpu")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    seed_pool, pretrain_texts = load_wikitext(tok, n_seed, seed)
    report.kv("seed pool / pretrain corpus", f"{len(seed_pool)} / {len(pretrain_texts)} lines")

    model = build_model(seed).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    report.kv("model parameters", f"{n_params / 1e6:.1f}M (6 layers, 6 heads, d=384, vocab 50257)")

    # ---- pretraining: identical and deterministic on every node ----------- #
    report.banner(f"Pretraining the seed LM on WikiText-2 ({pretrain_steps} steps)")
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pre_batches = []
    order: list[int] = []
    for i in range(pretrain_steps):
        if len(order) < pretrain_batch:
            order = list(rng.permutation(len(pretrain_texts)))
        take, order = order[:pretrain_batch], order[pretrain_batch:]
        pre_batches.append(encode(tok, [pretrain_texts[j] for j in take], device))
    # Make the padding mask auditable: if this fraction were 1.0 the loss would again be
    # dominated by padding, which is the failure that produced the earlier empty samples.
    supervised = float(np.mean([(lab != -100).float().mean().item() for _, lab in pre_batches]))
    report.kv("supervised (non-padding) positions", f"{supervised:.1%} of {MAX_LEN} per sequence")
    pre_loss = train_steps(model, pre_batches, lr=3e-4, label="pretrain", log_every=200)
    report.kv("pretrain final loss / wall clock", f"{pre_loss:.4f} / {time.time() - t0:.0f}s")

    targets = [T_A] if single_reward else [T_A, float(T_B)]

    # ---- recursive retraining loop ---------------------------------------- #
    report.banner(f"Recursive curated retraining: {rounds} rounds ({label})")
    pool = list(seed_pool)
    rows = []
    for rnd in range(1, rounds + 1):
        tr0 = time.time()
        curated, cinfo = bt_curate(pool, targets, q, n_curated, tau, rng)
        batches = []
        for _ in range(epochs):
            order = rng.permutation(len(curated))
            for s in range(0, len(curated) - batch + 1, batch):
                batches.append(encode(tok, [curated[i] for i in order[s:s + batch]], device))
        loss = train_steps(model, batches, lr=lr, label=f"round{rnd}", log_every=0)
        gen = generate(model, tok, n_generate, temperature, top_p,
                       batch=int(params.get("gen_batch", 50)), gen_seed=seed * 1000 + rnd)
        H = length_entropy(gen)
        lens = [word_count(t) for t in gen]
        survivors, _ = bt_curate(gen, targets, q, n_generate, tau, rng)
        pool = pool + list(dict.fromkeys(survivors))
        rows.append({
            "round": rnd, "H_L": H,
            "mean_len": float(np.mean(lens)), "std_len": float(np.std(lens)),
            "n_distinct_lengths": len(set(lens)),
            "frac_near_T_A": float(np.mean([abs(l - T_A) <= 2 for l in lens])),
            "frac_near_T_B": (float(np.mean([abs(l - float(T_B)) <= 2 for l in lens]))
                              if not single_reward else float("nan")),
            "train_loss": loss, "pool_size": len(pool),
            "mean_selected_length": cinfo["mean_selected_length"],
            "leakage_proxy": cinfo["leakage_proxy"],
            "seconds": time.time() - tr0,
        })
        report.kv(f"round {rnd:>2d}", f"H(L)={H:.4f}  mean_len={np.mean(lens):5.1f}  "
                                     f"distinct={len(set(lens)):3d}  nearA={rows[-1]['frac_near_T_A']:.2f}  "
                                     f"nearB={rows[-1]['frac_near_T_B']:.2f}  loss={loss:.3f}  "
                                     f"{rows[-1]['seconds']:.0f}s")
        report.write_csv(out / f"rounds_{label.replace('=', '')}_seed{seed}.csv", rows)
        with (out / f"samples_{label.replace('=', '')}_seed{seed}.json").open("w") as fh:
            json.dump({"round": rnd, "examples": gen[:20], "lengths": lens}, fh, indent=2)

    H_series = [r["H_L"] for r in rows]
    H_first, H_last = H_series[0], H_series[-1]
    H_tail = float(np.mean(H_series[-5:]))
    report.kv("H(L) first / last / mean of last 5", f"{H_first:.4f} / {H_last:.4f} / {H_tail:.4f}")

    v = Verdict(
        claim_id="claim6/E4-text-length-entropy",
        title="E4: pluralistic curation sustains length entropy in a real LM retraining loop",
        status=VERIFIED,
        statement=(
            "Curation under two competing length-based preferences sustains the entropy "
            "H(L) of the generated length distribution over recursive retraining, and "
            "larger conflict distance d = |T_A - T_B| gives higher sustained entropy."
        ),
    )
    v.add(
        "a real GPT-2-style LM was pretrained and then fine-tuned on its own curated "
        "generations for every round",
        pre_loss < 9.0 and all(np.isfinite(r["train_loss"]) for r in rows),
        f"{n_params / 1e6:.1f}M-parameter decoder (6 layers, 6 heads, d=384, vocab 50257) "
        f"as specified in Appendix C.6; pretrain loss {pre_loss:.4f}; {rounds} rounds of "
        f"AdamW lr={lr} for {epochs} epochs at batch {batch} on {n_curated} BT-curated samples",
        n_params=n_params, pretrain_loss=pre_loss,
    )
    v.numbers = {
        "config": label, "T_A": T_A, "T_B": T_B, "d": d, "seed": seed,
        "H_first": H_first, "H_last": H_last, "H_tail_mean": H_tail,
        "H_series": H_series,
        "mean_len_last": rows[-1]["mean_len"],
        "n_params": n_params,
        "total_seconds": sum(r["seconds"] for r in rows),
    }
    report.write_json(out / f"summary_{label.replace('=', '')}_seed{seed}.json", v.numbers)

    if single_reward:
        v.status = VERIFIED
        v.claim_id = "claim6/E4-single-reward-control"
        v.title = "E4 negative control: single-reward curation"
        v.add_control(
            "single-reward curation drives H(L) down (the collapse pluralism is meant to avoid)",
            H_tail < H_first,
            f"H(L) falls from {H_first:.4f} to a tail mean of {H_tail:.4f} under a single "
            f"length preference T_A={T_A:g}; mean generated length converges to "
            f"{rows[-1]['mean_len']:.1f}",
        )
        v.add(
            "the control ran the identical loop, so any entropy difference is due to the "
            "preference structure alone",
            True,
            "same model, seed, pretraining, rounds, optimiser and decoding; only the "
            "reward mixture differs",
        )
    else:
        sustained = H_tail > 0.5 * H_first and H_tail > 1.0
        v.add(
            "H(L) is sustained rather than collapsing across the 20 recursive rounds",
            bool(sustained),
            f"H(L) starts at {H_first:.4f} and the mean over the final 5 rounds is "
            f"{H_tail:.4f} (min over all rounds {min(H_series):.4f}); the generated length "
            f"distribution still spans {rows[-1]['n_distinct_lengths']} distinct lengths in "
            f"the final round",
        )
        v.add_control(
            "both length basins stay populated, so the model hedges instead of picking one",
            rows[-1]["frac_near_T_A"] > 0.02 and rows[-1]["frac_near_T_B"] > 0.02,
            f"final round: {rows[-1]['frac_near_T_A']:.3f} of generations within 2 words of "
            f"T_A={T_A:g} and {rows[-1]['frac_near_T_B']:.3f} within 2 words of T_B={T_B}. "
            "If the model had collapsed to one compromise length, one of these would be ~0.",
        )
    v.limitations = [
        "Downscaling relative to Appendix C.6: none in the retraining loop (rounds, "
        f"curated count, tau, q, lr, epochs, batch and decoding temperature are the "
        f"paper's values), but the seed LM is pretrained here for {pretrain_steps} steps "
        "on WikiText-2 because the paper does not release or specify a checkpoint for its "
        "6-layer/6-head/384-dim architecture.",
        "Sequences are capped at 96 tokens, so word counts above roughly 60 cannot occur; "
        "targets are chosen inside that range.",
        "top_p = 0.95 for nucleus sampling: the paper states the temperature (0.8) but not "
        "the nucleus mass.",
    ]
    v.artifacts = [str(p) for p in sorted(out.rglob("*")) if p.is_file()]
    return v
