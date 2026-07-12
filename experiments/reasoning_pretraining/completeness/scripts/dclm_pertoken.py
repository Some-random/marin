#!/usr/bin/env python3
"""Per-token perplexity of the real DCLM target continuation, under base / +complete / +incomplete / +placebo
(prose rationales, mid-continuation insertion — the exact scoring of perplexity_complete.py). Shows the full
context, the target, all three rationales (complete / incomplete-with-gap / placebo), and each target token's
NLL in every condition, so you can SEE: complete≈incomplete (completeness null), placebo hurts, complete≪placebo.
Judge = DCLM-1.4B base. Usage: dclm_pertoken.py [id ...]"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"
OUT = "docs/DCLM_PERTOKEN.md"

# replicate perplexity_complete.py ordering + leak filter, so placebo rotation matches the aggregate
rows = [json.loads(l) for l in open("data/complete_dataset_prose.jsonl") if l.strip()]
kept = {}
for r in rows:
    if r["target"].strip() and r["complete"].strip() and r["incomplete"].strip():
        cw = [w for w in r["target"].lower().split() if len(w) > 4]
        if cw and sum(w in r["complete"].lower() for w in cw) / len(cw) > 0.6:
            continue  # leaked
        kept[r["id"]] = (r["context"], r["target"].strip(), r["complete"], r["incomplete"])
ids = list(kept.keys())
want = [int(x) for x in sys.argv[1:]] or ids[:4]

def placebo_of(i):
    k = ids.index(i)
    return kept[ids[(k + 1) % len(ids)]][2], ids[(k + 1) % len(ids)]  # next doc's complete rationale

tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def per_token(prefix, cont):
    pre = tok(prefix + "\n", return_tensors="pt").input_ids
    full = tok(prefix + "\n" + cont, return_tensors="pt").input_ids.to(dev)
    n = pre.shape[1]
    with torch.no_grad():
        lg = model(full).logits[0]
    lab = full[0, n:]
    lp = torch.log_softmax(lg[n - 1:-1].float(), dim=-1)
    nlls = [-lp[i, lab[i]].item() for i in range(len(lab))]
    toks = [tok.decode([t]) for t in lab]
    return toks, nlls

L = ["# DCLM — per-token perplexity of the real target continuation\n",
     "For each real DCLM doc: the full context, the target span that actually follows, and three rationales that",
     "get spliced BETWEEN context and target — `complete` (all steps), `incomplete` (same chain, load-bearing",
     "middle steps deleted → a real gap), `placebo` (an UNRELATED doc's complete rationale). Then every target",
     "token's NLL (nats/tok) under `base` (no rationale) and each splice. Judge = DCLM-1.4B.\n",
     "**What to look for:** `+complete` ≈ `+incomplete` (completeness makes no difference — the model fills the",
     "gap); `+placebo` is *higher* than base (an off-topic splice hurts); `+complete` ≪ `+placebo` (the relevant",
     "reasoning is real). The net `+complete` vs `base` is small because the splice's insertion cost ~cancels it.\n"]

import math
for i in want:
    context, target, complete, incomplete = kept[i]
    placebo, pid = placebo_of(i)
    setups = {"base": per_token(context, target),
              "complete": per_token(context + "\n" + complete, target),
              "incomplete": per_token(context + "\n" + incomplete, target),
              "placebo": per_token(context + "\n" + placebo, target)}
    toks = setups["base"][0]
    L += [f"\n---\n\n## doc {i}\n",
          "**Context (real DCLM doc):**\n", "> " + context.strip().replace("\n", "\n> ") + "\n",
          f"**Target scored (the real continuation):** `{target}`\n",
          "**① complete rationale (all steps):**\n", "> " + complete.strip().replace("\n", "\n> ") + "\n",
          "**② incomplete rationale (middle steps deleted — note the gap):**\n", "> " + incomplete.strip().replace("\n", "\n> ") + "\n",
          f"**③ placebo (unrelated doc {pid}'s complete rationale):**\n", "> " + placebo.strip().replace("\n", "\n> ") + "\n",
          "Per-token NLL of the target:\n",
          "| # | token | base | +complete | +incomplete | +placebo | Δ(compl−base) |",
          "|---:|---|---:|---:|---:|---:|---:|"]
    for j, t in enumerate(toks):
        b, c, ic, p = setups["base"][1][j], setups["complete"][1][j], setups["incomplete"][1][j], setups["placebo"][1][j]
        L.append(f"| {j} | `{t}` | {b:.3f} | {c:.3f} | {ic:.3f} | {p:.3f} | {c-b:+.3f} |")
    mean = {k: sum(setups[k][1]) / len(setups[k][1]) for k in setups}
    L.append(f"| | **MEAN** | **{mean['base']:.3f}** | **{mean['complete']:.3f}** | **{mean['incomplete']:.3f}** | **{mean['placebo']:.3f}** | **{mean['complete']-mean['base']:+.3f}** |")
    L.append(f"\n**Perplexity:** base {math.exp(mean['base']):.2f} · +complete {math.exp(mean['complete']):.2f} · "
             f"+incomplete {math.exp(mean['incomplete']):.2f} · +placebo {math.exp(mean['placebo']):.2f}  ·  "
             f"complete−incomplete {mean['complete']-mean['incomplete']:+.3f} (completeness) · "
             f"complete−placebo {mean['complete']-mean['placebo']:+.3f} (content)\n")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(want)} docs: {want})")
