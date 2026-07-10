#!/usr/bin/env python3
"""Dump the FULL per-token perplexity calculation to a readable file: for chosen docs, the context, the
complete rationale, the continuation being scored, and per-token NLL of that continuation under
`context` (base) vs `context + rationale`, with the diff, mean, and perplexity. Judge = DCLM-1.4B base.
Each model uses ITS OWN tokenizer (AutoTokenizer.from_pretrained(judge))."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"
OUT = "docs/PERTOKEN_EXAMPLES.md"
idxs = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4]

data = [json.loads(l) for l in open("data/complete_dataset.jsonl")]
tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def per_token(prefix, cont):
    pre = tok(prefix + "\n", return_tensors="pt").input_ids
    full = tok(prefix + "\n" + cont, return_tensors="pt").input_ids.to(dev)
    n = pre.shape[1]
    with torch.no_grad():
        logits = model(full).logits[0]
    labels = full[0, n:]
    logp = torch.log_softmax(logits[n - 1:-1].float(), dim=-1)
    nlls = [-logp[i, labels[i]].item() for i in range(len(labels))]
    toks = [tok.decode([t]) for t in labels]
    return toks, nlls

L = ["# Per-token perplexity calculation (DCLM-1.4B base judge)\n",
     "Judge tokenizer = the model's own (`AutoTokenizer.from_pretrained`). For each doc: the real continuation",
     "target's tokens, NLL (nats/token) under `context` (base) vs `context + complete rationale`, and the diff.",
     "**diff < 0 = the rationale made that token more predictable.** The reported per-doc Δ is the mean-diff.\n"]
for IX in idxs:
    d = data[IX]; ctx, target, comp = d["context"], d["target"], d["complete"]
    toks, base = per_token(ctx, target)
    _, rat = per_token(ctx + "\n" + comp, target)
    mb, mr = sum(base) / len(base), sum(rat) / len(rat)
    L += [f"\n---\n\n## doc id {d['id']}\n",
          f"**CONTEXT:**\n\n> {ctx.strip()}\n",
          f"**COMPLETE RATIONALE (inserted before the target):**\n\n> " + comp.strip().replace(chr(10), chr(10)+'> ') + "\n",
          f"**CONTINUATION being scored (the real next text):** `{target.strip()}`\n",
          "| # | token | NLL base | NLL +rationale | diff |",
          "|---:|---|---:|---:|---:|"]
    for i, (t, b, r) in enumerate(zip(toks, base, rat)):
        L.append(f"| {i} | `{t}` | {b:.3f} | {r:.3f} | {r-b:+.3f} |")
    L += [f"| | **MEAN** | **{mb:.3f}** | **{mr:.3f}** | **{mr-mb:+.3f}** |",
          f"\n**perplexity:** base {torch.tensor(mb).exp():.2f} → +rationale {torch.tensor(mr).exp():.2f} · "
          f"**mean-diff (the reported Δ) = {mr-mb:+.4f} nats/token**\n"]
open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(idxs)} docs)")
