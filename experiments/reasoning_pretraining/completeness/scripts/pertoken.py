#!/usr/bin/env python3
"""Show EXACTLY how the perplexity reduction is computed, token by token, on one real doc.
For the target's tokens, print NLL under `context` (base) vs `context+complete rationale`, and the diff."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IDX = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d = [json.loads(l) for l in open("data/complete_dataset.jsonl")][IDX]
ctx, target, comp = d["context"], d["target"], d["complete"]

tok = AutoTokenizer.from_pretrained("/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
model = AutoModelForCausalLM.from_pretrained("/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf",
                                             torch_dtype=torch.bfloat16, device_map="auto").eval()
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

print(f"\n=== doc id {d['id']} ===")
print(f"CONTEXT (tail): ...{ctx[-220:].strip()}")
print(f"\nCOMPLETE RATIONALE inserted: {comp.strip()[:300]}...")
print(f"\nTARGET (real next text we score): {target.strip()}\n")

toks, base = per_token(ctx, target)
_, rat = per_token(ctx + "\n" + comp, target)
print(f"{'token':>16s} | {'NLL base':>9s} | {'NLL +rat':>9s} | {'Δ (rat-base)':>12s}")
print("-" * 56)
for t, b, r in zip(toks, base, rat):
    mark = "  ← rationale HELPS" if (r - b) < -0.15 else ("  ← rationale HURTS" if (r - b) > 0.15 else "")
    print(f"{repr(t):>16s} | {b:9.3f} | {r:9.3f} | {r-b:+12.3f}{mark}")
mb, mr = sum(base)/len(base), sum(rat)/len(rat)
print("-" * 56)
print(f"{'MEAN':>16s} | {mb:9.3f} | {mr:9.3f} | {mr-mb:+12.3f}   (this Δ is what we report per doc)")
print(f"\nperplexity: base {torch.tensor(mb).exp():.2f} -> +rationale {torch.tensor(mr).exp():.2f}")
