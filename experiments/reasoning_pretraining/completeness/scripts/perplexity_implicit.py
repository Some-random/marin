#!/usr/bin/env python3
"""Same test as perplexity_conclusion.py but on IMPLICIT-reasoning docs (NO marker filtering — per
Dongwei: filtering for 'thus/therefore' selects docs whose reasoning is already explicit, defeating the
purpose). Agents picked, for real DCLM docs, a `target` = a real continuation span that follows from the
context's IMPLICIT (unstated) reasoning. Score that real span: base vs +rationale vs +placebo."""
import json, glob, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
ap.add_argument("--out", default="data/implicit_results.jsonl")
args = ap.parse_args()

ctxs = {}
for f in sorted(glob.glob("data/implicit_batches/batch_*.jsonl")):
    for l in open(f):
        if l.strip():
            r = json.loads(l); ctxs[r["id"]] = r["context"]
kept = {}
for f in sorted(glob.glob("data/implicit_batches/out_*.jsonl")):
    for l in open(f):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
            if r.get("keep") and r.get("rationale", "").strip() and r.get("target", "").strip() and r["id"] in ctxs:
                kept[r["id"]] = (r["target"].strip(), r["rationale"])
        except Exception:
            continue
ids = list(kept.keys())
print(f"{len(ids)} kept implicit-reasoning docs (real target + rationale)", flush=True)

print(f"loading judge {args.judge} ...", flush=True)
tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def nll(prefix, cont):
    pre = tok(prefix + "\n", return_tensors="pt").input_ids
    full = tok(prefix + "\n" + cont, return_tensors="pt").input_ids.to(dev)
    n = pre.shape[1]
    if full.shape[1] <= n + 1 or full.shape[1] > 3500:
        return None
    with torch.no_grad():
        lg = model(full).logits[0]
    lab = full[0, n:]
    lp = torch.log_softmax(lg[n - 1:-1].float(), dim=-1)
    return -lp[torch.arange(len(lab)), lab].mean().item()

out = open(args.out, "w")
dr, dp = [], []
leaked = 0
for k, i in enumerate(ids):
    ctx = ctxs[i]; target, rat = kept[i]
    cw = [w for w in target.lower().split() if len(w) > 4]
    if cw and sum(w in rat.lower() for w in cw) / len(cw) > 0.6:
        leaked += 1; continue
    placebo = kept[ids[(k + 1) % len(ids)]][1]
    b = nll(ctx, target); r = nll(ctx + "\n" + rat, target); p = nll(ctx + "\n" + placebo, target)
    if None in (b, r, p):
        continue
    dr.append(r - b); dp.append(p - b)
    out.write(json.dumps({"id": i, "target": target, "base": round(b, 3), "real": round(r, 3),
                          "placebo": round(p, 3), "real_minus_base": round(r - b, 3),
                          "real_minus_placebo": round((r - b) - (p - b), 3)}) + "\n")
    out.flush()
out.close()
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"\n=== IMPLICIT-REASONING TARGETS (no marker filter) n={len(dr)}, {leaked} leaked ===")
print(f"real − base    : {mean(dr):+.3f}  ({100*sum(d<0 for d in dr)/len(dr):.0f}% drop)")
print(f"placebo − base : {mean(dp):+.3f}  ({100*sum(d<0 for d in dp)/len(dp):.0f}%)")
print(f"real − placebo : {mean(dr)-mean(dp):+.3f}   (real<placebo on {sum(a<b for a,b in zip(dr,dp))}/{len(dr)})")
print("DONE ->", args.out, flush=True)
