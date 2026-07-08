#!/usr/bin/env python3
"""Perplexity drop on ORIGINAL data (no synthetic Q&A). For real DCLM docs whose continuation IS the
doc's own conclusion (re-split at 'Thus/Therefore/As a result/…'), measure the NLL of that REAL conclusion
under the judge, given: context (base) vs context+rationale (real) vs context+ANOTHER-doc's-rationale
(placebo). real<base = adding the rationale lowers the real conclusion's perplexity; real<placebo = it's the
specific reasoning, not generic priming."""
import json, glob, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
ap.add_argument("--out", default="data/conclusion_results.jsonl")
args = ap.parse_args()

doc = {}
for f in sorted(glob.glob("data/concl_batches/batch_*.jsonl")):
    for l in open(f):
        if l.strip():
            r = json.loads(l); doc[r["id"]] = r
kept = {}
for f in sorted(glob.glob("data/concl_batches/out_*.jsonl")):
    for l in open(f):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
            if r.get("keep") and r.get("rationale", "").strip() and r["id"] in doc:
                kept[r["id"]] = r["rationale"]
        except Exception:
            continue
ids = list(kept.keys())
print(f"{len(ids)} kept conclusion-docs (genuine argument→conclusion, with rationale)", flush=True)

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
    ctx, concl, rat = doc[i]["context"], doc[i]["conclusion"], kept[i]
    # leakage guard: skip if most of the conclusion's content words are already in the rationale
    cwords = [w for w in concl.lower().split() if len(w) > 4]
    if cwords and sum(w in rat.lower() for w in cwords) / len(cwords) > 0.6:
        leaked += 1; continue
    placebo_rat = kept[ids[(k + 1) % len(ids)]]
    b = nll(ctx, concl)
    r = nll(ctx + "\n" + rat, concl)
    p = nll(ctx + "\n" + placebo_rat, concl)
    if None in (b, r, p):
        continue
    dr.append(r - b); dp.append(p - b)
    out.write(json.dumps({"id": i, "conclusion": concl, "base": round(b, 3), "real": round(r, 3),
                          "placebo": round(p, 3), "real_minus_base": round(r - b, 3),
                          "real_minus_placebo": round((r - b) - (p - b), 3)}) + "\n")
    out.flush()
out.close()

mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"\n=== PERPLEXITY ON REAL CONCLUSIONS (n={len(dr)}, {leaked} leaked/skipped) ===")
print(f"real − base    : {mean(dr):+.3f}   ({100*sum(d<0 for d in dr)/len(dr):.0f}% of docs drop)   <0 = rationale lowers the REAL conclusion's perplexity")
print(f"placebo − base : {mean(dp):+.3f}   ({100*sum(d<0 for d in dp)/len(dp):.0f}%)")
print(f"real − placebo : {mean(dr)-mean(dp):+.3f}   <0 = the SPECIFIC reasoning helps, not generic priming")
print("DONE ->", args.out, flush=True)
