#!/usr/bin/env python3
"""Completeness test on real DCLM data. For docs whose real continuation needs a MULTI-STEP reasoning chain,
score the real target under the judge given: context (base), context+COMPLETE rationale, context+INCOMPLETE
(gap-broken) rationale, context+PLACEBO (another doc's complete rationale). Key numbers:
  complete − base       : does adding the complete chain lower the real target's perplexity
  complete − placebo    : is it the specific reasoning (not priming) — placebo should hurt
  complete − incomplete : THE COMPLETENESS EFFECT — does a gap-free chain beat a gap-broken one (<0 = yes)
"""
import json, glob, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
ap.add_argument("--out", default="data/complete_results.jsonl")
args = ap.parse_args()

ctx = {}
kept = {}
for l in open("data/complete_dataset.jsonl"):
    if not l.strip():
        continue
    r = json.loads(l)
    if r.get("target", "").strip() and r.get("complete", "").strip() and r.get("incomplete", "").strip():
        ctx[r["id"]] = r["context"]
        kept[r["id"]] = (r["target"].strip(), r["complete"], r["incomplete"])
ids = list(kept.keys())
print(f"{len(ids)} kept multi-step docs (target + complete + incomplete)", flush=True)

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
dcomp, dincomp, dplac, dcompVincomp = [], [], [], []
leaked = 0
for k, i in enumerate(ids):
    c = ctx[i]; target, comp, incomp = kept[i]
    cw = [w for w in target.lower().split() if len(w) > 4]
    if cw and sum(w in comp.lower() for w in cw) / len(cw) > 0.6:
        leaked += 1; continue
    placebo = kept[ids[(k + 1) % len(ids)]][1]  # another doc's complete rationale
    b = nll(c, target)
    rc = nll(c + "\n" + comp, target)
    ri = nll(c + "\n" + incomp, target)
    rp = nll(c + "\n" + placebo, target)
    if None in (b, rc, ri, rp):
        continue
    dcomp.append(rc - b); dincomp.append(ri - b); dplac.append(rp - b); dcompVincomp.append(rc - ri)
    out.write(json.dumps({"id": i, "target": target, "base": round(b, 3), "complete": round(rc, 3),
                          "incomplete": round(ri, 3), "placebo": round(rp, 3),
                          "complete_minus_base": round(rc - b, 3),
                          "complete_minus_incomplete": round(rc - ri, 3),
                          "complete_minus_placebo": round(rc - rp, 3)}) + "\n")
    out.flush()
out.close()
m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"\n=== COMPLETENESS TEST on real multi-step targets (n={len(dcomp)}, {leaked} leaked) ===")
print(f"complete   − base : {m(dcomp):+.3f}  ({100*sum(d<0 for d in dcomp)/len(dcomp):.0f}% drop)")
print(f"incomplete − base : {m(dincomp):+.3f}  ({100*sum(d<0 for d in dincomp)/len(dincomp):.0f}% drop)")
print(f"placebo    − base : {m(dplac):+.3f}  ({100*sum(d<0 for d in dplac)/len(dplac):.0f}%)")
print(f"complete − placebo   (reasoning-specific): {m(dcomp)-m(dplac):+.3f}")
print(f"complete − incomplete (COMPLETENESS effect): {m(dcompVincomp):+.3f}  (<0 = complete chain beats gap-broken; complete<incomplete on {sum(d<0 for d in dcompVincomp)}/{len(dcompVincomp)})")
print("DONE ->", args.out, flush=True)
