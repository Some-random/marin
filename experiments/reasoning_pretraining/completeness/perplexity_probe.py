#!/usr/bin/env python3
"""Probe-target test: does making the reasoning explicit help predict the answer to a reasoning question?
For each R doc: NLL(answer | context + Q)  vs  NLL(answer | context + rationale + Q), DCLM-1.4B base judge.
delta = with_rationale - base ; delta<0 = the rationale LOWERS the answer's perplexity.

Leakage guard: skip any probe whose answer appears verbatim in the rationale or context (that would be
copying, not reasoning). This is the honest test of 'reasoning helps predict reasoning-dependent content'."""
import json, glob, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
ap.add_argument("--out", default="data/probe_results.jsonl")
args = ap.parse_args()

# docs: id -> {context, rationale}
doc = {}
for f in sorted(glob.glob("data/probe_batch_*.jsonl")):
    for l in open(f):
        if l.strip():
            r = json.loads(l); doc[r["id"]] = r
# probes: id -> {question, answer}
probe = {}
for f in sorted(glob.glob("data/probe_out_*.jsonl")):
    for l in open(f):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l); probe[r["id"]] = r
        except Exception:
            continue
ids = [i for i in probe if i in doc]
print(f"{len(ids)} probes joined to docs", flush=True)

print(f"loading judge {args.judge} ...", flush=True)
tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def nll_answer(prefix, answer):
    pre = tok(prefix, return_tensors="pt").input_ids
    full = tok(prefix + " " + answer.strip(), return_tensors="pt").input_ids.to(dev)
    n_pre = pre.shape[1]
    if full.shape[1] <= n_pre or full.shape[1] > 3500:
        return None
    with torch.no_grad():
        logits = model(full).logits[0]
    labels = full[0, n_pre:]
    logp = torch.log_softmax(logits[n_pre - 1:-1].float(), dim=-1)
    return -logp[torch.arange(len(labels)), labels].mean().item()

out = open(args.out, "w")
deltas = []; leaked = 0
for i in ids:
    ctx, rat = doc[i]["context"], doc[i]["rationale"]
    q, a = probe[i]["question"], probe[i]["answer"]
    if a.strip().lower() in rat.lower() or a.strip().lower() in ctx.lower():
        leaked += 1
        continue  # leakage guard
    base_pref = f"{ctx}\nQuestion: {q}\nAnswer:"
    rat_pref = f"{ctx}\n{rat}\nQuestion: {q}\nAnswer:"
    b = nll_answer(base_pref, a); r = nll_answer(rat_pref, a)
    if b is None or r is None:
        continue
    d = r - b
    deltas.append(d)
    out.write(json.dumps({"id": i, "question": q, "answer": a,
                          "base": round(b, 4), "rat": round(r, 4), "delta": round(d, 4)}) + "\n")
    out.flush()
out.close()

mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
print(f"\n=== PROBE-TARGET RESULT (n={len(deltas)} after leakage guard; {leaked} leaked/skipped) ===")
print(f"mean delta (rationale - base): {mean(deltas):+.3f} nats/token  (<0 = rationale LOWERS answer ppl)")
print(f"fraction with a drop: {100*sum(d<0 for d in deltas)/len(deltas):.1f}%" if deltas else "no probes")
print("DONE ->", args.out, flush=True)
