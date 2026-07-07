#!/usr/bin/env python3
"""Robustness for the probe-target drop: is the answer-perplexity drop caused by the RELEVANT reasoning,
or would ANY inserted rationale help (just more tokens before the answer)? Adds a placebo arm = another
doc's rationale, format-matched but irrelevant. If real << placebo, the drop is the reasoning's content."""
import json, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"
doc = {}
for f in sorted(glob.glob("data/probe_batch_*.jsonl")):
    for l in open(f):
        if l.strip():
            r = json.loads(l); doc[r["id"]] = r
probe = {}
for f in sorted(glob.glob("data/probe_out_*.jsonl")):
    for l in open(f):
        if l.strip():
            try:
                r = json.loads(l); probe[r["id"]] = r
            except Exception:
                pass
ids = [i for i in probe if i in doc]

print(f"loading judge ...", flush=True)
tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto").eval()
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

real, plac, rows = [], [], []
for k, i in enumerate(ids):
    ctx, rat = doc[i]["context"], doc[i]["rationale"]
    q, a = probe[i]["question"], probe[i]["answer"]
    if a.strip().lower() in rat.lower() or a.strip().lower() in ctx.lower():
        continue
    placebo_rat = doc[ids[(k + 1) % len(ids)]]["rationale"]   # another doc's rationale (irrelevant)
    b = nll_answer(f"{ctx}\nQuestion: {q}\nAnswer:", a)
    r = nll_answer(f"{ctx}\n{rat}\nQuestion: {q}\nAnswer:", a)
    p = nll_answer(f"{ctx}\n{placebo_rat}\nQuestion: {q}\nAnswer:", a)
    if None in (b, r, p):
        continue
    real.append(r - b); plac.append(p - b)
    rows.append({"id": i, "answer": a, "real": round(r - b, 3), "placebo": round(p - b, 3),
                 "real_minus_placebo": round((r - b) - (p - b), 3)})

with open("data/probe_placebo_perprobe.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
mean = lambda xs: sum(xs) / len(xs)
print(f"\n=== PROBE placebo control (n={len(real)}) ===")
print(f"real rationale     delta: {mean(real):+.3f}  ({100*sum(d<0 for d in real)/len(real):.0f}% drop)")
print(f"placebo rationale  delta: {mean(plac):+.3f}  ({100*sum(d<0 for d in plac)/len(plac):.0f}% drop)")
print(f"real − placebo (isolates RELEVANT reasoning): {mean(real)-mean(plac):+.3f}")
print("\n-- per-probe, sorted by real-minus-placebo (most reasoning-specific first) --")
for x in sorted(rows, key=lambda z: z["real_minus_placebo"]):
    tag = "REASONING" if x["real_minus_placebo"] < -0.1 else ("priming" if x["real"] < -0.1 else "flat")
    print(f"  [{x['id']:4d}] real {x['real']:+.2f} placebo {x['placebo']:+.2f} r-p {x['real_minus_placebo']:+.2f} {tag:9s}| {x['answer'][:26]}")
print("DONE", flush=True)
