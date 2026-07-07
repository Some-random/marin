#!/usr/bin/env python3
"""Confirmatory test: strict SPECIFIC-answer probes (no vague/directional answers). Measures, per probe,
base vs real-rationale vs placebo-rationale NLL of the answer under the DCLM-1.4B judge. The pre-stated
hypothesis (from the earlier per-probe split): on specific answers, the RELEVANT reasoning drops the
answer's perplexity while an irrelevant (placebo) rationale does not — i.e. real ≪ placebo."""
import json, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"
doc = {}
for f in sorted(glob.glob("data/probe_batch_*.jsonl")):
    for l in open(f):
        if l.strip():
            r = json.loads(l); doc[r["id"]] = r
probes = []
for f in sorted(glob.glob("data/probe_strict_*.jsonl")):
    for l in open(f):
        if l.strip():
            try:
                r = json.loads(l)
                if r["id"] in doc:
                    probes.append(r)
            except Exception:
                pass
doc_ids = list(doc.keys())
print(f"{len(probes)} strict probes over {len(set(p['id'] for p in probes))} docs", flush=True)

print("loading judge ...", flush=True)
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
leaked = 0
for k, p in enumerate(probes):
    i = p["id"]; ctx, rat = doc[i]["context"], doc[i]["rationale"]
    q, a = p["question"], p["answer"]
    if a.strip().lower() in rat.lower() or a.strip().lower() in ctx.lower():
        leaked += 1; continue
    # placebo = a different doc's rationale
    pid = doc_ids[(doc_ids.index(i) + 1) % len(doc_ids)]
    placebo_rat = doc[pid]["rationale"]
    b = nll_answer(f"{ctx}\nQuestion: {q}\nAnswer:", a)
    r = nll_answer(f"{ctx}\n{rat}\nQuestion: {q}\nAnswer:", a)
    pl = nll_answer(f"{ctx}\n{placebo_rat}\nQuestion: {q}\nAnswer:", a)
    if None in (b, r, pl):
        continue
    real.append(r - b); plac.append(pl - b)
    rows.append({"id": i, "answer": a, "real": round(r - b, 3), "placebo": round(pl - b, 3),
                 "r_minus_p": round((r - b) - (pl - b), 3)})

with open("data/probe_strict_results.jsonl", "w") as f:
    for x in rows:
        f.write(json.dumps(x) + "\n")
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"\n=== STRICT SPECIFIC-ANSWER PROBES (n={len(real)}, {leaked} leaked) ===")
print(f"real     delta: {mean(real):+.3f}  ({100*sum(d<0 for d in real)/len(real):.0f}% drop)")
print(f"placebo  delta: {mean(plac):+.3f}  ({100*sum(d<0 for d in plac)/len(plac):.0f}% drop)")
print(f"real − placebo (reasoning-specific): {mean(real)-mean(plac):+.3f}  <0 = reasoning helps beyond priming")
print(f"probes where real < placebo (reasoning wins): {sum(x['r_minus_p']<0 for x in rows)}/{len(rows)}")
print("DONE", flush=True)
