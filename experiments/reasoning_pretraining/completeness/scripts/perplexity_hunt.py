#!/usr/bin/env python3
"""Perplexity-drop hunt: find a config where inserting the rationale LOWERS continuation NLL under a
BASE learner judge (our DCLM-trained 1.4B) — the model we'd actually augment, and a base LM that won't
over-penalize an off-distribution format the way an instruct model does.

Sweeps insertion-style x target x docs. Per config reports: mean delta (rat - base), the fraction of
docs where the rationale drops perplexity, and the mean over that drop-subset. delta<0 = improvement.
"""
import json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dclm_aug_qwen32b.jsonl")
ap.add_argument("--judge", default="/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--out", default="data/hunt_results.jsonl")
args = ap.parse_args()

print(f"loading judge {args.judge} ...", flush=True)
tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def nll(prefix, cont):
    if not cont.strip():
        return None
    pre = tok(prefix + "\n", return_tensors="pt").input_ids
    full = tok(prefix + "\n" + cont, return_tensors="pt").input_ids.to(dev)
    n_pre = pre.shape[1]
    if full.shape[1] <= n_pre + 1 or full.shape[1] > 3500:
        return None
    with torch.no_grad():
        logits = model(full).logits[0]
    labels = full[0, n_pre:]
    logp = torch.log_softmax(logits[n_pre - 1:-1].float(), dim=-1)
    return -logp[torch.arange(len(labels)), labels].mean().item()

def insert(ctx, reas, style):
    return {"bracketed": ctx + "\nReasoning:\n" + reas,
            "natural":   ctx + "\n\n" + reas,
            "nosep":     ctx + " " + reas}[style]

def first_sentence(t):
    p = t.find(". ")
    return t[:p + 1] if p != -1 else t

STYLES = ["bracketed", "natural", "nosep"]
TARGETS = ["full", "first"]
rows = [json.loads(l) for l in open(args.data)][:args.n]
agg = {(s, t): [] for s in STYLES for t in TARGETS}
noctx_list = []
out = open(args.out, "w")
n_scored = 0
for r in rows:
    ctx, cont, reas = r["context"], r["continuation"], r.get("reasoning", "")
    if not reas.strip() or not cont.strip():
        continue
    tgt = {"full": cont, "first": first_sentence(cont)}
    base = {t: nll(ctx, tgt[t]) for t in TARGETS}
    if any(base[t] is None for t in TARGETS):
        continue
    rec = {"id": r["id"], "base_full": round(base["full"], 4), "base_first": round(base["first"], 4)}
    ok = True
    for s in STYLES:
        pref = insert(ctx, reas, s)
        for t in TARGETS:
            v = nll(pref, tgt[t])
            if v is None:
                ok = False; break
            agg[(s, t)].append(v - base[t])
            rec[f"{s}_{t}"] = round(v - base[t], 4)
        if not ok:
            break
    if not ok:
        continue
    nx = nll("Passage:", cont)
    if nx is not None:
        noctx_list.append(nx)
    out.write(json.dumps(rec) + "\n"); out.flush()
    n_scored += 1
    if n_scored % 50 == 0:
        print(f"  scored {n_scored} ...", flush=True)
out.close()

mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
print(f"\n=== HUNT RESULTS  judge={args.judge.rstrip('/').split('/')[-1]}  n_scored={n_scored} ===")
print(f"{'style':10s} {'target':6s} {'mean_delta':>11s} {'%drop':>7s} {'drop_mean':>10s}   (delta<0 = LOWERS ppl)")
best = None
for s in STYLES:
    for t in TARGETS:
        ds = agg[(s, t)]
        drops = [d for d in ds if d < 0]
        m = mean(ds); frac = (len(drops) / len(ds)) if ds else 0
        print(f"{s:10s} {t:6s} {m:>+11.3f} {100*frac:>6.1f}% {mean(drops):>+10.3f}")
        if best is None or m < best[0]:
            best = (m, s, t, frac)
print(f"\nBEST config: {best[1]}+{best[2]}  mean_delta {best[0]:+.3f}  ({100*best[3]:.0f}% of docs drop)")
print(f"memorization: noctx mean {mean(noctx_list):.3f}  (low <~1.5 => continuations memorized)")
print("DONE ->", args.out, flush=True)
