#!/usr/bin/env python3
"""Judge-calibration comparison: run the probe-target + strict-probe measurement under several BASE judges
of increasing quality/size, and compare the GAINS (real−base, placebo−base, real−placebo). If the
reasoning-specific gain (real−placebo) is consistent across well-calibrated judges it is robust; if it only
appears in the small/noisy 1.4B, it is judge-dependent. Loads one judge at a time (frees GPU between)."""
import json, glob, gc, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGES = [
    ("DCLM-1.4B-base", "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"),
    ("Llama-2-7B-base", "NousResearch/Llama-2-7b-hf"),
    ("Qwen2.5-72B-base", "Qwen/Qwen2.5-72B"),
]

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

doc = {}
for f in sorted(glob.glob("data/probe_batch_*.jsonl")):
    for r in load(f):
        doc[r["id"]] = r
doc_ids = list(doc.keys())
pt = [r for f in sorted(glob.glob("data/probe_out_*.jsonl")) for r in load(f) if r["id"] in doc]
st = [r for f in sorted(glob.glob("data/probe_strict_*.jsonl")) for r in load(f)
      if r["id"] in doc and "question" in r and "answer" in r]

def measure(model, tok, dev, probes):
    def nll(prefix, answer):
        pre = tok(prefix, return_tensors="pt").input_ids
        full = tok(prefix + " " + answer.strip(), return_tensors="pt").input_ids.to(dev)
        n = pre.shape[1]
        if full.shape[1] <= n or full.shape[1] > 3500:
            return None
        with torch.no_grad():
            lg = model(full).logits[0]
        lab = full[0, n:]
        lp = torch.log_softmax(lg[n - 1:-1].float(), dim=-1)
        return -lp[torch.arange(len(lab)), lab].mean().item()
    real, plac = [], []
    for k, p in enumerate(probes):
        i = p["id"]; ctx, rat = doc[i]["context"], doc[i]["rationale"]
        q, a = p["question"], p["answer"]
        if a.strip().lower() in rat.lower() or a.strip().lower() in ctx.lower():
            continue
        pl_rat = doc[doc_ids[(doc_ids.index(i) + 1) % len(doc_ids)]]["rationale"]
        b = nll(f"{ctx}\nQuestion: {q}\nAnswer:", a)
        r = nll(f"{ctx}\n{rat}\nQuestion: {q}\nAnswer:", a)
        pl = nll(f"{ctx}\n{pl_rat}\nQuestion: {q}\nAnswer:", a)
        if None in (b, r, pl):
            continue
        real.append(r - b); plac.append(pl - b)
    m = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return len(real), m(real), m(plac), m(real) - m(plac)

results = []
for name, path in JUDGES:
    print(f"\n=== loading {name} ({path}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = model.device
    n_pt, r_pt, p_pt, rp_pt = measure(model, tok, dev, pt)
    n_st, r_st, p_st, rp_st = measure(model, tok, dev, st)
    results.append((name, n_pt, r_pt, p_pt, rp_pt, n_st, r_st, p_st, rp_st))
    print(f"  {name}: probe-target real {r_pt:+.3f} placebo {p_pt:+.3f} real-placebo {rp_pt:+.3f} | strict real-placebo {rp_st:+.3f}", flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

print("\n\n=== JUDGE CALIBRATION COMPARISON ===")
print(f"{'judge':18s} | probe-target: real / placebo / real−placebo | strict: real−placebo")
for name, n_pt, r_pt, p_pt, rp_pt, n_st, r_st, p_st, rp_st in results:
    print(f"{name:18s} | {r_pt:+.3f} / {p_pt:+.3f} / {rp_pt:+.3f}   | {rp_st:+.3f}")
print("\nreal−placebo<0 = reasoning-specific gain survives on that judge. Consistent across judges = robust.")
print("DONE", flush=True)
