#!/usr/bin/env python3
"""Winogrande as a reasoning-perplexity probe: does prepending a rationale help the judge pick the correct
option? Partial scoring — for each option, score NLL of the shared post-blank suffix given
[rationale +] sentence-up-to-blank + option; the model picks the option with lower suffix-NLL.
Conditions: base / principle / full / complete / placebo (another example's complete rationale).
Reports accuracy (% correct) and margin (NLL_wrong - NLL_correct; +=favors correct) per condition.
Each judge uses ITS OWN tokenizer. Usage: winogrande_score.py <judge_dir> <out.jsonl>"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = sys.argv[1] if len(sys.argv) > 1 else "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/winogrande_results.jsonl"

data = [json.loads(l) for l in open("data/winogrande_200.jsonl")]
rat = {json.loads(l)["idx"]: json.loads(l) for l in open("data/winogrande_rationales.jsonl")}
ids = [x["idx"] for x in data]

tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def nll(prefix, target):
    pre = tok(prefix, return_tensors="pt").input_ids
    full = tok(prefix + target, return_tensors="pt").input_ids.to(dev)
    n = pre.shape[1]
    if full.shape[1] <= n or full.shape[1] > 3500:
        return None
    with torch.no_grad():
        lg = model(full).logits[0]
    lab = full[0, n:]
    lp = torch.log_softmax(lg[n - 1:-1].float(), dim=-1)
    return -lp[torch.arange(len(lab)), lab].mean().item()

def score(x, rationale):
    b = x["sentence"].index("_")
    pre_txt, suf = x["sentence"][:b], x["sentence"][b + 1:]
    pfx = (rationale.strip() + "\n") if rationale else ""
    n1 = nll(pfx + pre_txt + x["option1"], suf)
    n2 = nll(pfx + pre_txt + x["option2"], suf)
    if n1 is None or n2 is None:
        return None
    pick = "1" if n1 < n2 else "2"
    ncorr = n1 if x["answer"] == "1" else n2
    nwrong = n2 if x["answer"] == "1" else n1
    return {"correct": pick == x["answer"], "margin": nwrong - ncorr,
            "nll_correct": ncorr, "nll_wrong": nwrong}

CONDS = ["base", "principle", "full", "complete", "placebo"]
results = []
with open(OUT, "w") as fout:
    for k, x in enumerate(data):
        r = rat.get(x["idx"])
        if not r:
            continue
        placebo = rat[ids[(k + 1) % len(ids)]]["complete"]  # rotate: an unrelated example's complete rationale
        row = {"idx": x["idx"]}
        for c in CONDS:
            rationale = None if c == "base" else (placebo if c == "placebo" else r.get(c))
            s = score(x, rationale)
            if s:
                row[c] = s
        results.append(row)
        fout.write(json.dumps(row) + "\n")

import math
print(f"\n=== {JUDGE.split('/')[-1]}  (n={len(results)}) ===")
print("NLL(correct cont) = perplexity of the TRUE continuation (suffix under the correct option), teacher-forced.")
print("ppl = exp(that). ΔNLL vs base < 0 => the rationale LOWERED the true continuation's perplexity.")
print("margin = NLL(wrong option) - NLL(correct option); + => model favors correct (drives accuracy).\n")
base_acc = sum(r['base']['correct'] for r in results if 'base' in r) / len(results)
base_nc = sum(r['base']['nll_correct'] for r in results if 'base' in r) / len(results)
print(f"{'condition':>10s} | {'acc':>6s} | {'Δacc':>7s} | {'NLL(corr)':>9s} | {'ppl':>6s} | {'ΔNLL':>8s} | {'margin':>7s}")
print("-" * 74)
for c in CONDS:
    rr = [r[c] for r in results if c in r]
    acc = sum(x['correct'] for x in rr) / len(rr)
    nc = sum(x['nll_correct'] for x in rr) / len(rr)
    marg = sum(x['margin'] for x in rr) / len(rr)
    dacc = "" if c == "base" else f"{acc-base_acc:+.3f}"
    dnc = "" if c == "base" else f"{nc-base_nc:+.4f}"
    print(f"{c:>10s} | {acc:6.3f} | {dacc:>7s} | {nc:9.4f} | {math.exp(nc):6.2f} | {dnc:>8s} | {marg:+7.3f}")
