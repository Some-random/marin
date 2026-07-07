#!/usr/bin/env python3
"""Completeness check: does the generated reasoning make the doc's OWN continuation more predictable?
For each (context, reasoning, continuation): compute the judge model's NLL of the continuation given
(a) context alone vs (b) context + reasoning. If (b) < (a), the reasoning closed a gap. Also an
ablation: drop one reasoning line and check NLL rises (a minimality/completeness probe).

The document's continuation is the free 'answer key' — no symbolic verifier needed. The judge should
be a base-ish model standing in for the learner; a small model is fine for a first read."""
import json, argparse, math, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)                 # augmented jsonl
ap.add_argument("--judge", required=True)                # HF path of judge model
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--ablate", action="store_true")         # also do the drop-a-step probe
ap.add_argument("--dump", default=None)                  # write per-doc base/aug NLL for case analysis
args = ap.parse_args()
dump_f = open(args.dump, "w") if args.dump else None

tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def cont_nll(prefix, continuation):
    """mean NLL per continuation token under the judge, conditioned on prefix."""
    if not continuation.strip():
        return None
    pre = tok(prefix + "\n", return_tensors="pt").input_ids
    full = tok(prefix + "\n" + continuation, return_tensors="pt").input_ids.to(dev)
    n_pre = pre.shape[1]
    if full.shape[1] <= n_pre + 1 or full.shape[1] > 3500:
        return None
    with torch.no_grad():
        logits = model(full).logits[0]                    # [T, V]
    labels = full[0, n_pre:]                              # continuation tokens
    logp = torch.log_softmax(logits[n_pre - 1:-1].float(), dim=-1)
    nll = -logp[torch.arange(len(labels)), labels].mean().item()
    return nll

rows = [json.loads(l) for l in open(args.data)][: args.n * 2]
random.seed(0); random.shuffle(rows); rows = rows[: args.n]

helped = 0; total = 0; base_sum = 0.0; aug_sum = 0.0; abl_rise = 0; abl_total = 0
for r in rows:
    ctx, reas, cont = r["context"], r.get("reasoning", ""), r["continuation"]
    b = cont_nll(ctx, cont)
    a = cont_nll(ctx + "\nReasoning: " + reas, cont)
    if b is None or a is None:
        continue
    total += 1; base_sum += b; aug_sum += a
    if a < b:
        helped += 1
    if dump_f:
        dump_f.write(json.dumps({"id": r["id"], "base_nll": round(b, 4), "aug_nll": round(a, 4),
                                 "delta": round(a - b, 4), "context": ctx, "reasoning": reas,
                                 "continuation": cont}) + "\n")
        dump_f.flush()
    if args.ablate and reas.count("\n") >= 2:
        lines = [x for x in reas.split("\n") if x.strip()]
        drop = random.randrange(len(lines))
        abl = "\n".join(lines[:drop] + lines[drop + 1:])
        a2 = cont_nll(ctx + "\nReasoning: " + abl, cont)
        if a2 is not None:
            abl_total += 1
            if a2 > a:          # removing a step raised NLL → that step was load-bearing (complete)
                abl_rise += 1

print(f"judge: {args.judge.split('/')[-3] if '/' in args.judge else args.judge}")
print(f"scored: {total} docs")
print(f"mean continuation NLL — context only : {base_sum/total:.4f}  (ppl {math.exp(base_sum/total):.2f})")
print(f"mean continuation NLL — +reasoning   : {aug_sum/total:.4f}  (ppl {math.exp(aug_sum/total):.2f})")
print(f"mean NLL reduction from reasoning    : {(base_sum-aug_sum)/total:+.4f} nats/token")
print(f"fraction of docs where reasoning helped (lower NLL): {helped/total:.1%}")
if args.ablate and abl_total:
    print(f"ablation (drop 1 step): NLL rose in {abl_rise/abl_total:.1%} of {abl_total} docs "
          f"(higher = steps are load-bearing / chain is minimal-complete)")
