#!/usr/bin/env python3
"""R vs N test: does adding a rationale lower continuation perplexity on reasoning-dependent (R) docs but
not on non-reasoning (N) docs? Merges the agents' out_*.jsonl (id,label,reasoning=Claude, context-only),
joins context+continuation, and measures `rationale − base` under the DCLM 1.4B base judge, split by label.
Compares the Claude rationale against the original Qwen rationale on the same docs."""
import json, glob, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf")
ap.add_argument("--out", default="data/rn_results.jsonl")
args = ap.parse_args()

# merge agent outputs: {id: {label, claude_reasoning}}
lab = {}
for f in sorted(glob.glob("data/hunt_batches/out_*.jsonl")):
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            lab[r["id"]] = {"label": r["label"].strip().upper()[:1], "claude": r["reasoning"]}
        except Exception:
            continue
src = {json.loads(l)["id"]: json.loads(l) for l in open("data/dclm_aug_qwen32b.jsonl")}
docs = [{"id": i, "label": lab[i]["label"], "claude": lab[i]["claude"],
         "qwen": src[i].get("reasoning", ""), "context": src[i]["context"],
         "continuation": src[i]["continuation"]}
        for i in lab if i in src]
print(f"merged {len(docs)} labeled docs "
      f"(R={sum(d['label']=='R' for d in docs)}, N={sum(d['label']=='N' for d in docs)})", flush=True)

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

def brk(ctx, reas):
    return ctx + "\nReasoning:\n" + reas

out = open(args.out, "w")
res = {"R": {"claude": [], "qwen": []}, "N": {"claude": [], "qwen": []}}
for d in docs:
    if d["label"] not in ("R", "N"):
        continue
    base = nll(d["context"], d["continuation"])
    if base is None:
        continue
    dc = nll(brk(d["context"], d["claude"]), d["continuation"])
    dq = nll(brk(d["context"], d["qwen"]), d["continuation"]) if d["qwen"].strip() else None
    if dc is None:
        continue
    row = {"id": d["id"], "label": d["label"], "base": round(base, 4),
           "claude_delta": round(dc - base, 4)}
    res[d["label"]]["claude"].append(dc - base)
    if dq is not None:
        row["qwen_delta"] = round(dq - base, 4)
        res[d["label"]]["qwen"].append(dq - base)
    out.write(json.dumps(row) + "\n"); out.flush()
out.close()

mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
frac_drop = lambda xs: (sum(x < 0 for x in xs) / len(xs)) if xs else float("nan")
print("\n=== R vs N — rationale delta (nats/token; <0 = LOWERS perplexity) ===")
print(f"{'label':6s} {'teacher':7s} {'n':>4s} {'mean_delta':>11s} {'%drop':>7s}")
for L in ("R", "N"):
    for T in ("claude", "qwen"):
        xs = res[L][T]
        print(f"{L:6s} {T:7s} {len(xs):>4d} {mean(xs):>+11.3f} {100*frac_drop(xs):>6.1f}%")
print("\nKEY: if R/claude mean_delta < 0 and N/claude > 0 → rationales lower perplexity exactly on "
      "reasoning-dependent docs. If Claude beats Qwen on R, teacher quality converts to a bigger drop.")
print("DONE ->", args.out, flush=True)
