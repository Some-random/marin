#!/usr/bin/env python3
"""Controlled completeness check (v2) — separates rationale CONTENT from the format/flow confound,
and probes memorization. Judge = a local HF model (default Qwen2.5-32B-Instruct).

For each doc, teacher-forcing NLL of the HELD-OUT continuation under the judge, given:
  base    : context only
  +claude : context + Claude rationale
  +qwen   : context + Qwen rationale
  +placebo: context + ANOTHER doc's Claude rationale  (format-matched, but irrelevant content)
  noctx   : a minimal prefix only  (memorization / intrinsic-predictability probe)

Signals (nats/token, lower NLL = continuation better predicted):
  content  = (+claude) - (+placebo)   # <0 => the REAL rationale helps BEYOND merely having a list here
  format   = (+placebo) - base        # >0 => inserting ANY list raises surprise (concern B, the confound)
  teacher  = (+claude) - (+qwen)      # <0 => Claude's rationale beats Qwen's at the SAME format
  memorize = noctx                    # low => continuation predictable without its context (concern A)

The placebo cancels the format penalty (both +arms pay it), so `content` isolates relevance; comparing
`content` to `format` tells us whether last night's negative was the format confound or a real effect.
"""
import json, argparse, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/dclm_aug_claude.jsonl")
ap.add_argument("--judge", default="Qwen/Qwen2.5-32B-Instruct")
ap.add_argument("--out", default="data/completeness_v2_results.jsonl")
ap.add_argument("--style", choices=["bracketed", "natural"], default="bracketed",
                help="bracketed: '\\nReasoning:\\n<list>'  |  natural: '\\n\\n<prose>' (in-distribution)")
args = ap.parse_args()

print(f"loading judge {args.judge} ...", flush=True)
tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def cont_nll(prefix, continuation):
    """mean per-token NLL of `continuation` under the judge, conditioned on `prefix`."""
    if not continuation.strip():
        return None
    pre = tok(prefix + "\n", return_tensors="pt").input_ids
    full = tok(prefix + "\n" + continuation, return_tensors="pt").input_ids.to(dev)
    n_pre = pre.shape[1]
    if full.shape[1] <= n_pre + 1 or full.shape[1] > 3500:
        return None
    with torch.no_grad():
        logits = model(full).logits[0]
    labels = full[0, n_pre:]
    logp = torch.log_softmax(logits[n_pre - 1:-1].float(), dim=-1)
    return -logp[torch.arange(len(labels)), labels].mean().item()

SEP = {"bracketed": "\nReasoning:\n", "natural": "\n\n"}[args.style]
def with_reasoning(ctx, reas):
    return ctx + SEP + reas

rows = [json.loads(l) for l in open(args.data)]
out = open(args.out, "w")
agg = {k: [] for k in ["base", "claude", "qwen", "placebo", "noctx", "content", "format", "teacher"]}
print(f"judge={args.judge}  docs={len(rows)}\n", flush=True)

for i, r in enumerate(rows):
    ctx, cont = r["context"], r["continuation"]
    claude_r, qwen_r = r["reasoning"], r.get("qwen_reasoning", "")
    placebo_r = rows[(i + 1) % len(rows)]["reasoning"]           # another doc's Claude rationale
    base    = cont_nll(ctx, cont)
    claude  = cont_nll(with_reasoning(ctx, claude_r), cont)
    qwen    = cont_nll(with_reasoning(ctx, qwen_r), cont)
    placebo = cont_nll(with_reasoning(ctx, placebo_r), cont)
    noctx   = cont_nll("Passage:", cont)                         # memorization / intrinsic probe
    content = claude - placebo
    fmt     = placebo - base
    teacher = claude - qwen
    rec = {"id": r["id"], "base": base, "claude": claude, "qwen": qwen, "placebo": placebo,
           "noctx": noctx, "content": content, "format": fmt, "teacher": teacher}
    out.write(json.dumps(rec) + "\n"); out.flush()
    for k, v in rec.items():
        if k != "id":
            agg[k].append(v)
    print(f"doc {r['id']:5d}: base {base:.2f} | +claude {claude:.2f} | +qwen {qwen:.2f} | "
          f"+placebo {placebo:.2f} | noctx {noctx:.2f}  ||  content(cl−pl) {content:+.2f}  "
          f"format(pl−base) {fmt:+.2f}  teacher(cl−qw) {teacher:+.2f}", flush=True)

mean = lambda xs: sum(xs) / len(xs)
print("\n=== MEANS (nats/token; lower = continuation better predicted) ===")
for k in ["base", "claude", "qwen", "placebo", "noctx"]:
    print(f"  {k:9s} {mean(agg[k]):.3f}  (ppl {math.exp(mean(agg[k])):.2f})")
print("\n=== SIGNALS (n={} docs — illustrative) ===".format(len(rows)))
print(f"  content  (+claude − +placebo): {mean(agg['content']):+.3f}  <0 = real rationale beats a format-matched irrelevant one")
print(f"  format   (+placebo − base)   : {mean(agg['format']):+.3f}  >0 = inserting ANY list raises surprise (the confound)")
print(f"  teacher  (+claude − +qwen)   : {mean(agg['teacher']):+.3f}  <0 = Claude rationale beats Qwen (same format)")
print(f"  memorize (noctx mean)        : {mean(agg['noctx']):.3f}  low (<~1.5) => continuations predictable w/o context")
out.close()
print("\nDONE ->", args.out, flush=True)
