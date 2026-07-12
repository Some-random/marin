#!/usr/bin/env python3
"""Reverse-filter: score base NLL of the target (first sentence of the continuation) given context, for DCLM
docs, with NO rationale. High NLL = headroom. Supports sharding over the raw file's LINE INDEX
(--shard/--nshards) and an optional --ids-file (score only docs whose line index is listed). id = raw line
index, so it maps back to the pool and is unique across shards.
Usage: score_uncertainty.py --judge P --src F --out O [--n N] [--shard i --nshards K] [--ids-file F]"""
import json, argparse, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--judge", required=True)
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=0)        # 0 = all docs in this shard
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--ids-file", default=None)        # only score these line indices (overrides sharding)
args = ap.parse_args()

ids_filter = set(int(x) for x in open(args.ids_file) if x.strip()) if args.ids_file else None

def split(t):
    cut = int(len(t) * 0.65)
    p = t.find(". ", cut); p = p + 1 if p != -1 else len(t)
    ctx, cont = t[:p].strip(), t[p:].strip()
    q = cont.find(". "); target = (cont[:q + 1] if q != -1 else cont).strip()
    return ctx, target

pairs = []  # (line_index, ctx, target)
for li, line in enumerate(open(args.src)):
    if ids_filter is not None:
        if li not in ids_filter:
            continue
    elif li % args.nshards != args.shard:
        continue
    t = (json.loads(line).get("text") or "").strip()
    if not (400 < len(t) < 2000 and t.count("\n") < 12):
        continue
    ctx, tgt = split(t)
    if len(ctx) > 150 and 15 < len(tgt) < 400:
        pairs.append((li, ctx, tgt))
    if args.n and len(pairs) >= args.n:
        break

tok = AutoTokenizer.from_pretrained(args.judge)
model = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def nll(ctx, target):
    pre = tok(ctx + "\n", return_tensors="pt").input_ids
    full = tok(ctx + "\n" + target, return_tensors="pt").input_ids.to(dev)
    n = pre.shape[1]
    if full.shape[1] <= n or full.shape[1] > 2048:
        return None
    with torch.no_grad():
        lg = model(full).logits[0]
    lab = full[0, n:]
    lp = torch.log_softmax(lg[n - 1:-1].float(), dim=-1)
    return -lp[torch.arange(len(lab)), lab].mean().item()

t0 = time.time(); out = open(args.out, "w"); done = 0
for li, ctx, tgt in pairs:
    v = nll(ctx, tgt)
    if v is None:
        continue
    out.write(json.dumps({"id": li, "base_nll": round(v, 4), "target": tgt, "ctx_tail": ctx[-120:]}) + "\n")
    out.flush(); done += 1
    if done % 2000 == 0:
        print(f"  {done}/{len(pairs)} @ {done/(time.time()-t0):.1f} docs/s", flush=True)
out.close()
print(f"[DONE] {args.judge.split('/')[-1]} shard {args.shard}/{args.nshards}: {done} docs in {time.time()-t0:.0f}s", flush=True)
