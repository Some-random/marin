#!/usr/bin/env python3
"""Generate implicit-reasoning augmentations for real DCLM docs (HF transformers, bulletproof).
For each doc: split into (context, continuation); ask a teacher to write the concise implicit
reasoning behind the context; save {context, reasoning, continuation}. This is the raw material
for completeness-augmented pretraining. Perplexity/completeness check is a separate script."""
import json, argparse, random, time, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--max_new", type=int, default=220)
ap.add_argument("--src", default="outputs/raw/dclm_5000docs.jsonl")
ap.add_argument("--out", default="experiments/data_efficiency/reasoning_completeness/data/dclm_augmented.jsonl")
args = ap.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# --- load + split DCLM docs ---
docs = []
with open(args.src) as f:
    for line in f:
        t = (json.loads(line).get("text") or "").strip()
        if 400 < len(t) < 2000 and t.count("\n") < 12:
            docs.append(t)
        if len(docs) >= args.n * 5:  # early break for large sources (dclm_1500m is 7GB)
            break
random.seed(0); random.shuffle(docs); docs = docs[:args.n]

def split(t):
    cut = int(len(t) * 0.65)
    p = t.find(". ", cut)
    p = p + 1 if p != -1 else len(t)
    return t[:p].strip(), t[p:].strip()

pairs = [split(t) for t in docs]

# --- teacher ---
print(f"loading {args.model} ...", flush=True)
tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

SYS = "You expose the hidden reasoning that a piece of text takes for granted."
def build(ctx):
    user = (
        f'Passage (may be truncated):\n\n"""{ctx}"""\n\n'
        "This passage leaves most of its reasoning implicit. Reconstruct the COMPLETE chain of reasoning "
        "that connects its setup to its point: make every load-bearing inferential step explicit so there "
        "are NO gaps between steps, but do NOT state facts any adult already knows (skip 'objects are "
        "physical', 'people can carry things', etc.). Write a short NUMBERED chain where each step follows "
        "from the previous steps plus common knowledge. 3 to 6 steps. Output only the numbered steps."
    )
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )

prompts = [build(c) for c, _ in pairs]
out_f = open(args.out, "w")
results = []
t0 = time.time()
for i in range(0, len(prompts), args.batch):
    batch = prompts[i:i + args.batch]
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1800).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                             temperature=0.7, top_p=0.9, pad_token_id=tok.pad_token_id)
    for j, g in enumerate(gen):
        text = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        idx = i + j
        rec = {"id": idx, "context": pairs[idx][0], "continuation": pairs[idx][1], "reasoning": text}
        out_f.write(json.dumps(rec) + "\n"); out_f.flush()
        results.append(rec)
    rate = (i + len(batch)) / max(1e-9, time.time() - t0)
    print(f"[{i + len(batch)}/{len(prompts)}] {rate:.1f} docs/s", flush=True)
out_f.close()

# --- human-readable sample for morning review ---
with open(args.out.replace(".jsonl", "_readable.md"), "w") as m:
    m.write(f"# DCLM implicit-reasoning augmentation\nmodel: {args.model} · n={len(results)}\n\n")
    for r in results[:40]:
        m.write(f"## doc {r['id']}\n**Context:** {r['context'][:700]}\n\n"
                f"**→ Generated reasoning:** {r['reasoning']}\n\n"
                f"**Actual continuation:** {r['continuation'][:300]}\n\n---\n\n")
print(f"DONE {len(results)} docs in {time.time()-t0:.0f}s -> {args.out}", flush=True)
