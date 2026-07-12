#!/usr/bin/env python3
"""Per-token perplexity of the Winogrande scored span WITH THE BLANK INCLUDED — i.e. the scored span is
`option + suffix` ("Maria always got the easier cases."), given the pre-blank stem, under base / +principle /
+full / +complete. Shows the full context, the three rationales, and every scored token's NLL in each
condition. The option (blank) token(s) come first, then the suffix. Judge = DCLM-1.4B base."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE = "/fsx/users/dongweij/marin/checkpoints/1ep_dclm_step14672_hf"
OUT = "docs/WINOGRANDE_PERTOKEN.md"
idxs = [int(x) for x in sys.argv[1:]] or [0, 3, 4, 2]

data = {json.loads(l)["idx"]: json.loads(l) for l in open("data/winogrande_200.jsonl")}
rat = {json.loads(l)["idx"]: json.loads(l) for l in open("data/winogrande_rationales.jsonl")}
tok = AutoTokenizer.from_pretrained(JUDGE)
model = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.bfloat16, device_map="auto").eval()
dev = model.device

def per_token(prefix, target):
    pre = tok(prefix, return_tensors="pt").input_ids
    full = tok(prefix + target, return_tensors="pt").input_ids.to(dev)
    n = pre.shape[1]
    with torch.no_grad():
        lg = model(full).logits[0]
    lab = full[0, n:]
    lp = torch.log_softmax(lg[n - 1:-1].float(), dim=-1)
    nlls = [-lp[i, lab[i]].item() for i in range(len(lab))]
    toks = [tok.decode([t]) for t in lab]
    return toks, nlls

def mean_nll(prefix, target):
    _, n = per_token(prefix, target)
    return sum(n) / len(n)

CONDS = ["base", "principle", "full", "complete"]
L = ["# Winogrande — per-token perplexity of the scored span (BLANK INCLUDED)\n",
     "Scored span = **`option + suffix`** (e.g. `Maria always got the easier cases.`), given the pre-blank stem —",
     "so the blank/answer token itself is now part of what's scored, followed by the rest of the sentence. Each",
     "row is one token of that span with its NLL (nats/token) under `base` (no rationale) and each rationale",
     "prepended, all under the **correct** option. The `‹opt›` rows are the blank token(s). Judge = DCLM-1.4B.",
     "The model picks whichever option gives the lower mean span-NLL; margin vs the wrong option shown too.\n"]

for IX in idxs:
    d = data[IX]; r = rat[IX]
    b = d["sentence"].index("_")
    pre_txt, suf = d["sentence"][:b], d["sentence"][b + 1:]
    stem = pre_txt.rstrip(); gap = pre_txt[len(stem):]  # natural spacing between stem and option
    correct, wrong = d["correct"], (d["option2"] if d["answer"] == "1" else d["option1"])
    span = gap + correct + suf  # the scored span: option + suffix, naturally spaced
    n_opt = len(tok(stem + gap + correct).input_ids) - len(tok(stem).input_ids)  # leading tokens that are the option/blank (BOS cancels)
    L += [f"\n---\n\n## example idx {IX}\n",
          f"**Sentence:** {d['sentence'].replace('_', '▁')}",
          f"**option1:** `{d['option1']}`  ·  **option2:** `{d['option2']}`  ·  **answer:** {d['answer']} → **`{correct}`** (correct)\n",
          f"**Scored span (blank + suffix, correct option):** `{(correct + suf).strip()}`\n",
          f"**principle:** {r['principle']}\n",
          f"**full:** {r['full']}\n",
          f"**complete:** {r['complete']}\n",
          "Per-token NLL of the scored span (option first, then suffix), under the **correct** option:\n",
          "| # | token | base | +principle | +full | +complete | Δ(compl−base) |",
          "|---:|---|---:|---:|---:|---:|---:|"]
    cols = {}
    for c in CONDS:
        rr = None if c == "base" else r[c]
        pfx = (rr.strip() + "\n") if rr else ""
        toks, nl = per_token(pfx + stem, span)
        cols[c] = nl; cols[c + "_toks"] = toks
    toks = cols["base_toks"]
    for i, t in enumerate(toks):
        dc = cols["complete"][i] - cols["base"][i]
        mark = " ⟵blank" if i < n_opt else ""
        L.append(f"| {i} | `{t}`{mark} | {cols['base'][i]:.3f} | {cols['principle'][i]:.3f} | {cols['full'][i]:.3f} | {cols['complete'][i]:.3f} | {dc:+.3f} |")
    mrow = "| | **MEAN** |"
    for c in CONDS:
        mrow += f" **{sum(cols[c])/len(cols[c]):.3f}** |"
    dmean = sum(cols['complete'])/len(cols['complete']) - sum(cols['base'])/len(cols['base'])
    L.append(mrow + f" **{dmean:+.3f}** |")
    L.append("\n**Perplexity (correct span) & margin vs the wrong option (`" + wrong + "`):**\n")
    L.append("| condition | ppl(correct) | mean NLL correct | mean NLL wrong | margin (wrong−correct) | model picks |")
    L.append("|---|---:|---:|---:|---:|:---:|")
    for c in CONDS:
        rr = None if c == "base" else r[c]
        pfx = (rr.strip() + "\n") if rr else ""
        nc = sum(cols[c]) / len(cols[c])
        nw = mean_nll(pfx + stem, gap + wrong + suf)
        pick = "✓ correct" if nc < nw else "✗ wrong"
        L.append(f"| {c} | {torch.tensor(nc).exp():.2f} | {nc:.3f} | {nw:.3f} | {nw-nc:+.3f} | {pick} |")

open(OUT, "w").write("\n".join(L) + "\n")
print(f"wrote {OUT} ({len(idxs)} examples, blank-included scoring)")
