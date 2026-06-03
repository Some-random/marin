"""Arithmetic decomposition probe (Path B, Phase 1).

Probes 5 levels of arithmetic difficulty on a 1.4B-class model. Tests whether
the model has any arithmetic capability at all, independent of word-problem
parsing (which is what GSM8K mostly measures).

Levels:
  A1 — single-digit addition (a, b ∈ [0, 9])
  A2 — two-digit addition without carry (a + b ≤ 99)
  A3 — two-digit addition with carry (a + b ≤ 199)
  A4 — single-digit multiplication (a, b ∈ [2, 9], uses '*' for tokenizer-portability)
  A5 — two-digit subtraction (a ≥ b, both ∈ [10, 99])

100 problems per level. Same seed=0 across models so the same problem set is
seen by every checkpoint.

Format: 0-shot, single-line.
  prompt:  "1 + 2 = "
  target:  "3"

Scoring: greedy generate up to max_new_tokens=4, strip whitespace,
compare integer prefix to ground truth.

Usage:
  .venv/bin/python -m experiments.data_efficiency.probes_arithmetic \\
    --model /fsx/users/dongweij/marin/checkpoints/1ep_dclm_final_hf \\
    --output /tmp/arith_probe_A5final.json

  # Then aggregate:
  .venv/bin/python -m experiments.data_efficiency.probes_arithmetic \\
    --summary /tmp/arith_probe_*.json
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

SEED = 0
PROBLEMS_PER_LEVEL = 100
MAX_NEW_TOKENS = 4

LEVELS = ("A1", "A2", "A3", "A4", "A5")
LEVEL_LABELS = {
    "A1": "single-digit addition (a, b ∈ [0, 9])",
    "A2": "two-digit addition without carry (a + b ≤ 99, no carry between cols)",
    "A3": "two-digit addition with carry (a + b ≤ 199, requires carry)",
    "A4": "single-digit multiplication (a, b ∈ [2, 9])",
    "A5": "two-digit subtraction (a ≥ b ∈ [10, 99])",
}


@dataclass
class Problem:
    level: str
    prompt: str
    answer: int


def _has_carry(a: int, b: int) -> bool:
    sa, sb = str(a), str(b)
    pad = max(len(sa), len(sb))
    sa, sb = sa.zfill(pad), sb.zfill(pad)
    carry = 0
    for ca, cb in zip(reversed(sa), reversed(sb)):
        total = int(ca) + int(cb) + carry
        if total >= 10:
            return True
        carry = 0
    return False


def generate_problems() -> dict[str, list[Problem]]:
    rng = np.random.default_rng(SEED)
    out: dict[str, list[Problem]] = {lvl: [] for lvl in LEVELS}

    # A1: single-digit add
    pairs = [(int(a), int(b)) for a in range(10) for b in range(10)]
    rng.shuffle(pairs)
    for a, b in pairs[:PROBLEMS_PER_LEVEL]:
        out["A1"].append(Problem("A1", f"{a} + {b} = ", a + b))

    # A2: two-digit no-carry
    seen: set[tuple[int, int]] = set()
    while len(out["A2"]) < PROBLEMS_PER_LEVEL:
        a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
        if a + b > 99 or _has_carry(a, b) or (a, b) in seen:
            continue
        seen.add((a, b))
        out["A2"].append(Problem("A2", f"{a} + {b} = ", a + b))

    # A3: two-digit with-carry
    seen.clear()
    while len(out["A3"]) < PROBLEMS_PER_LEVEL:
        a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
        if not _has_carry(a, b) or (a, b) in seen:
            continue
        seen.add((a, b))
        out["A3"].append(Problem("A3", f"{a} + {b} = ", a + b))

    # A4: single-digit mult (skip a=1 or b=1 — trivial)
    pairs = [(int(a), int(b)) for a in range(2, 10) for b in range(2, 10)]
    rng.shuffle(pairs)
    # Need 100, only 64 unique — repeat with shuffles
    expanded: list[tuple[int, int]] = []
    while len(expanded) < PROBLEMS_PER_LEVEL:
        rng.shuffle(pairs)
        expanded.extend(pairs)
    for a, b in expanded[:PROBLEMS_PER_LEVEL]:
        out["A4"].append(Problem("A4", f"{a} * {b} = ", a * b))

    # A5: two-digit subtraction
    seen.clear()
    while len(out["A5"]) < PROBLEMS_PER_LEVEL:
        a, b = int(rng.integers(10, 100)), int(rng.integers(10, 100))
        if a < b or (a, b) in seen:
            continue
        seen.add((a, b))
        out["A5"].append(Problem("A5", f"{a} - {b} = ", a - b))

    return out


_NUM_RE = re.compile(r"-?\d+")


def parse_first_int(text: str) -> int | None:
    m = _NUM_RE.search(text)
    if m is None:
        return None
    return int(m.group(0))


def score(generation: str, target: int) -> bool:
    """Generation is correct iff the first integer in it equals target."""
    pred = parse_first_int(generation)
    return pred is not None and pred == target


def run_one_model(model_path: str, output_path: str) -> None:
    logger.info("loading tokenizer + model from %s", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    probs = generate_problems()
    n_per_level = sum(len(probs[l]) for l in LEVELS)
    logger.info("probing %d problems across %d levels", n_per_level, len(LEVELS))

    results: dict[str, list[dict]] = {l: [] for l in LEVELS}
    summary: dict[str, dict[str, float]] = {}

    for lvl in LEVELS:
        n_correct = 0
        for p in probs[lvl]:
            input_ids = tok(p.prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
            correct = score(text, p.answer)
            n_correct += int(correct)
            results[lvl].append({
                "prompt": p.prompt,
                "target": p.answer,
                "generation": text,
                "correct": correct,
            })
        acc = n_correct / len(probs[lvl])
        summary[lvl] = {"n": len(probs[lvl]), "n_correct": n_correct, "accuracy": acc}
        logger.info("%s (%s): %d/%d = %.3f",
                    lvl, LEVEL_LABELS[lvl], n_correct, len(probs[lvl]), acc)

    output = {
        "model": model_path,
        "seed": SEED,
        "summary": summary,
        "details": results,
    }
    Path(output_path).write_text(json.dumps(output, indent=2))
    logger.info("wrote %s", output_path)


def summarize(json_paths: Iterable[str]) -> None:
    rows: list[tuple[str, dict[str, float]]] = []
    for jp in json_paths:
        data = json.loads(Path(jp).read_text())
        model = data["model"]
        # Use the basename of the model path as the label
        label = Path(model).name if model.startswith("/") else model
        rows.append((label, data["summary"]))

    header = f"{'model':<35}" + "".join(f" {lvl:>10}" for lvl in LEVELS)
    print(header)
    print("-" * len(header))
    for label, summ in rows:
        row = f"{label:<35}"
        for lvl in LEVELS:
            row += f" {summ[lvl]['accuracy']:>10.3f}"
        print(row)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="HF model path or hub id (single-model mode)")
    parser.add_argument("--output", help="Output JSON path (single-model mode)")
    parser.add_argument("--summary", nargs="+", help="Result JSON paths (summary mode)")
    args = parser.parse_args()

    if args.summary:
        summarize(args.summary)
        return
    if not args.model or not args.output:
        parser.error("must give --model + --output, or --summary")
    run_one_model(args.model, args.output)


if __name__ == "__main__":
    main()
