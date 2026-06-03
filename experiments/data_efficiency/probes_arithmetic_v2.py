"""Arithmetic decomposition probe v2 — longer generation + answer regex.

v1 (probes_arithmetic.py) used max_new_tokens=4 and parsed the first integer
in the generation. That works for models that respond with bare answers
(our 4 1.4B models) but penalizes phi-1.5 which writes "\\n\\nSimplifying..."
chain-of-thought style.

v2 fixes this by:
  - max_new_tokens=64
  - Looking for the LAST integer in the generation, not the first
    (heuristic: chain-of-thought ends with the answer)
  - Stop on second newline (the model often writes "...\\n\\n" then starts
    a new problem; we don't care about that)

Same 5 problem levels and seed as v1.

Usage:
  .venv/bin/python -m experiments.data_efficiency.probes_arithmetic_v2 \\
    --model microsoft/phi-1_5 \\
    --output /tmp/arith_probe_v2_phi-1.5.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.data_efficiency.probes_arithmetic import (
    LEVELS,
    LEVEL_LABELS,
    SEED,
    generate_problems,
)

logger = logging.getLogger(__name__)

MAX_NEW_TOKENS = 64

# Integer-token regex. Match optional negative sign + digits.
_INT_RE = re.compile(r"-?\d+")


def parse_last_int(text: str) -> int | None:
    """Return the last integer in `text`, or None."""
    matches = _INT_RE.findall(text)
    return int(matches[-1]) if matches else None


def truncate_at_double_newline(text: str) -> str:
    """Cut at the first occurrence of two newlines — the model often writes
    the answer, then double-newlines, then starts a new problem."""
    idx = text.find("\n\n")
    return text[:idx] if idx >= 0 else text


def score(generation: str, target: int) -> bool:
    """Generation is correct iff the last integer in it (within the first
    paragraph) equals target."""
    trimmed = truncate_at_double_newline(generation)
    pred = parse_last_int(trimmed)
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
    logger.info("probing %d problems across %d levels (v2: max_new_tokens=%d, last-int)",
                n_per_level, len(LEVELS), MAX_NEW_TOKENS)

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
        "probe_version": "v2",
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "summary": summary,
        "details": results,
    }
    Path(output_path).write_text(json.dumps(output, indent=2))
    logger.info("wrote %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_one_model(args.model, args.output)


if __name__ == "__main__":
    main()
