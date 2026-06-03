"""Wu et al §3.1 arithmetic counterfactual runner — phi models + ours.

Replicates Wu et al's prompt template + parsing exactly. Uses Wu's
released data files for the problem set and uses Wu's parse_output to
extract the answer from the model's generation.

Wu repo (must be cloned): /tmp/counterfactual-evaluation/

Per-(base, digit-width) accuracy. Default mode = 0-shot CoT with the
'You are a mathematician...' prompt template.

Usage:
  .venv/bin/python -m experiments.data_efficiency.wu_arithmetic_runner \
    --model microsoft/phi-1_5 --output /tmp/wu_arith_phi-1.5.json \
    --digit-widths 2,3,4 --bases 8,9,10,11,16 --n_per_base 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

WU_REPO = Path("/fsx/users/dongweij/marin/outputs/counterfactual-evaluation")
if not WU_REPO.exists():
    raise SystemExit(
        "Wu et al repo not present. Clone via: "
        "git clone --depth 1 https://github.com/ZhaofengWu/counterfactual-evaluation.git "
        f"into {WU_REPO}"
    )
sys.path.insert(0, str(WU_REPO))

# Wu's templatize function — inlined here to avoid pulling in their query
# pipeline (which imports anthropic/openai). The scoring uses Wu's
# `arithmetic.eval` module which has no API deps.
from arithmetic.eval import parse_output, get_label  # type: ignore


def templatize(expr: str, base: int, cot: bool = True, n_shots: int = 0) -> str:
    """Exact copy of Wu et al arithmetic/query.py templatize(), no API deps."""
    if n_shots > 0:
        raise NotImplementedError("n_shots > 0 not supported in this runner")
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if cot:
        return (
            f"You are a mathematician. Assuming that all numbers are in base-{base} "
            f"where the digits are \"{digits[:base]}\", what is {expr}? "
            f"Let's think step by step, and end the response with the result "
            f"in \"\\boxed{{result}}\"."
        )
    return (
        f"You are a mathematician. Assuming that all numbers are in base-{base} "
        f"where the digits are \"{digits[:base]}\", what is {expr}? "
        f"End the response with the result in \"\\boxed{{result}}\"."
    )

logger = logging.getLogger(__name__)


def load_problems(digit_width: int, base: int) -> list[str]:
    """Load Wu's released arithmetic problems for a given digit-width and base."""
    subdir = "0shot" if digit_width == 2 else f"0shot_{digit_width}digits"
    f = WU_REPO / "arithmetic" / "data" / subdir / f"base{base}.txt"
    with f.open() as fp:
        return [line.strip() for line in fp if line.strip()]


def score_one(generation: str, expr: str, base: int) -> tuple[bool, str]:
    """Wu-style: parse the model output, compare to the ground-truth label.

    Wu's parse_output raises AssertionError when the output is in a format
    it doesn't know how to handle (e.g., phi-1.5 returning Python code).
    Treat those as 'FAILED' (incorrect), not as a crash.
    """
    try:
        pred = parse_output(generation)
    except (AssertionError, Exception) as e:  # pragma: no cover
        pred = f"PARSE_FAILED({type(e).__name__})"
    gold = get_label(expr, base)
    return pred == gold, pred


def run_one_base(
    model,
    tok,
    digit_width: int,
    base: int,
    n: int,
    max_new_tokens: int,
) -> dict:
    problems = load_problems(digit_width, base)[:n]
    logger.info("digit=%d base=%d  %d problems", digit_width, base, len(problems))
    prompts = [templatize(p, base, cot=True, n_shots=0) for p in problems]

    n_correct = 0
    details: list[dict] = []
    for expr, prompt in zip(problems, prompts):
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        ok, parsed = score_one(text, expr, base)
        n_correct += int(ok)
        details.append({
            "expr": expr,
            "gold": get_label(expr, base),
            "generation": text,
            "parsed": parsed,
            "correct": ok,
        })
    return {
        "digit_width": digit_width,
        "base": base,
        "n": len(problems),
        "n_correct": n_correct,
        "accuracy": n_correct / max(1, len(problems)),
        "details": details,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--digit-widths", default="2,3,4")
    parser.add_argument("--bases", default="8,9,10,11,16")
    parser.add_argument("--n_per_base", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    digit_widths = [int(d) for d in args.digit_widths.split(",")]
    bases = [int(b) for b in args.bases.split(",")]

    logger.info("loading %s", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    results = {
        "model": args.model,
        "config": {
            "digit_widths": digit_widths, "bases": bases,
            "n_per_base": args.n_per_base, "max_new_tokens": args.max_new_tokens,
            "prompt": "Wu et al §3.1 templatize(cot=True, n_shots=0)",
            "scoring": "Wu et al arithmetic.eval.parse_output",
        },
        "by_base": [],
        "summary": {},
    }

    for dw in digit_widths:
        for base in bases:
            r = run_one_base(model, tok, dw, base, args.n_per_base, args.max_new_tokens)
            results["by_base"].append({k: v for k, v in r.items() if k != "details"})
            results["by_base"][-1]["details_path_hint"] = "in raw details list"
            # Also save full details under a sibling key.
            results.setdefault("_details", []).append(r)
            logger.info("digit=%d base=%d acc=%.3f", dw, base, r["accuracy"])

    # summary table
    for dw in digit_widths:
        for base in bases:
            r = next(x for x in results["by_base"] if x["digit_width"] == dw and x["base"] == base)
            results["summary"][f"d{dw}_b{base}"] = r["accuracy"]

    Path(args.output).write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", args.output)
    # Print summary
    print("\n=== summary ===")
    for dw in digit_widths:
        row = "  digit=%d:" % dw
        for base in bases:
            acc = results["summary"][f"d{dw}_b{base}"]
            row += f"  b{base}={acc:.2f}"
        print(row)


if __name__ == "__main__":
    main()
