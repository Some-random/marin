#!/usr/bin/env python
"""
Comprehensive evaluation suite matching benchmarks from papers in reasoning_curriculum.md.

Organized by category to cover all three research objectives:
  (1) Data efficiency — measured via training loss/perplexity (not here, done during training)
  (2) Reasoning quality — math, code, compositional, logic benchmarks
  (3) General NL performance — commonsense, world knowledge, reading comprehension, grammaticality

Usage:
  # Run all loglikelihood tasks on 300M baseline (fast, no generation)
  python run_comprehensive_evals.py --checkpoint checkpoints/300m_hf --suite logprob

  # Run everything including generation tasks (slow)
  python run_comprehensive_evals.py --checkpoint checkpoints/300m_hf --suite all

  # Run on a specific category
  python run_comprehensive_evals.py --checkpoint checkpoints/300m_hf --suite reasoning

  # Run all checkpoints
  python run_comprehensive_evals.py --all-checkpoints --suite logprob

  # Dry run — just print what would be evaluated
  python run_comprehensive_evals.py --checkpoint checkpoints/300m_hf --suite all --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

secrets_path = Path("/fsx/users/dongweij/marin/.secrets")
if secrets_path.exists():
    for line in secrets_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BASE_DIR = "/fsx/users/dongweij/marin"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs/eval_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "outputs/comprehensive_evals.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# =============================================================================
# Benchmark definitions by category
# =============================================================================

# --- NL Reasoning (Aryabumi's 11 benchmarks + extras) ---
# These are all loglikelihood-based (multiple choice)
NL_REASONING = [
    "arc_easy",        # grade-school science MCQ (Aryabumi, Petty, ours)
    "arc_challenge",   # harder science MCQ (Aryabumi, ours)
    "hellaswag",       # commonsense completion (Aryabumi, ours, Beyond Random)
    "piqa",            # physical intuition (Aryabumi, ours, Beyond Random)
    "winogrande",      # coreference/commonsense (Aryabumi, ours, Beyond Random)
    "openbookqa",      # multi-step science reasoning (Aryabumi, Beyond Random)
    "copa",            # causal reasoning (Aryabumi, Beyond Random)
    "boolq",           # yes/no reading comprehension (Aryabumi, Beyond Random)
    "social_iqa",      # social commonsense (Aryabumi)
    "commonsense_qa",  # 5-way commonsense MCQ (Aryabumi)
    "logiqa",          # logical reasoning MCQ (Aryabumi)
    "sciq",            # science MCQ (ours)
]

# --- World Knowledge ---
# triviaqa and nq_open require generation
WORLD_KNOWLEDGE_LOGPROB = [
    "mmlu",            # massive multitask (Aryabumi, ours, Kang)
    "truthfulqa_mc2",  # factual accuracy MCQ
]
WORLD_KNOWLEDGE_GEN = [
    "triviaqa",        # open-domain QA (Aryabumi)
    "nq_open",         # Natural Questions open (Aryabumi)
]

# --- Code ---
# humaneval requires generation + code execution
CODE_GEN = [
    "humaneval",       # code generation (Aryabumi, Petty)
    "mbpp",            # code generation (Aryabumi)
]

# --- Math/Arithmetic ---
MATH_LOGPROB = [
    "gsm8k",           # grade school math MCQ variant (Front-Loading, Echo Chamber)
    "mathqa",          # math word problems MCQ
]
MATH_GEN = [
    "gsm8k_cot",       # grade school math with CoT generation (Front-Loading)
    "minerva_math",    # MATH benchmark (Front-Loading)
]

# --- Linguistic / Grammaticality ---
LINGUISTIC = [
    "blimp",           # grammaticality judgments (Between Circuits)
    "lambada_openai",  # word prediction (language modeling quality)
]

# --- BigBench ---
# bbh_zeroshot is logprob-based, bbh_cot_fewshot requires generation
BIGBENCH_LOGPROB = [
    "bbh_zeroshot",    # BigBench Hard zero-shot MCQ (Petty uses full BigBench 204 tasks)
]
BIGBENCH_GEN = [
    "bbh_cot_fewshot", # BigBench Hard with CoT (3-shot)
]

# --- Reading Comprehension ---
READING_LOGPROB = [
    "race",            # reading comprehension MCQ
]
READING_GEN = [
    "drop",            # discrete reasoning over paragraphs
]

# --- Composites ---
ALL_LOGPROB = (
    NL_REASONING
    + WORLD_KNOWLEDGE_LOGPROB
    + MATH_LOGPROB
    + LINGUISTIC
    + BIGBENCH_LOGPROB
    + READING_LOGPROB
)

ALL_GEN = (
    WORLD_KNOWLEDGE_GEN
    + CODE_GEN
    + MATH_GEN
    + BIGBENCH_GEN
    + READING_GEN
)

ALL_TASKS = ALL_LOGPROB + ALL_GEN

SUITES = {
    "logprob": ALL_LOGPROB,
    "generation": ALL_GEN,
    "all": ALL_TASKS,
    "reasoning": NL_REASONING + MATH_LOGPROB + BIGBENCH_LOGPROB,
    "knowledge": WORLD_KNOWLEDGE_LOGPROB + WORLD_KNOWLEDGE_GEN,
    "code": CODE_GEN,
    "linguistic": LINGUISTIC,
    "quick": ["arc_easy", "arc_challenge", "piqa", "sciq", "hellaswag", "winogrande", "mmlu"],
}

ALL_HF_CHECKPOINTS = [
    ("300m_baseline", os.path.join(CKPT_DIR, "300m_hf")),
    ("300m_C_OT_DCLM", os.path.join(CKPT_DIR, "run_C_phase2_hf")),
    ("300m_D_DCLM_OT", os.path.join(CKPT_DIR, "run_D_phase2_hf")),
    ("300m_mixed_80_20", os.path.join(CKPT_DIR, "kb6hhnxn_mixed_hf")),
    ("600m_baseline", os.path.join(CKPT_DIR, "600m_baseline_hf")),
    ("600m_C_OT_DCLM", os.path.join(CKPT_DIR, "600m_run_C_phase2_hf")),
    ("600m_D_DCLM_OT", os.path.join(CKPT_DIR, "600m_run_D_phase2_hf")),
    ("1_4b_baseline", os.path.join(CKPT_DIR, "1_4b_baseline_hf")),
    ("1_4b_B_OT_only", os.path.join(CKPT_DIR, "1_4b_run_B_hf")),
    ("1_4b_C_OT_DCLM", os.path.join(CKPT_DIR, "1_4b_run_C_hf")),
    ("1_4b_D_DCLM_OT", os.path.join(CKPT_DIR, "1_4b_run_D_hf")),
]


METRIC_PRIORITY = [
    "acc,none",
    "acc_norm,none",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match,get-answer",
    "exact_match,remove_whitespace",
    "exact_match,none",
    "math_verify,none",
    "pass@1,create_test",
    "pass_at_1,none",
    "f1,none",
    "em,none",
]


def _best_metric(metrics: dict) -> str:
    for key in METRIC_PRIORITY:
        if key in metrics:
            return f"{metrics[key]:.4f} ({key})"
    return "N/A"


def _best_metric_value(metrics: dict) -> float | None:
    for key in METRIC_PRIORITY:
        if key in metrics:
            return metrics[key]
    return None


def run_eval(
    label: str,
    hf_path: str,
    task_list: list[str],
    batch_size: int = 32,
    limit: int | None = None,
    samples_dir: str | None = None,
) -> dict | None:
    """Run lm-eval-harness with per-example sample logging.

    Per-sample outputs (prompt, model response, target, score) are always saved when
    `samples_dir` is provided. We never just save aggregate scores — per-example
    outputs are required by the project's eval rules.
    """
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    if not os.path.exists(hf_path) and "/" not in hf_path:
        log.error("Checkpoint not found: %s — skipping %s", hf_path, label)
        return None

    if "1_4b" in label or "1.4b" in label:
        batch_size = min(batch_size, 16)

    log.info("Evaluating %s on %d tasks: %s", label, len(task_list), task_list)
    t0 = time.time()

    model = HFLM(pretrained=hf_path, dtype="bfloat16", device="cuda:0", batch_size=batch_size)
    results = simple_evaluate(
        model=model,
        tasks=task_list,
        batch_size=batch_size,
        limit=limit,
        confirm_run_unsafe_code=True,
        log_samples=True,
    )
    elapsed = time.time() - t0

    extracted = {}
    for task, r in results["results"].items():
        extracted[task] = {k: v for k, v in r.items() if isinstance(v, (int, float))}

    log.info("Finished %s in %.1fs", label, elapsed)
    for task in sorted(extracted.keys()):
        log.info("  %s: %s", task, _best_metric(extracted[task]))

    if samples_dir is not None:
        task_samples_dir = os.path.join(samples_dir, label)
        os.makedirs(task_samples_dir, exist_ok=True)
        samples = results.get("samples", {}) or {}
        for task_name, task_samples in samples.items():
            with open(os.path.join(task_samples_dir, f"{task_name}.json"), "w") as f:
                json.dump(task_samples, f, indent=2, default=str)
        log.info("Saved per-sample outputs for %d tasks to %s", len(samples), task_samples_dir)

    del model
    torch.cuda.empty_cache()

    return extracted


def print_summary(all_results: dict, task_list: list[str]):
    """Print a comparison table across all evaluated checkpoints."""
    if not all_results:
        return

    # Collect all task names that actually have results
    all_task_names = set()
    for res in all_results.values():
        all_task_names.update(res.keys())

    # For aggregate tasks like mmlu, collect subtask averages
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Header
    labels = list(all_results.keys())
    header = f"{'Task':<25}" + "".join(f"{l:<18}" for l in labels)
    print(header)
    print("-" * len(header))

    for task in sorted(all_task_names):
        row = f"{task:<25}"
        for label in labels:
            res = all_results.get(label, {}).get(task, {})
            acc = _best_metric_value(res)
            if acc is not None:
                row += f"{acc:<18.4f}"
            else:
                row += f"{'—':<18}"
        print(row)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LM evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to HF checkpoint")
    parser.add_argument("--label", type=str, help="Label for this checkpoint (auto-detected if not given)")
    parser.add_argument("--all-checkpoints", action="store_true", help="Run on all known HF checkpoints")
    parser.add_argument("--suite", type=str, default="logprob", choices=list(SUITES.keys()),
                        help="Which benchmark suite to run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (useful for smoke tests)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--samples-dir", type=str, default=None,
                        help="Directory to save per-sample outputs (default: <output_dir>/samples_<suite>)")
    parser.add_argument("--dry-run", action="store_true", help="Just print what would be evaluated")
    parser.add_argument("--resume", action="store_true", help="Skip checkpoints already in output file")
    args = parser.parse_args()

    task_list = SUITES[args.suite]

    if args.dry_run:
        print(f"Suite: {args.suite}")
        print(f"Tasks ({len(task_list)}):")
        for t in task_list:
            print(f"  - {t}")
        if args.all_checkpoints:
            print(f"\nCheckpoints ({len(ALL_HF_CHECKPOINTS)}):")
            for label, path in ALL_HF_CHECKPOINTS:
                exists = os.path.exists(path)
                print(f"  {'✓' if exists else '✗'} {label}: {path}")
        elif args.checkpoint:
            print(f"\nCheckpoint: {args.checkpoint}")
        return

    output_path = args.output or os.path.join(RESULTS_DIR, f"comprehensive_{args.suite}.json")
    samples_dir = args.samples_dir or os.path.join(RESULTS_DIR, f"samples_{args.suite}")

    all_results = {}
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            all_results = json.load(f)
        log.info("Loaded %d existing results from %s", len(all_results), output_path)

    checkpoints = []
    if args.all_checkpoints:
        checkpoints = [(l, p) for l, p in ALL_HF_CHECKPOINTS if os.path.exists(p)]
    elif args.checkpoint:
        label = args.label or Path(args.checkpoint).name.replace("_hf", "")
        checkpoints = [(label, args.checkpoint)]
    else:
        parser.error("Specify --checkpoint or --all-checkpoints")

    for label, path in checkpoints:
        if args.resume and label in all_results:
            log.info("Skipping %s (already in results)", label)
            continue

        try:
            result = run_eval(
                label, path, task_list,
                batch_size=args.batch_size,
                limit=args.limit,
                samples_dir=samples_dir,
            )
            if result is not None:
                all_results[label] = result
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2)
                log.info("Saved results (%d checkpoints) to %s", len(all_results), output_path)
        except Exception as e:
            log.error("Failed to evaluate %s: %s", label, e, exc_info=True)

    print_summary(all_results, task_list)


if __name__ == "__main__":
    main()
