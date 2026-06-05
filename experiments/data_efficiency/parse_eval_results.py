"""Parse lm-eval JSON results from a v2 eval run directory and print a table.

Usage:
  python parse_eval_results.py <eval_dir>

Looks for results_*.json files (recursively) and emits a one-row-per-task
summary using each task's canonical primary metric. Falls through a list of
acceptable metric keys per task (lm-eval names vary between tasks).
"""

import argparse
import json
from pathlib import Path

# Per-task primary metric. Order matters — first key in this list that
# exists in the result dict is what gets used. List of (key, label).
# Keep this in sync with EVALUATION.md so the table cells match.
PRIMARY = {
    "arc_easy": [("acc_norm,none", "acc_norm")],
    "arc_challenge": [("acc_norm,none", "acc_norm")],
    "hellaswag": [("acc_norm,none", "acc_norm")],
    "winogrande": [("acc,none", "acc")],
    "mmlu": [("acc,none", "acc")],
    "gsm8k": [("exact_match,strict-match", "EM strict")],
    "gsm8k_cot": [("exact_match,flexible-extract", "EM flex"),
                  ("exact_match,strict-match", "EM strict")],
    "piqa": [("acc,none", "acc"), ("acc_norm,none", "acc_norm")],
    "boolq": [("acc,none", "acc")],
    "sciq": [("acc,none", "acc"), ("acc_norm,none", "acc_norm")],
    "openbookqa": [("acc_norm,none", "acc_norm")],
    "openbookqa_fact": [("acc_norm,none", "acc_norm")],
    "commonsense_qa": [("acc,none", "acc")],
    "social_iqa": [("acc,none", "acc")],
    "logiqa": [("acc_norm,none", "acc_norm")],
    "lambada_openai": [("acc,none", "acc")],
    "copa": [("acc,none", "acc")],
    "wsc": [("acc,none", "acc")],
    "agieval_lsat_ar": [("acc_norm,none", "acc_norm")],
    "gpqa_diamond_zeroshot": [("acc,none", "acc"), ("acc_norm,none", "acc_norm")],
    "gpqa_diamond_n_shot": [("acc_norm,none", "acc_norm")],
    # bbh's lm-eval YAML uses get-answer (not get-response). Try both.
    "bbh": [("exact_match,get-answer", "EM get-answer"),
            ("exact_match,get-response", "EM get-response"),
            ("exact_match,none", "EM")],
    "mmlu_pro": [("exact_match,custom-extract", "EM custom-extract"),
                 ("exact_match,none", "EM")],
    "humaneval": [("pass@1,create_test", "pass@1")],
    "mbpp": [("pass_at_1,none", "pass@1")],
    "minerva_math": [("exact_match,none", "EM"),
                     ("math_verify,none", "math_verify")],
}

# Tasks whose subtask scores we want to suppress (only show aggregate)
AGGREGATED_PREFIXES = {"mmlu_", "bbh_", "mmlu_pro_", "minerva_math_", "gpqa_"}
# But these aggregate names ARE rows we want to show
AGGREGATE_NAMES = {"mmlu", "bbh", "mmlu_pro", "minerva_math", "gpqa_diamond_zeroshot", "gpqa_diamond_n_shot"}


def scan_dir(eval_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for p in sorted(eval_dir.rglob("results_*.json")):
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"WARN: could not parse {p}")
            continue
        for task, metrics in data.get("results", {}).items():
            # Skip subtask rows (e.g. mmlu_high_school_world_history) unless
            # explicitly an aggregate row name.
            if task not in AGGREGATE_NAMES and any(
                task.startswith(p) and task != p.rstrip("_") for p in AGGREGATED_PREFIXES
            ):
                continue
            results.setdefault(task, metrics)
    return results


def best_metric(task: str, metrics: dict) -> tuple[str, float] | None:
    candidates = PRIMARY.get(task)
    if candidates:
        for key, label in candidates:
            if key in metrics and isinstance(metrics[key], (int, float)):
                return label, float(metrics[key])
    # Fallback: any acc-like key.
    for k in ("acc_norm,none", "acc,none", "exact_match,none",
              "exact_match,strict-match", "exact_match,flexible-extract",
              "pass@1,create_test", "pass_at_1,none"):
        if k in metrics and isinstance(metrics[k], (int, float)):
            return k, float(metrics[k])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", type=Path)
    args = ap.parse_args()
    results = scan_dir(args.eval_dir)
    if not results:
        print(f"No results in {args.eval_dir}")
        return

    # Pull bigcode HumanEval if present.
    bigcode_path = args.eval_dir / "bigcode_humaneval" / "metrics.json"
    bigcode_he = None
    if bigcode_path.exists():
        try:
            bc = json.loads(bigcode_path.read_text())
            bigcode_he = bc.get("humaneval", {}).get("pass@1")
        except Exception:
            pass

    print(f"{'task':<40} {'metric':<22} {'value':>10}")
    print("-" * 76)
    for task in sorted(results):
        m = best_metric(task, results[task])
        if m is None:
            continue
        label, value = m
        print(f"{task:<40} {label:<22} {value:>10.4f}")
    if bigcode_he is not None:
        print(f"{'humaneval (bigcode)':<40} {'pass@1':<22} {bigcode_he:>10.4f}")


if __name__ == "__main__":
    main()
