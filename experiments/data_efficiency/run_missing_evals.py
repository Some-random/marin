#!/usr/bin/env python
"""
Run all missing evaluations on existing HF checkpoints.
Saves results to outputs/eval_results/missing_evals.json.

Checkpoints and their missing evals:
- 300m_hf (Run A baseline): already has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande
- run_C_phase2_hf (300M Run C): already has arc_challenge/hellaswag/winogrande/mmlu => need arc_easy, piqa, sciq
- run_D_phase2_hf (300M Run D): no easy benchmark evals => need arc_easy, piqa, sciq
- 600m_baseline_hf: already has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande
- 600m_run_C_phase2_hf: already has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande
- 600m_run_D_phase2_hf: already has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande
- 1_4b_baseline_hf: has arc_challenge/arc_easy/piqa/sciq => need hellaswag, winogrande, mmlu
- 1_4b_run_B_hf (OT-only): has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande, mmlu
- 1_4b_run_C_hf: has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande, mmlu
- 1_4b_run_D_hf: has arc_easy/piqa/sciq => need arc_challenge, hellaswag, winogrande, mmlu
- kb6hhnxn (mixed DCLM+OWM 300M): needs HF conversion first, then all evals
"""

import os
import sys
import json
import logging
import time
from pathlib import Path

# Load secrets
secrets_path = Path("/fsx/users/dongweij/marin/.secrets")
if secrets_path.exists():
    for line in secrets_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

BASE_DIR = "/fsx/users/dongweij/marin"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs/eval_results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "missing_evals.json")
LOG_FILE = os.path.join(BASE_DIR, "outputs/missing_evals.log")

os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---- Step 1: Convert kb6hhnxn checkpoint to HF format ----

MIXED_RUN_RAW = os.path.join(CKPT_DIR, "kb6hhnxn/step-5402")
MIXED_RUN_HF = os.path.join(CKPT_DIR, "kb6hhnxn_mixed_hf")


def convert_kb6hhnxn():
    """Convert the mixed DCLM+OWM 300M checkpoint to HF format."""
    if os.path.exists(MIXED_RUN_HF) and os.path.exists(os.path.join(MIXED_RUN_HF, "config.json")):
        log.info("kb6hhnxn HF checkpoint already exists, skipping conversion")
        return

    log.info("Converting kb6hhnxn (mixed DCLM+OWM 300M) to HF format...")

    import jax
    import haliax as hax
    import equinox as eqx
    from levanter.checkpoint import load_checkpoint
    from levanter.compat.hf_checkpoints import load_tokenizer

    # Add experiments to path so we can import model configs
    sys.path.insert(0, BASE_DIR)
    from experiments.data_efficiency.models import model_dict

    model_config = model_dict["300m4k"]
    tokenizer = load_tokenizer("meta-llama/Meta-Llama-3.1-8B")
    Vocab = hax.Axis("vocab", len(tokenizer))
    key = jax.random.PRNGKey(0)
    mesh = jax.sharding.Mesh(jax.devices("cpu")[:1], ("data",))

    with hax.partitioning.set_mesh(mesh):
        model = eqx.filter_eval_shape(model_config.build, Vocab, key=key)
        model = load_checkpoint(model, MIXED_RUN_RAW, subpath="model")
        converter = model_config.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
        converter.save_pretrained(model, MIXED_RUN_HF)
        tokenizer.save_pretrained(MIXED_RUN_HF)

    log.info("Conversion complete: %s", MIXED_RUN_HF)


# ---- Step 2: Define all eval jobs ----

EASY_TASKS = ["arc_easy", "piqa", "sciq"]
HARD_TASKS = ["arc_challenge", "hellaswag", "winogrande"]
MMLU_TASK = ["mmlu"]

# Each entry: (label, hf_path, tasks_to_run)
EVAL_JOBS = [
    # 300M models - missing harder benchmarks
    ("300m_A_baseline", os.path.join(CKPT_DIR, "300m_hf"), HARD_TASKS),
    ("300m_C_OT_then_DCLM", os.path.join(CKPT_DIR, "run_C_phase2_hf"), EASY_TASKS),
    ("300m_D_DCLM_then_OT", os.path.join(CKPT_DIR, "run_D_phase2_hf"), EASY_TASKS + HARD_TASKS),

    # 600M models - missing harder benchmarks
    ("600m_A_baseline", os.path.join(CKPT_DIR, "600m_baseline_hf"), HARD_TASKS),
    ("600m_C_OT_then_DCLM", os.path.join(CKPT_DIR, "600m_run_C_phase2_hf"), HARD_TASKS),
    ("600m_D_DCLM_then_OT", os.path.join(CKPT_DIR, "600m_run_D_phase2_hf"), HARD_TASKS),

    # 1.4B models - baseline missing hellaswag/winogrande/mmlu; B/C/D missing harder benchmarks + mmlu
    ("1_4b_A_baseline", os.path.join(CKPT_DIR, "1_4b_baseline_hf"), ["hellaswag", "winogrande"] + MMLU_TASK),
    ("1_4b_B_OT_only", os.path.join(CKPT_DIR, "1_4b_run_B_hf"), HARD_TASKS + MMLU_TASK),
    ("1_4b_C_OT_then_DCLM", os.path.join(CKPT_DIR, "1_4b_run_C_hf"), HARD_TASKS + MMLU_TASK),
    ("1_4b_D_DCLM_then_OT", os.path.join(CKPT_DIR, "1_4b_run_D_hf"), HARD_TASKS + MMLU_TASK),

    # Mixed DCLM+OWM 300M - all evals needed
    ("300m_mixed_dclm80_owm20", MIXED_RUN_HF, EASY_TASKS + HARD_TASKS),
]


def run_eval(label, hf_path, tasks):
    """Run lm-eval-harness on a single checkpoint."""
    import torch
    from lm_eval.models.huggingface import HFLM
    from lm_eval import simple_evaluate

    log.info("Evaluating %s at %s on tasks: %s", label, hf_path, tasks)

    if not os.path.exists(hf_path):
        log.error("Checkpoint not found: %s -- skipping %s", hf_path, label)
        return None

    bs = 16 if "1_4b" in label else 64
    t0 = time.time()
    model = HFLM(pretrained=hf_path, dtype="bfloat16", device="cuda:0", batch_size=bs)
    results = simple_evaluate(model=model, tasks=tasks, batch_size=bs)
    elapsed = time.time() - t0

    # Extract numeric results only
    extracted = {}
    for task, r in results["results"].items():
        extracted[task] = {k: v for k, v in r.items() if isinstance(v, (int, float))}

    log.info("Finished %s in %.1fs", label, elapsed)
    for task in tasks:
        if task in extracted:
            acc = extracted[task].get("acc,none", "N/A")
            log.info("  %s: acc=%s", task, acc)
        elif task == "mmlu" and "mmlu" not in extracted:
            # mmlu expands into subtasks; look for the aggregate
            for k, v in extracted.items():
                if k.startswith("mmlu") and "acc" in str(v):
                    break

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return extracted


def main():
    log.info("=" * 60)
    log.info("Starting missing evaluations")
    log.info("=" * 60)

    # Step 1: Convert kb6hhnxn
    try:
        convert_kb6hhnxn()
    except Exception as e:
        log.error("Failed to convert kb6hhnxn: %s", e, exc_info=True)
        log.info("Will skip mixed run evals and continue with other checkpoints")

    # Step 2: Run all evals
    all_results = {}

    # Load existing results if the file exists (for resumability)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        log.info("Loaded %d existing results from %s", len(all_results), RESULTS_FILE)

    for label, hf_path, tasks in EVAL_JOBS:
        if label in all_results:
            log.info("Skipping %s (already in results file)", label)
            continue

        try:
            result = run_eval(label, hf_path, tasks)
            if result is not None:
                all_results[label] = result
                # Save after each eval for resumability
                with open(RESULTS_FILE, "w") as f:
                    json.dump(all_results, f, indent=2)
                log.info("Saved intermediate results (%d total)", len(all_results))
        except Exception as e:
            log.error("Failed to evaluate %s: %s", label, e, exc_info=True)

    log.info("=" * 60)
    log.info("All evaluations complete. Results saved to %s", RESULTS_FILE)
    log.info("Total checkpoints evaluated: %d", len(all_results))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
