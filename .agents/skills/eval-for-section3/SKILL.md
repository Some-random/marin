---
name: eval-for-section3
description: Run the canonical v2-suite evaluation on a model checkpoint and populate or update a column in §3 of `experiments/data_efficiency/EVALUATION.md`. Use whenever the user asks to evaluate a model for the EVALUATION.md table, fill missing §3 cells, refresh historical mbpp/humaneval, or compute Mean rows. NEVER use `eval_intermediate.sh` for §3 — it skips ~12 tasks. This skill always uses `run_eval_v2.sh`.
---

# Skill: Evaluate a model for §3 of EVALUATION.md

This skill enforces the canonical evaluation pipeline so §3 columns are never half-empty. It always uses `run_eval_v2.sh` (the full v2 suite — 14 task groups including lambada / copa / wsc / agieval / gpqa / bbh / mmlu_pro / bigcode humaneval).

## Required info

If any is missing, ask before proceeding.

1. `LABEL` — the column name to use in §3 (e.g. `c5v3_p2_a6_step14671`). Used as both the directory tag for results and the §3 column label after stripping markdown markers.
2. `LEVANTER_DIR` OR `HF_DIR` — either the Levanter checkpoint dir (`checkpoints/<run_id>/step-N`) or an existing HF checkpoint dir (`checkpoints/<name>_hf`). If only Levanter is given, the skill converts to HF first.
3. `EVAL_NODE` — which single GPU node to run on (e.g. `gpu-st-p4d24xlarge-2`). Must be reachable via ssh and have <1 GB GPU memory in use. If the user didn't specify, default to a known free node (run `nvidia-smi` first to confirm).

## Procedure

### Step 1: Verify the node is free

```bash
ssh <EVAL_NODE> "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc"
```

If >1000 MiB, find another free node or abort. Do not start an eval on a node that's already running training/eval.

### Step 2: Run the eval via `convert_and_eval_v2.sh`

```bash
TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
LOG=/fsx/users/dongweij/marin/logs/v2_${LABEL}_${TS}.log
nohup bash /fsx/users/dongweij/marin/experiments/data_efficiency/convert_and_eval_v2.sh \
  --label <LABEL> \
  --src <LEVANTER_DIR> \
  --hf-dst /fsx/users/dongweij/marin/checkpoints/<LABEL>_hf \
  --node <EVAL_NODE> \
  > $LOG 2>&1 < /dev/null &
disown
```

`convert_and_eval_v2.sh` handles both the Levanter→HF conversion (skip if HF dir already exists) and the FULL `run_eval_v2.sh` invocation on the eval node. The first line of its output must be `=== convert + v2 eval ===` — if it says `=== Intermediate eval ===`, the wrong script is being called.

### Step 3: Arm a Monitor

```python
Monitor(
  description="<LABEL> v2-suite eval: task DONE/FAILED + ALL DONE",
  persistent=True,
  timeout_ms=3600000,
  command='tail -F -q /fsx/users/dongweij/marin/logs/v2_<LABEL>_*.log | grep -E --line-buffered "tasks=.*DONE|FAILED-CONTINUE|ALL DONE|panic|SIGSEGV|Traceback|RuntimeError|RESOURCE_EXHAUSTED"'
)
```

Plus a 25-min `ScheduleWakeup` backstop in case the log goes silent.

### Step 4: Wait for `ALL DONE`

The full v2 suite takes ~45 min. Don't end the loop until the log emits `[<LABEL>] ALL DONE → /fsx/users/dongweij/marin/outputs/eval_results/v2_<LABEL>_<TS>/`. If any task hits `FAILED-CONTINUE`, surface it to the user — the run continues but that cell will be missing.

### Step 5: Pull scores

Results live under `outputs/eval_results/v2_<LABEL>_<TS>/`. Each task group has its own subdir with a `results_*.json`. Key extraction patterns:

- Standard tasks: `find <RESULTS_DIR> -name "*results*.json"` then read `results.<task>.<metric>` where metric ends in `,none` / `,strict-match` / `,create_test`.
- `mmlu[5]` → use the **mean** of all `mmlu_<subject>` `acc,none` values (exclude `mmlu_humanities`, `mmlu_social_sciences`, `mmlu_stem`, `mmlu_other` category meta-tasks).
- `lambada_openai[0]` → use `acc,none`, NOT `perplexity,none`.
- `humaneval[0] (bigcode)` → read `bigcode_humaneval/metrics.json` field `humaneval.pass@1`.
- `minerva_math[4]` → prefer `exact_match,none` (matches existing §3 values).

### Step 6: Update §3 of EVALUATION.md

If the model is a NEW column:
- Insert the column header between `C5-v2 final ‖` and `4B final ª` (or wherever logically fits).
- Add a row to §2 describing the model (params, tokens, FLOPs, recipe).
- Pick a footnote symbol for the model variant (e.g. `◊`) and define it.
- Reuse the Python pattern in `/tmp/finalize_evaluation_md.py` for bulk cell insertion.

If filling cells in an EXISTING column:
- Run the Python helper at `/tmp/fill_c5v3_small_final.py` (model template — copy + change `RESULTS_DIR` + column index).
- Validate after: `grep "^| sciq\[0\]" experiments/data_efficiency/EVALUATION.md` — confirm new value present.

### Step 7: Recompute Mean rows

After any cell update, the 5 Mean rows (Mean Open-book, Mean Closed-book NL, Mean Aggregate, Mean Math (standard), Mean Code) must be recomputed in place. The fill script template already does this.

### Step 8: Commit and push

```bash
git add experiments/data_efficiency/EVALUATION.md
git commit -m "EVALUATION.md: <what changed>"
git push origin main
```

## Validation gates (mandatory)

- The eval log's first line must be `=== convert + v2 eval ===`. If `Intermediate eval` appears, stop and switch to `convert_and_eval_v2.sh`.
- Before committing, every row in §3 must have 19 pipes (= 18 cells). If any row has fewer, the table is misaligned.
- For each new column, every task this skill ran for must have a number; missing tasks are `—` (no silent zeros).

## Anti-patterns

- Calling `eval_intermediate.sh` directly. It skips ~12 §3 tasks.
- Patching a running supervisor with `sed -i` — bash caches the script; you must kill + relaunch.
- Reporting `mbpp` or `humaneval` as `FAILED-CONTINUE` without re-running with `--num_processes 1` (or `convert_and_eval_v2.sh`, which has the per-rank metrics cache fix).
- Updating §3 without recomputing Mean rows.

## Why this skill exists

On 2026-06-10–11 I used `eval_intermediate.sh` for the C5-v3 family, then noticed half the §3 cells were `—` because that script skips lambada / copa / wsc / agieval / gpqa / bbh / mmlu_pro / bigcode humaneval / paloma per-subset / gsm_symbolic / gsm_noop. Re-running everything cost an extra ~2 hours of wall time and user frustration. This skill enforces the right script at the gate.
