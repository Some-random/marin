---
name: eval-for-section3
description: Evaluate a model for §3 of `experiments/data_efficiency/EVALUATION.md` and fill its column without breaking the table. Use whenever the user asks to evaluate a model for the EVALUATION.md table, fill missing §3 cells, refresh historical mbpp/humaneval, or compute Mean rows. Always delegate to `experiments/data_efficiency/eval_section3.py` (NOT raw scripts) — that tool has the canonical task config, all metric fallbacks, Mean row computation, and table-structure validation built in.
---

# QUICK PATH — ONE COMMAND FOR A NEW MODEL

```bash
# End-to-end: pick 3 free nodes, insert §3 column + §2 row, run v2-suite + paloma + gsm in parallel,
# extract dclm_200m_val from training log, fill every fillable cell, strict-validate.
.venv/bin/python experiments/data_efficiency/eval_section3.py add-model \
  --label <LABEL> \
  --src <LEVANTER_OR_HF_DIR> \
  --train-log <PATH/TO/levanter_stdout.log> \
  --footnote-marker '◊' \
  --insert-before "4B final" \
  --params "1.4 B" --tokens "30.77 B" --flops "2.6 × 10²⁰" \
  --unique "..." --notes "..."
```

Flags: `--no-v2`, `--no-paloma`, `--no-gsm` skip individual sub-evals.
`--background` launches everything then exits without waiting (re-run `fill-from-results` later).

# QUICK PATH — sharded v2-suite (4 nodes, ~16-19 min instead of ~67)

When 4 GPU nodes are free, the v2-suite can be split across them. Each
shard runs ~25% of task groups in parallel into the SAME OUT_ROOT:

```bash
nohup bash /fsx/users/dongweij/marin/experiments/data_efficiency/convert_and_eval_v2_sharded.sh \
  --label <LABEL> \
  --src <LEVANTER_DIR> \
  --hf-dst /fsx/users/dongweij/marin/checkpoints/<LABEL>_hf \
  --shard-nodes "node1,node2,node3,node4" \
  > /fsx/users/dongweij/marin/logs/v2_<LABEL>_<TS>.log 2>&1 < /dev/null &
disown
```

Verify each node is <1 GB GPU memory before launch. The 4 shards (A/B/C/D)
are balanced to ~16-19 min each based on observed C5-v6-NEW v7 timings.
fill-from-results works the same on the shared OUT_ROOT (file layout is
identical to single-node mode).

# QUICK PATH — sub-commands (when add-model can't handle a case)

```bash
# v2-suite only (no paloma / gsm / column insert):
.venv/bin/python experiments/data_efficiency/eval_section3.py run <LABEL> <LEVANTER_OR_HF_DIR> [--node NODE]
# Then when 'ALL DONE' appears in the log:
.venv/bin/python experiments/data_efficiency/eval_section3.py fill-from-results <RESULTS_DIR> "<COLUMN_LABEL_SUBSTR>"
# Strict-fail by default if any v2-suite task is missing. Add --allow-missing
# only for intentional partial backfill (e.g. one cell from one re-run).
# Manual cell fill (e.g. from grepped training log):
.venv/bin/python experiments/data_efficiency/eval_section3.py fill-cell --row "<ROW_LABEL>" --col "<COL_SUBSTR>" --value 1.234
# Validate after manual edits:
.venv/bin/python experiments/data_efficiency/eval_section3.py validate
# Strict — fails on any (model, task) cell missing a value (excluding documented blanks):
.venv/bin/python experiments/data_efficiency/eval_section3.py validate --strict
```

The full procedure (only as backup when the tool can't handle a case):

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

### Step 2b: ALSO run the auxiliary task runners (NOT in run_eval_v2.sh)

`run_eval_v2.sh` does not cover the full §3 row set. After the v2 suite, ALSO dispatch these on the same (or another) node:

- `run_paloma_for_model.sh <LABEL> <HF_DIR>` — paloma_macro (16 subsets via lm-eval). For 4B+ models, set `BATCH_SIZE=4`.
- `run_gsm_for_model.sh <LABEL> <HF_DIR>` — gsm_symbolic_main + gsm_noop.
- `run_aryabumi_nl_extras.sh <LABEL> <HF_DIR>` — storycloze_2018_local + cb (super_glue/CB).
- `run_quac_for_model.sh <LABEL> <HF_DIR>` — quac_first_turn (F1, 1000 single-shot QA).

**ALWAYS use the absolute path to the script when ssh-invoking on a remote node.** SSH starts you in `$HOME` on the remote, not in the marin checkout — `bash experiments/data_efficiency/run_xxx.sh` will fail with `No such file or directory` because it looks for `$HOME/experiments/...`. The script itself does `cd /fsx/users/dongweij/marin` internally on line 7, but bash never finds the script in the first place. Canonical copy-paste snippet for ALL aux runners:

```bash
LABEL=<LABEL>
HF=/fsx/users/dongweij/marin/checkpoints/${LABEL}_hf
TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
MARIN=/fsx/users/dongweij/marin

# paloma (set BATCH_SIZE=4 inline before "bash" for 4B+ models)
nohup ssh <NODE_A> "bash $MARIN/experiments/data_efficiency/run_paloma_for_model.sh $LABEL $HF" \
  > $MARIN/logs/paloma_${LABEL}_${TS}.log 2>&1 < /dev/null & disown

# gsm
nohup ssh <NODE_B> "bash $MARIN/experiments/data_efficiency/run_gsm_for_model.sh $LABEL $HF" \
  > $MARIN/logs/gsm_${LABEL}_${TS}.log 2>&1 < /dev/null & disown

# aryabumi-nl-extras (storycloze + cb)
nohup ssh <NODE_C> "bash $MARIN/experiments/data_efficiency/run_aryabumi_nl_extras.sh $LABEL $HF" \
  > $MARIN/logs/aryabumi_nl_extras_${LABEL}_${TS}.log 2>&1 < /dev/null & disown

# quac (single task)
nohup ssh <NODE_D> "bash $MARIN/experiments/data_efficiency/run_quac_for_model.sh $LABEL $HF" \
  > $MARIN/logs/quac_${LABEL}_${TS}.log 2>&1 < /dev/null & disown
```

For dclm_200m_val (bpb): if you have a fresh-trained model with a Levanter wandb log, grep `eval/dclm_200m_val/loss` and convert via `bpb = nats × 0.3273` (4.408 bytes/Llama-token on dclm). For external models (no in-training eval), use the custom `dclm_200m_val.yaml` lm-eval task on the same 5000-doc dclm slice.

For phi-1/phi-1.5 specifically: also run `run_aryabumi_nl_extras.sh` and `run_quac_for_model.sh` using `microsoft/phi-1` and `microsoft/phi-1_5` as the HF_DIR (no local checkpoint dir needed).

The `add-model` subcommand orchestrates v2-suite + paloma + gsm. The aryabumi-nl-extras and QUAC runners are not yet integrated into `add-model` — call them manually after add-model completes (use the canonical snippet above with absolute paths). TODO: extend `add-model` to fan these out too.

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

The full v2 suite takes ~45 min. Don't end the loop until the log emits the terminal marker. As of 2026-06-22 the runners emit a **truthful** marker (they can no longer print a clean "ALL DONE" when a task crashed — PASS is decided by whether a `results_*.json` was actually written, not by exit code):
- Clean: `[<LABEL>] ALL DONE (0 failures) → <dir>` — proceed.
- Dirty: `[<LABEL>] ALL DONE WITH FAILURES (N task-group(s) FAILED: <names>) → <dir>` and the script **exits non-zero**. Surface the named failures to the user and **re-run them** before extracting; do NOT fill §3 from a run that printed `WITH FAILURES`. The same convention applies to the aux runners (`paloma`, `gsm`, `aryabumi-nl`, `quac`): `… ALL DONE (N/N ok)` vs `… ALL DONE WITH FAILURES (M/N ok, K FAILED: …)`.

### Step 4b: When tasks fail — triage, don't blind-retry

On any `WITH FAILURES`, the runner **automatically** runs `analyze_eval_failures.py "$OUT_ROOT"`, which writes `FAILURES.md` into the result dir and prints a per-task diagnosis: the root-cause class (`NCCL_GATHER_OOM` / `CUDA_OOM` / `OFFLINE_MODE` / `HUB_CONN` / `CODE_EVAL_CACHE` / `KILLED` / `UNKNOWN` / …), whether it's `transient` (safe to retry) or `permanent` (fix the config first), the suggested fix, and the first real traceback. **Read `FAILURES.md` before re-running.** Retry only the `transient` classes; for `permanent` ones, apply the suggested fix (e.g. lower batch, set OFFLINE=0) and only then re-run. Run it manually on any result dir with:
```bash
.venv/bin/python experiments/data_efficiency/analyze_eval_failures.py <RESULTS_DIR> --now "$(TZ='America/Los_Angeles' date '+%H:%M %Z')"
```

**Resume instead of redo.** All runners accept `OUT_ROOT` via env and **skip any task that already has a `results_*.json`**. So after you fix the cause, re-run into the SAME dir to re-execute only the failed tasks — completed ones are skipped:
```bash
OUT_ROOT=<existing_result_dir> bash experiments/data_efficiency/run_paloma_for_model.sh <LABEL> <HF_DIR>
# (run_eval_v2.sh already honored OUT_ROOT; paloma/gsm/aryabumi/quac now do too)
```

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

- **Reaching for `--num_processes 1` when mmlu or a large paloma subset crashes with `NCCL Error 1: unhandled cuda error` at `gather_object`. ROOT-CAUSED + FIXED 2026-06-22.** This is an OOM in NCCL's P2P/CUMEM IPC-buffer allocator on A100-40GB during the end-of-task `gather_object` (evaluator.py:677/691), not a fabric fault. The fix `export NCCL_P2P_DISABLE=1` is now baked into `run_eval_v2.sh`, `run_paloma_for_model.sh`, `run_gsm_for_model.sh`, `run_aryabumi_nl_extras.sh`, `run_quac_for_model.sh` — so these tasks now run at FULL 8-GPU speed. Do NOT fall back to single-GPU and do NOT remove the `NCCL_P2P_DISABLE=1` line. (Proven: P2P-on = 8/8 fail, P2P-off = 6/6 pass; validated on 4×600M = 32/32 task-runs.) The SEPARATE 4B-model paloma OOM is a real CUDA OOM — fix that with `BATCH_SIZE=4`, not P2P_DISABLE.
- Calling `eval_intermediate.sh` directly. It skips ~12 §3 tasks.
- Patching a running supervisor with `sed -i` — bash caches the script; you must kill + relaunch.
- Reporting `mbpp` or `humaneval` as `FAILED-CONTINUE` without re-running with `--num_processes 1` (or `convert_and_eval_v2.sh`, which has the per-rank metrics cache fix).
- Updating §3 without recomputing Mean rows.
- **Running paloma with `HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1`.** `allenai/paloma` uses a legacy builder-script that lives on the Hub and is fetched on every `load_dataset` call; offline mode breaks this with `ConnectionError: Couldn't reach 'allenai/paloma' on the Hub (OfflineModeIsEnabled)`. `run_paloma_for_model.sh` now defaults to `OFFLINE=0`. Other tasks (gsm, v2-suite) are fine offline because their datasets ship as plain parquet/json.
- **Trusting `ALL DONE` markers** from runner scripts that swallow per-task failures with `||`. The runner can print `ALL DONE` even when all subsets failed. Always check at least one `<results_dir>/<subset>/*results*.json` exists before extracting; if the dir is empty, grep the `.log` for `Traceback|ConnectionError|FAILED`.
- **Mixing paloma_macro values from Levanter in-training eval and lm-eval-harness.** They disagree by up to +0.05 bpb on average (and +0.55 on twitterAAE). All paloma_macro values in §3 must come from `run_paloma_for_model.sh` (lm-eval-harness), NOT from a wandb-logged Levanter in-training value. If you discover a value that was sourced from Levanter, re-run paloma via `run_paloma_for_model.sh` to get the apples-to-apples number. Discovered 2026-06-12 after C5-v4 paloma 1.093 looked like it beat A5's 1.122 (Levanter-sourced); the apples-to-apples A5 value via lm-eval was actually 1.077 — C5-v4 is close but doesn't beat A5.
- **Running paloma on models bigger than 1.4B at default `batch_size=16`.** 4B model OOMs on 8×40GB GPUs. Set `BATCH_SIZE=4` env var when running `run_paloma_for_model.sh` for the 4B model (the script honors `BATCH_SIZE` since 2026-06-12).
- **Passing `--allow-missing` to `fill-from-results` to bypass strict-fail.** Since 2026-06-15 the command refuse-fails when any v2-suite task is missing a results JSON or its metric key. The strict-fail catches partial v2 failures (mbpp/humaneval cache collision, paloma offline-mode crash, runner-script `||`-swallowed crash). NEVER pass `--allow-missing` just to make the table validate — that hides the bug the check exists to surface. Use it only for an intentional single-task backfill where you understand exactly which task is being filled.
- **Adding a new TASKS row with `runs_in_v2_suite=True` without confirming `run_eval_v2.sh` actually runs it.** Strict-fail will then block on every fill until either the task is added to the v2 script or the TASKS row is marked `runs_in_v2_suite=False` (the right choice for storycloze, cb, quac — they live in the aux runners).

## Why this skill exists

On 2026-06-10–11 I used `eval_intermediate.sh` for the C5-v3 family, then noticed half the §3 cells were `—` because that script skips lambada / copa / wsc / agieval / gpqa / bbh / mmlu_pro / bigcode humaneval / paloma per-subset / gsm_symbolic / gsm_noop. Re-running everything cost an extra ~2 hours of wall time and user frustration. This skill enforces the right script at the gate.
