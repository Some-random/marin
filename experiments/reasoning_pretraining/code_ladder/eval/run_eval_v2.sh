#!/usr/bin/env bash
# Improved eval runner — uses shared /fsx HF_DATASETS_CACHE + per-rank
# HF_METRICS_CACHE wrapper. Fixes:
#   (a) HF 504 transient failures on cais/mmlu, EleutherAI/hendrycks_math,
#       SaylorTwift/bbh — data is local, set HF_DATASETS_OFFLINE=1.
#   (b) code_eval cache race on multi-GPU mbpp/humaneval — wrapper sets
#       HF_METRICS_CACHE=/tmp/hf_metrics_rank_$LOCAL_RANK per worker.
#
# Usage:
#   run_eval_v2.sh <LABEL> <HF_DIR>             (run all 15 task groups, ~67 min)
#   run_eval_v2.sh <LABEL> <HF_DIR> <SHARD>     (run a shard: A/B/C/D, ~16-19 min)
#
# When invoked with a SHARD arg, the dispatcher driver
# (run_eval_v2_sharded.sh) MUST pass OUT_ROOT via env so all 4 shards
# write into the same timestamped results dir. Without the env var the
# script generates its own OUT_ROOT (back-compat for single-node mode).
set -uo pipefail
cd /fsx/users/dongweij/marin
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_ALLOW_CODE_EVAL=1
# Shared cache on /fsx — same across all nodes.
export HF_DATASETS_CACHE=/fsx/users/dongweij/marin/outputs/hf_cache/datasets
# Offline mode: skip HF metadata calls, use the local cache. Robust to
# HF backend outages.
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# NCCL P2P/CUMEM IPC-buffer OOM fix (root-caused 2026-06-22). lm-eval's end-of-task
# torch.distributed.gather_object builds a per-peer P2P/CUMEM gather channel to rank 0 and
# cudaMallocs shareable IPC buffers; on A100-40GB that intermittently OOMs and surfaces as the
# misleading "NCCL Error 1: unhandled cuda error" at evaluator.py:677/691 — killing mmlu (57
# subtasks = biggest gather) and large paloma subsets. Disabling P2P routes the gather off the
# CUMEM IPC path and fixes it (verified: P2P on = 8/8 fail, P2P off = 6/6 pass) at full 8-GPU
# speed. Eval is data-parallel inference + one gather, so P2P perf is irrelevant here.
export NCCL_P2P_DISABLE=1

LABEL="${1:?LABEL required}"
HF_DST="${2:?HF_DST required}"
SHARD="${3:-all}"   # all | A | B | C | D

# OUT_ROOT may be passed via env by the sharded dispatcher so all 4 shards
# write into the same dir. Otherwise generate a new timestamped path.
if [ -z "${OUT_ROOT:-}" ]; then
  OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/v2_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
fi
mkdir -p "$OUT_ROOT"
WRAPPER=/fsx/users/dongweij/marin/outputs/lm_eval_wrapper.py
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL eval suite → $OUT_ROOT"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

# Track task-groups that did NOT produce a result, so the final marker can't lie
# ("ALL DONE" must never print clean when a task crashed). See OPS.md "don't trust ALL DONE".
FAILED_TASKS=()

run_lm_eval_one() {
  # Single attempt at a specific batch size. Returns 0 on success, 1 on failure.
  local TASKS="$1" NSHOT="$2" BATCH="$3" EXTRA="$4" OUT="$5"
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS n-shot=$NSHOT batch=$BATCH attempt"
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    "$WRAPPER" --model hf \
    --model_args "pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" --num_fewshot "$NSHOT" --batch_size "$BATCH" \
    --log_samples --output_path "$OUT" \
    --include_path /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval \
    --trust_remote_code \
    $EXTRA >> "$OUT.log" 2>&1
}

run_lm_eval() {
  # Wrapper that auto-retries with halved batch on OOM. Stops at batch=1.
  local TASKS="$1" NSHOT="$2" BATCH="$3" EXTRA="$4"
  local OUT="$OUT_ROOT/${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  mkdir -p "$OUT"
  if find "$OUT" -name 'results_*.json' 2>/dev/null | grep -q .; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS SKIP (already has results)" | tee -a "$OUT.log"
    return 0
  fi
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS n-shot=$NSHOT batch=$BATCH start" | tee -a "$OUT.log"
  local cur="$BATCH"
  while [ "$cur" -ge 1 ]; do
    if run_lm_eval_one "$TASKS" "$NSHOT" "$cur" "$EXTRA" "$OUT"; then
      echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS DONE at batch=$cur" | tee -a "$OUT.log"
      return 0
    fi
    # Decide whether to retry. Only retry on CUDA OOM; not on HF 504 etc.
    if grep -q "OutOfMemoryError\|RESOURCE_EXHAUSTED" "$OUT.log" 2>/dev/null && [ "$cur" -gt 1 ]; then
      local nxt=$(( cur / 2 ))
      [ "$nxt" -lt 1 ] && nxt=1
      echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS OOM at batch=$cur, retrying batch=$nxt" | tee -a "$OUT.log"
      cur="$nxt"
      continue
    fi
    break
  done
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS FAILED-CONTINUE (last batch=$cur)" | tee -a "$OUT.log"
  FAILED_TASKS+=("$TASKS")
  return 1
}

# bigcode HumanEval (separate venv) — packaged as a function so shards can call it.
run_bigcode_humaneval() {
  local BC_OUT="$OUT_ROOT/bigcode_humaneval"
  mkdir -p "$BC_OUT"
  if [ -s "$BC_OUT/metrics.json" ]; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] bigcode humaneval SKIP (already has metrics)"
    return 0
  fi
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] bigcode humaneval start"
  .venv_bigcode/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    /fsx/users/dongweij/marin/bigcode-evaluation-harness/main.py \
    --model "$HF_DST" \
    --tasks humaneval \
    --max_length_generation 512 \
    --temperature 0.0 \
    --do_sample False \
    --n_samples 1 \
    --batch_size 1 \
    --precision bf16 \
    --allow_code_execution \
    --save_generations \
    --save_generations_path "$BC_OUT/generations.json" \
    --metric_output_path "$BC_OUT/metrics.json" \
    --trust_remote_code > "$BC_OUT/eval.log" 2>&1 || true
  # Ground-truth PASS = metrics.json was written (see OPS.md "don't trust ALL DONE").
  if [ -s "$BC_OUT/metrics.json" ]; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] bigcode humaneval DONE"
  else
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] bigcode humaneval FAILED-CONTINUE"
    FAILED_TASKS+=("humaneval_bigcode")
  fi
}

# Task groups grouped by SHARD. Balanced from observed C5-v6-NEW v7 timings
# (each shard ~16-19 min vs 67 min serial).
# Shard A (~16 min): mmlu_pro [13] + openbookqa_fact [1] + lambada group [2]
# Shard B (~16 min): minerva_math [12] + arc [4]
# Shard C (~17 min): gpqa [7] + bbh [6] + hellaswag [4]
# Shard D (~19 min): bigcode [3] + humaneval [3] + mbpp [2] + gsm8k_cot [3] +
#                    mmlu [3] + winogrande+gsm8k [3] + piqa group [2]
shard_A() {
  run_lm_eval "mmlu_pro" 5 8 "--limit 0.1"
  run_lm_eval "openbookqa_fact" 0 16 ""
  run_lm_eval "lambada_openai,copa,wsc,agieval_lsat_ar" 0 16 ""
}
shard_B() {
  run_lm_eval "minerva_math" 4 8 ""
  run_lm_eval "arc_easy,arc_challenge" 25 16 ""
}
shard_C() {
  run_lm_eval "gpqa" 0 8 ""
  run_lm_eval "bbh" 3 8 "--limit 0.1"
  run_lm_eval "hellaswag" 10 16 ""
}
shard_D() {
  run_bigcode_humaneval
  run_lm_eval "humaneval" 0 8 "--confirm_run_unsafe_code"
  run_lm_eval "mbpp" 3 8 "--confirm_run_unsafe_code"
  run_lm_eval "gsm8k_cot" 8 8 "--confirm_run_unsafe_code"
  run_lm_eval "mmlu" 5 16 ""
  run_lm_eval "winogrande,gsm8k" 5 16 ""
  run_lm_eval "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" 0 16 ""
}

case "$SHARD" in
  A) shard_A ;;
  B) shard_B ;;
  C) shard_C ;;
  D) shard_D ;;
  all)
    shard_A
    shard_B
    shard_C
    shard_D
    ;;
  *) echo "unknown SHARD '$SHARD' — must be A/B/C/D/all" >&2; exit 2 ;;
esac

NF=${#FAILED_TASKS[@]}
if [ "$SHARD" = "all" ]; then
  if [ "$NF" -eq 0 ]; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] ALL DONE (0 failures) → $OUT_ROOT"
  else
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] ALL DONE WITH FAILURES ($NF task-group(s) FAILED: ${FAILED_TASKS[*]}) → $OUT_ROOT"
  fi
else
  if [ "$NF" -eq 0 ]; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] SHARD $SHARD DONE (0 failures) → $OUT_ROOT"
  else
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] SHARD $SHARD DONE WITH FAILURES ($NF FAILED: ${FAILED_TASKS[*]}) → $OUT_ROOT"
  fi
fi
if [ "$NF" -ne 0 ]; then
  # Triage the failures (writes FAILURES.md + prints why each died) — fix, don't blind-retry.
  .venv/bin/python experiments/reasoning_pretraining/code_ladder/eval/analyze_eval_failures.py "$OUT_ROOT" --now "$(TZ='America/Los_Angeles' date '+%H:%M %Z')" || true
  exit 1
fi
