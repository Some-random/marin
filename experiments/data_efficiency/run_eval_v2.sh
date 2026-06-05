#!/usr/bin/env bash
# Improved eval runner — uses shared /fsx HF_DATASETS_CACHE + per-rank
# HF_METRICS_CACHE wrapper. Fixes:
#   (a) HF 504 transient failures on cais/mmlu, EleutherAI/hendrycks_math,
#       SaylorTwift/bbh — data is local, set HF_DATASETS_OFFLINE=1.
#   (b) code_eval cache race on multi-GPU mbpp/humaneval — wrapper sets
#       HF_METRICS_CACHE=/tmp/hf_metrics_rank_$LOCAL_RANK per worker.
#
# Usage: run_eval_v2.sh <LABEL> <HF_DIR>
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

LABEL="${1:?LABEL required}"
HF_DST="${2:?HF_DST required}"
OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/v2_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
WRAPPER=/fsx/users/dongweij/marin/outputs/lm_eval_wrapper.py
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL eval suite → $OUT_ROOT"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

run_lm_eval_one() {
  # Single attempt at a specific batch size. Returns 0 on success, 1 on failure.
  local TASKS="$1" NSHOT="$2" BATCH="$3" EXTRA="$4" OUT="$5"
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] tasks=$TASKS n-shot=$NSHOT batch=$BATCH attempt"
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    "$WRAPPER" --model hf \
    --model_args "pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$TASKS" --num_fewshot "$NSHOT" --batch_size "$BATCH" \
    --log_samples --output_path "$OUT" \
    --include_path /fsx/users/dongweij/marin/experiments/data_efficiency \
    --trust_remote_code \
    $EXTRA >> "$OUT.log" 2>&1
}

run_lm_eval() {
  # Wrapper that auto-retries with halved batch on OOM. Stops at batch=1.
  local TASKS="$1" NSHOT="$2" BATCH="$3" EXTRA="$4"
  local OUT="$OUT_ROOT/${NSHOT}shot__$(echo "$TASKS" | tr ',' '_' | cut -c1-30)"
  mkdir -p "$OUT"
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
  return 1
}

# Standard suite — same task set + nshot as the previous final eval.
run_lm_eval "arc_easy,arc_challenge" 25 16 ""
run_lm_eval "hellaswag" 10 16 ""
run_lm_eval "winogrande,gsm8k" 5 16 ""
run_lm_eval "mmlu" 5 16 ""
run_lm_eval "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" 0 16 ""
run_lm_eval "openbookqa_fact" 0 16 ""
run_lm_eval "gsm8k_cot" 8 8 "--confirm_run_unsafe_code"
# code_eval tasks now safe multi-GPU thanks to per-rank metrics cache.
run_lm_eval "mbpp" 3 8 "--confirm_run_unsafe_code"
run_lm_eval "humaneval" 0 8 "--confirm_run_unsafe_code"
run_lm_eval "minerva_math" 4 8 ""
run_lm_eval "lambada_openai,copa,wsc,agieval_lsat_ar" 0 16 ""
run_lm_eval "gpqa" 0 8 ""
run_lm_eval "bbh" 3 8 "--limit 0.1"
run_lm_eval "mmlu_pro" 5 8 "--limit 0.1"

# bigcode HumanEval — same env vars apply, openai_humaneval is in shared cache.
BC_OUT="$OUT_ROOT/bigcode_humaneval"
mkdir -p "$BC_OUT"
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
  --trust_remote_code > "$BC_OUT/eval.log" 2>&1 && \
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] bigcode humaneval DONE" || \
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] bigcode humaneval FAILED-CONTINUE"

echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] [$LABEL] ALL DONE → $OUT_ROOT"
