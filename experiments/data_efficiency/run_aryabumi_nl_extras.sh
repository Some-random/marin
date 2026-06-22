#!/usr/bin/env bash
# Run Aryabumi NL Reasoning tasks NOT in our v2 suite: storycloze_2018_local + cb (super_glue).
# Designed to run via ssh on a free node.
# Usage: run_aryabumi_nl_extras.sh <LABEL> <HF_DIR>

set -uo pipefail
cd /fsx/users/dongweij/marin
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
# Fix the NCCL gather_object P2P/CUMEM IPC-buffer OOM (root-caused 2026-06-22; see run_eval_v2.sh).
export NCCL_P2P_DISABLE=1

LABEL="${1:?LABEL required}"
HF_DST="${2:?HF_DST required}"
INCLUDE_DIR=/fsx/users/dongweij/marin/experiments/data_efficiency

OUT_ROOT=/fsx/users/dongweij/marin/outputs/eval_results/aryabumi_nl_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT"
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL aryabumi-nl → $OUT_ROOT"

# storycloze_2018_local: 0-shot, MC2 sentence-completion. Custom YAML at INCLUDE_DIR.
# cb: super_glue/CB 0-shot, 3-way NLI MC. Native lm-eval task.
TASKS=(storycloze_2018_local cb)

for T in "${TASKS[@]}"; do
  OUT="$OUT_ROOT/${T}"
  mkdir -p "$OUT"
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T start"
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval \
    --include_path "$INCLUDE_DIR" \
    --model hf \
    --model_args "pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$T" \
    --batch_size 16 \
    --output_path "$OUT" \
    --log_samples \
    --trust_remote_code > "$OUT.log" 2>&1 \
    && echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T DONE" \
    || echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T FAILED-CONTINUE"
done

echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL aryabumi-nl ALL DONE → $OUT_ROOT"
