#!/usr/bin/env bash
set -uo pipefail
cd /fsx/users/dongweij/marin
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
# Fix the NCCL gather_object P2P/CUMEM IPC-buffer OOM (root-caused 2026-06-22; see run_eval_v2.sh).
export NCCL_P2P_DISABLE=1
LABEL="${1:?LABEL required}"
HF_DST="${2:?HF_DST required}"
INCLUDE_DIR=/fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval
# OUT_ROOT may be passed via env to RESUME into an existing dir.
OUT_ROOT="${OUT_ROOT:-/fsx/users/dongweij/marin/outputs/eval_results/quac_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)}"
OUT="$OUT_ROOT/quac_first_turn"
mkdir -p "$OUT"
if find "$OUT" -name 'results_*.json' 2>/dev/null | grep -q .; then
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL quac SKIP (already has results) → $OUT_ROOT"
  exit 0
fi
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL quac start"
.venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval \
  --include_path "$INCLUDE_DIR" \
  --model hf \
  --model_args "pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True" \
  --tasks quac_first_turn \
  --batch_size 16 \
  --output_path "$OUT" \
  --trust_remote_code > "$OUT.log" 2>&1 || true
# Ground-truth PASS = a results JSON was written (see OPS.md "don't trust ALL DONE").
if find "$OUT" -name 'results_*.json' 2>/dev/null | grep -q .; then
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL quac ALL DONE (1/1 ok) → $OUT_ROOT"
else
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL quac ALL DONE WITH FAILURES (0/1 ok, 1 FAILED: quac_first_turn) → $OUT_ROOT"
  .venv/bin/python experiments/reasoning_pretraining/code_ladder/eval/analyze_eval_failures.py "$OUT_ROOT" --now "$(TZ='America/Los_Angeles' date '+%H:%M %Z')" || true
  exit 1
fi
