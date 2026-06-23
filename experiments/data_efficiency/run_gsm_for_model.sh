#!/usr/bin/env bash
# Run gsm_symbolic_main + gsm_noop for one model on one node.
# Uses local YAML task configs under experiments/data_efficiency/.
# Usage: run_gsm_for_model.sh <LABEL> <HF_DIR>
# Designed to run inside an ssh session on the eval node.

set -uo pipefail
cd /fsx/users/dongweij/marin
export HF_TOKEN=$(cat /fsx/users/dongweij/.cache/huggingface/token)
export HF_DATASETS_CACHE=/fsx/users/dongweij/marin/outputs/hf_cache/datasets
# Allow live HF Hub access for apple/GSM-Symbolic (offline cache may not have it).
export HF_DATASETS_OFFLINE=0
export HF_HUB_OFFLINE=0
# Fix the NCCL gather_object P2P/CUMEM IPC-buffer OOM (root-caused 2026-06-22; see run_eval_v2.sh).
export NCCL_P2P_DISABLE=1

LABEL="${1:?LABEL required}"
HF_DST="${2:?HF_DST required}"
INCLUDE_DIR=/fsx/users/dongweij/marin/experiments/data_efficiency

# OUT_ROOT may be passed via env to RESUME into an existing dir — already-done tasks are skipped.
OUT_ROOT="${OUT_ROOT:-/fsx/users/dongweij/marin/outputs/eval_results/gsm_${LABEL}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)}"
mkdir -p "$OUT_ROOT"
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL gsm → $OUT_ROOT"

GSM_TASKS=(gsm_symbolic_main gsm_noop)
FAILED_TASKS=()

for T in "${GSM_TASKS[@]}"; do
  OUT="$OUT_ROOT/${T}"
  mkdir -p "$OUT"
  if find "$OUT" -name 'results_*.json' 2>/dev/null | grep -q .; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T SKIP (already has results)"
    continue
  fi
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T start"
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 -m lm_eval \
    --include_path "$INCLUDE_DIR" \
    --model hf \
    --model_args "pretrained=$HF_DST,dtype=bfloat16,trust_remote_code=True" \
    --tasks "$T" \
    --batch_size 16 \
    --output_path "$OUT" \
    --log_samples \
    --trust_remote_code > "$OUT.log" 2>&1 || true
  # Ground-truth PASS = a results JSON was written (see OPS.md "don't trust ALL DONE").
  if find "$OUT" -name 'results_*.json' 2>/dev/null | grep -q .; then
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T DONE"
  else
    echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL $T FAILED-CONTINUE"
    FAILED_TASKS+=("$T")
  fi
done

NTOTAL=${#GSM_TASKS[@]}
NF=${#FAILED_TASKS[@]}
if [ "$NF" -eq 0 ]; then
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL gsm ALL DONE ($NTOTAL/$NTOTAL ok) → $OUT_ROOT"
else
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $LABEL gsm ALL DONE WITH FAILURES ($((NTOTAL-NF))/$NTOTAL ok, $NF FAILED: ${FAILED_TASKS[*]}) → $OUT_ROOT"
  .venv/bin/python experiments/data_efficiency/analyze_eval_failures.py "$OUT_ROOT" --now "$(TZ='America/Los_Angeles' date '+%H:%M %Z')" || true
  exit 1
fi
