#!/usr/bin/env bash
# Audit re-eval campaign: re-run v2-suite for 12 §3 columns on one node sequentially.
# Goal: verify each existing §3 column under the current canonical pipeline (post-bugfix).
# Output: results dirs under outputs/eval_results/v2_<LABEL>_AUDIT_<TS>/.

set -uo pipefail

EVAL_NODE="${EVAL_NODE:-gpu-dy-p4d24xlarge-5}"
MARIN=/fsx/users/dongweij/marin
QUEUE_LOG=$MARIN/logs/audit_campaign_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S).log

# Format: LABEL HF_CHECKPOINT_DIRNAME
COLS=(
  "C5_stage1_AUDIT          c5_stage1_step14672_hf"
  "C5v2_stage1_AUDIT        c5v2_stage1_step14672_hf"
  "C5v2_final_AUDIT         c5v2_final_step29343_hf"
  "C5v2_small_stage1_AUDIT  c5v2_small_stage1_step6400_hf"
  "C5v2_small_final_AUDIT   c5v2_small_step12799_hf"
  "C5v3_phase1_AUDIT        c5v3_phase1_step14671_hf"
  "C5v3_final_AUDIT         c5v3_p2_a6_step14671_hf"
  "C5v3_small_phase1_AUDIT  c5v3_small_phase1_step6399_hf"
  "C5v3_small_phase2_AUDIT  c5v3_small_phase2_step6399_hf"
  "C5v6_final_AUDIT         c5v6_phase2_step14671_hf"
  "C5V7_final_AUDIT         c5v7_final_hf"
  "4Bfinal_AUDIT            4b_dclm_short_final_hf"
)

mkdir -p "$MARIN/logs" "$MARIN/outputs/eval_results"

log() { echo "[$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$QUEUE_LOG"; }

log "=== Audit campaign START — node=$EVAL_NODE, ${#COLS[@]} columns ==="

i=0
for entry in "${COLS[@]}"; do
  i=$((i+1))
  read -r label ckpt <<<"$entry"
  TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
  ITEM_LOG=$MARIN/logs/audit_v2_${label}_${TS}.log
  HF_DIR=$MARIN/checkpoints/${ckpt}

  if [ ! -d "$HF_DIR" ]; then
    log "  [${i}/${#COLS[@]}] SKIP $label — missing ${HF_DIR}"
    continue
  fi

  log "=== [${i}/${#COLS[@]}] START $label ($ckpt) on $EVAL_NODE ==="

  # convert_and_eval_v2.sh launches its own nohup. We invoke it foreground here
  # so this loop blocks until ALL DONE. The script auto-skips convert when src
  # is already an HF dir (path ending in _hf).
  bash "$MARIN/experiments/reasoning_pretraining/code_ladder/eval/convert_and_eval_v2.sh" \
    --label "$label" \
    --src "$HF_DIR" \
    --hf-dst "$HF_DIR" \
    --node "$EVAL_NODE" \
    > "$ITEM_LOG" 2>&1
  rc=$?

  if grep -q "ALL DONE" "$ITEM_LOG"; then
    log "  [${i}/${#COLS[@]}] DONE $label (rc=$rc)"
  else
    log "  [${i}/${#COLS[@]}] FAILED-or-no-ALL-DONE $label (rc=$rc) — see $ITEM_LOG"
  fi
done

log "=== Audit campaign FINISHED — see $QUEUE_LOG ==="
