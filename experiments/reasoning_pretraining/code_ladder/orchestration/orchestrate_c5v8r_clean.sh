#!/usr/bin/env bash
# Orchestrator for C5-v8r-clean overnight (June 17 → June 18 PDT).
#
# Sequence:
#   1. Poll for C5-v8r phase 1 final checkpoint.
#   2. Update PHASE1_INIT_FROM in run_1_4b_c5v8r_phase2.py.
#   3. Launch phase 2 on st-1..4.
#   4. Launch phase 1 endpoint eval on dy-5 (serial v2 + sequential aux).
#
# Logs:
#   /fsx/users/dongweij/marin/logs/orchestrate_c5v8r_clean_<TS>.log

set -uo pipefail
cd /fsx/users/dongweij/marin

MARIN=/fsx/users/dongweij/marin
TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
OLOG=$MARIN/logs/orchestrate_c5v8r_clean_${TS}.log

log() { echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $*" >> "$OLOG"; }

log "=== orchestrator START ==="

# === Step 1: poll for phase 1 final checkpoint ===
log "polling for C5-v8r phase 1 step-14671 checkpoint..."
until ls -d $MARIN/checkpoints/1_4b_c5v8r_phase1/*/step-14671 2>/dev/null | head -1 >/dev/null; do
  sleep 60
done

P1_CKPT=$(ls -d $MARIN/checkpoints/1_4b_c5v8r_phase1/*/step-14671 | head -1)
P1_RUN_ID=$(basename $(dirname "$P1_CKPT"))
log "phase 1 final ckpt: $P1_CKPT (run_id=$P1_RUN_ID)"

# === Step 2: edit phase 2 PHASE1_INIT_FROM ===
P2_SCRIPT=$MARIN/experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_c5v8r_phase2.py
OLD_INIT="checkpoints/1_4b_1ep_c5_code_then_text/7mnu0nch/step-14672"
NEW_INIT="checkpoints/1_4b_c5v8r_phase1/${P1_RUN_ID}/step-14671"
log "editing $P2_SCRIPT: $OLD_INIT -> $NEW_INIT"
sed -i "s|$OLD_INIT|$NEW_INIT|" "$P2_SCRIPT"
grep -n "PHASE1_INIT_FROM" "$P2_SCRIPT" | head -2 >> "$OLOG"

# === Step 3: launch phase 2 on st-1..4 ===
P2_TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
P2_TAG="c5v8r_p2_4n_${P2_TS}"
P2_LL=$MARIN/logs/launcher_${P2_TAG}.log
log "launching phase 2 on st-1..4 TAG=$P2_TAG"
nohup $MARIN/experiments/reasoning_pretraining/code_ladder/orchestration/multi_node_launch.sh \
  --nodes "gpu-st-p4d24xlarge-1,gpu-st-p4d24xlarge-2,gpu-st-p4d24xlarge-3,gpu-st-p4d24xlarge-4" \
  --config experiments/reasoning_pretraining/code_ladder/scripts/run_1_4b_c5v8r_phase2.py \
  --run-tag "$P2_TAG" \
  --coordinator-port 33620 \
  > "$P2_LL" 2>&1 < /dev/null &
disown
echo "$P2_TAG" > /tmp/last_tag_c5v8r_p2.txt
log "phase 2 launcher fired -> $P2_LL"
sleep 15
log "phase 2 launcher tail:"
tail -20 "$P2_LL" >> "$OLOG"

# === Step 4: phase 1 endpoint eval on dy-5 (serial v2 + sequential aux) ===
P1_HF=$MARIN/checkpoints/c5v8r_p1_step14671_hf
P1_LABEL=c5v8r_p1_step14671
V2_TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
V2_LOG=$MARIN/logs/v2_${P1_LABEL}_${V2_TS}.log

log "converting phase 1 ckpt to HF + serial v2 + sequential aux on dy-5"

nohup bash -c "
bash $MARIN/experiments/reasoning_pretraining/code_ladder/eval/convert_and_eval_v2.sh \
  --label $P1_LABEL \
  --src $P1_CKPT \
  --hf-dst $P1_HF \
  --node gpu-dy-p4d24xlarge-5

# wait for v2 ALL DONE marker
while ! grep -q 'ALL DONE' $V2_LOG 2>/dev/null; do sleep 30; done

# now sequential aux runners on dy-5
ssh gpu-dy-p4d24xlarge-5 'bash $MARIN/experiments/reasoning_pretraining/code_ladder/eval/run_paloma_for_model.sh $P1_LABEL $P1_HF' \
  > $MARIN/logs/paloma_${P1_LABEL}_${V2_TS}.log 2>&1
ssh gpu-dy-p4d24xlarge-5 'bash $MARIN/experiments/reasoning_pretraining/code_ladder/eval/run_gsm_for_model.sh $P1_LABEL $P1_HF' \
  > $MARIN/logs/gsm_${P1_LABEL}_${V2_TS}.log 2>&1
ssh gpu-dy-p4d24xlarge-5 'bash $MARIN/experiments/reasoning_pretraining/code_ladder/eval/run_aryabumi_nl_extras.sh $P1_LABEL $P1_HF' \
  > $MARIN/logs/aryabumi_nl_extras_${P1_LABEL}_${V2_TS}.log 2>&1
ssh gpu-dy-p4d24xlarge-5 'bash $MARIN/experiments/reasoning_pretraining/code_ladder/eval/run_quac_for_model.sh $P1_LABEL $P1_HF' \
  > $MARIN/logs/quac_${P1_LABEL}_${V2_TS}.log 2>&1
" > "$V2_LOG" 2>&1 < /dev/null &
disown
log "phase 1 eval dispatched -> $V2_LOG (serial v2 then sequential aux on dy-5)"

log "=== orchestrator END (phase 2 training is live; phase 1 eval running on dy-5) ==="
