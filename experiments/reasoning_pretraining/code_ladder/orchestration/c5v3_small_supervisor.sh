#!/bin/bash
# C5-v3-small DAG supervisor (v3, single-node).
#
# Pipeline:
#   1. Launch phase 1 on $TRAIN_NODE (1 node × 8 GPUs, batch=64, per_device=8).
#   2. Wait for phase 1 step-6399 checkpoint.
#   3. Fan out: phase 1 eval on $EVAL_NODE (parallel) + phase 2 train on $TRAIN_NODE.
#   4. Wait for phase 2 step-6399 checkpoint.
#   5. Phase 2 eval on $EVAL_NODE.
#
# Config matches C5-v2-small exactly: batch=64, ~3.36 B tokens total.
#
# Runs as a long-lived background process. Kill with:
#   pkill -f c5v3_small_supervisor.sh

set -uo pipefail

TRAIN_NODE=${TRAIN_NODE:-gpu-dy-p4d24xlarge-5}
EVAL_NODE=${EVAL_NODE:-gpu-st-p4d24xlarge-2}
P1_BASE=/fsx/users/dongweij/marin/checkpoints/1_4b_c5v3_small_phase1
P2_BASE=/fsx/users/dongweij/marin/checkpoints/1_4b_c5v3_small_phase2
P1_STEP=6399
P2_STEP=6399
LOG_ROOT=/fsx/users/dongweij/marin/logs

log() {
  echo "[$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

# === Phase 1 launch (1 node) ===
TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
P1_LOG=$LOG_ROOT/c5v3-small-phase1-${TS}.log
log "phase 1 launching on $TRAIN_NODE (batch=64) → $P1_LOG"

ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $TRAIN_NODE "
  cd /fsx/users/dongweij/marin
  export JAX_DIST_NUM_PROCESSES=1
  export JAX_DIST_PROCESS_ID=0
  export JAX_DIST_COORDINATOR=\$(hostname -I | awk '{print \$1}'):33360
  export WANDB_MODE=offline
  nohup .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.scripts.run_1_4b_c5v3_small_phase1 > $P1_LOG 2>&1 < /dev/null &
  disown
"

log "waiting for phase 1 step-$P1_STEP checkpoint..."
while true; do
  found=$(ls -d $P1_BASE/*/step-$P1_STEP 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    log "phase 1 checkpoint detected: $found"
    P1_CKPT=$found
    break
  fi
  if ! ssh -o ConnectTimeout=3 $TRAIN_NODE "pgrep -f run_1_4b_c5v3_small_phase1 >/dev/null" 2>/dev/null; then
    log "WARN: phase 1 python gone but no step-$P1_STEP checkpoint — crash. Halting."
    log "  inspect: tail $P1_LOG"
    exit 1
  fi
  sleep 60
done

# === Fan out: phase 1 eval (parallel) + phase 2 train ===
P1_EVAL_LOG=$LOG_ROOT/c5v3-small-phase1-eval-${TS}.log
log "fanning out: phase 1 eval on $EVAL_NODE + phase 2 train on $TRAIN_NODE"

nohup bash /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval/convert_and_eval_v2.sh \
  --label c5v3-small-phase1-step$P1_STEP \
  --src $P1_CKPT \
  --hf-dst /fsx/users/dongweij/marin/checkpoints/c5v3_small_phase1_step${P1_STEP}_hf \
  --node $EVAL_NODE \
  > $P1_EVAL_LOG 2>&1 < /dev/null &
log "phase 1 eval launched (PID $!) → $P1_EVAL_LOG"
disown

TS2=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
P2_LOG=$LOG_ROOT/c5v3-small-phase2-${TS2}.log
log "phase 2 launching on $TRAIN_NODE → $P2_LOG"
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $TRAIN_NODE "
  cd /fsx/users/dongweij/marin
  export JAX_DIST_NUM_PROCESSES=1
  export JAX_DIST_PROCESS_ID=0
  export JAX_DIST_COORDINATOR=\$(hostname -I | awk '{print \$1}'):33361
  export WANDB_MODE=offline
  export C5V3_SMALL_PHASE1_CKPT=$P1_CKPT
  nohup .venv/bin/python -m experiments.reasoning_pretraining.code_ladder.scripts.run_1_4b_c5v3_small_phase2 > $P2_LOG 2>&1 < /dev/null &
  disown
"

log "waiting for phase 2 step-$P2_STEP checkpoint..."
while true; do
  found=$(ls -d $P2_BASE/*/step-$P2_STEP 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    log "phase 2 checkpoint detected: $found"
    P2_CKPT=$found
    break
  fi
  if ! ssh -o ConnectTimeout=3 $TRAIN_NODE "pgrep -f run_1_4b_c5v3_small_phase2 >/dev/null" 2>/dev/null; then
    log "WARN: phase 2 python gone but no step-$P2_STEP checkpoint — crash. Halting."
    exit 1
  fi
  sleep 60
done

# === Phase 2 eval (sequential, after phase 1 eval likely done) ===
TS3=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
P2_EVAL_LOG=$LOG_ROOT/c5v3-small-phase2-eval-${TS3}.log
log "phase 2 eval launching on $EVAL_NODE → $P2_EVAL_LOG"
nohup bash /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval/convert_and_eval_v2.sh \
  --label c5v3-small-phase2-step$P2_STEP \
  --src $P2_CKPT \
  --hf-dst /fsx/users/dongweij/marin/checkpoints/c5v3_small_phase2_step${P2_STEP}_hf \
  --node $EVAL_NODE \
  > $P2_EVAL_LOG 2>&1 < /dev/null &
log "phase 2 eval launched (PID $!)"
disown

log "supervisor done. Phase 2 eval continues in background."
