#!/bin/bash
# C5-v3 phase 2 (hero run, 8n) DAG supervisor.
#
# This run was launched independently (multi_node_launch.sh c5v3-p2-a6-8n).
# Supervisor just polls for the final step-14671 checkpoint and then kicks off
# the v2 eval on a node freed by the training ending.
#
# Pipeline:
#   1. Poll $P2_BASE/85ip8s5o/step-14671 every 60s.
#   2. On detection, kick off eval_intermediate.sh on $EVAL_NODE.
#
# Note: the 8 training nodes free up automatically when the python process
# exits, so any of them is a valid EVAL_NODE choice. Default st-1.
#
# Runs as a long-lived background process. Kill with:
#   pkill -f c5v3_p2_supervisor.sh

set -uo pipefail

P2_BASE=/fsx/users/dongweij/marin/checkpoints/1_4b_c5v3_phase2
P2_RUN_ID=${P2_RUN_ID:-85ip8s5o}
P2_FINAL_STEP=14671
EVAL_NODE=${EVAL_NODE:-gpu-st-p4d24xlarge-1}
LOG_ROOT=/fsx/users/dongweij/marin/logs

log() {
  echo "[$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

log "watching for c5v3-p2-a6 final checkpoint at $P2_BASE/$P2_RUN_ID/step-$P2_FINAL_STEP..."

# Poll for the final step checkpoint
while true; do
  if [ -d "$P2_BASE/$P2_RUN_ID/step-$P2_FINAL_STEP" ]; then
    log "phase 2 final checkpoint detected at step-$P2_FINAL_STEP"
    P2_CKPT="$P2_BASE/$P2_RUN_ID/step-$P2_FINAL_STEP"
    break
  fi
  sleep 60
done

# Wait an extra 30s for GPU memory to free on the eval node + processes to fully exit
sleep 30

TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
EVAL_LOG=$LOG_ROOT/c5v3-p2-a6-eval-${TS}.log
log "launching c5v3-p2-a6 v2 eval on $EVAL_NODE → $EVAL_LOG"

bash /fsx/users/dongweij/marin/experiments/data_efficiency/convert_and_eval_v2.sh \
  --label c5v3-p2-a6-step$P2_FINAL_STEP \
  --src $P2_CKPT \
  --hf-dst /fsx/users/dongweij/marin/checkpoints/c5v3_p2_a6_step${P2_FINAL_STEP}_hf \
  --node $EVAL_NODE \
  > $EVAL_LOG 2>&1

log "c5v3-p2-a6 eval done. results in /fsx/users/dongweij/marin/outputs/eval_results/intermediate_c5v3-p2-a6-step${P2_FINAL_STEP}_*"
log "supervisor done."
