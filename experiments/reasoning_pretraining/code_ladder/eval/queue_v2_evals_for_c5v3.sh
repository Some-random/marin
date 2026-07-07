#!/bin/bash
# Wait for mbpp/he retries to finish on st-1 + st-3, then launch v2-suite evals
# for c5v3 phase 1 (on st-1) and c5v3-small phase 1 (on st-3).
#
# Hero is already running on st-2 (launched at 22:39 PDT).

set -uo pipefail
cd /fsx/users/dongweij/marin

log() { echo "[$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

wait_node_free() {
  local NODE=$1
  log "waiting for $NODE to be free (8 MiB GPU mem)..."
  while true; do
    local mem=$(timeout 5 ssh -o ConnectTimeout=3 $NODE "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc" 2>/dev/null)
    if [ "${mem:-100000}" -lt 1000 ]; then
      log "  $NODE free (mem=${mem})"
      return 0
    fi
    sleep 30
  done
}

launch_eval() {
  local LABEL=$1
  local HF_DIR=$2
  local NODE=$3
  local TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
  local LOG=/fsx/users/dongweij/marin/logs/v2_${LABEL}_${TS}.log
  log "launching v2-suite eval [$LABEL] on $NODE → $LOG"
  nohup ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $NODE "
    cd /fsx/users/dongweij/marin
    bash experiments/reasoning_pretraining/code_ladder/eval/run_eval_v2.sh $LABEL /fsx/users/dongweij/marin/checkpoints/$HF_DIR
  " > $LOG 2>&1 < /dev/null &
  disown
  log "  launched (PID $!)"
}

wait_node_free gpu-st-p4d24xlarge-1
launch_eval c5v3_phase1_step14671 c5v3_phase1_step14671_hf gpu-st-p4d24xlarge-1
sleep 5

wait_node_free gpu-st-p4d24xlarge-3
launch_eval c5v3_small_phase1_step6399 c5v3_small_phase1_step6399_hf gpu-st-p4d24xlarge-3

log "both queued. Watch logs/v2_c5v3_phase1_*.log + logs/v2_c5v3_small_phase1_*.log"
