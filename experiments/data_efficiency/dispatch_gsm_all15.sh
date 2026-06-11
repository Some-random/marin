#!/usr/bin/env bash
# Dispatch gsm_symbolic_main + gsm_noop on all 15 internal §3 models across
# 6 free nodes. Each node runs N models sequentially.
# Launches all 6 ssh pipes in parallel; the orchestrator process exits.

set -uo pipefail
cd /fsx/users/dongweij/marin
TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
LOG_DIR=/fsx/users/dongweij/marin/logs

# Each entry: NODE:LABEL=HF_DIR LABEL=HF_DIR ...
# 15 models split 3-3-3-2-2-2 across 6 nodes.
declare -A PLAN=(
  [gpu-st-p4d24xlarge-2]="base=1_4b_wd1_6_x16_nocrossblock_hf code25_v2=1_4b_25code_alg_v2_hf c5v2_small_stage1=c5v2_small_stage1_step6400_hf"
  [gpu-dy-p4d24xlarge-2]="c5v2_small_final=c5v2_small_step12799_hf a5_final=1ep_dclm_final_hf b4_final=1ep_code25_final_hf"
  [gpu-dy-p4d24xlarge-3]="c5_stage1=c5_stage1_step14672_hf c5v2_stage1=c5v2_stage1_step14672_hf c5_final=c5_final_step29343_hf"
  [gpu-dy-p4d24xlarge-4]="c5v2_final=c5v2_final_step29343_hf c5v3_phase1=c5v3_phase1_step14671_hf"
  [gpu-dy-p4d24xlarge-5]="c5v3_final=c5v3_p2_a6_step14671_hf c5v3_small_phase1=c5v3_small_phase1_step6399_hf"
  [gpu-dy-p4d24xlarge-8]="c5v3_small_final=c5v3_small_phase2_step6399_hf 4b_final=4b_dclm_short_final_hf"
)

for NODE in "${!PLAN[@]}"; do
  LOG="$LOG_DIR/gsm_all15_${NODE##*-}_${TS}.log"
  PLAN_STR="${PLAN[$NODE]}"
  echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] dispatching to $NODE → $LOG"
  ssh -o ConnectTimeout=10 "$NODE" "
    set -uo pipefail
    cd /fsx/users/dongweij/marin
    echo \"[\$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $NODE GSM-15 batch START on \$(hostname)\"
    for ENTRY in $PLAN_STR; do
      LABEL=\${ENTRY%%=*}
      HF=\${ENTRY##*=}
      HF_DIR=/fsx/users/dongweij/marin/checkpoints/\$HF
      if [ ! -d \"\$HF_DIR\" ]; then
        echo \"[\$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $NODE \$LABEL MISSING-HF-DIR \$HF_DIR — SKIP\"
        continue
      fi
      bash /fsx/users/dongweij/marin/experiments/data_efficiency/run_gsm_for_model.sh \"\$LABEL\" \"\$HF_DIR\"
    done
    echo \"[\$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] $NODE GSM-15 batch ALL DONE\"
  " > "$LOG" 2>&1 < /dev/null &
  disown
  sleep 1
done

echo ""
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] All 6 ssh pipes dispatched. Logs under $LOG_DIR/gsm_all15_*_${TS}.log"
echo "TIMESTAMP=$TS"
