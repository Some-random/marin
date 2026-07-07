#!/bin/bash
# Retry mbpp + humaneval for all previously-tracked models that failed under
# the old multi-GPU eval (HF evaluate code_eval cache collision under torchrun).
# Single-GPU per task, 1 model per node, parallel across 8 nodes.
#
# Output: outputs/eval_results/mbpp_he_retry_<LABEL>_<TS>/
#   mbpp_results.json
#   humaneval_results.json
#
# Run: nohup bash retry_mbpp_humaneval_all.sh > logs/mbpp_he_retry_all.log 2>&1 &

set -uo pipefail

# (model_label, hf_path, node) — 11 models distributed across 8 nodes (3 share)
declare -a JOBS=(
  "base_x16          1_4b_wd1_6_x16_nocrossblock_hf  gpu-st-p4d24xlarge-1"
  "code25_v2         1_4b_25code_alg_v2_hf           gpu-st-p4d24xlarge-3"
  "c5v2small_stage1  c5v2_small_stage1_step6400_hf   gpu-st-p4d24xlarge-4"
  "c5v2small_final   c5v2_small_step12799_hf         gpu-dy-p4d24xlarge-1"
  "A5_final          1ep_dclm_final_hf               gpu-dy-p4d24xlarge-2"
  "B4_final          1ep_code25_final_hf             gpu-dy-p4d24xlarge-3"
  "C5_stage1         c5_stage1_step14672_hf          gpu-dy-p4d24xlarge-4"
  "C5_final          c5_final_step29343_hf           gpu-dy-p4d24xlarge-8"
  # Remaining 3 share already-busy nodes after the first 8 finish; serialize them after
  "C5v2_stage1       c5v2_stage1_step14672_hf        gpu-st-p4d24xlarge-1"
  "C5v2_final        c5v2_final_step29343_hf         gpu-st-p4d24xlarge-3"
  "4B_final          4b_dclm_short_final_hf          gpu-st-p4d24xlarge-4"
)

LOG_ROOT=/fsx/users/dongweij/marin/logs
RES_ROOT=/fsx/users/dongweij/marin/outputs/eval_results

log() { echo "[$(TZ='America/Los_Angeles' date '+%Y-%m-%d %H:%M:%S %Z')] $*"; }

run_one() {
  local LABEL=$1
  local HF_DIR=$2
  local NODE=$3
  local TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
  local OUT=$RES_ROOT/mbpp_he_retry_${LABEL}_${TS}
  local LOG=$LOG_ROOT/mbpp_he_retry_${LABEL}_${TS}.log
  mkdir -p $OUT
  log "[$LABEL] launching mbpp+humaneval single-GPU on $NODE → $LOG"
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no $NODE "
    cd /fsx/users/dongweij/marin
    export HF_TOKEN=\$(cat /fsx/users/dongweij/.cache/huggingface/token)
    export HF_ALLOW_CODE_EVAL=1

    for TASK_NSHOT in 'mbpp 3' 'humaneval 0'; do
      set -- \$TASK_NSHOT
      TASK=\$1; NSHOT=\$2
      SUBOUT=$OUT/\${NSHOT}shot__\$TASK
      mkdir -p \$SUBOUT
      echo \"[$LABEL][\$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] \$TASK n-shot=\$NSHOT batch=4 nproc=1\"
      .venv/bin/python -m lm_eval --model hf \
        --model_args 'pretrained=/fsx/users/dongweij/marin/checkpoints/$HF_DIR,dtype=bfloat16,trust_remote_code=True' \
        --tasks \$TASK --num_fewshot \$NSHOT --batch_size 4 \
        --log_samples --output_path \$SUBOUT \
        --include_path /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval \
        --trust_remote_code --confirm_run_unsafe_code \
        > \$SUBOUT.log 2>&1 && echo '  DONE' || echo '  FAILED'
    done
  " > $LOG 2>&1
  log "[$LABEL] finished"
}

# Launch first 8 in parallel (one per unique node)
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
  read LABEL HF_DIR NODE <<< "${JOBS[$i]}"
  run_one $LABEL $HF_DIR $NODE &
  PIDS+=($!)
done
log "first batch of 8 launched in parallel: ${PIDS[*]}"

# Wait for first 8
for p in "${PIDS[@]}"; do
  wait $p
done
log "first 8 finished"

# Launch remaining 3 (reusing first 3 nodes)
PIDS2=()
for i in 8 9 10; do
  read LABEL HF_DIR NODE <<< "${JOBS[$i]}"
  run_one $LABEL $HF_DIR $NODE &
  PIDS2+=($!)
done
log "remaining 3 launched: ${PIDS2[*]}"

for p in "${PIDS2[@]}"; do
  wait $p
done
log "all 11 models done. Results under $RES_ROOT/mbpp_he_retry_*"
