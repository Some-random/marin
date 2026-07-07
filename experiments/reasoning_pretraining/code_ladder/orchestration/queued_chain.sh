#!/usr/bin/env bash
# Wait for the running step-10000 eval suite (pid in $STEP10000_PID), then
# chain:
#   1. wd=3.2/x16 eval suite (both step-10000 and final)
#   2. Re-run the failed social_iqa-containing batch for baseline/v1/v2 step-10000
#      with --trust_remote_code (failed earlier due to new HF gating)
#   3. wd=1.6/x8 standalone training (~5h on 8x A100-40GB)

cd /fsx/users/dongweij/marin

STEP10000_PID="${STEP10000_PID:-}"
if [ -n "$STEP10000_PID" ]; then
  echo "=== waiting for step-10000 eval pid $STEP10000_PID ==="
  while kill -0 "$STEP10000_PID" 2>/dev/null; do sleep 30; done
  echo "=== step-10000 eval pid $STEP10000_PID exited at $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
fi

echo "=== 1. wd=3.2/x16 eval suite start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
bash /fsx/users/dongweij/marin/experiments/reasoning_pretraining/code_ladder/eval/run_wd3_2_x16_evals.sh
echo "=== 1. wd=3.2/x16 eval suite DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="

echo "=== 2. social_iqa rerun (failed earlier on baseline/v1/v2 step-10000) start ==="
OUT_ROOT_RERUN=/fsx/users/dongweij/marin/outputs/eval_results/step10000_siqa_rerun_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M)
mkdir -p "$OUT_ROOT_RERUN"
for LABEL in baseline_step10000 v1_step10000 v2_step10000; do
  CKPT=/fsx/users/dongweij/marin/checkpoints/${LABEL}_hf
  OUT=$OUT_ROOT_RERUN/${LABEL}
  mkdir -p $OUT
  echo "  [$LABEL] start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')"
  .venv/bin/accelerate launch --multi_gpu --num_processes 8 --num_machines 1 \
    -m lm_eval --model hf \
    --model_args "pretrained=$CKPT,dtype=bfloat16,trust_remote_code=True" \
    --tasks "piqa,boolq,sciq,openbookqa,commonsense_qa,social_iqa,logiqa" \
    --num_fewshot 0 --batch_size 16 \
    --log_samples --output_path $OUT \
    --trust_remote_code > $OUT.log 2>&1 && \
    echo "  [$LABEL] DONE" || echo "  [$LABEL] FAILED-CONTINUE"
done
echo "=== 2. social_iqa rerun DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="

echo "=== 3. wd=1.6/x8 training start $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
.venv/bin/python -m experiments.reasoning_pretraining.code_ladder.scripts.run_1_4b_wd1_6_x8_nocrossblock
echo "=== 3. wd=1.6/x8 training DONE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="

echo "=== ALL CHAINED JOBS COMPLETE $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
