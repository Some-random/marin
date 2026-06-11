#!/bin/bash
cd /fsx/users/dongweij/marin
export JAX_DIST_NUM_PROCESSES=1
export JAX_DIST_PROCESS_ID=0
export JAX_DIST_COORDINATOR="$(hostname -I | awk '{print $1}'):33334"
export WANDB_MODE=online
LOG=/fsx/users/dongweij/marin/logs/c5v2_small_logs/run_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)_FROMSCRIPT.log
nohup .venv/bin/python -m experiments.data_efficiency.run_1_4b_c5v2_small > $LOG 2>&1 &
echo "PID: $!"
echo "LOG: $LOG"
