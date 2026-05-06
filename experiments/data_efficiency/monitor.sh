#!/bin/bash
cd /fsx/users/dongweij/marin
MONITOR_LOG=outputs/monitor.log

while true; do
    echo "--- $(date) ---" >> $MONITOR_LOG
    
    # Check if any training is running
    PIDS=$(ps aux | grep "run_reasoning_experiment\|run_all_reasoning\|run_1_4b\|toy_300m\|toy_8b" | grep -v grep | awk '{print $2}')
    if [ -z "$PIDS" ]; then
        echo "NO TRAINING PROCESS RUNNING" >> $MONITOR_LOG
    else
        echo "Running PIDs: $PIDS" >> $MONITOR_LOG
    fi
    
    # GPU status
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1 >> $MONITOR_LOG
    
    # Latest training progress
    for logfile in outputs/reasoning_experiments_v2.log outputs/1_4b_training_655M.log outputs/openthoughts_pretrain.log; do
        if [ -f "$logfile" ]; then
            LAST=$(grep "Progress on:train" "$logfile" | tail -1)
            if [ -n "$LAST" ]; then
                echo "$(basename $logfile): $LAST" >> $MONITOR_LOG
            fi
        fi
    done
    
    # Latest eval
    for logfile in outputs/reasoning_experiments_v2.log; do
        if [ -f "$logfile" ]; then
            LAST_EVAL=$(grep "dclm_val loss" "$logfile" | tail -1)
            if [ -n "$LAST_EVAL" ]; then
                echo "eval: $LAST_EVAL" >> $MONITOR_LOG
            fi
        fi
    done
    
    echo "" >> $MONITOR_LOG
    sleep 300
done
