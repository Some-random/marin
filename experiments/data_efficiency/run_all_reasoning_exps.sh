#!/bin/bash
set -e
cd /fsx/users/dongweij/marin

MODEL=${1:-300m4k}
LOG=outputs/reasoning_experiments_${MODEL}.log

log() { echo "$(date): $1" | tee -a $LOG; }

log "=== Starting reasoning experiments for $MODEL ==="

log "=== Run B: OpenThoughts filtered only ==="
.venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run B --model $MODEL >> $LOG 2>&1
log "=== Run B complete ==="

log "=== Run C Phase 1: OpenThoughts first (3200 steps) ==="
.venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run C_phase1 --model $MODEL >> $LOG 2>&1
log "=== Run C Phase 1 complete ==="

CKPT=$(ls -td /fsx/users/dongweij/marin/checkpoints/*/step-* 2>/dev/null | head -1)
log "=== Run C Phase 2: DCLM (3200 steps), checkpoint: $CKPT ==="
.venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run C_phase2 --model $MODEL --checkpoint "$CKPT" >> $LOG 2>&1
log "=== Run C complete ==="

log "=== Run D Phase 1: DCLM first (3200 steps) ==="
.venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run D_phase1 --model $MODEL >> $LOG 2>&1
log "=== Run D Phase 1 complete ==="

CKPT=$(ls -td /fsx/users/dongweij/marin/checkpoints/*/step-* 2>/dev/null | head -1)
log "=== Run D Phase 2: OpenThoughts (3200 steps), checkpoint: $CKPT ==="
.venv/bin/python experiments/data_efficiency/run_reasoning_experiment.py --run D_phase2 --model $MODEL --checkpoint "$CKPT" >> $LOG 2>&1
log "=== Run D complete ==="

log "=== All $MODEL experiments done ==="
