#!/bin/bash
set -euo pipefail

SCRIPT="experiments/data_efficiency/run_h1_continuous.py"
CKPT_BASE="checkpoints"

find_latest_checkpoint() {
    local run_id=$1
    local ckpt_dir="$CKPT_BASE/$run_id"
    ls -d "$ckpt_dir"/step-* 2>/dev/null | sort -t- -k2 -n | tail -1
}

echo "=========================================="
echo "H1 Experiment: OWM+Code vs DCLM (Continuous Cosine LR)"
echo "=========================================="
echo "Phase 0: 3096 steps (4 epochs of 203M DCLM)"
echo "Phase 1: 1667 steps (437M tokens - OWM+Code treatment / DCLM control)"
echo "Phase 2: 3052 steps (800M tokens - DCLM both arms)"
echo "Cosine LR spans phases 1+2: 4719 total steps"
echo "=========================================="

# --- Phase 0: Pretrain from scratch ---
echo ""
echo "[Phase 0] Pretraining 300M on DCLM 200M, 4 epochs (3096 steps)..."
.venv/bin/python $SCRIPT --phase phase0 2>&1 | tee logs/h1_phase0.log

# Find phase0 checkpoint
PHASE0_RUN_ID=$(grep -oP 'View run .* at: .*/runs/\K\S+' logs/h1_phase0.log | tail -1)
if [ -z "$PHASE0_RUN_ID" ]; then
    echo "ERROR: Could not find phase0 run ID in logs"
    exit 1
fi
PHASE0_CKPT=$(find_latest_checkpoint "$PHASE0_RUN_ID")
echo "[Phase 0] Checkpoint: $PHASE0_CKPT"

# --- Phase 1: Treatment (OWM+Code) and Control (DCLM) ---
echo ""
echo "[Phase 1 - Treatment] OWM+Code mix (219M + 218M tokens) from phase0 checkpoint..."
.venv/bin/python $SCRIPT --phase treatment_p1 --checkpoint "$PHASE0_CKPT" 2>&1 | tee logs/h1_treatment_p1.log

TREAT_P1_RUN_ID=$(grep -oP 'View run .* at: .*/runs/\K\S+' logs/h1_treatment_p1.log | tail -1)
TREAT_P1_CKPT=$(find_latest_checkpoint "$TREAT_P1_RUN_ID")
echo "[Phase 1 - Treatment] Checkpoint: $TREAT_P1_CKPT"

echo ""
echo "[Phase 1 - Control] Disjoint DCLM (~440M tokens) from phase0 checkpoint..."
.venv/bin/python $SCRIPT --phase control_p1 --checkpoint "$PHASE0_CKPT" 2>&1 | tee logs/h1_control_p1.log

CTRL_P1_RUN_ID=$(grep -oP 'View run .* at: .*/runs/\K\S+' logs/h1_control_p1.log | tail -1)
CTRL_P1_CKPT=$(find_latest_checkpoint "$CTRL_P1_RUN_ID")
echo "[Phase 1 - Control] Checkpoint: $CTRL_P1_CKPT"

# --- Phase 2: Both arms on disjoint DCLM ---
echo ""
echo "[Phase 2 - Treatment] Disjoint DCLM (~810M tokens), initialize_from_step=1667..."
.venv/bin/python $SCRIPT --phase treatment_p2 --checkpoint "$TREAT_P1_CKPT" 2>&1 | tee logs/h1_treatment_p2.log

echo ""
echo "[Phase 2 - Control] Disjoint DCLM (~810M tokens), initialize_from_step=1667..."
.venv/bin/python $SCRIPT --phase control_p2 --checkpoint "$CTRL_P1_CKPT" 2>&1 | tee logs/h1_control_p2.log

echo ""
echo "=========================================="
echo "H1 Experiment Complete!"
echo "=========================================="
echo "Check WandB for results: https://wandb.ai/dongwei_jiang/dongwei-data-efficiency"
echo "Filter by tag: h1-continuous"
