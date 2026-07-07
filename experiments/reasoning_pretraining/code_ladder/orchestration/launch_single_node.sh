#!/usr/bin/env bash
# Single-node launcher for parameterized templates (smallscale battery).
#
# Usage:
#   ./launch_single_node.sh \
#     --node gpu-st-p4d24xlarge-1 \
#     --config experiments/reasoning_pretraining/code_ladder/scripts/run_smallscale_single_phase.py \
#     --run-tag 300m_a5 \
#     --env MODEL_KEY=300m4k \
#     --env TEXT_SOURCE=dclm \
#     --env TARGET_TOKENS=6e9 \
#     --env RUN_TAG=300m_a5
#
# Each --env KEY=VAL is forwarded into the remote shell before exec.
# Logs land in /fsx/users/dongweij/marin/logs/single_<run-tag>_<TS>.log

set -euo pipefail

NODE=""
CONFIG=""
RUN_TAG=""
ENV_PAIRS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --node) NODE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --env) ENV_PAIRS+=("$2"); shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$NODE" ] || [ -z "$CONFIG" ] || [ -z "$RUN_TAG" ]; then
  echo "Usage: $0 --node <name> --config <python-script> --run-tag <name> [--env KEY=VAL ...]" >&2
  exit 1
fi

TS=$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
LOG=/fsx/users/dongweij/marin/logs/single_${RUN_TAG}_${TS}.log

mem=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$NODE" \
  "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc" 2>/dev/null || echo 0)
echo "[launch_single_node] $NODE GPU mem in use = ${mem} MiB"
if [ "${mem:-0}" -gt 1000 ]; then
  echo "ERROR: $NODE has >1GB GPU mem in use; aborting." >&2
  exit 1
fi

EXPORTS=""
for pair in "${ENV_PAIRS[@]:-}"; do
  EXPORTS="${EXPORTS}export ${pair} && "
done

echo "[launch_single_node] run-tag=$RUN_TAG"
echo "[launch_single_node] config=$CONFIG"
echo "[launch_single_node] log=$LOG"
echo "[launch_single_node] env: ${ENV_PAIRS[*]:-}"

nohup ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "$NODE" "
  cd /fsx/users/dongweij/marin && \
  ${EXPORTS}\
  export WANDB_MODE=online && \
  .venv/bin/python $CONFIG
" > "$LOG" 2>&1 < /dev/null &
disown

echo "[launch_single_node] launched, tail $LOG"
