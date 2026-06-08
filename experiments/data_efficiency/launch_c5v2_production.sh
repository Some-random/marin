#!/usr/bin/env bash
# Production launch for C5-v2 — uses multi_node_launch.sh with the clean-code config.
#
# Default 8 nodes. If --4node-fallback is set and 8 GPU-idle nodes can't be
# claimed within 90s, falls back to 4 nodes (same global batch=256, just
# per_device_parallelism=8 instead of 4 — matches A5 exactly).
#
# Usage:
#   ./launch_c5v2_production.sh [--4node-fallback] [--coordinator-port N]

set -euo pipefail
cd /fsx/users/dongweij/marin

NODE_COUNT=8
ALLOW_FALLBACK=0
COORD_PORT=33333
RUN_TAG="c5v2-1ep-clean-code-then-text"

while [ $# -gt 0 ]; do
  case "$1" in
    --4node-fallback) ALLOW_FALLBACK=1; shift ;;
    --coordinator-port) COORD_PORT="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --nodes) NODE_COUNT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Find idle GPU nodes (per `sinfo`, "idle" or "idle~" both work — "~" = powered off but allocatable)
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] looking for $NODE_COUNT idle GPU nodes..."
IDLE_NODES=$(sinfo -h -p gpu -o "%n %T" 2>/dev/null \
  | awk '$2 ~ /^idle/ {print $1}' \
  | grep -E "gpu-(st|dy)-p4d24xlarge" \
  | head -"$NODE_COUNT")

N_FOUND=$(echo "$IDLE_NODES" | grep -c .)
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] found $N_FOUND idle nodes"

if [ "$N_FOUND" -lt "$NODE_COUNT" ] && [ "$ALLOW_FALLBACK" -eq 1 ] && [ "$N_FOUND" -ge 4 ]; then
  echo "  not enough nodes; falling back to 4 nodes"
  NODE_COUNT=4
  IDLE_NODES=$(echo "$IDLE_NODES" | head -4)
elif [ "$N_FOUND" -lt "$NODE_COUNT" ]; then
  echo "ERROR: only $N_FOUND idle nodes; needed $NODE_COUNT (use --4node-fallback to allow 4-node)" >&2
  exit 1
fi

NODES_CSV=$(echo "$IDLE_NODES" | head -"$NODE_COUNT" | paste -sd, -)
echo "[$(TZ='America/Los_Angeles' date '+%H:%M:%S %Z')] launching on $NODE_COUNT nodes: $NODES_CSV"

./experiments/data_efficiency/multi_node_launch.sh \
  --nodes "$NODES_CSV" \
  --config experiments/data_efficiency/run_1_4b_c5v2_clean_code.py \
  --run-tag "$RUN_TAG" \
  --coordinator-port "$COORD_PORT"
