#!/usr/bin/env bash
# SSH-based multi-node launcher for Levanter training jobs.
#
# Why this exists: Slurm sbatch on this cluster shares allocations across users
# (alloc state in sinfo even when GPUs are empty), making sbatch unreliable for
# claiming idle nodes. This script SSHes into a pre-chosen list of empty nodes
# and starts the training process on each with the right
# jax.distributed.initialize() environment.
#
# Usage:
#   ./multi_node_launch.sh \
#     --nodes "gpu-st-p4d24xlarge-4,gpu-dy-p4d24xlarge-1" \
#     --config experiments/data_efficiency/run_1_4b_oneep.py \
#     --run-tag "1ep-baseline-test" \
#     [--coordinator-port 33333]
#
# Each node runs: cd /fsx/users/dongweij/marin && uv run python <config>
# with these env vars set:
#   JAX_DIST_NUM_PROCESSES=N
#   JAX_DIST_PROCESS_ID=i
#   JAX_DIST_COORDINATOR=<coord_ip>:<port>
# Levanter's DistributedConfig reads these and calls jax.distributed.initialize().
# Coordinator is the first node in --nodes.
#
# Logs land in /fsx/users/dongweij/marin/logs/<run-tag>/node-<i>.log

set -euo pipefail

NODES=""
CONFIG=""
RUN_TAG=""
COORD_PORT=33333

while [ $# -gt 0 ]; do
  case "$1" in
    --nodes) NODES="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --coordinator-port) COORD_PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$NODES" ] || [ -z "$CONFIG" ] || [ -z "$RUN_TAG" ]; then
  echo "Usage: $0 --nodes <comma-sep> --config <python-script> --run-tag <name> [--coordinator-port N]" >&2
  exit 1
fi

IFS=',' read -ra NODE_ARR <<< "$NODES"
NUM_NODES=${#NODE_ARR[@]}
COORD_NODE="${NODE_ARR[0]}"
COORD_IP=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$COORD_NODE" "hostname -I | awk '{print \$1}'" 2>/dev/null)

if [ -z "$COORD_IP" ]; then
  echo "ERROR: could not get IP of coordinator node $COORD_NODE" >&2
  exit 1
fi

LOG_DIR=/fsx/users/dongweij/marin/logs/multinode_${RUN_TAG}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

echo "=== Multi-node launch ==="
echo "Run tag: $RUN_TAG"
echo "Config:  $CONFIG"
echo "Nodes:   $NODES (num=$NUM_NODES)"
echo "Coord:   $COORD_NODE @ $COORD_IP:$COORD_PORT"
echo "Logs:    $LOG_DIR"
echo "==="

# Sanity: verify all nodes have free GPUs
for i in "${!NODE_ARR[@]}"; do
  node="${NODE_ARR[$i]}"
  mem=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$node" \
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd+ | bc" 2>/dev/null)
  echo "  node $i ($node): GPU mem in use = ${mem} MiB"
  if [ "${mem:-0}" -gt 1000 ]; then
    echo "  WARNING: node $node has >1GB GPU memory used — skipping" >&2
    exit 1
  fi
done

# Launch each process_id with the right env
for i in "${!NODE_ARR[@]}"; do
  node="${NODE_ARR[$i]}"
  log_file="$LOG_DIR/node-${i}-${node}.log"
  echo "Launching process $i on $node -> $log_file"
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "$node" "
    cd /fsx/users/dongweij/marin && \
    export JAX_DIST_NUM_PROCESSES=$NUM_NODES && \
    export JAX_DIST_PROCESS_ID=$i && \
    export JAX_DIST_COORDINATOR=$COORD_IP:$COORD_PORT && \
    export NCCL_DEBUG=INFO && \
    export FI_PROVIDER=efa && \
    export FI_EFA_USE_DEVICE_RDMA=1 && \
    export WANDB_MODE=$([ $i -eq 0 ] && echo 'online' || echo 'offline') && \
    $([ $i -ne 0 ] && echo 'export WANDB_DISABLED=true &&') \
    .venv/bin/python $CONFIG > $log_file 2>&1
  " > "${log_file}.launch" 2>&1 &
  LAUNCH_PIDS[$i]=$!
done

echo ""
echo "All processes launched. PIDs:"
for i in "${!LAUNCH_PIDS[@]}"; do
  echo "  process $i: PID ${LAUNCH_PIDS[$i]}"
done
echo ""
echo "Monitor with: tail -F $LOG_DIR/node-0-*.log"
echo "Stop all:     for n in ${NODES//,/ }; do ssh \$n 'pkill -f \"python.*$CONFIG\"' & done; wait"

# Wait for them all
for pid in "${LAUNCH_PIDS[@]}"; do
  wait "$pid" || echo "  WARN: process pid=$pid exited non-zero"
done

echo ""
echo "=== All processes exited at $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ==="
