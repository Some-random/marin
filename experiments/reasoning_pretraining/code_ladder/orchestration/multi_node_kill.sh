#!/usr/bin/env bash
# Kill a running multi-node training job by SSHing into each node and pkilling
# the matching python process.
#
# Usage:
#   ./multi_node_kill.sh --nodes "gpu-st-p4d24xlarge-4,gpu-dy-p4d24xlarge-1" --config run_smoke_multinode.py
#   ./multi_node_kill.sh --nodes "<all 4 nodes>" --config run_1_4b_1ep_dclm.py
#
# Always confirm with --yes-i-mean-it because killing is destructive.

set -euo pipefail

NODES=""
CONFIG=""
CONFIRM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --nodes) NODES="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --yes-i-mean-it) CONFIRM="yes"; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$NODES" ] || [ -z "$CONFIG" ]; then
  echo "Usage: $0 --nodes <comma-sep> --config <script.py> [--yes-i-mean-it]" >&2
  exit 1
fi

PATTERN=$(basename "$CONFIG")

if [ -z "$CONFIRM" ]; then
  echo "About to kill processes matching '$PATTERN' on nodes: $NODES"
  echo "Re-run with --yes-i-mean-it to confirm"
  exit 1
fi

IFS=',' read -ra NODE_ARR <<< "$NODES"
for node in "${NODE_ARR[@]}"; do
  echo "Killing on $node..."
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$node" \
    "pkill -f 'python.*${PATTERN}' && echo '  killed' || echo '  nothing to kill'" 2>&1 | tail -2
done
