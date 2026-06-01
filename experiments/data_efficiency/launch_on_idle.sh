#!/usr/bin/env bash
# launch_on_idle.sh — pick an idle GPU node from a candidate list and
# nohup-launch a command on it. The candidate list is one node per line in
# experiments/data_efficiency/cluster_nodes.txt (lines starting with `#` skipped).
#
# Idleness check: ssh to candidate, sample GPU utilization. If all GPUs are
# below $IDLE_THRESHOLD percent and we can run nvidia-smi (i.e. the node is
# actually up), we use it.
#
# Usage:
#   experiments/data_efficiency/launch_on_idle.sh \
#       --script /path/to/script.sh \
#       [--log /fsx/users/dongweij/marin/logs/myjob.log] \
#       [--name myjob]                          # used in default log name
#       [--exclude gpu-dy-p4d24xlarge-5]        # comma list to skip
#       [--threshold 5]                         # max %% util to count as idle
#       [--require-static]                      # only consider gpu-st-* nodes
#
# Prints to stdout:
#   chosen_node=<hostname>
#   log=<absolute log path>
#   remote_pid=<pid on the remote node>
#
# Errors with exit code 1 if no idle node found.

set -uo pipefail

NODES_FILE=/fsx/users/dongweij/marin/experiments/data_efficiency/cluster_nodes.txt
SCRIPT=""
LOG=""
NAME="job"
EXCLUDE=""
THRESHOLD=5
REQUIRE_STATIC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --script) SCRIPT="$2"; shift 2;;
    --log) LOG="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --exclude) EXCLUDE="$2"; shift 2;;
    --threshold) THRESHOLD="$2"; shift 2;;
    --require-static) REQUIRE_STATIC=1; shift;;
    -h|--help) sed -n 's/^# \?//;1,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$SCRIPT" ]]; then
  echo "ERROR: --script is required" >&2; exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: script '$SCRIPT' does not exist" >&2; exit 1
fi

if [[ -z "$LOG" ]]; then
  LOG=/fsx/users/dongweij/marin/logs/${NAME}_$(TZ='America/Los_Angeles' date +%Y%m%d_%H%M%S).log
fi
mkdir -p "$(dirname "$LOG")"

HERE=$(hostname)

# Build candidate list from config, applying filters.
candidates=()
while IFS= read -r line; do
  line=$(echo "$line" | sed 's/[[:space:]]*#.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//')
  [[ -z "$line" ]] && continue
  [[ "$line" == "$HERE" ]] && continue   # don't pick the node we're on
  if [[ -n "$EXCLUDE" ]]; then
    if echo ",$EXCLUDE," | grep -q ",$line,"; then continue; fi
  fi
  if (( REQUIRE_STATIC )); then
    [[ "$line" != gpu-st-* ]] && continue
  fi
  candidates+=("$line")
done < "$NODES_FILE"

if (( ${#candidates[@]} == 0 )); then
  echo "ERROR: no candidate nodes after filtering" >&2; exit 1
fi

# Probe each candidate for idleness.
chosen=""
for cand in "${candidates[@]}"; do
  # 4-second timeout — dynamic nodes that are powered down won't answer.
  util=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=4 -o BatchMode=yes "$cand" \
         'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1' 2>/dev/null)
  if [[ -z "$util" || ! "$util" =~ ^[0-9]+$ ]]; then
    echo "  $cand — unreachable or no GPUs visible" >&2
    continue
  fi
  if (( util <= THRESHOLD )); then
    echo "  $cand — max GPU util ${util}%% (≤${THRESHOLD}%%), choosing" >&2
    chosen="$cand"
    break
  else
    echo "  $cand — max GPU util ${util}%% (>${THRESHOLD}%%), skip" >&2
  fi
done

if [[ -z "$chosen" ]]; then
  echo "ERROR: no idle node found (all candidates busy or unreachable)" >&2; exit 1
fi

# Launch.
echo "=== launching $SCRIPT on $chosen at $(TZ='America/Los_Angeles' date '+%H:%M:%S %Z') ===" >&2
remote_pid=$(ssh -o StrictHostKeyChecking=no "$chosen" \
  "nohup bash $SCRIPT > $LOG 2>&1 & echo \$!" 2>/dev/null)

if [[ -z "$remote_pid" ]]; then
  echo "ERROR: failed to launch on $chosen" >&2; exit 1
fi

echo "chosen_node=$chosen"
echo "log=$LOG"
echo "remote_pid=$remote_pid"
