#!/usr/bin/env bash
# keep_nodes_warm.sh — keep our dynamic GPU nodes (gpu-dy-p4d24xlarge-1..9)
# powered-up and reserved so training relaunches don't stall waiting for nodes,
# OR surface exactly why a node can't come up (AWS capacity vs recoverable).
#
# WHY THIS EXISTS (confirmed on the cluster 2026-06-23):
#   - SuspendTime=600s: any CLOUD (dy) node idle >10 min is auto-powered-down by
#     ParallelCluster's slurm_suspend. SuspendExcNodes protects ONLY the static
#     nodes (gpu-st-p4d24xlarge-1..4, cpu-st-c512xlarge-1..4) — every dy node is
#     subject to auto-suspend. That is normal autoscaling, not a fault.
#   - A RUNNING job on a node prevents the idle-suspend → a 7-day "holder"
#     sleep job is what actually keeps a dy node warm.
#   - Manual revive of a powered-down cloud node is `scontrol update State=POWER_UP`
#     (NOT RESUME — RESUME returns "Invalid node state" for cloud nodes).
#   - If AWS has no p4d.24xlarge capacity, resume fails: node → DOWN+POWERING_UP
#     with Reason=(Code:ReservationCapacityExceeded)Failure when resuming nodes,
#     and any holder job NODE_FAILs. That is AWS-side and CANNOT be fixed from the
#     cluster — only retried later. Do NOT hammer POWER_UP in a tight loop.
#
# USAGE:
#   keep_nodes_warm.sh scan                 # read-only: classify every target node
#   keep_nodes_warm.sh acquire [nodes...]   # POWER_UP + holder for recoverable nodes
#   keep_nodes_warm.sh release [nodes...]   # scancel our holders (free the nodes)
# Default target set is all of OUR dy nodes; pass an explicit node list to narrow.
set -uo pipefail

OUR_DY=(gpu-dy-p4d24xlarge-1 gpu-dy-p4d24xlarge-2 gpu-dy-p4d24xlarge-3 \
        gpu-dy-p4d24xlarge-4 gpu-dy-p4d24xlarge-5 gpu-dy-p4d24xlarge-6 \
        gpu-dy-p4d24xlarge-7 gpu-dy-p4d24xlarge-8 gpu-dy-p4d24xlarge-9)
HOLDER_DAYS=7
SCONTROL=$(command -v scontrol || echo /opt/slurm/bin/scontrol)

# Classify one node from its scontrol record. Echoes "CLASS|STATE|REASON".
# Classes: UP_INUSE | UP_IDLE | SUSPENDED | CAPACITY_BLOCKED | DOWN_OTHER | UNKNOWN
classify() {
  local node="$1" info state reason
  info=$("$SCONTROL" show node "$node" 2>/dev/null)
  [ -z "$info" ] && { echo "UNKNOWN|NO_RECORD|"; return; }
  state=$(printf '%s\n' "$info" | grep -oE 'State=[^ ]+' | head -1 | cut -d= -f2)
  reason=$(printf '%s\n' "$info" | grep -oE 'Reason=.*' | head -1 | cut -d= -f2- \
           | sed 's/ \[root.*//')
  local cls
  # NOTE: the ReservationCapacityExceeded reason is only present TRANSIENTLY in the
  # ~minute after a failed resume; a settled powered-down node reverts to its prior
  # reason ("Scheduler health check failed"). So a point-in-time scan cannot always
  # tell capacity-blocked from recoverable — the definitive verdict comes from
  # 'acquire' + re-scan (a node that bounces back to SUSPENDED is capacity-blocked).
  # POWER*_DOWN must be checked before the bare DOWN token, else POWERING_DOWN /
  # POWERED_DOWN (transitional suspend) get mis-bucketed as a hard DOWN.
  if   printf '%s' "$reason" | grep -qiE 'ReservationCapacityExceeded|InsufficientInstanceCapacity|Failure when resuming'
  then cls=CAPACITY_BLOCKED
  elif printf '%s' "$state" | grep -qE 'ALLOCATED|MIXED';                then cls=UP_INUSE
  elif printf '%s' "$state" | grep -qE 'POWERED_DOWN|POWERING_DOWN|POWER_DOWN'; then cls=SUSPENDED
  elif printf '%s' "$state" | grep -qE '(^|\+)DOWN(\+|$)|NOT_RESPONDING'; then cls=DOWN_OTHER
  elif printf '%s' "$state" | grep -qE 'IDLE';                          then cls=UP_IDLE
  else cls=UNKNOWN; fi
  echo "${cls}|${state}|${reason}"
}

# Does a holder job of ours already sit on this node?
has_holder() { squeue -w "$1" -h -o '%j' 2>/dev/null | grep -q '^hold_'; }

submit_holder() {
  local node="$1" tag="${1##*-}" jid
  jid=$(sbatch --no-requeue --nodes=1 --nodelist="$node" --time="${HOLDER_DAYS}-00:00:00" \
        --partition=gpu --gres=gpu:8 --job-name="hold_${tag}" \
        --wrap='exec sleep infinity' 2>&1 | grep -oE '[0-9]+' | head -1)
  echo "${jid:-FAILED}"
}

scan() {
  printf '%-26s %-16s %s\n' NODE CLASS "STATE / REASON"
  printf '%-26s %-16s %s\n' "----" "-----" "--------------"
  for n in "${TARGETS[@]}"; do
    IFS='|' read -r cls state reason <<<"$(classify "$n")"
    local h=""; has_holder "$n" && h=" [holder:ours]"
    printf '%-26s %-16s %s%s\n' "$n" "$cls" "${state}${reason:+  ($reason)}" "$h"
  done
}

acquire() {
  local up=0 cap=0 held=0
  for n in "${TARGETS[@]}"; do
    IFS='|' read -r cls _ reason <<<"$(classify "$n")"
    case "$cls" in
      UP_INUSE)
        echo "  $n: UP_INUSE — already busy, leaving as-is" ;;
      CAPACITY_BLOCKED)
        echo "  $n: CAPACITY_BLOCKED ($reason) — AWS-side, cannot fix; retry later"; cap=$((cap+1)) ;;
      SUSPENDED|DOWN_OTHER)
        echo "  $n: $cls — issuing POWER_UP + holder"
        sudo "$SCONTROL" update NodeName="$n" State=POWER_UP 2>&1 | grep -vi '^$' | sed 's/^/      /'
        echo "      holder job: $(submit_holder "$n")"; up=$((up+1)) ;;
      UP_IDLE)
        if has_holder "$n"; then echo "  $n: UP_IDLE — holder already present"; held=$((held+1))
        else echo "  $n: UP_IDLE — submitting holder to prevent 10-min auto-suspend"
             echo "      holder job: $(submit_holder "$n")"; held=$((held+1)); fi ;;
      *) echo "  $n: $cls — no action" ;;
    esac
  done
  echo "summary: power_up_attempted=$up held_warm=$held capacity_blocked=$cap"
  echo "NOTE: re-run 'scan' in ~15 min. Nodes that POWER_UP'd should be UP_INUSE"
  echo "      (holder running). Any still DOWN with ReservationCapacityExceeded are"
  echo "      AWS-capacity-blocked — schedule a retry every 30-60 min, do not loop tight."
}

release() {
  for n in "${TARGETS[@]}"; do
    local jids; jids=$(squeue -w "$n" -h -o '%i %j' 2>/dev/null | awk '$2 ~ /^hold_/{print $1}')
    [ -n "$jids" ] && { echo "  $n: scancel $jids"; scancel $jids 2>/dev/null; } || echo "  $n: no holder of ours"
  done
}

CMD="${1:-scan}"; shift || true
if [ "$#" -gt 0 ]; then TARGETS=("$@"); else TARGETS=("${OUR_DY[@]}"); fi
case "$CMD" in
  scan)    scan ;;
  acquire) acquire ;;
  release) release ;;
  *) echo "usage: keep_nodes_warm.sh {scan|acquire|release} [nodes...]" >&2; exit 2 ;;
esac
