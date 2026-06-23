# Skill: keep-nodes-warm

Keep our dynamic GPU nodes (`gpu-dy-p4d24xlarge-1..9`) powered-up and reserved so
a training relaunch doesn't stall waiting for nodes — or surface **exactly why** a
node can't come up (AWS capacity vs. a recoverable idle-suspend).

Helper: `.agents/skills/keep-nodes-warm/keep_nodes_warm.sh {scan|acquire|release} [nodes...]`

## Why dy nodes keep going away (confirmed on the cluster 2026-06-23)

Two independent mechanisms — don't conflate them:

1. **Idle auto-suspend (normal, expected).** `SuspendTime=600s`: any CLOUD node idle
   >10 min is powered down by ParallelCluster's `slurm_suspend`. `SuspendExcNodes`
   protects **only** the static nodes (`gpu-st-p4d24xlarge-1..4`,
   `cpu-st-c512xlarge-1..4`). Every `dy` node is subject to auto-suspend. This is
   autoscaling working as designed, not a fault. A node sitting at `idle~`
   (`IDLE+CLOUD+POWERED_DOWN`) is just asleep.

2. **AWS capacity on resume (NOT fixable from the cluster).** When slurm tries to
   resume a `dy` node (via `POWER_UP`, a holder job, or any pending allocation) and
   AWS has no `p4d.24xlarge` capacity, the resume fails: the node goes
   `DOWN+CLOUD+NOT_RESPONDING+POWERING_UP` with
   `Reason=(Code:ReservationCapacityExceeded)Failure when resuming nodes`, then
   settles back to `idle~`/`POWERING_DOWN`. Any holder job `NODE_FAIL`s. **This is
   AWS-side. POWER_UP / holders cannot fix it — only a later retry when capacity
   returns.** Do NOT hammer `POWER_UP` in a tight loop.

**Key subtlety:** the `ReservationCapacityExceeded` reason is only visible for ~a
minute right after a failed resume. Once the node settles back to powered-down, its
reason reverts to whatever it was before (often `Scheduler health check failed`).
So a point-in-time `scan` **cannot** reliably distinguish "asleep, will wake" from
"asleep, AWS-capacity-blocked". The definitive verdict comes from `acquire` +
re-scan: a node that comes up `UP_INUSE` is warm; a node that bounces straight back
to `SUSPENDED`/`DOWN` is capacity-blocked.

## How a node actually stays warm

A **running job prevents idle-suspend**. So the durable mechanism is a 7-day
"holder" sleep job per node (`sbatch --nodelist=<n> --time=7-00:00:00 --gres=gpu:8
--wrap='exec sleep infinity'`). `acquire` submits these. Revive a powered-down
cloud node with `scontrol update State=POWER_UP` — **NOT `RESUME`** (RESUME returns
"Invalid node state" for cloud nodes).

## Classification (what `scan` prints)

| Class | Meaning | `acquire` action |
|---|---|---|
| `UP_INUSE` | `ALLOCATED`/`MIXED` — a job is on it | leave as-is |
| `UP_IDLE` | up but free — will auto-suspend in ≤10 min | submit holder to pin it warm |
| `SUSPENDED` | `POWERED_DOWN`/`POWERING_DOWN` — asleep | `POWER_UP` + holder (then re-scan) |
| `CAPACITY_BLOCKED` | reason shows `ReservationCapacityExceeded` | **skip** — AWS-side, retry later |
| `DOWN_OTHER` | hard `DOWN`/`NOT_RESPONDING`, no capacity reason | `POWER_UP` + holder once; if it bounces, treat as capacity |

## Procedure

### 1. Scan (read-only, safe anytime)
```bash
.agents/skills/keep-nodes-warm/keep_nodes_warm.sh scan
```

### 2. Acquire the nodes you want warm
```bash
# all dy nodes, or pass an explicit subset:
.agents/skills/keep-nodes-warm/keep_nodes_warm.sh acquire gpu-dy-p4d24xlarge-8 gpu-dy-p4d24xlarge-9
```
This `POWER_UP`s + submits holders for `SUSPENDED`/`DOWN_OTHER` nodes, pins
`UP_IDLE` nodes with a holder, and reports `CAPACITY_BLOCKED` nodes without
hammering them.

### 3. Verify after ~15 min (ResumeTimeout is 2100s / 35 min)
Re-run `scan`. A node that reached `UP_INUSE` with a `[holder:ours]` tag is warm.
A node still `SUSPENDED`/`DOWN` with a capacity reason is AWS-blocked.

### 4. Recurring keep-in-check (the actual "keeping them in check")
Arm a low-frequency loop — **30–60 min**, never tight — that runs `acquire` only
for the nodes you need and reports capacity-blocked ones. Either:
- `/loop 45m run keep_nodes_warm.sh acquire <nodes> and report`, or
- a cron via the `schedule` skill for a durable version.
Capacity-blocked nodes simply stay blocked until AWS frees `p4d` — the loop's job is
to grab them the moment capacity returns, and to keep `UP_IDLE` nodes from sleeping.

### 5. Release (give nodes back)
```bash
.agents/skills/keep-nodes-warm/keep_nodes_warm.sh release gpu-dy-p4d24xlarge-8
```
`scancel`s our `hold_*` jobs on the listed nodes.

## Anti-patterns

- **Tight POWER_UP / holder retry loops on a capacity-blocked node.** Each failed
  resume churns slurm and AWS API calls and changes nothing. Back off to 30–60 min.
- **Using `RESUME` instead of `POWER_UP`** for a cloud node — returns "Invalid node
  state specified".
- **Reading capacity-blocked from a settled `scan`.** The capacity reason is
  transient; confirm via `acquire` + re-scan, not a single snapshot.
- **Holding nodes you don't need.** Holders occupy `p4d` capacity others want; only
  pin what an imminent / running job actually needs, and `release` when done.
- **`--dependency=afterok` chains of holders.** When a holder hits its TimeLimit it
  exits non-zero, so `afterok` becomes `DependencyNeverSatisfied`. Submit independent
  7-day holders.

## Relationship to other skills

`babysit-job` watches a *running job's* forward progress. This skill manages *node
lifecycle* underneath jobs — getting/keeping the hardware so a (re)launch has
somewhere to land. Use them together: keep-nodes-warm secures the nodes, then the
launcher + babysit run the job on them.
