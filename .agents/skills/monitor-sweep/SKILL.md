---
name: monitor-sweep
description: Sweep stale Monitor tasks accumulated across the session. Use whenever the user asks to "kill stale monitors" / "clean up monitors" / "sweep monitors" / lists lurking monitor descriptions, OR proactively after a long-running task (training, eval) completes and its monitor is no longer needed. Reads the per-session registry at `~/.claude/monitor-registry.jsonl`, filters by an optional label substring, and calls TaskStop on each match.
---

# What this skill does

Each Monitor I arm during a session has a `task_id` (e.g. `bafiiysgy`) that I should write to a registry at `~/.claude/monitor-registry.jsonl` when I arm it. This skill reads that registry and stops monitors matching a user-supplied substring (label / run-tag / description).

Without the registry discipline, Monitor IDs cannot be enumerated after-the-fact (Monitor IDs don't surface in any UI the user can see). The registry IS the only way to recover them.

# How to invoke

```
/monitor-sweep <substring>    # stop all registered monitors matching substring (case-insensitive)
/monitor-sweep --all          # stop EVERY registered monitor (use after a multi-task session ends)
/monitor-sweep --list         # show registered monitors without stopping anything
```

# Procedure

## Step 1: read + filter registry

```bash
REG=~/.claude/monitor-registry.jsonl
[ -f "$REG" ] || { echo "no registry at $REG — nothing to sweep"; exit 0; }

# Filter by substring (case-insensitive). For --all, show everything.
case "$ARG" in
  --all|all) MATCH=$(cat "$REG") ;;
  --list)    cat "$REG" | jq -r '"\(.task_id)  \(.label)  \(.description)"'; exit 0 ;;
  *)         MATCH=$(grep -i "$ARG" "$REG") ;;
esac

if [ -z "$MATCH" ]; then
  echo "no registered monitors match '$ARG'"; exit 0
fi

echo "$MATCH" | jq -r '"\(.task_id)  \(.label)  \(.description)"'
```

## Step 2: call TaskStop on each match

For each `task_id` in MATCH, call TaskStop. Some will already be stopped (no-op) — don't fail on those.

```python
import json
matched_ids = [json.loads(ln)['task_id'] for ln in match_stdin.splitlines() if ln.strip()]
for mid in matched_ids:
    # call TaskStop tool (this is pseudocode — actually call the TaskStop tool in the conversation)
    TaskStop(task_id=mid)
```

## Step 3: prune registry

Remove the stopped entries from the registry file. Don't delete the whole file (other matches may be valid).

```bash
case "$ARG" in
  --all|all) > "$REG" ;;
  *)         grep -iv "$ARG" "$REG" > "${REG}.tmp" && mv "${REG}.tmp" "$REG" ;;
esac
```

## Step 4: report what was stopped

One line per stopped monitor: `stopped <task_id>  (<description>)`. Final summary line: `swept N monitor(s) matching '<arg>'`.

# Registry format

`~/.claude/monitor-registry.jsonl` — one JSON object per line:

```json
{"task_id": "bafiiysgy", "label": "code25b_v2_4n_20260616_121502", "description": "code25b_v2 training: Progress + crashes", "created": "2026-06-16T12:15:08-07:00", "associated_with": "training-run"}
```

Required fields:
- `task_id` — the Monitor's task ID returned by the Monitor tool.
- `label` — the run-tag / model-label / eval-label this monitor was created for. Used for substring matching.
- `description` — copy of the Monitor's `description` arg. Used for human-readable reporting.
- `created` — ISO timestamp.
- `associated_with` — broad category: `training-run`, `eval-pipeline`, `tokenize`, `fetch`, `infra-watch`, etc.

# Discipline I (the agent) must follow

**Every time I create a Monitor in a conversation, I MUST immediately append a registry entry.**

Pattern after each Monitor call:

```bash
# Inside the same bash where I just called Monitor:
echo "{\"task_id\": \"$MID\", \"label\": \"$LABEL\", \"description\": \"$DESC\", \"created\": \"$(date -Iseconds)\", \"associated_with\": \"$KIND\"}" >> ~/.claude/monitor-registry.jsonl
```

Where `$MID` is the task ID returned by the Monitor tool call. If I forget, the registry stays empty and the monitor becomes un-sweepable except at session end.

# Backstop: stale-by-mtime sweep

If the registry is empty/missing OR there are unregistered monitors lurking (created before the discipline existed), the .output files at `/tmp/claude-1000/-fsx-users-dongweij-marin/<session-id>/tasks/<task_id>.output` are the closest thing to a backup. A `<task_id>` whose .output hasn't been touched in >24 h is almost certainly stale. Cannot enumerate live monitor IDs from .output filenames alone (TaskStop on a dead/missing ID is a no-op anyway), so this is best-effort.

```bash
# Identify candidates older than 24 h:
TASK_DIR=/tmp/claude-1000/-fsx-users-dongweij-marin/$(ls -d /tmp/claude-1000/-fsx-users-dongweij-marin/*/ | xargs -I{} basename {} | head -1)/tasks
find "$TASK_DIR" -name "*.output" -mtime +1 -exec basename {} .output \; | head -50
# Then call TaskStop on each ID. Most will no-op; persistent ones will actually be stopped.
```

# Anti-patterns

- **Calling TaskStop on EVERY ID in the .output dir blindly.** 342+ files accumulate per long session — stopping 342 monitors at once is fine in cost but useless if most are already stopped. Use the registry first.
- **Wiping the registry without filtering.** `--all` should only be used at session-end or when the user explicitly confirms.
- **Forgetting the registry append after creating a Monitor.** If I create a Monitor without registering it, this skill cannot help later. The discipline is mine to keep; the skill only acts on what's been registered.
- **Treating "TaskStop returned no-op" as failure.** Already-stopped monitors return errors; that's expected when sweeping stale entries.

# Why this skill exists

In long multi-day sessions (this one had ~50 monitors armed across June 15-17), monitors with `persistent=true` survive until session-end. They generate noise notifications (a few per day) and clutter the user's monitor UI. Without a registry, there's no way to enumerate live monitors from inside the agent — Monitor IDs aren't surfaced anywhere except in the agent's own conversation context, which gets compacted away. The registry + this skill close that loop.

Worked example from 2026-06-17: user listed 17 lurking monitors by description ("C5-v6-NEW-v7 eval", "Code25B resume orchestrator", "SE-Python fetch", etc.). None had registry entries (skill was created later that day). Skill exists so future sessions never repeat this mess.
