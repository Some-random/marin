---
name: train-and-eval
description: Chain a Levanter training run with its full §3 eval pipeline (v2 suite + paloma + gsm + aryabumi-nl-extras + quac) in one in-session orchestration, including a pre-launch design-verification step (data sample inspection + baseline declaration) and a post-eval comparison report against named baselines. Use whenever the user wants to launch training that should auto-eval on completion — instead of forgetting to fire eval and burning hours of idle compute. Keeps the agent in session across all stages so crashes / hangs are caught at every step.
---

# Quick path

```
/train-and-eval \
  --nodes "<comma-sep-node-list>" \
  --config experiments/data_efficiency/<train_script>.py \
  --run-tag "<unique-tag>" \
  --label "<§3-column-label>" \
  --compare-against "<col-substr-1>" \
  [--compare-against "<col-substr-2>" ...]
```

Example:
```
/train-and-eval \
  --nodes "gpu-st-p4d24xlarge-1,gpu-st-p4d24xlarge-2,gpu-dy-p4d24xlarge-1,gpu-dy-p4d24xlarge-4" \
  --config experiments/data_efficiency/run_1_4b_c5v6_strict_phase2.py \
  --run-tag c5v6-strict-8n \
  --label c5v6_strict_step14671 \
  --compare-against "C5-v6 final" \
  --compare-against "C5-v4 final"
```

`--compare-against` is **repeatable** — declare every baseline this run should be measured against.

# Why this skill exists

Without it: training finishes → agent doesn't notice for hours → eval doesn't fire → free GPUs idle until manually noticed. Burned hours of free compute on c5v6-strict (training done 04:32 PDT, eval not fired until 11:41 PDT — 7h gap).

Worse: when eval finally lands, the agent (or user) might forget to actually *compare* it against the run it was designed to evaluate. The whole point of c5v6-strict was vs. c5v6-final — if no comparison gets written, the experiment is half-done.

With this skill: pre-launch data review → training → eval → §3 fill → comparison report(s) happen as one chained operation, all in-session, with monitors at every stage.

# Procedure

## Stage -1: Pre-launch design verification (NEW)

Before firing training, the skill prints a **structured experiment spec** so the user can sanity-check it. The skill does not proceed to Stage 1 until the user explicitly confirms (or auto-confirms with `--yes`).

### -1a: Parse the training config and print the data + hparam spec

For the `--config` script, extract and print:

- **Init checkpoint** (if any — `initialize_from_checkpoint_path`).
- **Component caches and weights** (`LmDataConfig.components` + `train_weights`). Group as: text sources, code sources, markup sources, eval-only (weight = 0). Show absolute weights AND % of training mix.
- **Token budget**: `NUM_TRAIN_STEPS × batch × seq_len`. Compute Chinchilla ratio (`tokens / params`).
- **Hyperparameters**: batch size, LR + schedule + warmup, weight decay, max_grad_norm, model size.

Format as a table the user can scan in <30 seconds. Example:

```
Training spec for c5v6-strict-8n:
  Model: 1.4B (LLaMA-style, 4096 seq)
  Init from: checkpoints/1_4b_c5v3_phase1/8dtdcear/step-14671
  Budget: 14,672 steps × batch 256 × seq 4096 = 15.39B tokens (Chinchilla ratio: 11.0)
  Hparams: LR 3e-4 cosine→0 (warmup 1%), wd 0.1, max_grad_norm 1.0, β=(0.9,0.95)

Data mix (% of training tokens):
  TEXT (70.0%):
    dclm_shard0..6:  10.0% each = 70.0%
  CODE+MARKUP (30.0%):
    code_se_python:           12.97%  (8.8/16.3 × 0.80 × 0.30)
    code_nemotron_cc:         10.75%  (7.3/16.3 × 0.80 × 0.30)
    code_nemotron_ua:          0.28%  (0.2/16.3 × 0.80 × 0.30)
    markup_se_markdown:        6.00%  (1.00 × 0.20 × 0.30)
  EVAL-ONLY (weight=0):  dclm_200m_val + 16 paloma_*
```

### -1b: Sample 10 docs from each non-zero-weighted cache

Per the standing rule "ALWAYS inspect actual content of training and eval data" (CLAUDE.local.md):

- For each cache referenced with non-zero training weight, sample 10 random documents.
- Decode tokens via the Llama-3.1-8B tokenizer to text (the cache's tokenizer).
- Write all samples to `$LOG_DIR/PRE_LAUNCH_SAMPLES.md` grouped by cache name, with metadata (token count, doc index).
- Print the path to chat and **block** until the user replies (or `--yes` was passed). If `--yes`, log a warning that samples were not human-reviewed.

### -1c: Declare baselines

Print the list of `--compare-against` substrings, resolve each to a §3 column header (via `eval_section3.py`'s column-substring resolution), and fail loudly if any is ambiguous (matches >1) or missing.

For each resolved baseline, also print its §2 row description so the user has the recipe in front of them and can confirm the comparison is meaningful. Example:

```
Baseline 1: "C5-v6 final" → §3 column 'C5-v6 final ★'
  §2 row: 70% DCLM + 30% code+markup replay, separate cosine, init from C5-v3 phase 1.
  This skill will diff every numeric task row + Mean rows: <new> vs C5-v6 final.

Baseline 2: "C5-v4 final" → §3 column 'C5-v4 final'
  §2 row: 90% SP-NL + 10% code+markup replay, separate cosine.
  This skill will diff every numeric task row + Mean rows: <new> vs C5-v4 final.
```

### -1d: Confirmation gate

Print the experiment hypothesis prompt (per the standing experiment-design rules) and wait for user OK:

```
About to launch c5v6-strict-8n.
  Hypothesis: strict prefix replay > same-cache-different-shuffle replay
              at the 30% code+markup mix ratio (C5-v6 setup).
  Falsifiable: if §3 Mean Open-book improves >0.005 over C5-v6 final → strict replay helps.
  Reference baselines: C5-v6 final, C5-v4 final.
  Samples written: $LOG_DIR/PRE_LAUNCH_SAMPLES.md

Have you reviewed the spec + samples? [y/N]
```

If user says no or doesn't answer in a reasonable window — abort, leave a marker in `$LOG_DIR/ABORTED_PRELAUNCH.json` explaining why.

## Stage 0: validate inputs

- Verify `--config` file exists.
- Verify `--nodes` are all reachable and have <1 GB GPU memory.
- Compute `LOG_DIR = logs/multinode_<run-tag>_<TS>`.
- Write a state file `$LOG_DIR/CHAIN_STATE.json` with `{stage: "training", run_tag, label, baselines: [...], started_at}`.

## Stage 1: launch training

```bash
nohup bash experiments/data_efficiency/multi_node_launch.sh \
  --nodes "<nodes>" --config "<config>" --run-tag "<run-tag>" \
  > "$LOG_DIR/launch.log" 2>&1 < /dev/null &
disown
```

Then arm a Monitor on `$LOG_DIR/node-0-*.log` with patterns:
- Forward-progress: `Progress on:train [0-9]+\.0+kit` (dedup per-whole-kit via awk)
- Completion: `All training steps completed` OR `Saved checkpoint to .*/step-<NUM_TRAIN_STEPS-1>` (parse NUM_TRAIN_STEPS from launcher init lines)
- Crashes: `Traceback|panic:|SIGSEGV|SIGBUS|RuntimeError|FAILED|nil pointer|rendezvous\.cc|coordination_service.*before final step|OOM|Killed`

The crash regex includes `coordination_service` ONLY BEFORE final step — after the final step, the same pattern is the normal worker-shutdown noise. The skill differentiates by checking whether the final-step ckpt was saved first.

Also arm a **ScheduleWakeup every 25 min** as a backstop in case the log goes silent.

## Stage 2: handle training completion

When the completion event fires:
- Parse final ckpt path: `grep -oE "Saved checkpoint to checkpoints/[^ ]+/step-[0-9]+" $LOG_DIR/node-0-*.log | tail -1`
- Update state: `{stage: "training_done", ckpt: <path>, completed_at: <ts>}`
- TaskStop the training Monitor.
- Proceed to Stage 3.

If a crash event fires before completion: TaskStop the Monitor, write `{stage: "training_failed", error_excerpt: ...}` to state file, alert user. Do NOT fire eval on a crashed checkpoint.

## Stage 3: re-scan nodes and fire eval

Re-scan availability (NOT trusting the launcher's launch-time assumption):

```bash
FREE=()
for n in gpu-dy-p4d24xlarge-{1..9} gpu-st-p4d24xlarge-{1..4}; do
  mem=$(ssh -o ConnectTimeout=5 $n "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc" 2>/dev/null || echo 9999)
  if [ "${mem:-9999}" -lt 1000 ]; then FREE+=($n); fi
done
```

Graceful degrade based on `${#FREE[@]}`:

| Free | Plan |
|------|------|
| ≥ 5 | v2 on FREE[0]; paloma FREE[1]; gsm FREE[2]; aryabumi-nl-extras FREE[3]; quac FREE[4]. Parallel. ~1h total. |
| 3–4 | v2 on FREE[0]; aux split across FREE[1..N]; one node may handle 2 aux sequentially. ~1h 20min. |
| 1–2 | v2 first on FREE[0] (~45 min), then 4 aux sequentially on same node (~30 min). ~1.5h total. |
| 0 | Abort eval. Write `{stage: "eval_blocked_no_nodes"}` to state file. Alert user with the marker file path so they can manually fire later via `/eval-for-section3`. |

For each task fired, capture ssh PID + log path in `$LOG_DIR/CHAIN_STATE.json` under `eval_runners: [{kind: "v2", node, log_path, pid, started_at}, ...]`.

Arm a Monitor on every eval log with the `ALL DONE | tasks=.*DONE | FAILED-CONTINUE | Traceback | RESOURCE_EXHAUSTED` regex.

## Stage 4: collect eval completion

For each eval log, watch for `ALL DONE`. Update state file as each lands.

When ALL 5 (v2 + 4 aux) report ALL DONE:
- Run `eval_section3.py fill-from-results <v2-results-dir> "<label>"` (the v2 fill).
- For aux results, the helper `fill-from-results` may not pick them up (per current eval_section3.py limitations). Use `eval_section3.py fill-cell` for any aux task you can extract from the result JSONs (paloma_macro mean, gsm_symbolic_main, gsm_noop, storycloze, cb, quac, etc.).
- Update state: `{stage: "section3_filled", filled_cells: <count>}`.

## Stage 5: comparison against EACH declared baseline

For every `--compare-against` baseline given in Stage -1c:

For each task row in §3 where BOTH the new column AND that reference column have a value:
- Compute `delta = new - reference` (or `delta = reference - new` for bpb/loss rows where lower is better — the skill knows which is which from the existing TASKS table in `eval_section3.py`).
- Generate a markdown comparison table:

```
| Task | <new label> | <reference> | Δ | Better? |
|------|-----:|-----:|-----:|:---:|
| arc_challenge[25] | 0.456 | 0.423 | +0.033 | ✓ |
| paloma_macro (bpb) | 1.072 | 1.085 | -0.013 | ✓ |
...
| Mean Open-book | ... | ... | ... | ... |
```

Save each baseline's diff to `$LOG_DIR/COMPARISON_vs_<baseline_label_sanitized>.md`.

Also generate a **summary table** at `$LOG_DIR/COMPARISON_SUMMARY.md` showing only the Mean rows (Open-book, Closed-book NL, Aggregate, Math, Code) against ALL baselines side-by-side:

```
| Mean row | <new> | C5-v6 final | Δ | C5-v4 final | Δ |
|---|---:|---:|---:|---:|---:|
| Mean Open-book      | 0.434 | 0.429 | +0.005 | 0.413 | +0.021 |
| Mean Closed-book NL | 0.252 | 0.246 | +0.006 | 0.241 | +0.011 |
| Mean Aggregate      | 0.359 | 0.354 | +0.005 | 0.341 | +0.018 |
| Mean Math           | 0.082 | 0.078 | +0.004 | 0.071 | +0.011 |
| Mean Code           | 0.183 | 0.196 | -0.013 | 0.142 | +0.041 |
```

Update state: `{stage: "done", comparison_files: [...]}`.

## Stage 6: surface to user

Push notification with:
- "Chain done: <label> trained + evaluated + filled into §3"
- One-line summary per baseline:
  - "vs C5-v6 final: Mean Open-book +0.005, Mean Code -0.013 — replay-strict slightly better on NL, slightly worse on Code"
  - "vs C5-v4 final: Mean Open-book +0.021, Mean Code +0.041 — beats C5-v4 across the board"
- Paths to each per-baseline file + the summary file.

# State file format

`$LOG_DIR/CHAIN_STATE.json` — single source of truth so a fresh session can resume mid-chain:

```json
{
  "stage": "prelaunch" | "training" | "training_done" | "training_failed" | "eval_fired" | "eval_blocked_no_nodes" | "section3_filled" | "done" | "aborted_prelaunch",
  "run_tag": "c5v6-strict-8n",
  "label": "c5v6_strict_step14671",
  "baselines": ["C5-v6 final", "C5-v4 final"],
  "started_at": "2026-06-18T20:07:23 PDT",
  "log_dir": "logs/multinode_c5v6-strict-8n_20260618_200724",
  "config_spec": {
    "init_from": "checkpoints/1_4b_c5v3_phase1/8dtdcear/step-14671",
    "tokens": 1.539e10,
    "chinchilla_ratio": 11.0,
    "hparams": {"lr": 3e-4, "wd": 0.1, "batch": 256, "seq": 4096, "steps": 14672},
    "data_mix_pct": {"text": 70.0, "code": 24.0, "markup": 6.0}
  },
  "pre_launch_samples_file": "logs/multinode_c5v6-strict-8n_20260618_200724/PRE_LAUNCH_SAMPLES.md",
  "ckpt": "checkpoints/1_4b_c5v6_strict_phase2/vg3ez4v4/step-14671",
  "completed_at": "2026-06-19T04:32:00 PDT",
  "eval_runners": [
    {"kind": "v2", "node": "gpu-dy-p4d24xlarge-1", "log_path": "logs/v2_..._<ts>.log", "started_at": "..."}
    /* one entry per kind */
  ],
  "filled_cells": 31,
  "comparison_files": [
    "logs/.../COMPARISON_vs_C5-v6_final.md",
    "logs/.../COMPARISON_vs_C5-v4_final.md",
    "logs/.../COMPARISON_SUMMARY.md"
  ]
}
```

# Recovery from a dropped session

If a fresh agent session starts and finds an in-flight `CHAIN_STATE.json` under any `logs/multinode_*_*/`:
- If `stage == "prelaunch"`: re-print spec + samples + baselines, re-prompt for confirmation. If the samples file already exists, link to it instead of regenerating.
- If `stage == "training"`: re-arm Monitor on node-0 log; proceed to Stage 2 when training finishes.
- If `stage == "training_done"`: skip directly to Stage 3 (eval fire).
- If `stage == "eval_fired"`: re-arm Monitors on the captured eval log paths; proceed to Stage 4.
- If `stage == "section3_filled"`: proceed to Stage 5 (comparison) using the `baselines` array.
- If `stage == "done"`, `"training_failed"`, `"eval_blocked_no_nodes"`, or `"aborted_prelaunch"`: nothing to do (terminal states).

The user can also explicitly resume via `/train-and-eval --resume <log-dir>`.

# Anti-patterns

- **Don't auto-add a new §3 column or §2 row.** Those are editorial decisions (footnote marker, paper description). The skill fills cells into a column whose name was passed via `--label`; if the column doesn't exist yet in EVALUATION.md, fill fails with a clear message and the user adds the column manually then re-runs the fill step.
- **Don't skip Stage -1.** The pre-launch step is not optional. It enforces three rules the user has corrected the agent on multiple times: (1) read 10 samples before any training run, (2) state hypothesis + baselines before launching, (3) confirm the comparison reference actually matches the experimental claim.
- **Don't proceed without baselines.** `--compare-against` is required (≥1). Running training-then-eval without naming what the run is being compared against produces results that drift into limbo — the whole experiment was *to compare against something specific*; if we don't write that comparison, the experiment is half-done.
- **Don't fire eval if training crashed.** Even partial checkpoints from a crashed run shouldn't be evaluated as if they were the intended final-step model.
- **Don't proceed to comparison if a reference column substring matches >1 column** — eval_section3.py's column-resolution is single-match. Fail loudly for every ambiguous baseline.
- **Don't trust launch-time node availability for eval-fire-time.** Always re-scan via nvidia-smi.
- **Don't compare against the same column the run was bug-derived from** without flagging the relation explicitly. Example: c5v6-strict was specifically designed to fix the shuffle-key bug in c5v6-final; the comparison is meaningful BECAUSE the reference is the buggy version. State the relation in the COMPARISON_*.md so the reader doesn't misread the diff as a clean A/B.

# Why this is a skill (not a launcher flag)

The user explicitly rejected `multi_node_launch.sh --auto-eval` for these reasons:

1. **Decouples eval pipeline from training.** When the eval task list changes (new task, new runner), only this skill changes — `multi_node_launch.sh` stays minimal.
2. **Keeps agent in session.** The skill runs inside a Claude session, so monitors / wakeups fire to the agent. A launcher running as a detached shell can't loop the agent back in.
3. **Easier to reason about as one task.** "train and eval c5v6-strict" is a single conceptual operation; one skill invocation, one state file, one resumable chain.
