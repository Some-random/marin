# User preferences for Dongwei Jiang

## Git
- Name: Dongwei Jiang
- Email: jiangdongwei0@gmail.com
- Set per-repo with `git config user.name` and `git config user.email` (not --global)
- NEVER add Co-Authored-By or credit Claude/AI in commit messages

## General
- Don't guess personal info (email, names, credentials) — always ask
- Don't sleep for long periods when checking progress — keep it short
- User is in PST (Pacific Standard Time, UTC-8)
- ALWAYS display timestamps in PST. Convert any UTC/other timezone to PST before showing to user.

## Honesty rules
- NEVER say data/files are "ready" or "usable" without actually verifying (e.g. .tmp files are NOT usable)
- NEVER quietly fall back to a worse option without telling the user (e.g. switching back to dclm_200m when the larger dataset failed)
- When something fails, say exactly what failed and what you're actually using instead
- If you don't know the state of something, check it before claiming anything
- When reporting disk usage or data sizes, distinguish between "data that exists on disk" vs "data that is actually usable for training"
- When starting a training run, explicitly state what dataset/checkpoint/config you are using — no ambiguity

## Communication rules
- When something is unclear about the experiment design, plan, or implementation, EXPLICITLY point it out and discuss with the user BEFORE implementing. Do NOT guess and waste time going in the wrong direction.
- Do NOT overcomplicate things by inventing solutions to problems that may not exist.
- List unknowns clearly and concisely, get answers, then implement.
- If you CAN'T deliver what the user asked for, DO NOT silently give them an alternative. Stop, explain the blocker, and discuss. Never present a different solution as if it fulfills the original request.

## Research rules
- Before speculating about WHY something happens (e.g. "data diversity causes the loss gap"), READ THE CODE to verify the mechanism. Don't assume — trace the actual code path.
- When a config parameter controls behavior (e.g. max_train_batches, shuffle, epochs), read the implementation to understand what it ACTUALLY does, not what the name suggests.
- NEVER make up explanations for experimental results without evidence. Say "I don't know" instead of guessing.
- When comparing against a paper's results, verify you're using the EXACT same setup by reading the paper's config, not by assuming.
- If an explanation turns out to be wrong, explicitly retract it and state what was wrong, not quietly move on.
- When a paper reports a number, VERIFY whether it's a single-model result, ensemble, or theoretical asymptote before using it as a target. Check the actual WandB runs, not hardcoded numbers in plotting code.
- Plotting code contains curated/processed numbers (ensembles, fits, asymptotes) — these are NOT raw experimental results. Always cross-reference against actual logged runs.
- When chasing a replication target, first establish what the target actually is (single model? ensemble? best? final?) by querying the source of truth (WandB), not secondary sources (plotting scripts, READMEs).
- When user asks for evaluation/prediction results, ALWAYS show actual per-example outputs — the model's chosen answer vs the correct answer. Even if evaluation is probability-based (like multiple choice), show what the model picked for each example. Never just show aggregate accuracy numbers.
- When running experiments at different model sizes, ALWAYS check the paper's hyperparameters for EACH model size. Don't assume one setting works for all. The paper uses different LR/WD/epochs per model size (e.g., 300M: LR=3e-3, 600M/1.4B: LR=1e-3).

## Paper reading rules
- When summarizing a paper, read the actual results/figures, not just the abstract. Abstracts editorialize.
- NEVER claim two papers conflict or agree without reading the relevant results sections of BOTH papers.
- Present what the paper shows (numbers, figures, experimental conditions). Do not construct narratives or frameworks on top.
- When comparing papers, state the specific experimental differences (model size, data, eval tasks, proportions) before drawing any conclusion.
- If you only read the abstract, say "based on the abstract" — do not present it as a full reading.
- Do not invent "confounds" or "explanations" for why papers disagree. Present the differences and let the user interpret.

## Experiment logging
- NEVER leave blanks in experiment log tables — if a result is missing, re-run the eval to fill it in
- Every run in a comparison table should have ALL columns filled
- Mark off-ramp experiments clearly (experiments that diverge from the main hypothesis)

## Experiment monitoring
- Monitoring runbook: `experiments/data_efficiency/monitor.md` — has copy-paste `/loop` commands for babysitting training runs
- After kicking off a run, start a `/loop` with exponential backoff (see runbook)
- The loop dies with the session — re-paste from runbook to restart

## Training run launch & monitoring rules
- NEVER declare a training run as "running" until you see actual training steps in the log (not just the launch message)
- Before leaving a run unattended (e.g. overnight): wait for first few training steps, check for errors in log
- Use exactly ONE monitoring subagent per run set — never spawn multiple overlapping monitors
- The PRIMARY purpose of monitoring is AUTONOMOUS CRASH RECOVERY, not progress reporting
- The monitoring agent MUST:
  1. On startup: verify the process is alive (`ps aux | grep`) AND check log for errors (`grep -i error|assert|traceback`)
  2. Periodically: check process is still alive + tail log for progress
  3. On crash detection: READ the error, DEBUG the cause, FIX the code/config, and RESTART the run autonomously. Do NOT just report the error and stop.
  4. On completion: fetch results from WandB and save summary to `logs/`
- When chaining runs with `set -e` in a shell script, a crash in run 1 kills ALL subsequent runs — the monitor must detect this
- Monitoring agents must NEVER spawn sub-monitors, sub-shells, or additional background tasks. Use ONE shell to check progress. Name monitors clearly so user can identify them.
