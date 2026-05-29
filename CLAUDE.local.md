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
- **Print PST timestamp before each meaningful tool action, not just at the start of a turn.** For chained operations (multiple tool calls, lots of text between them), run `TZ='America/Los_Angeles' date '+%H:%M:%S %Z'` before each significant step. Reason: the user may be asleep during long sessions and needs to scrub the transcript to see exactly when each step happened ("step 5 ran at 02:45" vs "this turn started at 02:00, somewhere in here step 5 ran"). Compact `%H:%M:%S %Z` is enough unless we cross midnight. Skip for trivial single-line acknowledgments.

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
- Do NOT write anything that isn't directly backed by evidence. If you can't point to the specific source (paper figure, code, log, data), don't write it.
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
- **ALWAYS inspect actual content of training and eval data whenever discussing them.** Before making any claim about what a dataset contains, what a benchmark tests, why scores differ across tasks, or what a model is learning — sample at least 2-3 actual examples and read them. Names and descriptions can mislead (e.g. "NL reasoning benchmark" sciq vs piqa look identical by name but sciq provides a passage with the answer literally in it while piqa requires commonsense from weights — completely different mechanisms). Use the per-example samples (e.g. `samples_*.jsonl` from lm-eval) when explaining benchmark behavior, and inspect the actual jsonl/parquet content when describing training data. Never reason about data behavior from the dataset name alone.

## Paper reading rules
- When summarizing a paper, read the FULL paper including appendix. Key evidence often lives in appendix figures, not the main text.
- Read the actual results/figures, not just the abstract. Abstracts editorialize and overstate.
- NEVER claim two papers conflict or agree without reading the relevant results sections of BOTH papers.
- Present what the paper shows (numbers, figures, experimental conditions). Do not construct narratives or frameworks on top.
- When comparing papers, state the specific experimental differences (model size, data, eval tasks, proportions) before drawing any conclusion.
- If you only read part of the paper, say exactly which pages/sections you read — do not present partial reading as a full reading.
- Do not invent "confounds" or "explanations" for why papers disagree. Present the differences and let the user interpret.
- **When making any claim about what a paper says — its method, data, training setup, results, framing — READ the relevant section and QUOTE the exact text.** Do not paraphrase from memory, summary notes, or by projecting from another paper's approach. If a claim about paper X cannot be backed by a verbatim quote from paper X, do not make the claim. State "I don't have the quote — let me read it" and read.
- Do not project one paper's mechanism onto another (e.g., do not assume Aryabumi's synthetic code is "textbook-style with NL explanations" because Phi does that — read what Aryabumi actually says).

## Experiment design rules
- BEFORE launching ANY experiment, EXPLICITLY DISCUSS each of the following with the user, IN THE CHAT, and wait for confirmation. Do NOT just write them in a script comment. Do NOT launch until the user has acknowledged each one.
  1. **Hypothesis being tested.** State the falsifiable claim — what would success look like, what would failure look like.
  2. **Why this hypothesis.** What observation or prior result motivates testing it. If the goal is to fix a problem, name the problem and the proposed cause.
  3. **Why this specific configuration tests the hypothesis.** Justify each non-default choice (data, hyperparams, eval set) by how it relates to the hypothesis. If a config matches a reference run, name the reference run and confirm that reference run actually exhibits the behavior we want to study.
  4. **Data.** Exact dataset, revision/hash, total tokens, source path. Do NOT use vague names ("DCLM") — name the specific subset.
  5. **Hyperparameters.** Every non-default param, including: LR, WD, schedule, min_lr_ratio, batch size, seq len, steps, epochs, data_seed, optimizer betas, warmup, max_grad_norm. Cross-check against the reference run if there is one.
  6. **Eval sets.** Which datasets, why each one. Eval sets must enable the comparison the hypothesis demands. If the hypothesis is "match konwoo's loss", we must eval on the SAME sets konwoo evaluated on, not adjacent ones.
  7. **What result would confirm the hypothesis** and **what result would refute it.** If you can't say in advance, the experiment is not well-designed.
- Format: present these as a clear numbered list in chat. Mark anything you are unsure about. Pause and ask before proceeding to the next step.
- Critical anti-pattern to avoid: replicating a config that already exhibits the bug you want to fix. If model X with config C has problem P, replicating config C will reproduce problem P — that does not fix P. To fix P, pick the config of the model that DOES NOT have P.
- When the user's goal is "fix behavior X" (e.g. fix looping), the reference run to match is the one that DOES NOT exhibit X, not the one that does.

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
- **For ANY background process expected to run >30 min, immediately arm BOTH a Monitor and a ScheduleWakeup before doing anything else.** They're complementary:
  - **Monitor** = event-driven, real-time. Wide grep filter on per-step DONE markers AND every crash-shaped string (`Traceback|FAILED|SIGBUS|RuntimeError|OOM|Killed|ChildFailedError|assert`). Catches "something happened" within seconds. Use `tail -F log | grep -E --line-buffered ...` and set `persistent=true`.
  - **ScheduleWakeup** = time-based backstop, 20–30 min cadence. Catches "nothing happened" failures (process hung silently, killed without traceback, log went quiet). Independent of Monitor filter coverage.
  - **Why both:** Monitor goes silent if your grep doesn't match an unexpected error mode; Wakeup goes silent until the next tick. Together: instant alert on known failures, bounded delay on unknown ones.
  - **Anti-pattern (don't do this):** launch the job, switch attention to doc edits or another side task, forget to arm watchdogs. This caused a 5-hour-20-minute gap on a dead eval job on 2026-05-28. Arm watchdogs *first*, do unrelated work *after*.
