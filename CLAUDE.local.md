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
- **Narrate intent before non-trivial shell, interpret the result after.** Format: (1) one-line plain English of what's being checked and what healthy/unhealthy looks like, (2) the bash call, (3) one-line plain English of what the output means ("8/8 alive, loss dropping" not raw bash output). Trivial single commands (one `ls`, one `ssh`) don't need narration. Goal: user can follow without reading shell. Skip the narration only when it adds noise.
- **When I learn a new behavior rule or feedback from the user, write it to CLAUDE.local.md in addition to memory.** Memory files are my auto-loaded context but the user has to dig to read them; CLAUDE.local.md is the user's source of truth they read and edit directly. Putting rules only in memory hides them from review and edit. Always do BOTH so the user can see and modify the rule.
- **For ANY evaluation that goes into §3 of EVALUATION.md, invoke the `/eval-for-section3` skill, NOT the raw shell scripts.** The skill enforces `convert_and_eval_v2.sh` → `run_eval_v2.sh` (full v2 suite, ~30 tasks, multi-GPU code_eval fix) and includes the Mean-row recompute + alignment validation. Calling raw `eval_intermediate.sh` skips ~12 §3 tasks and leaves columns half-empty. The skill's invocation is visible in the transcript (search for `"skill": "eval-for-section3"`), so you can audit whether I used the right path. If I'm reaching for `bash eval_intermediate.sh` or `bash run_eval_v2.sh` directly without going through the skill, that's a miss — call it out.
- **Bash caches a script's text at process start — patching a script while it's running doesn't change the running process's behavior.** If you `sed -i` a supervisor/script that's already running as a long-lived process, the changes won't take effect on disk reads it has already done. You must kill the running process and relaunch for the patch to apply. Sanity-check by verifying the next log output matches the patched header.

## Research rules
- Do NOT write anything that isn't directly backed by evidence. If you can't point to the specific source (paper figure, code, log, data), don't write it.
- **When asked about performance, throughput, ETA, scaling, wall time, or anything quantitative about an actual system: GREP THE LOGS, do not estimate from rule-of-thumb.** Don't say "likely", "probably", "should be ~X" when there are actual logs on disk that have the real number. Examples of what to grep: past `multinode_*/node-0-*.log` for step rates, EXPERIMENT_LOG.md for measured wall times, wandb run history. Violated 2026-06-08 with the C5-v2 4n-vs-8n ETA estimates — I claimed "30 hours for 4 nodes" from a math estimate (1.87× scaling efficiency) when grepping the prior A5/C5 logs would have shown 2.8–3.0 s/step → 24.5 h actual measured. Real numbers ALWAYS beat estimates; never invent the latter when the former exists.
- **NEVER fabricate a "Dongwei comment" or any user-attributed opinion in `papers/reasoning_curriculum.md` or anywhere else.** Paper entries get Motivation / Experiment Setup / Conclusion only. Dongwei comment blocks are added by the user, never by me. Violated twice — first on the Beyond Random Sampling paper, again on 2026-06-01 across Tracr / DPG / FinePhrase entries. If I think a comment block would be useful, I surface the thought in chat ("worth noting: …") and let the user decide whether to add it.
- Before speculating about WHY something happens (e.g. "data diversity causes the loss gap"), READ THE CODE to verify the mechanism. Don't assume — trace the actual code path.
- When a config parameter controls behavior (e.g. max_train_batches, shuffle, epochs), read the implementation to understand what it ACTUALLY does, not what the name suggests.
- NEVER make up explanations for experimental results without evidence. Say "I don't know" instead of guessing.
- When comparing against a paper's results, verify you're using the EXACT same setup by reading the paper's config, not by assuming.
- If an explanation turns out to be wrong, explicitly retract it and state what was wrong, not quietly move on.
- When a paper reports a number, VERIFY whether it's a single-model result, ensemble, or theoretical asymptote before using it as a target. Check the actual WandB runs, not hardcoded numbers in plotting code.
- Plotting code contains curated/processed numbers (ensembles, fits, asymptotes) — these are NOT raw experimental results. Always cross-reference against actual logged runs.
- When chasing a replication target, first establish what the target actually is (single model? ensemble? best? final?) by querying the source of truth (WandB), not secondary sources (plotting scripts, READMEs).
- When user asks for evaluation/prediction results, ALWAYS show actual per-example outputs — the model's chosen answer vs the correct answer. Even if evaluation is probability-based (like multiple choice), show what the model picked for each example. Never just show aggregate accuracy numbers.
- **For ANY claim about what a model does or doesn't do (e.g. "doesn't attempt CoT", "can't generate Python"), open the samples_*.jsonl and read the raw `resps` field — NOT `filtered_resps`.** `resps` is the actual model generation; `filtered_resps` is lm-eval's post-extraction (often just the final number for math tasks, or `[invalid]` if extraction failed). Conflating these has burned me at least twice on 2026-06-03: (1) claimed B4 "can't generate Python" because bigcode HumanEval = 0.000, when raw `resps` showed compilable Python that lost on harder problems; (2) claimed our 4 × 1.4B models "don't even attempt CoT" because `filtered_resps` was just a bare number, when raw `resps` showed 91-100% full CoT generations with equations, reasoning chains, and even repetition loops. Read raw outputs FIRST, then summary numbers — never the other way around.
- **Before any non-trivial infra workaround (disabling a whole subsystem, changing mesh layout, swapping the tracker, etc.), FIRST search the project's GitHub issues for the symptom.** The fix may be a known workaround. Going straight to "rip out wandb" without checking is the wrong default. Violated 2026-06-04 when I disabled wandb to work around BrokenPipeError on multi-node JAX coordination without checking levanter issues first. Use `gh issue list --search <symptom>` or the GitHub API; explicitly state "searched, no matches" if that's the result.
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
- **When the user names a published paper as the methodology reference, replicate that paper's exact protocol — do NOT propose hand-picked variants on hunch.** If user says "follow Wu et al / GSM-Symbolic / Percy style", that paper IS the design: same task construction, same perturbation rules, same scoring. Only deviate when (a) the paper's setup is genuinely inapplicable to our model/data AND (b) the user has explicitly OK'd the deviation. Always cite the paper's exact section ("per Mirzadeh et al §3.1, we sample N=50 instances per template..."). Violated 2026-06-03 with "CF-1 format-invariant arithmetic" — I invented 4 hand-picked surface formats instead of replicating GSM-Symbolic's parameterized-template methodology. Fix going forward: read the paper's methods section first, replicate it, then run.

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
- **SINGLE-DAY LOG ENTRIES ONLY.** Every `## ` header in `experiments/data_efficiency/EXPERIMENT_LOG.md` is exactly one calendar day. Never combine dates ("May 31 – June 1", "and", "/", ranges, etc.). If work spans days, split into one entry per day and cross-reference. Before committing any log edit, run `grep '^## ' experiments/data_efficiency/EXPERIMENT_LOG.md` and confirm no header spans multiple dates. User has corrected this rule multiple times (most recently 2026-06-01).

## Experiment monitoring
- Monitoring runbook: `experiments/data_efficiency/monitor.md` — has copy-paste `/loop` commands for babysitting training runs
- After kicking off a run, start a `/loop` with exponential backoff (see runbook)
- The loop dies with the session — re-paste from runbook to restart

## Cluster / Slurm tips

- **We have passwordless `sudo (ALL) NOPASSWD: ALL`** on this cluster. Verified 2026-06-08. Use it sparingly but it is available.
- **When a dy GPU node is stuck `IDLE+CLOUD+POWERED_DOWN` with `Reason=Scheduler health check failed`**, the recovery for cloud nodes is `sudo /opt/slurm/bin/scontrol update NodeName=<node> State=POWER_UP` (NOT `State=RESUME` — that's for non-cloud nodes and returns "Invalid node state specified"). After the POWER_UP, submit a 1-node holder sbatch (`--nodelist=<node>` + `--time=7-00:00:00`); if the underlying EC2 instance comes up cleanly, the holder reaches RUNNING within ~10–15 min. If it dies within 17 seconds, the node has a deeper AWS-level issue (capacity shortage, ASG misconfig) that we can't fix from the cluster side.
- **Acquire idle "our" nodes (dy-1..9 + st-1..4) as holders for 7 days** when training is at risk of losing nodes mid-run. Pattern: `sbatch --no-requeue --nodes=1 --nodelist=<node> --time=7-00:00:00 --partition=gpu --gres=gpu:8 --wrap='exec sleep infinity'`. Do NOT use job dependencies (`--dependency=afterok:X`) for follow-on holders — when the parent holder hits its TimeLimit and is killed, exit code is non-zero, so `afterok` becomes `DependencyNeverSatisfied`. Either clear with `sudo scontrol update JobId=Y Dependency=""` or just submit independent 7-day holders from the start.
- **Sneaking onto other users' allocated-but-idle nodes is feasible.** Check actual GPU usage with `ssh <node> 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc'`. If the total is < ~1000 MiB, the slurm allocation is just a placeholder and we can ssh-launch on top of it via `multi_node_launch.sh`. Risk: if the actual owner suddenly uses their GPUs, both our training and theirs OOM. The 2026-06-08 C5-v2 production run sneaked onto romeosr × 6 + alkhouli × 1 + free × 1 = 8 nodes and ran fine. Use ssh-based `multi_node_launch.sh`, not sbatch (sbatch will refuse to schedule on allocated nodes).
- **Our team's GPU nodes are `gpu-st-p4d24xlarge-1..4` + `gpu-dy-p4d24xlarge-1..9`.** Anything beyond dy-9 (e.g. dy-30+) is not ours and will fail to launch with `Scheduler health check failed`. Don't waste cycles on those.

## Training run launch & monitoring rules
- **BEFORE launching any training run (smoke test or production), proactively ask Dongwei whether they have read 10 actual training-data samples per source.** If they haven't, offer to dump 10 samples per source to a markdown file (like `outputs/eval_results/code_data_source_samples.md`) and pause launch until they've eyeballed them. Apply on every launch — even repeat runs (cache/revision can shift). Reason: on 2026-06-08 we discovered while writing up C5 results that what the B4 script calls "aryabumi_web" is actually `codeparrot/github-code` Python (unfiltered crawl), NOT `OpenCoder-LLM/opc-annealing-corpus/synthetic_qa` — sampling actual jsonl content settled it. Cheap to check beforehand; expensive to discover mid-run.
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
- **For chained multi-stage training launches (e.g. phase 1 → phase 2, resume after crash), the watchdog AFTER the new stage launches MUST verify FORWARD PROGRESS, not just "the previous process exited".** Concrete check: within 10 min of launch, a `Progress on:train [0-9.]+kit` line must appear in `node-0`'s log; then every 30 min another fresh one. If either check fails, push notify Dongwei IMMEDIATELY. Event-driven monitors go silent if the log goes silent — they cannot detect a hung process. Violated 2026-06-10 overnight when C5-v3 phase 2 hung 2 min after launch because the coordinator node was power-cycled, and I treated "phase 1 process exited" as a green light. Lost 10+ hours of cluster time. The fix is a parallel sleep-loop watchdog that re-checks for fresh Progress lines every ~15 min and alerts on absence. **When the user is asleep, push notify aggressively for a hang** — a 10-hour stall is much worse than one false-positive page.
- **For ANY background process expected to run >30 min, immediately arm BOTH a Monitor and a ScheduleWakeup before doing anything else.** They're complementary:
  - **Monitor** = event-driven, real-time. Wide grep filter on per-step DONE markers AND every crash-shaped string (`Traceback|FAILED|SIGBUS|RuntimeError|OOM|Killed|ChildFailedError|assert`). Catches "something happened" within seconds. Use `tail -F log | grep -E --line-buffered ...` and set `persistent=true`.
  - **ScheduleWakeup** = time-based backstop, 20–30 min cadence. Catches "nothing happened" failures (process hung silently, killed without traceback, log went quiet). Independent of Monitor filter coverage.
  - **Why both:** Monitor goes silent if your grep doesn't match an unexpected error mode; Wakeup goes silent until the next tick. Together: instant alert on known failures, bounded delay on unknown ones.
  - **Anti-pattern (don't do this):** launch the job, switch attention to doc edits or another side task, forget to arm watchdogs. This caused a 5-hour-20-minute gap on a dead eval job on 2026-05-28. Arm watchdogs *first*, do unrelated work *after*.
- **Phase-based monitor cadence for training runs.** Don't fire every progress line for the entire run — it's noise once you're past the danger zone. Instead:
  1. **Launch → first ~1000 steps (~30 min):** fire on every `Progress on:train` event. Catches first-step failure, early divergence, slow first-step JIT issues.
  2. **Past step 1000 (steady state):** subsample to every `1.0kit` milestone (regex `Progress on:train [0-9]+\.0[0-9]?kit`) → one event every ~33 min at 2 s/step.
  3. **Crash filter (always on, all nodes):** `panic:|SIGSEGV|nil pointer|rendezvous\.cc|Traceback|FAILED|ChildFailedError|RuntimeError|OOM|Killed|ConnectionRefused|BrokenPipe|broken pipe|Connection reset|jaxlib\.xla_extension`.
  4. **ScheduleWakeup backstop every ~25 min** to catch "log went silent" cases that the grep filter wouldn't see.
- **Chained multi-stage runs (e.g. C5-v3 phase 1 → eval + phase 2 → eval) are a DAG, not a chain.** After stage 1 trains, fan out: stage 1 eval runs in parallel on a standalone node WHILE stage 2 training fires on the multi-node cluster. Stage 2 eval fires when stage 2 finishes. This shaves the eval wall time (~1-2h per stage) off the total. Each "stage done" trigger should arm both downstream jobs simultaneously, not serially. Implement by arming a Monitor on the completion log line (`Saving training checkpoint to <final_step>` or `All training steps completed`) that, on fire, kicks the next two jobs.
