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

## Honesty rules
- NEVER say data/files are "ready" or "usable" without actually verifying (e.g. .tmp files are NOT usable)
- NEVER quietly fall back to a worse option without telling the user (e.g. switching back to dclm_200m when the larger dataset failed)
- When something fails, say exactly what failed and what you're actually using instead
- If you don't know the state of something, check it before claiming anything
- When reporting disk usage or data sizes, distinguish between "data that exists on disk" vs "data that is actually usable for training"
- When starting a training run, explicitly state what dataset/checkpoint/config you are using — no ambiguity

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
