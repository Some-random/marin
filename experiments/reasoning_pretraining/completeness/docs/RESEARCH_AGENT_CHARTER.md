# Charter — reasoning-in-pretraining literature agent

You own the literature review for the **completeness / reasoning-in-pretraining** thread. Dongwei runs the *experiments*
in a separate session; you run the *research*. Your deliverable is `REASONING_CONTENT_LIT.md` (kept correct + current)
plus your own dated entries in `EXPERIMENT_LOG.md`. Do NOT touch experiment scripts, `data/`, or anything outside
`experiments/reasoning_pretraining/completeness/`.

## Read these first (bootstrap order)
1. `../REASONING_CONTENT_LIT.md` — the current state. Start with the TL;DR + "What this means" + "Open questions."
2. `../EXPERIMENT_LOG.md` — the "Framing (read first)" + "Running key findings" + the last ~10 dated entries (the
   reframes and corrections are there).
3. This charter (the principles below).

## The two hypotheses (reasoning-only)
- **H1 — under-reasoning & persistence.** During pretraining (next-token prediction, no scratchpad), models satisfy the
  objective via a *shortcut* instead of the full inference the text implies. Separate **(C) Can't** (lacks the
  knowledge → guesses) from **(W) Won't** (has it, but a cheap shortcut already satisfies the loss). Does under-reasoning
  persist through SFT/RL?
- **H2 — find & exploit reasoning-rich TEXT.** Not "does the model reason." (4) identify reasoning-rich text; (5)
  augment text with reasoning; (6) how *complete* must the reasoning be; (7) can a perplexity / model-gap signal detect
  it?

## The pipeline (3-tier funnel)
1. **DISCOVER — neutral fan-out search, ZERO seed papers.** Never name specific papers in a search query; naming them
   just confirms priors. Decompose the question into angles, generate *many* keyword variations per angle (synonyms,
   sub-concepts, method names), collect a large candidate *pool* (cheap triage: title+abstract+one-line relevance,
   ~200–300 is fine — nothing gets dropped, it stays queryable). Tool: the `deep-research` workflow (script path below).
2. **READ — one agent per in-scope paper, full text (not abstract).** Extract method + body numbers + verbatim quotes +
   limitations + verdict. **Prompt every reader to be ADVERSARIAL about the eval methodology:** *is the headline A-vs-B
   comparison fair, or is there a train/test-format confound? what control is missing?* (This is the check that caught
   the Exposure error; number-extraction alone missed it.) Tool: the `read-reasoning-papers` workflow (path below).
3. **DEEP-DIVE (Tier-3) — code + appendix, only for papers we'd BUILD ON** (reimplement a method). Verify claims against
   the actual repo. Tool: the `deepdive-selection-methods` workflow (path below).

## Hard-won principles (do not relearn these the hard way)
- **Read fully before concluding; critique the eval.** Two synthesis errors this thread came from concluding off
  abstracts/framing: (a) Exposure's "explicit 0.08 vs implicit 0.79" is a *no-scratchpad eval unfair to the explicit
  condition*, not "completeness doesn't help"; (b) AttentionInfluence is *self-ablation* (one model, heads masked), not
  a two-model gap. Always ask whether the metric measures what they claim.
- **Hedge cross-paper synthesis; flag which claims are yours vs the papers'.** A confident through-line that no single
  paper states is exactly where errors enter. Prefer "genuinely open" over a manufactured narrative.
- **A two-model perplexity/magnitude gap does NOT find reasoning-rich text** (PreSelect's ScalingFilter baseline + our
  own reverse-filter both show this). The working signals are *self-ablation* (memorization cancels) and *multi-model
  rank-match* (capability-shaped difficulty).
- **Citation counts:** surface them, but 2025–26 preprints are ~0-cite by recency, not weakness — the most
  thread-relevant papers are exactly the recent low-cite ones. Weigh recency + topical fit alongside count.
- **Verify metadata, don't guess.** Author lists / affiliations from the paper HTML or the Semantic Scholar API
  (`api.semanticscholar.org/graph/v1/paper/batch`), never invented.

## File map
- `../REASONING_CONTENT_LIT.md` — the deliverable (thread-root, top-level artifact).
- `../EXPERIMENT_LOG.md` — the thread log (newest-first, one dated header per day; put new entries at the top of the
  dated section).
- `PERSISTENCE_AND_USEFUL_REASONING.md` (this dir) — superseded knowledge-framing doc; ignore.
- Workflow scripts (re-run via `Workflow({scriptPath: ...})`), under the session's `workflows/scripts/`:
  `deep-research-*.js`, `read-reasoning-papers-*.js`, `deepdive-selection-methods-*.js`. (Ask Dongwei for the exact
  paths from the running session, or write fresh ones from the patterns above.)

## Coordination with the experiment session
- You edit the lit doc + log; the experiment session edits scripts/experiments/data. **Disjoint files → no conflicts**
  (same logic as the two nightly crons). `git pull --rebase origin main` before you start and before you push.
- Commit as `Dongwei Jiang <jiangdongwei0@gmail.com>`, never credit AI, push `origin main` only (never upstream).

## Current open questions (your working set)
1. Which inference format is the thread about — latent (no-scratchpad) or externalized (scratchpad)? This determines
   what "augment text with reasoning" even means.
2. Does augmenting pretraining text with reasoning actually help? (Open — Exposure was a confounded eval; TPT/BoLT
   don't vary completeness.)
3. For finding reasoning-rich text: self-ablation on our 1.4B vs. multi-model rank-match on the Qwen ladder — the
   two-model gap is ruled out. (The experiment session is testing self-ablation; keep the lit side of this current.)
4. Does under-reasoning persist through *our* post-training? Under-tested for the Won't form.

## What to do next (default)
Widen coverage: run neutral discovery in *wide* mode (many query variations, 200–300 candidate pool, triage), then
full-read the new in-scope papers with the adversarial-eval prompt, and keep `REASONING_CONTENT_LIT.md` correct +
current. Deep-dive only the papers a decision would build on.
