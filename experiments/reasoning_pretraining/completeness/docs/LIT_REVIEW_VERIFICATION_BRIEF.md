# Verification brief — independent audit of the 143-paper lit review

**For:** a separate verification agent (fresh session). **Written:** 2026-07-23 by the lit-review session.
**Your deliverable:** a discrepancy report (severity, doc line refs, record citations). **Do NOT edit any file** —
report first; Dongwei decides what gets fixed.

## What was done (the pipeline you are auditing)

The deliverable under audit is `experiments/reasoning_pretraining/completeness/REASONING_CONTENT_LIT.md` — a
143-paper literature review on reasoning in pretraining (two hypotheses: H1 under-reasoning/shortcuts/persistence;
H2 identify/augment/completeness/loss-signals for reasoning-rich text). Recent commits: `15c13aef6`, `8e7df7023`,
`fa8683d9e`, `a8e2dea75`, plus a patch commit dated 2026-07-23/24 folding in the final 9 verifications.

Pipeline: (1) zero-seed discovery, 533-candidate triaged pool → `docs/DISCOVERY_POOL_2026-07-23.md`; (2) three read
passes — 24 + 40 papers at an early **HTML-tier** protocol (later shown unreliable), then 80 papers at a hardened
**PDF-tier** protocol (download PDF, read page-by-page incl. appendices, figures inspected visually, released code
checked when load-bearing); (3) page-by-page **PDF verification** of the 64 HTML-tier reads — NOTE a workflow bug
here: index-based assignment caused 9 papers to be silently skipped and 8 double-verified; the 9 were re-verified
in a follow-up run with explicit per-agent assignment; (4) 7-bucket **synthesis** over the corrected records,
auditing ~40 provisional claims (~18 survived, 11 revised, 6 weakened, 5 killed); (5) full **doc rewrite** by
7 writer agents + assembly; (6) a patcher pass folding the last 9 verification results into the doc.

Read-depth tiers across 143 papers: **55** read (HTML) + independently PDF-verified · **78** single-pass PDF-tier
(**single-witness — never independently audited; this is the biggest open risk**) · **9** read (HTML) +
late-verified · **1** abstract-only (TechRxiv, bot-walled; flagged in its entry).

Raw evidence (session-tied paths; may not exist if the session's tmp was cleaned — the workflow journals under the
session dir are the durable copy):
- Per-paper read records: `/tmp/claude-1000/-fsx-users-dongweij-marin/8eafed95-a507-4a25-a339-dfd512fa5c7b/scratchpad/reads/` (24), `.../reads2/` (40), `.../reads3/` (80)
- Verification records: `.../scratchpad/verify/` (64 + 9)
- Bucket syntheses (claims audit, doc-corrections lists): `.../scratchpad/synth/*.json`
- Workflow journals: `/fsx/users/dongweij/.claude/projects/-fsx-users-dongweij-marin/8eafed95-a507-4a25-a339-dfd512fa5c7b/subagents/workflows/<runId>/journal.jsonl` — runIds: reads `wf_13d49562-ffa`, `wf_4006ecb6-289`, `wf_e87a77d2-8da`; verifications `wf_a05ac667-dc2`, `wf_de1bb509-eaf`; synthesis `wf_777fa169-9d7`; writers `wf_e0c0abd9-e1e`

## Measured error base-rates (calibrate your suspicion)

- HTML-tier records: 21/64 material corrections on verification — including one **fabricated table pairing**
  (Beyond-Code "College Math 41.59→65.73"), wrong-version reads (papers revise fast; v1 vs v3 changed headline
  numbers), figure misreads, and **overclaimed adversarial critiques in both directions** (papers accused of
  missing controls that exist in their appendices).
- The late-9 verification: 3/9 material — two were our critiques being refuted by appendix content, one a
  version-drift case (When-LLMs-Stop v1 said 14 models/61%; v3 says 15 models/63%).
- PDF-tier records (the 78): error rate **unmeasured**. That is your job #1.

## What to check (priority order)

1. **Spot-audit the single-witness tier.** Randomly sample ~10 of the 78 PDF-tier papers (records in `reads3/`;
   entries in the doc). For each: fetch the LATEST arXiv version, read it fully (PDF, appendices, figures), compare
   against both the record and the doc entry. Report a measured material-error rate; if ≥ ~10%, recommend full
   verification of the tier.
2. **Version check.** For every full-writeup paper, confirm the version read is still the latest on arXiv (abs-page
   dateline). 2025–26 preprints revise fast and two version-drift errors already slipped through.
3. **Doc-vs-record fidelity.** Sample ~15 load-bearing numbers from the TL;DR and bucket synthesis sections; trace
   each to its per-paper record and its (Table/Fig) source. Flag any doc number with no record backing.
4. **Stale-claim grep.** Killed claims must not survive anywhere. Grep the doc for: `+4.09`, `14 models`,
   `61% (5 steps)`, `18.43`, `41.59→65.73`, `barely moves`, `not significant on real LLMs`, `[unverified`,
   `+5.2` (Logic-Corpus math transfer — correct value +0.7). Any hit is a bug.
5. **Contradiction honesty.** The doc is required to present genuine cross-paper disagreements two-sided (e.g.
   chains-produce-reasoning vs -templates; RL-erases vs RL-preserves shortcuts; textbook-style best vs worst) and
   to tag thread inferences `[our inference]`. Flag any manufactured resolution or untagged inference.
6. **Coverage arithmetic.** 146 entries = 143 distinct papers + 3 cross-reference stubs (Exposure → full writeup in
   H2.6; AttentionInfluence and PreSelect → full writeups in H2.4). Every paper in the pool's read batches appears
   exactly once; the abstract-only paper is explicitly flagged in its entry.
7. **The freshest patch.** The last commit folded 9 verification results in (papers: ProRL 2505.24864, RLVR-Boundary
   2510.04028, When-LLMs-Stop 2605.00817, ScalingFilter 2408.08310, ProcKN 2411.12580, Echo Chamber 2504.07912,
   Attrib 2505.19949, Inefficient-Reasoning 2507.05362, CompCollapse 2605.26789). Check their doc entries against
   `.../scratchpad/verify/<id>.md` — especially: When-LLMs-Stop now 15 models/63%→20%/−23.85pp (v3); Boundary's
   crossover demonstrated (Table 3) and eval sampling stated (App C.2); CompCollapse's main prompt PERMITS a
   reasoning block.
8. **Process hygiene.** Commits authored as Dongwei Jiang (no AI credit), pushed to origin main only;
   `EXPERIMENT_LOG.md` headers are single-day.

## Pitfalls we hit — avoid repeating them

- **Never trust HTML renderings for numbers** — always the PDF, latest version, figures inspected visually.
- **Reader/verifier agents overstate read depth** — require a precise read_scope (what was NOT read) and treat any
  vague scope as unread.
- **Workflow `args` may arrive as a JSON string** — parse-guard (`typeof args === "string" ? JSON.parse(args) : args`).
- **Never assign work by "read entry at index i" of a shared file** — agents miscount; embed each agent's full
  assignment in its own prompt (this caused the 9-skipped/8-duplicated bug).
- **Budget:** stagger heavy fan-outs across 5-hour session windows (a ~16M-token burst locked the user out for 2h);
  workflows resume from cache, so staggering is free. Check headroom before any >2M-token fan-out.
- Adversarial critique needs the same verification as extraction — half our material errors were **overclaimed
  critiques**, not misread results.
