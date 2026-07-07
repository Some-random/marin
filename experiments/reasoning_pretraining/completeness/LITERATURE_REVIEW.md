# Literature review — reasoning-completeness transfer (code → text)

Integrates 5 research threads (2026-07-04). Detail + quotes + citation counts in the thread files:
CODE_REASONING_LIT.md · PRETRAIN_AUGMENTATION_LIT.md · ENTAILMENT_LIT.md · STOPPING_RULES.md · DATASETS.md.
All quotes were fetched from arxiv HTML/PDF by research agents; re-verify load-bearing ones against source PDFs before any writeup.

## The three things the literature settles

**1. Code helps reasoning — and the active ingredient is STRUCTURE, not executability.**
Aryabumi 2024 (+8.2% NL reasoning from code). Waheed 2025: structural perturbations hurt far more than
semantic; **pseudocode/flowcharts recover most of the benefit.** Zhao 2026 (10T-token controlled): with
Code-NL controlled, standalone executable code is "not a general reasoning enhancer"; gains are "better
explained by cross-domain structured reasoning traces, such as code-text and math-text mixtures, rather
than by executable code alone." Kim 2024: code = explicit multi-step **state tracking**. Petty 2024
scoping: code helps structured-output/math, can **HARM** syntax/morphology/knowledge — **the same code↔NL
tradeoff we already measured in our code-budget ladder.**

**2. Augmenting text with explicit reasoning at pretraining time WORKS — and is already an active recipe family (2025).**
BoLT (infer latent thoughts, prefix, EM bootstrap — "outperforms training on an equivalent amount of
unique raw tokens"), TPT (append expert thinking, ~3× data efficiency), Reasoning CPT (reconstruct
author's hidden reasoning, gains grow with difficulty + cross-domain). Dominant cost = **teacher inference**,
not student training. Main failure mode outside math/code = **no cheap correctness signal** (verification gap).

**3. The STOPPING RULE (how explicit is explicit enough) is the unsolved problem.**
Carroll's regress: appending premises never certifies completeness — needs an outside-the-chain certificate
(rule/verifier) or a social floor (common ground/default). Code has a natural floor (primitives); text
doesn't. Selection-Inference is candid: "we halt after a fixed number of steps… Addressing the issue of
halting is left for future work." **No purely-textual method has a learned, dynamic stopping rule.** Six
implementable stopping rules enumerated in STOPPING_RULES.md.

## Honest novelty assessment (corrected)
My first-pass claim ("nobody has run this experiment") was **too strong** — TPT/BoLT/Reasoning CPT/MIND
are essentially "augment text with reasoning, train normally," and they work. What is **genuinely open**,
and where this project can contribute, is narrower and sharper:

- **(N1) Completeness/stopping-rule as a CONTROLLED VARIABLE.** Existing methods use one fixed "expert
  thinking" / "latent thought" style. Nobody sweeps *completeness depth* (shallow single-hop → deep
  bounded chain) and measures the reasoning-gain vs distribution-shift (perplexity) tradeoff as a function
  of the stopping rule. This is the core of Dongwei's idea and it is untested.
- **(N2) Reasoning-augmentation as a CODE SUBSTITUTE, measured on the code↔NL tradeoff.** The augmentation
  literature evaluates on math/reasoning benchmarks in isolation; the code-transfer literature never tries
  to *replace* code with reasoning-explicit text. Slotting this into our existing code-budget ladder
  (§3 EVALUATION.md) directly asks: *can completeness-augmented text buy the code→reasoning transfer
  WITHOUT paying code's cost to NL/perplexity?*
- **(N3) A near-zero-cost first test using data we already have.** openthoughts + openwebmath are already
  tokenized — a "found completeness" arm tests the core hypothesis before spending any generation compute.

## Reusable assets
- **Recipes:** BoLT (2503.18866), TPT (2509.20186), Reasoning CPT (2505.10182) — static gen pass + normal trainer.
- **Structure templates:** EntailmentBank (multi-step chains, `[BECAUSE]/[INFER]`), ProofWriter (depth-graded + abduction/missing-premise), allenai/art (natural single-hop, verified).
- **Grounding atoms:** ATOMIC/COMET (commonsense floor).
- **Already local (free):** openwebmath, openthoughts_filtered/flat, phi_1_5.
- **Eval sets (multi-step reasoning):** EntailmentBank, ProofWriter, aNLI/art, QASC, OpenBookQA + our §3 suite.

## Key risk carried into the design
Distribution-shift from natural text: over-explicit augmentation may help reasoning while hurting plain-LM
perplexity — literally the code↔NL tradeoff again. The experiment MUST measure both (reasoning evals AND
dclm/paloma bpb), exactly as the ladder does.
