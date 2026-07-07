# Stopping rules — "how explicit is explicit enough" (the core design knob)

Grounded in: Carroll's regress, frame/qualification problem, defeasible reasoning (Reiter),
Gricean implicature, common ground. Full sourcing in LITERATURE_REVIEW.md.

## The load-bearing lesson (Carroll, 1895)
**Appending premises NEVER certifies completeness** — the Tortoise defeats "add one more premise"
forever, because any inference *rule* can be re-demanded as a *premise*. Completeness must be
certified by something **outside the chain** (an inference rule / verifier) OR by a **social floor**
(common ground / default), never by "one more premise." This reframes every stopping rule.

Corollary: **code has a natural floor** (primitives / type system / machine instructions);
natural language does not — so we must *impose* one. The knob IS the stopping rule.

## Six implementable stopping rules
Tradeoff axis: **tractability ↔ completeness ↔ distribution-shift from natural text**
(over-explaining drifts pretraining data off the human-writing manifold).

| Rule | Concept | Tradeoff |
|---|---|---|
| **(a) Fixed depth k** — expand to hard depth k | truncate mutual-knowledge hierarchy | max tractable/deterministic; k arbitrary; shift grows with k. Use as a **safety cap**, not primary. |
| **(b) Common-knowledge atomic** — stop when premise is common ground (corpus freq / KB / classifier) | Grice Quantity + Clark common ground | **lowest distribution-shift** (where humans actually stop); but "common knowledge" fuzzy, audience-relative, no formal certificate. |
| **(c) Fixed vocab/ontology** — stop when premises use only closed predicate set | circumscription domain-closure | machine-checkable, guaranteed termination; ontology gaps → brittle, **high shift** for open-domain. |
| **(d) Verifier/entailment threshold** — add premises while NLI `premises ⊢ concl` < τ | Carroll-honest (rule lives in verifier, outside chain) | content-adaptive, closes the actual gap; verifier fallible, τ arbitrary, Carroll gap can reappear inside verifier. |
| **(e) Primitive ops / axioms** — decompose to fixed primitives (code analogy) | formal-system resolution | clean verifiable termination; NL rarely decomposes cleanly → **strongest shift**; good for math/code, poor for web. |
| **(f) Default / normal-case** — include premise iff omitting it (assuming normal case) changes the conclusion | Reiter defaults + qualification problem | **matches human text** (state defeaters not defaults), low shift, dense signal; needs a "what's normal" model (itself the qualification problem). |

## Recommended combination (agent's synthesis, [inference])
**Primary floor = (f)/(b)** — stop at non-default, common-ground premises so the augmentation stays
on the natural-text manifold — **gated by (d)** an entailment verifier to certify the gap is closed —
with **(a) a hard depth cap** as a runaway guard. (c)/(e) reserved for formal/closed domains.

## Why this matters for the experiment
The augmentation strategy IS a choice of stopping rule. Different rules → different data distributions
→ this is the primary experimental axis to test (e.g. shallow-natural (f/b) vs deep-formal (e) vs
depth-k sweep). Distribution-shift from natural text is the key risk: too-explicit text may hurt
plain LM/perplexity even if it helps reasoning (mirrors the code↔NL tradeoff we already measured).
