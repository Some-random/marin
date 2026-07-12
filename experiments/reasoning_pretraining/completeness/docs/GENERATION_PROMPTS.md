# Rationale generation prompts (what the generator saw)

The single most important confound in this thread is **whether the rationale generator was shown the
continuation / target it would later be scored against.** This file records the exact prompt + inputs for
every rationale dataset, so it's never guesswork.

---

## 1. DCLM completeness — `complete_dataset.jsonl` (complete + incomplete)

**Generator:** Claude subagents, 60-way, via workflow `complete-reasoning-mine` (`wf_65774a0b-5a8`, session 6fd63298).
**Input each agent saw per doc:** `{id, context, continuation}` — **BOTH the context AND the real continuation.**
**Target-awareness:** **YES, the generator saw the continuation** (it had to, to decide `keep` and to extract
`target` as a verbatim span of the continuation). **Mitigation in the prompt:** it was told to write the
rationale *"all from the CONTEXT only; never copy the target's words"* and to end *"one step before the target."*
So: target-visible, but with an explicit context-only / no-copy instruction — imperfectly followed (e.g. doc
101447's rationale still names "Latakia," a place that appears only in the target). A downstream leak filter in
`perplexity_complete.py` drops any doc where >60% of the target's long words appear in the `complete` rationale
(catches gross lexical copying, not directional tailoring).

**Verbatim agent prompt:**
```
You are mining real DCLM web documents for COMPLETE MULTI-STEP reasoning. Read <DIR>/batch_<b>.jsonl — each
line is {id, context, continuation}: context = the earlier part of a real web doc; continuation = the real
text that follows.

For EACH doc, decide keep:
- keep=true ONLY if the continuation contains a CONCLUSION/CONSEQUENCE that genuinely requires a MULTI-STEP
  reasoning chain — at least THREE non-obvious inferential steps — to derive from the context, where the
  context gives the premises but leaves the CHAIN implicit. NOT a single-hop inference, NOT a restatement,
  NOT new facts / narration / lists, NOT a doc that already spells out its own reasoning.
- Be STRICT. Most docs are keep=false. Quality over quantity.

If keep=true, produce three things (all from the CONTEXT only; never copy the target's words):
- target: the shortest verbatim span from the START of the continuation that states the conclusion. MUST be
  an exact substring of the continuation.
- complete: the FULL reasoning chain deriving the target from the context — a numbered list where EVERY
  load-bearing step is explicit, each following from the previous plus common knowledge, NO gaps, ending one
  step before the target. This is the COMPLETE rationale (aim for 3-6 steps).
- incomplete: the SAME chain but with the 1-2 MOST load-bearing MIDDLE steps DELETED — keep the opening setup
  and the near-final step but remove the critical connecting inference(s), leaving a real logical GAP.

Write to <DIR>/out_<b>.jsonl, one JSON per line:
{"id":<id>,"keep":true/false,"target":"...","complete":"1. ...\n2. ...","incomplete":"1. ...\n3. ..."}.
```

---

## 2. DCLM prose rewrite — `complete_dataset_prose.jsonl`

**Generator:** Claude subagents, 6 batches (session 6fd63298, 2026-07-10). **Style-only rewrite** of the
numbered rationales above into natural prose. **Input each agent saw:** `{id, context, target,
complete_numbered, incomplete_numbered}` — so it saw the target too, but its job was ONLY to restyle the
existing rationale (numbered → prose), **preserving content and the incomplete-gap exactly**, no new reasoning.
Inherits the target-awareness of #1 (it's the same content, restyled). Rules given: natural flowing prose, no
numbered lists / "Step"/"Therefore" scaffolding, do not copy the post-blank/target words, keep every step for
`complete` and the same deleted steps for `incomplete`.

---

## 3. Winogrande — `winogrande_rationales.jsonl` (principle / full / complete)

**Generator:** Claude subagents, 10 batches (session 6fd63298, 2026-07-10).
**Input each agent saw per example:** `{idx, sentence, option1, option2, answer, correct}` — where `sentence`
is the **full Winogrande sentence including the post-blank continuation** (e.g. "…always got the easier
cases") and `correct` **states the answer explicitly**.
**Target-awareness:** **YES, fully** — the generator saw both the continuation and the answer, by design.
Controlled by the three rungs: `principle` = the general rule with **no entity names and no answer stated**
(text-level leak-free, though the generator knew the answer); `full` = one sentence naming the answer;
`complete` = full multi-step chain naming the answer. Rules given: natural prose, no lists, `principle` names
no entity and never reveals the correct option, do not copy the exact post-blank suffix words.

---

**Takeaway for reading any result:** none of these generators were blind to the target (the truly-blind one is
`generate_reasoning.py`, the separate *implicit* experiment). So `complete − placebo` / `complete − base`
magnitudes are upper bounds inflated by target-awareness. The within-comparisons that are NOT affected:
`complete − incomplete` (both equally target-aware) on DCLM, and `principle` vs `full`/`complete` on
Winogrande (principle's text is answer-free).
