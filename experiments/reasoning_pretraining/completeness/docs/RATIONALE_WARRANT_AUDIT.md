# Rationale-Warrant Audit — how much DCLM text actually warrants an added rationale?

**Run:** 2026-07-05 · **Method:** 30-agent classification workflow (`dclm-rationale-warrant-audit`,
`wf_0494a236`) · **Numbers below are recomputed from the workflow journal, not from memory.**

## The question (why we ran this)

The completeness-of-reasoning idea is: augment pretraining **text** with the implicit reasoning it leaves
out, so a text-trained model learns *complete* reasoning the way code teaches it. But that only makes sense
for text that **has** latent reasoning. Bolt a "reasoning chain" onto a product listing, a news blurb, or an
opinion rant and you're **manufacturing** reasoning that isn't in the source — noise at best, spurious
inferences at worst. So before scaling any augmentation, measure the denominator: **of raw DCLM web docs,
what fraction genuinely warrant an added rationale?**

This is the "found-completeness gate" (novelty **N3** in `EXPERIMENT_LOG.md`): use the classifier as a **filter** —
augment only the reasoning-bearing subset, leave the rest untouched — rather than a blanket transform.

## Method (verified)

- **Corpus:** the 3,000 DCLM docs we had augmented with Qwen2.5-32B (`data/dclm_aug_qwen32b.jsonl`),
  classified on each doc's `context` field (first ~65% of the raw doc, first ~900 chars shown to the classifier).
- **Fan-out:** 30 agents × 100 docs each = all 3,000, in parallel.
- **Per doc:** a strict binary + a dominant doctype (8 buckets that must sum to the batch size).
  - **WARRANTS** = at least one *genuine inferential move* — a claim/conclusion that follows from earlier
    content by non-trivial reasoning (a real argument step, a causal/explanatory link, a justification), such
    that to follow the text a reader must make an inference the text leaves implicit.
  - **DOES NOT WARRANT** = information/narrative you follow *without* inference: a fact list, a description,
    sequential news, a product listing, a table, nav/boilerplate, or opinion/dialogue that merely asserts.
    Plus junk/fragmentary text.
  - **Strict rule:** the mere presence of "because / so / therefore" is **not** enough — the inference must
    be real and non-obvious; when genuinely unsure → DOES-NOT-WARRANT (conservative).

## Result (verified from the journal)

- **347 / 3,000 = 11.6%** of DCLM docs warrant a rationale.
- **Robust:** per-agent warrant fraction spread is tight — **min 6%, median 12%, max 18%** across the 30
  independent classifiers. This isn't one model's quirk.

| doctype | count | % of corpus |
|---|---:|---:|
| narrative_or_news | 924 | 30.8% |
| opinion_or_dialogue | 739 | 24.6% |
| factlist_or_reference | 626 | 20.9% |
| **argument_or_analysis** | 188 | **6.3%** |
| **causal_or_explanatory** | 182 | **6.1%** |
| other | 148 | 4.9% |
| boilerplate_or_nav | 101 | 3.4% |
| howto_procedure | 92 | 3.1% |
| **TOTAL** | **3,000** | 100% |

**Internal consistency check:** the two reasoning-bearing doctypes — `argument_or_analysis` (6.3%) +
`causal_or_explanatory` (6.1%) = **12.4%** — track the 11.6% warrant fraction almost exactly. The docs that
warrant a rationale *are* the argument/causal docs; the classifier isn't finding "hidden reasoning" scattered
through narrative and opinion. If you also count `howto_procedure` as reasoning-adjacent, the generous
ceiling is **~15.5%**.

## What warrants vs. what doesn't (real docs from the run)

**WARRANTS — a genuine implicit inference to make explicit:**
- *"By issuing stock, the company is carved into additional slices, making each piece worth less… called
  dilution…"* → multi-step causal chain (issue stock → more slices → each worth less → dilution → share
  price falls) the reader must reconstruct.
- *"The statute as applied in this case conflicts with federal law…"* → the legal holding requires inferring
  **why** the state ban is preempted (it undercuts a federal leverage strategy) — not stated outright.
- *"A standard CD-R holds 703MB — about 737m characters — so two discs hold 1.474bn. That would only be 59
  characters per record…"* → quantitative deduction: capacity + record count → chars/record → a conclusion.
- *"television is a pulsating blue light, which is why from outside a house it looks like the living room is
  glowing blue…"* → explicit causal mechanism for why a fake-TV device fools burglars.

**DOES NOT WARRANT — followable with no inference:**
- *"Local officials closed a Newtown, Conn. elementary school following a threat… Classes were scheduled
  to begin at…"* → sequential news reporting; events in order, nothing implicit.
- *"Easy Workout Routines / 10 Reasons Why Spring Is The Best Time To Lose Weight / …"* → listicle nav/boilerplate.
- *"Obama and his administration, if given 4 more years, will 'help' this country right over the cliff…"* →
  opinion rant; asserts a stance, no non-trivial inferential move.
- *"Cups are 30% thicker… with four holes for better air flow so you can stack harder and faster!"* →
  product listing; "so you can" is a marketing feature claim, not latent reasoning.

## Implications

1. **~7 of every 8 DCLM docs are not reasoning-bearing.** Blanket augmentation (add a chain to *every* doc)
   would manufacture reasoning for ~88% of docs that have none. This quantitatively confirms the standing
   intuition that most normal web text isn't reasoning — the completeness idea has to be **selective**.
2. **The classifier is a FILTER, not a transform.** Keep the ~1/6 reasoning-bearing subset; augment only
   those; pass narrative / opinion / fact-list / boilerplate through unchanged. This is N3.
3. **Yield (doc side):** at 11.6%, producing *N* augmented reasoning docs means classifying ~**8.6×N** raw
   DCLM docs first. The **token** yield (what matters for filling a 30%-of-mix slot on a ~15B-token run) is
   **not yet measured** — needs avg-tokens-per-warranting-doc, TBD. Don't estimate it; grep it.

## Caveats (honest limits of this number)

- **Classified on the `context` prefix (first ~65%, ~900 chars), not the full doc.** A doc could carry its
  reasoning in the tail we didn't show. This can only *under*-count warrants.
- **Measured on the pre-filtered 32B set** (docs already restricted to 400–2000 chars, <12 newlines), i.e.
  the fraction among *moderately clean prose*, **not raw crawl**. Boilerplate/junk was partly excluded
  upfront, so **11.6% is if anything an over-estimate** of the fraction over the full raw DCLM stream.
- **Single classifier family (Claude agents).** A second, non-Claude classifier on the same 3,000 docs is a
  cheap robustness check — noted, **not yet run** (and not greenlit).
- **`howto_procedure` (3.1%) is a judgment edge.** Procedures have structure but the "reasoning" is often
  procedural, not inferential — we did not fold them into the warrant count. Worth a deliberate ruling.

## Next (not launched — for review)

- Measure **token yield** of the warranting subset (not just doc %), to size a real augmentation slot.
- Optional **second-model cross-check** of the 11.6%.
- Rule on whether `howto_procedure` counts as warrant.
- The real test is downstream: does augmenting *only* the filtered reasoning-bearing subset transfer better
  than augmenting everything (or nothing)? That needs the training experiment, not another classification pass.

*See also: `OVERNIGHT_RESULTS.md` (the augmentation run + completeness-check), `SURPRISING_CASES.md`
(why the zero-shot perplexity probe failed), `EXPERIMENT_LOG.md` (the thread's running index).*
