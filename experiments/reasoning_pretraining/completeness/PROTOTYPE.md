# Prototype: what "completeness-augmented" pretraining text looks like

**Status: hand-authored ILLUSTRATION (no model output, no compute spent).** Purpose: make the
augmentation tangible and show how the *stopping rule* (see STOPPING_RULES.md) changes the output —
including the distribution-shift risk. At scale these would be model-generated (recipes in
LITERATURE_REVIEW.md); here they're written by hand to pin down the target format.

Each example shows the **source** (implicit, as-is web text) and 3 augmentation levels:
- **A1 — single-hop enthymeme completion** (stopping rule (f)/(b): state only the one non-default,
  non-common-ground premise). Closest to natural text → lowest distribution-shift.
- **A2 — bounded-depth entailment chain** (rule (a)+(d): expand to depth k≈2–3, verifier-gated).
  EntailmentBank `[BECAUSE]…[INFER]` style. The "code-like completeness" target.
- **A3 — full regress (anti-example)** (no stopping rule): shows why you can't do this — pedantic,
  off-manifold, and still not "complete" (the Carroll point).

---

## Example A — the canonical case (Dongwei's Alice sentence)

**Source (implicit):**
> The sidewalk was wet, so Alice brought an umbrella.

**A1 — single-hop enthymeme completion** *(state the one suppressed premise that licenses "so")*
> The sidewalk was wet, so Alice brought an umbrella. *(A wet sidewalk suggests recent or likely rain,
> and an umbrella keeps off rain — so Alice was preparing for rain.)*

**A2 — bounded-depth entailment chain (depth ~3, verifier-gated)** *(EntailmentBank-style)*
> Claim: Alice brought an umbrella because the sidewalk was wet.
> [P1] A wet sidewalk is evidence it rained recently or may rain soon.
> [P2] Rain gets people wet; people generally prefer to stay dry.
> [P3] Umbrellas reduce exposure to rain.
> [INFER P1+P2] Alice had reason to expect rain and to want to avoid getting wet.
> [INFER +P3] Therefore bringing an umbrella is a reasonable way to stay dry → she brought one.

**A3 — full regress (anti-example — DO NOT generate)**
> Alice is a person; people persist through time; umbrellas are physical objects; objects are solid;
> water makes surfaces wet; gravity pulls rain downward; Earth-like weather exists; the sentence is
> literal not sarcastic; English is used conventionally; … *(never terminates, drifts far off the
> text manifold, and STILL isn't "complete" — Carroll's regress. This is why a stopping rule is
> mandatory, not optional.)*

---

## Example B — real DCLM doc (neutral, verbatim from dclm_5000docs.jsonl)

**Source (implicit):**
> Dell should retire the ST2220T and replace it with a capacitive touch version… I bought the ST2220T
> and ended up returning it because of the bezel and the lack of pen support. If capacitive pens are
> good enough… I'd buy it at double this price.

**A1 — single-hop enthymeme completion** *(make the "because" premises explicit)*
> …I returned it because of the bezel and the lack of pen support. *(A thick bezel interrupts the
> touch surface at the screen edge, and without pen support you can't do precise stylus input — both
> defeat the point of a touch display, which is why those two flaws were enough to return it.)*

**A2 — bounded-depth chain**
> Claim: the reviewer returned the ST2220T.
> [P1] The ST2220T is a touch display; its value is accurate direct/stylus input.
> [P2] A large bezel creates a dead border, breaking edge touches.
> [P3] No pen support removes precise stylus input.
> [INFER P1+P2+P3] The two flaws each undercut the display's core value → net value too low → returned.

**Note (distribution-shift):** A1 reads like something a careful writer might actually add; A2 is
visibly "code-like / proof-like" and drifts from how web text is written. **That drift is the central
empirical risk** — it may help structured reasoning while hurting plain-LM perplexity (the same
code↔NL tradeoff we measured in the code-budget ladder). The experiment must measure both.

---

## What this pins down for the plan
1. The augmentation target format is concrete: **inline parenthetical single-hop (A1)** vs **explicit
   bracketed chain (A2)** — these are two distinct treatment arms differing by stopping rule.
2. The **stopping rule is the primary experimental axis** (A1 shallow-natural vs A2 deep-structured).
3. Distribution-shift vs reasoning-gain is the key tension to measure (NL/perplexity vs reasoning evals).
4. At scale, A1/A2 are produced by an LLM augmenter (Quiet-STaR / WRAP / latent-thoughts style) — model
   choice + cost is an open decision (see EXPERIMENT_PLAN.md).
