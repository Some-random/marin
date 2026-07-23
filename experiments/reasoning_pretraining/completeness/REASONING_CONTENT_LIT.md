# Reasoning in pretraining: under-reasoning (H1) & finding/exploiting reasoning-rich text (H2)

**Status: FULL READS DONE (2026-07-21).** 16 in-scope papers read end-to-end (workflow `wf_e16faf72-dc2`, one
agent per paper, full HTML/PDF, body numbers + verbatim quotes + author/venue confirmed); 4 more (TPT, BoLT,
RHO-1, Quiet-STaR) read earlier; 2 secondary surveys skipped. This version is written to be **read cold** — every
paper below has a plain-English "what it is / what they did / what they found / why it matters" writeup, not just
numbers. Provenance at the end. Supersedes the earlier abstract-only map.

---

## TL;DR — the reads support H1; on H2 two of my earlier confident readings were WRONG and are corrected below

Almost every paper supports **H1** (that language models take reasoning *shortcuts* instead of doing the full
inference, and this is baked in during pretraining). On **H2** — whether rewriting pretraining text with explicit,
complete reasoning helps — the evidence is genuinely **open**, and two of my earlier confident claims were wrong and
are corrected here (🔴 and 🟢). Three things to hold onto:

1. **🔴 The "Exposure" paper is NOT a counter-result to completeness (corrected 2026-07-23).** I earlier read its
   "explicit → 0.08, implicit → 0.79" as "spelling reasoning out completely doesn't help." That was a **misread**: the
   evaluation is *unfair to the explicit condition* because it forces a **single forward pass with no scratchpad** —
   the bridge entity is not allowed in the context. But explicit *training* teaches `P(answer | …bridge token… )`;
   it relies on the bridge token being physically present in the prefix, and the test deletes exactly that. So the
   explicit condition isn't shown to be *worse at reasoning* — it's tested in a format that removes what it learned to
   lean on. The paper never runs the fair comparison (let the explicit model emit the bridge, then score the final
   answer); the 2×2 of {implicit, explicit} training × {direct-answer, scratchpad} test is a **missing experiment** —
   only the direct-answer column is filled. **The defensible claim is narrow:** explicit compositional text does *not
   automatically compile into a **latent** (no-scratchpad) direct-answer computation* — it says nothing about explicit
   reasoning *with* a scratchpad. **What robustly holds** is *exposure-boundedness*: composition transfers only to
   entities that appeared in compositional pretraining contexts; atomic facts alone are not enough (97% 1-hop, ~1%
   2-hop for unexposed entities, invariant to model scale).

2. **🟡 Completeness helps at least one regime; the "it fails silent reasoning" side was the misread above.** The
   *Enthymeme* paper shows filling in unstated premises steadily improves formal logical-argument verification
   (0.53→0.73). The apparent "completeness fails latent composition" from Exposure was a **train/test-format
   confound**, not evidence about completeness. So the real axis is **latent (no-scratchpad) vs externalized
   (scratchpad) inference** — a question about *inference format*, NOT about "completeness good vs bad." Which one our
   thread targets is the open design question.

3. **🟢 The reverse-filter fix I proposed is itself wrong — a two-model gap is a *documented near-failure* (corrected
   2026-07-23, from a code-level deep-dive).** I said "re-run our filter as a weak-vs-strong two-model gap." Wrong on
   both counts: (a) our *original* reverse-filter's gold criterion (1.4B-high AND Qwen-low) was **already** a two-model
   gap, and it found *knowledge, not reasoning*; (b) PreSelect ran exactly this ("ScalingFilter" = big-vs-small
   perplexity difference) as a controlled baseline — it beats random by only **+0.4**, selects **short/easy junk**, and
   is **uncorrelated (Spearman 0.05)** with the signal that works. And AttentionInfluence — which I called a
   "cross-model gap" — is actually **self-ablation** (one model vs. itself with reasoning heads masked; memorization
   cancels because both losses come from the same weights). The two recipes that actually work are **not** a two-model
   gap: **(A)** self-ablation on our own 1.4B, or **(B)** multi-model *rank-match* on a same-family size ladder
   (Qwen 0.5B→72B).

**Bottom line (hedged — my cross-paper synthesis has been unreliable this pass):** H1 (under-reasoning is real,
pretraining-laid, persists) is well-supported. On H2, the naive "spell everything out → model stops shortcutting"
thesis is **neither supported nor cleanly refuted** — the one paper I cited against it was a confounded eval. What
robustly holds: reasoning-in-pretraining pays off and compounds (Front-Loading); composition is exposure-bound
(Exposure); a two-model perplexity gap does *not* find reasoning-rich text (PreSelect + our own result). The genuinely
open question: does augmenting pretraining text with reasoning help, and does it hinge on whether we want **latent**
(no-scratchpad) or **externalized** (scratchpad) reasoning at inference?

---

## The two hypotheses (corrected, reasoning-only)

**H1 — UNDER-REASONING AND ITS PERSISTENCE.** When predicting the next word, a model can get the word right by a
**shortcut** — matching a surface pattern, recalling a memorized association, or guessing plausibly — instead of
actually working through the multi-step inference the text implies. Call that **under-reasoning**. Two causes we
must keep apart: **(C) Can't** — the model lacks the knowledge/information to do the inference, so it guesses; and
**(W) Won't** — the model *could* do it, but a cheaper shortcut already satisfies next-word prediction, so it never
practices or learns the full inference. The claim: under-reasoning (especially the **Won't** kind) is learned in
pretraining and **persists** through fine-tuning and RL.

**H2 — FINDING & EXPLOITING REASONING-RICH TEXT.** Not "does the *model* reason" — rather, "does this *document*
contain reasoning, and can we use that?" Four parts: **(4) identify** reasoning-rich text; **(5) augment** text with
reasoning; **(6) completeness** — how fully must the reasoning chain be spelled out to help; **(7)** can a
**perplexity / weak-vs-strong-model gap** detect reasoning content?

---

## H1 — under-reasoning is real, and "Can't vs Won't" turns out to be a spectrum, not a switch

Your Can't/Won't split was the right lens, but the papers show it's not one clean line — there are at least four
genuinely different failure modes, and they matter because *data can fix some of them and not others*:

- **Won't (a cheap shortcut wins first).** *Grokked Transformers* trains a tiny model on synthetic facts and watches
  it learn: it first builds a **memorization** circuit that fits the training data by rote, and only much later — if
  you keep training far past the point of "done" (this is "grokking") — does it build a genuine **reasoning** circuit.
  The shortcut is the default; real reasoning is the expensive latecomer. *Bag of Heuristics* finds the same flavor
  in real models doing arithmetic: they never learn a real algorithm, just a pile of pattern-matching rules, and
  that pile forms early in pretraining and is *never* replaced.

- **A "Can't" that was manufactured by the training distribution.** The *Exposure* paper shows a model that knows all
  the individual facts (97% on single facts) yet can't chain two of them together — and making the model bigger
  doesn't help at all. It's not missing knowledge, and it's not a lazy shortcut hiding a usable skill; the ability to
  compose those particular facts simply was never created, because the training text never asked for it.

- **An architectural timing limit.** *Hopping Too Late* finds cases where the model knows fact A and fact B, wants to
  combine them, but literally runs out of layers: it figures out the intermediate answer too late in its own
  processing to still use it. (They prove this by surgically feeding a late-stage internal state back to an earlier
  layer — which fixes 66% of the failures.)

- **A capacity/coverage limit.** The *k-hop* paper shows deeper chains are learnable but the training data needed
  grows *exponentially* with the number of steps — so below some budget the model just guesses.

**The unifying H1 takeaway:** under-reasoning is genuine and mechanistically real, but "the model just can't reason"
is wrong (that framing got refuted). The accurate picture is *partial, shortcut-inflated, and bounded by what the
training data exposed and what the architecture can reach.* And critically: **making the reasoning explicit at
answer-time (chain-of-thought) reliably rescues it** — one paper shows silent composition at ~8% jumping to ~93%
once the model is allowed to write the intermediate step out. Our thread's question is whether we can bake that same
benefit into the *training text* instead of relying on it at answer-time.

**Persistence (H1's central claim) — the strongest single result.** NVIDIA's *Front-Loading Reasoning* is the
cleanest evidence that reasoning put into *pretraining* doesn't just survive later training — it **compounds**. A
model pretrained with reasoning data leads a plain model by ~9% before any fine-tuning, ~9.3% after fine-tuning, and
**~18.5% after RL** — the gap *widens* at every stage. And you can't cheat your way there with more fine-tuning:
doubling the fine-tuning data on the plain model still leaves it behind even the *weakest* reasoning-pretrained
model. Their line: front-loading reasoning "cannot be fully replicated by later-stage SFT, even with more data."

---

## H2 — how to find reasoning-rich text, and the completeness question that reshapes the thread

**(4) Identifying reasoning-rich text is a largely solved problem — three working recipes:**
- *AttentionInfluence* takes a small model, deliberately breaks its "retrieval" attention heads so it gets worse at
  reasoning, and flags the documents where that breakage hurts the most — those are the reasoning-heavy ones.
- *AutoDS* just asks a big model two yes/no questions ("is this mathematically intelligent? is it educational?") and
  keeps the documents it says yes to.
- *FineWeb-Edu* trains a cheap classifier to imitate a big model's 0–5 "educational value" rating. (Caveat: "educational"
  is deliberately grade-school-flavored and *down-weights* technical/arXiv content — so it's broader than "reasoning.")

**(6) Completeness — corrected reading (2026-07-23).**
- The *Enthymeme* paper: for explicit logical arguments, filling in the unstated premises **steadily helped**
  (0.53→0.73). Completeness helps *this* regime.
- The *Exposure* paper does **not** show the opposite. Its "explicit did nothing (0.08)" result is a
  **train/test-format confound** — the eval forbids the scratchpad that explicit training relies on (see 🔴 above), so
  it is *not* evidence that completeness hurts. What it actually shows: explicit traces don't auto-compile into
  *latent* one-pass computation, and composition is exposure-bound.
- The *Faithfulness* paper gives us a precise, borrowable definition of completeness — a reasoning chain is complete
  if it "screens off" the question from the answer (you can't reach the answer except through the chain) — plus a
  crucial warning: a chain can *look* complete while the model ignores it and answers directly. So **surface
  completeness isn't enough; the chain has to be one the model actually uses** ("necessity").

**(7) The perplexity-gap — corrected by a code-level deep-dive (2026-07-23).** I earlier wrote "our single-model
perplexity failed; re-run as a two-model weak-vs-strong gap." That is **wrong on both counts:**
- Our *original* reverse-filter already used a two-model gap (1.4B-high AND Qwen-low) and it found *knowledge, not
  reasoning*. PreSelect's appendix runs exactly this "big-vs-small perplexity difference" (they call it **ScalingFilter**)
  as a controlled baseline: **+0.4 over random, selects short/easy junk, uncorrelated (Spearman 0.05)** with the signal
  that works. The two-model magnitude gap is a documented near-failure — and it *explains* our own result.
- **AttentionInfluence is NOT a two-model gap.** It is **self-ablation**: one model vs. itself with its top-5%
  retrieval heads masked to uniform attention. Because both losses come from the *same weights*, everything memorized
  (frequency, n-grams) cancels in the difference — which is precisely why it isolates reasoning where a cross-model gap
  does not.
- The two recipes that actually work: **(A)** self-ablation on our own 1.4B (detect + mask its retrieval heads, score
  each doc by the loss gap; one model, no tokenizer mismatch, has a same-day go/no-go sanity check), or **(B)**
  multi-model **rank-match** on a same-family size ladder (Qwen 0.5B→72B; score by whether per-char loss ranks match
  the models' ability order — the *sign* over many pairs, not one magnitude gap). Exact recipes in the
  AttentionInfluence and PreSelect entries below.

---

## What this means for our thread (honest read, corrected 2026-07-23)

1. The naive "rewrite text to spell reasoning out completely → model stops shortcutting" thesis is **not supported and
   not cleanly refuted.** The one result I earlier cited against it (Exposure) was a confounded eval, so it counts as
   evidence *neither* way.
2. **Reasoning-in-pretraining is worth it** — it persists and compounds (Front-Loading), and explicit reasoning
   reliably helps *at answer-time* (chain-of-thought; SOCRATES ~8%→~93%). Whether that benefit can be baked into
   *training text* is the open question.
3. The real design fork is about **inference format**, not completeness-per-se: do we want the model to reason
   **latently** (no scratchpad — then Exposure says match the inference distribution *and* get the entities exposed
   compositionally) or **with a scratchpad** (chain-of-thought — untested by Exposure)? Faithfulness's "necessity"
   (does the model actually *use* the chain?) is the metric that cuts across both.
4. **On data-selection:** a two-model perplexity gap does NOT find reasoning-rich text (our result + PreSelect's
   ScalingFilter). The candidates worth testing are self-ablation on our own 1.4B (one model) or multi-model
   rank-match on a same-family ladder.

---

## Paper metadata — citations, venue, first/last-author institutions

*Citation counts via Semantic Scholar API (2026-07-22). **2026 preprints show 0 — too new to accrue citations, NOT a
quality signal**; weigh recency + topical fit alongside count (per our paper-reading rules). Institutions shown are
the **first** and **last** author's.*

| Paper (id) | First author (institution) | Last author (institution) | Venue | Cites |
|---|---|---|---|---:|
| **H1.1 — reasoning shortcuts** | | | | |
| Bag of Heuristics (`2410.21272`) | Yaniv Nikankin (Technion) | Yonatan Belinkov (Technion) | ICLR 2025 | 105 |
| When LLMs Stop (`2605.00817`) | Sailesh Panda (IIT Gandhinagar) | Mayank Singh (IIT Gandhinagar) | preprint 2026 | 0† |
| GSM-Symbolic (`2410.05229`) | Iman Mirzadeh (Apple) | Mehrdad Farajtabar (Apple) | ICLR 2025 | 591 |
| **H1.2 — latent multi-hop** | | | | |
| Latent Multi-Hop / Yang (`2402.16837`) | Sohee Yang (UCL / DeepMind) | Sebastian Riedel (UCL / DeepMind) | ACL 2024 | 207 |
| Hopping Too Late (`2406.12775`) | Eden Biran (Tel Aviv U.) | Amir Globerson (Tel Aviv U. / Google) | EMNLP 2024 | 97 |
| Grokked Transformers (`2405.15071`) | Boshi Wang (Ohio State) | Huan Sun (Ohio State) | NeurIPS 2024 | 90 |
| SOCRATES (`2411.16679`) | Sohee Yang (UCL / DeepMind) | Mor Geva (Google Research / Tel Aviv) | ACL 2025 Findings | 33 |
| k-hop needs data / Yao (`2505.17923`) | Yuekun Yao (Saarland U.) | Alexander Koller (Saarland U.) | EMNLP 2025 | 10 |
| **H1.3 — persistence** | | | | |
| Front-Loading Reasoning (`2510.03264`) | Syeda Nahida Akter (CMU / NVIDIA) | Bryan Catanzaro (NVIDIA) | preprint 2025 | 23 |
| Yue RL-beyond-base (`2504.13837`) | Yang Yue (Tsinghua) | Gao Huang (Tsinghua) | preprint 2025 | 924 |
| ProRL (`2505.24864`) | Mingjie Liu (NVIDIA) | Yi Dong (NVIDIA) | preprint 2025 | 156 |
| RLVR Boundary debate (`2510.04028`) | Xinhao Yao (Renmin U. / Ant) | Yong Liu (Renmin U.) | preprint 2025 | 10 |
| **H2.4 — identify reasoning-rich text** | | | | |
| AttentionInfluence (`2505.07293`) | Kai Hua (ByteDance Seed) | Ke Shen (ByteDance Seed) | preprint 2025 | 5 |
| PreSelect (`2503.00808`) | Kashun Shum (HKUST) | Junxian He (HKUST) | ICML 2025 | 20 |
| AutoDS / AutoMathText (`2402.07625`) | Yifan Zhang (Tsinghua) | Andrew C. Yao (Tsinghua) | ACL 2025 Findings | 26 |
| FineWeb-Edu (`2406.17557`) | Guilherme Penedo (HuggingFace) | Thomas Wolf (HuggingFace) | NeurIPS 2024 | 1029 |
| **H2.5/6 — augment + completeness** | | | | |
| **Exposure** (`2606.09338`) | Yannis Karmim (Inria) | Valentin Barrière (U. de Chile) | preprint 2026 | 0† |
| Faithfulness as Info Flow (`2605.24286`) | Jinghan Jia (Michigan State / Anthropic) | Eric Easley (Anthropic) | preprint 2026 | 0† |
| Enthymemes (`2603.06114`) | Xuyao Feng (UCL) | Anthony Hunter (UCL) | preprint 2026 | 0† |
| TPT (`2509.20186`) | Liang Wang (Microsoft Research) | Furu Wei (Microsoft Research) | preprint 2025 | 3 |
| BoLT (`2503.18866`) | Yangjun Ruan (U. of Toronto) | Tatsunori Hashimoto (Stanford) | preprint 2025 | 40 |
| Quiet-STaR (`2403.09629`) | Eric Zelikman (Stanford) | Noah D. Goodman (Stanford) | COLM 2024 | 319 |
| **H2.7 — perplexity-gap** | | | | |
| RHO-1 (`2404.07965`) | Zhenghao Lin (Xiamen U.) | Weizhu Chen (Microsoft) | NeurIPS 2024 | 126 |
| Perplexity Correlations (`2409.05816`) | Tristan Thrush (Stanford) | Tatsunori Hashimoto (Stanford) | ICLR 2025 | 54 |

† 2026 preprint — too new to have accrued citations.

*Reading the counts: the high-citation anchors (FineWeb 1029, Yue 924, GSM-Symbolic 591, Quiet-STaR 319, Yang 207)
are the field-defining papers; the most thread-relevant results (Exposure, Faithfulness, When-LLMs-Stop, Front-Loading)
are all 2025–2026 and thus low/zero-citation by recency, not by weakness — several are the sharpest evidence we have.*

---

# The papers, one by one (readable writeups)

## H1.1 — evidence that models take shortcuts instead of reasoning

### 📖 Arithmetic Without Algorithms: LLMs Solve Math With a "Bag of Heuristics"
Yaniv Nikankin (Technion) … Yonatan Belinkov (Technion) · ICLR 2025 · **105 citations** · `2410.21272`

**What it is.** A "how does the model actually do it?" study of mental arithmetic (e.g. `36 + 59 =` answered in one
shot, no scratch-work). The question: when a model gets arithmetic right, is it running a real algorithm, is it just
memorizing answer tables, or is it something else?

**What they did.** They opened up the model (Llama-3-8B, plus Pythia and GPT-J) and traced which neurons actually
drive the answer. They found a small set of neurons, each firing on a simple pattern — one fires when an operand is
in a certain range, another on operands ending in the same digit, another on multiples of some number — and the
model just *adds up* these little rules' votes to land on an answer. No carrying, no place value, no algorithm. They
confirmed these neurons are the real mechanism (they account for 96% of the model's arithmetic behavior; deleting
the ones relevant to a given problem drops accuracy by ~29 points). Then they replayed the model's *entire
pretraining history* (Pythia checkpoints) to see when this mechanism forms.

**What they found.** The "bag of heuristics" is there almost from the start of training and is **never replaced by a
real algorithm** — the same rough set of rules explains ~79% of the model's arithmetic ability at every checkpoint.
So the model found a cheap trick early, it worked well enough, and it never had any pressure to learn something
better.

**Why it matters here.** This is a clean, mechanistic example of your **Won't**: the model reasons by shortcut, the
shortcut is laid down early in pretraining, and continued training doesn't fix it. The sobering note for us — the
authors think fixing this "may require fundamental changes to training and architectures," which is a caution that
simply feeding better text may not dislodge an entrenched shortcut.

### 📖 When LLMs Stop Following Steps
Sailesh Panda … Mayank Singh (both IIT Gandhinagar) · 2026 preprint · **0 citations** (too new) · `2605.00817`

**What it is.** A stress test of whether models actually *follow a procedure* they're given, versus just landing on a
plausible final answer. The trick: they hand the model the complete step-by-step recipe *in the prompt*, so the
model is never missing any knowledge — the only question is whether it faithfully executes.

**What they did.** They generate arithmetic procedures of varying length (5 steps up to 95 steps), give the model the
full recipe and two numbers, and check both the final answer and whether the model actually walked through every
step. 15 models, ~55,000 problems.

**What they found.** As procedures get longer, accuracy collapses from **63% (5 steps) to 20% (95 steps)** — and the
reason isn't arithmetic mistakes, it's that the model **stops following the recipe partway through** (it "under-
executes": completing all steps drops from 71% to 47%, giving up early rises from 24% to 51%). Crucially this
happens *even though the complete procedure is right there in the prompt*, and it happens to reasoning/RL-tuned
models too.

**Why it matters here.** Two things for us. First, it isolates **Won't/execution** from **Can't/knowledge** — the
knowledge is literally supplied, and the model still fails. Second, it's a caution for the completeness thesis: even
a *maximally complete* chain sitting in front of the model doesn't get followed over long horizons. Completeness of
the text is not the same as the model using it.

---

## H1.2 — does the model reason "silently" (in one forward pass), or just recall?

This cluster is the richest, and it's the most directly relevant to our thread, because "silent multi-hop reasoning"
is exactly reasoning that a model must do *without writing anything down* — the kind that pretraining either installs
or doesn't.

### 📖 Do Large Language Models Latently Perform Multi-Hop Reasoning?
Sohee Yang (UCL / Google DeepMind) … Sebastian Riedel (UCL / Google DeepMind) · ACL 2024 · **207 citations** · `2402.16837`

**What it is.** Tests whether a model, asked something like *"the mother of the singer of 'Superstition' is ___"*,
internally does the two hops — first figure out the singer (Stevie Wonder), then find his mother — or just jumps
straight to a memorized answer.

**What they did.** Built 45,595 such two-hop questions and used interpretability probes to watch, inside the model,
(a) whether it recalls the bridge entity (Stevie Wonder) and (b) whether recalling it more strongly makes the final
answer more correct. Tested on LLaMA-2 at 7B, 13B, 70B.

**What they found.** The **first hop is real and gets better with size** (the model recalls the bridge entity in
~71–78% of cases, rising with scale). But the **second hop — actually *using* that recalled entity to get the answer
— is only moderate and does NOT improve with size** (stuck around 61–65% from 7B to 70B). So the model has the
pieces but frequently fails to connect them, and making it bigger doesn't help.

**Why it matters here.** Direct evidence of under-reasoning that isn't a knowledge gap: the bridge entity is *right
there internally*, unused. And because scale doesn't fix the second hop, it hints that the fix has to come from
*how the model is trained*, not from more parameters — which is at least consistent with a data intervention.

### 📖 Hopping Too Late: The Limitations of LLMs on Multi-Hop Queries
Eden Biran (Tel Aviv University) … Amir Globerson (Tel Aviv University / Google) · EMNLP 2024 · **97 citations** · `2406.12775`

**What it is.** A follow-up that asks *why* the second hop fails, focusing on the hardest, cleanest cases: questions
where the model provably knows both facts on their own but still blows the combined question.

**What they did.** They traced, layer by layer, where inside the model the bridge entity gets resolved and where the
second hop happens. Then they ran a surgical intervention ("back-patching"): take the model's internal state from a
*later* layer and paste it back into an *earlier* layer, then let it finish — to test whether the answer was
computable if only the second hop had started sooner.

**What they found.** The model resolves the bridge entity in *early* layers, but the second hop only starts in *late*
layers — sometimes so late that those layers no longer hold the knowledge needed to finish. It's a **timing/traffic
problem inside the network**. The back-patch (giving early layers the later information) fixes **up to 66%** of the
previously-wrong cases, proving the answer was reachable — the model just ran out of runway.

**Why it matters here.** This is a *third* category beyond Can't/Won't: the model knows the facts AND wants to
compose them, but the architecture runs out of layers. Data augmentation wouldn't directly fix this particular
mechanism — but the authors note that writing the intermediate step out explicitly (chain-of-thought) sidesteps it,
which is indirect support for externalizing the hidden step.

### 📖 Grokked Transformers are Implicit Reasoners
Boshi Wang … Huan Sun (both Ohio State University) · NeurIPS 2024 · **90 citations** · `2405.15071`

**What it is.** A controlled from-scratch study of *when* a transformer learns to reason silently versus just
memorize — training a small model on made-up facts so they can watch the whole learning process.

**What they did.** They train on synthetic facts and two task types: **composition** (chain two facts) and
**comparison** (is A bigger than B). They train far past the normal stopping point and watch the internal circuits
evolve, and they test on held-out combinations to see if the skill *generalizes* or is memorized.

**What they found.** Transformers *can* learn genuine silent reasoning, but **only through "grokking"** — training
far beyond the point where the training data is already fit (roughly 50× longer). Before grokking, generalization is
~9%; after, ~98%. Mechanistically they see two circuits: a fast **memorizing** circuit that forms first (the
shortcut) and a slow **generalizing** circuit that only wins later because it's more efficient. Twist: after
grokking, the model generalizes to *new* comparison problems but **still fails on new composition problems** — a
hard architectural limit. And frontier models (GPT-4, Gemini) score near random (~28–37%) on the hard version.

**Why it matters here.** A textbook demonstration of your Won't (memorize-first shortcut) *and* a hard Can't
(architecture can't generalize composition out-of-distribution). The uncomfortable note for the augmentation thesis:
the model can internalize reasoning *with no explicit chains in the data at all* — so explicit chains aren't strictly
necessary; the lever the paper found is the *ratio* of reasoning-examples to plain-facts in the mix.

### 📖 Do LLMs Perform Latent Multi-Hop Reasoning *Without Exploiting Shortcuts*? (SOCRATES)
Sohee Yang (UCL / Google DeepMind) … Mor Geva (Google Research / Tel Aviv University) · ACL 2025 Findings · **33 citations** · `2411.16679`

**What it is.** The most careful version of "does the model *really* reason silently, or is it cheating?" It builds a
test set specifically designed so the model *can't* get the answer by a memorized shortcut (e.g. because the start
and end entities never appear together in any document), and it only counts cases where the model provably already
knows both individual facts.

**What they did.** Built 7,232 two-hop questions, filtered out every shortcut, and measured "latent composability" —
how often the model chains the two facts silently — versus "chain-of-thought composability" where it's allowed to
write the middle step out. Tested ~41 models.

**What they found.** With shortcuts removed, silent composition is **terrible** — Claude 3.5 at 8.4%, GPT-4o at 7.6%,
Gemini Flash at 2.4% — *even though the model knows both facts*. Let the same model write the intermediate step out,
and it jumps to **~85–93%** (GPT-4o: 7.6% → 92.8%). They also show the shortcut inflation is real (scores are ~5×
higher if you *don't* filter shortcuts) and that it's wildly uneven: when the bridge is a *country* the model
composes ~83% of the time, when it's a *year* only ~6%.

**Why it matters here.** This is close to a *definition* of the problem our thread cares about: the model has the
knowledge but does not silently run the inference — and making the middle step explicit fixes it. It strongly
supports "explicit reasoning helps," while cautioning that merely having the facts present in text does *not* teach
silent composition (in their pretraining trace, silent 2-hop reasoning emerged for only ~11% of eligible cases).

### 📖 Language Models Can Learn Implicit Multi-Hop Reasoning, But Only With Lots of Data
Yuekun Yao (Saarland University) … Alexander Koller (Saarland University) · EMNLP 2025 · **10 citations** · `2505.17923`

**What it is.** Asks the quantitative version: *how much* data does it take to teach a model to do k-step reasoning
silently, as k grows?

**What they did.** Trained small GPT-2 models from scratch on synthetic k-hop reasoning (k = 2, 3, 4), sweeping the
amount of training data, and also derived a theoretical minimum on how many layers are needed.

**What they found.** It's learnable, but brutally data-hungry: 2-hop is easy, but the data needed **grows
exponentially with the number of hops** (4-hop needed up to 100× the base budget), and the required depth grows
linearly with hops. Below the needed data, the model just guesses (~1%, chance). The bright spot: a **curriculum**
(teach 2-hop, then 3-hop, then 4-hop) cut the 4-hop data requirement ~20× — ordering the reasoning by difficulty
beat dumping it all in uniformly.

**Why it matters here.** It reframes some "under-reasoning" as a plain **capacity/coverage** problem, not a removable
shortcut — a caution for us. But the curriculum result is encouraging: *how* you stage reasoning examples matters a
lot, which is an argument for thoughtful augmentation rather than raw volume.

---

## H1.3 — does the reasoning gap persist through fine-tuning and RL?

### 📖 Front-Loading Reasoning: The Synergy between Pretraining and Post-Training Data
Syeda Nahida Akter (CMU / NVIDIA) … Bryan Catanzaro (NVIDIA) · 2025 preprint · **23 citations** · `2510.03264`

**What it is.** The most direct test of your persistence claim: does reasoning ability have to be built in
*pretraining*, or can you add it later with fine-tuning? They pretrain an 8B model from scratch with vs without
reasoning data, then push both through the full fine-tuning + RL pipeline and compare at every stage.

**What they did.** Four base models (varying how much/what reasoning data went into the 1-trillion-token pretraining
mix), each then fine-tuned and then RL-trained, evaluated at each stage on reasoning benchmarks. They specifically
test the "catch-up hypothesis" — can extra fine-tuning let a plain base model catch a reasoning-pretrained one?

**What they found.** Reasoning-in-pretraining doesn't just persist, it **compounds**: the lead of the
reasoning-pretrained model over the plain one *grows* from ~9% (before fine-tuning) to ~9.3% (after fine-tuning) to
**~18.5% (after RL)**. And you can't catch up: **doubling** the fine-tuning data on the plain model gains only +4%
and still leaves it behind even the *weakest* reasoning-pretrained model. There's even a "latent" effect where high-
quality pretraining data shows no benefit until fine-tuning "unlocks" it.

**Why it matters here.** This is the paper that most directly backs "get reasoning into the base model, because you
can't paper over its absence later." Big caveat for *our specific* method: their "reasoning data" is
question-answer / long chain-of-thought fine-tuning-style data mixed into pretraining — **not ordinary web text
rewritten to expose its reasoning** — and their proxy for "quality" is basically how long the reasoning traces are.
So it strongly supports front-loading reasoning, but doesn't itself test "rewrite normal text to be more complete."

### 📖 The Debate on RLVR's Reasoning Boundary: Shrinkage, Expansion, or Both?
Xinhao Yao (Renmin University of China / Ant Group) … Yong Liu (Renmin University of China) · 2025 preprint · **10 citations** · `2510.04028`

**What it is.** Referees the fight over whether RL actually adds new reasoning ability or just sharpens what the base
model already had. (One camp — Yue et al. — says RL narrows the model to what it could already do; another — ProRL —
says long RL discovers genuinely new things.)

**What they did.** Analyzed the training dynamics mathematically and ran RL on a math model, tracking "how many
distinct problems can it eventually solve" (Pass@k) across training.

**What they found.** Both camps are right about *different phases*. Early RL **over-concentrates** the model and can
actually *shrink* the set of solvable problems (one benchmark drops from 100% to 91% coverage); only prolonged,
diversity-preserving RL **expands** it (another benchmark 47% → 67%). So for ordinary/short RL, the base model's
ability is the effective ceiling.

**Why it matters here.** Indirect support for reasoning-in-pretraining: since ordinary post-training is largely
bounded by what the base model can already do, it's better to get the reasoning into the base. Neutral on our
specific text-augmentation question.

---

## H2.4 — how to identify reasoning-rich text in a corpus

### 📖 AttentionInfluence: Weak-to-Strong Pretraining Data Selection
Kai Hua … Ke Shen (both ByteDance Seed) · 2025 preprint · **5 citations** · `2505.07293`

**What it is.** A training-free trick for finding reasoning-heavy documents *without* a classifier. **Important: despite
the title's "weak-to-strong," this is NOT two different models — it is one model vs. *itself with reasoning heads
disabled* (self-ablation).** Verified against code in a Tier-3 deep-dive (2026-07-23).

**What they did.** Take one small (1.3B) model. Detect its "retrieval heads" (attention heads that fetch information)
via a synthetic key-value needle task, then build a *weak* copy by setting those top-5% heads to **uniform attention**.
Score each doc by the relative loss gap `(L_masked − L_base) / L_base` between the crippled copy and the intact model,
ranked **within-domain**. Keep top 20%, upsample into the corpus, pretrain a 7B model. (The retrieval-head-detection
code is public — `nightdessert/Retrieval_Head` — and was verified; the scoring/masking loop has no released code.)

**What they found.** Selected data measurably improves reasoning benchmarks (**HumanEval +3.5, GSM8K +2.7, MMLU-Pro
+2.7 pts**), and leans more "reasoning" than an educational classifier (GPT-4o rated its picks 0.88 vs 0.52 on math).
The decisive internal check (Table 6): masking the top-5% retrieval heads **collapses** GSM8K 0.18→0.007 and BBH
0.32→0.04, while masking *random* heads barely moves them — so the ablation really is hitting reasoning machinery.
Caveats: within-domain-comparable only; also lifts pure-knowledge benchmarks.

**Why it matters here (corrected).** I earlier called this a "weak-vs-strong-*model* gap — the shape our reverse-filter
should have been." **Wrong.** The whole reason it works is that `L_masked` and `L_base` come from the *same weights*, so
everything the model memorized (frequency, n-grams) appears in **both** terms and **cancels** in the difference — the
only thing left is reliance on the reasoning heads. A two-*different*-model gap (our 1.4B-vs-72B) does the opposite:
nothing cancels, and the gap is dominated by what the 72B memorized differently = frequency/knowledge again. So this is
the **self-ablation** recipe **(A)** for our reverse-filter: run it on our *own* 1.4B (detect + mask its retrieval
heads, score by the gap). Cheap same-day go/no-go: replicate the Table-6 check on our 1.4B — if masking its retrieval
heads collapses GSM8K/BBH while random-head masking doesn't, the method transfers; if not, our 1.4B lacks strong
retrieval heads and we pivot to PreSelect's rank-match instead.

### 📖 Predictive Data Selection: "The Data That Predicts Is the Data That Teaches" (PreSelect)
Kashun Shum … Junxian He (both HKUST) · ICML 2025 · **20 citations** · `2503.00808`

**What it is.** Another "which documents are worth training on?" method, built on a neat idea: a document is valuable
if *stronger models compress it better than weaker models, in exactly the order of the models' overall ability.*

**What they did.** Take a ladder of models of known ability (Llama-1-7B up to Llama-1-65B). For each document, check
whether their perplexities line up with their ability ranking. Documents where the ranking matches perfectly get a
high "predictive strength" score. Then train a cheap classifier to imitate that score and run it over the whole
corpus.

**What they found.** Very effective: models trained on 30B PreSelect-chosen tokens beat models trained on **300B**
random tokens (a 10× efficiency win), and it beats other selection methods. Caveat: the signal targets *general*
downstream ability (knowledge, code, comprehension), **not reasoning specifically** — the word "reasoning" barely
appears.

**Why it matters here (deep-dive-verified 2026-07-23).** This is recipe **(B)** for our reverse-filter — and crucially,
it is **not a two-model magnitude gap.** It scores by whether *many* models' per-char losses **rank-match** the models'
ability order (the *sign* over C(6,2)=15 pairs), which is a *different and nearly orthogonal signal* from "how much
better is the big model than the small one." Their appendix runs our exact 1.4B-vs-72B idea as a controlled baseline
("ScalingFilter" = big-vs-small perplexity difference): it beats random by only **+0.4**, selects **short/easy junk**,
and is **uncorrelated (Spearman 0.05)** with the rank-match signal that works. So: (a) a two-model gap is a documented
near-failure — do not run it; (b) if we want recipe (B), use the **Qwen size ladder (0.5B→72B)** — same family, same
tokenizer, known ability order — and to target *reasoning* specifically, define the ability order by a reasoning
benchmark (their A.7.2 shows the ranking is steerable, at some cost to other axes). Two more caveats it flags: the
signal targets *general* downstream ability (not reasoning per se), and you must normalize per-character (tokenizer-
agnostic), never per-token.

### 📖 Autonomous Data Selection with Zero-Shot Generative Classifiers for Math (AutoDS / AutoMathText)
Yifan Zhang … Andrew Chi-Chih Yao (both Tsinghua University) · ACL Findings 2025 · **26 citations** · `2025.findings-acl.216` (arXiv 2402.07625)

**What it is.** The simplest "find the reasoning text" recipe: just ask a big model whether a document is
mathematically substantive, and keep the ones it says yes to.

**What they did.** Feed each document to a Qwen-72B base model with a fixed prompt asking two yes/no questions ("does
this show mathematical intelligence?" and "is it good for learning math?"), read how confident it is in "YES," keep
the high-scoring documents, and continue-pretrain on them.

**What they found.** It works: a Mistral-7B continue-pretrained on the selected math text improves **MATH 12.9 →
16.1** and **GSM8K 38.8 → 45.4**, at ~2.4× the token efficiency of using the unfiltered math corpus, and it beats
other selectors. On a small model (Gemma-2B) it only ties the baseline.

**Why it matters here.** A clean example of "identify reasoning-rich text and it helps," and a useful *contrast* for
the perplexity-gap question: its signal is a *single strong model's* confidence, **not** perplexity and **not** a
weak-vs-strong gap — so it shows there's more than one way to find this text.

### 📖 The FineWeb Datasets (incl. FineWeb-Edu)
Guilherme Penedo … Thomas Wolf (both HuggingFace) · NeurIPS 2024 · **1029 citations** · `2406.17557`

**What it is.** The famous web-data-cleaning paper; the relevant piece is **FineWeb-Edu**, a filter that keeps
"educational" web pages.

**What they did.** Had Llama-3-70B rate 460,000 web pages 0–5 for "educational value," trained a cheap classifier to
reproduce those ratings (82% F1), and used it to filter 15 trillion tokens down to a 1.3-trillion-token
"educational" subset.

**What they found.** Big gains on knowledge/reasoning benchmarks (**MMLU 33 → 37, ARC 46 → 57**) from just filtering.

**Why it matters here.** It's the canonical "quality selection helps," but it's a **contrast case, not our target**:
"educational" is deliberately aimed at grade-school knowledge and *down-weights* technical/arXiv content — so it's
broader than, and partly orthogonal to, "reasoning-rich." Useful to distinguish "educational" from "reasoning" when
we design our own signal.

---

## H2.5 / H2.6 — augmenting text with reasoning, and how *complete* it must be

### 📖 Multi-Hop Knowledge Composition is Bound by Pretraining Exposure
Yannis Karmim (Inria, Paris & Chile) … Valentin Barrière (Universidad de Chile) · 2026 preprint · **0 citations** (too new) · `2606.09338`

**⚠️ Corrected 2026-07-23 — I first mis-read this as "completeness/explicit reasoning doesn't help." It is NOT that;
the eval is unfair to the explicit condition (no scratchpad). Corrected reading below** (Dongwei read the paper and
flagged the error; the key correction is merged into this entry).

**What it is.** A controlled study of whether a model that has *memorized two facts separately* can chain them in one
forward pass to answer a question whose answer isn't stored (e.g. knows "Marcus's friend is Delia" and "Delia was born
in Ashford," but is asked "where was Marcus's friend born?"). The clean question it isolates — separate from "does the
model know the facts?" — is the **compositionality gap**.

**What they did.** Invent 100k people, split into two disjoint groups: everyone appears in *atomic* one-hop
biographies, but only one group (`P_comp`) also appears in *compositional* two-hop pretraining sequences; the other
(`P_held`) never does. Train GPT-2 (124M–774M) from scratch, then test two-hop composition **in a single forward pass,
with the bridge entity NOT in the context** (so the model must recover it internally). They try nine ways of writing
the compositional data, crossing format (natural language vs. RDF triples) × **explicitness** (name the bridge entity
vs. omit it). Then LoRA/full fine-tune on QA and evaluate transfer.

**What they found.**
- **The robust, load-bearing result — exposure-boundedness.** Both groups learn every atomic fact (**97% 1-hop**), but
  only `P_comp` can compose (2-hop up to **0.83**); `P_held` stays at **chance (~1%)** across *all* nine augmentation
  formats and *all* model scales. A conditional analysis (only cases where the model answers both single hops
  correctly) confirms `P_held` still can't compose — so it's **not** missing knowledge; the composition operation was
  never installed for entities absent from compositional contexts. This is the paper's real contribution.
- **The explicit-vs-implicit result — and why it is NOT "completeness doesn't help."** Explicit augmentation (bridge
  entity named) gave **0.08** (= baseline); implicit (bridge omitted) gave **0.62** NL / **0.79** RDF. *But the
  evaluation is unfair to the explicit condition:* explicit training teaches `P(Ashford | …Delia Crane was born in)` —
  it relies on the bridge token *being in the prefix* — while the test forbids any scratchpad and never puts the
  bridge in context. So this measures **train/test-format alignment**, not "explicit is worse at reasoning." The paper
  never runs the fair test (let the explicit model emit the bridge first, then score the answer); the 2×2
  {implicit,explicit}×{direct-answer, scratchpad} is a **missing experiment**. The logit-lens "explicit recalls the
  bridge but composes 8%, implicit never emits it but composes 79%" shows only that *latent* one-pass composition
  isn't taught by explicit traces — it says nothing about explicit-with-scratchpad.

**Why it matters here (corrected).** The defensible claim is **narrow**: explicit compositional text does *not
auto-compile into a latent (no-scratchpad) direct-answer computation*, and composition is **exposure-bound** (you must
expose the composition itself, atomic facts don't suffice). It is **not** evidence that completeness/explicit reasoning
is useless — that would require the scratchpad column they never ran. For our thread the real takeaways are: (1) if we
want *latent* reasoning, the training format must match the no-scratchpad inference distribution, and the *entities/
operations* (not just facts) must be exposed; (2) if we want *externalized* (chain-of-thought) reasoning, this paper
says nothing against it. Heavy caveats: fully synthetic, two relation types, GPT-2 scale, silent-reasoning-only.

### 📖 Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought
Jinghan Jia (Michigan State University / Anthropic Fellows) … Eric Easley (Anthropic) · 2026 preprint · **0 citations** (too new) · `2605.24286`

**What it is.** A rigorous attempt to define what it even *means* for a reasoning chain to be "complete" and
"faithful" — and it hands us a borrowable definition plus a warning.

**What they did.** They frame a good reasoning trace as one where all the answer-relevant information flows *through*
the written chain (prompt → chain → answer), and define three properties information-theoretically: **sufficiency**
(the chain alone determines the answer), **completeness** (given the chain, the prompt adds nothing more — i.e. the
chain "screens off" the prompt; a violation is a leftover prompt→answer shortcut), and **necessity** (the answer
actually depends on the chain, not just correlates with it). Then they try to *train* models to be more faithful by
tweaking the RL update.

**What they found.** Models routinely use a hidden prompt→answer shortcut while emitting a plausible-looking chain
that they don't actually rely on — a chain can look complete and still be a rationalization. Their training
interventions make the shortcut more *visible* in the chain but don't remove it, which is exactly why they insist on
the separate "necessity" property.

**Why it matters here.** Two gifts. First, a **precise, measurable definition of completeness** — "the chain screens
off the prompt; any leftover direct path is an incompleteness" — that we could actually compute. Second, the
**necessity caveat that reshapes the thread**: a complete-*looking* chain isn't enough; what matters is whether the
model *uses* it. That's the pivot from "is it complete?" to "does the encoding make the model actually run it?"

### 📖 Making Implicit Premises Explicit in Enthymemes  ← completeness helps *this* regime
Xuyao Feng … Anthony Hunter (both UCL) · 2026 preprint · **0 citations** (too new) · `2603.06114`

**What it is.** The paper on the *other* side of the completeness split: for explicit logical arguments, does filling
in the unstated premise help? (An "enthymeme" is an argument with a missing premise — "Socrates is a man, therefore
mortal" leaves out "all men are mortal.")

**What they did.** Build a pipeline: an LLM generates the missing intermediate premise(s) — one, two, or three steps
— then a formal logic checker (converting the sentences to logic and running a SAT solver) verifies whether the
argument now actually goes through. They vary how many gap-filling steps are added and measure.

**What they found.** More completeness helps, **monotonically**: verification accuracy rises steadily as you add
steps (one dataset 0.53 → 0.73, another 0.29 → 0.56), and the LLM's generated premises even beat the datasets' own
original terse premises.

**Why it matters here.** This is the counterpoint to the Exposure paper: **for explicit, step-by-step logical
argument, filling in the missing premises clearly helps.** So completeness isn't universally useless — it's
regime-dependent. The open question for us is which regime our thread targets: silent one-shot reasoning (where
explicit hurt) or explicit multi-step argument (where it helped).

### 📖 (read earlier) Thinking-Augmented Pretraining (TPT) · `2509.20186` · and BoLT · `2503.18866`
Both **augment text with generated reasoning uniformly** (no selection) and get real gains (TPT ~3× data efficiency;
BoLT lifts MATH 5.7 → 25.4) — but on already math-heavy corpora, and they never test *completeness* as a variable.
They establish "adding reasoning to pretraining text works"; they don't answer "how complete must it be."

### 📖 (read earlier) Quiet-STaR · `2403.09629`
Teaches a model to generate a little private rationale after each token to predict the next text better; the
perplexity gain concentrates on the *hard* tokens — the closest thing to a per-token "this needed reasoning" signal,
but measured *after* inserting the rationale, not used to find reasoning text beforehand.

---

## H2.7 — can a perplexity / weak-vs-strong gap detect reasoning content?

The short answer from the deep-dive (2026-07-23): **single-model perplexity — no; a two-*different*-model magnitude gap
— also no (documented near-failure); what works is either self-ablation (one model) or multi-model rank-match.**
AttentionInfluence is *self-ablation* (one model vs. itself with reasoning heads masked — memorization cancels because
both losses share weights). PreSelect is a *multi-model rank-match* (does per-char loss rank-order match the models'
ability order?), and it explicitly shows the two-model magnitude gap ("ScalingFilter") barely beats random (+0.4),
picks short/easy junk, and is uncorrelated (Spearman 0.05) with the rank-match. Our own reverse-filter's "gold"
criterion *was* a two-model gap (1.4B-high AND 72B-low) and it found knowledge, not reasoning — exactly what
ScalingFilter predicts. RHO-1 (read earlier) is the token-level reference-vs-training excess-loss idea, and its authors
are explicit it finds "high-quality" tokens, not specifically "reasoning" tokens.

---

## Open questions (genuinely open — not prescriptions)

1. **Which inference format is our thread about?** Latent, no-scratchpad reasoning (then Exposure says: match the
   no-scratchpad inference distribution *and* expose the composition/operation, not just facts) or externalized
   chain-of-thought (which Exposure does not test at all)? This determines what "augmenting text with reasoning" even
   means, and it is a question about inference format, not about how completely the text is written.
2. **Does augmenting pretraining text with reasoning actually help?** Still open — the paper I earlier cited against it
   (Exposure) was a confounded eval, and the papers that *do* augment (TPT/BoLT) don't vary completeness. No clean
   evidence either way yet.
3. **For the data-selection side (reasoning-rich text):** the two candidates that survive the deep-dive are
   self-ablation on our own 1.4B (recipe A) and multi-model rank-match on the Qwen ladder (recipe B); the two-model
   1.4B-vs-72B gap is ruled out (ScalingFilter). Which — if either — actually surfaces *reasoning* (vs general quality)
   on DCLM is untested.
4. **Does under-reasoning persist through *our* post-training?** Still under-tested for the Won't form specifically;
   Front-Loading is the closest, but it uses fine-tuning-style reasoning data, not rewritten web text.

---

*Provenance: full-read workflow `wf_e16faf72-dc2` (2026-07-21), 16 agents (opus), one per paper, HTML/PDF full text,
schema-validated extraction (method / numbers / verbatim quotes / can't-vs-won't / completeness / limitations /
verdict). Raw journal: `subagents/workflows/wf_e16faf72-dc2/journal.jsonl`; full structured results in the session
task output. Papers discovered by a zero-seed neutral search (`wf_869397f2-d8b`). The earlier abstract-only map and
the knowledge-framing doc `docs/PERSISTENCE_AND_USEFUL_REASONING.md` are both superseded by this.*
