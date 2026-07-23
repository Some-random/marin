# Reasoning in pretraining: under-reasoning (H1) & finding/exploiting reasoning-rich text (H2)

**Status: 64 PAPERS FULL-READ (2026-07-23).** The 24 core papers (full writeups below) plus a 40-paper batch from
the wide zero-seed discovery pool (`docs/DISCOVERY_POOL_2026-07-23.md`; compact entries in the Batch-2 section).
Every reader was **required to critique the paper's own eval methodology** — is the headline A-vs-B comparison
fair? is the baseline token/compute-matched? is an augmentation gain separable from distilling the generator? what
control is missing? So every writeup carries both the paper's result **and** the confounds that bound what it can
prove. Written to be **read cold**. Provenance at the end.

---

## TL;DR

**H1 (under-reasoning is real, pretraining-laid, and persists) is well-supported. H2 (whether augmenting pretraining
text with reasoning helps) is genuinely open.** Three things to hold onto:

1. **🔴 Composition is bound by pretraining EXPOSURE — and latent vs scratchpad reasoning are different problems.** A
   model that knows two facts atomically (97% 1-hop) cannot chain them in one forward pass unless those entities
   appeared in *compositional* pretraining contexts, and this is invariant to model scale (~1% 2-hop for unexposed
   entities at 124M–774M). Exposing the composition itself — not just the facts — is what installs it. Separately,
   explicit-bridge augmentation (0.08) loses to implicit-bridge (0.79) *on the no-scratchpad test*, but that only
   measures train/test-format alignment: explicit training relies on the bridge token being in the prefix, and the
   test removes it. The fair explicit+scratchpad comparison is never run (the paper explicitly rejects the paradigm),
   so the narrow claim is "explicit traces don't auto-compile into *latent* one-pass computation" — nothing more.

2. **🟡 Completeness: the dose-response evidence now exists in fragments (batch 2), and it is consistent —
   completeness pays off through STRUCTURE and DIFFICULTY, not volume.** Across the papers that actually vary chain
   granularity: (a) completeness is **difficulty-conditional** — skipping steps is harmless or even helpful on easy
   algorithmic tasks but complete chains are required where genuine multi-step inference happens (Skip-Steps,
   Less-is-More-Tokens); (b) there is a **training-granularity floor** — SFT cannot decompose reasoning below the
   granularity of its training traces; only RL re-derives skipped steps (Zipping); (c) **incrementality beats
   length** — longer, locally-incremental traces beat compressed ones at matched tokens AND matched data, while pure
   padding/redundancy hurts (Inefficient-Reasoning), trace length is uncorrelated with gain (MIND), and verbose noisy
   thoughts are monotonically worse than concise ones (ToW); (d) explicitness that helps is produced by
   **knowledge-gap dynamics** — dialogues between mismatched-knowledge participants force premises explicit, while
   zero-gap expert dialogue stays surface-level and gains nothing (MIND); (e) **order matters** — computation-order
   chains prevent the shortcut from forming (Implicit-Shortcut), and teacher-forced complete chains can *entrench* a
   cheat when intermediate steps leak the answer (Pitfalls-of-NTP). The older explicit-regime evidence (Enthymeme
   0.53→0.73) remains only suggestive (positive-class-only recall; gold premise barely beats none). For latent
   (no-scratchpad) reasoning the lever is still format-match/exposure (Exposure), and Faithfulness's *necessity*
   caveat stands: a complete-looking chain can still be unused.

3. **🟢 No loss/perplexity-family signal has been shown to find *reasoning* — and our own experiments have now ruled
   out two of them.** The two-model magnitude gap selects short/easy/memorized text (ScalingFilter — whose primary
   source, now read, validates it only on commonsense with +0.6–1.1% no-error-bar gains and zero reasoning
   benchmarks; PreSelect's controlled baseline; our own 1.4B-vs-72B reverse-filter found knowledge). **Self-ablation
   (recipe A) is now empirically NO-GO on our own models** (2026-07-23, `docs/RECIPE_A_SELF_ABLATION.md`): across 5
   base models and 6 sources, the retrieval-head gap detects *in-context copy dependence* — config files, parallel
   translations, reference boilerplate — not reasoning; the random-head control separates as well or better; it
   misses verbal reasoning for 3 of 4 models; and Qwen 7B→72B *inverts* which reasoning type it flags. Single-model
   perplexity pruning actively REMOVES reasoning-dense domains (Perplexed-by-Perplexity cuts code and papers ~3×);
   frequency/co-occurrence signals track knowledge recall, not reasoning (Generalization-vs-Memorization). What
   remains standing, untested by us: **multi-model rank-match** (PreSelect; cousin Perplexity Correlations — whose
   own appendix concedes plain mean loss predicts nearly as well), plus two newer self-contained variants worth
   noting (Quiet-STaR's with-vs-without-thought gap; TEMP's noise-perturbed-vs-clean gap). Every published signal in
   this family is validated on *general* value, none on reasoning specifically.

**Bottom line:** reasoning-in-pretraining pays off and compounds (Front-Loading, with its token-matching caveat;
now reinforced by Echo Chamber — RL amplifies whatever mode dominates pretraining — and Interplay — RL cannot
transfer to contexts below ~1% pretraining exposure, and pass@128 ceilings are set upstream). Composition is
exposure-bound. On augmentation, the distillation-confound story is now **two-sided**: the flagship results (TPT,
BoLT, Swallow, REWIRE) remain strong-teacher-confounded — REWIRE's rewritten text *alone* is worse than raw — but
batch 2 adds real deconfounding evidence that the *structure* effect exists without strong teachers: MIND's
generator swap (8B ≈ 70B) and same-generator rephrase control, Demystify's generator ablation (70B-written data
trains WORSE than 8B-written), GrokWild (even factually *incorrect* augmentation works — the lever is the
inferred-to-atomic ratio), and the programmatic-trace results (Kinetics, Logic-Corpus, Internalize) that have no
teacher at all. A first-principles argument for externalizing reasoning also landed: latent multi-hop costs ~250×
more parameters per bit than knowledge storage (U-shape, ~0.008 vs ~2 bits/param). What remains genuinely open:
whether reasoning-augmentation of *natural, general* web text helps at pretraining scale with a weak generator —
and the completeness dose-response on natural text, which no one has run.

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
  processing to still use it. (Their surgical fix — feeding a late-stage internal state back to an earlier layer —
  repairs up to 66% of failures on the best model, though the intervention is under-controlled; see the entry.)

- **A capacity/coverage limit.** The *k-hop* paper shows deeper chains are learnable but the training data needed
  grows *exponentially* with the number of steps — so below some budget the model just guesses.

**The unifying H1 takeaway:** under-reasoning is genuine and mechanistically real, but "the model just can't reason"
is wrong (that framing got refuted). The accurate picture is *partial, shortcut-inflated, and bounded by what the
training data exposed and what the architecture can reach.* And critically: **making the reasoning explicit at
answer-time (chain-of-thought) reliably rescues it** — one paper shows silent composition at ~8% jumping to ~93%
once the model is allowed to write the intermediate step out. Our thread's question is whether we can bake that same
benefit into the *training text* instead of relying on it at answer-time.

**Persistence (H1's central claim).** NVIDIA's *Front-Loading Reasoning* shows reasoning put into *pretraining*
doesn't just survive later training — it **compounds**: a model pretrained with reasoning data leads a plain model by
~9% before any fine-tuning, ~9.3% after fine-tuning, and **~18.5% after RL** — the gap *widens* at every stage.
Their headline "cannot be fully replicated by later-stage SFT, even with more data" is weaker than it sounds,
though: the catch-up baseline is never given a post-training reasoning budget anywhere near the ~200B reasoning
tokens the pretrained models saw (it merely doubles SFT), so the *compounding direction* is well-evidenced while the
*irreplaceability* claim is unproven at matched budgets. The RL-side papers (Yue, ProRL) add the complementary
point: ordinary RLVR mostly re-weights reasoning paths the base model already has — new capability comes from new
information (distillation, or pretraining data), which is exactly why the base model's training data matters.

---

## H2 — how to find reasoning-rich text, and the completeness question that reshapes the thread

**(4) Identifying reasoning-rich text — three recipes that work, but none cleanly isolates *reasoning*:**
- *AttentionInfluence* takes a small model, deliberately breaks its "retrieval" attention heads so it gets worse at
  reasoning, and flags the documents where that breakage hurts the most. Caveats: the downstream gain is modest
  (+0.75pp average, with commonsense regressions; the abstract's +1.4–3.5pp is the reasoning subset only), the
  selected docs are ~2× longer (an un-ablated confound — no matched-upsampling control), and its "more reasoning than
  the edu-classifier" advantage exists only in math/code domains.
- *AutoDS* just asks a big model two yes/no questions ("is this mathematically intelligent? is it educational?") and
  keeps the documents it says yes to. Caveats: the 2.4× efficiency headline is one model on one task read off noisy
  curves with no error bars; on Gemma-2B it inverts; and the scorer is Qwen-72B, so it's large-model-curates-for-small,
  not truly "autonomous."
- *FineWeb-Edu* trains a cheap classifier to imitate a big model's 0–5 "educational value" rating. Caveats:
  "educational" is deliberately grade-school-flavored and *down-weights* technical/arXiv content; and the MMLU/ARC
  gains have a target-eval circularity (school-exam-flavored filter scored on school-exam benchmarks — aggressive
  filtering *hurts* HellaSwag, so it's domain steering, not a free lunch).

**(6) Completeness — regime-dependent, with only suggestive evidence in the explicit regime.**
- For **explicit** reasoning, completeness *may* help: filling in unstated premises improves formal logical-argument
  verification (Enthymeme, 0.53→0.73) — but that headline is entailment-class-only recall with no specificity
  control, and the gold complete premise barely beats no premise (0.558 vs 0.530), so part of the gain is verifier
  plumbing rather than completeness. Balanced metrics show a modest gain. Treat as suggestive.
- For **latent** (no-scratchpad) reasoning, the lever is format-match and exposure, not completeness: what installs
  composition is matching the no-scratchpad inference distribution and exposing the composition itself, not how
  completely the text spells the steps out (Exposure). Making the bridge explicit does not, by itself, teach the model
  to use it silently.
- *Faithfulness* gives a precise, borrowable definition of completeness — a chain is complete if it "screens off" the
  question from the answer (you can't reach the answer except through the chain) — plus the warning that a chain can
  *look* complete while the model ignores it. So **surface completeness isn't enough; the chain must be one the model
  actually uses** ("necessity").

**(7) The perplexity-gap — a two-model magnitude gap is a dead end; two other signals work, with caveats.**
- **A big-model-minus-small-model loss gap does NOT find reasoning text.** PreSelect runs exactly this
  ("ScalingFilter") as a controlled baseline: **+0.4 over random, selects short/easy junk, uncorrelated (Spearman
  0.05)** with the signal that works. Our own reverse-filter's gold criterion (1.4B-high AND Qwen-low) was this same
  gap and it found *knowledge, not reasoning* — the magnitude gap is maximized by text that's easy once you're big.
- **Self-ablation works (AttentionInfluence).** One model vs. itself with its top-5% retrieval heads masked to uniform
  attention. Because both losses come from the *same weights*, everything memorized (frequency, n-grams) **cancels** in
  the difference — the only thing left is reliance on the reasoning heads. That cancellation is why it isolates
  reasoning where a two-model gap does not.
- **Multi-model rank-match works (PreSelect).** Score by whether per-char loss **rank-orders** across a model ladder in
  the same order as the models' ability — the *sign* over many pairs, not one magnitude gap.
- **A third multi-model signal exists (Perplexity Correlations, ICLR 2025)** — correlate ~90 public models' per-domain
  bits-per-byte with their benchmark accuracy, no training needed. It works at *domain* granularity, but its own
  appendix shows **plain mean loss predicts nearly as well** as the correlation, its top-correlated domains for a
  reasoning benchmark are optometry-clinic and children's-hospital sites, and the signal **evaporates on pre-filtered
  pools**. Multi-model perplexity structure is real; the naive version detects general quality/language, not reasoning.
- **Warning from RHO-1:** its excess-loss token selection looks like a weak-vs-strong gap that works — but its own
  self-reference ablation shows **~80% of the headline gain (+16.5pp → +3.3pp) comes from the curated reference-model
  data**, i.e. it distills a curated distribution through a token mask, and the authors never claim it finds
  *reasoning* tokens ("closely related to mathematics", "aligned with the desired distribution").
- Recipe status for our reverse-filter: **(A) self-ablation — NO-GO, empirically ruled out on our own models
  (2026-07-23).** The go/no-go experiment was run generalized (5 base models, 3 families, 2 scales, 6 sources): the
  retrieval-head gap ranks copyable boilerplate on top, the random-head control separates reasoning-vs-web as well
  or better, 3 of 4 models rank verbal reasoning *below* web text, and Qwen 7B→72B inverts per-source (GSM8K AUC
  0.955→0.051). It measures in-context-copy dependence, not reasoning. Full record: `docs/RECIPE_A_SELF_ABLATION.md`
  (note: this does not refute AttentionInfluence's *downstream* result — their gain may come from selecting
  structured/technical data). **(B) multi-model rank-match on the Qwen ladder (0.5B→72B, same tokenizer) — the one
  candidate still standing, untested.** Exact recipe in the PreSelect entry below.

---

## What this means for our thread

1. **Reasoning-in-pretraining is worth it** — it persists and compounds through fine-tuning and RL (Front-Loading;
   though "you can't catch up later" is unproven at matched reasoning-token budgets — see its entry), and explicit
   reasoning reliably helps *at answer-time* (chain-of-thought; SOCRATES ~8%→~93%). Whether that benefit can be baked
   into *training text* — for reasons beyond distilling the teacher that writes the reasoning — is the open question.
2. **The design fork is inference format, not completeness-per-se.** For **latent** (no-scratchpad) reasoning, the
   levers are matching the no-scratchpad inference distribution and exposing the composition itself (Exposure). For
   **externalized** (scratchpad) reasoning, completeness *may* help — the Enthymeme result points that way but is
   confounded (positive-class-only metric, gold-premise anomaly) — and Exposure says nothing against it.
   Faithfulness's *necessity* (does the model actually *use* the chain?) is the metric that cuts across both.
3. **Composition is exposure-bound.** A corpus can contain every fact yet fail to teach the operations over them; data
   coverage must be defined over *operations/paths*, not just facts.
4. **For finding reasoning-rich text:** a two-model perplexity gap is ruled out (ScalingFilter + our own result),
   and **self-ablation (recipe A) is now also ruled out by our own multi-model experiments** (it detects copy
   dependence, is model/scale-dependent, and misses verbal reasoning — `docs/RECIPE_A_SELF_ABLATION.md`). The one
   surviving candidate is multi-model rank-match on the Qwen ladder (recipe B) — with the warning that no published
   signal in this family has demonstrated it surfaces *reasoning* rather than general quality, and Perplexity
   Correlations' own appendix shows plain mean loss carries nearly the same information as its correlation signal.

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

*Each entry ends with the eval-methodology fine print — the confounds and missing controls that bound what the paper
can prove. Read the fine print before citing the headline.*

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
confirmed these neurons matter causally (the circuit accounts for 96% of the model's arithmetic behavior; deleting
the neurons relevant to a given problem drops accuracy by ~29 points). Then they replayed the model's *entire
pretraining history* (Pythia checkpoints) to see when this mechanism forms.

**What they found.** The "bag of heuristics" is there almost from the start of training and is **never replaced by a
real algorithm** — the same rough set of rules explains ~79% of the model's arithmetic ability at every checkpoint.
So the model found a cheap trick early, it worked well enough, and it never had any pressure to learn something
better.

**The fine print.** The random-ablation control is not matched for activation magnitude (heuristic neurons are by
construction high-activation, so "causally specific to arithmetic" isn't fully separated from "generally important
neurons"), and no non-arithmetic control task is run. The 0.96 faithfulness is in-distribution (same operand regime;
no held-out carry/large-operand test). The 79% developmental number is survivorship-framed — "mutual heuristics" are
*defined* by the final checkpoint, so any early mechanism outside the researcher-defined taxonomy is invisible. The
authors scope the claim to multi-digit-tokenization models ("a similar analysis might lead to different conclusions
for models that perform single-digit tokenization").

**Why it matters here.** A clean, mechanistic example of your **Won't**: the model reasons by shortcut, the shortcut
is laid down early in pretraining, and continued training doesn't fix it — with the scope caveats above keeping the
"never replaced by an algorithm" claim narrower than the headline. The sobering note for us — the authors think
fixing this "may require fundamental changes to training and architectures," a caution that simply feeding better
text may not dislodge an entrenched shortcut.

### 📖 When LLMs Stop Following Steps
Sailesh Panda … Mayank Singh (both IIT Gandhinagar) · 2026 preprint · **0 citations** (too new) · `2605.00817`

**What it is.** A stress test of whether models actually *follow a procedure* they're given, versus just landing on a
plausible final answer. The trick: they hand the model the complete step-by-step recipe *in the prompt*, so the
model is never missing any knowledge — the only question is whether it faithfully executes.

**What they did.** They generate arithmetic procedures of varying length (5 steps up to 95 steps), give the model the
full recipe and two numbers, and check both the final answer and whether the model actually walked through every
step. 14 models, 55 datasets × 1,000 = ~55,000 problems.

**What they found.** As procedures get longer, accuracy collapses from **61% (5 steps) to 20% (95 steps)** — and a
large share of failures is that the model **stops following the recipe partway through** (exact-step execution drops
from ~71% to ~47%; under-execution rises from ~24% to ~51%). This happens *even though the complete procedure is
right there in the prompt*, and it happens to reasoning/RL-tuned models too.

**The fine print — a no-scratchpad eval with a contaminated metric.** (1) The prompt *forbids* written reasoning
("You MUST NOT explain your reasoning… MUST NOT output anything except the final result") — thinking models get a de
facto scratchpad in their hidden reasoning channel while the one non-reasoning model is denied token-space
computation entirely and predictably floors; the written-execution control is never run. (2) The "under-execution"
metric counts *obedient direct answering* as failure (the paper itself notes GPT-oss-120B "appears to generate the
final output directly"), and in single-op variants batching 95 identical additions into one multiplication is
*correct math*, also scored as under-execution. (3) A big share of the decline is per-operation float arithmetic,
not lost procedure-state: add/sub stay ~97–99% while mult/div sit at ~43–53% for the best models, and exact
3-decimal matching accrues rounding drift over 95 chained ops. Two models keep exact-step execution high while
accuracy still declines — the authors concede "factors beyond premature termination also contribute."

**Why it matters here.** The construction rules out missing *knowledge* (the full recipe is supplied), so something
execution-shaped fails at long horizons — but the no-scratchpad prompt, the contaminated under-execution metric, and
the arithmetic confound entangle a Can't-execute (precision/capacity) component, so it cannot cleanly separate Won't
from Can't. The durable lesson: a maximally complete chain in-context does not guarantee faithful long-horizon
execution — and its eval forbids exactly the externalization that would test our thesis fairly.

### 📖 GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs
Iman Mirzadeh … Mehrdad Farajtabar (both Apple) · ICLR 2025 · **591 citations** · `2410.05229`

**What it is.** The widely-cited fragility benchmark: take 100 GSM8K questions, turn each into a parameterized
template (names/numbers become variables with validity constraints), sample 50 instances per template, and measure
how much performance moves when only the surface changes — plus controlled variants that add/remove clauses
(M1/P1/P2) and **GSM-NoOp**, which inserts a relevant-*sounding* but logically inert clause.

**What they found.** (~25 models, 2B–27B open + 4 closed.) Number swaps and added clauses hurt: Gemma2-9B goes
79.1→44.0→41.8 as one then two clauses are added (drops ~35pp, far beyond the ±3–6 std); GSM-NoOp is the killer —
Phi-3-mini 85.0→18.0 (−67pp), Gemma2-9B −64.7pp; models "blindly subtract" the irrelevant quantity. RL-trained
reasoners resist much better but not fully: o1-preview −18.6pp on NoOp, essentially flat on clause-scaling. And
8-shot prompts of the *same question with full correct reasoning* (NoOp-Symb) do **not** fix it.

**The fine print.** The headline NoOp drop is measured against GSM8K-original — which the paper *itself* shows is
contaminated (original sits >1 std right of the GSM-Symbolic distribution for 21/25 models), so the −67pp mixes
contamination-loss with the distractor effect; the fair baseline (GSM-Symbolic vs NoOp) still shows the effect but
smaller. Their own significance test (which they admit is inappropriate for two-dataset comparisons) finds GPT-4o
and Llama3-8B's name/number fragility *not significant*. Clause-scaling confounds reasoning depth with prompt
length. And the NoOp failure is equally well explained as a learned Gricean prior ("every stated quantity is
relevant" — true of all training word problems), i.e. a distributional shortcut, not missing capacity; the paper
defaults to the strongest "no formal reasoning" interpretation without running any training intervention.

**Why it matters here.** Strong behavioral evidence for H1's premise — the shortcut is real, and RL shrinks but does
not erase it (o1 still drops on NoOp). For H2 it's a caution in both directions: no training intervention was run
(so it says nothing about whether augmentation fixes the shortcut), and its one adjacent result — complete worked
reasoning in-context failing to teach premise-relevance — echoes the Exposure lesson that *presenting* reasoning is
not the same as *installing* it. The distributional-shortcut reading suggests the fix may be training-distribution
coverage (include distractor-bearing problems) as much as reasoning completeness.

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

**The fine print.** Three probe-validity caveats. The first-hop metric (EntRec) cannot separate genuine bridge
resolution from shallow n-gram co-occurrence ("Superstition"↔"Stevie Wonder" co-occur in pretraining) — the authors
list this competing pathway themselves. The two hops are measured with *different* metrics, and the "both hops"
joint rate assumes independence of two events derived from the same hidden state (the 25% null is unjustified). And
the probes are never conditioned on the model actually *answering* the two-hop question correctly — this is a
representational signature, not behavioral reasoning. The authors hedge appropriately ("lower bound", "one pathway").

**Why it matters here.** Suggestive evidence of under-reasoning that isn't a knowledge gap: the bridge entity is
*right there internally*, unused — with the caveat that the probe is representational, not behavioral. Because scale
doesn't fix the second hop, it hints the fix has to come from *how the model is trained*, not from more parameters —
consistent with a data intervention.

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
problem inside the network**. Back-patching fixes **up to 66%** of previously-wrong cases (best model, Pythia-6.9B;
LLaMA models are 41–58%), suggesting the answer was reachable — the model just ran out of runway.

**The fine print.** "Fixed" means *there exists* at least one (source-layer × target-layer) patch that flips the
answer — a max over a large intervention grid selected on the outcome, with **no random/placebo back-patch
baseline**, so an unknown share could be generic perturbation nudging. Back-patching also re-runs extra layers on
the later state, so "knowledge arrived too late" is confounded with "extra effective depth constructs the answer."
Credit where due: the incorrect-case construction (model provably answers both hops in isolation) genuinely holds
knowledge constant, and Patchscopes independently confirms the bridge entity is encoded even in failing cases.

**Why it matters here.** A *third* category beyond Can't/Won't: the model knows the facts AND wants to compose them,
but the architecture runs out of layers — plausible but under-controlled. Data augmentation wouldn't directly fix
this mechanism, but the authors note that writing the intermediate step out explicitly (chain-of-thought) sidesteps
it, which is indirect support for externalizing the hidden step.

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
hard architectural limit.

**The fine print.** The internal result (composition-vs-comparison, same architecture/regime) is fair and
well-controlled. The paper's famous side-by-side — grokked transformer 99.3% vs GPT-4-Turbo ~33% / Gemini ~11–28% —
is **not** apples-to-apples: it pits a model trained from scratch *on this exact knowledge graph* (facts in weights,
grokked over hundreds of thousands of steps) against frontier models seeing the same facts once in-context,
zero-shot — a parametric-vs-in-context memory mismatch, not a reasoning-skill gap. Don't quote it as "tiny grokked
transformer out-reasons GPT-4."

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

**The fine print.** The latent-vs-CoT contrast is fair *as an elicitation diagnostic* (no author-side training, so no
train/test-format trap), and conditioning on 1-hop knowledge is a genuinely strong control. Three caveats: single-hop
and composed queries use *different prompt templates*, so some "latent failures" could be query-parsing failures
(the missing control: put both facts in-context, still forbid CoT); the country-bridge ~83% bin is plausibly the
*least* shortcut-controlled (a soft prior over a small country set survives their single-prompt-ablation filter), so
the honest latent numbers are the year/city bins (~6% and below); and their own ~5× inflation figure implies
residual undetected shortcuts could still inflate the 8%.

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

**The fine print.** The main test set is not guaranteed shortcut-free: the *curriculum* section builds
rejection-sampled test sets specifically "to avoid shortcut solutions" (no sub-chain overlap with training), but
that control is not described for the main evaluation — so a held-out 4-hop query can share its first 3 hops with a
training query and be solved by one fresh hop. The data-budget curve is therefore optimistic (a strictly
shortcut-free test would demand at least as much data); the "genuinely learnable" claim leans more on their
mechanistic layer-wise evidence than the accuracy numbers. Also the hardest cell (4-hop_large) got 2× the training
steps of other cells, and the depth lower-bound is conditional on a query-independent attention pattern (which the
authors flag as possibly relaxable).

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
mix — the reasoning variants replace 20% of the corpus, ~200B tokens, with curated QA/CoT-format reasoning data),
each then fine-tuned and then RL-trained, evaluated at each stage. They specifically test the "catch-up hypothesis"
— can extra fine-tuning let a plain base model catch a reasoning-pretrained one?

**What they found.** Reasoning-in-pretraining doesn't just persist, it **compounds**: the lead of the
reasoning-pretrained model over the plain one *grows* from ~9% (before fine-tuning) to ~9.3% (after fine-tuning) to
**~18.5% (after RL)**. Doubling the fine-tuning data on the plain model gains only +4.09% and still leaves it behind
even the *weakest* reasoning-pretrained model (37.33 vs 34.01). There's also a "latent" effect where high-quality
pretraining data shows ~no benefit at the base stage but pays +4.25% after fine-tuning "unlocks" it — and at the
pretraining stage, *diversity and scale* of reasoning data beat curated long-CoT quality.

**The fine print — the catch-up comparison is not token-matched.** The reasoning-pretrained models saw ~200B
reasoning tokens in pretraining *plus* 1× SFT; the "catch-up" baseline got reasoning only at SFT and merely doubled
it (4.8M→9.6M samples — far below 200B; the paper never reports the token counts). The fair test — give the baseline
a post-training reasoning budget equal to the 200B, or ablate placement at fixed total reasoning tokens — is not
run. Their own Table 6 shows naive SFT-doubling with mixed-quality data *harms* math (−4.9%), so "doubling didn't
help" partly reflects quality-scaling damage. Two more confounds: base-stage models trained on QA/CoT format are
evaluated few-shot on QA benchmarks (format familiarity; no format-matched control), and the 20% reasoning data
*substitutes* for general corpus (adding reasoning is entangled with removing web text).

**Why it matters here.** The strongest evidence that reasoning content belongs early and **compounds** through the
pipeline — cite it for the direction, not for "cannot be replicated later," which is unproven at matched budgets.
Big caveat for *our specific* method: their "reasoning data" is question-answer / long chain-of-thought
fine-tuning-style data mixed into pretraining — **not ordinary web text rewritten to expose its reasoning** — and
their proxy for "quality" is basically trace length. So it supports front-loading reasoning, but doesn't test
"rewrite normal text to be more complete."

### 📖 Does RL Really Incentivize Reasoning Capacity Beyond the Base Model? (Yue et al.)
Yang Yue (Tsinghua) … Gao Huang (Tsinghua) · 2025 preprint · **924 citations** · `2504.13837`

**What it is.** The anchor of the "RL doesn't add capacity" camp. Sample a base model and its RLVR-trained
descendant many times per problem (up to n=1024) and compare **pass@k**: does RL ever solve a problem the base
model *couldn't* solve in any of k tries?

**What they found.** RLVR raises pass@1 (reliability) but pass@k *coverage* shrinks as training progresses
(train-set pass@1 26.1→42.5 while pass@256 declines); at large k the base matches or beats the RL model (Minerva
32B: base +~9% at k=128); RL uniquely solves ~0% of problems the base can't; and RL outputs sit in the
low-perplexity region of the *base's own* distribution — RL re-weights existing paths rather than creating new
ones. Distillation, by contrast, genuinely lifts the whole pass@k curve. Their sampling-efficiency gap stays >40pts
across all six RLVR algorithms tested.

**The fine print.** Pass@k at huge k structurally favors the higher-entropy model — RL *by design* trades coverage
for per-sample reliability, so "boundary narrows" is partly a framing choice on an entropy-sensitive metric (their
own perplexity result supports "re-weighted, not deleted"). The top of the curve is high-variance (n is set equal to
the largest k). The lucky-guess control is a manual CoT audit on only ~25 GSM8K + ~7 AIME problems. On
deployment-relevant metrics (pass@1, maj@k) RL wins.

**Why it matters here.** The key indirect argument that capacity is set upstream: if the base lacks a reasoning
path, ordinary RLVR won't install it — only distillation (new information from a teacher) does. That rhymes with
"put the reasoning into pretraining," without being a test of it. For the can't/won't lens: RLVR fixes a *Won't*
(path exists, rarely sampled) and does nothing for a *Can't* (path absent).

### 📖 ProRL: Prolonged RL Expands Reasoning Boundaries
Mingjie Liu (NVIDIA) … Yi Dong (NVIDIA) · 2025 preprint · **156 citations** · `2505.24864`

**What it is.** The "RL expands" camp's flagship: >2,000 RL steps (unusually long) on DeepSeek-R1-Distill-Qwen-1.5B
with a KL penalty, periodic hard resets of the reference policy, and a diverse 136K-example verifiable-reward suite
(math/code/STEM/logic-puzzles/instruction-following).

**What they found.** Large pass@1 gains (math avg 44.5→60.1; Reasoning Gym 4.2→59.1; GPQA +25.9), approaching a
distilled 7B. The boundary evidence: tasks where the base has pass@128≈0 and the RL model reaches high pass rates
(e.g. boxnet 0→7.9 pass@1; family_relationships ~0→near-perfect) — presented as RL creating genuinely new capability
under matched sampling budget (256 samples each).

**The fine print.** Three untested confounds: eval temperature (0.6) is the RL-optimal setting, not tuned per model
— base-model pass@k is very temperature-sensitive, so "base fails entirely" may be partly a decoding artifact; the
base is *already distilled from R1* (a ~671B reasoner), so "expansion" is confounded with resurfacing
distilled-but-suppressed capability (no non-distilled-base control); and no ablation isolates the "prolonged"
ingredients. Their own "Diminished" category concedes pass@128 *declines* where the base is already competent —
expansion is selective and base-competence-dependent, not a uniform property.

**Why it matters here.** Together with Yue, the honest synthesis: ordinary RLVR sharpens; *prolonged,
diversity-preserving* RL may expand — but the clean demonstration (non-distilled base, per-model temperature sweep)
hasn't been run. Either way, both camps agree the base model's distribution is the dominant term — which is the
thread-relevant point: what pretraining installs is what post-training has to work with.

### 📖 The Debate on RLVR's Reasoning Boundary: Shrinkage, Expansion, or Both?
Xinhao Yao (Renmin University of China / Ant Group) … Yong Liu (Renmin University of China) · 2025 preprint · **10 citations** · `2510.04028`

**What it is.** Referees the Yue-vs-ProRL fight with a two-stage story: early RL over-concentrates the model
(entropy collapses, coverage can shrink); only prolonged, diversity-preserving RL expands it.

**What they did.** Analyzed the training dynamics mathematically and ran RL (GRPO and entropy-preserving "-N"
variants) on Qwen2.5-Math-7B, tracking Pass@k across training.

**What they found.** Evidence for the second stage: the entropy-preserving variants keep improving at large k where
vanilla GRPO stagnates (e.g. AIME2025 Pass@256 base 46.7 vs GRPO-N 66.7).

**The fine print.** Eval temperature/top-p is not stated (so the -N variants' large-k gains could be a diversity
artifact — the missing control is a raised-temperature base); the load-bearing Stage-2 result is a 14/30-vs-20/30
gap on AIME's 30 problems (inside noise, no CIs); and in their own Table 1 the RL models meet or beat base at the
largest k *everywhere* — the "shrinkage" half rests on theory and entropy curves, not a demonstrated base>RL
crossover. Treat the two-stage reconciliation as a hypothesis, not a settled referee decision.

**Why it matters here.** Indirect support for reasoning-in-pretraining: since ordinary post-training is largely
bounded by what the base model can already do, it's better to get the reasoning into the base. Neutral on our
specific text-augmentation question.

---

## H2.4 — how to identify reasoning-rich text in a corpus

### 📖 AttentionInfluence: Weak-to-Strong Pretraining Data Selection
Kai Hua … Ke Shen (both ByteDance Seed) · 2025 preprint · **5 citations** · `2505.07293`

**What it is.** A training-free trick for finding reasoning-heavy documents *without* a classifier. **Important:
despite the title's "weak-to-strong," this is NOT two different models — it is one model vs. *itself with reasoning
heads disabled* (self-ablation).** Verified against code in a Tier-3 deep-dive.

**What they did.** Take one small (1.3B) model. Detect its "retrieval heads" (attention heads that fetch information)
via a synthetic key-value needle task, then build a *weak* copy by setting those top-5% heads to **uniform attention**.
Score each doc by the relative loss gap `(L_masked − L_base) / L_base` between the crippled copy and the intact model,
ranked **within-domain**. Keep top 20%, upsample into the corpus, pretrain a 7B model on 1T tokens (both runs
token-matched). (The retrieval-head-detection code is public — `nightdessert/Retrieval_Head` — and was verified; the
scoring/masking loop has no released code.)

**What they found.** The reasoning subset improves (**HumanEval +3.5, GSM8K +2.7, MMLU-Pro +2.7 pts**) — but the
*overall* average gain is **+0.75pp**, with commonsense regressions (WinoGrande −2.2, OpenBookQA −1.4, PIQA −1.1);
the abstract's "+1.4 to +3.5pp" is the reasoning subset only. The internal check (Table 6): masking the top-5%
retrieval heads **collapses** GSM8K 0.182→0.007 and BBH 0.317→0.043; masking random heads leaves BBH/MMLU-Pro/AGIEval
essentially intact (0.301/0.128/0.207), though GSM8K still drops ~30% relative (→0.127). GPT-4o rates its picks more
"reasoning" than the FineWeb-Edu classifier's **only in math/code domains** (OpenWebMath 0.52→0.88, Python-Edu
0.76→0.87); in FineWeb-Edu-Dedup and Cosmopedia the *classifier* scores higher.

**The fine print.** Two missing controls matter for us: (1) the intervention is **upsampling**, and selected docs are
**~2× longer** (OpenWebMath 1023→2256 tokens) — no matched-upsampling baseline (random 20%, length-matched, or
classifier-top-20% at the same token budget) is ever run, so "+0.75pp because *reasoning* content" is not isolated
from "upweighting longer/higher-quality docs"; (2) there is **no downstream training bake-off** against the
FineWeb-Edu classifier — the "better than a classifier" claim rests on data-composition analyses only. Table 6
validates the *head selection*, not the *data selection's* reasoning-vs-quality claim, and the
retrieval-head→reasoning equivalence is inherited from Wu et al., not validated here.

**Why it matters here.** This was recipe **(A)** for our reverse-filter — the cancellation logic (both losses share
weights, so memorization cancels) made it the most principled candidate. **We ran the generalized go/no-go on
2026-07-23 and it is a NO-GO** (`docs/RECIPE_A_SELF_ABLATION.md`): across 5 base models and 6 data sources the gap
detects *in-context copy dependence* — it ranks config files, parallel translations, and reference boilerplate at
the top of raw DCLM; the random-head control separates reasoning-vs-web as well or better (so the signal isn't even
retrieval-head-specific); 3 of 4 models rank verbal reasoning *below* web text; and the same family inverts across
scale (Qwen 7B→72B: GSM8K AUC 0.955→0.051). The mechanistic reason: retrieval heads are *copy* heads — the gap is
high exactly where next tokens are copyable from context, which is close to the opposite of a forced inferential
guess. This does not refute the paper's downstream pretraining result (their gain may well come from upweighting
long/structured/technical docs — see the missing controls above); it rules the gap out as a *reasoning* detector.

### 📖 Predictive Data Selection: "The Data That Predicts Is the Data That Teaches" (PreSelect)
Kashun Shum … Junxian He (both HKUST) · ICML 2025 · **20 citations** · `2503.00808`

**What it is.** A "which documents are worth training on?" method built on a sharp idea: a document is valuable if the
*ranking* of several models' compression of it (per-char loss) matches the models' known *ability* ranking. Call that
the document's **predictive strength**.

**Why it works (the mechanism — this answers "why are perplexity and capability correlated on some docs?").** Model
capability *is*, largely, "ability to predict structured text" — a model scores higher on benchmarks because it
captured more of the structure in its training data. So on a document whose predictability *depends* on that structure,
better models genuinely compress better, monotonically with ability. On surface-predictable text (frequency/n-grams)
every model does about equally well; on noise every model does equally badly — neither *rank-separates* the ladder. So
"does the whole ladder rank-order this doc's loss by ability?" is a filter for text whose difficulty is
*capability-shaped* — the text that capability is made of. Hence "the data that predicts is the data that teaches."
This is also why it beats the magnitude gap: a big-minus-small gap is maximized by text that's *easy once you're big*
(short/frequent), whereas rank-match requires the *entire* ladder to line up — a cleaner, nearly-orthogonal signal
(they measured Spearman 0.05 between the two).

**What they did.** Take a ladder of same-family models of known ability (six Llamas, 7B→65B). For each document, check
whether their per-char losses rank-order in the same order as the models' benchmark ability. Documents where the
ranking matches perfectly get top predictive strength. Train a cheap fastText classifier to imitate that score and run
it over the whole corpus.

**What they found.** Very effective: models trained on 30B PreSelect-chosen tokens beat models trained on **300B**
random tokens (a 10× efficiency win as a training-FLOPs statement — you still crawl and score the full pool), and it
beats other selection methods.

**The fine print.** Three caveats on the headline (the ScalingFilter comparison is untouched by them — same pool,
same setup, only the metric differs; it's the paper's cleanest result): (1) home-field confound — PreSelect's
fastText is trained *natively* on the eval pool with a target anchored to benchmarks that overlap the eval suite,
while the DCLM/FineWeb-Edu baselines are imported foreign scorers; (2) the "20% gains in Math and Code" are
**bits-per-char compression, not accuracy** — the authors state GSM8K/HumanEval accuracy is "negligible" at these
scales, so the math/code claim is surface-familiarity, not verified reasoning; (3) MMLU is flat at chance (~26) for
every method at 1B/3B, so real gains concentrate in ARC-E/SciQ/BBH/LAMBADA. The signal targets *general* downstream
ability — the word "reasoning" barely appears.

**Why it matters here.** This is recipe **(B)** for our reverse-filter — and crucially, it is **not a two-model
magnitude gap.** It scores by whether *many* models' per-char losses **rank-match** the models' ability order (the
*sign* over C(6,2)=15 pairs), which is a *different and nearly orthogonal signal* from "how much better is the big
model than the small one." Their appendix runs our exact 1.4B-vs-72B idea as a controlled baseline ("ScalingFilter" =
big-vs-small perplexity difference): it beats random by only **+0.4**, selects **short/easy junk**, and is
**uncorrelated (Spearman 0.05)** with the rank-match signal that works. So: (a) a two-model gap is a documented
near-failure — do not run it; (b) if we want recipe (B), use the **Qwen size ladder (0.5B→72B)** — same family, same
tokenizer, known ability order — and to target *reasoning* specifically, define the ability order by a reasoning
benchmark (their A.7.2 shows the ranking is steerable, at some cost to other axes). Normalize per-character
(tokenizer-agnostic), never per-token.

### 📖 Autonomous Data Selection with Zero-Shot Generative Classifiers for Math (AutoDS / AutoMathText)
Yifan Zhang … Andrew Chi-Chih Yao (both Tsinghua University) · ACL Findings 2025 · **26 citations** · `2025.findings-acl.216` (arXiv 2402.07625)

**What it is.** The simplest "find the reasoning text" recipe: ask a big model whether a document is mathematically
substantive, and keep the ones it says yes to.

**What they did.** Feed each document to a Qwen-72B base model with a fixed prompt asking two yes/no questions ("does
this show mathematical intelligence?" and "is it good for learning math?"), read how confident it is in "YES," keep
the high-scoring documents, and continue-pretrain on them.

**What they found.** It works at face value: a Mistral-7B continue-pretrained on the selected math text improves
**MATH 12.9 → 16.1** and **GSM8K 38.8 → 45.4**, at ~2.4× the token efficiency of using the unfiltered math corpus,
and it beats other selectors. On a small model (Gemma-2B) it only ties the baseline.

**The fine print.** The core AutoDS-vs-Uniform comparison is genuinely fair (same pool, token-matched, same
hyperparameters, same few-shot eval format). But: the 2.4× figure is one model (Mistral-7B) on one task (MATH) read
off noisy curves with no error bars or seeds; gaps over Uniform are often 1–2 points near the MATH floor; DSIR is a
strawman baseline (it targets Pile-Wikipedia as the "math" domain and lands *below* the no-pretrain base); and no
ablation tests what the Qwen-72B judgment actually keys on (reasoning vs generic cleanliness). "Autonomous"
overstates — the scorer is a 72B model curating for 2B–7B trainees, i.e. weak-to-strong curation.

**Why it matters here.** An example of "identify reasoning-rich text and it helps," and a useful *contrast* for the
perplexity-gap question: its signal is a *single strong model's* confidence, **not** perplexity and **not** a
weak-vs-strong gap — so there's more than one way to find this text.

### 📖 The FineWeb Datasets (incl. FineWeb-Edu)
Guilherme Penedo … Thomas Wolf (both HuggingFace) · NeurIPS 2024 · **1029 citations** · `2406.17557`

**What it is.** The famous web-data-cleaning paper; the relevant piece is **FineWeb-Edu**, a filter that keeps
"educational" web pages.

**What they did.** Had Llama-3-70B rate 460,000 web pages 0–5 for "educational value," trained a cheap classifier to
reproduce those ratings (82% F1), and used it to filter 15 trillion tokens down to a 1.3-trillion-token
"educational" subset.

**What they found.** Big gains on knowledge benchmarks (**MMLU 33 → 37, ARC 46 → 57**) from just filtering, at
matched model size and token budget.

**The fine print.** The MMLU/ARC gains carry a target-eval circularity: the filter's supervision target (a 70B
model's grade-school-to-college "educational value") is drawn from the same distribution as the school-exam
benchmarks it's scored on — and their own threshold choice concedes aggressive edu-filtering *hurts* HellaSwag. It's
domain steering that trades commonsense for exam-style knowledge; no control tests whether a generic
well-written-text signal would produce the same lift, and calling the gain "reasoning" overstates it (MMLU is
largely recall).

**Why it matters here.** The canonical "quality selection helps," but a **contrast case, not our target**:
"educational" is deliberately aimed at grade-school knowledge and *down-weights* technical/arXiv content — broader
than, and partly orthogonal to, "reasoning-rich." Useful for distinguishing "educational" from "reasoning" when we
design our own signal.

---

## H2.5 / H2.6 — augmenting text with reasoning, and how *complete* it must be

### 📖 Multi-Hop Knowledge Composition is Bound by Pretraining Exposure
Yannis Karmim (Inria, Paris & Chile) … Valentin Barrière (Universidad de Chile) · 2026 preprint · **0 citations** (too new) · `2606.09338`

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
  never installed for entities absent from compositional contexts. This is the paper's real contribution — and it
  survives the format critique below, because the *implicit* (eval-matched) formats also fail on `P_held`.
- **The explicit-vs-implicit result — read it only as a format statement.** Explicit augmentation (bridge entity
  named) gave **0.08** (= baseline); implicit (bridge omitted) gave **0.62** NL / **0.79** RDF. *The evaluation is
  unfair to the explicit condition:* explicit training teaches `P(Ashford | …Delia Crane was born in)` — it relies on
  the bridge token *being in the prefix* — while the test forbids any scratchpad and never puts the bridge in
  context. So this measures **train/test-format alignment**, not "explicit is worse at reasoning." The fair test —
  let the explicit model emit the bridge first, then score the answer — is never run; the paper explicitly rejects
  that paradigm as "externalizing composition at inference time," so the {explicit}×{scratchpad} cell of the 2×2 is
  missing by design. Their logit-lens analysis is a partial mechanism-level defense of the single-pass framing: the
  explicit model's bridge probability rises monotonically toward the output (a generation trajectory, not an
  intermediate variable) — i.e. explicit training installs "bridge as thing-to-generate," not "bridge as internal
  variable." Informative about mechanism; not a substitute for the missing experiment. (Conditions are also not
  strictly token-matched — augmentation adds tokens on a shared atomic backbone.)

**Why it matters here.** The claim is **narrow**: explicit compositional text does *not auto-compile into a latent
(no-scratchpad) direct-answer computation*, and composition is **exposure-bound** (you must expose the composition
itself, atomic facts don't suffice). This is not evidence that explicit/complete reasoning is useless — that would
require the scratchpad column they never ran. For our thread the real takeaways are: (1) if we want *latent*
reasoning, the training format must match the no-scratchpad inference distribution, and the *entities/operations*
(not just facts) must be exposed; (2) if we want *externalized* (chain-of-thought) reasoning, this paper says
nothing against it. Heavy caveats: fully synthetic, two relation types, GPT-2 scale, silent-reasoning-only.

### 📖 Faithfulness as Information Flow: Evaluating and Training Faithful Chain-of-Thought
Jinghan Jia (Michigan State University / Anthropic Fellows) … Eric Easley (Anthropic) · 2026 preprint · **0 citations** (too new) · `2605.24286`

**What it is.** A rigorous attempt to define what it even *means* for a reasoning chain to be "complete" and
"faithful" — it hands us a borrowable definition plus a warning.

**What they did.** They frame a good reasoning trace as one where all the answer-relevant information flows *through*
the written chain (prompt → chain → answer), and define three properties information-theoretically: **sufficiency**
(the chain alone determines the answer), **completeness** (given the chain, the prompt adds nothing more — i.e. the
chain "screens off" the prompt; a violation is a leftover prompt→answer shortcut), and **necessity** (the answer
actually depends on the chain, not just correlates with it). Then they try to *train* models to be more faithful by
tweaking the RL update.

**What they found.** Models routinely use a hidden prompt→answer shortcut while emitting a plausible-looking chain
they don't actually rely on — a chain can look complete and still be a rationalization. Their training interventions
make the shortcut more *visible* in the chain but don't remove it ("the interventions do not eliminate reward
hacking here, but they make it monitorable") — which is exactly why they insist on the separate "necessity" property.

**The fine print.** The training comparison itself is clean (identical rollouts/rewards, only the policy-update
gradient differs — no data/token confound). Two validity caveats: the external validation legitimizing the gradient
metrics rests on just **two** off-the-shelf models differing simultaneously in family, size, and RL recipe — weak
evidence the metrics track "faithfulness" rather than size/family/entropy; and the "~0.9 faithfulness" numbers are
verbalization/transparency metrics computed among hint-followers only (a conditioned subset), while the shortcut
*behavior* is unchanged (wrong-hint following persists; hidden-test pass stuck ~0.32). Borrow the definitions; don't
treat the metrics as validated instruments.

**Why it matters here.** Two gifts. First, a **precise, measurable definition of completeness** — "the chain screens
off the prompt; any leftover direct path is an incompleteness" — that we could actually compute. Second, the
**necessity caveat that reshapes the thread**: a complete-*looking* chain isn't enough; what matters is whether the
model *uses* it. That's the pivot from "is it complete?" to "does the encoding make the model actually run it?"

### 📖 Making Implicit Premises Explicit in Enthymemes
Xuyao Feng … Anthony Hunter (both UCL) · 2026 preprint · **0 citations** (too new) · `2603.06114`

**What it is.** The paper closest to our stopping-rule framing: for explicit logical arguments, does filling in the
unstated premise help? (An "enthymeme" is an argument with a missing premise — "Socrates is a man, therefore
mortal" leaves out "all men are mortal.")

**What they did.** Build a pipeline: an LLM (DeepSeek v3.2) generates the missing intermediate premise(s) — one,
two, or three steps — then a formal checker (AMR parse → logic → neuro-matching relaxation → SAT solver) verifies
whether the argument now goes through, scored against the dataset's gold entailment label.

**What they found.** On the headline metric, more steps help monotonically: entailment accuracy rises 0.53 → 0.73
(ANLI) and 0.29 → 0.56 (ARCT) as LLM-generated premises go from none to 3-step, and the LLM's premises beat the
datasets' own terse gold premises.

**The fine print — the headline is confounded; treat as suggestive.** (1) The headline table is computed **only on
items labeled entailment** — positive-class recall at fixed thresholds with **no paired specificity**; adding
premise steps mechanically supplies more overlapping AMR atoms for the neuro-matching relaxation, which makes the
SAT stack say "entails" more often *regardless of premise quality* — so the rise could partly be a loosened trigger.
(2) The **gold complete human premise barely beats no premise** (ANLI 0.558 vs 0.530; ARCT 0.303 vs 0.293) — a
correct, complete single premise does NOT help the verifier; only verbose multi-step LLM chains do, which points at
verifier plumbing (the stack needs many atoms to fire) driving part of the gain. (3) Balanced metrics are much more
modest: entailment-class best-F1 tops out at 0.59–0.67, overall accuracy peaks ~0.65–0.72, and the best-F1 table
cherry-picks a different threshold pair per row.

**Why it matters here.** The right *framing* paper — it operationalizes enthymeme-completion with formal
verification, exactly our stopping-rule setting. But what it demonstrates is "verbose LLM-generated premise chains
make a particular neuro-symbolic verifier fire more often on entailment items" — suggestive for
completeness-helps-explicit-reasoning, not clean evidence. The regime split (latent vs explicit) stands as a
hypothesis on the strength of Exposure's side; the explicit side awaits a cleaner test.

### 📖 Thinking Augmented Pre-training (TPT)
Liang Wang (Microsoft Research) … Furu Wei (Microsoft Research) · 2025 preprint · **3 citations** · `2509.20186`

**What it is.** The biggest "augment pretraining text with reasoning" result: append an automatically-generated
"thinking trajectory" to *every* document and pretrain on the concatenation.

**What they did.** One fixed prompt ("Simulate an expert's in-depth thought process… Use Feynman technique whenever
possible"), generated by **Qwen3-8B** for the from-scratch runs (DeepSeek-R1-Distill-7B for mid-training), thinking
capped at 8k tokens (~3× token inflation). From-scratch 8B on 100B tokens of FineWeb-Edu+MegaMath, token- and
step-matched against a vanilla baseline (which therefore sees ~3× more raw documents). Augmentation is **uniform** —
no per-document selection — and the completeness/depth/length of the thinking is **never varied** (their ablations
vary generation *strategy*, which barely moves the metric — a 1.5B generator even beats the 7B default).

**What they found.** Big: GSM8K 19.2→50.1, MATH 9.1→21.8, 5-task average 26.2→43.9 (vs LLaMA-3.1-8B@15T at 46.8);
gains persist and amplify through SFT (AIME24 1.0→35.2, MATH-500 33.8→82.4). A genuinely good data-matched control:
at a fixed 40B budget with ≤10B raw tokens (vanilla 4 epochs vs TPT 1), TPT still roughly doubles the average.

**The fine print.** (1) **Teacher-distillation confound, the big one:** the from-scratch thinking is written by
Qwen3-8B — a fully-trained, RL-tuned reasoner — so the headline partly *distills Qwen3-8B* rather than demonstrating
"decomposition improves learnability"; the weak-teacher ablation that would deconfound this is run only in
mid-training, never from scratch. (2) Every eval is a reasoning/CoT benchmark — no perplexity, HellaSwag, or
knowledge-recall control — so reasoning-specificity vs CoT-format-match is never isolated. (3) The loss comparison
is apples-to-oranges (augmented vs raw data distributions).

**Why it matters here.** The strongest existence proof for H2.5 that uniform reasoning-augmentation of pretraining
text pays off at scale — but for *our* question it's doubly incomplete: the gain can't be separated from distilling
a strong teacher, and completeness is applied at one fixed setting, never varied. The completeness dose-response
experiment remains unclaimed territory.

### 📖 Reasoning to Learn from Latent Thoughts (BoLT)
Yangjun Ruan (U. of Toronto) … Tatsunori Hashimoto (Stanford) · 2025 preprint · **40 citations** · `2503.18866`

**What it is.** The other flagship augmentation paper: treat the reasoning that *produced* a document as a latent
variable, generate it ("latent thoughts"), and train on thought+text; then bootstrap — the model generates its own
latents in an EM loop.

**What they did.** TinyLlama-1.1B continued-pretrained on FineMath-4+ (math web — the corpus is pre-selected as
reasoning-rich; augmentation *within* it is uniform, chunk-by-chunk). Section 5: latents from GPT-4o-mini, 480M
unique raw tokens, all baselines compute-matched at an 8B-token budget. Section 6 (BoLT proper): the 1.1B model
generates its own latents, importance-resamples them (targeting the IWAE bound), retrains, iterates — only a 240M
GPT-4o-mini warmstart.

**What they found.** Headline: MATH 5.74 (Raw-Repeat) → 25.38; GSM8K 5.76→33.59. The fair, teacher-matched
comparison is **Latent-Thought 25.38 vs WRAP-CoT 19.36** (both GPT-4o-mini-generated, same budget) — the novel
latent-thought design contributes **+6.0 MATH**, roughly a third of the headline delta; the rest is shared with a
generic "have GPT-4o-mini rewrite with reasoning" baseline. It also beats 8B *fresh unique* raw tokens (11.18) — the
genuinely surprising data-efficiency result, though that win also bundles distillation. The self-bootstrap (no
strong teacher in the loop) is real but modest: few-shot MATH ~13%→~20% over 3 iterations, plateauing; GSM8K
*deteriorates* over iterations on the fixed-data setup.

**The fine print.** Quote 25.38-vs-19.36 as the method's isolated contribution, not 5.7→25.4 (which bundles
GPT-4o-mini distillation). One eval prompt set is GPT-4o-mini-synthesized CoT (a train/test format-match advantage;
a standard prompt set mitigates). Missing: a GPT-4o-mini direct-answer skyline (to bound teacher capability) and any
completeness/length ablation.

**Why it matters here.** Conceptually the closest paper to our thesis — it names the exact mechanism ("web text is
the compressed final outcome of a verbose human thought process") and instantiates completeness-restoration. But its
strong-teacher results are distillation-confounded, its teacher-free loop yields modest plateauing gains, and
completeness is never varied. Together with TPT: "augmenting works" is established *only* in the
strong-teacher-distillation regime, on math-heavy corpora, at one completeness setting.

### 📖 Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
Eric Zelikman (Stanford) … Noah D. Goodman (Stanford) · COLM 2024 · **319 citations** · `2403.09629`

**What it is.** Teaches a model (Mistral-7B, light continued pretraining) to generate a short private rationale at
*every token position* to better predict the following text — REINFORCE-rewarded by whether the thought improves
prediction of the true next tokens, with a mixing head interpolating with/without-thought logits.

**What they found.** Zero-shot GSM8K 5.9→10.9, CommonsenseQA 36.3→47.2 (no task finetuning), scaling with the
*training* thought length. The result we cite most: perplexity gains are negligible on average but **concentrate
disproportionately on hard tokens** (theorem names, the start of the next proof step) — most tokens don't need
reasoning; a sparse tail does.

**The fine print.** The scored numbers are *direct answering* (multiple choice scored by answer-token logits — the
gains live in the weights, not test-time thinking), and a data-matched control exists (same model, same OpenWebMath,
no thought tokens — Fig 2), though the headline deltas are stated against off-the-shelf Mistral rather than clearly
against that control. Real limits: single 7B model, GSM8K stays near floor (10.9%), and "scales with rationale
length" confounds the mechanism with extra training compute.

**Why it matters here.** The hard-token concentration result is the closest thing in the literature to a per-token
"this token needed reasoning" signal (a *self*-generated with-vs-without-thought gap — one model, so memorization
largely cancels; same family as self-ablation). Its motivation is pure enthymeme-framing ("the steps not stated
between the lines of a proof") — but the method fills gaps *latently*, never making the text itself complete, so it
motivates rather than tests our completeness thesis.

---

## H2.7 — can a perplexity / weak-vs-strong gap detect reasoning content?

The short answer: **single-model perplexity — no; a two-*different*-model magnitude gap — also no (documented
near-failure); what works is either self-ablation (one model) or multi-model rank-match.** AttentionInfluence is
*self-ablation* (one model vs. itself with reasoning heads masked — memorization cancels because both losses share
weights). PreSelect is a *multi-model rank-match* (does per-char loss rank-order match the models' ability order?),
and it explicitly shows the two-model magnitude gap ("ScalingFilter") barely beats random (+0.4), picks short/easy
junk, and is uncorrelated (Spearman 0.05) with the rank-match. Our own reverse-filter's "gold" criterion *was* a
two-model gap (1.4B-high AND 72B-low) and it found knowledge, not reasoning — exactly what ScalingFilter predicts.
The sharper caveat from the full reads: **every working perplexity-family signal is validated as finding *generally
valuable* text, not reasoning specifically.**

### 📖 Rho-1: Not All Tokens Are What You Need
Zhenghao Lin (Xiamen U. / Microsoft) … Weizhu Chen (Microsoft) · NeurIPS 2024 · **126 citations** · `2404.07965`

**What it is.** Token-level selection by a loss gap: train a reference model on 0.5B *curated* math tokens
(GPT-synthesized + manually curated), score every corpus token by excess loss (training-model loss − reference loss),
and backprop only on the top 60–70% of tokens.

**What they found.** Big math gains at face value: GSM8K +23.4pp at 1B / +24.0pp at 7B over all-token continual
pretraining on the same 15B OpenWebMath corpus; "up to 30%" few-shot gains; 5–10× token-efficiency framing.

**The fine print — the key ablation is the paper's own.** With a **self-referential** reference model (trained on
OpenWebMath itself, no curated data), the average gain collapses **+16.5pp → +3.3pp** — i.e. ~80% of the headline
comes from the curated reference *distribution* leaking through the token mask, not from token selection as such.
This is distillation of a curated dataset through a mask. Also: the authors never claim the signal finds *reasoning*
tokens — their own words are "closely related to mathematics" and "aligned with the desired distribution"; the
efficiency framing counts only loss-bearing tokens (both conditions forward-pass the full corpus, plus RM
training/scoring overhead); the GPT-synthesized RM data is math-CoT-styled with no audit of overlap with GSM8K/MATH;
and after identical SFT the baseline recovers most of the gap (+2.2/+3.4pp on MATH).

**Why it matters here.** For H2.7 this reframes excess-loss selection: the detector detects "looks like my reference
set." That's a warning *and* an actionable idea — **if the reference model were trained on complete-reasoning text,
the same gap signal would point at completeness**; nobody has run that variant.

### 📖 Improving Pretraining Data Using Perplexity Correlations
Tristan Thrush (Stanford) … Tatsunori Hashimoto (Stanford) · ICLR 2025 · **54 citations** · `2409.05816`

**What it is.** The third multi-model signal, and the most observational: no training at all. Take ~90 public models
(33M–9B), compute per-*domain* bits-per-byte on ~10k web domains, and select the domains where loss **correlates**
with the models' benchmark accuracy (a rank-based single-index estimator robust to model-family heterogeneity), then
scale to page level with a fastText classifier.

**What they found.** At 160M/3.2B tokens it beats DSIR everywhere and roughly ties DCLM's handcrafted fastText
(which *wins* once given a manual language filter); preregistered runs show gains growing to 1.4B — but **only on
raw pools**; on pre-filtered pools the signal evaporates (correlation coefficients become homogeneous). Notably
honest: they preregistered the follow-up and reported the null.

**The fine print.** Their own Appendix I concedes **plain mean loss predicts model rank nearly as well** as the
correlation estimator (no individually significant comparison) — the incremental value of *correlation* over generic
"good models find it easy" is never isolated on the selection side. The top-correlated domains for a reasoning
benchmark (ARC Easy) are optometry-clinic and children's-hospital websites; for the DCLM aggregate they're
weather/finance/currency sites — a general-ability/quality/**language** detector, not a reasoning detector. The
90-model set includes many partially-trained checkpoints from one Pythia run (pseudo-replication), and the estimator
fails on models trained on atypical data (Phi).

**Why it matters here.** Validates that multi-model perplexity structure carries real selection signal (H2.7
feasible in principle) — PreSelect's within-family ladder and this cross-model rank-correlation are cousins — while
warning that (a) it operates at *domain*, not document, granularity; (b) it dies exactly in our regime (DCLM is a
pre-filtered pool); and (c) the naive version finds quality/language, not reasoning. A reasoning-specific target
would have to be engineered in (as PreSelect's A.7.2 steerability suggests).

---

# Batch 2 — discovery-pool full reads (40 papers, compact entries)

*From the wide zero-seed discovery sweep (`docs/DISCOVERY_POOL_2026-07-23.md`), the 40 highest-priority must-reads,
each full-read with the same adversarial protocol. Compact format: claim → numbers → fine print. The remaining ~86
must-reads stay queued in the pool doc.*

## Batch 2 · H2.5 — augmenting pretraining text with reasoning

**📖 MIND: Math Informed syNthetic Dialogues (NVIDIA, `2410.12881`)** — *the best-controlled augmentation result in
either batch.* Rewrites OpenWebMath chunks into knowledge-gap dialogues (7 styles); continued-pretrain 7B, 64B tokens,
**token-matched**: GSM8K +13.4 abs, and 33B of dialogue-augmented OWM **beats a 3.6× larger raw corpus**. Three
controls most augmentation papers lack: same-generator *rephrase* baseline ≈ raw (structure, not passthrough, is the
active ingredient); generator swap 70B→8B keeps ~all the gain (weakens distillation); and the *zero-knowledge-gap*
style (TWO PROFESSORS — experts assume shared knowledge, skip premises) gains **nothing** — the cleanest evidence yet
that *reasoning left implicit doesn't teach*, while forced explicitation does. Fine print: dialogue *length* is
uncorrelated with gain (completeness ≠ verbosity); their own rubric rewards "new knowledge" injection, never
quantified; no code transfer (HumanEval flat).

**📖 ToW: Thoughts of Words (`2410.16235`)** — token-level augmentation: GPT-4o writes ≤15-word inter-word rationales
(with an information bottleneck: generator can't see the gold word), continual pretraining on 6k docs. Reasoning avg
+2.7 to +9.0 across five 7–8B models; GSM8K +22.7 on Llama-3-8B. Fine print: no token-matched plain-GPT-4o control
(distillation unseparated); models learn to *emit* inline thoughts at eval (a learned scratchpad format, not better
raw next-token prediction); and the built-in dose-response runs **against verbosity** — raw 67-token thoughts <
denoised 30 < summarized 14.4 tokens/thought on all six benchmarks. Reusable ideas: the can't-see-the-answer
bottleneck, and honestly marking *unpredictable* words as unpredictable (drives their hallucination gains).

**📖 Rewriting Pre-Training Data (SwallowCode/Math, `2505.02881`)** — Llama-3.3-70B rewrites Stack-v2 Python
(style-guide + self-containment passes) and Finemath-4+ (restore missing context, step-by-step): HumanEval +17.0,
GSM8K +12.4 at matched 50B budget. Fine print: **purest distillation confound in the batch** — no weak-rewriter
control, and their own model-fixed comparison (LLM *scoring* <1pt vs LLM *rewriting* >14pt) points at
token-injection as the channel; MBPP dropped ~10pts from style-renaming and was excluded post-hoc.

**📖 Recycling the Web / REWIRE (`2506.04689`)** — rewrites the *discarded* low-quality DCLM pool with 70B-Instruct;
mix of top-raw + top-rewritten beats raw-only token-matched (+2.5pp CORE at 7B). Fine print, damning for the
naive reading: **rewritten text alone is WORSE than raw** (1B CORE 0.270 vs 0.289) — the gain only exists in the mix
and the authors credit *diversity/complementarity* (rank corr 0.179 between raw and rewritten quality), not
intrinsically better reasoning text; no weak-generator control; rewrites are *shorter* than sources.

**📖 Demystifying Synthetic Data (Meta, `2510.01631`)** — >1000 models, 100M–3B, up to 200B tokens: 1/3
WRAP-style-rephrased + 2/3 raw CC reaches equal loss with 5–10× fewer tokens; optimal synthetic ratio ~30%
(textbook-style: often <5%, and pure-textbook *hurts*). The anti-distillation datapoint: **Llama-3-70B-generated
rephrasings train consistently WORSE than 8B-generated ones.** Fine print: sole metric is perplexity on Pile
domains — the rephrase prompt targets Wikipedia style and Wikipedia-like domains are in the eval (style-match
confound); zero downstream reasoning benchmarks, so it is structurally blind to the thing we care about; no
quality-filtered natural baseline.

**📖 Kinetics of Reasoning (`2510.25791`)** — from-scratch synthetic, **ground-truth templated traces (no teacher —
no distillation possible)**: trace-supervised training flips OOD failure to success (Composition 0.00→1.00, Sorting
k=4 0.04→0.83), while answer-only training never gets there. Two sharp caveats: the model *still* finds the answer
shortcut early and only later aligns to the trace ("transient unfaithfulness" — CoT reshapes the trajectory, doesn't
delete the shortcut); and on the task that exceeds per-step capacity (Intersection), no trace template rescues it —
a true Can't. Not token/supervision-matched (CoT condition gets denser supervision).

**📖 Transformers Provably Learn to Internalize CoT (`2605.28600`)** — theory: k-parity is exponentially hard
sample-wise without CoT, polynomial with it, and the chain can then be *removed on a log-stage curriculum* leaving a
single-forward-pass solver — a proof-of-concept for the missing "internalization" cell of Exposure's 2×2. Fine
print: the architecture has gates *prescribed* from the parity tree (the model doesn't discover the structure), and
the empirical section is one qualitative run — "provably learn to internalize" is doing heavy lifting.

**📖 Grokking in the Wild (`2504.20752`)** — 124M model + data augmentation that raises the inferred-to-atomic fact
ratio φ above the grokking threshold → 2WikiMultiHopQA comparison OOD 0.59→0.96. The thread-relevant twist: **even
factually incorrect synthetic facts help** — the lever is corpus *statistics* (ratio), not teacher knowledge or
rationale quality. Fine print: composition OOD stays at 0.07 (only comparison groks); the "beats GPT-4o" framing is
a finetune-vs-zero-shot + format confound; grokking-scale training budgets are unrealistic for pretraining.

**📖 EntiGraph / Synthetic Continued Pretraining (`2409.07431`)** — for the *knowledge* regime: entity-graph
expansion of a 1.3M-token corpus to 600M synthetic tokens; closed-book QuALITY 39.5→56.4, log-linear in synthetic
tokens, and the CPT'd 8B **beats its own GPT-4-turbo generator** closed-book (56.4 vs 51.3) — hard to explain as
pure distillation. Fine print: not token-matched vs the rephrase baseline (~330× more tokens); "rearranges
knowledge," no reasoning-completeness variable; knowledge-internalization, not reasoning.

**📖 Procedural Knowledge at Scale (`2604.01348`)** — *shortlisting error (triage overreach): this is inference-time
RAG on frozen models, not pretraining.* Still one useful signal: retrieval of decomposed (subquestion → subroutine)
procedural memory beats generic document RAG, and the datastore-builder swap (QwQ-32B → Qwen3-8B) loses nothing —
another weak-generator-parity datapoint.

## Batch 2 · H2.6 — how complete must the reasoning be

**📖 Zipping the Thought (`2605.28008`)** — *the purest granularity dose-response so far* (synthetic mod-23
arithmetic, post-training). Explicit (every step) / Composed (ops named, values elided) / Implicit (ops+values
skipped): finer granularity learns from less data; compressed variants scale better with diverse data; and the key
asymmetry — **SFT cannot decompose below its training granularity (a completeness floor), but RLVR can re-derive the
skipped steps**. Fine print: measured in samples/steps, never tokens (explicit traces are longer — the efficiency
edge is partly token budget); synthetic-only, post-training-only.

**📖 Can LMs Learn to Skip Steps? (`2411.01855`)** — completeness as step-count, varied by self-training: moderate
skipping preserves ~99–100% in-domain accuracy and *improves* OOD on three easy algorithmic tasks — but their own
GSM8K probe shows the opposite: real reasoning "necessitates a complete reasoning chain" and skipped steps
"frequently contain errors." Fine print: the OOD gain is confounded with iterative self-distillation volume (no
full-step self-training control). Net: **completeness is difficulty-conditional.**

**📖 Less is More Tokens (`2509.05226`)** — difficulty-aware trace compression (post-training): over-compression
collapses hard benchmarks (SFT-only AIME 5.0, HMMT 0.0) while easy ones hold — restoring length recovers them. Same
difficulty-conditional message from the compression side. Fine print: no same-data full-trace control; the 7B
variant *loses* accuracy on every hard benchmark; a simpler RL length-controller (L1) beats its efficiency claim.

**📖 Inefficient-Reasoning Bias / shortest-path (`2507.05362`)** — 28.5M from-scratch on layered-DAG shortest paths;
**both token-matched AND data-matched** (rare): longer, locally-incremental DFS-style traces beat compressed DP
traces (~87% vs ~82%), no-intermediate-step traces collapse at depth — but length-matched *padding* (repeated steps)
does NOT help and induces repetition loops. The isolable mechanism is systematic local incrementality, not length.
Also note for H2.7: the valuable traces are the *lower*-perplexity ones here — opposite to loss-gap intuitions.

**📖 Principled Synthetic Logic Corpus / ALT (`2411.12498`)** — program-generated (no teacher) fully-explicit
multi-step deduction corpus + SFT: +4.1/+4.4 avg over 31 benchmarks at 8B/70B, with real cross-domain transfer
(MATH +5.2, HumanEval +10.3). Fine print: biggest gains are format-similar logic benchmarks (RobustLR +32); no
full-proof-vs-answer-only control at matched tokens, so completeness is baked in, not isolated; whole effect
contingent on an anti-forgetting regularizer.

**📖 The Model Says Walk (`2603.29025`)** — inference-time: models key on surface cues 8.7–38× more than the goal,
fail to use *unstated* constraints (<75% strict accuracy for all 14 frontier models — the Won't persists through
current post-training), and a one-word hint recovers +15.3pp (knowledge present). The completeness-relevant cell:
goal-decomposition prompting (enumerate necessary conditions first) beats generic CoT (+5.0 vs +3.1pp) — chain
*structure* over raw reasoning tokens. Fine print: nothing token-matched; entirely inference-time.

## Batch 2 · H2.4 — identifying reasoning-rich text

**📖 Procedural Knowledge in Pretraining Drives Reasoning (`2411.12580`)** — influence functions over 5M docs (7B +
35B Cohere models): for *reasoning* queries, influence is spread over procedurally-similar documents (code, worked
solutions) and the answer is NOT in the top docs (unlike factual queries, where it is); same-task reasoning queries
draw on correlated doc sets (p<4e-8). The best observational definition of "reasoning-rich = procedural/worked-out
text" we have. Fine print: correlational — no ablation/upweighting retrain; 2.5B-token subsample; "procedural"
labels are model-graded; MLP-only influence approximations.

**📖 Which Data Attributes Stimulate Reasoning (`2505.19949`)** — influence-function attributes + interventions: the
genuinely useful piece is the *exploration-truncation ablation* (strip verification/backtracking behaviors from SFT
traces → MATH500 −3.4) — causal evidence that trace *behaviors*, not just answers, carry value. Fine print: the
headline difficulty-flip swaps in a different source corpus (not source-matched), tokens unmatched, single-seed on
30-problem AIME.

**📖 Essential-Web v1.0 (`2506.14111`)** — 24T tokens with per-doc 12-field taxonomy incl. a 5-level
**reasoning-depth label**, annotated by a 0.5B distilled classifier that *beats its 32B teacher* on agreement
(κ 0.87) at 50× cheaper — reasoning-richness is a learnable, scalable annotation. Fine print: no ablation isolates
the reasoning-depth clause (subject alone recalls 96–98% of vetted math/code); on actual math benchmarks the
taxonomy filter *lags* FineMath (−8%); advertised wins are MMLU-knowledge subsets; and their own appendix shows the
one LLM-*rewritten* corpus (MegaMath-Web-Pro) tops every filter-only math corpus — an incidental augmentation-beats-
selection datapoint.

**📖 The Data-Quality Illusion (`2510.00866`)** — mechanism for why classifier filtering misleads: a CQF score is
provably a density ratio ("distance from web"), inherits the HQ set's task bias, and demonstrably latches onto
sequence length; training on filtered data can *hurt* loss on the HQ target itself. Compute/token/repetition-matched
comparisons (good). Directly explains FineWeb-Edu-style circularity and warns our own gate designs.

**📖 Reasoning Quality Emerges Early / TEMP (`2606.26797`)** — cheap difficulty signal: loss on the first ~100
tokens at a *noise-perturbed* checkpoint correlates r~0.9 with judged difficulty (beats length-proxy r~0.75); +1.1–1.7%
selecting 1k SFT examples. For H2.7: their brittleness signal is a **noise-weakened-vs-clean same-model loss gap** —
another self-contained weak-vs-strong variant. Fine print: pool is all-reasoning traces (no reasoning-vs-non-reasoning
control); small margins, mixed per-benchmark.

**📖 Beyond Pure Code (`2605.19762`)** — for our code-ladder: at a *fixed math token budget*, swapping ordinary math
for **cognitive-scaffold-structured math** (explicit subgoals/derivations, selected by classifier — no generator, no
distillation) gains Olympiad +47.8%, College Math +30.1%, while GSM8K *regresses* −6.3 (task-dependent optimum
again). Fine print: the "code hurts math" headline arm is confounded (removing code proportionally upsamples math);
the scaffold classifier tracks "external organization," i.e. structure/formatting, not verified reasoning.

## Batch 2 · H2.7 — loss/perplexity-family signals

**📖 ScalingFilter (`2408.08310`)** — the primary source of the two-model magnitude gap, now read directly: 124M-vs-
774M GPT-2 perplexity ratio as "quality." Its own validation: +1.12% avg over perplexity-gating and +0.62% over a
binary classifier on 7 *commonsense* tasks, 1.3B/25B — no error bars, no seeds, no reasoning or knowledge-intensive
benchmark anywhere. The signal was never even claimed to find reasoning; treat "two-model gap" as validated only for
filtering simple/repetitive text on raw pools.

**📖 Perplexed by Perplexity (`2405.20541`)** — single small-reference perplexity pruning works for general
downstream (+2.0 avg at 3B on the Pile, compute-matched vs full pool) — but the winning criterion (keep
high-perplexity) **removes code and scientific papers ~3×**, i.e. it actively de-selects reasoning-dense domains;
the optimal direction flips per corpus; no random-50% control. Anti-reasoning as a selector.

**📖 rBridge (`2509.21013`)** — small-proxy NLL evaluated **on frontier reasoning traces** predicts large-model
reasoning accuracy (R² 0.87 at 1B→13B vs 0.49 for standard NLL; 80.8% decision accuracy at ≥100× FLOPs savings).
Fine print: their own ablation shows ~all the gain is the *trace target* (a distilled notion of good reasoning), not
the weighting machinery; dataset-level not document-level; untested within a pre-filtered pool. The transferable
lesson: *what you evaluate the loss ON matters more than which models you difference.*

**📖 Generalization vs Memorization (`2407.14985`)** — corpus-frequency signals (task-gram co-occurrence over the
Pile/Dolma): strong for TriviaQA-style recall (and rising with scale), absent for GSM8K — reasoning performance is
not explained by task-relevant n-gram frequency. Fine print: partly a measurement floor (reasoning outputs are
un-memorizable-as-n-grams by construction), so read as "frequency signals find knowledge, not reasoning," not as
proof reasoning has no memorized substrate.

**📖 The Signal is in the Steps (`2510.03988`)** — for selecting distillation traces, *global* sequence perplexity
fails at long context; a *local* windowed self-perplexity (is each step justified by its immediate premises?) picks
better traces (+9.4pp avg over global on 32B). Fine print: local-pick is length/teacher-confounded (79% of picks
come from the two longest-CoT teachers; near-tie with just using the best single teacher). Directionally consistent
with our thread: the reasoning signal lives in step-to-step transitions, not whole-sequence fluency.

## Batch 2 · H1 — shortcuts, latent composition, persistence

**📖 The Pitfalls of Next-Token Prediction (`2403.06963`)** — path-star: teacher-forced training collapses to chance
(~1/d) because the *complete* target path lets the model cheat (later steps are trivial lookups), starving the
pivotal first-token decision of supervision — a Won't that hardens into a Can't. The thread-critical caveat: a
teacher-forced complete chain can *entrench* the shortcut whenever intermediate steps leak the answer — augmented
reasoning must be left-to-right-derivable (each step inferable from prior context). They test ordering and masking
but never the add-explicit-derivation-first condition — our exact lever remains untested. Synthetic-only.

**📖 Faith and Fate (`2305.18654`)** — compositional tasks as computation graphs: models succeed by "linearized
subgraph matching" (82.3% of correct answers contain computation *errors* — right answer, wrong process); GPT-3
finetuned on **maximally complete scratchpads** reaches near-perfect in-distribution but still collapses on
deeper/wider graphs. Fine print: scratchpad-vs-answer arms not compute-matched; OOD = complexity never seen in
training, so this bounds *extrapolation*, not whether complete reasoning helps in-distribution (it does, strongly).

**📖 Physics of LM 3.2: Knowledge Manipulation (`2309.14402`)** — bioS-controlled: knowledge retrieval ~97% but even
trivial manipulation (birth-month parity) is near-random *without a test-time scratchpad*, needs ~10k samples for
75% even fully supervised — and the killer control: **trained WITH CoT, tested without → still fails.** Convergent
with Exposure and CompCollapse: explicit reasoning in training text does not auto-compile into latent computation;
the deployed model must *emit* the chain. No distillation confound (fully synthetic).

**📖 Implicit Reasoning through Shortcuts (`2503.07604`)** — from-scratch GPT-2 on multi-step arithmetic: trained on
*fixed computation-order* chains → genuine, OOD-robust implicit reasoning (100%/99%/90%); trained on unfixed-order
data → number-chaining shortcut that collapses under variable-as-subtrahend (0.92→0.03). SoTA LLMs shortcut on the
same probe (GPT-4o ~100%→~30%). The demonstrated lever is chain ORDER/coherence — data structured so the shortcut
is unavailable. Fine print: the LLM probe forbids CoT and filters to no-CoT outputs (unquantified selection); the
reorder-as-intervention experiment is never run.

**📖 Composition Collapse (`2605.26789`)** — independent corroboration of Exposure's theme on *real* models with a
double-gate protocol (score composition only where atomic facts pass): failure rises to 100% by depth 8 in
single-pass mode, CoT recovers 70–75%, activation patching null → assembly bottleneck, not lookup. Two sharp
post-training data points: **SFT on reasoning traces *worsened* composition** (76.9% vs 69.8% baseline) while
outcome-RL (GRPO) helped but only on trained depths (OOD/ID 0.21). Fine print: the 47pp cross-recipe gap compares
different base models (correlational); residual-failure numbers are no-scratchpad-scored.

**📖 Two-Hop Curse / Lessons (`2411.16353`)** — fully-synthetic facts learned in *separate* documents: two-hop
no-CoT = chance with loss at the random baseline (no latent path exists — not a hidden-but-unused skill), while CoT
works; **same-document co-occurrence produces non-zero latent composition** — the failure is partly a data-
*arrangement* artifact. Fine print: the semi-synthetic ≈20% "success" bundles storage-depth/co-occurrence confounds
(their own lesson); no controlled scale sweep.

**📖 Identity Bridge (`2509.24653`)** — theory + toy: standard training provably settles into a shared-memory
shortcut (Thm 2: OOD two-hop margin <0 — unrecoverable) and a small *data-side* fix (identity two-hops "who is X's
[null-relation]") provably restores OOD composition. Fine print they under-advertise: on real pretrained LLMs the
improvement is "not significant" — cite as mechanism, not as a working intervention; no comparison against explicit
bridge-token supervision.

**📖 SynthWorlds (`2510.24427`)** — parallel real/synthetic worlds with identical reasoning structure: in the
knowledge-equalized reading-comprehension cell models reason *as well or better* on synthetic (KA ≤ 0) — the
"reasoning gap" in other cells is knowledge acquisition + retriever familiarity, not reasoning. But in navigation,
models inject memorized off-page entities in 35–48% of steps even when the correct page is in front of them — the
recall shortcut is *chosen* even when the reasoning path is fully available. A reusable can't-vs-won't instrument
(they suggest a code variant: rename numpy/pandas).

**📖 U-shaped implicit-reasoning scaling (`2504.03635`)** — from-scratch KG study: implicit multi-hop capacity is
**~0.008 bits/param vs ~2 bits/param for knowledge storage (~250×)**, optimal model size is a U (bigger memorizes
composables instead of reasoning), R²=0.85 scaling fit. The cleanest first-principles argument for externalizing
reasoning into tokens rather than expecting latent composition. Fine print: fixed-step (not convergence-matched)
training, weight_decay=0 with no regularization sweep, no 1-hop-conditioned scoring — cite as motivation, not law.

**📖 Echo Chamber (`2504.07912`)** — from-scratch 150M/1B + RL: RL collapses onto whichever solution format
*dominated pretraining* (regardless of whether it's the better one), pass@1 up while pass@64 *declines* — RL as
amplifier of pretraining modes, not creator. Direct persistence implication: if the dominant pretraining mode is a
shortcut, RL will amplify the shortcut. Fine print: the more-TinyGSM→more-gain curve is not token-matched;
format-vs-reasoning not fully separated.

**📖 Base Models Know How, Thinking Models Learn When (`2510.07364`)** — steering-vector elicitation of
reasoning behaviors from base models; headline 91% gap-recovery is the single best cell (spread 0–91%), and their
own ablation shows random-direction steering recovers most of the lift (specific direction adds ~7pts of ~21) —
the "base already knows how" claim is soft. Useful as a caution against strong elicitation claims, ours included.

**📖 Spurious Correlations in Post-Training (`2505.05704`)** — SFT vs DPO/KTO under injected shortcuts: orderings
flip by task; 90% contamination sometimes *helps*. Fine print that limits its use for us: no base-model or clean-data
control, so it cannot adjudicate whether pretraining-installed shortcuts persist — it measures uptake of freshly
planted artifacts only.

**📖 Pre/Mid-Training × RL Interplay (`2512.07783`)** — 100M from-scratch on synthetic DAG math, the elicit-vs-add
distinction done right (pass@1 vs pass@128): RL on covered tasks lifts pass@1 with **zero pass@128 gain**; RL cannot
transfer to a context below ~1% pretraining exposure (≥1% → +60% pass@128); at fixed compute, mid-training on gold
traces + RL beats RL-alone by +10.8% on OOD-hard. Fine print: the mid-training arm gets gold traces (information,
not just compute, is unmatched); ID saturation is by construction; synthetic-only. The ≥1%-exposure gate is the
sharpest quantitative version of "pretraining content bounds post-training" in either batch.

---

## Open questions (genuinely open — not prescriptions)

1. **Which inference format is our thread about?** Sharper after batch 2: three independent setups converge on
   "reasoning in training text pays off at inference only if the model *emits* the chain" (Physics-3.2's
   trained-with-CoT-tested-without still fails; Exposure; Composition Collapse's 70–75% CoT-recovery), and
   internalization into latent computation seems to need either compositional *exposure* (Exposure) or a staged
   removal *curriculum* (Internalize — theory only). Targeting latent reasoning also fights a ~250×-per-bit capacity
   penalty (U-shape); targeting externalized reasoning means the augmentation must teach chains the deployed model
   will actually produce.
2. **Does augmenting pretraining text with reasoning help — beyond distillation?** Partly de-confounded by batch 2:
   the structure effect is real without strong teachers in math/synthetic regimes (MIND's generator swap 8B≈70B plus
   same-generator rephrase control; Demystify's 70B-written-data-trains-WORSE-than-8B; GrokWild's
   incorrect-facts-still-work; the programmatic-trace results Kinetics/Logic-Corpus/Internalize), but where controls
   are missing the distillation channel is large (Swallow's own scoring-vs-rewriting comparison; REWIRE's
   rewritten-alone-worse-than-raw). **Unproven: natural, general (non-math) web text at pretraining scale with a
   weak generator.** The completeness dose-response now exists in fragments (Zipping's granularity ladder,
   Skip-Steps' difficulty-dependence, ToW's anti-verbosity, MIND's knowledge-gap lever, Inefficient-Reasoning's
   incrementality-not-length) — all synthetic or post-training. The natural-text version (fixed tokens, vary
   completeness of injected reasoning) is still unclaimed, and batch 2 supplies its design constraints: vary
   *granularity* not verbosity; respect difficulty-conditionality; keep chains computation-ordered and
   left-to-right-derivable (no answer leakage — Pitfalls-of-NTP); expect an SFT-side granularity floor (Zipping).
3. **For the data-selection side (reasoning-rich text):** recipe A (self-ablation) is **ruled out by our own
   experiments** (copy-dependence detector, scale-inverting — `docs/RECIPE_A_SELF_ABLATION.md`); the two-model gap
   was already ruled out (ScalingFilter — primary source now read: validated only on commonsense, never claimed for
   reasoning). Still standing, untested: **recipe B, multi-model rank-match on the Qwen ladder** — with PerpCorr's
   warning that plain mean loss predicts nearly as well, and rBridge's lesson that *what text the loss is evaluated
   on* (frontier reasoning traces) may matter more than the model-differencing scheme. Ideas worth holding: a
   reference model trained on complete-reasoning text pointing RHO-1's excess-loss at completeness; TEMP's
   noise-perturbed-vs-clean same-model gap. No published signal in this family has isolated *reasoning* from
   general quality.
4. **Does under-reasoning persist through *our* post-training?** Batch 2 sharpens the prior: RL is bounded by
   pretraining content (Echo Chamber's mode-amplification; Interplay's ≥1%-exposure gate and flat pass@128 on
   covered tasks), the Won't survives frontier post-training (Model-Says-Walk <75% strict for all 14 models), and
   Composition Collapse warns that *SFT on reasoning traces can worsen composition* while outcome-RL helps only on
   trained depths. Still untested on our own ladder with our rewritten-web-text intervention.

---

*Provenance: core 24 papers discovered by a zero-seed neutral search (`wf_869397f2-d8b`) and full-read by
`wf_13d49562-ffa` (2026-07-23; earlier pass `wf_e16faf72-dc2` covered 16). Batch 2: wide two-round discovery
(`wf_438a8a3c-3b1` + concept-expansion recall round `wf_66465130-feb`, 533 candidates, triage `wf_6e18afd1-6ab` —
full pool in `docs/DISCOVERY_POOL_2026-07-23.md`), top-40 full-read by `wf_4006ecb6-289`. All readers: one agent per
paper, HTML/PDF full text, schema-validated extraction (method / body numbers / verbatim quotes / can't-vs-won't /
completeness / limitations / verdict / mandatory eval-fairness critique); per-paper records in the session's
`subagents/workflows/<id>/journal.jsonl`. Tier-3 code deep-dive of AttentionInfluence + PreSelect against their
repos: `wf_1163664e-5a9` (2026-07-23). Recipe-A empirical no-go: `docs/RECIPE_A_SELF_ABLATION.md` (experiment
session, 2026-07-23). The earlier abstract-only map and `docs/PERSISTENCE_AND_USEFUL_REASONING.md` are superseded by
this document.*
