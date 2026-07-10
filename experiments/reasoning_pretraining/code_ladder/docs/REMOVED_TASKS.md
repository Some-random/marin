# Removed tasks — what they are, why removed, and 0/5/10-shot behaviour

Tasks pulled from the small-scale (300M–1.4B) suite into **Collapse**. Groups **C. Math** and **E. Aggregate** are dropped wholesale (no scale-matched reference — phi/Aryabumi/Suhas — uses them). Below are the individually-moved A/B tasks. Three models: **c5v6** (our code→text), **A5** (our full DCLM-text baseline), **phi-1.5** (reasoning-dense 1.3B reference).

> **What 'at chance' means:** a multiple-choice score equal to blind guessing (or to always picking the majority answer) — the model has *no* real ability to pick the correct option; its answers are driven by a prior (which letter/label it favours), not by understanding the question. The *answer-choice distribution* lines below show this directly: a model 'at chance' keeps firing the **same option** regardless of the question. Few-shot examples reshuffle *which* option it fires, but don't turn it into real choosing.

## boolq
**What it is:** A yes/no question plus a Wikipedia passage that contains the answer.
Chance is 50% (coin flip), but the real bar is the **62% 'yes' majority** — always answering 'yes' scores 62% without reading anything. So 'at chance' here means *at or below the 62% prior*.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | no49 yes51 · 55% | no59 yes41 · 51% | no65 yes35 · 50% |
| A5 (DCLM text) | no36 yes64 · 56% | no55 yes45 · 52% | no49 yes51 · 53% |
| phi-1.5 | no43 yes57 · 75% | no40 yes60 · 76% | no41 yes59 · 76% |

**Why removed:** Our models sit on (and below) the yes-prior instead of reading the passage — a label-collapse, so it's pulled out of the open-book group.

## mmlu
**What it is:** 4-option exam questions across 57 subjects (history, math, law…).
Chance is 25% (1 of 4). 'At chance' = the model can't separate the correct option from the 3 distractors, so it scores like blind guessing.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | A30 B23 C33 D14 · 24% | A59 B15 C19 D7 · 24% | — |
| A5 (DCLM text) | A6 B34 C44 D17 · 25% | A38 B16 C30 D16 · 24% | — |
| phi-1.5 | A25 B33 C24 D17 · 44% | A27 B27 C30 D17 · 44% | — |

**Why removed:** Letter-scored; our models collapse to one letter. Real task (phi clears it) but scale-limited to ~8B+. 10-shot is impossible — mmlu only ships 5 dev examples for the few-shot pool.

## commonsense_qa
**What it is:** Pick 1 of 5 answers (A–E) to a commonsense question.
Chance is 20% (1 of 5). 'At chance' = no real ability to pick the right answer; the score is whatever prior over the letters the model happens to have.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | A85 B1 C11 D3 E0 · 20% | A73 B4 C21 D2 E0 · 20% | A59 B5 C36 D1 E0 · 20% |
| A5 (DCLM text) | A29 B2 C45 D24 E0 · 19% | A98 B1 C0 D1 E0 · 19% | A95 B3 C1 D1 E0 · 20% |
| phi-1.5 | A15 B31 C29 D20 E6 · 51% | A21 B23 C25 D22 E8 · 54% | A21 B23 C26 D20 E10 · 54% |

**Why removed:** Letter-scored 5-way; collapses to a single letter. 0/5/25-shot all pinned at chance — few-shot never helps.

## cb
**What it is:** 3-way entailment: does the premise entail / contradict / leave neutral the hypothesis. N=56 (tiny).
Chance 33% (1 of 3); majority ~50%. 'At chance' = guessing among the 3 labels, dominated by whichever it favors.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | entail95 contra5 neutral0 · 39% | entail48 contra52 neutral0 · 48% | entail66 contra34 neutral0 · 43% |
| A5 (DCLM text) | entail71 contra21 neutral7 · 32% | entail61 contra39 neutral0 · 61% | entail62 contra38 neutral0 · 48% |
| phi-1.5 | entail55 contra32 neutral12 · 64% | entail41 contra55 neutral4 · 77% | entail48 contra52 neutral0 · 82% |

**Why removed:** Collapses to one class, never predicts the 3rd (neutral). Tiny N=56 + label collapse.

## winogrande
**What it is:** Which of 2 candidate nouns a pronoun refers to (adversarially filtered so word-association shortcuts don't work).
Chance 50% (2 options). 'At chance' = the model genuinely can't tell which candidate is right — balanced 50/50 picks, no artifact.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | opt146 opt254 · 50% | opt150 opt250 · 52% | opt151 opt249 · 52% |
| A5 (DCLM text) | opt151 opt249 · 54% | opt155 opt245 · 54% | opt155 opt245 · 54% |
| phi-1.5 | opt152 opt248 · 73% | opt150 opt250 · 71% | opt151 opt249 · 71% |

**Why removed:** Our models flat at chance across sizes; phi-1.5 0.71. KEEP as a reasoning TRIPWIRE (Regime 3) — it stays dark until ~8B, so it can't rank our runs, but it will light up when reasoning appears.

## arc_challenge
**What it is:** Hard grade-school science MC, 4 options (the ARC 'challenge' split).
Chance 25% (1 of 4). 'At chance' = can't beat guessing among 4 options.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | A36 B28 C20 D17 E0 · 25% | A35 B28 C21 D16 E0 · 26% | A36 B28 C21 D16 E0 · 27% |
| A5 (DCLM text) | A36 B26 C20 D17 E0 · 28% | A37 B27 C21 D16 E0 · 31% | A36 B27 C20 D16 E0 · 31% |
| phi-1.5 | A32 B27 C22 D20 E0 · 48% | A31 B28 C22 D19 E0 · 51% | A32 B28 C22 D19 E0 · 52% |

**Why removed:** Text-scored but the content is too hard for our scale — near-chance (raw acc is sub-chance, needs length-normalised acc_norm).

## logiqa
**What it is:** Formal logical-deduction puzzles, 4 options.
Chance 25% (1 of 4). 'At chance' = no deduction, just guessing among 4.

Each cell = **answer distribution** (% of questions the model assigned to each option) **· accuracy**.

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 (code→text) | A58 B13 C9 D20 · 27% | A58 B11 C12 D19 · 24% | A58 B11 C12 D18 · 25% |
| A5 (DCLM text) | A48 B15 C13 D24 · 32% | A61 B12 C10 D16 · 27% | A63 B12 C11 D14 · 27% |
| phi-1.5 | A51 B19 C16 D14 · 29% | A32 B17 C26 D25 · 27% | A30 B17 C29 D23 · 25% |

**Why removed:** Near-noise for our models (~chance) AND barely above chance for phi-1.5 (few-shot HURTS both) — closer to the fully-dead bucket than a scale-limited one.

---

**wsc (binary yes/no): removed and replaced — settled, not re-litigated here.** Swapped for `wsc273` (referent-choice, Marin-aligned, 0-shot only): c5v6 **0.601**, A5 **0.586**, phi-1.5 **0.769**. Above chance for us via surface co-occurrence, though winogrande (fluency-filtered) shows the reasoning itself is still at chance for our models.