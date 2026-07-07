# Why code pretraining improves reasoning — and the gap our project fills

Citation counts = Semantic Scholar, fetched 2026-07-04.

## The headline gap (motivation)
**No paper runs the direct experiment our idea implies:** take NL data, make its *latent* reasoning
**explicit/complete** (spell out intermediate steps/state), and measure whether that alone recovers
code-like transfer **without any code.** Closest proxies point the right way but don't isolate it.
→ The "make implicit reasoning explicit in prose ⇒ code-like transfer" mechanism is **plausible,
indirectly supported, not directly demonstrated.** This is the fillable gap.

## Consensus is converging on STRUCTURE, not executability
| Paper | arXiv | cites | Key finding + quote |
|---|---|---|---|
| **Aryabumi 2024** "To Code or Not To Code" | 2408.10914 | ~51 | code → **+8.2% NL reasoning**, +4.2% world knowledge, 12× code (470M–2.8B). Synthetic/cleaner code helps NL reasoning +9% over web code. "code is a critical building block for generalization far beyond coding tasks." **No mechanism claimed** — but the synthetic-code win says quality/structure, not raw volume, transfers. |
| **Madaan 2022** COCOGEN (LMs of Code are Commonsense Learners) | 2210.07128 | ~274 | "when we instead frame structured commonsense reasoning tasks as code generation tasks, pre-trained LMs of code are better structured commonsense reasoners than LMs of natural language, **even when the downstream task does not involve source code at all.**" → benefit is **structural framing**. Caveat: about output-as-code at inference, not code-in-pretraining. |
| **Petty 2024** How Does Code Pretraining Affect Task Perf | 2409.04556 | ~26 | Cleanest causal code-fraction study (~374M). "code improves performance on compositional tasks involving structured output... and mathematics. Conversely, increase code mixture can **harm** performance on... syntax or morphology, and... real-world knowledge." → **scoping constraint** (mirrors our own code↔NL tradeoff). |
| **Kim 2024** Code Pretraining Improves Entity Tracking | 2405.21068 | ~28 | Matched pairs (Llama-2 vs Code Llama, etc.) on "boxes" state-tracking. Code models win big on multi-op tracking. "keeping track of the states of variables is important for producing correct code... this kind of procedural input may provide a stronger training signal." → sub-mechanism = **explicit multi-step state tracking.** |
| **Waheed 2025** On Code-Induced Reasoning | 2509.21499 | ~3 | Isolates structure vs semantics, 10 langs, 0.6–8B, 3331 exps. "Structural perturbations consistently degrade performance more severely than semantic ones." **Pseudocode/flowcharts (≈structured text) recover most of the benefit.** → strongest support for "structure in TEXT should transfer like code." |
| **Zhao 2026** Structured Reasoning Signals Beyond Pure Code | 2605.19762 | ~1 | 10T-token controlled sep. "when code is restricted to standalone executable programs and Code-NL data are controlled for, code... **does not act as a general reasoning enhancer**; instead, it competes with... math reasoning" and "the reasoning gains... are better explained by **cross-domain structured reasoning traces, such as code-text and math-text mixtures**, rather than by executable code alone." → **most on-point paper** for our thesis (new, ~1 cite — treat as promising signal not settled). |
| Yang 2025 (survey) "Code to Think, Think to Code" | 2502.19411 | ~55 | received view: code "offers an abstract, modular, and logic-driven structure... provides verifiable execution paths, enforces logical decomposition." |

## Bearing on our design
- Effect size to beat / target: Aryabumi's +8.2% NL reasoning from code.
- Petty's scoping: expect gains on structured-output/math reasoning, risk of HARM on syntax/morphology/knowledge + perplexity — **the same code↔NL tradeoff we measured in the ladder.** Watch NL/perplexity closely.
- Waheed + Zhao: the active ingredient is structured *text* traces → our augmentation should work IF it captures that structure. Our contribution = isolate "implicit→explicit in ordinary prose" which none of them did.
