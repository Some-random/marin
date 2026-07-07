# Datasets — reasoning-completeness / claim→premise-chain (discovery, 2026-07-04)

Discovered via HF card fetches. **Honesty flags:** `allenai/entailment_bank` & `allenai/proofwriter`
return 401 (no public HF repo at those paths) → use community mirrors below. `nguyen-brat/worldtree`
card unfetched (sizes from paper). COMET is a model; the data is ATOMIC.

## Structure sources (claim → explicit premise chain) — the template to imitate
| HF path | size | fields / structure | note |
|---|---|---|---|
| `ariesutiono/entailment-bank-v3` | 46 MB | `hypothesis`, `proof` (`sent1 & sent3 -> int1; int1 & sent2 -> hyp`), `full_text_proof` ([BECAUSE]/[INFER]), `context` sent1..25, `depth_of_proof` | **BEST FIT** — explicit multi-step entailment tree w/ named intermediate conclusions |
| `tasksource/proofwriter` | 845k rows / 43 MB | `theory` (facts+rules), `question`, `answer`, `allProofs`, `maxD` (depth ≤5) | synthetic deductive proof chains at volume |
| `nguyen-brat/worldtree` | ~tens MB (unconf.) | explanation graphs, avg 6 facts from ~9k-fact tablestore | substrate EntailmentBank built from |
| `allenai/qasc` | 5.9 MB | `question`, `fact1`, `fact2`, `combinedfact` | cleanest 2-hop composition (2 premises → conclusion) |

## Pretrain-scale reasoning-dense corpora (source text to augment / or use directly)
| HF path | size | note |
|---|---|---|
| `open-web-math/open-web-math` | 27.4 GB / 14.7B tok | **ALREADY TOKENIZED locally as `openwebmath/`** — worked solutions = implicit chains |
| `HuggingFaceTB/cosmopedia` | 31M rows / ~25B tok | synthetic textbooks — reasoning-dense prose (not proof trees). NOT local. |
| `EleutherAI/proof-pile-2` | 51 GB / 55B tok | arxiv+owm+algebraic-stack; overlaps owm. NOT local. |

## Already tokenized locally (no cost) — flagged for the plan
- **`openwebmath/`** = open-web-math (reasoning-dense math web).
- **`openthoughts_filtered/`, `openthoughts_flat/`** = OpenThoughts distilled CoT reasoning traces — explicit reasoning chains, directly on-topic.
- `phi_1_5/` synthetic textbook-style.

## Eval sets (multi-step reasoning)
EntailmentBank test (proof/tree reconstruction) · ProofWriter test (depth-graded deductive QA) ·
aNLI `allenai/art` (abductive) · QASC / OpenBookQA-additional (multi-hop QA acc).

## Download cost
Structure shortlist (entailment-bank-v3 + proofwriter + qasc + openbookqa + worldtree) ≈ **<150 MB total** — trivial.
Pretrain-scale: open-web-math already local (free); cosmopedia/proof-pile-2 would need tokenizing.
