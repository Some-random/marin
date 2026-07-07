# §3 test-set inventory — exact sizes (from samples_*.jsonl on disk) + provenance

Counts are **line counts of the actual `samples_*.jsonl`** produced by our eval (source of truth), from
`outputs/eval_results/v2_code25b_clean_p2_15bt_step14671_20260624_0321/` + the aux dirs. n-shot in brackets.

## Open-book (Mean Open-book) — often a supporting passage is provided
| task | # examples | source / what it tests |
|---|---:|---|
| sciq[0] | 1,000 | SciQ (AI2) — science exam Qs **with a support passage** (answer often literally in it) |
| boolq[0] | 3,270 | BoolQ (Google, SuperGLUE) — yes/no over a Wikipedia passage |
| piqa[0] | 1,838 | PIQA (AI2/UW) — physical commonsense, 2-choice (adversarial) |
| openbookqa_fact[0] | 500 | OpenBookQA (AI2) — elementary science + an "open book" of facts |

## Closed-book NL (Mean Closed-book NL)
| task | # examples | source / what it tests |
|---|---:|---|
| arc_easy[25] | 2,376 | ARC-Easy (AI2) — grade-school science MC |
| arc_challenge[25] | 1,172 | ARC-Challenge (AI2) — **hard subset filtered to defeat retrieval solvers** |
| hellaswag[10] | 10,042 | HellaSwag (UW/AI2) — **adversarially-filtered** sentence completion |
| winogrande[5] | 1,267 | WinoGrande (AI2) — **adversarial** Winograd coreference |
| mmlu[5] | 14,042 | MMLU (Hendrycks) — 57-subject knowledge exam (mostly recall) |
| commonsense_qa[0] | 1,221 | CommonsenseQA (Talmor) — ConceptNet-derived commonsense MC |
| social_iqa[0] | 1,954 | Social IQa (AI2) — social commonsense |
| logiqa[0] | 651 | LogiQA — **logical deduction** from civil-service exams |
| lambada_openai[0] | 5,153 | LAMBADA (OpenAI) — last-word prediction over a passage |
| copa[0] | 100 | COPA (SuperGLUE) — causal choice |
| wsc[0] | 104 | Winograd Schema Challenge (SuperGLUE) |
| storycloze_2018_local[0] | 1,571 | Story Cloze / ROCStories 2018 — pick the right ending |
| cb[0] | 56 | CommitmentBank (SuperGLUE) — NLI |
| quac_first_turn[0] | 1,000 | QuAC (AI2) — QA-in-context, F1, first turn |

## Aggregate (Mean Aggregate) — designed-hard / reasoning
| task | # examples | source |
|---|---:|---|
| agieval_lsat_ar[0] | 230 | AGIEval — **LSAT analytical reasoning** |
| gpqa_diamond[0] | 198 | GPQA-Diamond (Rein) — graduate science, "Google-proof" |
| bbh[3] (limit=0.1) | 652 full (~65 scored) | BIG-Bench Hard — 27 reasoning-hard subtasks |
| mmlu_pro[5] (limit=0.1) | 1,209 full (~121 scored) | MMLU-Pro — harder, 10-choice MMLU |

## Math (standard) — genuinely multi-step
| task | # examples | source |
|---|---:|---|
| gsm8k[5] | 2,638 | GSM8K (OpenAI) — grade-school math word problems |
| gsm8k_cot[8] | 2,638 | same problems, CoT prompt |
| minerva_math[4] | 5,000 | MATH (Hendrycks) — 7 competition-math subjects |

## Math (perturbation-robust)
| task | # examples | source |
|---|---:|---|
| gsm_symbolic_main[8] | 10,000 | GSM-Symbolic (Apple) — templated GSM8K perturbations |
| gsm_noop[8] | 234 | GSM-NoOp — GSM8K + irrelevant clauses (distraction robustness) |

## Code
| task | # examples | source |
|---|---:|---|
| humaneval[0] (lm-eval + bigcode) | 164 | HumanEval (OpenAI) — Python function synthesis |
| mbpp[3] | 500 | MBPP (Google) — basic Python problems |

## Perplexity (bits/byte, not accuracy)
| task | # | source |
|---|---:|---|
| dclm_200m_val (bpb) | 5,000 docs | held-out DCLM slice |
| paloma_macro (bpb) | 16 subsets | Paloma (AI2) — perplexity across 16 domains |

## Totals
Roughly **~68,000 example-evaluations** across the accuracy tasks (before de-duping gsm8k/gsm8k_cot and
humaneval lm-eval/bigcode). Everything is **standard public benchmarks**, overwhelmingly from **AI2, OpenAI,
Google, Hendrycks et al** — English, multiple-choice or short-answer, mostly US-centric academic/commonsense.
No proprietary or internal eval data.
