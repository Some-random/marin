# Injecting reasoning into PRETRAINING — recipes, cost, gains, failure modes

Citations = Semantic Scholar 2026-07-04 (some rate-limited → "n/r"). Quotes from WebFetch of arxiv HTML/PDF; re-check before publication.

## Three recipe families
**(a) Objective-level (change loss/architecture):**
- **Quiet-STaR** (Zelikman 2024, 2403.09629, 310c): generate a rationale at EVERY token, learnable `<sot>/<eot>` meta-tokens + mixing head, REINFORCE reward = Δ log-likelihood of observed future tokens. No labels. GSM8K 5.9→10.9, CSQA 36.3→47.2 zero-shot. **Cost lives in the TRAINING loop** (per-token thoughts + RL) → heavy, changes trainer.
- **STaR** (Zelikman 2022, 2203.14465, 962c): ancestor; self-gen rationales, keep answer-correct ones, fine-tune, repeat. Needs verifiable answers.

**(b) Static corpus augmentation — MOST REUSABLE (generation pass + vanilla next-token loss):**
- **BoLT — Reasoning to Learn from Latent Thoughts** (Ruan 2025, 2503.18866, 39c): treat text X as generated from latent thoughts Z; GPT-4o-mini infers "missing reasoning steps or background knowledge", marked `<StartofLatent>/<EndofLatent>`; train both q(Z|X) (Z as suffix) and p(Z,X) (Z as prefix); **EM bootstrap** (E: sample K latents from own posterior, importance-weight w=p(Z,X)/q(Z|X), resample best; M: retrain) → self-improves without more teacher. **"Training with synthetic latent thoughts substantially outperforms all baselines, even outperforming training on an equivalent amount of unique raw tokens."** MATH 25.4% vs 5.7% raw / 19.4% WRAP-CoT. TinyLlama-1.1B, FineMath. **← closest to our idea.**
- **TPT — Thinking Augmented Pre-training** (2025, 2509.20186, n/r): x=[doc; thinking-trajectory], standard loss, no arch change. Teacher = Qwen3-8B / DeepSeek-R1-Distill-7B. Prompt: "Simulate an expert's in-depth thought process… focusing on complex and informative aspects." **~3× data efficiency** (GSM8K 19→50, MATH 9→22 at 100B tok, 8B model). ~20k A100-hrs just to generate.
- **Reasoning CPT — Mining Hidden Thoughts** (Ishibashi 2025, 2505.10182, n/r): frontier LLM (paper indicates **Claude** — verify) reconstructs author's unstated reasoning ("hidden thoughts"), insert between source and target, continual-pretrain Gemma-2-9B on STEM+Law. **Gains grow with difficulty (+8 pts hardest MMLU-Pro), transfer across domains.**

**(c) Synthetic-from-seed (generate docs from topic seeds — DON'T preserve source→conclusion link):**
- phi-1 "Textbooks Are All You Need" (2306.11644, 610c); Cosmopedia (Mixtral, 25B tok, audience×format prompt diversity, 10-gram decontam); MIND (2410.12881, restructure OpenWebMath into dialogues, GSM8K +13.4); WRAP (2401.16380, Mistral-7B paraphrase to 4 styles inc. Q/A, mix 1:1, ~3× speedup) — content-preserving cousin.

## Compute reality (the binding constraint)
Dominant cost = **teacher inference** (generation ∝ corpus size), NOT student training. Teachers used: GPT-4o-mini, Qwen3-8B, DeepSeek-R1-Distill-7B, Mixtral-8x7B, Claude, Mistral-7B. At pretraining scale = tens of thousands of GPU-hours to generate. Students are small (TinyLlama-1.1B, Gemma-2-9B, 128M–1.3B).

## Failure modes
1. **Hallucinated/unfaithful rationales** — teacher thoughts can be wrong; STaR/BoLT filter by answer-correctness/ELBO, but TPT/Reasoning-CPT use unfiltered teacher output.
2. **Teacher-capability ceiling** (distillation not discovery) — gains inherit the teacher; BoLT's EM is the main escape.
3. **Distribution shift** — synthetic drifts off natural web dist (WRAP studies it; Cosmopedia needs 10-gram decontam vs benchmark leakage).
4. **Cost** (above).
5. **Verification gap** — outside math/code (checkable answers), NO cheap correctness signal for reasoning injected into general web text → quality control is THE open problem.

## Bottom line
Three directly reusable blueprints preserve the source-doc link + need only a static generation pass + standard trainer: **BoLT (2503.18866), TPT (2509.20186), Reasoning CPT (2505.10182).** All three = "augment text with the latent reasoning behind it, train normally."
