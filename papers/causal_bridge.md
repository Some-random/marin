# Causal Bridge: Paper Notes

## Related Work

### [Synthetic Continued Pretraining (EntiGraph)](https://arxiv.org/abs/2409.07431) (Yang et al., Stanford, 2024)

**Motivation:** Knowledge acquisition through next-token prediction is data-inefficient -- models require hundreds to thousands of diverse representations of a fact to learn it. This poses a challenge for continued pretraining on small niche corpora where each fact may appear only once. Simply paraphrasing does not provide sufficient diversity.

**Experiment Setup:** EntiGraph extracts entities from 265 books in the QuALITY reading comprehension dataset (1.3M source tokens), forms a knowledge graph, and uses GPT-4-turbo to generate relation descriptions between entity pairs and triplets, producing a 455M token synthetic corpus (~350x expansion). Llama 3 8B Base is continually pretrained for 2 epochs with RedPajama replay and evaluated on 4,609 QuALITY multiple-choice questions in closed-book 5-shot CoT. Baselines include Raw CPT and Rephrase CPT.

**Conclusion:** EntiGraph CPT improves closed-book QA from 39.5% to 56.2%, with log-linear scaling in synthetic token count up to 455M. Achieves 80% of RAG's improvement (RAG: 60.4%). Raw CPT actually hurts performance, Rephrase CPT scales poorly. EntiGraph + RAG yields 62.6%, showing parametric and retrieval-based knowledge are complementary. The combinatorial relational structure between entities is what drives the diversity and quality of the synthetic corpus.
