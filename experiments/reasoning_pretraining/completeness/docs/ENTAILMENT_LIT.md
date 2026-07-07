# Making implicit premises explicit — entailment trees, proof chains, enthymeme reconstruction

Citation counts = Semantic Scholar, 2026-07-04. Quotes from full-text HTML (ar5iv/ACL); re-check load-bearing ones against PDF.

## Reusable assets, clean-synthetic → natural-single-hop spectrum
| Paper | arXiv | cites | What it gives | Stopping rule used |
|---|---|---|---|---|
| **EntailmentBank** (Dalvi 2021) | 2104.08661 | 232 | 1,840 human multi-step entailment trees (5,881 steps) over ARC science; leaves=WorldTree facts, internal=written intermediate conclusions. "T is valid if every node is entailed by its children." Guideline: "include all the knowledge that a young child would need." | **grounding in fixed fact corpus** — bottom out at accepted WorldTree facts |
| **RuleTaker** (Clark 2020) | 2002.05867 | 462 | synthetic NL facts+rules, depth-graded D0–D5, T/F deductive. | **bounded depth** (+ closed-world assumption) |
| **ProofWriter** (Tafjord 2021) | 2012.13048 | 453 | proofs (DAGs) over RuleTaker + **abduction** (find the one missing fact f_m s.t. C∪{f_m}⊢Q). Iterative model forward-chains "until... the implication 'None' is generated." | **fixpoint/closure** (symbolic) ; abduction = "one fact that makes Q provable" |
| **Selection-Inference** (Creswell 2022) | 2205.09712 | 478 | alternates select→infer, appends derived facts. **"we halt after a fixed number of steps... Addressing the issue of halting is left for future work."** | **fixed step budget — halting explicitly UNSOLVED** (the gap) |
| **ARCT** (Habernal 2018) | 1708.01425 | 164 | 1,970 args; pick correct implicit **warrant** (Toulmin). "claims and warrants are usually implicit... 'taken for granted'." | **single-hop by construction** |
| **Implicit Premise Generation** (Chakrabarty 2021) | 2109.05358 | 17 | generate the ONE implicit premise; transfers from ART + injects COMET `xIntent`; prefixes "And since". | **single premise by fiat** |
| **aNLI/ART** (Bhagavatula 2020) | 1908.05739 | 531 | **allenai/art (VERIFIED HF, 171k rows)**: choose/generate best abductive hypothesis h* = argmax P(H|O1,O2). Human 91.4% vs BERT 68.9%. | **single abductive hop** |
| **COMET/ATOMIC** (Bosselut 2019) | 1906.05317 | 1038 | generate commonsense if-then tuples (9 relations). Grounding **atoms** (the "bottom" of a chain). | community treats as **commonsense-atomic floor** (not a COMET claim) |
| Feng & Hunter 2026 (abstract only) | 2603.06114 | — | neuro-symbolic: LLM candidate premises → logic → symbolic entailment check. Implicit stop = "until claim is logically entailed" (unverified mechanism). | **entailment-check** (the principled one) |

## Synthesis (the stopping-rule question)
Prior work operationalizes "complete" 5 ways: (1) bounded depth (RuleTaker), (2) fixpoint/closure
(ProofWriter), (3) grounding in fact/commonsense KB (EntailmentBank+COMET), (4) single-hop by fiat
(ARCT/aNLI/Chakrabarty — sidestep depth), (5) fixed step budget w/ halting unsolved (SI).
**No purely-textual method has a learned, dynamic stopping rule** → confirms the stopping rule is THE
open design problem (see STOPPING_RULES.md).

Best assets: **EntailmentBank** (multi-step chain template), **ProofWriter** (depth-controlled + abduction/
missing-premise), **allenai/art** (largest natural single-hop, verified), **ATOMIC/COMET** (grounding atoms).
