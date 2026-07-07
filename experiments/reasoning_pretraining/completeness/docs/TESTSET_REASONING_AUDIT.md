# Do our §3 reasoning benchmarks actually NEED a rationale to answer? (raw examples)

## sciq

- **Q:** Compounds that are capable of accepting electrons, such as o 2 or f2, are called what?
  - **passage/support provided:** Oxidants and Reductants Compounds that are capable of accepting electrons, such as O 2 or F2, are calledoxidants (or oxidizing agents) because they can oxidize other compounds. In the process of accepting electrons, an oxidant is [...]
  - **gold:** oxidants
- **Q:** What kind of viscosity is found in long-chain hydrocarbons?
  - **passage/support provided:** There is also a correlation between viscosity and molecular shape. Liquids consisting of long, flexible molecules tend to have higher viscosities than those composed of more spherical or shorter-chain molecules. The longer the [...]
  - **gold:** highly viscous

## piqa

- **Q:** How do I ready a guinea pig cage for it's new occupants?
  - **choices:** ['Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.', 'Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to [...]
  - **gold:** 0
- **Q:** How to make tissue paper window decorations?
  - **choices:** ['Find tissue paper and fold it in half. Take scissors and cut out pieces of the paper in the middle. When you are done tape it to your window.', 'Find tissue paper and fold it in half. Take scissors and tear out pieces of the paper in the middle. When you are done tape it to your window.']
  - **gold:** 0

## boolq

- **Q:** does ethanol take more energy make that produces
  - **passage/support provided:** Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy [...]
  - **gold:** 0
- **Q:** can u drive in canada with us license
  - **passage/support provided:** American entry into Canada by land -- Persons driving into Canada must have their vehicle's registration document and proof of insurance.
  - **gold:** 1

## openbookqa

- **Q:** 
  - **choices:** {'text': ['make more phone calls', 'quit eating lunch out', 'buy less with monopoly money', 'have lunch with friends'], 'label': ['A', 'B', 'C', 'D']}
  - **gold:** B
- **Q:** 
  - **choices:** {'text': ['May', 'July', 'April', 'October'], 'label': ['A', 'B', 'C', 'D']}
  - **gold:** D

## commonsense_qa

- **Q:** A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
  - **choices:** {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['bank', 'library', 'department store', 'mall', 'new york']}
  - **gold:** A
- **Q:** Reading newspaper one of many ways to practice your what?
  - **choices:** {'label': ['A', 'B', 'C', 'D', 'E'], 'text': ['literacy', 'knowing how to read', 'money', 'buying', 'money bank']}
  - **gold:** A

## social_iqa

- **Q:** What does Tracy need to do before this?
  - **gold:** 3
- **Q:** What does Riley need to do before this?
  - **gold:** 3

## logiqa

- **Q:** Based on the above statement, which of the following can be derived?
  - **gold:** a
- **Q:** Which of the following is the conclusion must be assumed
  - **gold:** c

## hellaswag

- **Q:** Roof shingle removal: A man is sitting on a roof. He
  - **choices:** ['is using wrap to wrap a pair of skis.', 'is ripping level tiles off.', "is holding a rubik's cube.", 'starts pulling up roofing on a roof.']
  - **gold:** 3
- **Q:** Sharpening knives: A man is holding a pocket knife while sitting on some rocks in the wilderness. Then he
  - **choices:** ['opens a can of oil put oil on the knife, and puts oil on a knife and press it through a can filled with oil then cuts several pieces from the sandwiches.', 'takes a small stone from the flowing river and smashes it on another stone.', 'uses the knife to shave his leg.', 'sand the rocks and [...]
  - **gold:** 1

## winogrande

- **Q:** Sarah was a much better surgeon than Maria so _ always got the easier cases.
  - **gold:** 2
- **Q:** Natalie has a rich husband and lots of money, Jennifer is poor _ needs to make her clothes.
  - **gold:** 2

## arc_challenge

- **Q:** An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?
  - **choices:** {'text': ['Planetary density will decrease.', 'Planetary years will become longer.', 'Planetary days will become shorter.', 'Planetary gravity will become stronger.'], 'label': ['A', 'B', 'C', 'D']}
  - **gold:** C
- **Q:** Farmers in Wyoming were concerned because some of their chickens were being preyed upon by hawks that lived in areas around their ranches. The farmers grouped together and hunted the hawks until they were no longer in their area. Which would most likely happen next?
  - **choices:** {'text': ['The chicken population would go down.', 'Populations of mice and rats would increase.', 'Another bird of prey would replace the hawk.', 'The chickens would have a lower rate of disease.'], 'label': ['A', 'B', 'C', 'D']}
  - **gold:** B

## arc_easy

- **Q:** Which statement best explains why photosynthesis is the foundation of most food webs?
  - **choices:** {'text': ['Sunlight is the source of energy for nearly all ecosystems.', 'Most ecosystems are found on land instead of in water.', 'Carbon dioxide is more available than other gases.', 'The producers in all ecosystems are plants.'], 'label': ['A', 'B', 'C', 'D']}
  - **gold:** A
- **Q:** Plants use sunlight to make
  - **choices:** {'text': ['soil.', 'minerals.', 'food.', 'water.'], 'label': ['A', 'B', 'C', 'D']}
  - **gold:** C

## mmlu_formal_logic

- **Q:** Identify the conclusion of the following argument. It is hard not to verify in our peers the same weakened intelligence due to emotions that we observe in our everyday patients. The arrogance of our consciousness, which in general, belongs to the strongest defense mechanisms, blocks the [...]
  - **choices:** ['It is hard not to verify in our peers the same weakened intelligence due to emotions that we observe in our everyday patients.', 'The arrogance of our consciousness, which in general, belongs to the strongest defense mechanisms, blocks the unconscious complexes.', 'Because of this, it is [...]
  - **gold:** 3
- **Q:** Which of the following propositions is an immediate (one-step) consequence in PL of the given premises? ~E ⊃ ~F G ⊃ F H ∨ ~E H ⊃ I ~I
  - **choices:** ['E ⊃ F', 'F ⊃ G', 'H ⊃ ~E', '~H']
  - **gold:** 3

## mmlu_prehistory

- **Q:** Unlike most other early civilizations, Minoan culture shows little evidence of:
  - **choices:** ['trade.', 'warfare.', 'the development of a common religion.', 'conspicuous consumption by elites.']
  - **gold:** 3
- **Q:** The secondarily altricial condition of modern human babies may have been an evolutionary solution, in that:
  - **choices:** ['the brain size of hominids had not grown for more than a million years, making it difficult for babies to walk.', 'it allowed for bipedalism and independence from the mother at an earlier age.', 'it allowed for subsequent growth of the brain, since bipedalism had resulted in a narrowed birth [...]
  - **gold:** 2


---

## VERDICT: for most of our NL suite, a rationale is NOT needed to reach the answer

Classified by what actually gets you to the gold answer:

**Lookup / knowledge recall (no reasoning chain):**
- **sciq** — answer is literally in the provided passage ("...are called *oxidants*"). Pure lookup.
- **boolq** — passage reading comprehension, light.
- **mmlu (prehistory + most subjects)** — factual recall ("Minoan culture shows little evidence of: *warfare*").
- **commonsense_qa** — single-fact association ("revolving door + security = *bank*").
- **openbookqa** — single supporting fact / commonsense.

**One-step commonsense / plausibility (a single inference, not a chain):**
- **piqa** (paper vs jeans bedding), **hellaswag** (plausible next action), **social_iqa** (social default),
  **winogrande** (one-step coreference), **arc_challenge/easy** (one causal step: faster rotation → shorter day).

**Genuine multi-step reasoning (rationale actually needed):**
- **logiqa** — spatial/logical deduction from premises (admin SW of cultural, cultural SE of leisure → derive).
- **mmlu_formal_logic** — identify the argument's conclusion (structural reasoning).
- (+ the **Math** group: gsm8k, minerva, gsm_symbolic; and the **Aggregate** group: bbh, gpqa, mmlu_pro.)

## Strategic implication
The **open-book + closed-book NL suite is dominated by knowledge + one-step commonsense**, NOT multi-step
reasoning. So **reasoning-completeness augmentation would not move most of it** — those tasks need the model to
*know a fact* or *match a pattern in one step*, not to chain premises. (And per Petty 2024, structure/code-like
data can even *hurt* knowledge tasks.) The genuine multi-step targets are **logiqa, mmlu_formal_logic, and the
Math/Aggregate groups** — a minority of §3, and exactly where code pretraining already helps. So the honest
question is: **which metric are we trying to move?** If it's the NL suite → reasoning-completeness is the wrong
tool. If it's Math + multi-step logic → right tool, but narrow.

---

## CORRECTION (2026-07-06, after Dongwei pushback) — the earlier verdict was too glib

I under-counted the reasoning. Decomposed properly:
- **arc_challenge** needs 3–5 steps: disambiguate rotation vs orbit → day = one rotation → faster→shorter day
  → REJECT "years longer" (year=orbit) → reject density/gravity. ARC-*Challenge* is by design the subset
  filtered to defeat retrieval solvers. NOT one step.
- **piqa** needs ≥2: locate the single discriminating detail → identify the relevant physical property →
  apply it to the choices. PIQA/HellaSwag/Winogrande are all *adversarially filtered* to kill surface matching.
- **winogrande** ("better surgeon → harder cases → therefore the other got easier"), **hellaswag** (coherent
  next action + reject lures), **social_iqa** (situation → mental state → action) are all 2-step.

**Revised split:**
- Genuinely lookup / no chain: **sciq** (answer in passage), **boolq** (passage RC), **mmlu knowledge subjects**;
  **commonsense_qa** is borderline (association).
- Require 2+ reasoning steps: **arc_easy/challenge, piqa, hellaswag, winogrande, social_iqa, logiqa,
  mmlu_formal_logic**, plus Math + Aggregate groups. → **a MAJORITY of the suite needs reasoning.**

Two caveats: (1) "requires reasoning" ≠ "requires explicit verbalized CoT" — scored by log-likelihood, so the
reasoning can be latent in the weights (one forward pass), which means these benefit from better reasoning
*capability*, not necessarily from emitting chains. (2) Open puzzle: if these need reasoning, why did CODE
(a reasoning booster) barely move the NL suite vs. clearly lifting Math in our ladder? Likely because the NL
reasoning is commonsense a 1.4B already has, or code's procedural/formal reasoning ≠ physical/social commonsense.
