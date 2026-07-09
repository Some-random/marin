# Unreliable evals — sample content + 1.4B model predictions

For every task the systematic audit flagged **not trustworthy at our 300M–1.4B scale** (DEGENERATE / DEAD / SCALE-LIMITED), this shows the *actual* eval content and what a **1.4B** model predicts per example — so the failure mode is visible, not just asserted. MC tasks show the per-choice logprobs the model assigns (the collapse); generation tasks show the raw output.

**Reading note.** The `baseline / our / phi / collapse` line on each task comes from the systematic audit (measured across models). The examples show **c5v6**'s own raw predictions, so a given model can collapse to a *different* constant than the audit's headline (e.g. wsc audit says "always no" but c5v6 leans "yes"; gpqa audit says "(A)" but c5v6's logprobs are just clustered). The invariant is the same either way: **no content discrimination** — near-equal logprobs across choices, or degenerate/looping generations — which is why the score is noise at our scale. Model shown is 1.4B c5v6 (`c5v6_phase2_step14671_hf`) for all tasks.

---

## commonsense_qa  —  **DEGENERATE**
- type: 5-way, LETTER-scored | baseline: chance .200 / maj .209 | our .238  phi-1.5 .507
- failure: collapse .96 — picks the bare letter 'D'/'A' regardless of content
- 1.4B model shown: **c5v6**  (`samples_commonsense_qa_2026-06-13T17-52-09.429947.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...Question: A revolving door is convenient for two direction travel, but it also serves as a security measure at a what? A. bank B. library C. department store D. mall E. new york Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="A"  logprob=-1.180 ←MODEL PICK ✓GOLD
    [1] scored="B"  logprob=-1.805
    [2] scored="C"  logprob=-1.555
    [3] scored="D"  logprob=-1.805
    [4] scored="E"  logprob=-3.172
  → gold=A  model_pick=idx0  CORRECT
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...Question: Reading newspaper one of many ways to practice your what? A. literacy B. knowing how to read C. money D. buying E. money bank Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="A"  logprob=-1.117 ←MODEL PICK ✓GOLD
    [1] scored="B"  logprob=-1.867
    [2] scored="C"  logprob=-1.992
    [3] scored="D"  logprob=-1.742
    [4] scored="E"  logprob=-2.609
  → gold=A  model_pick=idx0  CORRECT
```
**Example 3** (doc_id 16):
```
  Prompt (tail): ...Question: When drinking booze what can you do to stay busy? A. reach tentative agreement B. stay in bed C. stop bicycle D. examine thing E. suicide Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="A"  logprob=-1.180 ←MODEL PICK
    [1] scored="B"  logprob=-1.680
    [2] scored="C"  logprob=-1.805
    [3] scored="D"  logprob=-1.930 ✓GOLD
    [4] scored="E"  logprob=-2.688
  → gold=D  model_pick=idx0  WRONG
```

## gpqa_diamond  —  **DEGENERATE**
- type: 4-way, LETTER-scored (PhD science) | baseline: chance .250 / maj .268 | our .328 (global max)  phi-1.5 .232
- failure: collapse .985 — picks '(A)' 195/198
- 1.4B model shown: **c5v6**  (`samples_gpqa_diamond_zeroshot_2026-06-13T18-18-29.209785.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...What is the correct answer to this question:Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?  Choices: (A) 10^-8 eV (B) 10^-9 eV (C) 10^-11 eV (D) 10^-4 eV Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="(A)"  logprob=-4.125
    [1] scored="(B)"  logprob=-3.750 ←MODEL PICK
    [2] scored="(C)"  logprob=-4.000
    [3] scored="(D)"  logprob=-4.344
  → gold=(D)  model_pick=idx1  WRONG
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...What is the correct answer to this question:which of the following molecules has c3h symmetry? triisopropyl borate quinuclidine benzo[1,2-c:3,4-c':5,6-c'']trifuran-1,3,4,6,7,9-hexaone triphenyleno[1,2-c:5,6-c':9,10-c'']trifuran-1,3,6,8,11,13-hexaone Choices: (A) triisopropyl borate (B) triphenylenotrifuran-1,3,6,8,11,13-hexaone (C) quinuclidine (D) benzotrifuran-1,3,4,6,7,9-hexaone Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="(A)"  logprob=-4.750
    [1] scored="(B)"  logprob=-4.531
    [2] scored="(C)"  logprob=-3.797 ←MODEL PICK
    [3] scored="(D)"  logprob=-4.688
  → gold=(B)  model_pick=idx2  WRONG
```
**Example 3** (doc_id 16):
```
  Prompt (tail): ...What is the correct answer to this question:What is the concentration of calcium ions in a solution containing 0.02 M stochiometric Ca-EDTA complex (we assume that the pH is ideal, T = 25 °C). KCa-EDTA = 5x10^10. Choices: (A) 2.0x10^-2 M (B) 5.0x10^-3 M (C) 1.0x10^-2 M (D) 6.3x10^-7 M Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="(A)"  logprob=-4.312
    [1] scored="(B)"  logprob=-3.969 ←MODEL PICK
    [2] scored="(C)"  logprob=-4.125
    [3] scored="(D)"  logprob=-4.500
  → gold=(D)  model_pick=idx1  WRONG
```

## cb  —  **DEGENERATE**
- type: 3-way NLI | baseline: chance .333 / maj .500 | our .464 (below maj)  phi-1.5 .643
- failure: collapse 1.0 — always predicts entailment; N=56
- 1.4B model shown: **c5v6**  (`samples_cb_2026-06-13T19-45-33.100375.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...Valence the void-brain, Valence the virtuous valet. Why couldn't the figger choose his own portion of titanic anatomy to shaft? Did he think he was helping? Question: Valence was helping. True, False, or Neither? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="True"  logprob=-1.234 ←MODEL PICK
    [1] scored="False"  logprob=-1.609 ✓GOLD
    [2] scored="Neither"  logprob=-3.484
  → gold=1  model_pick=idx0  WRONG
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...…flowing. They say, hey, we got to do something now. And then start playing the game the way the game should be played toward the last few minutes. B: Yeah. A: So, I don't know I'm looking for a good year. I guess we're always looking for a good year. B: So, obviously though, do you think they're going to do anything in the playoffs to make it to the Super Bowl this year Question: they're going to do anything in the playoffs to make it to the Super Bowl this year. True, False, or Neither? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="True"  logprob=-1.898 ←MODEL PICK
    [1] scored="False"  logprob=-2.516
    [2] scored="Neither"  logprob=-3.891 ✓GOLD
  → gold=2  model_pick=idx0  WRONG
```
**Example 3** (doc_id 16):
```
  Prompt (tail): ...B: And other than that I do not think it should be allowable. I think it should be illegal for them to want to do that. it's kind of the big brother syndrome, I mean, I just, anything like that just kind of scares me. A: I tend to view it, even though I don't think I'd work for a company that did that, I sort of want to defend an employer's rights uh, in addition to an individual's rights, Question: she would work for a company that did that. True, False, or Neither? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="True"  logprob=-1.828 ←MODEL PICK
    [1] scored="False"  logprob=-2.328 ✓GOLD
    [2] scored="Neither"  logprob=-4.062
  → gold=1  model_pick=idx0  WRONG
```

## wsc  —  **DEAD**
- type: binary coreference | baseline: maj .635 | our .654 (=class rate, beats phi)  phi-1.5 .606
- failure: collapse .865 — always 'no'; N=104
- 1.4B model shown: **c5v6**  (`samples_wsc_2026-06-13T18-12-02.449598.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...Passage: Bernard, who had not told the government official that he was less than 21 when he filed for a homestead claim, did not consider that he had done anything dishonest. Still, anyone who knew that he was 19 years old could take his claim away from *him*. Question: In the passage above, does the pronoun "*him*" refer to "*anyone*"? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="no"  logprob=-7.500 ✓GOLD
    [1] scored="yes"  logprob=-7.000 ←MODEL PICK
  → gold=0  model_pick=idx1  WRONG
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...Passage: Meanwhile, in the forest, the elephants are calling and hunting high and low for Arthur and Celeste, and their mothers are very worried. Fortunately, in flying over the town, an old marabou bird has seen *them* and come back quickly to tell the news. Question: In the passage above, does the pronoun "*them*" refer to "*their mothers*"? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="no"  logprob=-7.281 ✓GOLD
    [1] scored="yes"  logprob=-6.531 ←MODEL PICK
  → gold=0  model_pick=idx1  WRONG
```
**Example 3** (doc_id 16):
```
  Prompt (tail): ...Passage: Alice was dusting the living room and trying to find the button that Mama had hidden. No time today to look at old pictures in her favorite photo album. Today she had to hunt for a button, so she put the album on a chair without even opening *it*. Question: In the passage above, does the pronoun "*it*" refer to "*living room*"? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="no"  logprob=-6.312 ✓GOLD
    [1] scored="yes"  logprob=-5.938 ←MODEL PICK
  → gold=0  model_pick=idx1  WRONG
```

## agieval_lsat_ar  —  **DEAD**
- type: 5-way LSAT analytical | baseline: chance .200 / maj .230 | our .252 (GLOBAL max)  phi-1.5 .183 (<chance)
- failure: collapse .822 — picks option A; N=230; even phi below chance
- 1.4B model shown: **c5v6**  (`samples_agieval_lsat_ar_2026-06-13T18-12-02.449598.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...…ernoon: Kyle Wed. morning: Nina; Wed. afternoon: Lenore (C)Mon. morning: Lenore; Mon. afternoon: Helen Tues. morning: George; Tues. afternoon: Kyle Wed. morning: Robert; Wed. afternoon: Irving (D)Mon. morning: Nina; Mon. afternoon: Helen Tues. morning: Robert; Tues. afternoon: Irving Wed. morning: Olivia; Wed. afternoon: Lenore (E)Mon. morning: Olivia; Mon. afternoon: Nina Tues. morning: Irving; Tues. afternoon: Helen Wed. morning: Kyle; Wed. afternoon: George A: Among A through E, the answer is
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="(A)Mon. morning: Helen; Mon. afternoon: Robert Tues. morning…"  logprob=-20.625
    [1] scored="(B)Mon. morning: Irving; Mon. afternoon: Olivia Tues. mornin…"  logprob=-27.375
    [2] scored="(C)Mon. morning: Lenore; Mon. afternoon: Helen Tues. morning…"  logprob=-20.250 ←MODEL PICK
    [3] scored="(D)Mon. morning: Nina; Mon. afternoon: Helen Tues. morning: …"  logprob=-29.625
    [4] scored="(E)Mon. morning: Olivia; Mon. afternoon: Nina Tues. morning:…"  logprob=-35.750
  → gold=[2]  model_pick=idx2  WRONG
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...…d. At least as many French novels as Russian novels are selected. If both French plays are selected, then the Russian play is not selected.Q: If the works selected include three French novels, which one of the following could be a complete and accurate list of the remaining works selected? Answer Choices: (A)one Russian novel (B)two French plays (C)one Russian novel, one Russian play (D)one Russian novel, two French plays (E)two Russian novels, one French play A: Among A through E, the answer is
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="(A)one Russian novel"  logprob=-7.969 ←MODEL PICK
    [1] scored="(B)two French plays"  logprob=-10.062
    [2] scored="(C)one Russian novel, one Russian play"  logprob=-8.812
    [3] scored="(D)one Russian novel, two French plays"  logprob=-10.688
    [4] scored="(E)two Russian novels, one French play"  logprob=-10.000
  → gold=[2]  model_pick=idx0  WRONG
```
**Example 3** (doc_id 16):
```
  Prompt (tail): ...…compositions are performed either after F and before R, or after R and before F. O is performed either first or fifth. The eighth composition performed is either L or H. P is performed at some time before S. At least one composition is performed either after O and before S, or after S and before O.Q: If P is performed third and S is performed sixth, the composition performed fifth must be either Answer Choices: (A)F or H (B)F or O (C)F or T (D)H or L (E)O or R A: Among A through E, the answer is
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="(A)F or H"  logprob=-8.312 ←MODEL PICK
    [1] scored="(B)F or O"  logprob=-11.500
    [2] scored="(C)F or T"  logprob=-12.438
    [3] scored="(D)H or L"  logprob=-10.500
    [4] scored="(E)O or R"  logprob=-13.750
  → gold=[2]  model_pick=idx0  WRONG
```

## mmlu  —  **SCALE-LIMITED**
- type: 4-way, LETTER-scored | baseline: chance .250 / maj .271 | our .273  phi-1.5 .437
- failure: at chance for us (phi clears with balanced picks → real, just beyond our scale)
- 1.4B model shown: **c5v6**  (`samples_mmlu_world_religions_2026-06-13T17-50-03.257609.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...… In which dynasty was the "Mandate of Heaven" developed to legitimatize the new rulers? A. Shang B. Zhou C. Han D. Xia Answer: B  Which Japanese government promoted a kind of national cult based on the emperor and his associations with kami? A. Honen B. Tanaka C. Tokugawa D. Meiji Answer: D  How can the Upanishads be characterized? A. Ritual texts B. Philosophical texts C. Hymns D. Origin stories Answer: B  What can murtis be translated as? A. Offerings B. Prayers C. Apparitions D. Idols Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="A"  logprob=-1.133 ←MODEL PICK
    [1] scored="B"  logprob=-1.633
    [2] scored="C"  logprob=-1.508
    [3] scored="D"  logprob=-1.383 ✓GOLD
  → gold=3  model_pick=idx0  WRONG
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...…Shang B. Zhou C. Han D. Xia Answer: B  Which Japanese government promoted a kind of national cult based on the emperor and his associations with kami? A. Honen B. Tanaka C. Tokugawa D. Meiji Answer: D  How can the Upanishads be characterized? A. Ritual texts B. Philosophical texts C. Hymns D. Origin stories Answer: B  What is the name of the most famous dharmashastras, which probably dates from around the first century? A. Laws of Dharma B. Laws of Karma C. Laws of Vishnu D. Laws of Manu Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="A"  logprob=-1.094 ←MODEL PICK
    [1] scored="B"  logprob=-1.344
    [2] scored="C"  logprob=-1.844
    [3] scored="D"  logprob=-1.469 ✓GOLD
  → gold=3  model_pick=idx0  WRONG
```

## boolq  —  **SCALE-LIMITED**
- type: binary over a passage | baseline: maj .622 ('yes') | our .623 (=the yes-prior)  phi-1.5 .746
- failure: sits on the yes-majority; bigger models drop BELOW it
- 1.4B model shown: **c5v6**  (`samples_boolq_2026-06-13T17-52-09.429947.jsonl`)

**Example 1** (doc_id 0):
```
  Prompt (tail): ...…tion of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline. Question: does ethanol take more energy make that produces? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="no"  logprob=-2.938 ✓GOLD
    [1] scored="yes"  logprob=-2.562 ←MODEL PICK
  → gold=0  model_pick=idx1  WRONG
```
**Example 2** (doc_id 8):
```
  Prompt (tail): ...American entry into Canada by land -- Persons driving into Canada must have their vehicle's registration document and proof of insurance. Question: can u drive in canada with us license? Answer:
  Choices [what the model actually scores → the continuation], with its logprob:
    [0] scored="no"  logprob=-2.859
    [1] scored="yes"  logprob=-1.484 ←MODEL PICK ✓GOLD
  → gold=1  model_pick=idx1  CORRECT
```

## gsm8k  —  **SCALE-LIMITED**
- type: free-generation exact-match | baseline: ~0 | our .020  phi-1.5 .305
- failure: ~0 correct final answers
- 1.4B model shown: **c5v6**  (`samples_gsm8k_2026-06-13T17-47-43.512753.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...…. Therefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards. #### 84  Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Answer:
  Gold answer: Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18
  Model raw generation: "Janet’s ducks lay 16 eggs per day, so she makes $16 per day. She sells the remaining $2 per fresh duck egg for $2.50. #### 2"
  → extracted=2  correct=0.0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…r 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours? Answer:
  Gold answer: When he turned around he was 3*60=<<3*60=180>>180 miles from home
He was only able to drive 4-2=<<4-2=2>>2 hours in the first four hours
In half an hour he goes 30*.5=<<30*.5=15>>15 miles
He then drives another 2-.5=<<2-.5=1.5>>1.5 hours
In that time he goes 80*1.5=<<80*1.5=120>>120 miles
So he drove 120+15=<<120+15=135>>135 miles
So he is 180-135=<<180-135=45>>45 miles away from home
#### 45
  Model raw generation: "John spent 3 hours at 60 mph, 2 hours at 30 mph, and 1 hour at 80 mph. He spent 4 hours at 60 mph, 2 hours at 30 mph, and 1 hour at 80 mph. #### 2"
  → extracted=2  correct=0.0
```

## gsm8k_cot  —  **SCALE-LIMITED**
- type: CoT free-generation | baseline: ~0 | our .028  phi-1.5 .272
- failure: ~0
- 1.4B model shown: **c5v6**  (`samples_gsm8k_cot_2026-06-13T17-56-13.061300.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...…s for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.  Q: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? A:
  Gold answer: 18
  Model raw generation: "Janet started with 16 eggs. After she eats three for breakfast, she has 16 - 3 = 11 eggs. After she eats three for breakfast, she has 11 - 3 = 10 eggs. After she eats three for breakfast, she has 10 - 3 = 9 eggs. After she eats three for breakfast, she has 9 - 3 = 8 eggs. The answer is 8."
  → extracted=8  correct=0.0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…es for 3 hours at a speed of 60 mph and then turns around because he realizes he forgot something very important at home.  He tries to get home in 4 hours but spends the first 2 hours in standstill traffic.  He spends the next half-hour driving at a speed of 30mph, before being able to drive the remaining time of the 4 hours going at 80 mph.  How far is he from home at the end of those 4 hours? A:
  Gold answer: 45
  Model raw generation: "John started with 3 hours. After he got home, he spent 2 hours driving at 30mph. Then he spent the next 2 hours driving at 80mph. So he spent 4 hours driving at 80mph. The answer is 4."
  → extracted=4  correct=0.0
```

## gsm_symbolic_main  —  **SCALE-LIMITED**
- type: perturbed CoT math | baseline: ~0 | our .010  phi-1.5 .160
- failure: ~0
- 1.4B model shown: **c5v6**  (`samples_gsm_symbolic_main_2026-06-13T17-44-51.747451.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...…is 33.  Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?  A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.  Q: Rania saw a 210-foot whale with 7 72-inch remoras attached to it. What percentage of the whale's body length is the combined length of the remoras?  A:
  Gold answer: 20
  Model raw generation: "Rania started with 7 72-inch remoras. After seeing the whale, she had 7 - 72 = 7. The answer is 7."
  → extracted=7  correct=0.0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…33.  Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?  A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.  Q: Emma saw a 360-foot dolphin with 6 72-inch remoras attached to it. What percentage of the dolphin's body length is the combined length of the remoras?  A:
  Gold answer: 10
  Model raw generation: "Emma started with 6 72-inch remoras. After seeing the dolphin, she had 6 - 72 = 6. The answer is 6."
  → extracted=6  correct=0.0
```

## minerva_math  —  **DEAD**
- type: competition math | baseline: ~0 | our .016  phi-1.5 .031
- failure: ~0 even for phi-1.5
- 1.4B model shown: **c5v6**  (`samples_minerva_math_precalc_2026-06-13T18-10-32.248249.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...…frac{a}{b}=\boxed{-\frac{2}{3}}.$$ Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct.  Problem: Let $\mathbf{a}$ and $\mathbf{b}$ be vectors such that \[\mathbf{v} = \operatorname{proj}_{\mathbf{a}} \mathbf{v} + \operatorname{proj}_{\mathbf{b}} \mathbf{v}\]for all vectors $\mathbf{v}.$  Enter all possible values of $\mathbf{a} \cdot \mathbf{b},$ separated by commas.  Solution:
  Gold answer: 0
  Model raw generation: "We have that $\mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathb…"
  → extracted= We have that $\mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b} = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{b  correct=0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…rect.  Problem: A plane $P$ is parmeterized by \[\mathbf{v} = \begin{pmatrix} 1 \\ 6 \\ 7 \end{pmatrix} + t \begin{pmatrix} 2 \\ -1 \\ -1 \end{pmatrix} + s \begin{pmatrix} 2 \\ -3 \\ -5 \end{pmatrix},\]and line $L$ is parameterized by \[\mathbf{w} = \begin{pmatrix} 7 \\ 4 \\ 1 \end{pmatrix} + u \begin{pmatrix} 3 \\ 0 \\ -1 \end{pmatrix}.\]Find the intersection of plane $P$ and line $L.$  Solution:
  Gold answer: \begin{pmatrix}1\4\3\end{pmatrix}
  Model raw generation: "We have that $P$ is parmeterized by $P = \begin{pmatrix} 1 \\ 6 \\ 7 \end{pmatrix} + t \begin{pmatrix} 2 \\ -1 \\ -1 \end{pmatrix} + s \begin{pmatrix} 2 \\ -3 \\ -5 \end{pmatrix} + u \begin{pmatrix} 3 \\ 0 \\ -1 \end{pmatrix} = \boxed{[1,6,7,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2"
  → extracted= We have that $P$ is parmeterized by $P = \begin{pmatrix} 1 \\ 6 \\ 7 \end{pmatrix} + t \begin{pmatrix} 2 \\ -1 \\ -1 \end{pmatrix} + s \begin{pmatrix} 2 \\ -3 \\ -5 \end{pmatrix} + u \begin{pmatrix} 3 \\ 0 \\ -1 \end{pmatrix} = \boxed{[1,6,7,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2  correct=0
```
**Example 3** (doc_id 16):
```
  Question (tail): ...…the first equation by $-\frac{3}{2}$, we obtain  $$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have  $$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$ Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct.  Problem: Let $\mathbf{R}$ be the matrix for reflecting over the vector $\begin{pmatrix} 1 \\ 5 \end{pmatrix}.$  Find $\det \mathbf{R}.$  Solution:
  Gold answer: -1
  Model raw generation: "The determinant of the matrix $\mathbf{R}$ is $\det \mathbf{R} = \det \begin{pmatrix} 1 & 5 \end{pmatrix} = \det \begin{pmatrix} 1 & 5 \end{pmatrix} = 5.$ Final Answer: The final answer is $5$. I hope it is correct."
  → extracted= The determinant of the matrix $\mathbf{R}$ is $\det \mathbf{R} = \det \begin{pmatrix} 1 & 5 \end{pmatrix} = \det \begin{pmatrix} 1 & 5 \end{pmatrix} = 5.$
Final Answer: The final answer is $5$. I hope it is correct.

  correct=0
```

## gsm_noop  —  **DEAD**
- type: math + irrelevant distractor | baseline: ~0 | our .017  phi-1.5 .034
- failure: ~0 even for phi-1.5; N=117
- 1.4B model shown: **c5v6**  (`samples_gsm_noop_2026-06-13T17-46-25.589761.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...… will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.  Q: Andre works a 12-hour shift each day, 6 days a week. He earns €13 per hour, having received a raise from €11 per hour back in September. He also gets a €132 bonus each week if the company performs well. How much money did Andre make in October if the company performed very well for the whole month?  A:
  Gold answer: 4272
  Model raw generation: "Andre started with €13 per hour. After the month, he earned €132. So he earned €132 + €13 = €133. The answer is €133."
  → extracted=[invalid]  correct=0.0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…hopping for some athletic gear. She bought a t-shirt for $24, a pair of swimming shorts for $10 more than the t-shirt cost, and a pair of sneakers that were originally $50 but were on sale for 1/5 price. She had a coupon for $4 off the package of $6 athletic socks that she also bought. She also considered buying a $15 swim cap but decided not to get it. How much did she spend on athletic gear?  A:
  Gold answer: 70
  Model raw generation: "Keisha started with $24. She bought a t-shirt for $24. She bought a pair of swimming shorts for $10 more than the t-shirt cost. She also bought a pair of sneakers for $10 more than the t-shirt cost. She also bought a $15 swim cap for $4 off the package of $6 athletic socks that she also bought. She also considered buying a $15 swim cap but decided not to get it. The answer is $15."
  → extracted=[invalid]  correct=0.0
```
**Example 3** (doc_id 16):
```
  Question (tail): ...…es she have left?  A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.  Q: Roya scored 324 points in one game of basketball. Mei scored 14 points more than one-half as many as Roya, though 18 of Mei's points came from free throws rather than field goals. How many points did Roya and Mei have in total?  A:
  Gold answer: 500
  Model raw generation: "Roya started with 324 points. After scoring 14 points, she had 324 - 14 = 32 points. 18 of her points came from free throws rather than field goals. So she had 32 - 18 = 20 points. The answer is 20."
  → extracted=20  correct=0.0
```

## mmlu_pro  —  **DEAD**
- type: 10-way generative-extract | baseline: chance .100 / maj .130 | our .116  phi-1.5 n/a
- failure: extraction-emit floor
- 1.4B model shown: **c5v6**  (`samples_mmlu_pro_psychology_2026-06-13T18-36-15.613131.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...…. F. is functioning at an intellectual level that is superior for his/her age, education, and demographic background. G. is nonresponsive to visual or auditory stimuli and seems to be in a state of deep sleep. H. is alert and oriented and can remember and integrate remote and recent events but may have some impairment in judgment, planning, and abstract reasoning. Answer: Let's think step by step.
  Gold answer: E
  Model raw generation: "We refer to Wikipedia articles on psychology for help. The US scores highest on the cognitive function scale. The answer is (A)."
  → extracted=A  correct=0.0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…ogist feels char a client would be denied services if barbering were not allowed H. allowable only if the psychologist initiates the bartering proposal I. allowable only if the dollar value of the psychological service and of the payment service or goods are established in advance J. allowable only if it is established in a written agreement signed by both parties Answer: Let's think step by step.
  Gold answer: F
  Model raw generation: "We refer to Wikipedia articles on psychology for help. Bartering is a form of payment for psychological services. The answer is (A)."
  → extracted=A  correct=0.0
```
**Example 3** (doc_id 16):
```
  Question (tail): ...…uld seek her own professional help. G. "unacceptable because doing so creates a ""conﬂict of interest.”" H. acceptable only if the wife gives her written consent to participate in the therapy. I. unacceptable because it may create a bias in the therapist's approach towards the client. J. unacceptable because it infringes on the client's individual therapy process. Answer: Let's think step by step.
  Gold answer: E
  Model raw generation: "We refer to Wikipedia articles on psychology for help. Dr. Dinwoody invites the client's wife to participate in couple therapy. The client's wife is not allowed to participate in the therapy. The answer is (B)."
  → extracted=B  correct=0.0
```

## bbh  —  **DEAD**
- type: multi-subtask gen aggregate | baseline: ~.26 chance | our .235 (BELOW chance)  phi-1.5 .288
- failure: score is parseable-emit rate, below the chance floor
- 1.4B model shown: **c5v6**  (`samples_bbh_cot_fewshot_word_sorting_2026-06-13T18-23-22.754404.jsonl`)

**Example 1** (doc_id 0):
```
  Question (tail): ...…y looking at their second letters. The second letter: "arson": "r" (18). "aspect": "s" (19). We now have: (18) "arson" < (19) "aspect". Hence, we have ["arson" < "aspect"] < "census" < "mugho" < "newt" < "parthia" < "seismography". So the answer is arson aspect census mugho newt parthia seismography.  Q: Sort the following words alphabetically: List: syndrome therefrom A: Let's think step by step.
  Gold answer: syndrome therefrom
  Model raw generation: "The first letter: "syndrome": "s" (19). "therefrom": "t" (3). "therefore": "t" (3). We now have: (3) ["syndrome"? "therefrom"] < (19) "therefore". Hence, we have ["therefrom" < "syndrome"] < "therefore". So the answer is therefrom syndrome syndrome syndrome therefrom."
  → extracted=therefrom syndrome syndrome syndrome therefrom  correct=0.0
```
**Example 2** (doc_id 8):
```
  Question (tail): ...…t their second letters. The second letter: "arson": "r" (18). "aspect": "s" (19). We now have: (18) "arson" < (19) "aspect". Hence, we have ["arson" < "aspect"] < "census" < "mugho" < "newt" < "parthia" < "seismography". So the answer is arson aspect census mugho newt parthia seismography.  Q: Sort the following words alphabetically: List: thunderclap swab built poland A: Let's think step by step.
  Gold answer: built poland swab thunderclap
  Model raw generation: "The first letter: "swab": "s" (19). "built": "b" (1). "poland": "p" (16). "swab": "s" (19). "built": "b" (1). We now have: (1) ["swab"? "built"] < (19) "poland". Now let's sort this subpart ["swab"? "built"] by looking at their second letters. The second letter: "swab": "s" (19). "built": "b" (1). "poland": "p" (16). "swab": "s" (19). "built": "b" (1). We now have: (1) ["swab" < "built"] < (19) "poland". Hence, we have ["swab" < "built"] < "swab". So the answer is swab built poland thunderclap."
  → extracted=swab built poland thunderclap  correct=0.0
```
**Example 3** (doc_id 16):
```
  Question (tail): ...…, we have ["arson" < "aspect"] < "census" < "mugho" < "newt" < "parthia" < "seismography". So the answer is arson aspect census mugho newt parthia seismography.  Q: Sort the following words alphabetically: List: marlborough pyrotechnic filly grocer treadle transitive platelet deliver landau hotbox uncle siemens anger hessian gneiss convoy ninebark advent plat stapleton A: Let's think step by step.
  Gold answer: advent anger convoy deliver filly gneiss grocer hessian hotbox landau marlborough ninebark plat platelet pyrotechnic siemens stapleton transitive treadle uncle
  Model raw generation: "The first letter: "marlborough": "m" (8). "pyrotechnic": "p" (16). "filly": "f" (16). "grocer": "g" (16). "treadle": "t" (16). "transitive": "t" (16). "platelet": "p" (16). "deliver": "d" (3). We now have: (3) ["pyrotechnic"? "filly"] < (8) ["grocer"? "treadle"] < (16) ["platelet"? "deliver"] < (16) ["transitive"? "platelet"] < (16) ["platelet"? "deliver"] < (16) ["platelet"? "platelet"] < (16) ["platelet"? "platelet"] < (16) ["platelet"? "platelet"] < (16) ["platelet"? "platelet"] < (16) ["plat…"
  → extracted=[invalid]  correct=0.0
```
