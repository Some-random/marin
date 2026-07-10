# commonsense_qa — questions, options, and c5v6 vs A5 predictions (0/5/10-shot)

5-way multiple choice (A–E); **chance = 20%**. Both models score ~20% at every shot (see the summary). Below, per question, each model's pick at 0/5/10-shot — showing the model firing the same letter regardless of the question. `✓`=correct, `✗`=wrong.

---

### Example 1
**Q:** A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?
  (A) bank  (B) library  (C) department store  (D) mall  (E) new york
**Correct: A**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✓ | A ✓ | A ✓ |
| a5 | A ✓ | A ✓ | A ✓ |

### Example 2
**Q:** What do people aim to do at work?
  (A) complete job  (B) learn from each other  (C) kill animals  (D) wear hats  (E) talk to each other
**Correct: A**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | C ✗ | C ✗ | A ✓ |
| a5 | A ✓ | A ✓ | A ✓ |

### Example 3
**Q:** Where would you find magazines along side many other printed works?
  (A) doctor  (B) bookstore  (C) market  (D) train station  (E) mortuary
**Correct: B**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✗ | A ✗ | A ✗ |
| a5 | C ✗ | A ✗ | A ✗ |

### Example 4
**Q:** Where are  you likely to find a hamburger?
  (A) fast food restaurant  (B) pizza  (C) ground up dead cows  (D) mouth  (E) cow carcus
**Correct: A**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✓ | A ✓ | A ✓ |
| a5 | A ✓ | A ✓ | A ✓ |

### Example 5
**Q:** James was looking for a good place to buy farmland.  Where might he look?
  (A) midwest  (B) countryside  (C) estate  (D) farming areas  (E) illinois
**Correct: A**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | D ✗ | A ✓ | A ✓ |
| a5 | A ✓ | A ✓ | A ✓ |

### Example 6
**Q:** What island country is ferret popular?
  (A) own home  (B) north carolina  (C) great britain  (D) hutch  (E) outdoors
**Correct: C**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | D ✗ | C ✓ | A ✗ |
| a5 | C ✓ | A ✗ | A ✗ |

### Example 7
**Q:** In what Spanish speaking North American country can you get a great cup of coffee?
  (A) mildred's coffee shop  (B) mexico  (C) diner  (D) kitchen  (E) canteen
**Correct: B**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | C ✗ | A ✗ | A ✗ |
| a5 | D ✗ | A ✗ | A ✗ |

### Example 8
**Q:** What do animals do when an enemy is approaching?
  (A) feel pleasure  (B) procreate  (C) pass water  (D) listen to each other  (E) sing
**Correct: D**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✗ | C ✗ | A ✗ |
| a5 | A ✗ | A ✗ | A ✗ |

### Example 9
**Q:** Reading newspaper one of many ways to practice your what?
  (A) literacy  (B) knowing how to read  (C) money  (D) buying  (E) money bank
**Correct: A**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✓ | A ✓ | C ✗ |
| a5 | D ✗ | A ✓ | D ✗ |

### Example 10
**Q:** What do people typically do while playing guitar?
  (A) cry  (B) hear sounds  (C) singing  (D) arthritis  (E) making music
**Correct: C**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✗ | A ✗ | C ✓ |
| a5 | D ✗ | A ✗ | A ✗ |

### Example 11
**Q:** What would vinyl be an odd thing to replace?
  (A) pants  (B) record albums  (C) record store  (D) cheese  (E) wallpaper
**Correct: E**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✗ | C ✗ | C ✗ |
| a5 | D ✗ | A ✗ | A ✗ |

### Example 12
**Q:** If you want harmony, what is something you should try to do with the world?
  (A) take time  (B) make noise  (C) make war  (D) make peace  (E) make haste
**Correct: D**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✗ | A ✗ | C ✗ |
| a5 | C ✗ | A ✗ | A ✗ |

### Example 13
**Q:** Where does a heifer's master live?
  (A) farm house  (B) barnyard  (C) stockyard  (D) slaughter house  (E) eat cake
**Correct: A**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✓ | A ✓ | A ✓ |
| a5 | A ✓ | A ✓ | A ✓ |

### Example 14
**Q:** Aside from water and nourishment what does your dog need?
  (A) bone  (B) charm  (C) petted  (D) lots of attention  (E) walked
**Correct: D**

| model | 0-shot | 5-shot | 10-shot |
|---|---|---|---|
| c5v6 | A ✗ | A ✗ | C ✗ |
| a5 | C ✗ | A ✗ | A ✗ |
