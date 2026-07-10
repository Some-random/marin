# commonsense_qa — letter-scoring vs text-scoring (0-shot)

Same questions, two ways of reading the model's answer. The model never 'writes' an answer; it assigns a **log-likelihood** (higher = more likely) to each candidate continuation, and lm-eval picks the highest.

- **Letter-scored** (lm-eval default, what Marin uses): the candidates are the bare tokens ` A` ` B` ` C` ` D` ` E`. Pick = the letter with the highest log-likelihood. Correct if it equals the gold letter.
- **Text-scored** (`commonsense_qa_text.yaml`, like arc_easy): the candidates are the answer *texts* (` bank` ` library` …) placed after `Question: … Answer:`. Pick = the answer text with the highest log-likelihood (length-normalised for `acc_norm`). Correct if it's the gold answer.

**0-shot accuracy: c5v6 20.1% (letter) → 34.6% (text) ; A5 19.5% → 41.3%.** (chance = 20%.)

---

## How the pick is computed — 3 worked examples (c5v6, actual log-likelihoods; higher = preferred)

### Worked example 1
**Q:** Where would you find magazines along side many other printed works?   —   **correct: B (bookstore)**

_Letter-scored_ — log-likelihood of each **letter token**:
```
  A(doctor): -1.20   B(bookstore): -1.95   C(market): -1.70   D(train station): -1.70   E(mortuary): -3.08
  → highest = 'A'  ✗ (just prefers the letter token, ignores content)
```
_Text-scored_ — log-likelihood of each **answer text**:
```
  A(doctor): -15.06   B(bookstore): -12.19   C(market): -13.31   D(train station): -17.50   E(mortuary): -18.38
  → highest = 'B' (bookstore)  ✓ correct
```

### Worked example 2
**Q:** What island country is ferret popular?   —   **correct: C (great britain)**

_Letter-scored_ — log-likelihood of each **letter token**:
```
  A(own home): -1.80   B(north carolina): -2.06   C(great britain): -1.43   D(hutch): -1.30   E(outdoors): -2.94
  → highest = 'D'  ✗ (just prefers the letter token, ignores content)
```
_Text-scored_ — log-likelihood of each **answer text**:
```
  A(own home): -21.75   B(north carolina): -15.50   C(great britain): -16.12   D(hutch): -18.00   E(outdoors): -17.50
  → highest = 'B' (north carolina)  ✗
```

### Worked example 3
**Q:** In what Spanish speaking North American country can you get a great cup of coffee?   —   **correct: B (mexico)**

_Letter-scored_ — log-likelihood of each **letter token**:
```
  A(mildred's coffee shop): -1.58   B(mexico): -1.83   C(diner): -1.33   D(kitchen): -1.70   E(canteen): -2.58
  → highest = 'C'  ✗ (just prefers the letter token, ignores content)
```
_Text-scored_ — log-likelihood of each **answer text**:
```
  A(mildred's coffee shop): -30.38   B(mexico): -9.62   C(diner): -16.50   D(kitchen): -17.25   E(canteen): -15.94
  → highest = 'B' (mexico)  ✓ correct
```

---

## Pick comparison across more questions (gold≠A shown first)

| # | question | correct | c5v6 letter | c5v6 text | A5 letter | A5 text |
|---|---|---|---|---|---|---|
| 1 | Where would you find magazines along side many other printed… | B | A✗ | B✓ | C✗ | B✓ |
| 2 | What island country is ferret popular?… | C | D✗ | B✗ | C✓ | C✓ |
| 3 | In what Spanish speaking North American country can you get … | B | C✗ | B✓ | D✗ | B✓ |
| 4 | What do animals do when an enemy is approaching?… | D | A✗ | E✗ | A✗ | E✗ |
| 5 | What do people typically do while playing guitar?… | C | A✗ | C✓ | D✗ | C✓ |
| 6 | What would vinyl be an odd thing to replace?… | E | A✗ | E✓ | D✗ | B✗ |
| 7 | If you want harmony, what is something you should try to do … | D | A✗ | D✓ | C✗ | D✓ |
| 8 | Aside from water and nourishment what does your dog need?… | D | A✗ | A✗ | C✗ | A✗ |
| 9 | Janet was watching the film because she liked what?… | C | A✗ | B✗ | C✓ | D✗ |
| 10 | What are you waiting alongside with when you're in a recepti… | D | A✗ | D✓ | C✗ | D✓ |
| 11 | When drinking booze what can you do to stay busy?… | D | A✗ | E✗ | C✗ | E✗ |
| 12 | A fencing thrust with a sharp sword towards a person would r… | E | A✗ | A✗ | B✗ | A✗ |
| 13 | Unlike a spider and his many sight seers, people only have w… | E | A✗ | C✗ | A✗ | E✓ |
| 14 | Where do adults use glue sticks?… | D | A✗ | C✗ | C✗ | D✓ |
| 15 | What could go on top of wood?… | D | A✗ | C✗ | A✗ | C✗ |
| 16 | The artist was sitting quietly pondering, then suddenly he b… | C | A✗ | C✓ | C✓ | C✓ |