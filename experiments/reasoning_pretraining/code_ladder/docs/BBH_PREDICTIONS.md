# bbh — few-shot prompt, and c5v6 vs phi-1.5 predictions (3-shot CoT)

bbh is **free-generation**: the model is given 3 worked chain-of-thought examples, then a new question; it generates its own CoT ending in "So the answer is (X)" and the letter is extracted and matched to gold.

**Honest headline (correcting an earlier cherry-picked version of this doc):** on bbh, *both* models mostly **collapse to a constant answer letter** (or emit an unparseable `[invalid]`), not reason. phi-1.5 shows genuine balanced above-chance signal on essentially one of the subtasks below (logical_deduction); elsewhere its 36–40% is a collapsed constant that happens to match the gold rate. So bbh's overall ~0.24 (c5v6) / ~0.29 (phi) is a collapse-and-emit artifact for **both** models.

## Per-subtask accuracy + collapse (n=25 each, `limit=0.1`)

| subtask | chance | c5v6 acc | c5v6 top-pick | phi-1.5 acc | phi-1.5 top-pick |
|---|---|---|---|---|---|
| date_understanding | 17% | 16% | `(C)` 80% | 12% | `(A)` 76% |
| disambiguation_qa | 33% | 32% | `(B)` 100% | 40% | `(C)` 100% |
| logical_deduction_three_objects | 33% | 32% | `(A)` 68% | 52% | `(A)` 52% |
| reasoning_about_colored_objects | 6% | 4% | `[invalid]` 44% | 16% | `(B)` 44% |
| hyperbaton | 50% | 0% | `[invalid]` 92% | 36% | `(B)` 100% |
| navigate | 50% | 52% | `Yes` 56% | 52% | `Yes` 56% |

## The few-shot prompt bbh prepends (example: `date_understanding`)
Each subtask has its own fixed 3-shot CoT preamble in this shape:

```
Infer the date from context.

Q: Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
Options:
(A) 12/14/2026
(B) 12/14/1950
(C) 12/14/2007
(D) 12/14/1937
(E) 07/14/1938
(F) 12/14/1988
A: Let's think step by step.
If today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937. So the answer is (D).

Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
Options:
(A) 09/04/2018
(B) 11/11/2018
(C) 08/25/2018
(D) 11/02/2018
(E) 11/04/2018
A: Let's think step by step.
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is 11/11/2018. So the answer is (B).

Q: Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?
Options:
(A) 01/11/1961
(B) 01/03/1963
(C) 01/18/1961
(D) 10/14/1960
(E) 01/03/1982
(F) 12/03/1960
A: Let's think step by step.
If Jane and John married on Jan 2, 1958, then and if it is their 5-year anniversary today, then today's date is Jan 2, 1963. The date tomorrow is Jan 3, 1963, that is 01/03/1963. So the answer is (B).
```

## Subtask: date_understanding  (chance 17%) — first 3 items, not cherry-picked

### date_understanding — Example 1
**Problem:**
> Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?

> Options:
(A) 12/11/1937
(B) 12/25/1937
(C) 01/04/1938
(D) 12/04/1937
(E) 12/25/2006
(F) 07/25/1937

**Gold: (B)**

- **c5v6** ✗ → `(C)` — "If today is Christmas Eve of 1937, then today is 12/11/1937. The date tomorrow is 12/04/1937. So the answer is (C)."
- **phi-1.5** ✗ → `(A)` — "If today is Christmas Eve of 1937, then today's date is December 24, 1937. The date tomorrow is December 25, 1937, that is 12/25/1937. So the answer is (A)."

### date_understanding — Example 2
**Problem:**
> Jane thinks today is 6/18/2019, but John thinks today is 6/19/2019. Jane is correct. What is the date yesterday in MM/DD/YYYY?

> Options:
(A) 06/17/2063
(B) 05/18/2019
(C) 05/20/2019
(D) 06/17/2019
(E) 05/13/2019
(F) 06/08/2019

**Gold: (D)**

- **c5v6** ✗ → `(C)` — "If Jane and John think today is 6/18/2019, then today is 06/17/2019. The date yesterday is 05/18/2019. So the answer is (C)."
- **phi-1.5** ✗ → `(A)` — "If Jane thinks today is 6/18/2019, then today's date is 6/18/2019. The date yesterday is 6/17/2019, that is 05/17/2019. So the answer is (A)."

### date_understanding — Example 3
**Problem:**
> May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date today in MM/DD/YYYY?

> Options:
(A) 05/06/2095
(B) 05/06/1935
(C) 05/06/1912
(D) 08/06/2002
(E) 05/06/2002
(F) 05/06/1964

**Gold: (E)**

- **c5v6** ✗ → `(C)` — "If May 6, 1992 is like yesterday to Jane, then and if it is their 5-year anniversary today, then today's date is May 6, 1992, that is 05/06/2095. The date tomorrow is May 6, 1992, that is 05/06/1935. So the answer is (C)."
- **phi-1.5** ✗ → `(A)` — "If May 6, 1992 is like yesterday to Jane, then today is May 6, 1992. The date ten years ago is May 6, 1992 - 10 = May 4, 1992. So the answer is (A)."

## Subtask: disambiguation_qa  (chance 33%) — first 3 items, not cherry-picked

### disambiguation_qa — Example 1
**Problem:**
> In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The patient was referred to the specialist because he had a rare skin condition.

> Options:
(A) The patient had a skin condition
(B) The specialist had a skin condition
(C) Ambiguous

**Gold: (A)**

- **c5v6** ✗ → `(B)` — "Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the patient and the specialist. The verb phrase "refer" might be able to help us determine which one is more likely (if either). Let X be the patient and Y the specialist. The sentence is then of the form "X referred to  …"
- **phi-1.5** ✗ → `(C)` — "Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the patient and the specialist. The verb phrase "was referred to" might be able to help us determine which one is more likely (if either). Let X be the patient and Y the specialist. The sentence is then of the form "X wa …"

### disambiguation_qa — Example 2
**Problem:**
> In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The scientist collaborates with the artist, and he shares a story.

> Options:
(A) The scientist shares a story
(B) The artist shares a story
(C) Ambiguous

**Gold: (C)**

- **c5v6** ✗ → `(B)` — "Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the scientist and the artist. The verb phrase "collaborate" might be able to help us determine which one is more likely (if either). Let X be the scientist and Y the artist. The sentence is then of the form "X collaborat …"
- **phi-1.5** ✓ → `(C)` — "Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the scientist and the artist. The verb phrase "collaborates" might be able to help us determine which one is more likely (if either). Let X be the scientist and Y the artist. The sentence is then of the form "X collabora …"

### disambiguation_qa — Example 3
**Problem:**
> In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The developer was unable to communicate with the writer because he focuses on code.

> Options:
(A) The developer focuses on code
(B) The writer focuses on code
(C) Ambiguous

**Gold: (A)**

- **c5v6** ✗ → `(B)` — "Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the developer and the writer. The verb phrase "focus on code" might be able to help us determine which one is more likely (if either). Let X be the developer and Y the writer. The sentence is then of the form "X focuses  …"
- **phi-1.5** ✗ → `(C)` — "Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the developer and the writer. The verb phrase "was unable to communicate" might be able to help us determine which one is more likely (if either). Let X be the developer and Y the writer. The sentence is then of the form …"

## Subtask: logical_deduction_three_objects  (chance 33%) — first 3 items, not cherry-picked

### logical_deduction_three_objects — Example 1
**Problem:**
> The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a branch, there are three birds: a blue jay, a quail, and a falcon. The falcon is to the right of the blue jay. The blue jay is to the right of the quail.

> Options:
(A) The blue jay is the second from the left
(B) The quail is the second from the left
(C) The falcon is the second from the left

**Gold: (A)**

- **c5v6** ✓ → `(A)` — "(1) The blue jay is to the right of the quail: "(left)? blue? quail? (right)". (2) The blue jay is to the right of the falcon: "(left)? blue? quail? (right)". (3) Combining (1) and (2) we get the following ordering: "(left) blue quail (right)". According to this ordering, the blue jay is the second from the left. The blue jay is the secon …"
- **phi-1.5** ✓ → `(A)` — "(1) The blue jay is to the right of the quail: "(right)? quail? blue jay? (left)". (2) The falcon is to the right of the blue jay: "(right)? blue jay? falcon? (left)". (3) Combining (1) and (2) we get the following ordering: "(right) quail blue jay falcon (left)". According to this ordering, the second from the left is the blue jay. The b …"

### logical_deduction_three_objects — Example 2
**Problem:**
> The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. On a shelf, there are three books: an orange book, a yellow book, and a blue book. The blue book is to the right of the yellow book. The orange book is the second from the left.

> Options:
(A) The orange book is the second from the left
(B) The yellow book is the second from the left
(C) The blue book is the second from the left

**Gold: (A)**

- **c5v6** ✓ → `(A)` — "(1) The orange book is the second from the left: "(left)? orange? yellow? (right)". (2) The yellow book is the second from the left: "(left)? orange yellow? (right)". (3) Combining (1) and (2) we get the following ordering: "(left) orange yellow (right)". According to this ordering, the leftmost book is the orange book. The orange book is …"
- **phi-1.5** ✗ → `(B)` — "(1) The blue book is to the right of the yellow book: "(right)? yellow? blue? (left)". (2) The orange book is the second from the left: "(right)? yellow orange blue? (left)". (3) Combining (1) and (2) we get the following ordering: "(right) yellow orange blue (left)". According to this ordering, the second from the left is the yellow book …"

### logical_deduction_three_objects — Example 3
**Problem:**
> The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. A fruit stand sells three fruits: mangoes, watermelons, and kiwis. The watermelons are less expensive than the kiwis. The kiwis are the second-most expensive.

> Options:
(A) The mangoes are the most expensive
(B) The watermelons are the most expensive
(C) The kiwis are the most expensive

**Gold: (A)**

- **c5v6** ✓ → `(A)` — "(1) The mangoes are the most expensive: "(left)? mango? watermelons? (right)". (2) The watermelons are the second-most expensive: "(left)? mangoes? kiwis? (right)". (3) Combining (1) and (2) we get the following ordering: "(left) mangoes kiwis (right)". According to this ordering, the mangoes are the most expensive. The mangoes are the mo …"
- **phi-1.5** ✓ → `(A)` — "(1) The kiwis are the second-most expensive: "(second-most)? kiwis? mangoes? (most)". (2) The watermelons are less expensive than the kiwis: "(most)? kiwis? mangoes? (less)". (3) Combining (1) and (2) we get the following ordering: "(most) kiwis mangoes (less)". According to this ordering, the most expensive fruit is the mangoes. The mang …"
