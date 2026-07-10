# wsc273 — schemas, options, and c5v6 vs phi-1.5 predictions (0-shot)

The Winograd Schema Challenge in **referent-choice** format: the ambiguous pronoun is replaced by each candidate referent, and the model scores which full sentence is more likely. The differing word (the **referent**) is shown in bold. Chance = 50%.

Overall on all 273: **phi-1.5 = 76.9%**, **c5v6 (1.4B) = 60.1%**.  Counts — both✓ 140, phi✓c5v6✗ 70, phi✗c5v6✓ 24, both✗ 39.

---

### Example 1  (doc_id 1)
- (A) The city councilmen refused the demonstrators a permit because the **city councilmen** advocated violence.
- (B) The city councilmen refused the demonstrators a permit because the **demonstrators** advocated violence.
- **Correct: B**
- c5v6 picked: **B** ✓    |    phi-1.5 picked: **B** ✓

### Example 2  (doc_id 2)
- (A) The trophy doesn't fit into the brown suitcase because the **trophy** is too large.
- (B) The trophy doesn't fit into the brown suitcase because the **suitcase** is too large.
- **Correct: A**
- c5v6 picked: **A** ✓    |    phi-1.5 picked: **A** ✓

### Example 3  (doc_id 4)
- (A) Joan made sure to thank Susan for all the help **Joan** had recieved.
- (B) Joan made sure to thank Susan for all the help **Susan** had recieved.
- **Correct: A**
- c5v6 picked: **A** ✓    |    phi-1.5 picked: **A** ✓

### Example 4  (doc_id 5)
- (A) Joan made sure to thank Susan for all the help **Joan** had given.
- (B) Joan made sure to thank Susan for all the help **Susan** had given.
- **Correct: B**
- c5v6 picked: **B** ✓    |    phi-1.5 picked: **B** ✓

### Example 5  (doc_id 6)
- (A) Paul tried to call George on the phone, but **Paul** wasn't successful.
- (B) Paul tried to call George on the phone, but **George** wasn't successful.
- **Correct: A**
- c5v6 picked: **A** ✓    |    phi-1.5 picked: **A** ✓

### Example 6  (doc_id 3)
- (A) The trophy doesn't fit into the brown suitcase because the **trophy** is too small.
- (B) The trophy doesn't fit into the brown suitcase because the **suitcase** is too small.
- **Correct: B**
- c5v6 picked: **A** ✗    |    phi-1.5 picked: **B** ✓

### Example 7  (doc_id 9)
- (A) The lawyer asked the witness a question, but the **lawyer** was reluctant to answer it.
- (B) The lawyer asked the witness a question, but the **witness** was reluctant to answer it.
- **Correct: B**
- c5v6 picked: **A** ✗    |    phi-1.5 picked: **B** ✓

### Example 8  (doc_id 11)
- (A) The delivery truck zoomed by the school bus because the **delivery truck** was going so slow.
- (B) The delivery truck zoomed by the school bus because the **school bus** was going so slow.
- **Correct: B**
- c5v6 picked: **A** ✗    |    phi-1.5 picked: **B** ✓

### Example 9  (doc_id 14)
- (A) The man couldn't lift his son because the **man** was so weak.
- (B) The man couldn't lift his son because the **son** was so weak.
- **Correct: A**
- c5v6 picked: **B** ✗    |    phi-1.5 picked: **A** ✓

### Example 10  (doc_id 18)
- (A) John couldn't see the stage with Billy in front of him because **John** is so short.
- (B) John couldn't see the stage with Billy in front of him because **Billy** is so short.
- **Correct: A**
- c5v6 picked: **B** ✗    |    phi-1.5 picked: **A** ✓

### Example 11  (doc_id 22)
- (A) Although they ran at about the same speed, Sue beat Sally because **Sue** had such a good start.
- (B) Although they ran at about the same speed, Sue beat Sally because **Sally** had such a good start.
- **Correct: A**
- c5v6 picked: **B** ✗    |    phi-1.5 picked: **A** ✓

### Example 12  (doc_id 26)
- (A) Sam's drawing was hung just above Tina's and **Sam's** drawing did look much better with another one below it.
- (B) Sam's drawing was hung just above Tina's and **Tina's** drawing did look much better with another one below it.
- **Correct: A**
- c5v6 picked: **A** ✓    |    phi-1.5 picked: **B** ✗

### Example 13  (doc_id 35)
- (A) Jim comforted Kevin because **Jim** was so upset.
- (B) Jim comforted Kevin because **Kevin** was so upset.
- **Correct: B**
- c5v6 picked: **B** ✓    |    phi-1.5 picked: **A** ✗

### Example 14  (doc_id 0)
- (A) The city councilmen refused the demonstrators a permit because the **city councilmen** feared violence.
- (B) The city councilmen refused the demonstrators a permit because the **demonstrators** feared violence.
- **Correct: A**
- c5v6 picked: **B** ✗    |    phi-1.5 picked: **B** ✗

### Example 15  (doc_id 13)
- (A) Frank felt crushed when his longtime rival Bill revealed that **Frank** was the winner of the competition.
- (B) Frank felt crushed when his longtime rival Bill revealed that **Bill** was the winner of the competition.
- **Correct: B**
- c5v6 picked: **A** ✗    |    phi-1.5 picked: **A** ✗

### Example 16  (doc_id 17)
- (A) The large ball crashed right through the table because the **large ball** was made of styrofoam.
- (B) The large ball crashed right through the table because the **table** was made of styrofoam.
- **Correct: B**
- c5v6 picked: **A** ✗    |    phi-1.5 picked: **A** ✗
