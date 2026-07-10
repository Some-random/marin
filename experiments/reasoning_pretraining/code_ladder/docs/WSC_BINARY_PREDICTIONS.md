# wsc (SuperGLUE BINARY yes/no) — questions, options, and c5v6 vs phi-1.5 predictions (0-shot)

Same schemas as [WSC273_PREDICTIONS.md], but in the **binary** format: a passage with the pronoun, a yes/no question about one specific coreference, and the model scores the label tokens `no` / `yes`. The 'options' are content-free labels — so a model can score ~majority by just leaning on its yes/no prior instead of resolving the pronoun. Gold here is 63% 'no' / 37% 'yes' (majority baseline 63.5%).

Overall on all 104: **phi-1.5 = 60.6%** (picks 'no' 70% — rides the majority), **c5v6 = 40.6%** (collapses to 'yes' 88%). Counts — both✓ 22, phi✓c5v6✗ 41, phi✗c5v6✓ 20, both✗ 21.

---

### Example 1  (doc_id 0)
> Passage: Bernard, who had not told the government official that he was less than 21 when he filed for a homestead claim, did not consider that he had done anything dishonest. Still, anyone who knew that he was 19 years old could take his claim away from *him*.
> Question: In the passage above, does the pronoun "*him*" refer to "*anyone*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (A) no ✓    |    phi-1.5 picked: (A) no ✓

### Example 2  (doc_id 2)
> Passage: I tried to paint a picture of an orchard, with lemons in the lemon trees, but *they* came out looking more like light bulbs.
> Question: In the passage above, does the pronoun "*they*" refer to "*lemon trees*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (A) no ✓    |    phi-1.5 picked: (A) no ✓

### Example 3  (doc_id 13)
> Passage: Larry, a timid teen-ager, lives with his widowed mother in a Brooklyn housing project. Larry Larry's father, a gang leader, was shot to death; his father's disciple, Antonio, takes Larry under *his* wing, and quickly molds him into a drug runner.
> Question: In the passage above, does the pronoun "*his*" refer to "*Antonio*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(B) yes**
- c5v6 picked: (B) yes ✓    |    phi-1.5 picked: (B) yes ✓

### Example 4  (doc_id 14)
> Passage: Alice was dusting the living room and trying to find the button that Mama had hidden. No time today to look at old pictures in her favorite photo album. Today she had to hunt for a button, so she put the album on a chair without even opening *it*.
> Question: In the passage above, does the pronoun "*it*" refer to "*chair*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (A) no ✓    |    phi-1.5 picked: (A) no ✓

### Example 5  (doc_id 3)
> Passage: Always before, Larry had helped Dad with his work. But he could not help him now, for Dad said that *his* boss at the railroad company would not want anyone but him to work in the office.
> Question: In the passage above, does the pronoun "*his*" refer to "*Larry*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (A) no ✓

### Example 6  (doc_id 7)
> Passage: While Nancy and Ellen counted the silverware, Mrs. Smith hastened upstairs. In a few minutes she returned and one look at *her* stricken face told the girls that the precious map was gone.
> Question: In the passage above, does the pronoun "*her*" refer to "*Ellen*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (A) no ✓

### Example 7  (doc_id 8)
> Passage: Meanwhile, in the forest, the elephants are calling and hunting high and low for Arthur and Celeste, and their mothers are very worried. Fortunately, in flying over the town, an old marabou bird has seen *them* and come back quickly to tell the news.
> Question: In the passage above, does the pronoun "*them*" refer to "*their mothers*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (A) no ✓

### Example 8  (doc_id 10)
> Passage: Jane gave Joan candy because *she* was hungry.
> Question: In the passage above, does the pronoun "*she*" refer to "*Jane*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (A) no ✓

### Example 9  (doc_id 12)
> Passage: Alice was dusting the living room and trying to find the button that Mama had hidden. No time today to look at old pictures in *her* favorite photo album. Today she had to hunt for a button, so she put the album on a chair without even opening it.
> Question: In the passage above, does the pronoun "*her*" refer to "*Mama*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (A) no ✓

### Example 10  (doc_id 1)
> Passage: Mr. Moncrieff visited Chester's luxurious New York apartment, thinking that it belonged to his son Edward. The result was that Mr. Moncrieff has decided to cancel Edward's allowance on the ground that he no longer requires *his* financial support.
> Question: In the passage above, does the pronoun "*his*" refer to "*Mr. Moncrieff*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(B) yes**
- c5v6 picked: (B) yes ✓    |    phi-1.5 picked: (A) no ✗

### Example 11  (doc_id 4)
> Passage: Since Chester was dependent on Uncle Vernon, he couldn't very well marry without *his* approval
> Question: In the passage above, does the pronoun "*his*" refer to "*Chester*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (A) no ✓    |    phi-1.5 picked: (B) yes ✗

### Example 12  (doc_id 11)
> Passage: I tried to paint a picture of an orchard, with lemons in the lemon trees, but *they* came out looking more like light bulbs.
> Question: In the passage above, does the pronoun "*they*" refer to "*lemons*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(B) yes**
- c5v6 picked: (B) yes ✓    |    phi-1.5 picked: (A) no ✗

### Example 13  (doc_id 5)
> Passage: The large ball crashed right through the table because *it* was made of styrofoam.
> Question: In the passage above, does the pronoun "*it*" refer to "*the table*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(B) yes**
- c5v6 picked: (A) no ✗    |    phi-1.5 picked: (A) no ✗

### Example 14  (doc_id 6)
> Passage: The path to the lake was blocked, so we couldn't use *it*.
> Question: In the passage above, does the pronoun "*it*" refer to "*The path*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(B) yes**
- c5v6 picked: (A) no ✗    |    phi-1.5 picked: (A) no ✗

### Example 15  (doc_id 9)
> Passage: The customer walked into the bank and stabbed one of the tellers. *He* was immediately taken to the hospital.
> Question: In the passage above, does the pronoun "*He*" refer to "*The customer*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (B) yes ✗

### Example 16  (doc_id 15)
> Passage: Larry, a timid teen-ager, lives with his widowed mother in a Brooklyn housing project. Larry Larry's father, a gang leader, was shot to death; his father's disciple, Antonio, takes Larry under *his* wing, and quickly molds him into a drug runner.
> Question: In the passage above, does the pronoun "*his*" refer to "*Larry*"?

- Options: **(A) no**   **(B) yes**
- Correct: **(A) no**
- c5v6 picked: (B) yes ✗    |    phi-1.5 picked: (B) yes ✗
