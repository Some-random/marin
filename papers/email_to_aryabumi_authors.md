**To:** viraat@cohere.com, sara@cohere.com (or appropriate contact at Cohere For AI)
**Subject:** Quick question on §3.1 LR schedule — "To Code, or Not To Code?"

Dear Viraat and co-authors,

I'm replicating the two-stage code → text recipe from §3.1 of *To Code, or Not To Code?* at 1.4B, and I have one question I can't pin down from the paper:

Across the stage 1 → stage 2 boundary, did you use **a single continuous cosine LR schedule** (one cosine over the full combined budget — stage 2 inherits the bottom half), or **two separate cosines** (stage 2 fresh-warms to a peak then cosines to 0 over its own budget)?

And is there a public code/config release I can look at for this recipe?

Thanks very much,
Dongwei Jiang
