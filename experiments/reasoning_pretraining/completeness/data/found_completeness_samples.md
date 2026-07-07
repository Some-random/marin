# Samples: "found completeness" sources (Stage-0 candidates), read 2026-07-04

## openthoughts_flat — 113,957 rows, ~0.97B tokens, avg 33.9k chars/doc
Column: `text`. Clean long CoT reasoning traces — BUT SFT-chat-formatted (fixed preamble every doc):
> "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking
> process before providing the final precise and accurate solutions. This requires engaging in a
> comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and
> iteration to develop well-considered thinking process. Please structure your response into two main
> sections: [Thought] … [Solution] …"

→ This is exactly reasoning-explicit text (the "completeness" property), but in an **assistant/CoT format**,
not naturalistic prose. Distribution caveat: it's out-of-distribution vs DCLM in *style* (chat scaffold),
which confounds "does explicit reasoning transfer" with "does chat-format transfer." Consider stripping the
preamble, or treating format as a controlled factor.

## openwebmath — 110,794 rows in 2 LOCAL shards (~0.19B tok); full HF set is 14.7B (only a slice is local)
Columns: `url, text, date, metadata`. Naturalistic math/reasoning web prose (LaTeX preserved), e.g.:
> "Bayes and his Theorem… The previous discussion started from the result P(B|AC)=K⁻¹P(B|C)P(A|BC)…"
> "Physical Quantity Analogous to Inductance… electric field is analogous to temperature gradient…"

→ Naturalistic (unlike openthoughts) and reasoning-dense, but our LOCAL slice is small (~0.19B). Full
open-web-math (14.7B) would need re-download (27GB) + tokenize if we want a large owm arm.

## Consequence for Stage 0 (no-repeat constraint)
30% of 15.39B = 4.62B needed per single-source arm >> available (openthoughts 0.97B, owm-local 0.19B).
Options: (i) **small-scale ~3B-token gate** (30%×3B ≈ 0.9B ≤ openthoughts, <1 epoch) — cleanest, fast;
(ii) lower mix ratio; (iii) blend sources; (iv) download+tokenize full open-web-math for a large owm arm.
Recommend (i) for the first gate.
