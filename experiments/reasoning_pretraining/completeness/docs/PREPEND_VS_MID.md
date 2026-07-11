# Prepend vs. mid insertion — what the model actually sees

All these tests measure one thing: the **NLL (perplexity) of a fixed TARGET span**, teacher-forced, given
some prefix. "Does the rationale help" = does adding it to the prefix lower the target's NLL. The prefix can
be assembled two ways, and **where the rationale goes changes the result a lot** — independently of the
rationale's content. This doc shows the exact strings.

The `+ "\n" +` below is a literal newline join; the **scored span is always the TARGET** (bolded).

---

## DCLM — the target is a *continuation* of prior context

Real forum-post doc (id 102770). Context flows naturally into the target; the rationale explains the jump.

### base — no rationale (`context + target`)
```
…some of the links here are hard coded as .com rather than using
/whatever/wherever.php or ../here/there.php site relative links.
```
> **So pretty soon they end up back on the office server version of the site.**   ← scored

### mid — rationale BETWEEN context and target (`context + rationale + target`)
```
…some of the links here are hard coded as .com rather than …site relative links.
The site's internal navigation links are hard-coded as absolute .com URLs rather than
relative paths, and inside the office that .com domain doesn't point to the public web
host at all, … so even someone who opens the correct .co.uk version will, the moment they
click any internal link, have their browser request that link's hard-coded .com address.
```
> **So pretty soon they end up back on the office server version of the site.**   ← scored

The rationale sits **between** the context and its own continuation. The target now follows the *rationale*,
not the context — the original context→target adjacency is **broken**.

### prepend — rationale BEFORE the context (`rationale + context + target`)
```
The site's internal navigation links are hard-coded as absolute .com URLs … have their
browser request that link's hard-coded .com address.
…some of the links here are hard coded as .com rather than …site relative links.
```
> **So pretty soon they end up back on the office server version of the site.**   ← scored

Now the rationale is a **preamble**; the context→target flow is **intact** — but the rationale is far from
the target (the whole context sits in between), so its priming is diluted.

---

## Winogrande — the target is part of a *self-contained* sentence

There is **no prior context** to continue from — the sentence stands alone. So the rationale can only be
prepended, and there is **no flow to break**. (Scoring: fill the blank with the option, score the suffix.)

### base (`sentence-stem + option → suffix`)
```
Sarah was a much better surgeon than Maria so Maria
```
> **always got the easier cases.**   ← scored

### with rationale (always effectively "prepend", but nothing gets interrupted)
```
How tough an assignment a surgeon receives tracks their skill, with the abler one taking
the difficult cases and the less able one the routine ones. Sarah clearly outperformed
Maria in the operating room, which means the simpler work naturally fell to Maria.
Sarah was a much better surgeon than Maria so Maria
```
> **always got the easier cases.**   ← scored

The rationale precedes a self-contained sentence — no context→continuation adjacency exists to disrupt, and
the target is still close by (the sentence is short). **Both** benefits at once.

---

## Why the layout matters — measured with the placebo (an unrelated doc's rationale)

`placebo − base` isolates the pure **insertion cost** (off-topic rationale, so any effect is layout, not
content). `complete − placebo` is the **content benefit**. (1.4B judge, DCLM prose n=41, Winogrande n=200.)

| setup | insertion cost (placebo−base) | content benefit (complete−placebo) | net drop (complete−base) |
|---|---:|---:|---:|
| DCLM, **mid** | **+0.444** (big — breaks flow) | −0.546 | −0.103 |
| DCLM, **prepend** | **+0.068** (≈0 — flow intact) | −0.197 (diluted — far from target) | −0.130 |
| Winogrande (self-contained) | **−0.014** (≈0 — nothing to break) | −0.586 (full — close & clean) | **−0.600** |

**The trade-off on a continuation target (DCLM):**
- **mid** puts the rationale *near* the target (strong content, −0.546) but *breaks the flow* (+0.444 cost) → they cancel → net ≈ 0.
- **prepend** *keeps the flow* (+0.068 cost) but the rationale is now *far* from the target (content diluted to −0.197) → net still small.
- You cannot get the rationale **both** close to the target **and** without breaking coherence — so the drop is capped small either way.

**Winogrande escapes the trade-off entirely:** its scored span belongs to a self-contained sentence, so the
rationale prepends with **no flow to break** (cost ≈0) **and** stays close (short sentence → full content
−0.586). That structural difference — not richer reasoning — is why Winogrande shows a large clean drop
(−0.60) while DCLM caps out near zero.

*(The insertion cost also grows with model capability on DCLM-mid: 1.4B +0.444 → Llama-8B +0.556 →
Qwen-35B +0.713 — stronger, more coherence-sensitive models are more disrupted by a mid-document insertion.
See EXPERIMENT_LOG.md / WINOGRANDE_RESULTS.md.)*
