# Probe experiments — full data (probes, answers, base/real/placebo NLL)

**Judge:** DCLM-1.4B base (`1ep_dclm_step14672_hf`). NLL = nats/token of the ANSWER; delta = with-rationale − base; **delta<0 = rationale lowers the answer's perplexity.**
**Placebo** = the *same* measurement but the inserted rationale is **another doc's** rationale (irrelevant, format-matched) — isolates real reasoning from generic priming.

## 1. Probe-target (reasoning-determined answers) — raw −0.358, placebo −0.333

Answer NLL given `context + Q` (base) vs `context + rationale + Q` (real) vs `context + OTHER-doc rationale + Q` (placebo).

| id | answer | base | real | placebo | real−base | real−placebo |
|---|---|---:|---:|---:|---:|---:|
| 444 | spoilers never deployed | 8.2405 | 6.5854 | 8.237 | -1.655 | -1.651 |
| 458 | its low-cost advantage | 6.9959 | 5.7863 | 6.582 | -1.210 | -0.796 |
| 25 | stayed inaccurate | 13.0858 | 12.7958 | 13.189 | -0.290 | -0.393 |
| 89 | the boy's father | 3.5423 | 2.9302 | 3.323 | -0.612 | -0.393 |
| 404 | no, cause is ambiguous | 5.6795 | 5.3728 | 5.691 | -0.307 | -0.317 |
| 253 | it increases | 5.9574 | 5.4076 | 5.682 | -0.550 | -0.274 |
| 260 | they'd have risen too | 4.2249 | 4.0481 | 4.234 | -0.177 | -0.186 |
| 485 | undermines it | 9.2351 | 8.6765 | 8.806 | -0.559 | -0.129 |
| 302 | likely worse | 8.1914 | 8.3339 | 8.343 | +0.143 | -0.009 |
| 336 | it is superseded | 4.5483 | 4.5573 | 4.543 | +0.009 | +0.014 |
| 190 | 64 | 3.5637 | 3.1841 | 3.116 | -0.380 | +0.068 |
| 172 | mainly move it | 5.0856 | 4.8077 | 4.685 | -0.278 | +0.123 |
| 126 | it defends Twain's character | 5.4182 | 5.7423 | 5.481 | +0.324 | +0.261 |
| 430 | less tolerant | 7.9989 | 7.8776 | 7.419 | -0.121 | +0.459 |
| 153 | raise its requirements | 7.2892 | 6.3558 | 5.884 | -0.933 | +0.472 |
| 99 | it can't escape | 5.9989 | 6.0521 | 5.476 | +0.053 | +0.576 |
| 345 | worse | 15.0391 | 15.501 | 13.739 | +0.462 | +1.762 |

**mean real−base = −0.358 (70.6% drop); mean placebo−base = −0.333 → real−placebo ≈ −0.02 (mean).** The raw drop is mostly priming; a genuine reasoning-specific effect survives only on specific answers (see id 444, 458).

## 2. Strict specific-answer probes — real −0.406, placebo −0.307, reasoning-specific −0.099

| id | question | answer | real−base | placebo−base | real−placebo |
|---|---|---|---:|---:|---:|
| 444 | The pilot felt the airplane decelerating and assumed the air | they never deployed | -1.576 | -0.707 | -0.868 |
| 404 | A gardener gives plant A both more water and more sunlight t | confounding variables | -0.244 | +0.250 | -0.494 |
| 190 | With the value 0 reserved to mean 'no restriction', how many | 31 | -0.116 | -0.037 | -0.079 |
| 260 | Given Washington, Indiana gas runs 30 cents above Evansville | 10 cents | -0.151 | -0.149 | -0.002 |
| 25 | Roughly what fraction of the 62 analysts landed within 2.5 p | about 42 percent | +0.090 | +0.008 | +0.081 |
| 485 | Anna Friel survived for two months on the liquid Master Clea | malnutrition | -1.143 | -1.277 | +0.134 |
| 25 | How many of the 62 analysts did NOT come within 2.5 percent  | 36 | -0.044 | -0.226 | +0.182 |
| 190 | In the proposed bit-field scheme, how many distinct values c | 64 | -0.068 | -0.319 | +0.252 |

**Clean reasoning wins** (real≪placebo) only where the answer is a specific non-stated consequence (id 444 'they never deployed' −0.87; id 404 'confounding variables' −0.49). Numbers/terms ('64','malnutrition') stay priming-confounded.

## 3. Relationship: agent's R/N classification vs docs that ACTUALLY drop (Q6)

'R' = agent labelled the doc reasoning-dependent (i.e. 'adding a rationale WOULD help'). Continuation-ppl delta from `rn_results.jsonl`.

| label | n | mean delta | actually dropped |
|---|---:|---:|---:|
| R | 19 | +0.090 | 2/19 (11%) |
| N | 81 | +0.089 | 12/81 (15%) |

**Verdict: essentially NO relationship.** R docs dropped **11%** vs N docs **15%** — reasoning-dependent docs did NOT drop more (if anything slightly less). Of the 14 docs that actually dropped, only **2 were R** (14%), vs the 19% R base rate. So the classification 'a rationale would help this doc' does **not** predict which docs actually show a continuation-perplexity drop — the drops are priming/noise, not reasoning-dependence. This is the core reason the metric is confounded.

