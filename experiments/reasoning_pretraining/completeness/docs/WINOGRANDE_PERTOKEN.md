# Winogrande — per-token perplexity of the scored span (BLANK INCLUDED)

Scored span = **`option + suffix`** (e.g. `Maria always got the easier cases.`), given the pre-blank stem —
so the blank/answer token itself is now part of what's scored, followed by the rest of the sentence. Each
row is one token of that span with its NLL (nats/token) under `base` (no rationale) and each rationale
prepended, all under the **correct** option. The `‹opt›` rows are the blank token(s). Judge = DCLM-1.4B.
The model picks whichever option gives the lower mean span-NLL; margin vs the wrong option shown too.


---

## example idx 0

**Sentence:** Sarah was a much better surgeon than Maria so ▁ always got the easier cases.
**option1:** `Sarah`  ·  **option2:** `Maria`  ·  **answer:** 2 → **`Maria`** (correct)

**Scored span (blank + suffix, correct option):** `Maria always got the easier cases.`

**principle:** Between two professionals of unequal ability, the less capable one tends to be handed the more manageable work while the stronger takes on the demanding tasks.

**full:** Maria was the weaker of the two surgeons, so she was the one given the lighter, more routine operations.

**complete:** How tough an assignment a surgeon receives tracks their skill, with the abler one taking the difficult cases and the less able one the routine ones. Sarah clearly outperformed Maria in the operating room, which means the simpler, less demanding work naturally fell to Maria.

Per-token NLL of the scored span (option first, then suffix), under the **correct** option:

| # | token | base | +principle | +full | +complete | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | ` Maria` ⟵blank | 4.205 | 3.815 | 5.179 | 3.816 | -0.390 |
| 1 | ` always` | 4.919 | 4.528 | 5.072 | 5.541 | +0.621 |
| 2 | ` got` | 2.348 | 2.239 | 1.325 | 2.369 | +0.021 |
| 3 | ` the` | 1.360 | 1.052 | 0.726 | 0.738 | -0.622 |
| 4 | ` easier` | 6.521 | 3.613 | 3.275 | 2.510 | -4.011 |
| 5 | ` cases` | 1.411 | 3.234 | 2.833 | 0.452 | -0.959 |
| 6 | `.` | 0.518 | 1.332 | 1.579 | 0.930 | +0.412 |
| | **MEAN** | **3.040** | **2.831** | **2.856** | **2.336** | **-0.704** |

**Perplexity (correct span) & margin vs the wrong option (`Sarah`):**

| condition | ppl(correct) | mean NLL correct | mean NLL wrong | margin (wrong−correct) | model picks |
|---|---:|---:|---:|---:|:---:|
| base | 20.91 | 3.040 | 3.263 | +0.223 | ✓ correct |
| principle | 16.95 | 2.831 | 2.992 | +0.162 | ✓ correct |
| full | 17.38 | 2.856 | 2.862 | +0.006 | ✓ correct |
| complete | 10.34 | 2.336 | 2.527 | +0.191 | ✓ correct |

---

## example idx 3

**Sentence:** Terry tried to bake the eggplant in the toaster oven but the ▁ was too big.
**option1:** `eggplant`  ·  **option2:** `toaster`  ·  **answer:** 1 → **`eggplant`** (correct)

**Scored span (blank + suffix, correct option):** `eggplant was too big.`

**principle:** When an item won't go into a container, it is the object being inserted that is overlarge, not the space meant to hold it.

**full:** The eggplant would not go into the small toaster oven, meaning the eggplant was the oversized one.

**complete:** A toaster oven offers only a cramped interior, so anything placed inside has to stay compact. Terry could not get the eggplant to fit into that little space, and the only way something fails to go in is by exceeding the room available, so the eggplant was the oversized item.

Per-token NLL of the scored span (option first, then suffix), under the **correct** option:

| # | token | base | +principle | +full | +complete | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | ` egg` ⟵blank | 3.549 | 4.762 | 2.663 | 3.343 | -0.205 |
| 1 | `plant` ⟵blank | 0.032 | 0.019 | 0.010 | 0.011 | -0.021 |
| 2 | ` was` | 1.301 | 1.375 | 1.567 | 1.434 | +0.132 |
| 3 | ` too` | 2.323 | 0.979 | 1.164 | 0.816 | -1.507 |
| 4 | ` big` | 3.118 | 1.446 | 1.295 | 1.693 | -1.424 |
| 5 | `.` | 1.429 | 1.685 | 1.692 | 1.448 | +0.019 |
| | **MEAN** | **1.959** | **1.711** | **1.398** | **1.457** | **-0.501** |

**Perplexity (correct span) & margin vs the wrong option (`toaster`):**

| condition | ppl(correct) | mean NLL correct | mean NLL wrong | margin (wrong−correct) | model picks |
|---|---:|---:|---:|---:|:---:|
| base | 7.09 | 1.959 | 2.764 | +0.805 | ✓ correct |
| principle | 5.53 | 1.711 | 1.869 | +0.158 | ✓ correct |
| full | 4.05 | 1.398 | 2.254 | +0.855 | ✓ correct |
| complete | 4.30 | 1.457 | 2.157 | +0.700 | ✓ correct |

---

## example idx 4

**Sentence:** At night, Jeffrey always stays up later than Hunter to watch TV because ▁ wakes up late.
**option1:** `Jeffrey`  ·  **option2:** `Hunter`  ·  **answer:** 1 → **`Jeffrey`** (correct)

**Scored span (blank + suffix, correct option):** `Jeffrey wakes up late.`

**principle:** Someone who does not have to be up early in the morning can afford to stay awake later at night without losing sleep.

**full:** Jeffrey is the one who sleeps in, which is why he can remain awake into the night watching television.

**complete:** A person whose mornings start later can stay up longer without paying for it. Jeffrey routinely stays awake later than Hunter, and the natural reason is that his day begins later in the morning, so Jeffrey is the one who rises late.

Per-token NLL of the scored span (option first, then suffix), under the **correct** option:

| # | token | base | +principle | +full | +complete | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | ` Jeffrey` ⟵blank | 5.727 | 4.838 | 5.114 | 5.516 | -0.211 |
| 1 | ` wakes` | 6.934 | 5.275 | 6.064 | 4.309 | -2.625 |
| 2 | ` up` | 0.290 | 0.238 | 0.275 | 0.268 | -0.022 |
| 3 | ` late` | 4.315 | 4.293 | 4.467 | 3.370 | -0.945 |
| 4 | `.` | 1.453 | 2.635 | 2.281 | 1.882 | +0.429 |
| | **MEAN** | **3.744** | **3.456** | **3.640** | **3.069** | **-0.675** |

**Perplexity (correct span) & margin vs the wrong option (`Hunter`):**

| condition | ppl(correct) | mean NLL correct | mean NLL wrong | margin (wrong−correct) | model picks |
|---|---:|---:|---:|---:|:---:|
| base | 42.25 | 3.744 | 3.011 | -0.733 | ✗ wrong |
| principle | 31.68 | 3.456 | 3.118 | -0.337 | ✗ wrong |
| full | 38.10 | 3.640 | 2.964 | -0.676 | ✗ wrong |
| complete | 21.52 | 3.069 | 2.082 | -0.987 | ✗ wrong |

---

## example idx 2

**Sentence:** They were worried the wine would ruin the bed and the blanket, but the ▁ was't ruined.
**option1:** `blanket`  ·  **option2:** `bed`  ·  **answer:** 2 → **`bed`** (correct)

**Scored span (blank + suffix, correct option):** `bed was't ruined.`

**principle:** When a spill lands on stacked layers, the outer covering soaks up the liquid and shields whatever lies beneath it.

**full:** The blanket lay on top and absorbed the wine, so the bed underneath came through unharmed.

**complete:** When wine spills onto a made-up bed, the topmost layer catches the liquid before it can reach anything below. The blanket was that outer layer and took in the spill, which spared the mattress beneath it, so the bed itself escaped the damage.

Per-token NLL of the scored span (option first, then suffix), under the **correct** option:

| # | token | base | +principle | +full | +complete | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | ` bed` ⟵blank | 5.228 | 4.426 | 2.431 | 3.460 | -1.768 |
| 1 | ` was` | 0.997 | 1.140 | 0.888 | 0.993 | -0.004 |
| 2 | `'t` | 13.861 | 13.990 | 14.117 | 13.927 | +0.066 |
| 3 | ` ruined` | 3.223 | 2.435 | 2.679 | 2.092 | -1.131 |
| 4 | `.` | 1.314 | 2.015 | 1.783 | 1.222 | -0.093 |
| | **MEAN** | **4.925** | **4.801** | **4.380** | **4.339** | **-0.586** |

**Perplexity (correct span) & margin vs the wrong option (`blanket`):**

| condition | ppl(correct) | mean NLL correct | mean NLL wrong | margin (wrong−correct) | model picks |
|---|---:|---:|---:|---:|:---:|
| base | 137.66 | 4.925 | 5.117 | +0.193 | ✓ correct |
| principle | 121.66 | 4.801 | 4.655 | -0.146 | ✗ wrong |
| full | 79.82 | 4.380 | 4.612 | +0.232 | ✓ correct |
| complete | 76.62 | 4.339 | 4.415 | +0.076 | ✓ correct |
