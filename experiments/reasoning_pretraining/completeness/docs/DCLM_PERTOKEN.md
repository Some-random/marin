# DCLM — per-token perplexity of the real target continuation

For each real DCLM doc: the full context, the target span that actually follows, and three rationales that
get spliced BETWEEN context and target — `complete` (all steps), `incomplete` (same chain, load-bearing
middle steps deleted → a real gap), `placebo` (an UNRELATED doc's complete rationale). Then every target
token's NLL (nats/tok) under `base` (no rationale) and each splice. Judge = DCLM-1.4B.

**What to look for:** `+complete` ≈ `+incomplete` (completeness makes no difference — the model fills the
gap); `+placebo` is *higher* than base (an off-topic splice hurts); `+complete` ≪ `+placebo` (the relevant
reasoning is real). The net `+complete` vs `base` is small because the splice's insertion cost ~cancels it.


---

## doc 102770

**Context (real DCLM doc):**

> I have been taking care of a website that has both a .com and a .co.uk domain.
> The company who I have built this site for can access the site via the .co.uk address but the .com always takes them to their office server where an old version of the site is kept.
> 
> I have no experience with servers other than via a control panel for the company that I have webspace with (1and1).
> It is my understanding that the localhost on the office server has somehow been given a domain name of theukoffice .com and this is what is causing the error.
> 
> Now i know it would be simple just to get them to browse to the .co.uk domain but some of the links here are hard coded as .com rather than using /whatever/wherever.php or ../here/there.php site relative links.

**Target scored (the real continuation):** `So pretty soon they end up back on the office server version of the site.`

**① complete rationale (all steps):**

> The site's internal navigation links are hard-coded as absolute .com URLs rather than relative paths, and inside the office that .com domain doesn't point to the public web host at all, because the office server's own localhost was given the name theukoffice.com. Any request to the .com domain from an office machine is answered by the local office server, which is holding an outdated copy of the site, so even someone who deliberately opens the correct .co.uk version will, the moment they click any internal link, have their browser request that link's hard-coded .com address.

**② incomplete rationale (middle steps deleted — note the gap):**

> The site's internal navigation links are hard-coded as absolute .com URLs rather than relative paths, so even someone who deliberately opens the correct .co.uk version will, the moment they click any internal link, have their browser request that link's hard-coded .com address.

**③ placebo (unrelated doc 101869's complete rationale):**

> The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from a tax cheat, and that financial incentive is enough to tempt even someone's closest friends, colleagues, or partners into turning them in once they learn about the cheating. Social media has made it far easier for private information to be published out in the open where anyone can see it, which means any incriminating detail a tax cheat shares publicly online lands right in front of exactly those acquaintances who might be tempted by the reward to report them.

Per-token NLL of the target:

| # | token | base | +complete | +incomplete | +placebo | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `So` | 3.374 | 2.675 | 2.994 | 2.801 | -0.699 |
| 1 | ` pretty` | 9.167 | 8.553 | 8.679 | 9.048 | -0.613 |
| 2 | ` soon` | 2.812 | 3.043 | 3.144 | 2.635 | +0.230 |
| 3 | ` they` | 2.239 | 2.749 | 2.143 | 3.984 | +0.510 |
| 4 | ` end` | 4.907 | 3.505 | 3.151 | 6.683 | -1.403 |
| 5 | ` up` | 0.014 | 0.008 | 0.008 | 0.008 | -0.006 |
| 6 | ` back` | 4.916 | 4.480 | 4.587 | 5.889 | -0.437 |
| 7 | ` on` | 1.352 | 1.780 | 1.634 | 1.987 | +0.428 |
| 8 | ` the` | 0.405 | 0.325 | 0.359 | 0.336 | -0.080 |
| 9 | ` office` | 2.316 | 2.782 | 2.261 | 5.325 | +0.466 |
| 10 | ` server` | 0.186 | 0.425 | 0.192 | 0.313 | +0.238 |
| 11 | ` version` | 8.504 | 8.333 | 7.717 | 10.637 | -0.172 |
| 12 | ` of` | 0.608 | 0.802 | 0.590 | 0.280 | +0.194 |
| 13 | ` the` | 0.156 | 0.088 | 0.108 | 0.255 | -0.068 |
| 14 | ` site` | 0.242 | 0.131 | 0.178 | 0.504 | -0.110 |
| 15 | `.` | 3.338 | 3.089 | 3.408 | 2.741 | -0.249 |
| | **MEAN** | **2.784** | **2.673** | **2.572** | **3.339** | **-0.111** |

**Perplexity:** base 16.18 · +complete 14.48 · +incomplete 13.09 · +placebo 28.20  ·  complete−incomplete +0.101 (completeness) · complete−placebo -0.666 (content)


---

## doc 101966

**Context (real DCLM doc):**

> The cause of appendicitis isn't always clear. Sometimes appendicitis can occur as a result of:
> 
>   • An obstruction. Food waste or a hard piece of stool (fecal stone) can block the opening of the cavity that runs the length of your appendix.
>   • An infection. Appendicitis may also follow an infection, such as a gastrointestinal viral infection, or it may result from other types of inflammation.
> 
> In both cases, bacteria inside the appendix multiply rapidly, causing the appendix to become inflamed, swollen and filled with pus.

**Target scored (the real continuation):** `If not treated promptly, the appendix can rupture.`

**① complete rationale (all steps):**

> An obstruction blocks the opening of the appendix, sealing its cavity so that its contents can no longer drain out, and the bacteria trapped inside that sealed space multiply rapidly, generating pus and gas that keep piling up with nowhere to go. Since the sealed cavity has no way to vent, that buildup drives the internal pressure steadily higher and presses on the blood vessels running through the appendix wall, choking off its blood supply. Starved of blood and stretched thin by the mounting pressure inside, the wall of the appendix grows more and more fragile.

**② incomplete rationale (middle steps deleted — note the gap):**

> An obstruction blocks the opening of the appendix, sealing its cavity so that its contents can no longer drain out, and the bacteria trapped inside that sealed space multiply rapidly, generating pus and gas that keep piling up with nowhere to go. Starved of blood and stretched thin by the mounting pressure inside, the wall of the appendix grows more and more fragile.

**③ placebo (unrelated doc 101991's complete rationale):**

> As things stand, artists can put their music out through cheap, open online channels, holding on to most of the revenue and cutting the RIAA and the other middlemen out of the equation. A proprietary DRM format, though, isn't free to use: it's owned and controlled by whatever company holds its rights, and anyone who wants to use it needs a license from that owner. If a law were to mandate one specific DRM format as the only legal way to distribute music online, then every distributor, artists releasing their own work included, would be forced to use that format, and to do so they would all have to obtain and pay for a license from the company that owns it. That rights-holding company would then sit squarely between the artists and their audience, a gatekeeper everyone is legally required to pass through and pay.

Per-token NLL of the target:

| # | token | base | +complete | +incomplete | +placebo | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `If` | 4.388 | 3.322 | 3.290 | 3.477 | -1.066 |
| 1 | ` not` | 5.239 | 4.566 | 4.249 | 6.303 | -0.673 |
| 2 | ` treated` | 0.042 | 0.220 | 0.278 | 9.199 | +0.178 |
| 3 | ` promptly` | 2.707 | 2.927 | 3.114 | 3.801 | +0.220 |
| 4 | `,` | 0.060 | 0.120 | 0.104 | 0.211 | +0.060 |
| 5 | ` the` | 1.209 | 1.345 | 1.298 | 1.698 | +0.136 |
| 6 | ` appendix` | 1.129 | 2.088 | 1.577 | 3.995 | +0.958 |
| 7 | ` can` | 0.357 | 0.547 | 0.494 | 0.513 | +0.189 |
| 8 | ` rupture` | 1.294 | 1.160 | 1.020 | 1.583 | -0.133 |
| 9 | `.` | 2.854 | 2.925 | 3.036 | 2.831 | +0.071 |
| | **MEAN** | **1.928** | **1.922** | **1.846** | **3.361** | **-0.006** |

**Perplexity:** base 6.88 · +complete 6.83 · +incomplete 6.33 · +placebo 28.82  ·  complete−incomplete +0.076 (completeness) · complete−placebo -1.439 (content)


---

## doc 100989

**Context (real DCLM doc):**

> hairy ball
> 
> (topology)   A result in topology stating that a continuous vector field on a sphere is always zero somewhere. The name comes from the fact that you can't flatten all the hair on a hairy ball, like a tennis ball, there will always be a tuft somewhere (where the tangential projection of the hair is zero). An immediate corollary to this theorem is that for any continuous map f of the sphere into itself there is a point x such that f(x)=x or f(x) is the antipode of x.

**Target scored (the real continuation):** `Another corollary is that at any moment somewhere on the Earth there is no wind.`

**① complete rationale (all steps):**

> The hairy ball theorem tells us that any continuous vector field lying tangent to a sphere has to equal zero at at least one point on that sphere, and the surface of the Earth is, in its shape, a sphere. At every location on that surface the wind blows horizontally along the ground, so the wind at each point is a vector lying tangent to the spherical surface, and taken all together those winds form a tangent vector field on the sphere. Because wind direction and speed change gradually from one place to the next, this tangent vector field is continuous, and a continuous tangent vector field on a sphere satisfies exactly the theorem's hypothesis, so it must be zero somewhere — a location where the wind vector has no magnitude at all.

**② incomplete rationale (middle steps deleted — note the gap):**

> The hairy ball theorem tells us that any continuous vector field lying tangent to a sphere has to equal zero at at least one point on that sphere, and the surface of the Earth is, in its shape, a sphere. A continuous tangent vector field on a sphere satisfies exactly the theorem's hypothesis, so it must be zero somewhere — a location where the wind vector has no magnitude at all.

**③ placebo (unrelated doc 101673's complete rationale):**

> A star is an immense ball of gas held together as a single body, so all of its matter exerts a gravitational attraction, and that attraction pulls every part of the star inward toward its center. Left to act on its own, this inward pull would make the star collapse in on itself. At the same time, the nuclear fusion in the core that converts hydrogen to helium releases enormous energy as heat and radiation that pushes outward, and that outward push from the core's fusion works directly against the inward pull of gravity.

Per-token NLL of the target:

| # | token | base | +complete | +incomplete | +placebo | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `Another` | 8.110 | 5.288 | 5.656 | 6.620 | -2.822 |
| 1 | ` cor` | 1.897 | 1.914 | 2.272 | 4.278 | +0.017 |
| 2 | `oll` | 0.001 | 0.000 | 0.000 | 0.000 | -0.000 |
| 3 | `ary` | 0.035 | 0.023 | 0.015 | 0.015 | -0.012 |
| 4 | ` is` | 0.671 | 1.440 | 1.227 | 1.145 | +0.768 |
| 5 | ` that` | 0.275 | 0.335 | 0.337 | 0.249 | +0.059 |
| 6 | ` at` | 5.251 | 4.451 | 5.145 | 4.451 | -0.801 |
| 7 | ` any` | 1.689 | 1.430 | 1.978 | 1.355 | -0.258 |
| 8 | ` moment` | 6.394 | 7.067 | 7.174 | 4.650 | +0.673 |
| 9 | ` somewhere` | 9.408 | 7.908 | 8.886 | 8.907 | -1.500 |
| 10 | ` on` | 2.004 | 0.705 | 1.203 | 1.659 | -1.298 |
| 11 | ` the` | 0.360 | 0.406 | 0.444 | 0.349 | +0.045 |
| 12 | ` Earth` | 7.355 | 1.508 | 1.618 | 4.193 | -5.847 |
| 13 | ` there` | 1.130 | 1.848 | 1.700 | 1.592 | +0.718 |
| 14 | ` is` | 0.404 | 0.350 | 0.433 | 0.450 | -0.054 |
| 15 | ` no` | 3.141 | 2.895 | 2.912 | 3.669 | -0.247 |
| 16 | ` wind` | 8.022 | 2.240 | 5.680 | 6.101 | -5.782 |
| 17 | `.` | 1.754 | 2.070 | 1.875 | 1.825 | +0.316 |
| | **MEAN** | **3.217** | **2.327** | **2.697** | **2.862** | **-0.890** |

**Perplexity:** base 24.95 · +complete 10.24 · +incomplete 14.84 · +placebo 17.49  ·  complete−incomplete -0.371 (completeness) · complete−placebo -0.535 (content)


---

## doc 100293

**Context (real DCLM doc):**

> Tyco Electronics Corp. unveils its High Resolution Radar (HRR) for the next generation of "Smart Bumpers."
> 
> Tyco says its system offers significant advantages over ultrasonic sensing systems, including greater range, higher resolution between objects and important styling advantages.
> 
> HRR works by transmitting a short pulse into a desired area. This energy is reflected off objects within 66 ft. (20.1 m) and returns.

**Target scored (the real continuation):** `The travel time of the signal determines the range of the object.`

**① complete rationale (all steps):**

> The HRR fires a short pulse of radar energy out into the target area. That energy is electromagnetic, so it moves at a fixed, known speed that stays the same regardless of how far it goes. The pulse travels out to an object, bounces off it, and comes back to the sensor, covering a round trip equal to twice the object's range. Since the speed is constant, the distance covered is just that speed multiplied by the time elapsed, so a longer elapsed time means the pulse has traveled a longer round-trip distance. And because that round trip is exactly twice the range, a longer elapsed time maps directly onto a greater distance to the object.

**② incomplete rationale (middle steps deleted — note the gap):**

> The HRR fires a short pulse of radar energy out into the target area. The pulse travels out to an object, bounces off it, and comes back to the sensor, covering a round trip equal to twice the object's range. And because that round trip is exactly twice the range, a longer elapsed time maps directly onto a greater distance to the object.

**③ placebo (unrelated doc 101013's complete rationale):**

> He takes his medication to hold his blood glucose within a safe range, so the most likely immediate effect of skipping a dose is that his glucose climbs. A high glucose level can on its own bring on vomiting, dehydration, weakness, and a general sense of being unwell. He felt sick and vomited that very night he missed his medicine, and those symptoms line up with the known effects of high glucose, which points to the episode being driven by elevated glucose rather than being something harmless. On top of that, his glucose may well have been running high for several days before he ever skipped the dose, a sign that it isn't being held reliably in range even under his usual treatment. Since it is uncontrolled high glucose that produces these worrying symptoms, keeping future episodes at bay comes down to holding his glucose steadily within a safe range.

Per-token NLL of the target:

| # | token | base | +complete | +incomplete | +placebo | Δ(compl−base) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `The` | 1.953 | 1.306 | 1.368 | 2.104 | -0.647 |
| 1 | ` travel` | 10.397 | 10.673 | 10.643 | 10.726 | +0.276 |
| 2 | ` time` | 1.234 | 1.063 | 0.778 | 2.005 | -0.171 |
| 3 | ` of` | 1.660 | 1.544 | 1.895 | 1.841 | -0.116 |
| 4 | ` the` | 0.594 | 0.521 | 0.558 | 1.000 | -0.073 |
| 5 | ` signal` | 2.601 | 4.002 | 3.842 | 2.951 | +1.401 |
| 6 | ` determines` | 5.165 | 4.590 | 4.950 | 6.076 | -0.575 |
| 7 | ` the` | 0.727 | 0.737 | 0.672 | 0.957 | +0.010 |
| 8 | ` range` | 2.950 | 2.832 | 3.146 | 3.013 | -0.118 |
| 9 | ` of` | 1.299 | 1.626 | 1.421 | 1.457 | +0.327 |
| 10 | ` the` | 0.620 | 0.390 | 0.419 | 1.112 | -0.230 |
| 11 | ` object` | 2.860 | 1.675 | 1.345 | 6.545 | -1.185 |
| 12 | `.` | 1.304 | 1.116 | 1.140 | 2.584 | -0.188 |
| | **MEAN** | **2.567** | **2.467** | **2.475** | **3.259** | **-0.099** |

**Perplexity:** base 13.02 · +complete 11.79 · +incomplete 11.88 · +placebo 26.03  ·  complete−incomplete -0.008 (completeness) · complete−placebo -0.792 (content)

