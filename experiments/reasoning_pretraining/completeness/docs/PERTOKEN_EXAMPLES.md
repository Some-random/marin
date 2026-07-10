# Per-token perplexity calculation (DCLM-1.4B base judge)

Judge tokenizer = the model's own (`AutoTokenizer.from_pretrained`). For each doc: the real continuation
target's tokens, NLL (nats/token) under `context` (base) vs `context + complete rationale`, and the diff.
**diff < 0 = the rationale made that token more predictable.** The reported per-doc Δ is the mean-diff.


---

## doc id 102770

**CONTEXT:**

> I have been taking care of a website that has both a .com and a .co.uk domain.
The company who I have built this site for can access the site via the .co.uk address but the .com always takes them to their office server where an old version of the site is kept.

I have no experience with servers other than via a control panel for the company that I have webspace with (1and1).
It is my understanding that the localhost on the office server has somehow been given a domain name of theukoffice .com and this is what is causing the error.

Now i know it would be simple just to get them to browse to the .co.uk domain but some of the links here are hard coded as .com rather than using /whatever/wherever.php or ../here/there.php site relative links.

**COMPLETE RATIONALE (inserted before the target):**

> 1. The site's internal navigation links are hard-coded as absolute URLs on the .com domain instead of relative paths.
> 2. Inside the office, the .com domain does not point to the public web host, because the office server's localhost was given the name theukoffice.com.
> 3. So any request to the .com domain from an office machine is answered by the local office server, which stores an outdated copy of the site.
> 4. Even a user who deliberately opens the correct .co.uk copy will, on clicking any internal link, have the browser request that link's hard-coded .com address.

**CONTINUATION being scored (the real next text):** `So pretty soon they end up back on the office server version of the site.`

| # | token | NLL base | NLL +rationale | diff |
|---:|---|---:|---:|---:|
| 0 | `So` | 3.374 | 3.476 | +0.101 |
| 1 | ` pretty` | 9.167 | 8.196 | -0.970 |
| 2 | ` soon` | 2.812 | 2.827 | +0.015 |
| 3 | ` they` | 2.239 | 3.211 | +0.971 |
| 4 | ` end` | 4.907 | 3.289 | -1.619 |
| 5 | ` up` | 0.014 | 0.009 | -0.005 |
| 6 | ` back` | 4.916 | 4.742 | -0.174 |
| 7 | ` on` | 1.352 | 1.783 | +0.430 |
| 8 | ` the` | 0.405 | 0.271 | -0.134 |
| 9 | ` office` | 2.316 | 2.545 | +0.229 |
| 10 | ` server` | 0.186 | 0.450 | +0.264 |
| 11 | ` version` | 8.504 | 8.789 | +0.284 |
| 12 | ` of` | 0.608 | 0.472 | -0.136 |
| 13 | ` the` | 0.156 | 0.069 | -0.086 |
| 14 | ` site` | 0.242 | 0.119 | -0.123 |
| 15 | `.` | 3.338 | 2.982 | -0.357 |
| | **MEAN** | **2.784** | **2.702** | **-0.082** |

**perplexity:** base 16.18 → +rationale 14.91 · **mean-diff (the reported Δ) = -0.0818 nats/token**


---

## doc id 101869

**CONTEXT:**

> You have a big mouth
You have a big mouth
You may think you're a hot shot for pulling a fast one on the IRS. But when the friend you entrusted with your secret snitches on you in exchange for a fat check, you're going to be in big trouble.

"Most cases start the old-fashioned way," said Ian Comisky, a partner at law firm Blank Rome LLP who represents taxpayers whose returns were flagged by the IRS. "You blab about it to a friend, colleague, spouse or girlfriend, and one of them turns you in."

Even your closest pals may be tempted to tattle, since the IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.

And with the popularity of social media, it's now much easier to publish private information publicly.

**COMPLETE RATIONALE (inserted before the target):**

> 1. The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
> 2. That financial incentive tempts even a person's closest friends, colleagues, or partners to turn them in if they learn about the cheating.
> 3. The rise of social media has made it far easier for private information to be published publicly where anyone can see it.
> 4. So any incriminating detail a tax cheat shares publicly online becomes visible to exactly those acquaintances who could be tempted by the reward to report them to the IRS.

**CONTINUATION being scored (the real next text):** `So if you did something you think was questionable, don't post it all over Facebook.`

| # | token | NLL base | NLL +rationale | diff |
|---:|---|---:|---:|---:|
| 0 | `So` | 3.759 | 6.899 | +3.139 |
| 1 | ` if` | 2.364 | 2.085 | -0.279 |
| 2 | ` you` | 0.301 | 0.420 | +0.119 |
| 3 | ` did` | 6.339 | 6.287 | -0.052 |
| 4 | ` something` | 2.151 | 1.752 | -0.399 |
| 5 | ` you` | 3.166 | 3.483 | +0.316 |
| 6 | ` think` | 5.254 | 5.058 | -0.195 |
| 7 | ` was` | 2.186 | 2.040 | -0.146 |
| 8 | ` questionable` | 5.878 | 6.000 | +0.123 |
| 9 | `,` | 0.537 | 0.721 | +0.184 |
| 10 | ` don` | 3.282 | 3.024 | -0.258 |
| 11 | `'t` | 0.004 | 0.009 | +0.005 |
| 12 | ` post` | 4.771 | 4.535 | -0.236 |
| 13 | ` it` | 0.356 | 0.404 | +0.048 |
| 14 | ` all` | 5.453 | 5.351 | -0.102 |
| 15 | ` over` | 2.276 | 2.635 | +0.359 |
| 16 | ` Facebook` | 1.433 | 1.498 | +0.065 |
| 17 | `.` | 2.073 | 1.824 | -0.249 |
| | **MEAN** | **2.866** | **3.001** | **+0.136** |

**perplexity:** base 17.56 → +rationale 20.11 · **mean-diff (the reported Δ) = +0.1357 nats/token**


---

## doc id 100989

**CONTEXT:**

> hairy ball

(topology)   A result in topology stating that a continuous vector field on a sphere is always zero somewhere. The name comes from the fact that you can't flatten all the hair on a hairy ball, like a tennis ball, there will always be a tuft somewhere (where the tangential projection of the hair is zero). An immediate corollary to this theorem is that for any continuous map f of the sphere into itself there is a point x such that f(x)=x or f(x) is the antipode of x.

**COMPLETE RATIONALE (inserted before the target):**

> 1. The hairy ball theorem states that any continuous vector field lying tangent to a sphere must equal zero at at least one point on that sphere.
> 2. The surface of the Earth is, in shape, a sphere.
> 3. At every location on the Earth's surface the wind blows horizontally along the ground, so the wind at each point is a vector lying tangent to the spherical surface; taken together the winds form a tangent vector field on the sphere.
> 4. Wind direction and speed change gradually from one place to the next, so this tangent vector field is continuous.
> 5. A continuous tangent vector field on a sphere meets the theorem's hypothesis, so by the theorem it must be zero at some point — a location where the wind vector has zero magnitude.

**CONTINUATION being scored (the real next text):** `Another corollary is that at any moment somewhere on the Earth there is no wind.`

| # | token | NLL base | NLL +rationale | diff |
|---:|---|---:|---:|---:|
| 0 | `Another` | 8.110 | 10.083 | +1.972 |
| 1 | ` cor` | 1.897 | 2.453 | +0.556 |
| 2 | `oll` | 0.001 | 0.000 | -0.000 |
| 3 | `ary` | 0.035 | 0.031 | -0.005 |
| 4 | ` is` | 0.671 | 1.338 | +0.666 |
| 5 | ` that` | 0.275 | 0.520 | +0.245 |
| 6 | ` at` | 5.251 | 3.974 | -1.277 |
| 7 | ` any` | 1.689 | 1.266 | -0.423 |
| 8 | ` moment` | 6.394 | 6.807 | +0.413 |
| 9 | ` somewhere` | 9.408 | 8.210 | -1.198 |
| 10 | ` on` | 2.004 | 0.727 | -1.277 |
| 11 | ` the` | 0.360 | 0.348 | -0.013 |
| 12 | ` Earth` | 7.355 | 0.292 | -7.063 |
| 13 | ` there` | 1.130 | 2.537 | +1.407 |
| 14 | ` is` | 0.404 | 0.387 | -0.017 |
| 15 | ` no` | 3.141 | 3.182 | +0.041 |
| 16 | ` wind` | 8.022 | 1.923 | -6.099 |
| 17 | `.` | 1.754 | 2.243 | +0.489 |
| | **MEAN** | **3.217** | **2.573** | **-0.644** |

**perplexity:** base 24.95 → +rationale 13.11 · **mean-diff (the reported Δ) = -0.6435 nats/token**


---

## doc id 101673

**CONTEXT:**

> Star goes out with a whimper, not a bang, captured by Hubble telescope

  |   July 12, 2013 at 5:50 PM
GREENBELT, Md., July 12 (UPI) -- Not all stars go out with a bang, and NASA has released an image from the Hubble telescope of one going out with more of a cosmic whimper.

The image shows the planetary nebula IC 289, located in the northern constellation of Cassiopeia. Formerly a star like the sun, it is now just a thin cloud of ionized gas being pushed out into space by the remnants of the star's core, visible as a small bright dot in the middle of the cloud, NASA said Friday.

Planetary nebulae have nothing to do with planets -- the term is a relic from a time when early observers with small telescopes could only see undefined, hazy objects they thought looked like gaseous planets.

Stars are powered by nuclear fusion reactions in their cores that convert hydrogen to helium.

**COMPLETE RATIONALE (inserted before the target):**

> 1. A star is an immense ball of gas held together as a single body, so all of its matter exerts gravitational attraction.
> 2. That attraction pulls every part of the star inward toward its center.
> 3. Acting alone, this inward pull would make the star collapse in on itself.
> 4. The core's nuclear fusion, which converts hydrogen to helium, releases enormous energy as heat and radiation that pushes outward.
> 5. That outward push from the core's fusion acts directly against the inward pull.

**CONTINUATION being scored (the real next text):** `The star remains stable in a process that balances the inward squeeze caused by its gravity with the outward thrust from the inner fusion reaction in its core.`

| # | token | NLL base | NLL +rationale | diff |
|---:|---|---:|---:|---:|
| 0 | `The` | 1.878 | 4.572 | +2.695 |
| 1 | ` star` | 3.120 | 2.916 | -0.203 |
| 2 | ` remains` | 5.788 | 5.618 | -0.170 |
| 3 | ` stable` | 3.546 | 2.763 | -0.783 |
| 4 | ` in` | 2.584 | 2.781 | +0.197 |
| 5 | ` a` | 2.633 | 2.777 | +0.144 |
| 6 | ` process` | 5.081 | 5.463 | +0.382 |
| 7 | ` that` | 1.951 | 2.224 | +0.273 |
| 8 | ` balances` | 7.968 | 6.729 | -1.239 |
| 9 | ` the` | 0.769 | 1.221 | +0.452 |
| 10 | ` inward` | 4.331 | 1.290 | -3.041 |
| 11 | ` squeeze` | 9.057 | 7.873 | -1.184 |
| 12 | ` caused` | 4.422 | 5.693 | +1.271 |
| 13 | ` by` | 0.032 | 0.016 | -0.016 |
| 14 | ` its` | 2.619 | 2.471 | -0.148 |
| 15 | ` gravity` | 1.659 | 2.603 | +0.945 |
| 16 | ` with` | 0.834 | 1.168 | +0.334 |
| 17 | ` the` | 0.584 | 0.418 | -0.166 |
| 18 | ` outward` | 0.270 | 0.189 | -0.081 |
| 19 | ` thrust` | 3.381 | 3.970 | +0.589 |
| 20 | ` from` | 1.900 | 1.716 | -0.184 |
| 21 | ` the` | 0.915 | 0.718 | -0.197 |
| 22 | ` inner` | 5.645 | 5.407 | -0.238 |
| 23 | ` fusion` | 5.411 | 4.459 | -0.952 |
| 24 | ` reaction` | 1.214 | 2.151 | +0.936 |
| 25 | ` in` | 3.770 | 4.240 | +0.470 |
| 26 | ` its` | 0.629 | 0.804 | +0.175 |
| 27 | ` core` | 0.221 | 0.219 | -0.002 |
| 28 | `.` | 0.967 | 1.712 | +0.745 |
| | **MEAN** | **2.868** | **2.903** | **+0.035** |

**perplexity:** base 17.60 → +rationale 18.22 · **mean-diff (the reported Δ) = +0.0346 nats/token**


---

## doc id 100391

**CONTEXT:**

> Good News! NASA Is (Probably) Getting More MoneyS

NASA needs more money, because let's face it, rocket launches ain't cheap. The good news is, it looks like they'll be getting some. Not as much as they want, but some.

In October NASA said they would need $3 billion more per year to go forward with meaningful human space exploration, i.e. not just sending more robots up. For a while there were rumors going around that Washington was going to severely scale back the program's budget, but now according to Washington insider John Logsdon, "there will be more money."

He's also saying that Obama doesn't want to be that president who cuts a future oriented program. So he'll keep it alive, but he'll only give them a budget somewhere between their current spending and the $3 billion per year increase NASA is looking for.

**COMPLETE RATIONALE (inserted before the target):**

> 1. NASA said it needs an extra $3 billion per year to go forward with meaningful human space exploration rather than just sending more robots.
> 2. Obama will keep the program alive but only fund it somewhere between NASA's current spending and that full $3 billion increase.
> 3. So NASA will receive less than the $3 billion it asked for, leaving a funding shortfall relative to its stated needs.
> 4. With less money than it says it needs, NASA cannot fully afford meaningful human space exploration on its own budget.
> 5. The only way to still pursue those goals despite the gap is to bring in outside contributors who can share the costs and resources.

**CONTINUATION being scored (the real next text):** `But all that means is that NASA will have to buddy up with international space programs a little more.`

| # | token | NLL base | NLL +rationale | diff |
|---:|---|---:|---:|---:|
| 0 | `But` | 4.233 | 6.709 | +2.476 |
| 1 | ` all` | 5.536 | 5.226 | -0.310 |
| 2 | ` that` | 1.757 | 2.062 | +0.305 |
| 3 | ` means` | 3.279 | 2.835 | -0.444 |
| 4 | ` is` | 0.244 | 0.271 | +0.027 |
| 5 | ` that` | 0.343 | 0.536 | +0.193 |
| 6 | ` NASA` | 1.174 | 1.371 | +0.197 |
| 7 | ` will` | 1.279 | 0.977 | -0.302 |
| 8 | ` have` | 1.576 | 1.683 | +0.107 |
| 9 | ` to` | 0.286 | 0.323 | +0.036 |
| 10 | ` buddy` | 14.060 | 14.678 | +0.618 |
| 11 | ` up` | 0.100 | 0.090 | -0.011 |
| 12 | ` with` | 0.665 | 0.387 | -0.278 |
| 13 | ` international` | 7.286 | 5.820 | -1.467 |
| 14 | ` space` | 1.872 | 2.092 | +0.220 |
| 15 | ` programs` | 5.196 | 4.554 | -0.642 |
| 16 | ` a` | 4.873 | 7.011 | +2.138 |
| 17 | ` little` | 0.715 | 0.912 | +0.197 |
| 18 | ` more` | 0.965 | 0.955 | -0.011 |
| 19 | `.` | 1.222 | 1.592 | +0.370 |
| | **MEAN** | **2.833** | **3.004** | **+0.171** |

**perplexity:** base 17.00 → +rationale 20.17 · **mean-diff (the reported Δ) = +0.1710 nats/token**

