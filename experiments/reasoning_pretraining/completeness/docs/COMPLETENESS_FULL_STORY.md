# Completeness test — the full story (untruncated)

Real DCLM docs whose continuation needs a **≥3-step** reasoning chain. For each doc we score the perplexity
of the **target** (a real span of the continuation) under the DCLM-1.4B base judge, given four prefixes:
`context` (base), `context + COMPLETE rationale`, `context + INCOMPLETE rationale` (a load-bearing middle
step deleted), and `context + PLACEBO` (an unrelated doc's complete rationale). Lower NLL = more predictable.

**Aggregate (n=%d):** complete−base +0.048 · **complete−placebo −0.698** (relevant reasoning helps; placebo
hurts +0.745) · **complete−incomplete +0.004** (completeness makes ~no difference — the model fills the gap).

---

## Doc 102770
*NLL of the target: base **2.784** → +complete **2.702** (Δ-0.082) · +incomplete **2.753** (Δ-0.031) · +placebo **3.719** (Δ+0.935)*  ·  completeness effect (complete−incomplete) **-0.051**

### Original document (from DCLM, verbatim)
**Context:**

> I have been taking care of a website that has both a .com and a .co.uk domain.
> The company who I have built this site for can access the site via the .co.uk address but the .com always takes them to their office server where an old version of the site is kept.
> 
> I have no experience with servers other than via a control panel for the company that I have webspace with (1and1).
> It is my understanding that the localhost on the office server has somehow been given a domain name of theukoffice .com and this is what is causing the error.
> 
> Now i know it would be simple just to get them to browse to the .co.uk domain but some of the links here are hard coded as .com rather than using /whatever/wherever.php or ../here/there.php site relative links.

**Continuation (the real text that follows):**

> So pretty soon they end up back on the office server version of the site.
> 
> The server is samba with all the computers running xp or win 2000. How easy would it be to change the office server so it didnt resolve as the .com ? and if this change was made would all the office PC's need re-configuring because of this change
> 
> I know nothing, but learn fast...ish

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> So pretty soon they end up back on the office server version of the site.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The site's internal navigation links are hard-coded as absolute URLs on the .com domain instead of relative paths.
> 2. Inside the office, the .com domain does not point to the public web host, because the office server's localhost was given the name theukoffice.com.
> 3. So any request to the .com domain from an office machine is answered by the local office server, which stores an outdated copy of the site.
> 4. Even a user who deliberately opens the correct .co.uk copy will, on clicking any internal link, have the browser request that link's hard-coded .com address.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The site's internal navigation links are hard-coded as absolute URLs on the .com domain instead of relative paths.
> 4. Even a user who deliberately opens the correct .co.uk copy will, on clicking any internal link, have the browser request that link's hard-coded .com address.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101869, irrelevant to this doc):

> 1. The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
> 2. That financial incentive tempts even a person's closest friends, colleagues, or partners to turn them in if they learn about the cheating.
> 3. The rise of social media has made it far easier for private information to be published publicly where anyone can see it.
> 4. So any incriminating detail a tax cheat shares publicly online becomes visible to exactly those acquaintances who could be tempted by the reward to report them to the IRS.

---

## Doc 101869
*NLL of the target: base **2.866** → +complete **3.001** (Δ+0.136) · +incomplete **2.962** (Δ+0.096) · +placebo **3.423** (Δ+0.557)*  ·  completeness effect (complete−incomplete) **+0.040**

### Original document (from DCLM, verbatim)
**Context:**

> You have a big mouth
> You have a big mouth
> You may think you're a hot shot for pulling a fast one on the IRS. But when the friend you entrusted with your secret snitches on you in exchange for a fat check, you're going to be in big trouble.
> 
> "Most cases start the old-fashioned way," said Ian Comisky, a partner at law firm Blank Rome LLP who represents taxpayers whose returns were flagged by the IRS. "You blab about it to a friend, colleague, spouse or girlfriend, and one of them turns you in."
> 
> Even your closest pals may be tempted to tattle, since the IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
> 
> And with the popularity of social media, it's now much easier to publish private information publicly.

**Continuation (the real text that follows):**

> So if you did something you think was questionable, don't post it all over Facebook.
> 
> Join the Conversation
> Craziest tax deductions

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> So if you did something you think was questionable, don't post it all over Facebook.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
> 2. That financial incentive tempts even a person's closest friends, colleagues, or partners to turn them in if they learn about the cheating.
> 3. The rise of social media has made it far easier for private information to be published publicly where anyone can see it.
> 4. So any incriminating detail a tax cheat shares publicly online becomes visible to exactly those acquaintances who could be tempted by the reward to report them to the IRS.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
> 4. So any incriminating detail a tax cheat shares publicly online becomes visible to exactly those acquaintances who could be tempted by the reward to report them to the IRS.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100989, irrelevant to this doc):

> 1. The hairy ball theorem states that any continuous vector field lying tangent to a sphere must equal zero at at least one point on that sphere.
> 2. The surface of the Earth is, in shape, a sphere.
> 3. At every location on the Earth's surface the wind blows horizontally along the ground, so the wind at each point is a vector lying tangent to the spherical surface; taken together the winds form a tangent vector field on the sphere.
> 4. Wind direction and speed change gradually from one place to the next, so this tangent vector field is continuous.
> 5. A continuous tangent vector field on a sphere meets the theorem's hypothesis, so by the theorem it must be zero at some point — a location where the wind vector has zero magnitude.

---

## Doc 100989
*NLL of the target: base **3.217** → +complete **2.573** (Δ-0.644) · +incomplete **2.821** (Δ-0.396) · +placebo **3.265** (Δ+0.048)*  ·  completeness effect (complete−incomplete) **-0.247**

### Original document (from DCLM, verbatim)
**Context:**

> hairy ball
> 
> (topology)   A result in topology stating that a continuous vector field on a sphere is always zero somewhere. The name comes from the fact that you can't flatten all the hair on a hairy ball, like a tennis ball, there will always be a tuft somewhere (where the tangential projection of the hair is zero). An immediate corollary to this theorem is that for any continuous map f of the sphere into itself there is a point x such that f(x)=x or f(x) is the antipode of x.

**Continuation (the real text that follows):**

> Another corollary is that at any moment somewhere on the Earth there is no wind.
> 
> Last updated: 2002-01-07

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Another corollary is that at any moment somewhere on the Earth there is no wind.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The hairy ball theorem states that any continuous vector field lying tangent to a sphere must equal zero at at least one point on that sphere.
> 2. The surface of the Earth is, in shape, a sphere.
> 3. At every location on the Earth's surface the wind blows horizontally along the ground, so the wind at each point is a vector lying tangent to the spherical surface; taken together the winds form a tangent vector field on the sphere.
> 4. Wind direction and speed change gradually from one place to the next, so this tangent vector field is continuous.
> 5. A continuous tangent vector field on a sphere meets the theorem's hypothesis, so by the theorem it must be zero at some point — a location where the wind vector has zero magnitude.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The hairy ball theorem states that any continuous vector field lying tangent to a sphere must equal zero at at least one point on that sphere.
> 2. The surface of the Earth is, in shape, a sphere.
> 5. A continuous tangent vector field on a sphere meets the theorem's hypothesis, so by the theorem it must be zero at some point — a location where the wind vector has zero magnitude.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101673, irrelevant to this doc):

> 1. A star is an immense ball of gas held together as a single body, so all of its matter exerts gravitational attraction.
> 2. That attraction pulls every part of the star inward toward its center.
> 3. Acting alone, this inward pull would make the star collapse in on itself.
> 4. The core's nuclear fusion, which converts hydrogen to helium, releases enormous energy as heat and radiation that pushes outward.
> 5. That outward push from the core's fusion acts directly against the inward pull.

---

## Doc 101673
*NLL of the target: base **2.868** → +complete **2.903** (Δ+0.035) · +incomplete **2.89** (Δ+0.022) · +placebo **3.436** (Δ+0.568)*  ·  completeness effect (complete−incomplete) **+0.013**

### Original document (from DCLM, verbatim)
**Context:**

> Star goes out with a whimper, not a bang, captured by Hubble telescope
> 
>   |   July 12, 2013 at 5:50 PM
> GREENBELT, Md., July 12 (UPI) -- Not all stars go out with a bang, and NASA has released an image from the Hubble telescope of one going out with more of a cosmic whimper.
> 
> The image shows the planetary nebula IC 289, located in the northern constellation of Cassiopeia. Formerly a star like the sun, it is now just a thin cloud of ionized gas being pushed out into space by the remnants of the star's core, visible as a small bright dot in the middle of the cloud, NASA said Friday.
> 
> Planetary nebulae have nothing to do with planets -- the term is a relic from a time when early observers with small telescopes could only see undefined, hazy objects they thought looked like gaseous planets.
> 
> Stars are powered by nuclear fusion reactions in their cores that convert hydrogen to helium.

**Continuation (the real text that follows):**

> The star remains stable in a process that balances the inward squeeze caused by its gravity with the outward thrust from the inner fusion reaction in its core.
> 
> When all the hydrogen is consumed, as in IC 289, the equilibrium is broken. The gravitational forces become more powerful and crush the star's core into a helium-burning phase that is highly unstable, and eventually blows the whole star's atmosphere away, resulting in the state captured in the Hubble image.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> The star remains stable in a process that balances the inward squeeze caused by its gravity with the outward thrust from the inner fusion reaction in its core.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. A star is an immense ball of gas held together as a single body, so all of its matter exerts gravitational attraction.
> 2. That attraction pulls every part of the star inward toward its center.
> 3. Acting alone, this inward pull would make the star collapse in on itself.
> 4. The core's nuclear fusion, which converts hydrogen to helium, releases enormous energy as heat and radiation that pushes outward.
> 5. That outward push from the core's fusion acts directly against the inward pull.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. A star is an immense ball of gas held together as a single body, so all of its matter exerts gravitational attraction.
> 2. That attraction pulls every part of the star inward toward its center.
> 5. That outward push from the core's fusion acts directly against the inward pull.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100391, irrelevant to this doc):

> 1. NASA said it needs an extra $3 billion per year to go forward with meaningful human space exploration rather than just sending more robots.
> 2. Obama will keep the program alive but only fund it somewhere between NASA's current spending and that full $3 billion increase.
> 3. So NASA will receive less than the $3 billion it asked for, leaving a funding shortfall relative to its stated needs.
> 4. With less money than it says it needs, NASA cannot fully afford meaningful human space exploration on its own budget.
> 5. The only way to still pursue those goals despite the gap is to bring in outside contributors who can share the costs and resources.

---

## Doc 100391
*NLL of the target: base **2.833** → +complete **3.004** (Δ+0.171) · +incomplete **2.977** (Δ+0.144) · +placebo **3.223** (Δ+0.390)*  ·  completeness effect (complete−incomplete) **+0.027**

### Original document (from DCLM, verbatim)
**Context:**

> Good News! NASA Is (Probably) Getting More MoneyS
> 
> NASA needs more money, because let's face it, rocket launches ain't cheap. The good news is, it looks like they'll be getting some. Not as much as they want, but some.
> 
> In October NASA said they would need $3 billion more per year to go forward with meaningful human space exploration, i.e. not just sending more robots up. For a while there were rumors going around that Washington was going to severely scale back the program's budget, but now according to Washington insider John Logsdon, "there will be more money."
> 
> He's also saying that Obama doesn't want to be that president who cuts a future oriented program. So he'll keep it alive, but he'll only give them a budget somewhere between their current spending and the $3 billion per year increase NASA is looking for.

**Continuation (the real text that follows):**

> But all that means is that NASA will have to buddy up with international space programs a little more.
> 
> Let's face it, we weren't going to get to Mars on our own anyway. As long as NASA is still alive, and there's still a remote chance of me seeing a mission to Mars in my lifetime, I'm a happy camper. [New Scientist, image via Matthew Simantov]

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> But all that means is that NASA will have to buddy up with international space programs a little more.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. NASA said it needs an extra $3 billion per year to go forward with meaningful human space exploration rather than just sending more robots.
> 2. Obama will keep the program alive but only fund it somewhere between NASA's current spending and that full $3 billion increase.
> 3. So NASA will receive less than the $3 billion it asked for, leaving a funding shortfall relative to its stated needs.
> 4. With less money than it says it needs, NASA cannot fully afford meaningful human space exploration on its own budget.
> 5. The only way to still pursue those goals despite the gap is to bring in outside contributors who can share the costs and resources.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. NASA said it needs an extra $3 billion per year to go forward with meaningful human space exploration rather than just sending more robots.
> 2. Obama will keep the program alive but only fund it somewhere between NASA's current spending and that full $3 billion increase.
> 5. The only way to still pursue those goals despite the gap is to bring in outside contributors who can share the costs and resources.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101750, irrelevant to this doc):

> 1. The context reports that from 1951 to 1980 the top marginal tax rate was very high (70-91%) and average growth was 3.7%, whereas after the early 1980s the top rate was cut sharply (to 35-39%) and average growth fell to 3%.
> 2. Cutting the top marginal rate after the early 1980s let the highest earners keep a much larger share of their pre-tax income.
> 3. Over that same post-1980 period the top 1%'s share of total income more than doubled (from 10% to over 20%) — exactly the concentration of income at the top that the context says drained the middle class of purchasing power and produced the long-term slowdown (the 'trend').
> 4. Because the top-rate cuts directly enabled and accelerated that concentration of income at the top, the tax changes pushed in the same direction as the trend rather than counteracting it.

---

## Doc 101750
*NLL of the target: base **4.854** → +complete **4.648** (Δ-0.205) · +incomplete **4.967** (Δ+0.113) · +placebo **6.497** (Δ+1.643)*  ·  completeness effect (complete−incomplete) **-0.318**

### Original document (from DCLM, verbatim)
**Context:**

> Alan Reynolds ("Memo to Robert Reich: Why 70% Tax Rates Won't Work," op-ed, June 16) distorts my proposal and ignores my argument. I proposed a top marginal income tax rate of 70% only on incomes of more than $15 million. Under my plan, incomes between $5 million and $15 million would be subjected to a 60% rate, and incomes between $500,000 and $5 million to a 50% rate. I further proposed substantial rate reductions for people with incomes under $100,000.
> 
> My argument, which Mr. Reynolds doesn't rebut: During the almost three decades spanning 1951 to 1980, when the top rate was between 70% and 91%, average annual growth in the American economy was 3.7%. Between 1983 and the start of the "great recession," when the top rate ranged between 35% and 39%, average growth was 3%. The long-term slowdown is related to the fact that since the early 1980s a larger and larger share of total income has gone to the top (the richest 1% of Americans got 10% of total income in 1980, and get more than 20% now), leaving the vast middle class with insufficient purchasing power to boost the economy without eventually going deep into debt.

**Continuation (the real text that follows):**

> Tax rates exacerbated the trend. Giving the middle class more purchasing power by lowering its rates while raising the rates at the top will help spur growth, to the benefit of all. Top earners will do better with a smaller share of a more rapidly growing economy than with a larger share of a slower-growing one.
> 
> Robert B. Reich
> 
> Chancellor's Professor of Public Policy
> 
> University of California at Berkeley
> 
> Berkeley, Calif.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Tax rates exacerbated the trend.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The context reports that from 1951 to 1980 the top marginal tax rate was very high (70-91%) and average growth was 3.7%, whereas after the early 1980s the top rate was cut sharply (to 35-39%) and average growth fell to 3%.
> 2. Cutting the top marginal rate after the early 1980s let the highest earners keep a much larger share of their pre-tax income.
> 3. Over that same post-1980 period the top 1%'s share of total income more than doubled (from 10% to over 20%) — exactly the concentration of income at the top that the context says drained the middle class of purchasing power and produced the long-term slowdown (the 'trend').
> 4. Because the top-rate cuts directly enabled and accelerated that concentration of income at the top, the tax changes pushed in the same direction as the trend rather than counteracting it.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The context reports that from 1951 to 1980 the top marginal tax rate was very high (70-91%) and average growth was 3.7%, whereas after the early 1980s the top rate was cut sharply (to 35-39%) and average growth fell to 3%, while the top 1%'s income share more than doubled (from 10% to over 20%).
> 4. Because the top-rate cuts pushed in the same direction as the trend rather than counteracting it...

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101966, irrelevant to this doc):

> 1. An obstruction blocks the opening of the appendix, sealing its cavity so its contents cannot drain out.
> 2. Bacteria trapped inside the sealed appendix multiply rapidly, generating pus and gas that keep accumulating with nowhere to go.
> 3. Because the sealed cavity cannot vent, this accumulation drives the internal pressure steadily higher and compresses the blood vessels in the appendix wall, cutting off its blood supply.
> 4. Deprived of blood and stretched thin by the mounting internal pressure, the appendix wall becomes progressively more fragile.

---

## Doc 101966
*NLL of the target: base **1.928** → +complete **1.987** (Δ+0.060) · +incomplete **1.961** (Δ+0.033) · +placebo **4.202** (Δ+2.274)*  ·  completeness effect (complete−incomplete) **+0.026**

### Original document (from DCLM, verbatim)
**Context:**

> The cause of appendicitis isn't always clear. Sometimes appendicitis can occur as a result of:
> 
>   • An obstruction. Food waste or a hard piece of stool (fecal stone) can block the opening of the cavity that runs the length of your appendix.
>   • An infection. Appendicitis may also follow an infection, such as a gastrointestinal viral infection, or it may result from other types of inflammation.
> 
> In both cases, bacteria inside the appendix multiply rapidly, causing the appendix to become inflamed, swollen and filled with pus.

**Continuation (the real text that follows):**

> If not treated promptly, the appendix can rupture.
> 
> Aug. 13, 2011

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> If not treated promptly, the appendix can rupture.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. An obstruction blocks the opening of the appendix, sealing its cavity so its contents cannot drain out.
> 2. Bacteria trapped inside the sealed appendix multiply rapidly, generating pus and gas that keep accumulating with nowhere to go.
> 3. Because the sealed cavity cannot vent, this accumulation drives the internal pressure steadily higher and compresses the blood vessels in the appendix wall, cutting off its blood supply.
> 4. Deprived of blood and stretched thin by the mounting internal pressure, the appendix wall becomes progressively more fragile.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. An obstruction blocks the opening of the appendix, sealing its cavity so its contents cannot drain out.
> 2. Bacteria trapped inside the sealed appendix multiply rapidly, generating pus and gas that keep accumulating with nowhere to go.
> 4. Deprived of blood and stretched thin by the mounting internal pressure, the appendix wall becomes progressively more fragile.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101991, irrelevant to this doc):

> 1. Right now artists can distribute their music through cheap, open online channels, keeping most of the revenue and cutting the RIAA and other middlemen out of the equation.
> 2. A proprietary DRM format is not free to use: it is owned and controlled by whatever company holds its rights, and using it requires a license from that owner.
> 3. If a law mandated one specific DRM format as the only legal way to distribute music online, then every distributor, including artists releasing directly, would be forced to use that format.
> 4. To use the mandated format they would all have to obtain and pay for a license from the company that owns it.
> 5. That rights-holding company would therefore sit between the artists and their audience as a gatekeeper everyone is legally required to go through and pay.

---

## Doc 101991
*NLL of the target: base **3.057** → +complete **2.805** (Δ-0.252) · +incomplete **3.019** (Δ-0.038) · +placebo **3.465** (Δ+0.408)*  ·  completeness effect (complete−incomplete) **-0.214**

### Original document (from DCLM, verbatim)
**Context:**

> Forgot your password?
> 
> Comment: this can backfire (Score 1) 503
> 
> by spatenbrau (#15215649) Attached to: Senate Bill May Ban Streaming MP3s
> 
> What is to prevent the OSS community from making a more restrictive DRM standard based on ogg vorbis with some DRM-ish layer? Does this mean that the only legal streaming format will then be ogg-DRM-vorbis?
> 
> The RIAA and the other middlemen must really be worried that they are going to be cut out of the equation when the artists realise that they don't need to give up 99% of the revenues and could just as easily hire an online company to distribute their works for them at a much lower cost.

**Continuation (the real text that follows):**

> Legislating a certain format for the online distribution of music would turn the tables again and force the artists to deal with another middleman, in this case the company that owns the rights to that DRM format. The RIAA could simply buy those rights to that particular DRM and they would be guarenteed a revenue stream for quite a few years into the future.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Legislating a certain format for the online distribution of music would turn the tables again and force the artists to deal with another middleman

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Right now artists can distribute their music through cheap, open online channels, keeping most of the revenue and cutting the RIAA and other middlemen out of the equation.
> 2. A proprietary DRM format is not free to use: it is owned and controlled by whatever company holds its rights, and using it requires a license from that owner.
> 3. If a law mandated one specific DRM format as the only legal way to distribute music online, then every distributor, including artists releasing directly, would be forced to use that format.
> 4. To use the mandated format they would all have to obtain and pay for a license from the company that owns it.
> 5. That rights-holding company would therefore sit between the artists and their audience as a gatekeeper everyone is legally required to go through and pay.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Right now artists can distribute their music through cheap, open online channels, keeping most of the revenue and cutting the RIAA and other middlemen out of the equation.
> 2. A proprietary DRM format is not free to use: it is owned and controlled by whatever company holds its rights, and using it requires a license from that owner.
> 5. That rights-holding company would therefore sit between the artists and their audience as a gatekeeper everyone is legally required to go through and pay.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101165, irrelevant to this doc):

> 1. Culligan's charge -- that a longtime Clinton staffer used Clinton's taxpayer-funded office to receive a letter about her personal stock options -- is described as 'the most obscure imaginable charge,' a trivial matter.
> 2. The context notes it 'would have been a simple thing for the Clinton camp to brush off the charge as irrelevant,' so if the charge genuinely did not bother Clinton, the natural and effortless response would be to ignore it entirely and take no action.
> 3. Culligan registered the Clinton domain names back in the late '90s and had openly kept them as a joke for over a decade, during which time the Clinton camp took no action against them.
> 4. Yet Clinton's lawyer is pursuing legal action to reclaim those domains only now, precisely as Culligan is pressing the Williams charge -- and Culligan characterizes this move as 'payback.'
> 5. Choosing to launch retaliatory legal action now, instead of ignoring the charge, is the opposite of the effortless brush-off one would expect if the charge were truly irrelevant.

---

## Doc 101165
### Original document (from DCLM, verbatim)
**Context:**

> Bill Clinton Wants His Domain Names Back
> 
> In the late '90s, private investigator Joe Culligan registered and other Clintonesque domain names as a joke. Now Bill Clinton's lawyer is pursuing legal action to get the website addresses. It's payback, says Culligan.
> 
> For months, Culligan has been digging into the mystery of why Maggie Williams, a longtime Clinton staffer who served as Hillary Clinton's campaign manager and now works for her as a Secretary of State recruiter, used Clinton's taxpayer-funded office to receive correspondence about stock options she received from Delta Financial, a subprime lender.
> 
> It's the most obscure imaginable charge. What, does Culligan think Clinton ripped off taxpayers by having a government-paid clerk drop the letter off at Williams's desk? It's hardly a scandal compared to the $1 million-a-year bill the government has paid since 2001 to fund Clinton's post-presidential operation.
> 
> It would have been a simple thing for the Clinton camp to brush off the charge as irrelevant.

**Continuation (the real text that follows):**

> But the move to reclaim Clinton's domain names suggests that the charge has stung nonetheless. What is it about Williams's mailing address that has Clinton's lawyers so worried now — as opposed to any point in the past decade, during which time Culligan pointed,, and as a gag to the Republican National Committee's website?
> 
> (Photo by AP)

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> But the move to reclaim Clinton's domain names suggests that the charge has stung nonetheless.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Culligan's charge -- that a longtime Clinton staffer used Clinton's taxpayer-funded office to receive a letter about her personal stock options -- is described as 'the most obscure imaginable charge,' a trivial matter.
> 2. The context notes it 'would have been a simple thing for the Clinton camp to brush off the charge as irrelevant,' so if the charge genuinely did not bother Clinton, the natural and effortless response would be to ignore it entirely and take no action.
> 3. Culligan registered the Clinton domain names back in the late '90s and had openly kept them as a joke for over a decade, during which time the Clinton camp took no action against them.
> 4. Yet Clinton's lawyer is pursuing legal action to reclaim those domains only now, precisely as Culligan is pressing the Williams charge -- and Culligan characterizes this move as 'payback.'
> 5. Choosing to launch retaliatory legal action now, instead of ignoring the charge, is the opposite of the effortless brush-off one would expect if the charge were truly irrelevant.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Culligan's charge -- that a longtime Clinton staffer used Clinton's taxpayer-funded office to receive a letter about her personal stock options -- is described as 'the most obscure imaginable charge,' a trivial matter.
> 2. The context notes it 'would have been a simple thing for the Clinton camp to brush off the charge as irrelevant,' so if the charge genuinely did not bother Clinton, the natural and effortless response would be to ignore it entirely and take no action.
> 5. Choosing to launch retaliatory legal action now, instead of ignoring the charge, is the opposite of the effortless brush-off one would expect if the charge were truly irrelevant.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102695, irrelevant to this doc):

> 1. GLAAD's Network Responsibility Index shows the networks with the youngest audiences (the CW, ABC Family, Fox) carry the highest percentage of gay and lesbian characters, while the networks that skew oldest (CBS, USA, A&E, TBS) carry the least.
> 2. A network tailors its programming to the tastes of the demographic it targets.
> 3. Therefore the heavy gay content on the youth-oriented networks reflects that young viewers genuinely like and accept gay characters.
> 4. The media tastes and social attitudes a generation forms in its youth tend to persist as that generation ages rather than reverse.

---

## Doc 102695
*NLL of the target: base **2.88** → +complete **2.929** (Δ+0.050) · +incomplete **2.933** (Δ+0.053) · +placebo **3.119** (Δ+0.239)*  ·  completeness effect (complete−incomplete) **-0.003**

### Original document (from DCLM, verbatim)
**Context:**

> Only Teens Want to See Gay People on TV
> 
> GLAAD, the spayed and neutered gay media watchdog, released its annual Network Responsibility Index, which charts the gay content of television networks. Guess what they found? The highest percentage of programming with gay and lesbian characters is targeted at young audiences.
> 
> For the second year running, the barely post-pubescent CW had the highest percentage of primetime content with gay, lesbian, bisexual, transgender, or otherwise fruity characters. Most of these hours of television were on 90210, Gossip Gurrrlll, and America's Next Top. Shockingly enough, Fox came in second, mostly thanks to Glee, which is gayer than Cristiano Ronaldo's underwear drawer. Still the network has gay characters on many of its shows including American Dad and, appropriately, Bones.
> 
> The rankings for the networks didn't change at all, with ABC, NBC, and CBS coming in third fourth and fifth for the second consecutive year, but NBC and CBS did have a slightly higher percentage of gay content this year than last. The 3% hike in gay and lesbian "impressions" on CBS was enough for its grade to go from "failing" to "adequate."
> 
> Of all the cable networks ranked, number one, by a wide margin, was your high school-aged nieces' and Richard Lawson's favorite channel: ABC Family. Of it's original prime-time 55% of it had gays in it. Yes, Greek, Pretty Little Liars, Make It or Break It and all the other embarrassments on your DVR all have gay characters.
> 
> Meanwhile those that scored the lowest on the Index—CBS, USA, A&E, and TBS—all have audiences that skew older than their more gay-minded counterparts.

**Continuation (the real text that follows):**

> What does that mean? Well, what we knew all along—that the kids love the gays and that as they grow up, the entertainment aimed at their generation will probably continue to have just as many gays as you would want to see in the mainstream media. As for CBS and the rest, they'll eventually retire, live off their pensions, and yell at same-sex couples from their front porches until they eventually die and gays come in to gussy up the house and make it livable. The gay takeover is inevitable, so will you let us enjoy The Good Wife in our complacency while it happens?

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> What does that mean? Well, what we knew all along—that the kids love the gays and that as they grow up, the entertainment aimed at their generation will probably continue to have just as many gays as you would want to see in the mainstream media.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. GLAAD's Network Responsibility Index shows the networks with the youngest audiences (the CW, ABC Family, Fox) carry the highest percentage of gay and lesbian characters, while the networks that skew oldest (CBS, USA, A&E, TBS) carry the least.
> 2. A network tailors its programming to the tastes of the demographic it targets.
> 3. Therefore the heavy gay content on the youth-oriented networks reflects that young viewers genuinely like and accept gay characters.
> 4. The media tastes and social attitudes a generation forms in its youth tend to persist as that generation ages rather than reverse.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. GLAAD's Network Responsibility Index shows the networks with the youngest audiences (the CW, ABC Family, Fox) carry the highest percentage of gay and lesbian characters, while the networks that skew oldest (CBS, USA, A&E, TBS) carry the least.
> 4. The media tastes and social attitudes a generation forms in its youth tend to persist as that generation ages rather than reverse.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101807, irrelevant to this doc):

> 1. As originally drafted, the bill's whole purpose was to criminalize giving "material support" to Sharia, treating Islamic legal codes themselves as advocating violence and as a threat to the state and federal constitutions.
> 2. Under heavy public pressure, and after the sponsor met with Muslim leaders, the bill was rewritten to strike out all the language that painted Sharia as violent or as a constitutional threat.
> 3. The rewrite also explicitly declares that peaceful religious practice is not a violation, so the religion itself is no longer the thing being outlawed.
> 4. Once every reference to Sharia and religion is removed, the only operative content left from the original bill is its bare "material support" prohibition, with no religious object attached.
> 5. A "material support" prohibition stripped of any religious object reverts to punishing the ordinary category of dangerous conduct that such statutes are written to target.

---

## Doc 101807
*NLL of the target: base **3.083** → +complete **3.051** (Δ-0.032) · +incomplete **3.03** (Δ-0.053) · +placebo **3.603** (Δ+0.520)*  ·  completeness effect (complete−incomplete) **+0.021**

### Original document (from DCLM, verbatim)
**Context:**

> Tennessee Scraps Sharia References From Anti-Sharia Bill
> 
> | Thu Mar. 24, 2011 8:05 AM GMT
> 
> A few weeks back we told you about an extreme new bill proposed in Tennessee that defined Islamic law as prima facie treasonous, and made "material support" for Sharia punishable by 15 years in prison. That's a pretty harsh sentence for a constitutionally protected freedom, to be sure, but that was kind of the point. The bill, drafted by an Arizona-based attorney who'd once called for all Muslim non-citizens to be deported, went beyond warnings about some future invasion of Islamic extremists, and instead took on a core tenet of the religion itself.
> 
> In this case, at least, massive public pressure seems to have had an effect. After meeting with Muslim leaders, the bill's co-sponsor, Republican State Sen. Bill Ketron, submitted new language that sort of addresses the problem. From The Tennessean:
> 
> The new version removes language that described Shariah—the Islamic legal codes that cover everything from the rules of warfare to prayer and diet—as advocating violence and a threat to the United States and Tennessee constitutions. The change makes clear that peaceful religious practices would not be considered a violation, the bill's sponsors said in a statement.
> 
> The Council on American–Islamic Relations had promised to file a lawsuit to block the implementation if the bill became law, but now that the overt religious references have been removed, that becomes a lot less likely.

**Continuation (the real text that follows):**

> Basically, the bill has been converted into a fairly straightforward law concerning material support for terrorism. Of course, the federal government already has a material support for terrorism law—and a quite expansive one at that—so it's not entirely clear why Tennessee needs its own. Stay tuned next week when the Tennessee state legislature authorizes a no-fly zone over Libya.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Basically, the bill has been converted into a fairly straightforward law concerning material support for terrorism.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. As originally drafted, the bill's whole purpose was to criminalize giving "material support" to Sharia, treating Islamic legal codes themselves as advocating violence and as a threat to the state and federal constitutions.
> 2. Under heavy public pressure, and after the sponsor met with Muslim leaders, the bill was rewritten to strike out all the language that painted Sharia as violent or as a constitutional threat.
> 3. The rewrite also explicitly declares that peaceful religious practice is not a violation, so the religion itself is no longer the thing being outlawed.
> 4. Once every reference to Sharia and religion is removed, the only operative content left from the original bill is its bare "material support" prohibition, with no religious object attached.
> 5. A "material support" prohibition stripped of any religious object reverts to punishing the ordinary category of dangerous conduct that such statutes are written to target.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. As originally drafted, the bill's whole purpose was to criminalize giving "material support" to Sharia, treating Islamic legal codes themselves as advocating violence and as a threat to the state and federal constitutions.
> 2. Under heavy public pressure, and after the sponsor met with Muslim leaders, the bill was rewritten to strike out all the language that painted Sharia as violent or as a constitutional threat.
> 5. A "material support" prohibition stripped of any religious object reverts to punishing the ordinary category of dangerous conduct that such statutes are written to target.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102891, irrelevant to this doc):

> 1. Frame compatible 3D was designed to run over existing HD equipment, so an old AV receiver's hardware is physically capable of carrying the signal.
> 2. Therefore older receivers being blocked from passing DirecTV's 3D cannot be explained by a technical or hardware limitation.
> 3. RealD owns the patent on frame compatible 3D formats, and a device only appears on the list of supported devices (its EDID recognized) if its manufacturer pays RealD to license that patent.
> 4. So whether a given receiver is allowed to display the 3D signal is decided by whether its maker has licensed RealD, and unlicensed devices are precisely the ones left off the supported list.
> 5. Since the exclusion tracks licensing status rather than which receiver you happen to own, the real function of the block is to compel manufacturers into licensing RealD's patents.

---

## Doc 102891
*NLL of the target: base **2.323** → +complete **1.94** (Δ-0.383) · +incomplete **2.056** (Δ-0.267) · +placebo **2.719** (Δ+0.396)*  ·  completeness effect (complete−incomplete) **-0.116**

### Original document (from DCLM, verbatim)
**Context:**

> RealD logoFollow this industry and you learn to accept that manufacturers are only motivated to add new features to new products, but when we first heard that DirecTV's 3D signal wouldn't let you pass frame compatible 3D through older AV receivers, we were scratching our heads. Sure the receiver never claimed to be 3D compatible, but the entire point of using frame compatible 3D instead of doubling the HD signal for 3D like Blu-ray is so the signal can be transmitted via existing HD equipment. So while DirecTV gets away with making minimal changes to its infrastructure, you have to replace just about everything you own. What was a mystery, is now crystal clear and of course its always about money, it isn't necessarily about DirecTV's money, this time. You see RealD owns the patent on frame compatible 3D formats like side by side, and if a display or receiver manufacturer wants its EDID on the list of supported devices, they have to pay for that right.

**Continuation (the real text that follows):**

> So it isn't that DirecTV wants to prevent you from using your old receiver as much as it is about preventing those who don't license RealD's patents from being able to display 3D. Nice huh, but no one ever said it was about the customer.
> 
> 
> Follow the money; the real reason why your AVR doesn't support DirecTV's 3D

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> So it isn't that DirecTV wants to prevent you from using your old receiver as much as it is about preventing those who don't license RealD's patents from being able to display 3D.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Frame compatible 3D was designed to run over existing HD equipment, so an old AV receiver's hardware is physically capable of carrying the signal.
> 2. Therefore older receivers being blocked from passing DirecTV's 3D cannot be explained by a technical or hardware limitation.
> 3. RealD owns the patent on frame compatible 3D formats, and a device only appears on the list of supported devices (its EDID recognized) if its manufacturer pays RealD to license that patent.
> 4. So whether a given receiver is allowed to display the 3D signal is decided by whether its maker has licensed RealD, and unlicensed devices are precisely the ones left off the supported list.
> 5. Since the exclusion tracks licensing status rather than which receiver you happen to own, the real function of the block is to compel manufacturers into licensing RealD's patents.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Frame compatible 3D was designed to run over existing HD equipment, so an old AV receiver's hardware is physically capable of carrying the signal.
> 2. Therefore older receivers being blocked from passing DirecTV's 3D cannot be explained by a technical or hardware limitation.
> 5. Since the exclusion tracks licensing status rather than which receiver you happen to own, the real function of the block is to compel manufacturers into licensing RealD's patents.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100293, irrelevant to this doc):

> 1. The HRR sends out a short pulse of radar energy into the target area.
> 2. This energy is electromagnetic, so it travels at a fixed, known speed that does not change with distance.
> 3. The pulse travels out to an object, reflects off it, and returns to the sensor, so it covers a round-trip path equal to twice the object's range.
> 4. Because the speed is constant, the distance covered equals that speed multiplied by the elapsed time, so a longer elapsed time means the pulse traveled a longer round-trip distance.
> 5. Since that round-trip distance is exactly twice the range, a longer elapsed time corresponds directly to a greater object range.

---

## Doc 100293
*NLL of the target: base **2.567** → +complete **2.628** (Δ+0.062) · +incomplete **2.606** (Δ+0.039) · +placebo **3.369** (Δ+0.802)*  ·  completeness effect (complete−incomplete) **+0.022**

### Original document (from DCLM, verbatim)
**Context:**

> Tyco Electronics Corp. unveils its High Resolution Radar (HRR) for the next generation of "Smart Bumpers."
> 
> Tyco says its system offers significant advantages over ultrasonic sensing systems, including greater range, higher resolution between objects and important styling advantages.
> 
> HRR works by transmitting a short pulse into a desired area. This energy is reflected off objects within 66 ft. (20.1 m) and returns.

**Continuation (the real text that follows):**

> The travel time of the signal determines the range of the object.
> 
> The HRR system can "see through" plastic fascias commonly used on vehicles, eliminating the need for stylists to contend with unsightly external ultrasonic sensors.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> The travel time of the signal determines the range of the object.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The HRR sends out a short pulse of radar energy into the target area.
> 2. This energy is electromagnetic, so it travels at a fixed, known speed that does not change with distance.
> 3. The pulse travels out to an object, reflects off it, and returns to the sensor, so it covers a round-trip path equal to twice the object's range.
> 4. Because the speed is constant, the distance covered equals that speed multiplied by the elapsed time, so a longer elapsed time means the pulse traveled a longer round-trip distance.
> 5. Since that round-trip distance is exactly twice the range, a longer elapsed time corresponds directly to a greater object range.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The HRR sends out a short pulse of radar energy into the target area.
> 3. The pulse travels out to an object, reflects off it, and returns to the sensor, so it covers a round-trip path equal to twice the object's range.
> 5. Since that round-trip distance is exactly twice the range, a longer elapsed time corresponds directly to a greater object range.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101013, irrelevant to this doc):

> 1. His father takes medication for type 2 diabetes to keep his blood glucose within a safe range, so the most likely immediate consequence of missing a dose is that his blood glucose rises.
> 2. A high blood glucose level can itself produce vomiting, dehydration, weakness, and a general feeling of being sick.
> 3. The father felt sick and vomited the same night he skipped his medicine, and those symptoms match the known effects of high blood glucose, so his episode was most likely driven by elevated glucose rather than being harmless.
> 4. His glucose may also have been running high for several days before he ever missed the dose, which means his glucose is not being kept reliably in range even under his normal treatment.
> 5. Because it is uncontrolled high glucose that produces these dangerous symptoms, avoiding future episodes depends on keeping his glucose consistently within a safe range.

---

## Doc 101013
*NLL of the target: base **1.992** → +complete **2.093** (Δ+0.101) · +incomplete **2.041** (Δ+0.049) · +placebo **2.737** (Δ+0.745)*  ·  completeness effect (complete−incomplete) **+0.052**

### Original document (from DCLM, verbatim)
**Context:**

> My dad has type 2 diabetes. He recently forgot to take his medication for one day, and later that night he felt sick, with vomiting and a slight fever. He was fine after a little rest, but could this have been a reaction to missing his medicine? If it happens again, what should he (we) do?
> 
> — Amanda, Trinidad
> 
> I understand your concern, but this is a difficult question to answer without knowing details specific to your father's case, so I'll give you a general answer. The most likely immediate consequence of missing medicines for diabetes is a high glucose (sugar) level. If the level is very high, it causes frequent and copious urination, thirst, and hunger. Vomiting, dehydration, weakness, and confusion can also be caused by high glucose levels. If the glucose level has been high for a while, the individual might experience weight loss.
> 
> It is possible that your father's glucose was on the high side for several days before he missed that one day's medicine.

**Continuation (the real text that follows):**

> Therefore, it is important to make sure that his blood glucose is adequately controlled. When your father feels sick, check his blood glucose level. If it is high, make sure he remains well hydrated and seek the advice of his doctor for instructions on medicines.
> 
> Learn more in the Everyday Health Type 2 Diabetes Center.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Therefore, it is important to make sure that his blood glucose is adequately controlled.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. His father takes medication for type 2 diabetes to keep his blood glucose within a safe range, so the most likely immediate consequence of missing a dose is that his blood glucose rises.
> 2. A high blood glucose level can itself produce vomiting, dehydration, weakness, and a general feeling of being sick.
> 3. The father felt sick and vomited the same night he skipped his medicine, and those symptoms match the known effects of high blood glucose, so his episode was most likely driven by elevated glucose rather than being harmless.
> 4. His glucose may also have been running high for several days before he ever missed the dose, which means his glucose is not being kept reliably in range even under his normal treatment.
> 5. Because it is uncontrolled high glucose that produces these dangerous symptoms, avoiding future episodes depends on keeping his glucose consistently within a safe range.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. His father takes medication for type 2 diabetes to keep his blood glucose within a safe range, so the most likely immediate consequence of missing a dose is that his blood glucose rises.
> 2. A high blood glucose level can itself produce vomiting, dehydration, weakness, and a general feeling of being sick.
> 5. Because it is uncontrolled high glucose that produces these dangerous symptoms, avoiding future episodes depends on keeping his glucose consistently within a safe range.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102626, irrelevant to this doc):

> 1. Glutamate is the principal excitatory neurotransmitter, and its activity is normally balanced by inhibitory neurotransmitters; disrupting that balance produces abnormal levels of neural activity.
> 2. Mutations in genes involved in glutamate neurotransmission could cause exactly this imbalance and therefore raise the risk of autism — this is the hypothesis being tested.
> 3. If that hypothesis is true, then autistic individuals should carry these disease-causing mutations in the glutamate genes at a higher rate than people without autism.
> 4. Such causal mutations would show up as rare DNA sequence variants when the 38 glutamate-related genes are sequenced in the high-throughput screen.

---

## Doc 102626
*NLL of the target: base **2.32** → +complete **2.231** (Δ-0.089) · +incomplete **2.226** (Δ-0.094) · +placebo **2.526** (Δ+0.206)*  ·  completeness effect (complete−incomplete) **+0.005**

### Original document (from DCLM, verbatim)
**Context:**

> Skip navigation
> 
> Calls to Action
> 
> Understanding glutamate signaling defects in autism spectrum disorders
> 
> State/Province Full: 
> United States
> 
> A disruption of the balance between the activity of excitatory neurotransmitters (which increase neural activity) and inhibitory neurotransmitters (which decrease neural activity) may be involved in autism. The principal excitatory neurotransmitter, glutamate, is counteracted by the activity of inhibitory neurotransmitters. Disregulation of either neurotransmitter system results in abnormal levels of neural activity. Mutations in genes involved in glutamate neurotransmission could lead to this kind of imbalance, and therefore confer a risk of autism. In the present study, Dr. Wang and colleagues will conduct a high-throughput genetic screen in a cohort of autistic patients, looking for DNA sequence variants in 38 genes known to be involved in glutamate neurotransmission.

**Continuation (the real text that follows):**

> If the hypothesis that glutamate is involved in autism is correct, they expect to find multiple rare sequence variants of these genes in autistic patients, compared to a non-autistic control sample. This research may clarify the role of glutamate in autism, as well as identify new genetic risk factors and potential drug targets for the treatment and prevention of autism.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> If the hypothesis that glutamate is involved in autism is correct, they expect to find multiple rare sequence variants of these genes in autistic patients, compared to a non-autistic control sample.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Glutamate is the principal excitatory neurotransmitter, and its activity is normally balanced by inhibitory neurotransmitters; disrupting that balance produces abnormal levels of neural activity.
> 2. Mutations in genes involved in glutamate neurotransmission could cause exactly this imbalance and therefore raise the risk of autism — this is the hypothesis being tested.
> 3. If that hypothesis is true, then autistic individuals should carry these disease-causing mutations in the glutamate genes at a higher rate than people without autism.
> 4. Such causal mutations would show up as rare DNA sequence variants when the 38 glutamate-related genes are sequenced in the high-throughput screen.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Glutamate is the principal excitatory neurotransmitter, and its activity is normally balanced by inhibitory neurotransmitters; disrupting that balance produces abnormal levels of neural activity.
> 2. Mutations in genes involved in glutamate neurotransmission could cause exactly this imbalance and therefore raise the risk of autism — this is the hypothesis being tested.
> 4. Such causal mutations would show up as rare DNA sequence variants when the 38 glutamate-related genes are sequenced in the high-throughput screen.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102659, irrelevant to this doc):

> 1. Transferring the ketchup from one bottle to another introduced live microbes into the counterfeit product.
> 2. Unlike a legitimate bottling operation, no heat step was applied to kill them, so those microbes stayed alive inside the ketchup.
> 3. Unlike restaurant bottles that are opened and used up frequently, these bottles sat sealed and undisturbed, so nothing inside could vent.
> 4. Alive and sealed in, the microbes feed on the ketchup, and that metabolism releases gas as a byproduct.
> 5. Because the bottle is sealed, the gas cannot escape and steadily accumulates in the fixed container volume.

---

## Doc 102659
*NLL of the target: base **3.229** → +complete **2.982** (Δ-0.247) · +incomplete **3.092** (Δ-0.137) · +placebo **3.532** (Δ+0.303)*  ·  completeness effect (complete−incomplete) **-0.110**

### Original document (from DCLM, verbatim)
**Context:**

> Why Counterfeit Ketchup Is Exploding in New Jersey
> 
> In a warehouse in New Jersey, some messy, weird stuff went down. There was an explosion, but not just any explosion: a counterfeit ketchup explosion. No, nobody was trying to sabotage the illicit shipment, there's some science behind it.
> 
> Admittedly there are some other questions here? Why counterfeit ketchup? The plan was to put cheap, standard Heinz into bottle for the more expensive "Simply Heinz" and sell it at a profit. Why mess with bottle switching instead of counterfeiting labels? Who knows. Why are they exploding, on the other hand, is a little simpler to explain.
> 
> By transferring the condiment from one bottle to another, the hapless counterfeiters opened the ketchup up to contamination by all kinds of microbes. But how is that any different from standard bottling? Well in a legitimate bottling operation, you kill those microbes with heat first. And as for why unrefrigerated bottles of ketchup don't explode on restaurant tables, they're being opened frequently, and hopefully used up and replaced.
> 
> The counterfeit ketchup sitting in New Jersey, however, had none of these things going for it.

**Continuation (the real text that follows):**

> So when the microbes started chowing down and spewing out gas, the pressure built up and boom: spontaneous ketchup explosion. Weird as it is, it's a way better ending than if people got sick from the tainted product. It's not known if any made it onto shelves, but if so, it's probably destined to blow as well. So be glad those microrganisms are just making weird ketchup bombs instead of htiching a ride on your next plate of fries. And make sure you're on your toes in the condiment aisle. [Livescience]
> 
> Image by Picsfive/Shutterstock

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> So when the microbes started chowing down and spewing out gas, the pressure built up and boom: spontaneous ketchup explosion.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Transferring the ketchup from one bottle to another introduced live microbes into the counterfeit product.
> 2. Unlike a legitimate bottling operation, no heat step was applied to kill them, so those microbes stayed alive inside the ketchup.
> 3. Unlike restaurant bottles that are opened and used up frequently, these bottles sat sealed and undisturbed, so nothing inside could vent.
> 4. Alive and sealed in, the microbes feed on the ketchup, and that metabolism releases gas as a byproduct.
> 5. Because the bottle is sealed, the gas cannot escape and steadily accumulates in the fixed container volume.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Transferring the ketchup from one bottle to another introduced live microbes into the counterfeit product.
> 2. Unlike a legitimate bottling operation, no heat step was applied to kill them, so those microbes stayed alive inside the ketchup.
> 3. Unlike restaurant bottles that are opened and used up frequently, these bottles sat sealed and undisturbed, so nothing inside could vent.
> 5. Because the bottle is sealed, the gas cannot escape and steadily accumulates in the fixed container volume.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102075, irrelevant to this doc):

> 1. Japan's goal, when it began sending troops overseas, was to look like every other country that provides peacekeepers, so as to blend in and avoid criticism.
> 2. To stay within the pacifist restraints of its constitution, the 1992 law imposed tight restrictions on which kinds of missions the SDF is allowed to join.
> 3. No other peacekeeping nation places such constitution-driven limits on the missions its soldiers may take part in.
> 4. Operating under these unusual restrictions forces the SDF to behave visibly differently from every other national contingent on the ground.
> 5. Behaving differently from everyone else is the exact opposite of blending in and looking like the others.

---

## Doc 102075
*NLL of the target: base **3.133** → +complete **3.226** (Δ+0.093) · +incomplete **3.185** (Δ+0.052) · +placebo **3.953** (Δ+0.820)*  ·  completeness effect (complete−incomplete) **+0.041**

### Original document (from DCLM, verbatim)
**Context:**

> PACIFISTS can be better than military superpowers at being unilateralist. Junichiro Koizumi is counting on this as he updates, yet again, the role of Japan's Self Defence Force (SDF). Following promises made at the G8 summit in Georgia last week, the prime minister is seeking to redefine the SDF's mission in Iraq, and make it part of a new multinational force that was called for in last week's United Nations resolution.
> 
> The trouble with the new multinational outfit is that its members might have to defend people by firing weapons. Agreeing to this knowingly would further stretch, if not snap outright, Japan's self-imposed limits on using force. There is little risk of this in the SDF's current Iraq mission, which involves giving humanitarian help to the quiet southern town of Samawah. So Mr Koizumi announced a simple solution on June 15th. The 550 Japanese troops in Samawah will call themselves part of the multinational force, but carry on as before—operating in “non-combat” zones—and will take orders from nobody.
> 
> When its troops started venturing overseas more than a decade ago, Japan's goal was to look more like all of the other countries that provide peacekeepers, and thereby to avoid criticism. Yet the 1992 law authorising the SDF to deploy abroad laid down tight restrictions on which sorts of missions it could join, to avoid violating the pacifist restraints in Japan's constitution.

**Continuation (the real text that follows):**

> As a result, Japanese troops still stand out awkwardly wherever they go.
> 
> SDF troops on peacekeeping missions, for example, must adhere to prohibitions on “collective self defence”. So if anyone attacks the Japanese troops in Samawah, a nearby Dutch contingent will come to the SDF's aid; but if the Dutch troops should come under attack, they are on their own.
> 
> The 1992 law also limited SDF missions to cases in which a ceasefire is in place and Japan has a clear invitation from all local groups. In deploying troops to Samawah, Japan got round this by noting a UN resolution last year that called on willing countries to help rebuild Iraq. With a new interim government formally taking over at the end of this month, however, Japan would have had to negotiate a new and separate deal. Nominally joining the new UN-approved force makes things simpler.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> As a result, Japanese troops still stand out awkwardly wherever they go.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Japan's goal, when it began sending troops overseas, was to look like every other country that provides peacekeepers, so as to blend in and avoid criticism.
> 2. To stay within the pacifist restraints of its constitution, the 1992 law imposed tight restrictions on which kinds of missions the SDF is allowed to join.
> 3. No other peacekeeping nation places such constitution-driven limits on the missions its soldiers may take part in.
> 4. Operating under these unusual restrictions forces the SDF to behave visibly differently from every other national contingent on the ground.
> 5. Behaving differently from everyone else is the exact opposite of blending in and looking like the others.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Japan's goal, when it began sending troops overseas, was to look like every other country that provides peacekeepers, so as to blend in and avoid criticism.
> 2. To stay within the pacifist restraints of its constitution, the 1992 law imposed tight restrictions on which kinds of missions the SDF is allowed to join.
> 5. Behaving differently from everyone else is the exact opposite of blending in and looking like the others.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101050, irrelevant to this doc):

> 1. Canada's seal-fur industry depends on being able to export its furs to foreign markets, including the European Union.
> 2. The European Parliament adopted a regulation that prohibits the importation and marketing of seal products within the European Union.
> 3. Therefore the ban closes the EU market to Canadian seal furs, directly harming Canada's export trade.
> 4. Blocking another country's ability to sell its goods across borders is a barrier to international free trade.
> 5. The World Trade Organization is the international body that adjudicates disputes over cross-border trade barriers between member nations.
> 6. A nation harmed by another nation's trade barrier can seek relief by filing a formal complaint against it at the WTO.

---

## Doc 101050
### Original document (from DCLM, verbatim)
**Context:**

> World Trade Organization
> 
> Gandhi said that the greatness of a nation and its moral progress could be measured by the treatment that their animals. If true, this phrase would render greatness to countries that break the balance of nature to follow a model of unsustainable development. Such is the case of Denmark, where carried out massacres of dolphins in its coasts; or France, which allows their farmers get fatter to their geese until you bust them liver. Or Japan, where organized killing of whales in danger of extinction. They say that scientific research justifies the practices, but the greater part of the body of these cetaceans is used to feed farm animals. The slaughter of sharks in China and Japan for only the fins will cause an imbalance in the ecosystem that will pay all other species, including humans, if they do not begin to prohibit these practices. Spain, for example, did so with Fox hunting of sharks and hammerhead sharks, endangered species.
> 
> Another country in the crosshairs is Canada, which allows death to sticks of baby seals for producing and exporting furs. The European Parliament adopted a regulation which prohibits the importation and marketing of seal products in the European Union.

**Continuation (the real text that follows):**

> Canada is already preparing a lawsuit against the European Union before the World Trade Organization by hinder free trade, as if this was a supreme good over the protection of the planet and human well-being. To make a fur coat 8 adult seals or 20 baby, according to the Organization Equanimal seals, which publishes numbers of specimens of other species that are needed to produce a fur coat are needed: 17 Lynx, 60 mink, 20 otters, foxes 20, 60 marten, 250 squirrels and 12 wolves. Beyond the effects butterfly that these killings can occur in nature, also arises the suffering of many who are beaten, electrocuted or animals that are dying in traps for days.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Canada is already preparing a lawsuit against the European Union before the World Trade Organization

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Canada's seal-fur industry depends on being able to export its furs to foreign markets, including the European Union.
> 2. The European Parliament adopted a regulation that prohibits the importation and marketing of seal products within the European Union.
> 3. Therefore the ban closes the EU market to Canadian seal furs, directly harming Canada's export trade.
> 4. Blocking another country's ability to sell its goods across borders is a barrier to international free trade.
> 5. The World Trade Organization is the international body that adjudicates disputes over cross-border trade barriers between member nations.
> 6. A nation harmed by another nation's trade barrier can seek relief by filing a formal complaint against it at the WTO.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Canada's seal-fur industry depends on being able to export its furs to foreign markets, including the European Union.
> 2. The European Parliament adopted a regulation that prohibits the importation and marketing of seal products within the European Union.
> 5. The World Trade Organization is the international body that adjudicates disputes over cross-border trade barriers between member nations.
> 6. A nation harmed by another nation's trade barrier can seek relief by filing a formal complaint against it at the WTO.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102576, irrelevant to this doc):

> 1. Ektar is engineered for color accuracy, rendering the colors actually present in a scene instead of adding punch or softening tones of its own.
> 2. It is tightly engineered and valued for consistency, so it behaves the same way from one roll to the next.
> 3. A film that both stays faithful to the scene and performs consistently introduces no color shift or random variation of its own into the finished image.
> 4. It is also unforgiving of exposure and color-balance errors, neither masking nor compensating for them, so nothing in the result can be blamed on the film.

---

## Doc 102576
*NLL of the target: base **3.035** → +complete **2.989** (Δ-0.047) · +incomplete **2.95** (Δ-0.085) · +placebo **3.578** (Δ+0.543)*  ·  completeness effect (complete−incomplete) **+0.039**

### Original document (from DCLM, verbatim)
**Context:**

> Ektar does not give "punchy" color, unless punchy color is resident in the actual scene. It's fairly accurate for a color film. But neither does it
> artificially soften things like skintones, which is something Porta 160 does. Nor does it forgive errors in color balance as easily. If you are comfortable shooting chromes, Ektar should be easy to learn. If you want something more forgiving of exposure error, go Portra. These films are tightly engineered for specific categories of use. And consistency is one thing you tend to get in quality products like these. If you want high quality results, then the learning curve is going to be more consistent too.

**Continuation (the real text that follows):**

> Any mistakes which come out in the end result are likely to be your own. But you might pay a dollar more a roll for that privilege. Amateur films are made and marketed under less stringent conditions, and often have a buffer zone for sloppy use, so that you get "something", yet at the expense of something else.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Any mistakes which come out in the end result are likely to be your own.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Ektar is engineered for color accuracy, rendering the colors actually present in a scene instead of adding punch or softening tones of its own.
> 2. It is tightly engineered and valued for consistency, so it behaves the same way from one roll to the next.
> 3. A film that both stays faithful to the scene and performs consistently introduces no color shift or random variation of its own into the finished image.
> 4. It is also unforgiving of exposure and color-balance errors, neither masking nor compensating for them, so nothing in the result can be blamed on the film.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Ektar is engineered for color accuracy, rendering the colors actually present in a scene instead of adding punch or softening tones of its own.
> 4. It is also unforgiving of exposure and color-balance errors, neither masking nor compensating for them, so nothing in the result can be blamed on the film.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102589, irrelevant to this doc):

> 1. The interim Somali government, propped up by the occupying Ethiopian army, had just concluded its own government-sponsored reconciliation conference.
> 2. Only a week later the SCIC opposition, its leadership structure still intact, convened an entirely separate conference in Eritrea instead of taking part in that reconciliation.
> 3. At its conference the opposition demanded an immediate withdrawal of Ethiopian troops and set out to build an organization to "liberate the country."
> 4. Those aims run directly against the Ethiopian-backed interim government's position, so the two sides are pursuing incompatible goals through rival gatherings rather than one shared process.

---

## Doc 102589
*NLL of the target: base **3.619** → +complete **3.198** (Δ-0.421) · +incomplete **3.317** (Δ-0.302) · +placebo **3.968** (Δ+0.349)*  ·  completeness effect (complete−incomplete) **-0.119**

### Original document (from DCLM, verbatim)
**Context:**

> Somali Islamist leader emerges from hiding
> 
> The leader of Somalia's Islamist movement emerged from eight months of hiding yesterday to appear at an opposition meeting in Eritrea that called for an immediate withdrawal of Ethiopian troops.
> 
> Sheikh Hassan Dahir Aweys, who is accused by the US of links to al-Qaida, headed the Somali Council of Islamic Courts (SCIC) until it was driven from power in Mogadishu by Ethiopian forces last December. Having fled the capital he was thought to have been living in southern Somalia. Many people saw his hand in an ongoing insurgency against the occupying Ethiopian army and troops loyal to Somalia's interim government.
> 
> Mr Aweys's surprise appearance at the conference in Asmara, the Eritrean capital , which drew more than 300 delegates including observers from the UN and EU as well as disaffected members of the Somali government, confirmed recent reports that the leadership structure of the disbanded SCIC was still largely intact. The 72-year-old cleric sat alongside Sheikh Sharif Sheikh Ahmed, regarded as the SCIC's second-in-command, who said the aim of the 10-day meeting was to create "a political organisation that liberates the country ...".
> 
> The meeting came a week after the closure of a government-sponsored reconciliation conference in the capital.

**Continuation (the real text that follows):**

> The separate talks are indicative of the gulf between the two groups, whose differences are being played out on the streets of Mogadishu, where several people are being killed in fighting every day.
> 
> They also illustrate how Somalia has become a theatre for the proxy conflict between Ethiopia and Eritrea, whose relations have never recovered since they fought a war in the late 90s.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> The separate talks are indicative of the gulf between the two groups

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The interim Somali government, propped up by the occupying Ethiopian army, had just concluded its own government-sponsored reconciliation conference.
> 2. Only a week later the SCIC opposition, its leadership structure still intact, convened an entirely separate conference in Eritrea instead of taking part in that reconciliation.
> 3. At its conference the opposition demanded an immediate withdrawal of Ethiopian troops and set out to build an organization to "liberate the country."
> 4. Those aims run directly against the Ethiopian-backed interim government's position, so the two sides are pursuing incompatible goals through rival gatherings rather than one shared process.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The interim Somali government, propped up by the occupying Ethiopian army, had just concluded its own government-sponsored reconciliation conference.
> 4. Those aims run directly against the Ethiopian-backed interim government's position, so the two sides are pursuing incompatible goals through rival gatherings rather than one shared process.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101282, irrelevant to this doc):

> 1. In the past, doctors overprescribed antibiotics to huge numbers of children, including kids whose symptoms were mild, who had no clear diagnosis, or whose infections were likely viral and could not even be treated by the drugs.
> 2. This means bacteria living in and around a very large population of children were repeatedly and unnecessarily exposed to antibiotics.
> 3. Antibiotics kill the bacteria that are susceptible to them, but within any large bacterial population a few individuals carry random mutations that happen to make them resistant.
> 4. Under this constant drug exposure the susceptible bacteria are wiped out while the rare resistant ones survive and keep reproducing — a selective pressure favoring resistance.
> 5. Generation after generation the surviving resistant bacteria multiply and come to dominate the population.

---

## Doc 101282
*NLL of the target: base **3.789** → +complete **3.842** (Δ+0.053) · +incomplete **3.821** (Δ+0.032) · +placebo **4.406** (Δ+0.617)*  ·  completeness effect (complete−incomplete) **+0.021**

### Original document (from DCLM, verbatim)
**Context:**

> Tips For Ear Infections
> 
> Antibiotics Are Not Always the Answer
> 
> About 60 percent of ear infections are believed to be bacterial; the other 40 percent are sparked by viruses and can't be cured by antibiotics. (Unfortunately, there's no way for your doc to tell from looking in your child's ear whether an infection is viral or bacterial.) In 2004, the American Academy of Pediatrics (AAP) and the American Academy of Family Physicians (AAFP) jointly issued guidelines for treating acute ear infections in kids. The main message to doctors: Hand out fewer unnecessary prescriptions for antibiotics, and give the body's immune system a chance -- about two to three days -- to fight off the infection on its own. Studies have shown that approximately 80 percent of middle-ear infections in children go away without antibiotics in a week or so, and about 60 percent of kids have fewer symptoms after 24 hours, whether they take antibiotics or not. "Watchful waiting" is appropriate for a healthy child between 6 months and 2 years of age when her symptoms aren't severe (her fever is less than 102.2?F and she doesn't seem to be in a lot of pain) and her doctor isn't sure after looking in her ear that there's an infection. It's also appropriate for kids over 2 without severe symptoms. During the waiting period, your pediatrician will probably suggest a pain reliever such as acetaminophen, ibuprofen, or anesthetic ear drops. If your child's symptoms don't improve, contact the doctor.
> 
> Why not just take antibiotics ASAP? In the past, doctors overprescribed these drugs, experts say, giving them to kids whose symptoms were mild, who didn't have a clear-cut diagnosis, or whose infection was likely viral.

**Continuation (the real text that follows):**

> With children everywhere slurping down the "pink stuff," a scary problem began to arise: Some bacteria became resistant to the antibiotics. These strains can no longer be defeated by the traditional go-to remedies, which forces doctors to search for other alternatives. In Rochester, New York, a small group of kids had ear infections that didn't respond to any drug that's used to fight them in children, and doctors had to treat the bacteria (called the 19A strain) with a drug that was only approved for adults. The AAP/AAFP guidelines urge doctors to prescribe antibiotics more prudently to prevent resistant bacteria from spreading widely and putting all children at risk.
> 
> Parents Are Talking
> 
> Add a Comment

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> With children everywhere slurping down the "pink stuff," a scary problem began to arise: Some bacteria became resistant to the antibiotics.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. In the past, doctors overprescribed antibiotics to huge numbers of children, including kids whose symptoms were mild, who had no clear diagnosis, or whose infections were likely viral and could not even be treated by the drugs.
> 2. This means bacteria living in and around a very large population of children were repeatedly and unnecessarily exposed to antibiotics.
> 3. Antibiotics kill the bacteria that are susceptible to them, but within any large bacterial population a few individuals carry random mutations that happen to make them resistant.
> 4. Under this constant drug exposure the susceptible bacteria are wiped out while the rare resistant ones survive and keep reproducing — a selective pressure favoring resistance.
> 5. Generation after generation the surviving resistant bacteria multiply and come to dominate the population.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. In the past, doctors overprescribed antibiotics to huge numbers of children, including kids whose symptoms were mild, who had no clear diagnosis, or whose infections were likely viral and could not even be treated by the drugs.
> 2. This means bacteria living in and around a very large population of children were repeatedly and unnecessarily exposed to antibiotics.
> 5. Generation after generation the surviving resistant bacteria multiply and come to dominate the population.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100171, irrelevant to this doc):

> 1. The earlier paragraph shows that entering a value like `234 OR 1=1` forces the query WHERE ID=234 OR 1=1, which is always true, so the server retrieves every row in the table and incurs excessive load.
> 2. That injection exploits the structure of the SQL query, not the sensitivity of the rows, so it succeeds regardless of whether the data is confidential or publicly displayable.
> 3. Hence even a database holding only publicly available data can be driven to waste its resources running such full-table scans, i.e. a denial-of-service attack.
> 4. A server tied up serving those malicious scans becomes unresponsive to legitimate users, a genuine harm that is distinct from any leak of private data.

---

## Doc 100171
*NLL of the target: base **1.455** → +complete **4.079** (Δ+2.624) · +incomplete **3.767** (Δ+2.312) · +placebo **3.872** (Δ+2.417)*  ·  completeness effect (complete−incomplete) **+0.312**

### Original document (from DCLM, verbatim)
**Context:**

> 6.1.7 Client Programming Security Guidelines
> 
> Applications that access MySQL should not trust any data entered by users, who can try to trick your code by entering special or escaped character sequences in Web forms, URLs, or whatever application you have built. Be sure that your application remains secure if a user enters something like ; DROP DATABASE mysql;. This is an extreme example, but large security leaks and data loss might occur as a result of hackers using similar techniques, if you do not prepare for them.
> 
> A common mistake is to protect only string data values. Remember to check numeric data as well. If an application generates a query such as SELECT * FROM table WHERE ID=234 when a user enters the value 234, the user can enter the value 234 OR 1=1 to cause the application to generate the query SELECT * FROM table WHERE ID=234 OR 1=1. As a result, the server retrieves every row in the table. This exposes every row and causes excessive server load. The simplest way to protect from this type of attack is to use single quotation marks around the numeric constants: SELECT * FROM table WHERE ID='234'. If the user enters extra information, it all becomes part of the string. In a numeric context, MySQL automatically converts this string to a number and strips any trailing nonnumeric characters from it.
> 
> Sometimes people think that if a database contains only publicly available data, it need not be protected.

**Continuation (the real text that follows):**

> This is incorrect. Even if it is permissible to display any row in the database, you should still protect against denial of service attacks (for example, those that are based on the technique in the preceding paragraph that causes the server to waste resources). Otherwise, your server becomes unresponsive to legitimate users.
> 
> 
> Many application programming interfaces provide a means of escaping special characters in data values. Properly used, this prevents application users from entering values that cause the application to generate statements that have a different effect than you intend:
> 
> Other programming interfaces might have similar capabilities.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> This is incorrect.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The earlier paragraph shows that entering a value like `234 OR 1=1` forces the query WHERE ID=234 OR 1=1, which is always true, so the server retrieves every row in the table and incurs excessive load.
> 2. That injection exploits the structure of the SQL query, not the sensitivity of the rows, so it succeeds regardless of whether the data is confidential or publicly displayable.
> 3. Hence even a database holding only publicly available data can be driven to waste its resources running such full-table scans, i.e. a denial-of-service attack.
> 4. A server tied up serving those malicious scans becomes unresponsive to legitimate users, a genuine harm that is distinct from any leak of private data.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The earlier paragraph shows that entering a value like `234 OR 1=1` forces the query WHERE ID=234 OR 1=1, which is always true, so the server retrieves every row in the table and incurs excessive load.
> 4. A server tied up serving those malicious scans becomes unresponsive to legitimate users, a genuine harm that is distinct from any leak of private data.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100501, irrelevant to this doc):

> 1. A big-O claim such as "constant time" is a statement about asymptotic behavior — how the running time grows as the input size increases without bound.
> 2. If the input size is instead held fixed at a constant, there are only finitely many possible inputs, so any procedure's running time is bounded by a constant — making the "constant time" label apply trivially to every procedure.
> 3. Under this fixed-input view, even undecidable problems, which are the hardest problems there are, qualify as "constant time."
> 4. If both the hardest problems and the easiest problems land in the very same complexity class, the asymptotic language can no longer draw any distinction between the difficulty of one problem and another.

---

## Doc 100501
*NLL of the target: base **2.904** → +complete **2.626** (Δ-0.278) · +incomplete **2.61** (Δ-0.294) · +placebo **3.178** (Δ+0.274)*  ·  completeness effect (complete−incomplete) **+0.015**

### Original document (from DCLM, verbatim)
**Context:**

> note blokhead <blockquote><i>If the number of bits is constant, then any polynomial time based on it is constant.</i></blockquote> Big-O statements (like an algorithm taking constant or O(1) time) are statements about asymptotic behavior, i.e, how the function behaves in the limit (usually, as input size tends to infinity). If you don't look at them in the limit, then big-O-ish language (like constant time) is meaningless. <p> How meaningless? Even undecidable languages have a constant time "algorithm" if you consider the input size to be held to a constant.

**Continuation (the real text that follows):**

> So without viewing things in the limit, <i>all</i> problems become computationally equivalent in the asymptotic language. <p> <b>Update:</b> added citation from parent node <!-- Node text goes above. Div tags should contain sig only --> <div class="pmsig"><div class="pmsig-137386"> <p> blokhead </div></div> 143755 507875

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> So without viewing things in the limit, <i>all</i> problems become computationally equivalent in the asymptotic language.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. A big-O claim such as "constant time" is a statement about asymptotic behavior — how the running time grows as the input size increases without bound.
> 2. If the input size is instead held fixed at a constant, there are only finitely many possible inputs, so any procedure's running time is bounded by a constant — making the "constant time" label apply trivially to every procedure.
> 3. Under this fixed-input view, even undecidable problems, which are the hardest problems there are, qualify as "constant time."
> 4. If both the hardest problems and the easiest problems land in the very same complexity class, the asymptotic language can no longer draw any distinction between the difficulty of one problem and another.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. A big-O claim such as "constant time" is a statement about asymptotic behavior — how the running time grows as the input size increases without bound.
> 4. If both the hardest problems and the easiest problems land in the very same complexity class, the asymptotic language can no longer draw any distinction between the difficulty of one problem and another.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101527, irrelevant to this doc):

> 1. A fireplace normally needs a chimney or vent to carry away the smoke and soot it produces, because those combustion byproducts are harmful to breathe indoors.
> 2. The Pureflame burns ethanol and produces no soot or smoke at all — its only byproducts are steam and carbon dioxide.
> 3. It emits these in quantities similar to what a person gives off simply by breathing.
> 4. People exhale steam and carbon dioxide indoors constantly with no ventilation of that air, so those quantities are already harmless in an occupied room.
> 5. Therefore the Pureflame puts nothing into the room that has to be carried outside to keep the air safe.

---

## Doc 101527
*NLL of the target: base **2.576** → +complete **2.748** (Δ+0.173) · +incomplete **2.488** (Δ-0.088) · +placebo **4.725** (Δ+2.149)*  ·  completeness effect (complete−incomplete) **+0.260**

### Original document (from DCLM, verbatim)
**Context:**

> The Pureflame Is a Mobile Fireplace
> 
> Seems like every time the temperature in the Bay Area dips enough to actually warrant building a fire, the Air District will declare a Spare the Air Day, making wood-burning illegal. This wall-mounted fireplace from Pureflame could be the answer—not only does it run on soot-free ethanol, I can hang it anywhere in my house.
> 
> The Pureflame runs on plant-derived ethanol that burns without creating soot or smoke. According to the manufacturer its only byproducts are steam and carbon dioxide—both in quantities similar to what humans produce through respiration.

**Continuation (the real text that follows):**

> This eliminates the need for venting or chimneys.
> 
> And without the need to vent, the fireplace can be moved from room to room like a flaming space heater. A one-quart bottle of fuel will last produce yellow and orange flames for between two to five hours. The wall-mounted varieties retail for $650-880. [Pureflames via Gizmag]

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> This eliminates the need for venting or chimneys.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. A fireplace normally needs a chimney or vent to carry away the smoke and soot it produces, because those combustion byproducts are harmful to breathe indoors.
> 2. The Pureflame burns ethanol and produces no soot or smoke at all — its only byproducts are steam and carbon dioxide.
> 3. It emits these in quantities similar to what a person gives off simply by breathing.
> 4. People exhale steam and carbon dioxide indoors constantly with no ventilation of that air, so those quantities are already harmless in an occupied room.
> 5. Therefore the Pureflame puts nothing into the room that has to be carried outside to keep the air safe.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. A fireplace normally needs a chimney or vent to carry away the smoke and soot it produces, because those combustion byproducts are harmful to breathe indoors.
> 2. The Pureflame burns ethanol and produces no soot or smoke at all — its only byproducts are steam and carbon dioxide.
> 5. Therefore the Pureflame puts nothing into the room that has to be carried outside to keep the air safe.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102524, irrelevant to this doc):

> 1. The context states that, historically, the federal government has spent about 22 percent of GDP, and that this spending share held roughly steady across every administration since WWII, including Reagan, both Bushes, Clinton, and Obama.
> 2. A budget's balance is the difference between the revenue the government collects and the amount it spends.
> 3. Because the spending share was essentially the same under every administration, the spending side of the ledger cannot explain why some administrations ran deficits while others did not.
> 4. That means the only variable left that can account for the difference between a deficit and a surplus is how much revenue was collected.
> 5. Clinton was the administration that produced balanced budgets and even surpluses, unlike the others that ran deficits.

---

## Doc 102524
*NLL of the target: base **3.172** → +complete **2.916** (Δ-0.256) · +incomplete **2.944** (Δ-0.228) · +placebo **3.368** (Δ+0.196)*  ·  completeness effect (complete−incomplete) **-0.027**

### Original document (from DCLM, verbatim)
**Context:**

> Become a digitalPlus subscriber. $13 for 13 weeks.
> 
> 
> Gross Domestic Product
> Tax policies continue to constrain US economy
> Tax policies continue to constrain US economy
> 
> Some myths die hard, thanks to a feckless and ineffective American media, such as Cal Thomas' assertion (quoting Reagan) that the people aren't under-taxed but we have deficits because the government spends too much. A clear-eyed view of the facts/numbers shows that this belief is 180 degrees wrong. Historically, the federal government spends 22 percent of the gross domestic product — Reagan, Bush one, Bush two, Clinton and Obama, all administrations since WWII.

**Continuation (the real text that follows):**

> So how did Clinton manage balanced budgets and even surpluses? His tax policies (which did not stifle growth or kill jobs, as evidenced by the best economy of our lifetimes under Clinton) brought in revenues of 22...

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> So how did Clinton manage balanced budgets and even surpluses? His tax policies (which did not stifle growth or kill jobs, as evidenced by the best economy of our lifetimes under Clinton) brought in revenues of 22

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The context states that, historically, the federal government has spent about 22 percent of GDP, and that this spending share held roughly steady across every administration since WWII, including Reagan, both Bushes, Clinton, and Obama.
> 2. A budget's balance is the difference between the revenue the government collects and the amount it spends.
> 3. Because the spending share was essentially the same under every administration, the spending side of the ledger cannot explain why some administrations ran deficits while others did not.
> 4. That means the only variable left that can account for the difference between a deficit and a surplus is how much revenue was collected.
> 5. Clinton was the administration that produced balanced budgets and even surpluses, unlike the others that ran deficits.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The context states that, historically, the federal government has spent about 22 percent of GDP, and that this spending share held roughly steady across every administration since WWII, including Reagan, both Bushes, Clinton, and Obama.
> 2. A budget's balance is the difference between the revenue the government collects and the amount it spends.
> 5. Clinton was the administration that produced balanced budgets and even surpluses, unlike the others that ran deficits.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101359, irrelevant to this doc):

> 1. Question 6 would provide state funds specifically to permanently protect remaining farmland and to acquire open space (the context lists $4.5M to protect farmlands plus money to acquire parks, beaches, and shoreline).
> 2. In this program, 'protecting' farmland means the state buying the land or its development rights outright, not merely passing a regulation.
> 3. The remaining farmland and open space are highly desirable ('gems') and are under intense pressure to be developed.
> 4. Intense development pressure means private developers are actively bidding to buy those same parcels and will pay high prices for them.
> 5. To keep a parcel as farmland, the state must acquire it before a developer does, which means outbidding or matching those developers on price.

---

## Doc 101359
*NLL of the target: base **3.664** → +complete **3.191** (Δ-0.473) · +incomplete **3.129** (Δ-0.535) · +placebo **4.158** (Δ+0.494)*  ·  completeness effect (complete−incomplete) **+0.062**

### Original document (from DCLM, verbatim)
**Context:**

> Vote yes on Questions 5 and 6
> 
> Voters face difficult decisions when they enter the booths on Nov. 6: Who should sit on the next town council? Who should lead the school committee? Who are the best representatives to the Rhode Island General Assembly? Who should be U.S. president?
> Two questions on the ballot, however, should be easy to answer: Yes on Questions 5 and 6.
> Question 5 would provide $20 million for clean water infrastructure improvements, including $12 million for wastewater treatment plant improvements and $8 million for drinking water system improvements. The money will be used to leverage millions more in federal funds. Maybe nowhere else in the country are a state’s economy and quality of life more affected by water. In Rhode Island, we are blessed with a beautiful bay, rivers, streams, lakes and ponds. This money will be used to keep it that way.
> Question 6 would provide $20 million for farmland and open space protection, park development and bay restoration. Approximately $4.5 million would fund the state program to permanently protect farmlands; $2.5 million would be used to acquire state parks, beaches and shoreline areas; $2.5 million would create matching grants to municipalities, land trusts and other organizations involved in protecting wildlife habitat, farms, forests and water resources; $6.5 million would be spent improving and creating municipal parks and restoring historic parks; and $4 million would protect water supplies, ponds, rivers, streams, and Narragansett Bay from polluted storm water and establish a fish passage on the Blackstone River.
> In the past, residents of these towns voted overwhelmingly in support of environmental bond referenda and the payoff is evident everywhere from Tiverton’s Pardon Gray to progress toward the dream of an Aquidneck Greenway
> 
> Pressure to develop such gems is intense.

**Continuation (the real text that follows):**

> If it hopes to save any of the farmland that’s left, the state needs money to compete. 
> 
> These initiatives will pay dividends that dwarf the investment.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> If it hopes to save any of the farmland that’s left, the state needs money to compete.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Question 6 would provide state funds specifically to permanently protect remaining farmland and to acquire open space (the context lists $4.5M to protect farmlands plus money to acquire parks, beaches, and shoreline).
> 2. In this program, 'protecting' farmland means the state buying the land or its development rights outright, not merely passing a regulation.
> 3. The remaining farmland and open space are highly desirable ('gems') and are under intense pressure to be developed.
> 4. Intense development pressure means private developers are actively bidding to buy those same parcels and will pay high prices for them.
> 5. To keep a parcel as farmland, the state must acquire it before a developer does, which means outbidding or matching those developers on price.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Question 6 would provide state funds specifically to permanently protect remaining farmland and to acquire open space (the context lists $4.5M to protect farmlands plus money to acquire parks, beaches, and shoreline).
> 2. In this program, 'protecting' farmland means the state buying the land or its development rights outright, not merely passing a regulation.
> 5. To keep a parcel as farmland, the state must acquire it before a developer does, which means outbidding or matching those developers on price.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100135, irrelevant to this doc):

> 1. In old English, the "th" sound was written as a single special rune (thorn), not with the letters T and H.
> 2. That thorn rune was shaped so much like the letter Y that the two looked nearly identical on the page.
> 3. Printers who had no thorn in their type boxes therefore substituted an ordinary Y for it — a change of written symbol only, leaving unchanged the spoken sound the rune had represented.
> 4. Any word that now appears in print as "Ye" is thus the old thorn-word in disguise, so reading its Y as a Y-sound would be sounding out the substitute symbol as if it were the original.

---

## Doc 100135
*NLL of the target: base **2.625** → +complete **2.706** (Δ+0.080) · +incomplete **2.683** (Δ+0.058) · +placebo **2.896** (Δ+0.271)*  ·  completeness effect (complete−incomplete) **+0.022**

### Original document (from DCLM, verbatim)
**Context:**

> "Ye Olde" is one of those phrases we throw around for cheap laughs, but do you have any idea where that first part comes from? Or how to pronounce it? Probably not! Minute Physics takes a stab at explaining, and like most things in life, "Ye" was born out of laziness—and totally France's fault.
> 
> Back in the days of old English, the "Th" sound was represented by a single rune shaped something like a letter P. Over the years, it was Babelfished back and forth between French and modern English enough that printers threw their hands up and decided to use a Y instead of two scrunched together letters to represent a "th" sound.

**Continuation (the real text that follows):**

> Which means that "Ye" is actually "The," and is still supposed to be pronounced as "The." Which, you know, invalidates just about everything that ever happened at your local renaissance faire. Sorry, Harwin! [MinutePhysics]

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Which means that "Ye" is actually "The," and is still supposed to be pronounced as "The."

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. In old English, the "th" sound was written as a single special rune (thorn), not with the letters T and H.
> 2. That thorn rune was shaped so much like the letter Y that the two looked nearly identical on the page.
> 3. Printers who had no thorn in their type boxes therefore substituted an ordinary Y for it — a change of written symbol only, leaving unchanged the spoken sound the rune had represented.
> 4. Any word that now appears in print as "Ye" is thus the old thorn-word in disguise, so reading its Y as a Y-sound would be sounding out the substitute symbol as if it were the original.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. In old English, the "th" sound was written as a single special rune (thorn), not with the letters T and H.
> 4. Any word that now appears in print as "Ye" is thus the old thorn-word in disguise, so reading its Y as a Y-sound would be sounding out the substitute symbol as if it were the original.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100143, irrelevant to this doc):

> 1. In a sunlit snow scene, the small shadowed areas within the snow are illuminated mainly by blue skylight rather than by direct sunlight.
> 2. A yellow filter is "minus blue": it absorbs blue light, so anything lit predominantly by blue light records darker on the film.
> 3. Those small blue-lit shadow patches within the snow are therefore rendered darker by a yellow filter, while the directly sunlit snow stays bright.
> 4. Darkening the shadow patches while the sunlit snow stays bright widens the tonal gap between them, and longer (extended) development stretches that tonal separation further still.

---

## Doc 100143
*NLL of the target: base **3.899** → +complete **3.911** (Δ+0.011) · +incomplete **3.819** (Δ-0.080) · +placebo **4.381** (Δ+0.482)*  ·  completeness effect (complete−incomplete) **+0.091**

### Original document (from DCLM, verbatim)
**Context:**

> [QUOTE=dr bob] I made the mistake of thinking a Wratten 11 or 15 would bring out the "sparkles" in a local scene - wrong! The result was completely dark shadows without any detail (TX400). I rephotographed with a Wratten 47 (blue) and the shadows popped out perfectly. This seems to fly in the face of general logic until one considers that the shadows in a sunlit scene are illuminated by blue light.
> 
> 
> Dr. Bob it seems to me that your stated experience verifies that shadows are illuminated with blue light. Yellow filtration would be minus blue and that would account for the deepening of shadow values that you noted. The 47 blue filter is plus blue and would account for the lightening of shadow values that you indicated.
> 
> This blue filter would have the effect of lowering local contrast within the snow itself since local contrast within the snow itself would contain small shadow areas that are lit by the same blue light that you noted in the shadows.
> 
> While a full scale scene may have shadow and snow both included, the actual scene would need to be evaluated to determine the exposure and development considerations.

**Continuation (the real text that follows):**

> However for local contrast in the snow itself yellow filtration and expanded development would enhance local contrast.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> However for local contrast in the snow itself yellow filtration and expanded development would enhance local contrast.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. In a sunlit snow scene, the small shadowed areas within the snow are illuminated mainly by blue skylight rather than by direct sunlight.
> 2. A yellow filter is "minus blue": it absorbs blue light, so anything lit predominantly by blue light records darker on the film.
> 3. Those small blue-lit shadow patches within the snow are therefore rendered darker by a yellow filter, while the directly sunlit snow stays bright.
> 4. Darkening the shadow patches while the sunlit snow stays bright widens the tonal gap between them, and longer (extended) development stretches that tonal separation further still.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. In a sunlit snow scene, the small shadowed areas within the snow are illuminated mainly by blue skylight rather than by direct sunlight.
> 4. Darkening the shadow patches while the sunlit snow stays bright widens the tonal gap between them, and longer (extended) development stretches that tonal separation further still.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102162, irrelevant to this doc):

> 1. The model shows that the freedom to adjust one's work effort after the fact, including choosing when to retire, leads a person to take on more investment-portfolio risk beforehand.
> 2. How much of this freedom a person retains depends on how many working years still lie ahead: with a long career remaining, work effort and retirement date are still adjustable, but near the end of a career they are essentially fixed.
> 3. Therefore the amount of labor-supply flexibility a person has declines with age, so a younger person has far more of it than an older person.
> 4. Applying the model's mechanism from step 1, whoever holds more labor flexibility rationally chooses a riskier investment portfolio.

---

## Doc 102162
*NLL of the target: base **3.105** → +complete **2.813** (Δ-0.292) · +incomplete **2.899** (Δ-0.206) · +placebo **3.395** (Δ+0.290)*  ·  completeness effect (complete−incomplete) **-0.086**

### Original document (from DCLM, verbatim)
**Context:**

> 02028cam a22002417 4500001000600000003000500006005001700011008004100028100001600069245014900085260006600234490004100300500001800341520105500359530006101414538007201475538003601547700002201583700002601605710004201631830007601673856003701749w3954NBER20140317064302.0140317s1992 mau||||fs|||| 000 0 eng d1 aBodie, Zvi.10aLabor Supply Flexibility and Portfolio Choice in a Life-Cycle Modelh[electronic resource] /cZvi Bodie, Robert C. Merton, William F. Samuelson. aCambridge, Mass.bNational Bureau of Economic Researchc1992.1 aNBER working paper seriesvno. w3954 aJanuary 1992.3 aThis paper examines the effect of the labor-leisure choice on portfolio and consumption decisions over an individual's life cycle. The model incorporates the fact that individuals may have considerable flexibility in varying their work effort (including their choice of when to retire). Given this flexibility, the individual simultaneously determines optimal levels of current consumption, labor effort, and an optimal financial investment strategy at each point in his life cycle. We show that labor and investment choices are intimately related. The ability to vary labor supply ex post induces the individual to assume greater risks in his investment portfolio ex ante.

**Continuation (the real text that follows):**

> The model explains why the young (enjoying greater labor flexibility over their working lives) may take greater investment risks than the old. It also offers an explanation as to why consumption spending is relatively "smooth" despite volatility in asset prices. Finally, the paper provides a compact method for valuing the risky cash flows associated with future wage income. aHardcopy version available to institutional subscribers. aSystem requirements: Adobe [Acrobat] Reader required for PDF files. aMode of access: World Wide Web.1 aMerton, Robert C.1 aSamuelson, William F.2 aNational Bureau of Economic Research. 0aWorking Paper Series (National Bureau of Economic Research)vno. w3954.4 uhttp://www.nber.org/papers/w3954

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> The model explains why the young (enjoying greater labor flexibility over their working lives) may take greater investment risks than the old.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The model shows that the freedom to adjust one's work effort after the fact, including choosing when to retire, leads a person to take on more investment-portfolio risk beforehand.
> 2. How much of this freedom a person retains depends on how many working years still lie ahead: with a long career remaining, work effort and retirement date are still adjustable, but near the end of a career they are essentially fixed.
> 3. Therefore the amount of labor-supply flexibility a person has declines with age, so a younger person has far more of it than an older person.
> 4. Applying the model's mechanism from step 1, whoever holds more labor flexibility rationally chooses a riskier investment portfolio.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The model shows that the freedom to adjust one's work effort after the fact, including choosing when to retire, leads a person to take on more investment-portfolio risk beforehand.
> 4. Applying the model's mechanism from step 1, whoever holds more labor flexibility rationally chooses a riskier investment portfolio.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101594, irrelevant to this doc):

> 1. North Korea's core security problem is avoiding invasion; the context states that a nation without nuclear weapons must sustain a massive conventional military or else end up on the target list.
> 2. Because North Korea cannot rely on China to protect it, its own conventional armed forces are its only available deterrent, which is why it diverts nearly all of its GDP into maintaining them.
> 3. The context also holds that any nation possessing nuclear weapons is not invaded, even by superpowers, so a nuclear arsenal is itself a full deterrent against invasion.
> 4. A large conventional army and a nuclear arsenal serve the identical purpose of deterring invasion, so one deterrent can substitute for the other.
> 5. If North Korea acquires nuclear weapons plus a means to deliver them, it gains the same invasion deterrence it currently obtains only from its costly conventional forces.

---

## Doc 101594
*NLL of the target: base **3.955** → +complete **3.98** (Δ+0.026) · +incomplete **3.713** (Δ-0.242) · +placebo **4.432** (Δ+0.477)*  ·  completeness effect (complete−incomplete) **+0.268**

### Original document (from DCLM, verbatim)
**Context:**

> Comments     Threshold
> 
> 
> RE: say wha???
> By Darkskypoet on 2/9/2009 7:07:50 AM , Rating: 2
> This is it exactly. NK would never launch a nuke against another country, unless they were brutally invaded. The 'cold war' lessons learned guidebook (lol) has taught many nations one important fact; If you have nukes, other nations, even super powers, don't invade you. Conversely, if you don't have nukes; you had better spend your self into near oblivion supporting a massive conventional armed force, or you will be on the target list.
> 
> Their are 2 reasons the United States did not go after North Korea, 1) China 2) The losses in Iraq would have looked like a bubble bath in comparison. As NK can't depend on China to protect them, they feel they are in a position where they have to divert every penny of their GDP (that they possibly can, and even some they can't) to maintaining a large battle ready Armed Forces.

**Continuation (the real text that follows):**

> Nukes and a delivery platform change that equation and allow them to curtail some of their conventional force spending.
> 
> Its that simple. Its that rational. Nukes = less extensive conventional forces. Especially when you aren't trying to police the world, but simply defending your borders.
> 
> RE: say wha???

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Nukes and a delivery platform change that equation and allow them to curtail some of their conventional force spending.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. North Korea's core security problem is avoiding invasion; the context states that a nation without nuclear weapons must sustain a massive conventional military or else end up on the target list.
> 2. Because North Korea cannot rely on China to protect it, its own conventional armed forces are its only available deterrent, which is why it diverts nearly all of its GDP into maintaining them.
> 3. The context also holds that any nation possessing nuclear weapons is not invaded, even by superpowers, so a nuclear arsenal is itself a full deterrent against invasion.
> 4. A large conventional army and a nuclear arsenal serve the identical purpose of deterring invasion, so one deterrent can substitute for the other.
> 5. If North Korea acquires nuclear weapons plus a means to deliver them, it gains the same invasion deterrence it currently obtains only from its costly conventional forces.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. North Korea's core security problem is avoiding invasion; the context states that a nation without nuclear weapons must sustain a massive conventional military or else end up on the target list.
> 2. Because North Korea cannot rely on China to protect it, its own conventional armed forces are its only available deterrent, which is why it diverts nearly all of its GDP into maintaining them.
> 5. If North Korea acquires nuclear weapons plus a means to deliver them, it gains the same invasion deterrence it currently obtains only from its costly conventional forces.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100649, irrelevant to this doc):

> 1. Per the context, most annulments occur shortly after the wedding, typically between people who had not known each other for long before marrying.
> 2. A marriage that is dissolved so soon gives the couple little time to jointly accumulate significant assets or a shared home.
> 3. Such a brief union also makes it very unlikely that the couple had children together.
> 4. In an ordinary marital dissolution, the heavily contested matters are splitting up shared marital assets and setting arrangements for any children.

---

## Doc 100649
*NLL of the target: base **2.517** → +complete **2.511** (Δ-0.005) · +incomplete **2.546** (Δ+0.029) · +placebo **2.966** (Δ+0.449)*  ·  completeness effect (complete−incomplete) **-0.035**

### Original document (from DCLM, verbatim)
**Context:**

> Annulment or Divorce, What's the Difference? - Law and Daily Life
> 
> Annulment or Divorce, What's the Difference?
> 
> 
> 
> Annulments, as a general rule, require at least one of the following: 1) some type of fraud or concealment; 2) a refusal or inability to consummate the marriage (yes, that's pretty much just what it sounds like); or 3) a misunderstanding. Although "misunderstanding" sounds like a fairly broad category, it actually is usually interpreted fairly narrowly to mean a misunderstanding on some kind of a "deal-breaker" type of issue. It doesn't mean an argument or simply "not getting along". One example of a key misunderstanding might be if a couple had never discussed having children and they now found out they disagreed on the issue.
> 
> Fraud or hiding key information from a spouse might be grounds for an annulment, as well. For example, if a spouse lies about their ability to have kids (i.e. they physically can't), or hides that they have a sexually transmitted disease, these might be grounds for an annulment based on fraud or concealment.
> 
> The above noted requirements and the examples probably demonstrate why most annulments happen shortly after marriage, and in circumstances where the parties didn't know each other too long before marrying.

**Continuation (the real text that follows):**

> Lastly, if a short term marriage is involved, an annulment would probably not involve any major issues such as division of property, child custody and support, etc. Below are links to more information on the topic, as well as a questionnaire for those looking into annulments.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Lastly, if a short term marriage is involved, an annulment would probably not involve any major issues such as division of property, child custody and support

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Per the context, most annulments occur shortly after the wedding, typically between people who had not known each other for long before marrying.
> 2. A marriage that is dissolved so soon gives the couple little time to jointly accumulate significant assets or a shared home.
> 3. Such a brief union also makes it very unlikely that the couple had children together.
> 4. In an ordinary marital dissolution, the heavily contested matters are splitting up shared marital assets and setting arrangements for any children.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Per the context, most annulments occur shortly after the wedding, typically between people who had not known each other for long before marrying.
> 4. In an ordinary marital dissolution, the heavily contested matters are splitting up shared marital assets and setting arrangements for any children.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100654, irrelevant to this doc):

> 1. When commentators say outsourcing is costing the country jobs, they are counting only the positions that move from the U.S. to lower-wage economies.
> 2. But the context shows most foreign direct investment flows into the United States rather than to low-wage countries, and firms owned abroad employ millions of Americans while U.S. multinationals keep more than twice as many workers at home as overseas.
> 3. Those American positions that exist because the U.S. is the world's leading destination for investment are entirely omitted from the outsourcing complaint.
> 4. Once the inbound-investment jobs are set against the jobs lost to outsourcing, the balance for American workers comes out favorable rather than the pure loss the complaint implies.

---

## Doc 100654
*NLL of the target: base **1.993** → +complete **4.764** (Δ+2.771) · +incomplete **4.213** (Δ+2.220) · +placebo **4.732** (Δ+2.739)*  ·  completeness effect (complete−incomplete) **+0.550**

### Original document (from DCLM, verbatim)
**Context:**

> While that may be true for individual companies, the data show that overall, “offshoring” from other countries to the U.S. is greatly benefitting the American economy.
> 
> As this chart shows, most foreign direct investment does not go to low-wage countries like China and Mexico—it goes to the United States!
> 
> Protectionists would be shocked – shocked! to find that many huge multinational corporations actually prefer to produce in the United States with U.S. workers.
> 
> U.S.-based multinational corporations employ 22.9 million Americans—more than twice as many people as they employ in China, Mexico, and all other countries combined. Foreign-owned multinational corporations employ another 5.5 million people in the United States.
> 
> When talking heads or campaigning politicians assert that outsourcing is costing the U.S.

**Continuation (the real text that follows):**

> jobs, they’re telling only part of the story. The whole picture shows that U.S. workers do just fine competing for jobs in a global marketplace, and in fact the United States continues to win the war for global investment.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> jobs, they’re telling only part of the story

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. When commentators say outsourcing is costing the country jobs, they are counting only the positions that move from the U.S. to lower-wage economies.
> 2. But the context shows most foreign direct investment flows into the United States rather than to low-wage countries, and firms owned abroad employ millions of Americans while U.S. multinationals keep more than twice as many workers at home as overseas.
> 3. Those American positions that exist because the U.S. is the world's leading destination for investment are entirely omitted from the outsourcing complaint.
> 4. Once the inbound-investment jobs are set against the jobs lost to outsourcing, the balance for American workers comes out favorable rather than the pure loss the complaint implies.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. When commentators say outsourcing is costing the country jobs, they are counting only the positions that move from the U.S. to lower-wage economies.
> 4. Once the inbound-investment jobs are set against the jobs lost to outsourcing, the balance for American workers comes out favorable rather than the pure loss the complaint implies.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102491, irrelevant to this doc):

> 1. The engine won't crank, but the lights, radio, and heat all work, so the battery is charged and supplying power to the car's electrical system.
> 2. The battery is also brand new, which rules out a dead or worn-out battery as the cause.
> 3. A single click when turning the key to start is the sound of the starter solenoid engaging while the starter motor fails to spin the engine.
> 4. So the problem is not a lack of electrical power (the accessories work) and not the engine computer (which wouldn't produce a solenoid click) — the fault must lie between the charged battery and the starter's ability to crank.

---

## Doc 102491
*NLL of the target: base **3.36** → +complete **2.801** (Δ-0.559) · +incomplete **2.789** (Δ-0.571) · +placebo **3.959** (Δ+0.599)*  ·  completeness effect (complete−incomplete) **+0.012**

### Original document (from DCLM, verbatim)
**Context:**

> 1993 Jeep Grand Cherokee
> 
> I just recently had my Jeep in the show for an all around tune up, it needed plugs/wires, battery, brakes, all the normal maintenance. While my vehicle was in the shop, the mechanic had told me that there were a few other things wrong with it, the radiator was leaking, so was the water pump. And that the PCM (I think that's what it was called) should be replaced, but I needed my vehicle to move, and couldn't wait for them to order the part. Within 2 days of being 800 miles away from this shop in my new home. My Jeep won't turn over. The lights are on, the radio works, even the heat works. But I can't get it to start.

**Continuation (the real text that follows):**

> I called the shop and the guy told me he was positive it was the computer, but I don't think it is. What could be the problem? The battery is brand new. It makes a click sound when I try to start it, but then I let it be.
> January 28, 2013.
> 
> If the click is when you turn the key to start, I would be looking at a faulty starter or a bad battery connection, start here.
> 
> Jan 28, 2013.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> I would be looking at a faulty starter or a bad battery connection

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The engine won't crank, but the lights, radio, and heat all work, so the battery is charged and supplying power to the car's electrical system.
> 2. The battery is also brand new, which rules out a dead or worn-out battery as the cause.
> 3. A single click when turning the key to start is the sound of the starter solenoid engaging while the starter motor fails to spin the engine.
> 4. So the problem is not a lack of electrical power (the accessories work) and not the engine computer (which wouldn't produce a solenoid click) — the fault must lie between the charged battery and the starter's ability to crank.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The engine won't crank, but the lights, radio, and heat all work, so the battery is charged and supplying power to the car's electrical system.
> 4. So the problem is not a lack of electrical power (the accessories work) and not the engine computer (which wouldn't produce a solenoid click) — the fault must lie between the charged battery and the starter's ability to crank.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101184, irrelevant to this doc):

> 1. Female lesser house flies are strongly attracted to decaying fecal matter and livestock waste, which they visit to lay eggs and where the larvae feed.
> 2. Fecal matter and decaying organic material are reservoirs of disease-causing microorganisms.
> 3. As a fly walks and feeds on this contaminated material, those microorganisms cling to its legs, body hairs, and mouthparts.
> 4. The same flies then move on and land on other surfaces, including food and places where people live and eat.
> 5. As they land, the microorganisms carried on their bodies are deposited onto that food and those surfaces.

---

## Doc 101184
*NLL of the target: base **3.156** → +complete **3.337** (Δ+0.181) · +incomplete **3.368** (Δ+0.212) · +placebo **3.559** (Δ+0.403)*  ·  completeness effect (complete−incomplete) **-0.030**

### Original document (from DCLM, verbatim)
**Context:**

> Lesser House Flies
> 
> As its name implies, the lesser house fly, also called the little house fly, is noticeably smaller than the standard housefly (Musca domestica). It is approximately 5 to 6 mm in length and yellowish in coloration. The lesser house fly’s thorax hosts three black stripes.
> 
> Females are attracted to decaying fecal matter as egg-laying sites and can be a particular nuisance in chicken houses and other livestock grounds. Their eggs are white and thin, measuring 2 mm in length. Maggots develop fully within five to seven days and enter the pupal stage. , During the larval stage, they feed ravenously on the material on which the eggs were laid. Lesser house flies require a period of nine to 14 days.
> 
> Lesser houseflies move slightly faster than other species and fly in jerky, darting patterns. Lesser house fly eggs are capable of floating and can be found resting on standing water.

**Continuation (the real text that follows):**

> Like common houseflies, lesser houseflies are known carriers of pathogens resulting in human ailments, including typhoid, cholera, dysentery and anthrax. They pick up pathogens from fecal matter and other decaying material, and then transfer it to humans by landing on exposed food and other surfaces.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Like common houseflies, lesser houseflies are known carriers of pathogens resulting in human ailments

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Female lesser house flies are strongly attracted to decaying fecal matter and livestock waste, which they visit to lay eggs and where the larvae feed.
> 2. Fecal matter and decaying organic material are reservoirs of disease-causing microorganisms.
> 3. As a fly walks and feeds on this contaminated material, those microorganisms cling to its legs, body hairs, and mouthparts.
> 4. The same flies then move on and land on other surfaces, including food and places where people live and eat.
> 5. As they land, the microorganisms carried on their bodies are deposited onto that food and those surfaces.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Female lesser house flies are strongly attracted to decaying fecal matter and livestock waste, which they visit to lay eggs and where the larvae feed.
> 4. The same flies then move on and land on other surfaces, including food and places where people live and eat.
> 5. As they land, the microorganisms carried on their bodies are deposited onto that food and those surfaces.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102320, irrelevant to this doc):

> 1. Only the 20 highest-scoring brackets each receive $100,000, so Corey must finish among the top 20 to win anything.
> 2. Corey is tied for fourth right now, but tonight's championship game has not yet been scored, so those final points are still up for grabs.
> 3. Picking the champion correctly earns those final points, but Corey picked no champion and his bracket is locked, so he can earn zero additional points no matter who wins.
> 4. Every entrant who did pick the eventual champion will gain those points and climb the standings while Corey's score stays frozen.
> 5. More than 20 of those climbing entrants sit close enough behind Corey to pass him once they collect the champion's points, dropping him out of the top 20.

---

## Doc 102320
*NLL of the target: base **3.047** → +complete **3.517** (Δ+0.470) · +incomplete **3.328** (Δ+0.281) · +placebo **4.219** (Δ+1.172)*  ·  completeness effect (complete−incomplete) **+0.189**

### Original document (from DCLM, verbatim)
**Context:**

> This is the worst bracket ever filled out. Which is to say it's an excellent bracket. Corey nailed 11 of the Sweet 16 teams, seven of the Elite Eight, three of the Final Four, and has a UConn-Kentucky championship game. Except Corey forgot to pick a winner. [Update: We talked to Corey.]
> 
> Corey currently sits tied for fourth in Yahoo's billion-dollar bracket challenge group. That billion is long gone, but the creators of the 20 highest-scoring brackets each receive $100,000. If Corey correctly picked the winner for tonight's tournament final, he'd be in the money. But Corey didn't correctly pick the winner, because he didn't pick any winner, and his bracket is long locked.

**Continuation (the real text that follows):**

> There will be no money for Corey, because no matter who wins, he'll be leapfrogged by more than 20 people who did select a champion.
> 
> Corey's bracket is named "Corey's Champion Bracket."
> 
> Poor Schmo Forgets To Pick Bracket Winner, Costs Himself Shot At $100K
> 
> Corey, wherever you are, you have our sympathies.
> 
> [Lost Lettermen]

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> There will be no money for Corey

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Only the 20 highest-scoring brackets each receive $100,000, so Corey must finish among the top 20 to win anything.
> 2. Corey is tied for fourth right now, but tonight's championship game has not yet been scored, so those final points are still up for grabs.
> 3. Picking the champion correctly earns those final points, but Corey picked no champion and his bracket is locked, so he can earn zero additional points no matter who wins.
> 4. Every entrant who did pick the eventual champion will gain those points and climb the standings while Corey's score stays frozen.
> 5. More than 20 of those climbing entrants sit close enough behind Corey to pass him once they collect the champion's points, dropping him out of the top 20.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Only the 20 highest-scoring brackets each receive $100,000, so Corey must finish among the top 20 to win anything.
> 2. Corey is tied for fourth right now, but tonight's championship game has not yet been scored, so those final points are still up for grabs.
> 5. More than 20 of those climbing entrants sit close enough behind Corey to pass him once they collect the champion's points, dropping him out of the top 20.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101374, irrelevant to this doc):

> 1. The ActiveX control copies attacker-supplied input into a fixed-size buffer that is too small to hold it, and it does so without checking the length, so oversized input spills past the end of the buffer into adjacent memory.
> 2. That adjacent memory holds the program's control data (such as the saved return address and function pointers), so the overflow overwrites the values that decide where execution will continue next.
> 3. Turning the overflow into code execution requires the attacker to overwrite that control data with a precisely crafted value that redirects execution onto their own injected instructions.
> 4. An attempt that is not precisely crafted still overwrites the same control data, but with invalid, garbage values instead of a working payload.
> 5. Execution then transfers to an invalid address, so the process references bad memory and the host application (typically Internet Explorer) crashes.

---

## Doc 101374
*NLL of the target: base **3.357** → +complete **3.845** (Δ+0.488) · +incomplete **3.764** (Δ+0.407) · +placebo **4.326** (Δ+0.969)*  ·  completeness effect (complete−incomplete) **+0.082**

### Original document (from DCLM, verbatim)
**Context:**

> LeadTools Raster ISIS Object LTRIS14e.DLL ActiveX Control Buffer Overflow Vulnerability
> 
> LEADTOOLS Raster ISIS ActiveX control is prone to a buffer-overflow vulnerability because the application fails to bounds-check user-supplied data before copying it into an insufficiently sized buffer.
> 
> Successfully exploiting this issue allows remote attackers to execute arbitrary code in the context of the application using the ActiveX control (typically Internet Explorer).

**Continuation (the real text that follows):**

> Failed exploit attempts likely result in denial-of-service conditions.
> 
> LEADTOOLS ISIS ActiveX control is vulnerable to this issue; other versions may also be affected.
> 
> 
> Privacy Statement
> Copyright 2010, SecurityFocus

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Failed exploit attempts likely result in denial-of-service conditions.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The ActiveX control copies attacker-supplied input into a fixed-size buffer that is too small to hold it, and it does so without checking the length, so oversized input spills past the end of the buffer into adjacent memory.
> 2. That adjacent memory holds the program's control data (such as the saved return address and function pointers), so the overflow overwrites the values that decide where execution will continue next.
> 3. Turning the overflow into code execution requires the attacker to overwrite that control data with a precisely crafted value that redirects execution onto their own injected instructions.
> 4. An attempt that is not precisely crafted still overwrites the same control data, but with invalid, garbage values instead of a working payload.
> 5. Execution then transfers to an invalid address, so the process references bad memory and the host application (typically Internet Explorer) crashes.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The ActiveX control copies attacker-supplied input into a fixed-size buffer that is too small to hold it, and it does so without checking the length, so oversized input spills past the end of the buffer into adjacent memory.
> 2. That adjacent memory holds the program's control data (such as the saved return address and function pointers), so the overflow overwrites the values that decide where execution will continue next.
> 5. Execution then transfers to an invalid address, so the process references bad memory and the host application (typically Internet Explorer) crashes.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100075, irrelevant to this doc):

> 1. Gene's 'our characters DON'T' rule forbade TNG characters from having done the kinds of things Tasha's upbringing on the failed colony of Turkana IV would have entailed.
> 2. Because of that rule, the writers could not let Tasha draw on her Turkana IV past, so the one experience that made her distinctive was off-limits as story material.
> 3. With her defining backstory unusable, there was little unique material left to build episodes around her, so her role stayed thin and underdeveloped relative to the other main cast.
> 4. An actor cast in a thin, underdeveloped role with almost no meaningful material to perform grows creatively frustrated with the part.
> 5. A creatively frustrated actor tends to want to leave the production (as Denise Crosby, who played Tasha, in fact did after the first season).

---

## Doc 100075
### Original document (from DCLM, verbatim)
**Context:**

> View Single Post
> Old June 10 2009, 12:11 AM   #25
> Re: I like Tasha Yar.
> 
> I have always had a soft spot for the characters who get killed off because the writers 'can't work with their character' - I always think something along the lines of 'give ME a crack at them, I could do something!' Tasha especially, since she's a fascinating character - all the other main cast are these people who have lived the good life in the heart of the Federation, while she grew up on a colony that 'failed' (don't quite understand that - if the colony failed, how exactly are there still people there?)
> 
> Sadly, Tasha's character was a victim of two things - the writers trying to put a female in a position that traditionally was a 'man's' role while also trying to make it seem a casual 'yeah, happens all the time, and, the bigger problem of the two, the 'our characters DON'T' clause that Gene put in for TNG. Her existence on Turkana IV would have been filled with occasions were she engaged in activities that Gene had dictated that our character would not do, and as such, it made it hard to do anything with her, because she wouldn't have been allowed to draw on her prior experiences, given that as a character, they were things that she should have done, but as a Gene Roddenberry character, she wasn't allowed to do.
> 
> I know Star Trek wouldn't be around without him, but I firmly feel that by the time of TNG's production, Gene had bought into his own hype and let himself believe that all it took to overcome these inherent human traits he deemed as being 'negative' was the power of positive thinking - if TNG Gene had worked on TOS, we wouldn't have had the Spock-Bones banter, because the characters were 'too evolved' for such arguments.

**Continuation (the real text that follows):**

> All the TNG characters suffered because of the 'our characters DON'T' rule, but none so much as Tasha - if that hadn't been a rule, I would not be surprised if Denise Crosby would have wanted to remain with the show.
> DGCatAniSiri is offline   Reply With Quote

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> All the TNG characters suffered because of the 'our characters DON'T' rule, but none so much as Tasha - if that hadn't been a rule, I would not be surprised if Denise Crosby would have wanted to remain with the show.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Gene's 'our characters DON'T' rule forbade TNG characters from having done the kinds of things Tasha's upbringing on the failed colony of Turkana IV would have entailed.
> 2. Because of that rule, the writers could not let Tasha draw on her Turkana IV past, so the one experience that made her distinctive was off-limits as story material.
> 3. With her defining backstory unusable, there was little unique material left to build episodes around her, so her role stayed thin and underdeveloped relative to the other main cast.
> 4. An actor cast in a thin, underdeveloped role with almost no meaningful material to perform grows creatively frustrated with the part.
> 5. A creatively frustrated actor tends to want to leave the production (as Denise Crosby, who played Tasha, in fact did after the first season).

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Gene's 'our characters DON'T' rule forbade TNG characters from having done the kinds of things Tasha's upbringing on the failed colony of Turkana IV would have entailed.
> 2. Because of that rule, the writers could not let Tasha draw on her Turkana IV past, so the one experience that made her distinctive was off-limits as story material.
> 5. A creatively frustrated actor tends to want to leave the production (as Denise Crosby, who played Tasha, in fact did after the first season).

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100081, irrelevant to this doc):

> 1. As it currently works, get_posts() retrieves every column of each matching post, including the large post_content field, even when only the post ID is wanted.
> 2. Hauling post_content and the other unused fields from MySQL over to WordPress on every call wastes time and memory on data that is immediately thrown away.
> 3. A site's home page is requested very frequently, so the get_posts() query behind it runs an enormous number of times.
> 4. If that home page changes often, its output cannot be cached, so each visit must actually re-execute the query against the database instead of serving a stored copy.
> 5. Because the query is re-run on nearly every one of those many visits, the small per-call waste of fetching all fields is multiplied into a large aggregate load.
> 6. Adding a 'fields' argument that returns only the ID removes that wasted transfer from each of those many uncached executions.

---

## Doc 100081
*NLL of the target: base **3.106** → +complete **2.742** (Δ-0.364) · +incomplete **2.9** (Δ-0.206) · +placebo **3.398** (Δ+0.292)*  ·  completeness effect (complete−incomplete) **-0.157**

### Original document (from DCLM, verbatim)
**Context:**

> ﻿id summary reporter owner description type status priority milestone component version severity resolution keywords cc focuses 14777 "Adding ""fields"" to arguments array for get_posts()/query_posts()/WP_Query()" mikeschinkel "Hi all, I find myself more and more often needing to get a list of post IDs so I can call another WordPress database API function to include or exclude those posts. Calling `get_posts()` on a potentially large number of records and passing all the fields (especially `post_content`) between MySQL and WordPress is hugely inefficient when I only need the one ID. My two (2) options are: 1.) '''Code is directly in SQL.''' This is easy but I know it's definitely not a best practice and I would like to use the WordPress API wherever possible. 2.) '''To use a `post_fields` hook'''. Problem is that those are global and I have to start wrapping logic around my code to ensure I don't accidentally break some plugin or some other part of WordPress (this approach is much like trying to secure a server by starting with all the attack vectors open and then trying to close them all.) So I'd like to propose we simply add `""fields""` as a recognized argument for `get_posts()`, i.e. {{{ $posts = get_posts(array( 'fields' => 'ID,post_title', 'post_type' => 'movie', 'post_status' => 'publish', 'order' => 'ASC', 'posts_per_page' => -1 )); }}} I know I could make the same argument for `joins`, `where`, `orderbys` et. al. but I'd argue this is enough of a special case it could really use some early attention.

**Continuation (the real text that follows):**

> For a query on the home page of a high traffic site that changes often enough to not be able to be cached this tiny change could make a major difference in performance. If you'd like to see a use case here is one: - [http://wordpress.stackexchange.com/questions/1140/removing-duplicate-custom-taxonomy-terms-from-within-a-dropdown-select Removing Duplicate Custom Taxonomy Terms from within a Dropdown Select?] Unfortunately I still struggle with creating patches even though I have done it a few times in the past but each time I seem to have to start relearning from scratch. I seem to have a mental block for some reason on patches ([http://wordpress.stackexchange.com/questions/990/ '''can someone help me with this?''']) so I haven't gone ahead and written a patch but will tackle it if I get the task blessed. -Mike " enhancement closed normal 3.1 Query normal fixed

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> For a query on the home page of a high traffic site that changes often enough to not be able to be cached this tiny change could make a major difference in performance.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. As it currently works, get_posts() retrieves every column of each matching post, including the large post_content field, even when only the post ID is wanted.
> 2. Hauling post_content and the other unused fields from MySQL over to WordPress on every call wastes time and memory on data that is immediately thrown away.
> 3. A site's home page is requested very frequently, so the get_posts() query behind it runs an enormous number of times.
> 4. If that home page changes often, its output cannot be cached, so each visit must actually re-execute the query against the database instead of serving a stored copy.
> 5. Because the query is re-run on nearly every one of those many visits, the small per-call waste of fetching all fields is multiplied into a large aggregate load.
> 6. Adding a 'fields' argument that returns only the ID removes that wasted transfer from each of those many uncached executions.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. As it currently works, get_posts() retrieves every column of each matching post, including the large post_content field, even when only the post ID is wanted.
> 2. Hauling post_content and the other unused fields from MySQL over to WordPress on every call wastes time and memory on data that is immediately thrown away.
> 3. A site's home page is requested very frequently, so the get_posts() query behind it runs an enormous number of times.
> 6. Adding a 'fields' argument that returns only the ID removes that wasted transfer from each of those many uncached executions.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101654, irrelevant to this doc):

> 1. Human cloning by nuclear transfer is being pursued for reproductive purposes, and newer transgenic methods could even produce clones carrying new genes.
> 2. Pros and cons have arisen over reproductive human cloning, meaning there are serious unresolved objections to it.
> 3. Those unresolved objections center on whether the procedure is biologically safe and whether it is ethically acceptable.
> 4. Performing reproductive cloning while such safety and ethical problems remain unsolved would be irresponsible and unacceptable.
> 5. Finding solutions to these safety and ethical problems requires time and further study, which means cloning must be paused in the meantime.

---

## Doc 101654
*NLL of the target: base **2.397** → +complete **2.454** (Δ+0.057) · +incomplete **2.417** (Δ+0.020) · +placebo **3.137** (Δ+0.740)*  ·  completeness effect (complete−incomplete) **+0.037**

### Original document (from DCLM, verbatim)
**Context:**

> Kloning Manusia
> 
> Teresa L. Wargasetia
> 
> 
> In the last few years, very rapid progress in the cloning technology and its development towards human cloning has become a hotly-debated issue. Cloning, which is the process of formation of a number of individuals with the same genetic structure, can be done by means of embryo-splitting method and nuclear transfer. Human cloning through the nuclear transfer method is directed towards two purposes, i.e. reproduction and therapy. The relatively new transgenic technology can be combined with the cloning technique to produce clones with new genes. However, pros and cons arise concerning the development of research on human cloning, particularly cloning for reproductive purposes.

**Continuation (the real text that follows):**

> Therefore, there is need for a moratorium period before human cloning can be performed in order that solutions for all kinds of problems related to safety and ethics can be found.
> 
> 
> Full Text: PDF
> 
> 
>   • There are currently no refbacks.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Therefore, there is need for a moratorium period before human cloning can be performed

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Human cloning by nuclear transfer is being pursued for reproductive purposes, and newer transgenic methods could even produce clones carrying new genes.
> 2. Pros and cons have arisen over reproductive human cloning, meaning there are serious unresolved objections to it.
> 3. Those unresolved objections center on whether the procedure is biologically safe and whether it is ethically acceptable.
> 4. Performing reproductive cloning while such safety and ethical problems remain unsolved would be irresponsible and unacceptable.
> 5. Finding solutions to these safety and ethical problems requires time and further study, which means cloning must be paused in the meantime.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Human cloning by nuclear transfer is being pursued for reproductive purposes, and newer transgenic methods could even produce clones carrying new genes.
> 2. Pros and cons have arisen over reproductive human cloning, meaning there are serious unresolved objections to it.
> 5. Finding solutions to these safety and ethical problems requires time and further study, which means cloning must be paused in the meantime.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100611, irrelevant to this doc):

> 1. Azo printing paper is largely sensitive to UV light rather than to ordinary visible light.
> 2. Glass tends to block UV, and the thicker 1/4" plate glass reduces light transmission more than the thinner 1/8" clear glass, so the two pieces differ mainly in how much UV they pass.
> 3. Because Azo's exposure depends on UV, that difference in UV transmission between the two glasses translates into a large difference in printing time — which is exactly why switching glasses shortened the times so much.
> 4. Ordinary enlarging paper, by contrast, is sensitive to visible light and is not appreciably sensitive to UV.
> 5. Since ordinary paper does not rely on UV, the two glasses' difference in UV transmission would have little effect on how much usable light reaches it.

---

## Doc 100611
*NLL of the target: base **2.637** → +complete **1.972** (Δ-0.665) · +incomplete **2.353** (Δ-0.284) · +placebo **2.851** (Δ+0.214)*  ·  completeness effect (complete−incomplete) **-0.381**

### Original document (from DCLM, verbatim)
**Context:**

> Quote Originally Posted by Steve Sherman View Post
> The previous post didn't work out to well.
> 
> I used to use a piece of 1/4" plate glass with not frame. In using Azo I found the printing time quite long. Just happened to switch to a vacuum frame which uses a piece of clear 1/8" glass, the printing times were significantly shorter with the same negatives. I asked a glass person why and he said that plate glass is inherently stronger than ordinary glass and actually has a light reducing effect, he simply asked if the sides of the glass were at all green, that is an indication of "plate" glass which reduces light transmission.
> 
> Also, Azo is largely sensitive to UV and glass tends to block UV - that's why UV lenses are made of quartz.

**Continuation (the real text that follows):**

> I'd bet that if you were using ordinary enlarging paper, you wouldn't see nearly as much difference between the 2 pieces of glass.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> I'd bet that if you were using ordinary enlarging paper, you wouldn't see nearly as much difference between the 2 pieces of glass.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Azo printing paper is largely sensitive to UV light rather than to ordinary visible light.
> 2. Glass tends to block UV, and the thicker 1/4" plate glass reduces light transmission more than the thinner 1/8" clear glass, so the two pieces differ mainly in how much UV they pass.
> 3. Because Azo's exposure depends on UV, that difference in UV transmission between the two glasses translates into a large difference in printing time — which is exactly why switching glasses shortened the times so much.
> 4. Ordinary enlarging paper, by contrast, is sensitive to visible light and is not appreciably sensitive to UV.
> 5. Since ordinary paper does not rely on UV, the two glasses' difference in UV transmission would have little effect on how much usable light reaches it.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Azo printing paper is largely sensitive to UV light rather than to ordinary visible light.
> 2. Glass tends to block UV, and the thicker 1/4" plate glass reduces light transmission more than the thinner 1/8" clear glass, so the two pieces differ mainly in how much UV they pass.
> 5. Since ordinary paper does not rely on UV, the two glasses' difference in UV transmission would have little effect on how much usable light reaches it.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102443, irrelevant to this doc):

> 1. M-252 is described as a nickel-base high-temperature alloy, so nickel is its predominant constituent — the balance of the composition.
> 2. The analytical scheme measures molybdenum, cobalt, iron, and chromium by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy; and carbon by the high-frequency induction furnace.
> 3. Every element named is covered by one of those instruments, but nickel itself appears nowhere in the instrument list — it is the only principal constituent left unmeasured.
> 4. Because the paper states the alloy is "completely analyzed," the weight fractions of all constituents must together account for 100% of the material.
> 5. Nickel therefore does not require its own instrumental reading; its value is fixed automatically once all the other elements have been quantified.

---

## Doc 102443
*NLL of the target: base **4.592** → +complete **4.25** (Δ-0.342) · +incomplete **4.271** (Δ-0.321) · +placebo **5.313** (Δ+0.721)*  ·  completeness effect (complete−incomplete) **-0.021**

### Original document (from DCLM, verbatim)
**Context:**

> @article {Gerber:1954-08-01T00:00:00:0003-7028:120, author = "Gerber, W. O. and Tobin, W. H.", title = "A Spectrochemical Method for the Analysis of M-252 Nickel Base High Temperature Alloy and the Preparation of Standards by Powder Metallurgy", journal = "Applied Spectroscopy", volume = "8", number = "3", year = "1954-08-01T00:00:00", abstract = "M-252 nickel base high temperature alloy is completely analyzed by instruments at the general Electric River Works. Molybdenum, cobalt, iron, and chromium are determined by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy and carbon by the high frequency induction furnace.

**Continuation (the real text that follows):**

> The remainder, nickel, is calculated by difference.", pages = "120-125", url = "http://www.ingentaconnect.com/content/sas/sas/1954/00000008/00000003/art00004", doi = "doi:10.1366/000370254774634549" }

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> The remainder, nickel, is calculated by difference.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. M-252 is described as a nickel-base high-temperature alloy, so nickel is its predominant constituent — the balance of the composition.
> 2. The analytical scheme measures molybdenum, cobalt, iron, and chromium by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy; and carbon by the high-frequency induction furnace.
> 3. Every element named is covered by one of those instruments, but nickel itself appears nowhere in the instrument list — it is the only principal constituent left unmeasured.
> 4. Because the paper states the alloy is "completely analyzed," the weight fractions of all constituents must together account for 100% of the material.
> 5. Nickel therefore does not require its own instrumental reading; its value is fixed automatically once all the other elements have been quantified.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. M-252 is described as a nickel-base high-temperature alloy, so nickel is its predominant constituent — the balance of the composition.
> 2. The analytical scheme measures molybdenum, cobalt, iron, and chromium by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy; and carbon by the high-frequency induction furnace.
> 5. Nickel therefore does not require its own instrumental reading; its value is fixed automatically once all the other elements have been quantified.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101421, irrelevant to this doc):

> 1. A&P became dominant by aggressively discounting — selling goods at unusually low prices.
> 2. Those low prices let A&P capture huge market share (the first company ever to reach $1 billion in annual sales) and drove many small independent stores out of business.
> 3. Because it was seen as destroying small businesses, chain stores like A&P became political targets, and lawmakers moved to curb it through anti-trust law.
> 4. The Hartfords were then prosecuted and found guilty of criminal anti-trust activity for the very pricing strategy that had made them dominant.

---

## Doc 101421
*NLL of the target: base **3.506** → +complete **3.687** (Δ+0.181) · +incomplete **3.856** (Δ+0.350) · +placebo **5.349** (Δ+1.843)*  ·  completeness effect (complete−incomplete) **-0.169**

### Original document (from DCLM, verbatim)
**Context:**

> Book Nook: The Great A&P and the Struggle for Small Business in America, by Marc Levinson
> 
> Dec 6, 2012
> 
> 
> The author shifted his focus to the astonishing history of the A&P store chain. 100 years ago in 1912 the Hartfords implemented what became a seismic shift in retailing. They began opening what they called Economy stores. By 1929 the A&P's expansive presence and aggressive discounting had made them such a dominant force that they became the first company to ever rack up a billion dollars in annual sales.
> 
> By then chain stores like the A&P had become political targets. By the 1940's the Hartford brothers were found to be guilty of criminal anti-trust activity.

**Continuation (the real text that follows):**

> Their crime? Low prices. Too low. This government action ultimately led to the decline and eventual demise of that entire store chain.
> 
> Levinson tells a fascinating tale here. And on a side note; in this interview the author recalls his time at Antioch College back in the 1970's when he used to read the news on an eclectic little radio station known as WYSO.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> Their crime? Low prices.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. A&P became dominant by aggressively discounting — selling goods at unusually low prices.
> 2. Those low prices let A&P capture huge market share (the first company ever to reach $1 billion in annual sales) and drove many small independent stores out of business.
> 3. Because it was seen as destroying small businesses, chain stores like A&P became political targets, and lawmakers moved to curb it through anti-trust law.
> 4. The Hartfords were then prosecuted and found guilty of criminal anti-trust activity for the very pricing strategy that had made them dominant.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. A&P became dominant by aggressively discounting — selling goods at unusually low prices.
> 4. The Hartfords were then prosecuted and found guilty of criminal anti-trust activity.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 101447, irrelevant to this doc):

> 1. Rebels are steadily advancing in and around Damascus, and the regime has already lost its grip on most of the eastern and southern city, where rebels have seized army bases.
> 2. Because Damascus is the seat of Assad's power, this mounting pressure threatens the regime's survival and forces it to prioritize defending the capital above all else.
> 3. The regime's army is finite and already stretched thin, so reinforcing the capital is only possible by drawing forces away from distant, rebel-dominated fronts like the north.
> 4. The regime's irreducible core is its capital plus the Alawite coastal region around Latakia, the heartland of the sect the ruling Assads belong to — the territory it must hold to survive.

---

## Doc 101447
*NLL of the target: base **3.126** → +complete **2.986** (Δ-0.140) · +incomplete **2.991** (Δ-0.135) · +placebo **3.512** (Δ+0.386)*  ·  completeness effect (complete−incomplete) **-0.005**

### Original document (from DCLM, verbatim)
**Context:**

> REBEL advances in the east and north of Syria have captured most attention recently, with opposition fighters using an anti-aircraft missile to bring down a regime plane for the first time this week. But things have also been getting tough for President Bashar Assad in Damascus, the capital.
> 
> On Thursday the regime shut the city's airport, and airlines including Emirates and Egypt Air have cancelled flights there until further notice, citing deteriorating security. The closure apparently came after rebels operating in nearby suburbs fired a mortar at the facility. They say the airport is not only being used for military aircraft, but that Mr Assad's allies in Iran and Russia have used civilian planes to fly in money and other support, from advisers to riot equipment.
> 
> Heavy clashes ensued on the road from the city to the airport. At the same time, the disabling of the internet and most phone lines nationwide led to widespread fears that the government was planning a large operation in the area. Activists sending out news via satellite connections say this has not materialised so far, though the usual shelling and air strikes have continued across the country. By Friday evening the internet was still down but the airport road had reportedly reopened.
> 
> Fighting in Damascus has been increasing for several weeks, with the rebels edging ever closer to the heart of power despite suffering repeated strikes. Even regime figures admit the army no longer has a hold over almost all the eastern and southern areas of the city, where rebels have taken over at least two army bases.
> 
> Perhaps more importantly, the increasing pressure in the capital has ramifications for the wider conflict.

**Continuation (the real text that follows):**

> The regime has proved remarkably reluctant to pull garrisons and bases out of northern areas largely controlled by the rebels, but it may soon be forced to consolidate in Damascus and up the highway to Latakia, the coastal heartland of the Alawite sect, to which the ruling Assads belong. The government has already largely withdrawn its forces from the east, where rebels have moved in from the Iraqi border to the edge of Deir Ezzor, the main city in the region, snatching bases along the way. Military analysts now suspect it may do the same in the rural areas of Aleppo and Idleb in the north.
> 
> (Photo credit: AFP)

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> The regime has proved remarkably reluctant to pull garrisons and bases out of northern areas largely controlled by the rebels, but it may soon be forced to consolidate in Damascus and up the highway to Latakia

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. Rebels are steadily advancing in and around Damascus, and the regime has already lost its grip on most of the eastern and southern city, where rebels have seized army bases.
> 2. Because Damascus is the seat of Assad's power, this mounting pressure threatens the regime's survival and forces it to prioritize defending the capital above all else.
> 3. The regime's army is finite and already stretched thin, so reinforcing the capital is only possible by drawing forces away from distant, rebel-dominated fronts like the north.
> 4. The regime's irreducible core is its capital plus the Alawite coastal region around Latakia, the heartland of the sect the ruling Assads belong to — the territory it must hold to survive.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. Rebels are steadily advancing in and around Damascus, and the regime has already lost its grip on most of the eastern and southern city, where rebels have seized army bases.
> 4. The regime's irreducible core is its capital plus the Alawite coastal region around Latakia, the heartland of the sect the ruling Assads belong to — the territory it must hold to survive.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 100026, irrelevant to this doc):

> 1. The whole inner solar system condensed from one and the same primordial dust cloud, so iridium should be distributed in roughly equal proportion across all of its bodies.
> 2. Earth therefore should have formed containing the same fraction of iridium as the surrounding space and the other inner-solar-system bodies.
> 3. Yet iridium is observed to be even rarer on Earth than it is in space, so present-day Earth holds less iridium than it should have started with — Earth is depleted in iridium relative to expectation.
> 4. A depletion below the starting amount means the missing iridium was physically carried off the Earth at some point, not simply never present.
> 5. The Moon formed when a giant collision flung part of the proto-Earth's material off into orbit, which is precisely the mechanism capable of removing Earth's material.

---

## Doc 100026
*NLL of the target: base **2.578** → +complete **2.493** (Δ-0.085) · +incomplete **2.466** (Δ-0.112) · +placebo **3.193** (Δ+0.615)*  ·  completeness effect (complete−incomplete) **+0.027**

### Original document (from DCLM, verbatim)
**Context:**

> Comments     Threshold
> 
> 
> By dgingeri on 1/18/2011 4:13:47 PM , Rating: 2
> a while back, I saw a show on History that detailed what they think happened when the moon formed. They showed computer modeling that had a smaller earth and a larger planet collide, the heavier elements, including the iron cores, settle to the center of the new Earth and the lighter elements settle on what became the moon. There could be a lot to the theory of elements rare here that would be plentiful on the moon.
> 
> We know that Iridium is even more rare on the earth than in space, and with the whole solar system made from the same dust cloud, there should be pretty much equal distributions of iridium throughout the inner solar system.

**Continuation (the real text that follows):**

> so the iridium must have ended up on the moon after the collision. There's likely to be more elements like that. The moon could be a huge cash cow.

**Target scored** (the reasoning-consequence, a verbatim span of the continuation):

> so the iridium must have ended up on the moon after the collision.

### The three rationales that were inserted before the target
**① COMPLETE** (every load-bearing step, no gaps — written from the context only):

> 1. The whole inner solar system condensed from one and the same primordial dust cloud, so iridium should be distributed in roughly equal proportion across all of its bodies.
> 2. Earth therefore should have formed containing the same fraction of iridium as the surrounding space and the other inner-solar-system bodies.
> 3. Yet iridium is observed to be even rarer on Earth than it is in space, so present-day Earth holds less iridium than it should have started with — Earth is depleted in iridium relative to expectation.
> 4. A depletion below the starting amount means the missing iridium was physically carried off the Earth at some point, not simply never present.
> 5. The Moon formed when a giant collision flung part of the proto-Earth's material off into orbit, which is precisely the mechanism capable of removing Earth's material.

**② INCOMPLETE** (same chain, the 1–2 most load-bearing MIDDLE steps deleted → a real gap):

> 1. The whole inner solar system condensed from one and the same primordial dust cloud, so iridium should be distributed in roughly equal proportion across all of its bodies.
> 2. Earth therefore should have formed containing the same fraction of iridium as the surrounding space and the other inner-solar-system bodies.
> 5. The Moon formed when a giant collision flung part of the proto-Earth's material off into orbit, which is precisely the mechanism capable of removing Earth's material.

**③ PLACEBO** (an UNRELATED doc's complete rationale — from doc 102770, irrelevant to this doc):

> 1. The site's internal navigation links are hard-coded as absolute URLs on the .com domain instead of relative paths.
> 2. Inside the office, the .com domain does not point to the public web host, because the office server's localhost was given the name theukoffice.com.
> 3. So any request to the .com domain from an office machine is answered by the local office server, which stores an outdated copy of the site.
> 4. Even a user who deliberately opens the correct .co.uk copy will, on clicking any internal link, have the browser request that link's hard-coded .com address.

---

