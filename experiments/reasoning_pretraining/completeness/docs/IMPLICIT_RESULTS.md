# Perplexity drop on IMPLICIT-reasoning docs (no marker filter)

Per Dongwei: filtering for 'thus/therefore' selects docs whose reasoning is already explicit — defeats the
purpose. These are real DCLM docs with NO marker, where the `target` (verbatim from the doc) follows from the
context's UNSTATED reasoning. Score the real target under DCLM-1.4B: base vs +rationale vs +placebo.

**n=15: real−base −0.016 (60% drop), placebo−base +0.448 (0%), real−placebo −0.463 (real<placebo 14/15).**
An irrelevant rationale RAISES the target's perplexity; only the correct implicit reasoning lowers it.

---

### id 438 — base 3.773 → real 4.163 / placebo 5.381  (real−placebo -1.219)
**Context (from DCLM, tail):** …ge Monbiot. 'It was exhilarating; there's a feeling that permeates society that the only power we have as individuals is as consumers. But being a consumer's very different to being a citizen, and for once in my life, I felt like a citizen, someone doing what they believed in.'

He doesn't subscribe to the 'we're doomed' argument. 'But there's a definite possibility we're very close to a tipping point, where the biosphere takes over and there's nothing we can do to stop runaway climate change...

**Rationale (the IMPLICIT reasoning, added):** 1. The speaker warns we are very close to a climate tipping point.
2. Beyond that point, runaway warming could no longer be stopped by any human action.
3. Avoiding that outcome depends on cutting emissions before the point is crossed.

**REAL target (verbatim next text — scored):** We have to make rapid reductions

---

### id 389 — base 3.384 → real 3.324 / placebo 4.187  (real−placebo -0.863)
**Context (from DCLM, tail):** …White House Suddenly Decides Fake News Is a Bad Thing

Whenever you see a photograph of the president making a major address from inside the White House, it's really a picture of him saying "peas and carrots, peas and carrots" after the speech while photographers get their shots. Not anymore, though! For some reason, the White House has decided to stop participating in that particular form of fake-news manufacturing.

**Rationale (the IMPLICIT reasoning, added):** 1. The context describes the White House routinely staging the president saying "peas and carrots" after a speech so photographers can get their shots — manufacturing a fake image of a live address.
2. The White House has now decided to stop participating in that specific staged reshoot.
3. The decision is framed narrowly as ending only "that particular form" of fake-news manufacturing.
4. The qualifier "particular" signals other staged or managed news practices exist and are left untouched by the announcement.

**REAL target (verbatim next text — scored):** It will continue to fake other news events, though.

---

### id 418 — base 3.575 → real 3.652 / placebo 4.392  (real−placebo -0.740)
**Context (from DCLM, tail):** …n repeat steps 1 and 2 with the other antler.

3. Lay both antlers down on a bench or table. Slide the elastic section of a 24-inch bungee cord into the gap you made in the eye of each screw. Use the pliers to bend the eye closed again. Boom, done.

What’s so ingenious about this rig is that you can carry the antlers by wrapping the bungee cord around your waist like a belt, or wrap the bungee tightly around your pack or climber to both carry the horns and secure extra clothes to on the walk in.

**Rationale (the IMPLICIT reasoning, added):** 1. The rig is designed so you carry a full set of real deer antlers strapped around your waist like a belt or lashed to your pack while hiking in.
2. A set of antlers worn on a person's body closely resembles the rack of a live deer, especially at a distance or in low light.
3. In firearms season other hunters are out shooting at anything that looks like a deer.
4. So walking around wearing antlers while guns are in the woods would expose the hunter to being mistaken for game.

**REAL target (verbatim next text — scored):** (For safety’s sake, do either of the above only during bow season.)

---

### id 345 — base 3.981 → real 3.663 / placebo 4.338  (real−placebo -0.675)
**Context (from DCLM, tail):** …Q&A: Why are sticky notes not good for books?


Why is it not a good idea to use sticky notes in books?


While sticky notes are certainly useful, they can damage paper. The residue they leave behind attracts dirt, causes pages to stick together, and stains paper over time. When sticky notes are pulled from fragile pages in old books, they can also tear the paper.

**Rationale (the IMPLICIT reasoning, added):** 1. Sticky-note residue attracts dirt and stains paper over time.
2. The residue causes pages to stick together, and removing notes can tear fragile pages.
3. Therefore sticky notes actively damage books, especially old or fragile ones.
4. Protecting a library's collection requires marking pages with a method that leaves no residue and cannot tear the paper.

**REAL target (verbatim next text — scored):** To help preserve collections here in the libraries, please use bookmarks or strips of paper to mark your pages.

---

### id 317 — base 4.402 → real 3.94 / placebo 4.567  (real−placebo -0.627)
**Context (from DCLM, tail):** …kooky, oozing graphics of a bunny in a jumpsuit would look on a T-shirt?

Konrad Kirpluk is a U.K.-based illustrator who specializes in using vector graphics in various pop art styles. Commonly seen in his art pieces are hyperbolic characters with graffiti artwork influences. I’m most impressed with Kirpluk’s skill of emulating felt-tip marker strokes in the digital realm.

Implications - Consumers are obsessed with expressing their individuality, as it’s seen as an ideal in first-world nations.

**Rationale (the IMPLICIT reasoning, added):** 1. In first-world nations, consumers are described as obsessed with expressing their individuality, treating it as an ideal.
2. Expressing individuality means signaling a distinct personal identity.
3. Established subcultures carry strong, recognizable identity markers that people adopt to signal who they are.
4. A company selling to identity-seeking consumers therefore has a strong incentive to attach its brand to the identity markers those consumers already crave.

**REAL target (verbatim next text — scored):** As a method of invoking particular identity characteristics, sneaker corporations often align themselves with particular subcultures.

---

### id 327 — base 4.405 → real 4.259 / placebo 4.804  (real−placebo -0.545)
**Context (from DCLM, tail):** …e and seafood industry employee I know is in Paris during this week's Seafood Summit. He and his wife were really looking forward to checking out mega-watt, super Michelin-starred French chef Alain Ducasse's seafood-focussed restaurant, Rech. Within 24 hours of devouring a fruits de mer platter of oysters, mussels, shrimp and clams, his wife was knocked unconscious for 30 minutes, while he couldn't move his fingers - which had curled into a bizarre grip - and both of them were shaking violently.

**Rationale (the IMPLICIT reasoning, added):** 1. The couple ate a fruits de mer platter of shellfish (oysters, mussels, shrimp, clams).
2. Within 24 hours both developed severe symptoms: unconsciousness, loss of finger control, violent shaking.
3. Both partners who shared the same meal fell ill at the same time, pointing to a common dietary source.
4. Such acute symptoms so soon after eating raw/shellfish seafood indicate contamination or toxins in that food.

**REAL target (verbatim next text — scored):** As the hospital told them upon arrival Saturday, a pretty clear case of toxic food poisoning.

---

### id 385 — base 2.626 → real 2.643 / placebo 3.045  (real−placebo -0.401)
**Context (from DCLM, tail):** …e? After a sudden revolution in design a few years back we feel like maybe some designers are just adding more lines to cars they've already created rather than going bold.

Ray's orange Challenger looks pretty sharp, but it's not exactly a "new" design. We have a theory about this. In between big automotive design revolutions companies have to come out with "new" models so they get "new" upgrades, which is some variation of changing the shape of the fog lights and adding a new crease somewhere.

**Rationale (the IMPLICIT reasoning, added):** 1. Between major design revolutions, carmakers still must release "new" models.
2. Those refreshes are minor — reshaping fog lights or adding another crease.
3. This additive habit keeps stacking extra lines onto shapes that are already busy.
4. Layering yet more detail onto an already crowded body pushes its look past clean.

**REAL target (verbatim next text — scored):** In this case, there are so many lines on the cars already that adding more just makes them look stranger.

---

### id 368 — base 3.601 → real 3.415 / placebo 3.804  (real−placebo -0.389)
**Context (from DCLM, tail):** …The Boston Globe



Now marching online

The American Legislative Exchange Council, the group funded in part by the conservative Koch brothers to draft bills for state legislators, is the type of shadowy, under-the-radar operation that exerts huge influence over state lawmaking. It has written the “stand your ground’’ statutes in many states, and pushed voter-ID laws that strike at poor and minority citizens.

**Rationale (the IMPLICIT reasoning, added):** 1. ALEC is a shadowy, under-the-radar operation funded in part by the Koch brothers.
2. It quietly drafts model bills that state legislators adopt, wielding huge influence.
3. Its stand-your-ground and voter-ID laws are framed as harmful to poor and minority citizens.
4. A group that works unseen while doing damage invites a fitting, unseen reversal.

**REAL target (verbatim next text — scored):** So it’s more than just desserts that ALEC would get its comeuppance from another under-the-radar group

---

### id 363 — base 3.31 → real 3.263 / placebo 3.606  (real−placebo -0.342)
**Context (from DCLM, tail):** …ected the location, if the light is right it rarely takes me more than 10 minutes to set up, shoot, tear down, and move on. If the light isn't right but I think it might be better at another time, I put it into my mental checklist of things to look at again when the opportunity arises and I move on.

One of the reasons I love living where I do is that, between the hills and valleys and the lattitude, we not only have four seasons but we also have some fairly predictable light and other features.

**Rationale (the IMPLICIT reasoning, added):** 1. The photographer keeps a mental checklist of scenes whose light isn't right yet, to revisit when the opportunity arises.
2. Where he lives has four seasons plus fairly predictable light and weather features.
3. Predictable seasonal conditions mean each type of weather recurs at roughly foreseeable times.
4. So the moment a particular desired condition will next appear can be worked out ahead of time.

**REAL target (verbatim next text — scored):** If I know a scene would look better in the fog, I know about when I'll have to schedule a trip back.

---

### id 456 — base 3.417 → real 3.351 / placebo 3.642  (real−placebo -0.292)
**Context (from DCLM, tail):** …Why do you put the chisel on the front or left side of the blade?

This is an Emerson signature. Being the knifemaker who brought the chisel grind to worldwide recognition, we are often asked; Why do you put the grind on the opposite side of a traditional Japanese Chef’s knife? The answer is simple….We are not making chef’s knives. Our knives are hard knives meant for hard users. We do not cut many tomatoes.

**Rationale (the IMPLICIT reasoning, added):** 1. Emerson is asked why they put the chisel grind on the opposite side from a traditional Japanese chef's knife.
2. They answer that they are not making chef's knives; theirs are hard knives built for hard users.
3. They stress they do not cut many tomatoes, signaling that fine food-preparation slicing is not what these knives are for.
4. The reason a chef's knife's grind must sit on a particular side is tied to that delicate food-slicing purpose, which their tool/weapon use does not share.

**REAL target (verbatim next text — scored):** Our tests and those of a major government agency determined that there was no difference between right and left side grinds for use as a tool or weapon.

---

### id 379 — base 3.59 → real 3.569 / placebo 3.813  (real−placebo -0.244)
**Context (from DCLM, tail):** …sidewalk sale downtown, the buses were free all over Lawrence. Why did they decide this and why, at least, on one day where they could make some revenue from these buses with all the people downtown?

Karin Rexroad, the city transit administrator, said: "People are more likely to try the transit system on the day the fare is waived and they are going to a known event. Riders who take advantage of the incentive, free ridership, will experience two benefits of the T, no traffic or parking worries.

**Rationale (the IMPLICIT reasoning, added):** 1. Waiving the fare makes people more willing to try the transit system, especially for a known event.
2. First-time riders who take the free-ride incentive experience the T's benefits — no traffic or parking hassle.
3. A good first experience tends to turn a newcomer into a returning rider.
4. Those return trips would happen well after the single free-fare day.

**REAL target (verbatim next text — scored):** There is value in the future rides created by citizens riding the T for the first time to the annual sidewalk sale.

---

### id 420 — base 2.892 → real 2.915 / placebo 3.144  (real−placebo -0.228)
**Context (from DCLM, tail):** …ort money moving from wages to health plans, so economists assume the reverse would be true as well. But it sounded like there was no data to support that right now. Presumably that's because health plan costs have not dropped, making it an untested hypothesis.

I (and I suspect others like me) can supply anecdotal evidence of this, though. My husband and I are a two income couple and I have a relatively stable job at a University with good benefits, so we both have health insurance from my job.

**Rationale (the IMPLICIT reasoning, added):** 1. Heather offers her own household as anecdotal evidence for the claim that money not spent on employer health coverage can turn into higher pay.
2. She holds a stable University job whose benefits already cover both her and her husband.
3. That means her husband gains nothing extra from any health plan his own employer would provide.
4. A redundant benefit he does not need is something he could offer to give up in exchange for compensation he does value.

**REAL target (verbatim next text — scored):** My husband has repeatedly negotiated with employers for higher wages or more paid vacation based on the fact that he would not be using their health plan.

---

### id 320 — base 3.432 → real 3.334 / placebo 3.54  (real−placebo -0.206)
**Context (from DCLM, tail):** …until they "ship" (literally shipping the game, or they release it for download), and revenue is a really important thing for valuation purposes. It is, as TB mentioned, a decent source of data for financial models (for the purposes of investors, or other internal business decisions). Pre-orders can indicate a level of interest or a willingness-to-pay of a consumer which should feed into any good financial model.

I also think there are some additional considerations with respect to pre-orders.

**Rationale (the IMPLICIT reasoning, added):** 1. Pre-orders are committed and paid before a game launches, so their count is known in advance.
2. The comment establishes that pre-orders measure how many consumers are interested and willing to pay, i.e. expected demand.
3. Games differ in delivery: some run on the buyer's own machine, but online titles depend on infrastructure the publisher must provision.
4. For such online titles, the infrastructure and support burden rises directly with how many players actually show up.

**REAL target (verbatim next text — scored):** When speaking specifically about games which require company-run servers (MMO's and some FPS's), pre-orders are a great way to determine how to staff the IT teams and how much hardware to buy/lease.

---

### id 373 — base 2.605 → real 2.758 / placebo 2.939  (real−placebo -0.182)
**Context (from DCLM, tail):** …I got a letter from the IRS. Does that mean that I am being audited?

That depends but in most cases the answer is no. In this age of computers, the IRS is generating more and more letters based on a computer matching information on your return to information reported to them by employers, bank, and other third parties. If the computer is not able to find an item on your return the computer will write you a nice letter requesting clarification of where the item is included in your return.

**Rationale (the IMPLICIT reasoning, added):** 1. An IRS letter, in most cases, does not signal an audit.
2. The IRS computer matches return entries against employer, bank, and other third-party reports.
3. When it cannot locate an item, it merely sends a letter asking where that item appears.
4. Such a request implies the entry is probably present, just not where the computer looked.

**REAL target (verbatim next text — scored):** Hopefully the item has been included in the return so it is just a simple letter back explaining where in the return you have included the item in question.

---

### id 429 — base 3.588 → real 4.095 / placebo 4.095  (real−placebo +0.001)
**Context (from DCLM, tail):** …un boxee anymore since there's no hard drive in it. Very much like the iPhone with limited capacity, there are limited options at your disposal when it comes to attempting to "jailbreak" it. For now, we have not seen any news of a Boxee jailbreak or anything like that in the hacking community.

In this case your only other option is the Boxee Box, the Logitech Revue, or the Roku. While we've reviewed all of these items, Boxee still seems like the best combination of features & hardware specs vs.

**Rationale (the IMPLICIT reasoning, added):** 1. The context says the new Apple TV will not run Boxee.
2. The reason given is that the new unit has no hard drive and only limited storage.
3. This implies Boxee needs a device that actually includes a hard drive.
4. An earlier-generation Apple TV did ship with a built-in hard drive.

**REAL target (verbatim next text — scored):** price. Your final option is to grab an older 160GB ATV and run Boxee

---

