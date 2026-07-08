# Completeness test on real DCLM multi-step reasoning (n=41)

Real DCLM docs whose continuation needs a **>=3-step** reasoning chain. Score the REAL target under DCLM-1.4B given:
`base` (context only) vs `+complete` (full chain) vs `+incomplete` (same chain, 1-2 load-bearing MIDDLE steps deleted)
vs `+placebo` (another doc's complete chain). delta<0 = lowers the real target's perplexity.

## Headline (1.4B judge)
| contrast | mean | meaning |
|---|---:|---|
| complete − base | +0.048 (51%) | adding the complete chain ~neutral vs no rationale (format cost offsets) |
| **complete − placebo** | **-0.698** | RELEVANT reasoning helps a lot; an irrelevant chain HURTS (placebo−base +0.745) |
| **complete − incomplete** | **+0.004** (17/41) | **COMPLETENESS makes ~no difference** — the model fills the deleted step itself |

**Honest read:** on zero-shot perplexity of real multi-step targets, what matters is that the reasoning is
PRESENT and RELEVANT (vs a wrong-topic chain, which hurts) — NOT whether it is gap-free. A gap-broken chain
predicts the target as well as the complete one. Completeness-per-se is not demonstrable on this metric;
it may still matter for TRAINING (a model internalizing gap-free chains) — the open question.

---

### id 100611 — complete 1.972 / incomplete 2.353 / placebo 2.851 (base 2.637)
**Context (tail):** …printing times were significantly shorter with the same negatives. I asked a glass person why and he said that plate glass is inherently stronger than ordinary glass and actually has a light reducing effect, he simply asked if the sides of the glass were at all green, that is an indication of "plate" glass which reduces light transmission.

Also, Azo is largely sensitive to UV and glass tends to block UV - that's why UV lenses are made of quartz.

**COMPLETE rationale:** 1. Azo printing paper is largely sensitive to UV light rather than to ordinary visible light.
2. Glass tends to block UV, and the thicker 1/4" plate glass reduces light transmission more than the thinner 1/8" clear glass, so the two pieces differ mainly in how much UV they pass.
3. Because Azo's exposure depends on UV, that difference in UV transmission between the two glasses translates into a large difference in printing time — which is exactly why switching glasses shortened the times so much.
4. Ordinary enlarging paper, by contrast, is sensitive to visible light and is not appreciably sensitive to UV.
5. Since ordinary paper does not rely on UV, the two glasses' difference in UV transmission would have little effect on how much usable light reaches it.

**INCOMPLETE (gap-broken):** 1. Azo printing paper is largely sensitive to UV light rather than to ordinary visible light.
2. Glass tends to block UV, and the thicker 1/4" plate glass reduces light transmission more than the thinner 1/8" clear glass, so the two pieces differ mainly in how much UV they pass.
5. Since ordinary paper does not rely on UV, the two glasses' difference in UV transmission would have little effect on how much usable light reaches it.

**REAL target (verbatim):** I'd bet that if you were using ordinary enlarging paper, you wouldn't see nearly as much difference between the 2 pieces of glass.

---

### id 101750 — complete 4.648 / incomplete 4.967 / placebo 6.497 (base 4.854)
**Context (tail):** …art of the "great recession," when the top rate ranged between 35% and 39%, average growth was 3%. The long-term slowdown is related to the fact that since the early 1980s a larger and larger share of total income has gone to the top (the richest 1% of Americans got 10% of total income in 1980, and get more than 20% now), leaving the vast middle class with insufficient purchasing power to boost the economy without eventually going deep into debt.

**COMPLETE rationale:** 1. The context reports that from 1951 to 1980 the top marginal tax rate was very high (70-91%) and average growth was 3.7%, whereas after the early 1980s the top rate was cut sharply (to 35-39%) and average growth fell to 3%.
2. Cutting the top marginal rate after the early 1980s let the highest earners keep a much larger share of their pre-tax income.
3. Over that same post-1980 period the top 1%'s share of total income more than doubled (from 10% to over 20%) — exactly the concentration of income at the top that the context says drained the middle class of purchasing power and produced the long-term slowdown (the 'trend').
4. Because the top-rate cuts directly enabled and accelerated that concentration of income at the top, the tax changes pushed in the same direction as the trend rather than counteracting it.

**INCOMPLETE (gap-broken):** 1. The context reports that from 1951 to 1980 the top marginal tax rate was very high (70-91%) and average growth was 3.7%, whereas after the early 1980s the top rate was cut sharply (to 35-39%) and average growth fell to 3%, while the top 1%'s income share more than doubled (from 10% to over 20%).
4. Because the top-rate cuts pushed in the same direction as the trend rather than counteracting it...

**REAL target (verbatim):** Tax rates exacerbated the trend.

---

### id 100989 — complete 2.573 / incomplete 2.821 / placebo 3.265 (base 3.217)
**Context (tail):** …lt in topology stating that a continuous vector field on a sphere is always zero somewhere. The name comes from the fact that you can't flatten all the hair on a hairy ball, like a tennis ball, there will always be a tuft somewhere (where the tangential projection of the hair is zero). An immediate corollary to this theorem is that for any continuous map f of the sphere into itself there is a point x such that f(x)=x or f(x) is the antipode of x.

**COMPLETE rationale:** 1. The hairy ball theorem states that any continuous vector field lying tangent to a sphere must equal zero at at least one point on that sphere.
2. The surface of the Earth is, in shape, a sphere.
3. At every location on the Earth's surface the wind blows horizontally along the ground, so the wind at each point is a vector lying tangent to the spherical surface; taken together the winds form a tangent vector field on the sphere.
4. Wind direction and speed change gradually from one place to the next, so this tangent vector field is continuous.
5. A continuous tangent vector field on a sphere meets the theorem's hypothesis, so by the theorem it must be zero at some point — a location where the wind vector has zero magnitude.

**INCOMPLETE (gap-broken):** 1. The hairy ball theorem states that any continuous vector field lying tangent to a sphere must equal zero at at least one point on that sphere.
2. The surface of the Earth is, in shape, a sphere.
5. A continuous tangent vector field on a sphere meets the theorem's hypothesis, so by the theorem it must be zero at some point — a location where the wind vector has zero magnitude.

**REAL target (verbatim):** Another corollary is that at any moment somewhere on the Earth there is no wind.

---

### id 101991 — complete 2.805 / incomplete 3.019 / placebo 3.465 (base 3.057)
**Context (tail):** …ing a more restrictive DRM standard based on ogg vorbis with some DRM-ish layer? Does this mean that the only legal streaming format will then be ogg-DRM-vorbis?

The RIAA and the other middlemen must really be worried that they are going to be cut out of the equation when the artists realise that they don't need to give up 99% of the revenues and could just as easily hire an online company to distribute their works for them at a much lower cost.

**COMPLETE rationale:** 1. Right now artists can distribute their music through cheap, open online channels, keeping most of the revenue and cutting the RIAA and other middlemen out of the equation.
2. A proprietary DRM format is not free to use: it is owned and controlled by whatever company holds its rights, and using it requires a license from that owner.
3. If a law mandated one specific DRM format as the only legal way to distribute music online, then every distributor, including artists releasing directly, would be forced to use that format.
4. To use the mandated format they would all have to obtain and pay for a license from the company that owns it.
5. That rights-holding company would therefore sit between the artists and their audience as a gatekeeper everyone is legally required to go through and pay.

**INCOMPLETE (gap-broken):** 1. Right now artists can distribute their music through cheap, open online channels, keeping most of the revenue and cutting the RIAA and other middlemen out of the equation.
2. A proprietary DRM format is not free to use: it is owned and controlled by whatever company holds its rights, and using it requires a license from that owner.
5. That rights-holding company would therefore sit between the artists and their audience as a gatekeeper everyone is legally required to go through and pay.

**REAL target (verbatim):** Legislating a certain format for the online distribution of music would turn the tables again and force the artists to deal with another middleman

---

### id 101421 — complete 3.687 / incomplete 3.856 / placebo 5.349 (base 3.506)
**Context (tail):** …implemented what became a seismic shift in retailing. They began opening what they called Economy stores. By 1929 the A&P's expansive presence and aggressive discounting had made them such a dominant force that they became the first company to ever rack up a billion dollars in annual sales.

By then chain stores like the A&P had become political targets. By the 1940's the Hartford brothers were found to be guilty of criminal anti-trust activity.

**COMPLETE rationale:** 1. A&P became dominant by aggressively discounting — selling goods at unusually low prices.
2. Those low prices let A&P capture huge market share (the first company ever to reach $1 billion in annual sales) and drove many small independent stores out of business.
3. Because it was seen as destroying small businesses, chain stores like A&P became political targets, and lawmakers moved to curb it through anti-trust law.
4. The Hartfords were then prosecuted and found guilty of criminal anti-trust activity for the very pricing strategy that had made them dominant.

**INCOMPLETE (gap-broken):** 1. A&P became dominant by aggressively discounting — selling goods at unusually low prices.
4. The Hartfords were then prosecuted and found guilty of criminal anti-trust activity.

**REAL target (verbatim):** Their crime? Low prices.

---

### id 100081 — complete 2.742 / incomplete 2.9 / placebo 3.398 (base 3.106)
**Context (tail):** …ing to close them all.) So I'd like to propose we simply add `""fields""` as a recognized argument for `get_posts()`, i.e. {{{ $posts = get_posts(array( 'fields' => 'ID,post_title', 'post_type' => 'movie', 'post_status' => 'publish', 'order' => 'ASC', 'posts_per_page' => -1 )); }}} I know I could make the same argument for `joins`, `where`, `orderbys` et. al. but I'd argue this is enough of a special case it could really use some early attention.

**COMPLETE rationale:** 1. As it currently works, get_posts() retrieves every column of each matching post, including the large post_content field, even when only the post ID is wanted.
2. Hauling post_content and the other unused fields from MySQL over to WordPress on every call wastes time and memory on data that is immediately thrown away.
3. A site's home page is requested very frequently, so the get_posts() query behind it runs an enormous number of times.
4. If that home page changes often, its output cannot be cached, so each visit must actually re-execute the query against the database instead of serving a stored copy.
5. Because the query is re-run on nearly every one of those many visits, the small per-call waste of fetching all fields is multiplied into a large aggregate load.
6. Adding a 'fields' argument that returns only the ID removes that wasted transfer from each of those many uncached executions.

**INCOMPLETE (gap-broken):** 1. As it currently works, get_posts() retrieves every column of each matching post, including the large post_content field, even when only the post ID is wanted.
2. Hauling post_content and the other unused fields from MySQL over to WordPress on every call wastes time and memory on data that is immediately thrown away.
3. A site's home page is requested very frequently, so the get_posts() query behind it runs an enormous number of times.
6. Adding a 'fields' argument that returns only the ID removes that wasted transfer from each of those many uncached executions.

**REAL target (verbatim):** For a query on the home page of a high traffic site that changes often enough to not be able to be cached this tiny change could make a major difference in performance.

---

### id 102589 — complete 3.198 / incomplete 3.317 / placebo 3.968 (base 3.619)
**Context (tail):** …omali government, confirmed recent reports that the leadership structure of the disbanded SCIC was still largely intact. The 72-year-old cleric sat alongside Sheikh Sharif Sheikh Ahmed, regarded as the SCIC's second-in-command, who said the aim of the 10-day meeting was to create "a political organisation that liberates the country ...".

The meeting came a week after the closure of a government-sponsored reconciliation conference in the capital.

**COMPLETE rationale:** 1. The interim Somali government, propped up by the occupying Ethiopian army, had just concluded its own government-sponsored reconciliation conference.
2. Only a week later the SCIC opposition, its leadership structure still intact, convened an entirely separate conference in Eritrea instead of taking part in that reconciliation.
3. At its conference the opposition demanded an immediate withdrawal of Ethiopian troops and set out to build an organization to "liberate the country."
4. Those aims run directly against the Ethiopian-backed interim government's position, so the two sides are pursuing incompatible goals through rival gatherings rather than one shared process.

**INCOMPLETE (gap-broken):** 1. The interim Somali government, propped up by the occupying Ethiopian army, had just concluded its own government-sponsored reconciliation conference.
4. Those aims run directly against the Ethiopian-backed interim government's position, so the two sides are pursuing incompatible goals through rival gatherings rather than one shared process.

**REAL target (verbatim):** The separate talks are indicative of the gulf between the two groups

---

### id 102891 — complete 1.94 / incomplete 2.056 / placebo 2.719 (base 2.323)
**Context (tail):** …gets away with making minimal changes to its infrastructure, you have to replace just about everything you own. What was a mystery, is now crystal clear and of course its always about money, it isn't necessarily about DirecTV's money, this time. You see RealD owns the patent on frame compatible 3D formats like side by side, and if a display or receiver manufacturer wants its EDID on the list of supported devices, they have to pay for that right.

**COMPLETE rationale:** 1. Frame compatible 3D was designed to run over existing HD equipment, so an old AV receiver's hardware is physically capable of carrying the signal.
2. Therefore older receivers being blocked from passing DirecTV's 3D cannot be explained by a technical or hardware limitation.
3. RealD owns the patent on frame compatible 3D formats, and a device only appears on the list of supported devices (its EDID recognized) if its manufacturer pays RealD to license that patent.
4. So whether a given receiver is allowed to display the 3D signal is decided by whether its maker has licensed RealD, and unlicensed devices are precisely the ones left off the supported list.
5. Since the exclusion tracks licensing status rather than which receiver you happen to own, the real function of the block is to compel manufacturers into licensing RealD's patents.

**INCOMPLETE (gap-broken):** 1. Frame compatible 3D was designed to run over existing HD equipment, so an old AV receiver's hardware is physically capable of carrying the signal.
2. Therefore older receivers being blocked from passing DirecTV's 3D cannot be explained by a technical or hardware limitation.
5. Since the exclusion tracks licensing status rather than which receiver you happen to own, the real function of the block is to compel manufacturers into licensing RealD's patents.

**REAL target (verbatim):** So it isn't that DirecTV wants to prevent you from using your old receiver as much as it is about preventing those who don't license RealD's patents from being able to display 3D.

---

### id 102659 — complete 2.982 / incomplete 3.092 / placebo 3.532 (base 3.229)
**Context (tail):** …opened the ketchup up to contamination by all kinds of microbes. But how is that any different from standard bottling? Well in a legitimate bottling operation, you kill those microbes with heat first. And as for why unrefrigerated bottles of ketchup don't explode on restaurant tables, they're being opened frequently, and hopefully used up and replaced.

The counterfeit ketchup sitting in New Jersey, however, had none of these things going for it.

**COMPLETE rationale:** 1. Transferring the ketchup from one bottle to another introduced live microbes into the counterfeit product.
2. Unlike a legitimate bottling operation, no heat step was applied to kill them, so those microbes stayed alive inside the ketchup.
3. Unlike restaurant bottles that are opened and used up frequently, these bottles sat sealed and undisturbed, so nothing inside could vent.
4. Alive and sealed in, the microbes feed on the ketchup, and that metabolism releases gas as a byproduct.
5. Because the bottle is sealed, the gas cannot escape and steadily accumulates in the fixed container volume.

**INCOMPLETE (gap-broken):** 1. Transferring the ketchup from one bottle to another introduced live microbes into the counterfeit product.
2. Unlike a legitimate bottling operation, no heat step was applied to kill them, so those microbes stayed alive inside the ketchup.
3. Unlike restaurant bottles that are opened and used up frequently, these bottles sat sealed and undisturbed, so nothing inside could vent.
5. Because the bottle is sealed, the gas cannot escape and steadily accumulates in the fixed container volume.

**REAL target (verbatim):** So when the microbes started chowing down and spewing out gas, the pressure built up and boom: spontaneous ketchup explosion.

---

### id 102162 — complete 2.813 / incomplete 2.899 / placebo 3.395 (base 3.105)
**Context (tail):** …g their work effort (including their choice of when to retire). Given this flexibility, the individual simultaneously determines optimal levels of current consumption, labor effort, and an optimal financial investment strategy at each point in his life cycle. We show that labor and investment choices are intimately related. The ability to vary labor supply ex post induces the individual to assume greater risks in his investment portfolio ex ante.

**COMPLETE rationale:** 1. The model shows that the freedom to adjust one's work effort after the fact, including choosing when to retire, leads a person to take on more investment-portfolio risk beforehand.
2. How much of this freedom a person retains depends on how many working years still lie ahead: with a long career remaining, work effort and retirement date are still adjustable, but near the end of a career they are essentially fixed.
3. Therefore the amount of labor-supply flexibility a person has declines with age, so a younger person has far more of it than an older person.
4. Applying the model's mechanism from step 1, whoever holds more labor flexibility rationally chooses a riskier investment portfolio.

**INCOMPLETE (gap-broken):** 1. The model shows that the freedom to adjust one's work effort after the fact, including choosing when to retire, leads a person to take on more investment-portfolio risk beforehand.
4. Applying the model's mechanism from step 1, whoever holds more labor flexibility rationally chooses a riskier investment portfolio.

**REAL target (verbatim):** The model explains why the young (enjoying greater labor flexibility over their working lives) may take greater investment risks than the old.

---

### id 102770 — complete 2.702 / incomplete 2.753 / placebo 3.719 (base 2.784)
**Context (tail):** …her than via a control panel for the company that I have webspace with (1and1).
It is my understanding that the localhost on the office server has somehow been given a domain name of theukoffice .com and this is what is causing the error.

Now i know it would be simple just to get them to browse to the .co.uk domain but some of the links here are hard coded as .com rather than using /whatever/wherever.php or ../here/there.php site relative links.

**COMPLETE rationale:** 1. The site's internal navigation links are hard-coded as absolute URLs on the .com domain instead of relative paths.
2. Inside the office, the .com domain does not point to the public web host, because the office server's localhost was given the name theukoffice.com.
3. So any request to the .com domain from an office machine is answered by the local office server, which stores an outdated copy of the site.
4. Even a user who deliberately opens the correct .co.uk copy will, on clicking any internal link, have the browser request that link's hard-coded .com address.

**INCOMPLETE (gap-broken):** 1. The site's internal navigation links are hard-coded as absolute URLs on the .com domain instead of relative paths.
4. Even a user who deliberately opens the correct .co.uk copy will, on clicking any internal link, have the browser request that link's hard-coded .com address.

**REAL target (verbatim):** So pretty soon they end up back on the office server version of the site.

---

### id 100649 — complete 2.511 / incomplete 2.546 / placebo 2.966 (base 2.517)
**Context (tail):** …an annulment, as well. For example, if a spouse lies about their ability to have kids (i.e. they physically can't), or hides that they have a sexually transmitted disease, these might be grounds for an annulment based on fraud or concealment.

The above noted requirements and the examples probably demonstrate why most annulments happen shortly after marriage, and in circumstances where the parties didn't know each other too long before marrying.

**COMPLETE rationale:** 1. Per the context, most annulments occur shortly after the wedding, typically between people who had not known each other for long before marrying.
2. A marriage that is dissolved so soon gives the couple little time to jointly accumulate significant assets or a shared home.
3. Such a brief union also makes it very unlikely that the couple had children together.
4. In an ordinary marital dissolution, the heavily contested matters are splitting up shared marital assets and setting arrangements for any children.

**INCOMPLETE (gap-broken):** 1. Per the context, most annulments occur shortly after the wedding, typically between people who had not known each other for long before marrying.
4. In an ordinary marital dissolution, the heavily contested matters are splitting up shared marital assets and setting arrangements for any children.

**REAL target (verbatim):** Lastly, if a short term marriage is involved, an annulment would probably not involve any major issues such as division of property, child custody and support

---

### id 101184 — complete 3.337 / incomplete 3.368 / placebo 3.559 (base 3.156)
**Context (tail):** …white and thin, measuring 2 mm in length. Maggots develop fully within five to seven days and enter the pupal stage. , During the larval stage, they feed ravenously on the material on which the eggs were laid. Lesser house flies require a period of nine to 14 days.

Lesser houseflies move slightly faster than other species and fly in jerky, darting patterns. Lesser house fly eggs are capable of floating and can be found resting on standing water.

**COMPLETE rationale:** 1. Female lesser house flies are strongly attracted to decaying fecal matter and livestock waste, which they visit to lay eggs and where the larvae feed.
2. Fecal matter and decaying organic material are reservoirs of disease-causing microorganisms.
3. As a fly walks and feeds on this contaminated material, those microorganisms cling to its legs, body hairs, and mouthparts.
4. The same flies then move on and land on other surfaces, including food and places where people live and eat.
5. As they land, the microorganisms carried on their bodies are deposited onto that food and those surfaces.

**INCOMPLETE (gap-broken):** 1. Female lesser house flies are strongly attracted to decaying fecal matter and livestock waste, which they visit to lay eggs and where the larvae feed.
4. The same flies then move on and land on other surfaces, including food and places where people live and eat.
5. As they land, the microorganisms carried on their bodies are deposited onto that food and those surfaces.

**REAL target (verbatim):** Like common houseflies, lesser houseflies are known carriers of pathogens resulting in human ailments

---

### id 102524 — complete 2.916 / incomplete 2.944 / placebo 3.368 (base 3.172)
**Context (tail):** …ard, thanks to a feckless and ineffective American media, such as Cal Thomas' assertion (quoting Reagan) that the people aren't under-taxed but we have deficits because the government spends too much. A clear-eyed view of the facts/numbers shows that this belief is 180 degrees wrong. Historically, the federal government spends 22 percent of the gross domestic product — Reagan, Bush one, Bush two, Clinton and Obama, all administrations since WWII.

**COMPLETE rationale:** 1. The context states that, historically, the federal government has spent about 22 percent of GDP, and that this spending share held roughly steady across every administration since WWII, including Reagan, both Bushes, Clinton, and Obama.
2. A budget's balance is the difference between the revenue the government collects and the amount it spends.
3. Because the spending share was essentially the same under every administration, the spending side of the ledger cannot explain why some administrations ran deficits while others did not.
4. That means the only variable left that can account for the difference between a deficit and a surplus is how much revenue was collected.
5. Clinton was the administration that produced balanced budgets and even surpluses, unlike the others that ran deficits.

**INCOMPLETE (gap-broken):** 1. The context states that, historically, the federal government has spent about 22 percent of GDP, and that this spending share held roughly steady across every administration since WWII, including Reagan, both Bushes, Clinton, and Obama.
2. A budget's balance is the difference between the revenue the government collects and the amount it spends.
5. Clinton was the administration that produced balanced budgets and even surpluses, unlike the others that ran deficits.

**REAL target (verbatim):** So how did Clinton manage balanced budgets and even surpluses? His tax policies (which did not stifle growth or kill jobs, as evidenced by the best economy of our lifetimes under Clinton) brought in revenues of 22

---

### id 102443 — complete 4.25 / incomplete 4.271 / placebo 5.313 (base 4.592)
**Context (tail):** …y Powder Metallurgy", journal = "Applied Spectroscopy", volume = "8", number = "3", year = "1954-08-01T00:00:00", abstract = "M-252 nickel base high temperature alloy is completely analyzed by instruments at the general Electric River Works. Molybdenum, cobalt, iron, and chromium are determined by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy and carbon by the high frequency induction furnace.

**COMPLETE rationale:** 1. M-252 is described as a nickel-base high-temperature alloy, so nickel is its predominant constituent — the balance of the composition.
2. The analytical scheme measures molybdenum, cobalt, iron, and chromium by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy; and carbon by the high-frequency induction furnace.
3. Every element named is covered by one of those instruments, but nickel itself appears nowhere in the instrument list — it is the only principal constituent left unmeasured.
4. Because the paper states the alloy is "completely analyzed," the weight fractions of all constituents must together account for 100% of the material.
5. Nickel therefore does not require its own instrumental reading; its value is fixed automatically once all the other elements have been quantified.

**INCOMPLETE (gap-broken):** 1. M-252 is described as a nickel-base high-temperature alloy, so nickel is its predominant constituent — the balance of the composition.
2. The analytical scheme measures molybdenum, cobalt, iron, and chromium by X-ray fluorescence; titanium, aluminum, manganese, and silicon by ultraviolet emission spectroscopy; and carbon by the high-frequency induction furnace.
5. Nickel therefore does not require its own instrumental reading; its value is fixed automatically once all the other elements have been quantified.

**REAL target (verbatim):** The remainder, nickel, is calculated by difference.

---

### id 101447 — complete 2.986 / incomplete 2.991 / placebo 3.512 (base 3.126)
**Context (tail):** …portedly reopened.

Fighting in Damascus has been increasing for several weeks, with the rebels edging ever closer to the heart of power despite suffering repeated strikes. Even regime figures admit the army no longer has a hold over almost all the eastern and southern areas of the city, where rebels have taken over at least two army bases.

Perhaps more importantly, the increasing pressure in the capital has ramifications for the wider conflict.

**COMPLETE rationale:** 1. Rebels are steadily advancing in and around Damascus, and the regime has already lost its grip on most of the eastern and southern city, where rebels have seized army bases.
2. Because Damascus is the seat of Assad's power, this mounting pressure threatens the regime's survival and forces it to prioritize defending the capital above all else.
3. The regime's army is finite and already stretched thin, so reinforcing the capital is only possible by drawing forces away from distant, rebel-dominated fronts like the north.
4. The regime's irreducible core is its capital plus the Alawite coastal region around Latakia, the heartland of the sect the ruling Assads belong to — the territory it must hold to survive.

**INCOMPLETE (gap-broken):** 1. Rebels are steadily advancing in and around Damascus, and the regime has already lost its grip on most of the eastern and southern city, where rebels have seized army bases.
4. The regime's irreducible core is its capital plus the Alawite coastal region around Latakia, the heartland of the sect the ruling Assads belong to — the territory it must hold to survive.

**REAL target (verbatim):** The regime has proved remarkably reluctant to pull garrisons and bases out of northern areas largely controlled by the rebels, but it may soon be forced to consolidate in Damascus and up the highway to Latakia

---

### id 102695 — complete 2.929 / incomplete 2.933 / placebo 3.119 (base 2.88)
**Context (tail):** …anked, number one, by a wide margin, was your high school-aged nieces' and Richard Lawson's favorite channel: ABC Family. Of it's original prime-time 55% of it had gays in it. Yes, Greek, Pretty Little Liars, Make It or Break It and all the other embarrassments on your DVR all have gay characters.

Meanwhile those that scored the lowest on the Index—CBS, USA, A&E, and TBS—all have audiences that skew older than their more gay-minded counterparts.

**COMPLETE rationale:** 1. GLAAD's Network Responsibility Index shows the networks with the youngest audiences (the CW, ABC Family, Fox) carry the highest percentage of gay and lesbian characters, while the networks that skew oldest (CBS, USA, A&E, TBS) carry the least.
2. A network tailors its programming to the tastes of the demographic it targets.
3. Therefore the heavy gay content on the youth-oriented networks reflects that young viewers genuinely like and accept gay characters.
4. The media tastes and social attitudes a generation forms in its youth tend to persist as that generation ages rather than reverse.

**INCOMPLETE (gap-broken):** 1. GLAAD's Network Responsibility Index shows the networks with the youngest audiences (the CW, ABC Family, Fox) carry the highest percentage of gay and lesbian characters, while the networks that skew oldest (CBS, USA, A&E, TBS) carry the least.
4. The media tastes and social attitudes a generation forms in its youth tend to persist as that generation ages rather than reverse.

**REAL target (verbatim):** What does that mean? Well, what we knew all along—that the kids love the gays and that as they grow up, the entertainment aimed at their generation will probably continue to have just as many gays as you would want to see in the mainstream media.

---

### id 102626 — complete 2.231 / incomplete 2.226 / placebo 2.526 (base 2.32)
**Context (tail):** …gulation of either neurotransmitter system results in abnormal levels of neural activity. Mutations in genes involved in glutamate neurotransmission could lead to this kind of imbalance, and therefore confer a risk of autism. In the present study, Dr. Wang and colleagues will conduct a high-throughput genetic screen in a cohort of autistic patients, looking for DNA sequence variants in 38 genes known to be involved in glutamate neurotransmission.

**COMPLETE rationale:** 1. Glutamate is the principal excitatory neurotransmitter, and its activity is normally balanced by inhibitory neurotransmitters; disrupting that balance produces abnormal levels of neural activity.
2. Mutations in genes involved in glutamate neurotransmission could cause exactly this imbalance and therefore raise the risk of autism — this is the hypothesis being tested.
3. If that hypothesis is true, then autistic individuals should carry these disease-causing mutations in the glutamate genes at a higher rate than people without autism.
4. Such causal mutations would show up as rare DNA sequence variants when the 38 glutamate-related genes are sequenced in the high-throughput screen.

**INCOMPLETE (gap-broken):** 1. Glutamate is the principal excitatory neurotransmitter, and its activity is normally balanced by inhibitory neurotransmitters; disrupting that balance produces abnormal levels of neural activity.
2. Mutations in genes involved in glutamate neurotransmission could cause exactly this imbalance and therefore raise the risk of autism — this is the hypothesis being tested.
4. Such causal mutations would show up as rare DNA sequence variants when the 38 glutamate-related genes are sequenced in the high-throughput screen.

**REAL target (verbatim):** If the hypothesis that glutamate is involved in autism is correct, they expect to find multiple rare sequence variants of these genes in autistic patients, compared to a non-autistic control sample.

---

### id 102491 — complete 2.801 / incomplete 2.789 / placebo 3.959 (base 3.36)
**Context (tail):** …echanic had told me that there were a few other things wrong with it, the radiator was leaking, so was the water pump. And that the PCM (I think that's what it was called) should be replaced, but I needed my vehicle to move, and couldn't wait for them to order the part. Within 2 days of being 800 miles away from this shop in my new home. My Jeep won't turn over. The lights are on, the radio works, even the heat works. But I can't get it to start.

**COMPLETE rationale:** 1. The engine won't crank, but the lights, radio, and heat all work, so the battery is charged and supplying power to the car's electrical system.
2. The battery is also brand new, which rules out a dead or worn-out battery as the cause.
3. A single click when turning the key to start is the sound of the starter solenoid engaging while the starter motor fails to spin the engine.
4. So the problem is not a lack of electrical power (the accessories work) and not the engine computer (which wouldn't produce a solenoid click) — the fault must lie between the charged battery and the starter's ability to crank.

**INCOMPLETE (gap-broken):** 1. The engine won't crank, but the lights, radio, and heat all work, so the battery is charged and supplying power to the car's electrical system.
4. So the problem is not a lack of electrical power (the accessories work) and not the engine computer (which wouldn't produce a solenoid click) — the fault must lie between the charged battery and the starter's ability to crank.

**REAL target (verbatim):** I would be looking at a faulty starter or a bad battery connection

---

### id 101673 — complete 2.903 / incomplete 2.89 / placebo 3.436 (base 2.868)
**Context (tail):** …d gas being pushed out into space by the remnants of the star's core, visible as a small bright dot in the middle of the cloud, NASA said Friday.

Planetary nebulae have nothing to do with planets -- the term is a relic from a time when early observers with small telescopes could only see undefined, hazy objects they thought looked like gaseous planets.

Stars are powered by nuclear fusion reactions in their cores that convert hydrogen to helium.

**COMPLETE rationale:** 1. A star is an immense ball of gas held together as a single body, so all of its matter exerts gravitational attraction.
2. That attraction pulls every part of the star inward toward its center.
3. Acting alone, this inward pull would make the star collapse in on itself.
4. The core's nuclear fusion, which converts hydrogen to helium, releases enormous energy as heat and radiation that pushes outward.
5. That outward push from the core's fusion acts directly against the inward pull.

**INCOMPLETE (gap-broken):** 1. A star is an immense ball of gas held together as a single body, so all of its matter exerts gravitational attraction.
2. That attraction pulls every part of the star inward toward its center.
5. That outward push from the core's fusion acts directly against the inward pull.

**REAL target (verbatim):** The star remains stable in a process that balances the inward squeeze caused by its gravity with the outward thrust from the inner fusion reaction in its core.

---

### id 100501 — complete 2.626 / incomplete 2.61 / placebo 3.178 (base 2.904)
**Context (tail):** …/i></blockquote> Big-O statements (like an algorithm taking constant or O(1) time) are statements about asymptotic behavior, i.e, how the function behaves in the limit (usually, as input size tends to infinity). If you don't look at them in the limit, then big-O-ish language (like constant time) is meaningless. <p> How meaningless? Even undecidable languages have a constant time "algorithm" if you consider the input size to be held to a constant.

**COMPLETE rationale:** 1. A big-O claim such as "constant time" is a statement about asymptotic behavior — how the running time grows as the input size increases without bound.
2. If the input size is instead held fixed at a constant, there are only finitely many possible inputs, so any procedure's running time is bounded by a constant — making the "constant time" label apply trivially to every procedure.
3. Under this fixed-input view, even undecidable problems, which are the hardest problems there are, qualify as "constant time."
4. If both the hardest problems and the easiest problems land in the very same complexity class, the asymptotic language can no longer draw any distinction between the difficulty of one problem and another.

**INCOMPLETE (gap-broken):** 1. A big-O claim such as "constant time" is a statement about asymptotic behavior — how the running time grows as the input size increases without bound.
4. If both the hardest problems and the easiest problems land in the very same complexity class, the asymptotic language can no longer draw any distinction between the difficulty of one problem and another.

**REAL target (verbatim):** So without viewing things in the limit, <i>all</i> problems become computationally equivalent in the asymptotic language.

---

### id 101282 — complete 3.842 / incomplete 3.821 / placebo 4.406 (base 3.789)
**Context (tail):** …2 without severe symptoms. During the waiting period, your pediatrician will probably suggest a pain reliever such as acetaminophen, ibuprofen, or anesthetic ear drops. If your child's symptoms don't improve, contact the doctor.

Why not just take antibiotics ASAP? In the past, doctors overprescribed these drugs, experts say, giving them to kids whose symptoms were mild, who didn't have a clear-cut diagnosis, or whose infection was likely viral.

**COMPLETE rationale:** 1. In the past, doctors overprescribed antibiotics to huge numbers of children, including kids whose symptoms were mild, who had no clear diagnosis, or whose infections were likely viral and could not even be treated by the drugs.
2. This means bacteria living in and around a very large population of children were repeatedly and unnecessarily exposed to antibiotics.
3. Antibiotics kill the bacteria that are susceptible to them, but within any large bacterial population a few individuals carry random mutations that happen to make them resistant.
4. Under this constant drug exposure the susceptible bacteria are wiped out while the rare resistant ones survive and keep reproducing — a selective pressure favoring resistance.
5. Generation after generation the surviving resistant bacteria multiply and come to dominate the population.

**INCOMPLETE (gap-broken):** 1. In the past, doctors overprescribed antibiotics to huge numbers of children, including kids whose symptoms were mild, who had no clear diagnosis, or whose infections were likely viral and could not even be treated by the drugs.
2. This means bacteria living in and around a very large population of children were repeatedly and unnecessarily exposed to antibiotics.
5. Generation after generation the surviving resistant bacteria multiply and come to dominate the population.

**REAL target (verbatim):** With children everywhere slurping down the "pink stuff," a scary problem began to arise: Some bacteria became resistant to the antibiotics.

---

### id 101807 — complete 3.051 / incomplete 3.03 / placebo 3.603 (base 3.083)
**Context (tail):** …and diet—as advocating violence and a threat to the United States and Tennessee constitutions. The change makes clear that peaceful religious practices would not be considered a violation, the bill's sponsors said in a statement.

The Council on American–Islamic Relations had promised to file a lawsuit to block the implementation if the bill became law, but now that the overt religious references have been removed, that becomes a lot less likely.

**COMPLETE rationale:** 1. As originally drafted, the bill's whole purpose was to criminalize giving "material support" to Sharia, treating Islamic legal codes themselves as advocating violence and as a threat to the state and federal constitutions.
2. Under heavy public pressure, and after the sponsor met with Muslim leaders, the bill was rewritten to strike out all the language that painted Sharia as violent or as a constitutional threat.
3. The rewrite also explicitly declares that peaceful religious practice is not a violation, so the religion itself is no longer the thing being outlawed.
4. Once every reference to Sharia and religion is removed, the only operative content left from the original bill is its bare "material support" prohibition, with no religious object attached.
5. A "material support" prohibition stripped of any religious object reverts to punishing the ordinary category of dangerous conduct that such statutes are written to target.

**INCOMPLETE (gap-broken):** 1. As originally drafted, the bill's whole purpose was to criminalize giving "material support" to Sharia, treating Islamic legal codes themselves as advocating violence and as a threat to the state and federal constitutions.
2. Under heavy public pressure, and after the sponsor met with Muslim leaders, the bill was rewritten to strike out all the language that painted Sharia as violent or as a constitutional threat.
5. A "material support" prohibition stripped of any religious object reverts to punishing the ordinary category of dangerous conduct that such statutes are written to target.

**REAL target (verbatim):** Basically, the bill has been converted into a fairly straightforward law concerning material support for terrorism.

---

### id 100135 — complete 2.706 / incomplete 2.683 / placebo 2.896 (base 2.625)
**Context (tail):** …takes a stab at explaining, and like most things in life, "Ye" was born out of laziness—and totally France's fault.

Back in the days of old English, the "Th" sound was represented by a single rune shaped something like a letter P. Over the years, it was Babelfished back and forth between French and modern English enough that printers threw their hands up and decided to use a Y instead of two scrunched together letters to represent a "th" sound.

**COMPLETE rationale:** 1. In old English, the "th" sound was written as a single special rune (thorn), not with the letters T and H.
2. That thorn rune was shaped so much like the letter Y that the two looked nearly identical on the page.
3. Printers who had no thorn in their type boxes therefore substituted an ordinary Y for it — a change of written symbol only, leaving unchanged the spoken sound the rune had represented.
4. Any word that now appears in print as "Ye" is thus the old thorn-word in disguise, so reading its Y as a Y-sound would be sounding out the substitute symbol as if it were the original.

**INCOMPLETE (gap-broken):** 1. In old English, the "th" sound was written as a single special rune (thorn), not with the letters T and H.
4. Any word that now appears in print as "Ye" is thus the old thorn-word in disguise, so reading its Y as a Y-sound would be sounding out the substitute symbol as if it were the original.

**REAL target (verbatim):** Which means that "Ye" is actually "The," and is still supposed to be pronounced as "The."

---

### id 100293 — complete 2.628 / incomplete 2.606 / placebo 3.369 (base 2.567)
**Context (tail):** …Tyco Electronics Corp. unveils its High Resolution Radar (HRR) for the next generation of "Smart Bumpers."

Tyco says its system offers significant advantages over ultrasonic sensing systems, including greater range, higher resolution between objects and important styling advantages.

HRR works by transmitting a short pulse into a desired area. This energy is reflected off objects within 66 ft. (20.1 m) and returns.

**COMPLETE rationale:** 1. The HRR sends out a short pulse of radar energy into the target area.
2. This energy is electromagnetic, so it travels at a fixed, known speed that does not change with distance.
3. The pulse travels out to an object, reflects off it, and returns to the sensor, so it covers a round-trip path equal to twice the object's range.
4. Because the speed is constant, the distance covered equals that speed multiplied by the elapsed time, so a longer elapsed time means the pulse traveled a longer round-trip distance.
5. Since that round-trip distance is exactly twice the range, a longer elapsed time corresponds directly to a greater object range.

**INCOMPLETE (gap-broken):** 1. The HRR sends out a short pulse of radar energy into the target area.
3. The pulse travels out to an object, reflects off it, and returns to the sensor, so it covers a round-trip path equal to twice the object's range.
5. Since that round-trip distance is exactly twice the range, a longer elapsed time corresponds directly to a greater object range.

**REAL target (verbatim):** The travel time of the signal determines the range of the object.

---

### id 101966 — complete 1.987 / incomplete 1.961 / placebo 4.202 (base 1.928)
**Context (tail):** …as a result of:

  • An obstruction. Food waste or a hard piece of stool (fecal stone) can block the opening of the cavity that runs the length of your appendix.
  • An infection. Appendicitis may also follow an infection, such as a gastrointestinal viral infection, or it may result from other types of inflammation.

In both cases, bacteria inside the appendix multiply rapidly, causing the appendix to become inflamed, swollen and filled with pus.

**COMPLETE rationale:** 1. An obstruction blocks the opening of the appendix, sealing its cavity so its contents cannot drain out.
2. Bacteria trapped inside the sealed appendix multiply rapidly, generating pus and gas that keep accumulating with nowhere to go.
3. Because the sealed cavity cannot vent, this accumulation drives the internal pressure steadily higher and compresses the blood vessels in the appendix wall, cutting off its blood supply.
4. Deprived of blood and stretched thin by the mounting internal pressure, the appendix wall becomes progressively more fragile.

**INCOMPLETE (gap-broken):** 1. An obstruction blocks the opening of the appendix, sealing its cavity so its contents cannot drain out.
2. Bacteria trapped inside the sealed appendix multiply rapidly, generating pus and gas that keep accumulating with nowhere to go.
4. Deprived of blood and stretched thin by the mounting internal pressure, the appendix wall becomes progressively more fragile.

**REAL target (verbatim):** If not treated promptly, the appendix can rupture.

---

### id 100026 — complete 2.493 / incomplete 2.466 / placebo 3.193 (base 2.578)
**Context (tail):** …elements, including the iron cores, settle to the center of the new Earth and the lighter elements settle on what became the moon. There could be a lot to the theory of elements rare here that would be plentiful on the moon.

We know that Iridium is even more rare on the earth than in space, and with the whole solar system made from the same dust cloud, there should be pretty much equal distributions of iridium throughout the inner solar system.

**COMPLETE rationale:** 1. The whole inner solar system condensed from one and the same primordial dust cloud, so iridium should be distributed in roughly equal proportion across all of its bodies.
2. Earth therefore should have formed containing the same fraction of iridium as the surrounding space and the other inner-solar-system bodies.
3. Yet iridium is observed to be even rarer on Earth than it is in space, so present-day Earth holds less iridium than it should have started with — Earth is depleted in iridium relative to expectation.
4. A depletion below the starting amount means the missing iridium was physically carried off the Earth at some point, not simply never present.
5. The Moon formed when a giant collision flung part of the proto-Earth's material off into orbit, which is precisely the mechanism capable of removing Earth's material.

**INCOMPLETE (gap-broken):** 1. The whole inner solar system condensed from one and the same primordial dust cloud, so iridium should be distributed in roughly equal proportion across all of its bodies.
2. Earth therefore should have formed containing the same fraction of iridium as the surrounding space and the other inner-solar-system bodies.
5. The Moon formed when a giant collision flung part of the proto-Earth's material off into orbit, which is precisely the mechanism capable of removing Earth's material.

**REAL target (verbatim):** so the iridium must have ended up on the moon after the collision.

---

### id 100391 — complete 3.004 / incomplete 2.977 / placebo 3.223 (base 2.833)
**Context (tail):** …or a while there were rumors going around that Washington was going to severely scale back the program's budget, but now according to Washington insider John Logsdon, "there will be more money."

He's also saying that Obama doesn't want to be that president who cuts a future oriented program. So he'll keep it alive, but he'll only give them a budget somewhere between their current spending and the $3 billion per year increase NASA is looking for.

**COMPLETE rationale:** 1. NASA said it needs an extra $3 billion per year to go forward with meaningful human space exploration rather than just sending more robots.
2. Obama will keep the program alive but only fund it somewhere between NASA's current spending and that full $3 billion increase.
3. So NASA will receive less than the $3 billion it asked for, leaving a funding shortfall relative to its stated needs.
4. With less money than it says it needs, NASA cannot fully afford meaningful human space exploration on its own budget.
5. The only way to still pursue those goals despite the gap is to bring in outside contributors who can share the costs and resources.

**INCOMPLETE (gap-broken):** 1. NASA said it needs an extra $3 billion per year to go forward with meaningful human space exploration rather than just sending more robots.
2. Obama will keep the program alive but only fund it somewhere between NASA's current spending and that full $3 billion increase.
5. The only way to still pursue those goals despite the gap is to bring in outside contributors who can share the costs and resources.

**REAL target (verbatim):** But all that means is that NASA will have to buddy up with international space programs a little more.

---

### id 101654 — complete 2.454 / incomplete 2.417 / placebo 3.137 (base 2.397)
**Context (tail):** …ic structure, can be done by means of embryo-splitting method and nuclear transfer. Human cloning through the nuclear transfer method is directed towards two purposes, i.e. reproduction and therapy. The relatively new transgenic technology can be combined with the cloning technique to produce clones with new genes. However, pros and cons arise concerning the development of research on human cloning, particularly cloning for reproductive purposes.

**COMPLETE rationale:** 1. Human cloning by nuclear transfer is being pursued for reproductive purposes, and newer transgenic methods could even produce clones carrying new genes.
2. Pros and cons have arisen over reproductive human cloning, meaning there are serious unresolved objections to it.
3. Those unresolved objections center on whether the procedure is biologically safe and whether it is ethically acceptable.
4. Performing reproductive cloning while such safety and ethical problems remain unsolved would be irresponsible and unacceptable.
5. Finding solutions to these safety and ethical problems requires time and further study, which means cloning must be paused in the meantime.

**INCOMPLETE (gap-broken):** 1. Human cloning by nuclear transfer is being pursued for reproductive purposes, and newer transgenic methods could even produce clones carrying new genes.
2. Pros and cons have arisen over reproductive human cloning, meaning there are serious unresolved objections to it.
5. Finding solutions to these safety and ethical problems requires time and further study, which means cloning must be paused in the meantime.

**REAL target (verbatim):** Therefore, there is need for a moratorium period before human cloning can be performed

---

### id 102576 — complete 2.989 / incomplete 2.95 / placebo 3.578 (base 3.035)
**Context (tail):** …ng Porta 160 does. Nor does it forgive errors in color balance as easily. If you are comfortable shooting chromes, Ektar should be easy to learn. If you want something more forgiving of exposure error, go Portra. These films are tightly engineered for specific categories of use. And consistency is one thing you tend to get in quality products like these. If you want high quality results, then the learning curve is going to be more consistent too.

**COMPLETE rationale:** 1. Ektar is engineered for color accuracy, rendering the colors actually present in a scene instead of adding punch or softening tones of its own.
2. It is tightly engineered and valued for consistency, so it behaves the same way from one roll to the next.
3. A film that both stays faithful to the scene and performs consistently introduces no color shift or random variation of its own into the finished image.
4. It is also unforgiving of exposure and color-balance errors, neither masking nor compensating for them, so nothing in the result can be blamed on the film.

**INCOMPLETE (gap-broken):** 1. Ektar is engineered for color accuracy, rendering the colors actually present in a scene instead of adding punch or softening tones of its own.
4. It is also unforgiving of exposure and color-balance errors, neither masking nor compensating for them, so nothing in the result can be blamed on the film.

**REAL target (verbatim):** Any mistakes which come out in the end result are likely to be your own.

---

### id 101869 — complete 3.001 / incomplete 2.962 / placebo 3.423 (base 2.866)
**Context (tail):** …ank Rome LLP who represents taxpayers whose returns were flagged by the IRS. "You blab about it to a friend, colleague, spouse or girlfriend, and one of them turns you in."

Even your closest pals may be tempted to tattle, since the IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.

And with the popularity of social media, it's now much easier to publish private information publicly.

**COMPLETE rationale:** 1. The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
2. That financial incentive tempts even a person's closest friends, colleagues, or partners to turn them in if they learn about the cheating.
3. The rise of social media has made it far easier for private information to be published publicly where anyone can see it.
4. So any incriminating detail a tax cheat shares publicly online becomes visible to exactly those acquaintances who could be tempted by the reward to report them to the IRS.

**INCOMPLETE (gap-broken):** 1. The IRS offers whistleblowers a reward of up to 30% of any additional tax or penalties it collects from tax cheaters.
4. So any incriminating detail a tax cheat shares publicly online becomes visible to exactly those acquaintances who could be tempted by the reward to report them to the IRS.

**REAL target (verbatim):** So if you did something you think was questionable, don't post it all over Facebook.

---

### id 102075 — complete 3.226 / incomplete 3.185 / placebo 3.953 (base 3.133)
**Context (tail):** …re—operating in “non-combat” zones—and will take orders from nobody.

When its troops started venturing overseas more than a decade ago, Japan's goal was to look more like all of the other countries that provide peacekeepers, and thereby to avoid criticism. Yet the 1992 law authorising the SDF to deploy abroad laid down tight restrictions on which sorts of missions it could join, to avoid violating the pacifist restraints in Japan's constitution.

**COMPLETE rationale:** 1. Japan's goal, when it began sending troops overseas, was to look like every other country that provides peacekeepers, so as to blend in and avoid criticism.
2. To stay within the pacifist restraints of its constitution, the 1992 law imposed tight restrictions on which kinds of missions the SDF is allowed to join.
3. No other peacekeeping nation places such constitution-driven limits on the missions its soldiers may take part in.
4. Operating under these unusual restrictions forces the SDF to behave visibly differently from every other national contingent on the ground.
5. Behaving differently from everyone else is the exact opposite of blending in and looking like the others.

**INCOMPLETE (gap-broken):** 1. Japan's goal, when it began sending troops overseas, was to look like every other country that provides peacekeepers, so as to blend in and avoid criticism.
2. To stay within the pacifist restraints of its constitution, the 1992 law imposed tight restrictions on which kinds of missions the SDF is allowed to join.
5. Behaving differently from everyone else is the exact opposite of blending in and looking like the others.

**REAL target (verbatim):** As a result, Japanese troops still stand out awkwardly wherever they go.

---

### id 101013 — complete 2.093 / incomplete 2.041 / placebo 2.737 (base 1.992)
**Context (tail):** …medicines for diabetes is a high glucose (sugar) level. If the level is very high, it causes frequent and copious urination, thirst, and hunger. Vomiting, dehydration, weakness, and confusion can also be caused by high glucose levels. If the glucose level has been high for a while, the individual might experience weight loss.

It is possible that your father's glucose was on the high side for several days before he missed that one day's medicine.

**COMPLETE rationale:** 1. His father takes medication for type 2 diabetes to keep his blood glucose within a safe range, so the most likely immediate consequence of missing a dose is that his blood glucose rises.
2. A high blood glucose level can itself produce vomiting, dehydration, weakness, and a general feeling of being sick.
3. The father felt sick and vomited the same night he skipped his medicine, and those symptoms match the known effects of high blood glucose, so his episode was most likely driven by elevated glucose rather than being harmless.
4. His glucose may also have been running high for several days before he ever missed the dose, which means his glucose is not being kept reliably in range even under his normal treatment.
5. Because it is uncontrolled high glucose that produces these dangerous symptoms, avoiding future episodes depends on keeping his glucose consistently within a safe range.

**INCOMPLETE (gap-broken):** 1. His father takes medication for type 2 diabetes to keep his blood glucose within a safe range, so the most likely immediate consequence of missing a dose is that his blood glucose rises.
2. A high blood glucose level can itself produce vomiting, dehydration, weakness, and a general feeling of being sick.
5. Because it is uncontrolled high glucose that produces these dangerous symptoms, avoiding future episodes depends on keeping his glucose consistently within a safe range.

**REAL target (verbatim):** Therefore, it is important to make sure that his blood glucose is adequately controlled.

---

### id 101359 — complete 3.191 / incomplete 3.129 / placebo 4.158 (base 3.664)
**Context (tail):** …historic parks; and $4 million would protect water supplies, ponds, rivers, streams, and Narragansett Bay from polluted storm water and establish a fish passage on the Blackstone River.
In the past, residents of these towns voted overwhelmingly in support of environmental bond referenda and the payoff is evident everywhere from Tiverton’s Pardon Gray to progress toward the dream of an Aquidneck Greenway

Pressure to develop such gems is intense.

**COMPLETE rationale:** 1. Question 6 would provide state funds specifically to permanently protect remaining farmland and to acquire open space (the context lists $4.5M to protect farmlands plus money to acquire parks, beaches, and shoreline).
2. In this program, 'protecting' farmland means the state buying the land or its development rights outright, not merely passing a regulation.
3. The remaining farmland and open space are highly desirable ('gems') and are under intense pressure to be developed.
4. Intense development pressure means private developers are actively bidding to buy those same parcels and will pay high prices for them.
5. To keep a parcel as farmland, the state must acquire it before a developer does, which means outbidding or matching those developers on price.

**INCOMPLETE (gap-broken):** 1. Question 6 would provide state funds specifically to permanently protect remaining farmland and to acquire open space (the context lists $4.5M to protect farmlands plus money to acquire parks, beaches, and shoreline).
2. In this program, 'protecting' farmland means the state buying the land or its development rights outright, not merely passing a regulation.
5. To keep a parcel as farmland, the state must acquire it before a developer does, which means outbidding or matching those developers on price.

**REAL target (verbatim):** If it hopes to save any of the farmland that’s left, the state needs money to compete.

---

### id 101374 — complete 3.845 / incomplete 3.764 / placebo 4.326 (base 3.357)
**Context (tail):** …ISIS Object LTRIS14e.DLL ActiveX Control Buffer Overflow Vulnerability

LEADTOOLS Raster ISIS ActiveX control is prone to a buffer-overflow vulnerability because the application fails to bounds-check user-supplied data before copying it into an insufficiently sized buffer.

Successfully exploiting this issue allows remote attackers to execute arbitrary code in the context of the application using the ActiveX control (typically Internet Explorer).

**COMPLETE rationale:** 1. The ActiveX control copies attacker-supplied input into a fixed-size buffer that is too small to hold it, and it does so without checking the length, so oversized input spills past the end of the buffer into adjacent memory.
2. That adjacent memory holds the program's control data (such as the saved return address and function pointers), so the overflow overwrites the values that decide where execution will continue next.
3. Turning the overflow into code execution requires the attacker to overwrite that control data with a precisely crafted value that redirects execution onto their own injected instructions.
4. An attempt that is not precisely crafted still overwrites the same control data, but with invalid, garbage values instead of a working payload.
5. Execution then transfers to an invalid address, so the process references bad memory and the host application (typically Internet Explorer) crashes.

**INCOMPLETE (gap-broken):** 1. The ActiveX control copies attacker-supplied input into a fixed-size buffer that is too small to hold it, and it does so without checking the length, so oversized input spills past the end of the buffer into adjacent memory.
2. That adjacent memory holds the program's control data (such as the saved return address and function pointers), so the overflow overwrites the values that decide where execution will continue next.
5. Execution then transfers to an invalid address, so the process references bad memory and the host application (typically Internet Explorer) crashes.

**REAL target (verbatim):** Failed exploit attempts likely result in denial-of-service conditions.

---

### id 100143 — complete 3.911 / incomplete 3.819 / placebo 4.381 (base 3.899)
**Context (tail):** …lightening of shadow values that you indicated.

This blue filter would have the effect of lowering local contrast within the snow itself since local contrast within the snow itself would contain small shadow areas that are lit by the same blue light that you noted in the shadows.

While a full scale scene may have shadow and snow both included, the actual scene would need to be evaluated to determine the exposure and development considerations.

**COMPLETE rationale:** 1. In a sunlit snow scene, the small shadowed areas within the snow are illuminated mainly by blue skylight rather than by direct sunlight.
2. A yellow filter is "minus blue": it absorbs blue light, so anything lit predominantly by blue light records darker on the film.
3. Those small blue-lit shadow patches within the snow are therefore rendered darker by a yellow filter, while the directly sunlit snow stays bright.
4. Darkening the shadow patches while the sunlit snow stays bright widens the tonal gap between them, and longer (extended) development stretches that tonal separation further still.

**INCOMPLETE (gap-broken):** 1. In a sunlit snow scene, the small shadowed areas within the snow are illuminated mainly by blue skylight rather than by direct sunlight.
4. Darkening the shadow patches while the sunlit snow stays bright widens the tonal gap between them, and longer (extended) development stretches that tonal separation further still.

**REAL target (verbatim):** However for local contrast in the snow itself yellow filtration and expanded development would enhance local contrast.

---

### id 102320 — complete 3.517 / incomplete 3.328 / placebo 4.219 (base 3.047)
**Context (tail):** …ey forgot to pick a winner. [Update: We talked to Corey.]

Corey currently sits tied for fourth in Yahoo's billion-dollar bracket challenge group. That billion is long gone, but the creators of the 20 highest-scoring brackets each receive $100,000. If Corey correctly picked the winner for tonight's tournament final, he'd be in the money. But Corey didn't correctly pick the winner, because he didn't pick any winner, and his bracket is long locked.

**COMPLETE rationale:** 1. Only the 20 highest-scoring brackets each receive $100,000, so Corey must finish among the top 20 to win anything.
2. Corey is tied for fourth right now, but tonight's championship game has not yet been scored, so those final points are still up for grabs.
3. Picking the champion correctly earns those final points, but Corey picked no champion and his bracket is locked, so he can earn zero additional points no matter who wins.
4. Every entrant who did pick the eventual champion will gain those points and climb the standings while Corey's score stays frozen.
5. More than 20 of those climbing entrants sit close enough behind Corey to pass him once they collect the champion's points, dropping him out of the top 20.

**INCOMPLETE (gap-broken):** 1. Only the 20 highest-scoring brackets each receive $100,000, so Corey must finish among the top 20 to win anything.
2. Corey is tied for fourth right now, but tonight's championship game has not yet been scored, so those final points are still up for grabs.
5. More than 20 of those climbing entrants sit close enough behind Corey to pass him once they collect the champion's points, dropping him out of the top 20.

**REAL target (verbatim):** There will be no money for Corey

---

### id 101527 — complete 2.748 / incomplete 2.488 / placebo 4.725 (base 2.576)
**Context (tail):** …istrict will declare a Spare the Air Day, making wood-burning illegal. This wall-mounted fireplace from Pureflame could be the answer—not only does it run on soot-free ethanol, I can hang it anywhere in my house.

The Pureflame runs on plant-derived ethanol that burns without creating soot or smoke. According to the manufacturer its only byproducts are steam and carbon dioxide—both in quantities similar to what humans produce through respiration.

**COMPLETE rationale:** 1. A fireplace normally needs a chimney or vent to carry away the smoke and soot it produces, because those combustion byproducts are harmful to breathe indoors.
2. The Pureflame burns ethanol and produces no soot or smoke at all — its only byproducts are steam and carbon dioxide.
3. It emits these in quantities similar to what a person gives off simply by breathing.
4. People exhale steam and carbon dioxide indoors constantly with no ventilation of that air, so those quantities are already harmless in an occupied room.
5. Therefore the Pureflame puts nothing into the room that has to be carried outside to keep the air safe.

**INCOMPLETE (gap-broken):** 1. A fireplace normally needs a chimney or vent to carry away the smoke and soot it produces, because those combustion byproducts are harmful to breathe indoors.
2. The Pureflame burns ethanol and produces no soot or smoke at all — its only byproducts are steam and carbon dioxide.
5. Therefore the Pureflame puts nothing into the room that has to be carried outside to keep the air safe.

**REAL target (verbatim):** This eliminates the need for venting or chimneys.

---

### id 101594 — complete 3.98 / incomplete 3.713 / placebo 4.432 (base 3.955)
**Context (tail):** …massive conventional armed force, or you will be on the target list.

Their are 2 reasons the United States did not go after North Korea, 1) China 2) The losses in Iraq would have looked like a bubble bath in comparison. As NK can't depend on China to protect them, they feel they are in a position where they have to divert every penny of their GDP (that they possibly can, and even some they can't) to maintaining a large battle ready Armed Forces.

**COMPLETE rationale:** 1. North Korea's core security problem is avoiding invasion; the context states that a nation without nuclear weapons must sustain a massive conventional military or else end up on the target list.
2. Because North Korea cannot rely on China to protect it, its own conventional armed forces are its only available deterrent, which is why it diverts nearly all of its GDP into maintaining them.
3. The context also holds that any nation possessing nuclear weapons is not invaded, even by superpowers, so a nuclear arsenal is itself a full deterrent against invasion.
4. A large conventional army and a nuclear arsenal serve the identical purpose of deterring invasion, so one deterrent can substitute for the other.
5. If North Korea acquires nuclear weapons plus a means to deliver them, it gains the same invasion deterrence it currently obtains only from its costly conventional forces.

**INCOMPLETE (gap-broken):** 1. North Korea's core security problem is avoiding invasion; the context states that a nation without nuclear weapons must sustain a massive conventional military or else end up on the target list.
2. Because North Korea cannot rely on China to protect it, its own conventional armed forces are its only available deterrent, which is why it diverts nearly all of its GDP into maintaining them.
5. If North Korea acquires nuclear weapons plus a means to deliver them, it gains the same invasion deterrence it currently obtains only from its costly conventional forces.

**REAL target (verbatim):** Nukes and a delivery platform change that equation and allow them to curtail some of their conventional force spending.

---

### id 100171 — complete 4.079 / incomplete 3.767 / placebo 3.872 (base 1.455)
**Context (tail):** …way to protect from this type of attack is to use single quotation marks around the numeric constants: SELECT * FROM table WHERE ID='234'. If the user enters extra information, it all becomes part of the string. In a numeric context, MySQL automatically converts this string to a number and strips any trailing nonnumeric characters from it.

Sometimes people think that if a database contains only publicly available data, it need not be protected.

**COMPLETE rationale:** 1. The earlier paragraph shows that entering a value like `234 OR 1=1` forces the query WHERE ID=234 OR 1=1, which is always true, so the server retrieves every row in the table and incurs excessive load.
2. That injection exploits the structure of the SQL query, not the sensitivity of the rows, so it succeeds regardless of whether the data is confidential or publicly displayable.
3. Hence even a database holding only publicly available data can be driven to waste its resources running such full-table scans, i.e. a denial-of-service attack.
4. A server tied up serving those malicious scans becomes unresponsive to legitimate users, a genuine harm that is distinct from any leak of private data.

**INCOMPLETE (gap-broken):** 1. The earlier paragraph shows that entering a value like `234 OR 1=1` forces the query WHERE ID=234 OR 1=1, which is always true, so the server retrieves every row in the table and incurs excessive load.
4. A server tied up serving those malicious scans becomes unresponsive to legitimate users, a genuine harm that is distinct from any leak of private data.

**REAL target (verbatim):** This is incorrect.

---

### id 100654 — complete 4.764 / incomplete 4.213 / placebo 4.732 (base 1.993)
**Context (tail):** …e multinational corporations actually prefer to produce in the United States with U.S. workers.

U.S.-based multinational corporations employ 22.9 million Americans—more than twice as many people as they employ in China, Mexico, and all other countries combined. Foreign-owned multinational corporations employ another 5.5 million people in the United States.

When talking heads or campaigning politicians assert that outsourcing is costing the U.S.

**COMPLETE rationale:** 1. When commentators say outsourcing is costing the country jobs, they are counting only the positions that move from the U.S. to lower-wage economies.
2. But the context shows most foreign direct investment flows into the United States rather than to low-wage countries, and firms owned abroad employ millions of Americans while U.S. multinationals keep more than twice as many workers at home as overseas.
3. Those American positions that exist because the U.S. is the world's leading destination for investment are entirely omitted from the outsourcing complaint.
4. Once the inbound-investment jobs are set against the jobs lost to outsourcing, the balance for American workers comes out favorable rather than the pure loss the complaint implies.

**INCOMPLETE (gap-broken):** 1. When commentators say outsourcing is costing the country jobs, they are counting only the positions that move from the U.S. to lower-wage economies.
4. Once the inbound-investment jobs are set against the jobs lost to outsourcing, the balance for American workers comes out favorable rather than the pure loss the complaint implies.

**REAL target (verbatim):** jobs, they’re telling only part of the story

---

