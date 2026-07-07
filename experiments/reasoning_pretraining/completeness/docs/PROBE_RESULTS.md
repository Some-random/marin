# Probe experiments — full, readable data

Every probe below shows the **original context** (first ~65% of a DCLM doc), the **rationale** we inserted
(Claude, written from the context only), the **question**, the **answer**, and the answer's NLL under the
DCLM-1.4B base judge in three conditions:
- **base** = `context + question` (no rationale)
- **+real** = `context + THIS doc's rationale + question`
- **+placebo** = `context + ANOTHER (irrelevant) doc's rationale + question` (format-matched control)

`delta<0` = the rationale makes the answer more predictable. **real−placebo<0** = the *specific* reasoning
helped beyond generic priming (the honest signal).

---

## Part 1 — probe-target set (19 probes; raw mean −0.358, placebo −0.333)

Sorted most-reasoning-specific first (real−placebo ascending).

### id 444 — ★ reasoning-specific  (real−placebo -1.651)
**Context:** Preliminary Report: Hawker Overruns At TEB

 - November 2, 2006, 7:21 AM

Hawker 700A, Teterboro, N.J., March 8, 2005–At about 10 p.m. EST, Hawker N703TS sustained minor damage while landing at Teterboro Airport (TEB). No one was injured. The pilot instructed the copilot to lower flaps 15 degrees during the approach, about 30 miles from the airport. He requested 25 degrees on a right downwind for Runway 1. The airplane encountered severe low-level wind shear during the approach, so the pilot flew the final approach at Vref (119 knots) plus 15 knots to compensate for the wind. The airplane landed smoothly in the touchdown zone, at Vref plus 10 knots, and the pilot instructed the copilot to extend full flaps.

The pilot selected the airbrake to “dump” and felt the airplane slowing as if the spoilers were working correctly. He focused on runway alignment and did not divert his attention to check the position of the airbrake lever. (The airbrake lever cannot be positioned to “dump” unless the full flap extension is already selected.) The pilot also felt the antiskid braking system working correctly.

**Rationale inserted:** 1. On approach the pilot added speed to compensate for severe wind shear, then landed smoothly in the touchdown zone.
2. After landing he selected the airbrake to "dump" and felt the airplane slowing as if the spoilers were working.
3. But he never visually checked the airbrake lever, relying only on the feel of deceleration.
4. Because the lever can reach "dump" only after full flaps are set, whether the spoilers truly deployed hinged on that sequence.
5. This invites the inference that a mis-set airbrake, unnoticed because braking felt normal, contributed to the overrun.

**Question:** Because the pilot judged braking only by feel and never checked the lever, what might have failed to happen even though the airplane seemed to slow?
**Answer:** `spoilers never deployed`

**Answer NLL:** base **8.2405** → +real **6.5854** (-1.655) → +placebo **8.237** (-0.004)

---

### id 458 — ★ reasoning-specific  (real−placebo -0.796)
**Context:** Seeking Alpha
Seeking Alpha Portfolio App for iPad


SilverCrest Mines (SVLC) offers investors a producer with some of the lowest true all-in costs in the industry. With a gold-heavy production portfolio, these low costs will allow the company to be profitable even in the current low price precious metals environment. While many competitors in the gold and silver industry report negative second quarter incomes, SVLC should still report positive earnings and continue to be profitable. Additionally, a strong balance sheet gives the company much leeway in terms of its development objectives.

The company-specific challenges that investors should monitor are the company's transition from open-pit mining to underground operations and its impact on grades and true all-in costs.

**Rationale inserted:** 1. SilverCrest Mines has among the lowest true all-in production costs in the industry.
2. Low costs mean a positive margin survives even in the current low-price precious-metals environment.
3. So SVLC should still post positive earnings while higher-cost competitors report negative quarterly income.
4. A strong balance sheet funds development, though the open-pit-to-underground transition could raise costs and erode this cost edge.

**Question:** If the switch from open-pit to underground mining raises SVLC's all-in costs, which pillar of the bullish thesis is undermined?
**Answer:** `its low-cost advantage`

**Answer NLL:** base **6.9959** → +real **5.7863** (-1.210) → +placebo **6.582** (-0.414)

---

### id 25 — ★ reasoning-specific  (real−placebo -0.393)
**Context:** Apple 2.0's Philip Elmer-DeWitt put it the best in the subhead for his Apple Earnings Smackdown article this morning when he said "Apple management may have found a way to tame the wild beasts of Wall Street." What he was referring to was the fact that most Wall Street analysts were actually quite close on their estimates of how Apple performed in the second quarter of its fiscal year, thanks to new guidance rules that had been announced by CFO Peter Oppenheimer in January. The Smackdown is a quarterly roll call of analyst estimates and how well various institutional and independent analysts fared in their educated guesses of Apple performance.

On the Q1 2012 earnings call, Oppenheimer stated that the company would no longer give analysts conservative forecasts, or as Elmer-DeWitt put it: "Rather than tell them what the company was reasonably confident it could achieve the following quarter, it would offer a range of guidance that reflected what it believed it was likely to achieve (emphasis is Elmer-DeWitt's)."

Oppenheimer's pronouncement appears to have been embraced by the financial wizards, as 26 of 62 analysts picked the March quarter revenue and earnings within 2.5 percent. That's a vast improvement over the 2011 estimates for the same quarter that were off by a combined average of 11 percent.

**Rationale inserted:** 1. Apple previously gave analysts conservative guidance, so their estimates were far off (2011 off by ~11%).
2. In Q1 2012 the CFO changed policy to offer a realistic range of what Apple expected to achieve.
3. Better information should let analysts forecast more accurately.
4. Indeed 26 of 62 analysts hit revenue and earnings within 2.5%, a large improvement.
5. Thus the guidance change appears responsible for the tighter, more accurate estimates.

**Question:** Had Apple kept giving analysts conservative forecasts in 2012 instead of a realistic range, what would most likely have happened to the accuracy of their revenue and earnings estimates?
**Answer:** `stayed inaccurate`

**Answer NLL:** base **13.0858** → +real **12.7958** (-0.290) → +placebo **13.189** (+0.103)

---

### id 89 — ★ reasoning-specific  (real−placebo -0.393)
**Context:** 11-year-old does donuts in audi r8

Some would say the kid in the following video has an awesome dad. We know better... he most likely has an awesome uncle. A father might not let a son hop in the driver's seat of his shiny black Audi R8, turn the key and watch him enjoy some grass-fed donuts. Also, the father would be peeved that his lawn was torn up.

But seriously, someone gave this kid permission to hoon. Our legal team, parental sense of responsibility, and landscapers everywhere can't help but heap scorn on the adult(s) who helped make this happen, but a small part of us is pleased this young man already has a fine understanding of throttle-induced oversteer.

**Rationale inserted:** 1. A young boy is filmed doing donuts in an Audi R8, tearing up a lawn.
2. A father would be unwilling to risk his prized car and ruin his own lawn.
3. Therefore the permitting adult is more likely an uncle than the father.
4. The author scorns the responsible adult while grudgingly admiring the kid's skill.

**Question:** The reasoning concludes an uncle, not the father, gave permission. That implies the pricey car and torn-up lawn being risked belonged to whom?
**Answer:** `the boy's father`

**Answer NLL:** base **3.5423** → +real **2.9302** (-0.612) → +placebo **3.323** (-0.219)

---

### id 404 — ★ reasoning-specific  (real−placebo -0.317)
**Context:** How Science Works

A Step-by-Step Guide to the Process of Science

How do I answer the question?

Designing an experiment is a crucial part of the scientific process, and trickier than it might seem. To be a fair test, an experiment must look at only one factor at a time, and do so without interfering with the “system” under observation, be it atoms or a troop of baboons.

**Rationale inserted:** 1. A scientific experiment aims to determine the effect of a single factor.
2. If two or more factors changed at once, any result could not be attributed to one factor rather than another.
3. So to be a fair test the experiment must vary only one factor at a time.
4. It must also avoid interfering with the system, so the observation reflects the system's natural behavior rather than the experimenter's disturbance.

**Question:** If a scientist varies two factors at once, can the outcome be pinned on a single cause?
**Answer:** `no, cause is ambiguous`

**Answer NLL:** base **5.6795** → +real **5.3728** (-0.307) → +placebo **5.691** (+0.011)

---

### id 253 — ★ reasoning-specific  (real−placebo -0.274)
**Context:** ALG Statement on Announced $1.47 Trillion Deficit for 2010

July 23rd, 2010, Fairfax, VA—Americans for Limited Government President Bill Wilson today issued the following statement in response to the White House’s projection of a $1.47 trillion budget deficit for 2010:

“The announced $1.47 trillion deficit by the Obama Administration is just the latest insult to American taxpayers, who during these stressful economic times are scaling back their family budgets but watch helplessly as government dramatically expands without paying for it. That money will have to borrowed through Treasury auctions, and what cannot be borrowed will simply be printed by the Federal Reserve, which already holds $777 billion in U.S.

**Rationale inserted:** 1. The White House projects a $1.47 trillion federal deficit for 2010.
2. A deficit means government spends far more than it collects, expanding without paying for it.
3. That shortfall must be financed, so it will have to be borrowed through Treasury auctions.
4. Whatever cannot be borrowed will simply be printed by the Federal Reserve.
5. Therefore this deficit spending ultimately burdens taxpayers who are cutting their own budgets.

**Question:** As a direct result of financing this deficit by borrowing through Treasury auctions, what happens to the national debt?
**Answer:** `it increases`

**Answer NLL:** base **5.9574** → +real **5.4076** (-0.550) → +placebo **5.682** (-0.275)

---

### id 260 — ★ reasoning-specific  (real−placebo -0.186)
**Context:** Reply to a comment

Reply to this comment

shieldsawyer writes:

in response to john4757:

I just came back from driving in Tennessee, Georgia, and Alabama. I bought gasoline for as little as $3.11 and the average was probably around $3.25. Why was gas so cheap there and not here? Our gas prices go up every time there is a refinery problem somewhere else in the country but our refinery problems didn't affect people in those 3 states. Every year something else causes gas prices to rise. If you drive thru the city of Washington, In. their gas prices are almost always 30 cents higher than Evansville and 20 cents higher than the smaller surrounding towns. I remember in the 70's we boycotted a lot of overpriced gas stations and we need to do that now. If everyone would quit buying gas from owners who take advantage of their monopoly, prices would come down. Also, boycott buying food items from their stores.

**Rationale inserted:** 1. Gasoline was much cheaper in Tennessee, Georgia, and Alabama than at home.
2. Local prices rise whenever a refinery fails anywhere, yet those states' prices were unaffected.
3. This inconsistency implies local sellers exploit a monopoly to raise prices without real cause.
4. If buyers boycotted such sellers, falling demand would force prices back down.

**Question:** If nationwide refinery problems were truly the cause of the local price hikes, what should have happened to gas prices in Tennessee, Georgia, and Alabama?
**Answer:** `they'd have risen too`

**Answer NLL:** base **4.2249** → +real **4.0481** (-0.177) → +placebo **4.234** (+0.009)

---

### id 485 — ★ reasoning-specific  (real−placebo -0.129)
**Context:** General News

17 2013

Anna Friel criticised over extreme diet

8:34am EDT

British actress Anna Friel has become the latest star to face criticism over her promotion of a controversial liquid diet which she uses to maintain her youthful looks. The Pushing Daisies star, 36, has admitted using the Master Cleanse over a period of two months, cutting out solid food and surviving on a drink of maple syrup, cayenne pepper, lemon juice and water. Friel told Britain's Grazia magazine she also uses the diet to improve her skin, saying, "I've been drinking it for two months and I feel so much better and my skin has really benefited. If you're vain, as you get older you start thinking, 'I've got to do everything I can to save my skin'. I've tried everything." However, Friel's comments have sparked a backlash among critics, who fear the actress is setting a bad example to impressionable young girls. Sioned Quirke, of the British Dietetic Association, tells Britain's Daily Mail, "(The diet is) extreme and unnecessary... From a nutritional point of view, you are lacking in all the essential nutrients which your body requires on a daily basis to function. "How could you possibly feel better if you are depriving yourself of everything that your body needs to survive? If you are lacking in these nutrients for a matter of days, let alone weeks, your body will suffer.

**Rationale inserted:** 1. Anna Friel promotes a liquid "Master Cleanse" diet, claiming it makes her feel better and improves her skin.
2. A dietitian counters that the diet supplies none of the essential nutrients the body needs to function.
3. If the body is deprived of what it needs to survive, it cannot genuinely be healthier, contradicting the "feel better" claim.
4. Prolonged deprivation would instead make the body suffer, so critics fear she sets a harmful example for young girls.

**Question:** Does the dietitian's nutritional argument support or undermine Friel's claim that the cleanse makes her feel better?
**Answer:** `undermines it`

**Answer NLL:** base **9.2351** → +real **8.6765** (-0.559) → +placebo **8.806** (-0.429)

---

### id 302 — no help  (real−placebo -0.009)
**Context:** you are viewing a single comment's thread.

view the rest of the comments →

[–]S3xyInternalOrgans 1 point2 points ago


It somewhat saddens me that when an accomplishment makes me feel fantastic, part of it derives from comparing it with the accomplishments of the people who tormented me in primary and the beginning of secondary school. For example, when I got the second highest Leaving Cert results in my school (this is somewhat of a big deal- the LC is a final secondary exam that basically requires you to store two years of information in your head and will decide which college and course you'll end up in. Not a test of intelligence, by any means, but doing well requires a hell of a lot of hard work) I walked around with a shit-eating grin on my face for two days. Not just because the 14-hour study days had paid off, but because one of the bullies was pregnant and three were on welfare with no qualification.

The funny thing is though, they indirectly contributed to my success. I had suffered from acute anxiety and depression for years, due to being abused as a kid, constant bullying and bereavement. From the age of twelve, I found that studying really helped me take my mind off things. It allowed me to escape from other people (I'm still incredibly nervous in company), focus on things that had absolutely no relevance to my shitty life or turbulent emotions, and my mother couldn't complain about me being distant because I was doing something productive.

**Rationale inserted:** 1. The author reached near-top Leaving Cert results after sustained intense study (14-hour days).
2. They had endured abuse, bullying, and bereavement, producing acute anxiety and depression from a young age.
3. From age twelve, studying became a way to escape those emotions, avoid other people, and satisfy their mother.
4. Because studying served as emotional escape, it fueled the relentless hard work that produced the top results.
5. Therefore the tormentors 'indirectly contributed' to the success: the pain they caused pushed the author into the coping habit that yielded achievement.

**Question:** Had the author never been abused or bullied, what would most likely have happened to their exam results?
**Answer:** `likely worse`

**Answer NLL:** base **8.1914** → +real **8.3339** (+0.143) → +placebo **8.343** (+0.152)

---

### id 336 — no help  (real−placebo +0.014)
**Context:** Comment: Parens Patriae :

(See in situ)

Parens Patriae :

This is the "philosophy" used to explain the arrogation of power in these matters.

Parens patriae is Latin for "parent of the nation." In law, it refers to the public policy power of the state to intervene against an abusive or negligent parent, legal guardian or informal caretaker, and to act as the parent of any child or individual who is in need of protection. For example, some children, incapacitated individuals, and disabled individuals lack parents who are able and willing to render adequate care, requiring state intervention. In U.S. litigation, parens patriae can be invoked by the state to create its standing to sue; the state declares itself to be suing on behalf of its people.

**Rationale inserted:** 1. Parens patriae ('parent of the nation') is the state's policy power to act as parent for individuals needing protection.
2. Some children, incapacitated, and disabled individuals lack caretakers who are able and willing to provide adequate care.
3. Because those individuals lack adequate caretakers, the state may intervene against abusive or negligent guardians and assume the parental role.
4. On this basis the state can declare itself to be suing on behalf of its people, which is what creates its standing to sue.

**Question:** When a state successfully invokes parens patriae against a negligent guardian, what happens to that guardian's authority over the child?
**Answer:** `it is superseded`

**Answer NLL:** base **4.5483** → +real **4.5573** (+0.009) → +placebo **4.543** (-0.005)

---

### id 190 — priming/flat  (real−placebo +0.068)
**Context:** From: Fred Lindberg Date: March 26 1999 10:48pm Subject: Re: Inverse of ... like "...%" List-Archive: Message-Id: <> MIME-Version: 1.0 Content-Type: text/plain; charset="iso-8859-1" Content-Transfer-Encoding: 7bit On Fri, 26 Mar 1999 20:23:04 +0000, Fred T. Krogh wrote: >responses?) The approach I mention above, i.e. looking for D entries, >then DK entries and then DKB entries will take advantage of the indexing >that is available, and thus should be reasonable fast? Yes. Hard to think of a faster way, especially since it will only be at most 3 "sets". Combined in one query you'd also eliminate duplicates, i.e. WHERE x='D' OR x='DK' OR x='DKB'. An alternative would be to limit the mail categories to e.g. 32 and use 5 bits, limit the second set to 5 bits and the third level to 6 bits. In each case, the value 0 mens no restriction. You or course need to translate categories to a number, rather than a letter (you need to encode the categories anyway (i.e. it doesn't matter if crypto => 'C' or crypto => 10).

**Rationale inserted:** 1. Hierarchical mail categories can be matched by their prefixes: D, then DK, then DKB.
2. Because the column is indexed, at most three prefix lookups stay fast.
3. Combining them in one query (WHERE x='D' OR 'DK' OR 'DKB') also removes duplicates.
4. Alternatively, encode category levels as bit fields (5/5/6 bits), with 0 meaning no restriction.
5. This needs categories mapped to numbers, but the exact letter-to-number mapping is arbitrary.

**Question:** In the bit-field encoding, the first category level uses 5 bits, giving 32 possible values. How many possible values does the 6-bit third level give?
**Answer:** `64`

**Answer NLL:** base **3.5637** → +real **3.1841** (-0.380) → +placebo **3.116** (-0.448)

---

### id 172 — priming/flat  (real−placebo +0.123)
**Context:** Prosecuting prostitution abroad

MF says there's one country whose approach stands out: Sweden essentially has a felon offense against johns who buy sex, and traffickers, and pimps, but there's no arrest of the person selling sex. The Swedish law offers services, such as housing, medical and social services and long term vocational training, which women need to get out. (Has it rid Sweden of prostitution?) It has almost stopped trafficking entirely in Sweden. People are being moved to places where prostitution is decriminalized. (Is the Netherlands an example of this, with areas that are official red light zones? Does that work?) It does not seem to have worked, the mayor of Amsterdam has recently spoken of the overwhelming presence of illegal crime. (What about Thailand and other places where this is the only option for women and in some cases it works?) I once heard a john say he was a humanitarian because he was giving money so a woman could put food on the table.

**Rationale inserted:** 1. Sweden penalizes buyers, pimps, and traffickers but not those selling sex, and offers exit services.
2. This approach nearly eliminated trafficking within Sweden.
3. Demand appears to shift to countries where prostitution is decriminalized.
4. In decriminalized Netherlands it hasn't worked, as Amsterdam's mayor cites overwhelming illegal crime.
5. So targeting demand while supporting sellers looks more effective than decriminalization, though it displaces the trade.

**Question:** Given that trafficking nearly stopped inside Sweden but people were moved to decriminalized countries, does the Swedish approach appear to shrink prostitution overall or mainly move it elsewhere?
**Answer:** `mainly move it`

**Answer NLL:** base **5.0856** → +real **4.8077** (-0.278) → +placebo **4.685** (-0.401)

---

### id 126 — no help  (real−placebo +0.261)
**Context:** Twain Exonerated

Published: March 05, 1988

To the Editor:

The Auctions column (Weekend, Feb. 5) contains a significant error on Christie's auction of the Doheny Mark Twain material. The presentation copy of the first edition of ''Huckleberry Finn'' (from Clemens to his wife, Livy) does not contain, as was reported, the illustration on page 283 that was altered obscenely by someone during the printing. The illustration was in its original prealtered state (state 1 in Blanck's ''Bibliography of American Literature''), as Christie's catalogue makes clear.

This is not nit-picking. It is almost unthinkable that Clemens would have given his wife a first edition of Huckleberry Finn containing the obscene illustration.

**Rationale inserted:** 1. Christie's auctioned the first-edition Huckleberry Finn that Clemens presented to his wife.
2. A column claimed this copy had the obscenely altered page-283 illustration.
3. In fact the illustration was in its original unaltered state, per the catalogue.
4. Since Clemens would almost certainly not give his wife an obscene copy, the reported claim is both wrong and unfair, not mere nit-picking.

**Question:** Given that Clemens personally gave this copy to his wife, why does the writer insist correcting the illustration error is not mere nit-picking?
**Answer:** `it defends Twain's character`

**Answer NLL:** base **5.4182** → +real **5.7423** (+0.324) → +placebo **5.481** (+0.063)

---

### id 430 — priming/flat  (real−placebo +0.459)
**Context:** Ruby Roth has written a book for children, Vegan Is Love, which was featured on Today this morning. Inside, there are illustrations of animals hugging. There are also drawings of dead animals strung up and bleeding. Roth says, "My goal is not to scare any child." Roth has a stepdaughter, Akira, who declares: "My favorite food is kale." Probably because she can't eat McNuggets?

Obviously vegan parents are the target audience for this book, and they most likely wouldn't have a problem with the illustrations or the language inside. But, as Matt Lauer mentioned later in the segment, when you send the message "vegan is love," do you also send the message that "eating meat is hate"? And how does that affect a child's budding relationships? Surely no one could argue that you shouldn't educate a kid and teach compassion, even when it comes to food choices. But what about tolerance and acceptance of the choices of others? Even if your moral compass is very tightly wound, and you believe that meat is murder, should you let a kid decide for herself? Is little Akira existing in a world where she believes her teacher and classmates are cruel killers? And: While it's true that healthy eating habits should be taught early on, could this book trigger a constant worry about food choices, and lead to disordered eating? So many questions.

**Rationale inserted:** 1. The book Vegan Is Love pairs gentle animal images with disturbing drawings of dead animals to teach children compassion.
2. Framing veganism as "love" implicitly frames eating meat as its opposite, "hate."
3. A child who absorbs that framing may come to see meat-eating classmates and teachers as cruel killers, straining relationships.
4. Emphasizing morally-loaded food rules could also breed constant anxiety about eating, risking disordered eating.
5. So the reviewer questions whether such absolutist messaging conflicts with also teaching tolerance for others' choices.

**Question:** Does the reviewer worry the book would make a child more or less tolerant of people who eat meat?
**Answer:** `less tolerant`

**Answer NLL:** base **7.9989** → +real **7.8776** (-0.121) → +placebo **7.419** (-0.580)

---

### id 153 — priming/flat  (real−placebo +0.472)
**Context:** Pharmacy Times


Reps Steven LaTourette (R, OH) and Stephen Lynch (D, MA) have introduced sweeping federal legislation that will mandate training, education, registration, and certification requirements for pharmacy technicians nationwide.

Also known as "Emily's Act," named after a 2-year-old Ohio resident who died in 2006 after a pharmacy technician made an error with her chemotherapy dose, HR 5491 will set a floor for states to meet but will not weaken any state laws, according to the bill's sponsors. Currently, states oversee pharmacists and technicians, but regulations regarding training, certification, and continuing education vary from state to state.

The bill, which has been referred to the House Energy and Commerce Committee, would require states to register pharmacy technicians and have them pass the national Pharmacy Technician Board Certification exam, which triggers mandatory continuing education and renewal every 2 years.

**Rationale inserted:** 1. Pharmacy technician training and certification currently vary from state to state.
2. A technician's dosing error killed a child (Emily), motivating a bill named for her.
3. Federal legislation would set a uniform floor requiring registration and national certification, without weakening stronger state laws.
4. Passing certification triggers mandatory continuing education and biennial renewal, aimed at reducing such fatal errors.

**Question:** The bill sets a national floor for technician standards. For a state whose current requirements fall below that floor, what must it do to comply?
**Answer:** `raise its requirements`

**Answer NLL:** base **7.2892** → +real **6.3558** (-0.933) → +placebo **5.884** (-1.405)

---

### id 99 — no help  (real−placebo +0.576)
**Context:** Text Size

New Technology for Gas Absorption
Matthew Paragano
Yale University

The proposed research aims to develop technologies to make gas absorption systems for carbon dioxide or other waste gases smaller and more lightweight through a novel gas-liquid contact mechanism. Furthermore, this technology would permit use of liquid solvents in microgravity where previously only solids were permissible. The overall goal of this proposal is understand the transport mechanics for these waste gases into and within nanometer-sized droplets. Once transport mechanics are characterized, a prototype device may be designed and tested using flight-like conditions. Finally, a full size system will be constructed and tested to ensure scale-up is performed adequately. The full size system will be capable of reduced gravity testing.

Carbon dioxide capture is of particular interest to NASA since it is the primary waste gas of human metabolism and must be controlled within any closed environment (spacecraft, spacesuit, etc.).

**Rationale inserted:** 1. Closed environments like spacecraft accumulate CO2 from human metabolism, which must be removed.
2. A gas-liquid contact mechanism using nanometer droplets could shrink and lighten absorption systems and enable liquid solvents in microgravity, where previously only solids worked.
3. Designing such a device first requires understanding how waste gases transport into and within the droplets.
4. Once that transport is characterized, a flight-like prototype can be built, then scaled to a full reduced-gravity system.

**Question:** Why is removing carbon dioxide far more critical aboard a sealed spacecraft than in an open-air factory?
**Answer:** `it can't escape`

**Answer NLL:** base **5.9989** → +real **6.0521** (+0.053) → +placebo **5.476** (-0.523)

---

### id 345 — no help  (real−placebo +1.762)
**Context:** Q&A: Why are sticky notes not good for books?


Why is it not a good idea to use sticky notes in books?


While sticky notes are certainly useful, they can damage paper. The residue they leave behind attracts dirt, causes pages to stick together, and stains paper over time. When sticky notes are pulled from fragile pages in old books, they can also tear the paper.

**Rationale inserted:** 1. Sticky notes are useful but leave residue behind on paper.
2. That residue attracts dirt, causes pages to stick together, and stains paper over time.
3. On the fragile pages of old books, pulling off sticky notes can tear the paper.
4. Because they cause residue damage and tearing, sticky notes are not a good idea to use in books.

**Question:** Does leaving a sticky note inside a book for a long time make the paper damage better or worse?
**Answer:** `worse`

**Answer NLL:** base **15.0391** → +real **15.501** (+0.462) → +placebo **13.739** (-1.300)

---

## Part 2 — strict specific-answer probes (real −0.406, placebo −0.307, reasoning-specific −0.099)

Stricter answers (numbers/names/specific phrases only). Same three conditions.

### doc 444
**Context:** Preliminary Report: Hawker Overruns At TEB

 - November 2, 2006, 7:21 AM

Hawker 700A, Teterboro, N.J., March 8, 2005–At about 10 p.m. EST, Hawker N703TS sustained minor damage while landing at Teterboro Airport (TEB). No one was injured. The pilot instructed the copilot to lower flaps 15 degrees during the approach, about 30 miles from the airport. He requested 25 degrees on a right downwind for Runway 1. The airplane encountered severe low-level wind shear during the approach, so the pilot flew the final approach at Vref (119 knots) plus 15 knots to compensate for the wind. The airplane landed smoothly in the touchdown zone, at Vref plus 10 knots, and the pilot instructed the copilot to extend full flaps.

The pilot selected the airbrake to “dump” and felt the airplane slowing as if the spoilers were working correctly. He focused on runway alignment and did not divert his attention to check the position of the airbrake lever. (The airbrake lever cannot be positioned to “dump” unless the full flap extension is already selected.) The pilot also felt the antiskid braking system workin

**Rationale inserted:** 1. On approach the pilot added speed to compensate for severe wind shear, then landed smoothly in the touchdown zone.
2. After landing he selected the airbrake to "dump" and felt the airplane slowing as if the spoilers were working.
3. But he never visually checked the airbrake lever, relying only on the feel of deceleration.
4. Because the lever can reach "dump" only after full flaps are set, whether the spoilers truly deployed hinged on that sequence.
5. This invites the inference that a mis-set airbrake, unnoticed because braking felt normal, contributed to the overrun.

- **Q:** The pilot felt the airplane decelerating and assumed the airbrakes were working, but never visually confirmed the lever, which cannot reach 'dump' unless full flaps were already set. What does the reasoning conclude actually happened to the spoilers, explaining the runway overrun?  **A:** `they never deployed`
  → real -1.576 · placebo -0.707 · **real−placebo -0.868**

- **Q:** (not recorded)  **A:** `they never deployed`
  → real -1.576 · placebo -0.707 · **real−placebo -0.868**

### doc 404
**Context:** How Science Works

A Step-by-Step Guide to the Process of Science

How do I answer the question?

Designing an experiment is a crucial part of the scientific process, and trickier than it might seem. To be a fair test, an experiment must look at only one factor at a time, and do so without interfering with the “system” under observation, be it atoms or a troop of baboons.

**Rationale inserted:** 1. A scientific experiment aims to determine the effect of a single factor.
2. If two or more factors changed at once, any result could not be attributed to one factor rather than another.
3. So to be a fair test the experiment must vary only one factor at a time.
4. It must also avoid interfering with the system, so the observation reflects the system's natural behavior rather than the experimenter's disturbance.

- **Q:** A gardener gives plant A both more water and more sunlight than plant B, then finds plant A grew taller and concludes the extra water caused the growth. Applying the doc's fair-test principle, name the specific methodological flaw that invalidates that conclusion.  **A:** `confounding variables`
  → real -0.244 · placebo +0.250 · **real−placebo -0.494**

- **Q:** (not recorded)  **A:** `confounding variables`
  → real -0.244 · placebo +0.250 · **real−placebo -0.494**

### doc 190
**Context:** From: Fred Lindberg Date: March 26 1999 10:48pm Subject: Re: Inverse of ... like "...%" List-Archive: Message-Id: <> MIME-Version: 1.0 Content-Type: text/plain; charset="iso-8859-1" Content-Transfer-Encoding: 7bit On Fri, 26 Mar 1999 20:23:04 +0000, Fred T. Krogh wrote: >responses?) The approach I mention above, i.e. looking for D entries, >then DK entries and then DKB entries will take advantage of the indexing >that is available, and thus should be reasonable fast? Yes. Hard to think of a faster way, especially since it will only be at most 3 "sets". Combined in one query you'd also eliminate duplicates, i.e. WHERE x='D' OR x='DK' OR x='DKB'. An alternative would be to limit the mail categories to e.g. 32 and use 5 bits, limit the second set to 5 bits and the third level to 6 bits. In each case, the value 0 mens no restriction. You or course need to translate categories to a number, rather than a letter (you need to encode the categories anyway (i.e. it doesn't matter if crypto => 'C' or crypto => 10).

**Rationale inserted:** 1. Hierarchical mail categories can be matched by their prefixes: D, then DK, then DKB.
2. Because the column is indexed, at most three prefix lookups stay fast.
3. Combining them in one query (WHERE x='D' OR 'DK' OR 'DKB') also removes duplicates.
4. Alternatively, encode category levels as bit fields (5/5/6 bits), with 0 meaning no restriction.
5. This needs categories mapped to numbers, but the exact letter-to-number mapping is arbitrary.

- **Q:** With the value 0 reserved to mean 'no restriction', how many usable category codes does the 5-bit first set provide?  **A:** `31`
  → real -0.116 · placebo -0.037 · **real−placebo -0.079**

- **Q:** (not recorded)  **A:** `31`
  → real -0.116 · placebo -0.037 · **real−placebo -0.079**

### doc 260
**Context:** Reply to a comment

Reply to this comment

shieldsawyer writes:

in response to john4757:

I just came back from driving in Tennessee, Georgia, and Alabama. I bought gasoline for as little as $3.11 and the average was probably around $3.25. Why was gas so cheap there and not here? Our gas prices go up every time there is a refinery problem somewhere else in the country but our refinery problems didn't affect people in those 3 states. Every year something else causes gas prices to rise. If you drive thru the city of Washington, In. their gas prices are almost always 30 cents higher than Evansville and 20 cents higher than the smaller surrounding towns. I remember in the 70's we boycotted a lot of overpriced gas stations and we need to do that now. If everyone would quit buying gas from owners who take advantage of their monopoly, prices would come down. Also, boycott buying food items from their stores.

**Rationale inserted:** 1. Gasoline was much cheaper in Tennessee, Georgia, and Alabama than at home.
2. Local prices rise whenever a refinery fails anywhere, yet those states' prices were unaffected.
3. This inconsistency implies local sellers exploit a monopoly to raise prices without real cause.
4. If buyers boycotted such sellers, falling demand would force prices back down.

- **Q:** Given Washington, Indiana gas runs 30 cents above Evansville and 20 cents above the smaller surrounding towns, how much more do those smaller towns pay than Evansville?  **A:** `10 cents`
  → real -0.151 · placebo -0.149 · **real−placebo -0.002**

- **Q:** (not recorded)  **A:** `10 cents`
  → real -0.151 · placebo -0.149 · **real−placebo -0.002**

### doc 25
**Context:** Apple 2.0's Philip Elmer-DeWitt put it the best in the subhead for his Apple Earnings Smackdown article this morning when he said "Apple management may have found a way to tame the wild beasts of Wall Street." What he was referring to was the fact that most Wall Street analysts were actually quite close on their estimates of how Apple performed in the second quarter of its fiscal year, thanks to new guidance rules that had been announced by CFO Peter Oppenheimer in January. The Smackdown is a quarterly roll call of analyst estimates and how well various institutional and independent analysts fared in their educated guesses of Apple performance.

On the Q1 2012 earnings call, Oppenheimer stated that the company would no longer give analysts conservative forecasts, or as Elmer-DeWitt put it: "Rather than tell them what the company was reasonably confident it could achieve the following quarter, it would offer a range of guidance that reflected what it believed it was likely to achieve (emphasis is Elmer-DeWitt's)."

Oppenheimer's pronouncement appears to have been embraced by the finan

**Rationale inserted:** 1. Apple previously gave analysts conservative guidance, so their estimates were far off (2011 off by ~11%).
2. In Q1 2012 the CFO changed policy to offer a realistic range of what Apple expected to achieve.
3. Better information should let analysts forecast more accurately.
4. Indeed 26 of 62 analysts hit revenue and earnings within 2.5%, a large improvement.
5. Thus the guidance change appears responsible for the tighter, more accurate estimates.

- **Q:** Roughly what fraction of the 62 analysts landed within 2.5 percent of Apple's actual quarterly revenue and earnings?  **A:** `about 42 percent`
  → real +0.090 · placebo +0.008 · **real−placebo +0.081**

- **Q:** (not recorded)  **A:** `about 42 percent`
  → real +0.090 · placebo +0.008 · **real−placebo +0.081**

### doc 485
**Context:** General News

17 2013

Anna Friel criticised over extreme diet

8:34am EDT

British actress Anna Friel has become the latest star to face criticism over her promotion of a controversial liquid diet which she uses to maintain her youthful looks. The Pushing Daisies star, 36, has admitted using the Master Cleanse over a period of two months, cutting out solid food and surviving on a drink of maple syrup, cayenne pepper, lemon juice and water. Friel told Britain's Grazia magazine she also uses the diet to improve her skin, saying, "I've been drinking it for two months and I feel so much better and my skin has really benefited. If you're vain, as you get older you start thinking, 'I've got to do everything I can to save my skin'. I've tried everything." However, Friel's comments have sparked a backlash among critics, who fear the actress is setting a bad example to impressionable young girls. Sioned Quirke, of the British Dietetic Association, tells Britain's Daily Mail, "(The diet is) extreme and unnecessary... From a nutritional point of view, you are lacking in all the essential nutri

**Rationale inserted:** 1. Anna Friel promotes a liquid "Master Cleanse" diet, claiming it makes her feel better and improves her skin.
2. A dietitian counters that the diet supplies none of the essential nutrients the body needs to function.
3. If the body is deprived of what it needs to survive, it cannot genuinely be healthier, contradicting the "feel better" claim.
4. Prolonged deprivation would instead make the body suffer, so critics fear she sets a harmful example for young girls.

- **Q:** Anna Friel survived for two months on the liquid Master Cleanse. Based on the dietitian's nutritional critique of that regimen, what clinical condition would prolonged reliance on it predictably produce?  **A:** `malnutrition`
  → real -1.143 · placebo -1.277 · **real−placebo +0.134**

- **Q:** (not recorded)  **A:** `malnutrition`
  → real -1.143 · placebo -1.277 · **real−placebo +0.134**

- **Q:** How many of the 62 analysts did NOT come within 2.5 percent of Apple's actual results?  **A:** `36`
  → real -0.044 · placebo -0.226 · **real−placebo +0.182**

- **Q:** (not recorded)  **A:** `36`
  → real -0.044 · placebo -0.226 · **real−placebo +0.182**

- **Q:** In the proposed bit-field scheme, how many distinct values can the 6-bit third level encode?  **A:** `64`
  → real -0.068 · placebo -0.319 · **real−placebo +0.252**

- **Q:** (not recorded)  **A:** `64`
  → real -0.068 · placebo -0.319 · **real−placebo +0.252**

## Part 3 — does the agent's 'this doc warrants a rationale' predict actual drops? (NO)

| label | n | mean continuation-Δ | actually dropped |
|---|---:|---:|---:|
| R (reasoning-dependent) | 19 | +0.090 | 2/19 (11%) |
| N (not) | 81 | +0.089 | 12/81 (15%) |

**No relationship (slightly anti):** R docs dropped 11% vs N 15%. Of 14 docs that actually
dropped, only 2 were R (14%) vs the 19% base rate. The classification does NOT
predict which docs drop — evidence the continuation-perplexity signal is priming/noise, not reasoning.

