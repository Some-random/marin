# Surprising cases: reasoning RAISED continuation perplexity (judge = Qwen2.5-7B, 32B-generated data)

Scored **234** docs (those with a non-empty continuation). Reasoning helped (Δnll<0) in only **1**. Median Δnll = **+0.414** nats/token (positive = reasoning made the real continuation *more* surprising).

## The diagnosis these cases show
The generated reasoning is usually **good**. The problem is the **'continuation' (answer key)**: this data was made by splitting each raw DCLM doc at ~65% of characters and calling the rest the 'continuation'. In raw web crawl the last chunk is usually NOT a reasoned payoff — it's an attribution tag, a UI button, a comment jab, a nav link. So adding relevant reasoning puts the model in 'expect a logical conclusion' mode, and the actual junk continuation becomes *even more* surprising. That's also why the better 32B reasoning scored worse than 7B: better reasoning -> bigger mismatch with a junk continuation.

---

## Cases where reasoning HURT MOST (Δnll highest — ppl should have dropped, instead exploded)

### 1. Δnll = +4.255  (base ppl 103.9 -> +reasoning ppl 7318.8)

**CONTEXT:**

> Reply to a comment

Reply to this comment

Skeletor writes:

What this article doesn't tell you, is that Obummer is cutting military salaries 1% while RAISING federal salaries (including Biden's). Where are all the Libs that keep saying Democrats support the lower class? I'm sure there are plenty of lower class people in the military trying to better their lives!

Obama cannot pass a balanced budget, and he doesn't understand that you don't spend your way out of a debt crisis.

**GENERATED REASONING:**

1. The article mentions that Obama is cutting military salaries by 1% while increasing federal salaries.
2. This action implies a disparity in treatment between military personnel and federal employees under Obama's policies.
3. Since Democrats are often portrayed as supporters of the lower class, the inconsistency between this policy and their stated values is highlighted.
4. Military personnel are often considered part of the working or lower-middle class, thus the salary cut affects them adversely.
5. The author argues that Obama’s inability to balance the budget indicates poor financial management, contradicting the idea that spending will resolve the debt crisis.

**'CONTINUATION' it was scored against:**

> Obama the failure.

---

### 2. Δnll = +4.207  (base ppl 5.2 -> +reasoning ppl 346.6)

**CONTEXT:**

> Got some breadboardable electronics and a couple extra coffee cans lying around? Why not use them to build your very own Doppler radar?

MIT's OPENCOURSEWARE, the school's open education initiative, offers a free online course and resource materials that will guide you through the entire process, while educating you on the finer points of radar engineering. Your Folgers-hack won't be powerful enough to track precipitation, but it'll do just fine as a DIY speedometer, as the video up top demonstrates.

Visit MIT for more info on what you'll need to get started, and other ideas on how to make use of your radar system.

**GENERATED REASONING:**

1. The passage suggests using available materials like electronics and coffee cans to build a Doppler radar.
2. MIT’s OPENCOURSEWARE provides resources for learning radar engineering and constructing a basic Doppler radar system.
3. Given these resources, one can assemble a functional, albeit limited, radar system at home.
4. Although the homemade radar cannot detect weather phenomena, it is capable of functioning as a speedometer.
5. Interested individuals can visit MIT’s website for detailed instructions and additional project ideas involving radar systems.

**'CONTINUATION' it was scored against:**

> [Via MAKE]

---

### 3. Δnll = +3.957  (base ppl 4.1 -> +reasoning ppl 212.0)

**CONTEXT:**

> Twelve identical point charges q are equally spaced around the circumference of a circle of radius R. The circle is centered at the origin. One of the twelve charges, which happens to be on the positive x axis, is now moved to the center of the circle.

Find the magnitude of the net electric force exerted on this charge.
Express your answer in terms of some or all of the variables q, R, and appropriate constants.

Want an answer?

No answer yet.

**GENERATED REASONING:**

1. Twelve identical point charges q are arranged equidistantly on a circular path with radius R, centered at the origin; this implies a symmetrical distribution of charges.
2. By symmetry, the net force acting on any charge due to the others would originally be zero because forces from opposite charges cancel out.
3. Moving one charge to the center disrupts this symmetry; however, due to radial symmetry, the forces from charges positioned opposite to each other along the circle still cancel out in pairs, leaving only their radial components towards the center.
4. Each pair of opposite charges exerts a force on the central charge q, directed radially inward, whose magnitude can be calculated using Coulomb's law as \( F = k \frac{q^2}{R^2} \) for each charge, where k is Coulomb’s constant.
5. Since there are six such pairs, the total force is the sum of the individual forces from each charge, resulting in a net force magnitude of \( F_{net} = 6 \times k \frac{q^2}{R^2} \).

**'CONTINUATION' it was scored against:**

> Submit this
question to the community.

---

### 4. Δnll = +2.617  (base ppl 19.2 -> +reasoning ppl 263.1)

**CONTEXT:**

> Playing Hitman: Absolution professionally requires a lot of stealth and planning. You scope out the scene, and land your kills in such a fashion that no one will notice.

Playing Hitman: Absolution like—in timtimfed's video's words—an "idiot," however, doesn't require much planning at all. It just requires a trigger happy finger and maybe a chicken suit.

Watch the live action video to see one scene, played out in the two different methods.

**GENERATED REASONING:**

1. Playing Hitman: Absolution professionally demands stealth and strategic planning to execute missions without detection.
2. Conversely, playing the game haphazardly or recklessly, as termed "idiot" style, involves little to no planning and relies on impulsive actions and random use of items like a chicken suit.
3. The contrast between professional play and "idiot" play highlights the range of strategies available within the game.
4. A live-action demonstration comparing these two approaches would visually illustrate the differences in gameplay styles and outcomes.

**'CONTINUATION' it was scored against:**

> Then pick your favorite. I'll go with idiot, because: chicken suit.

---

### 5. Δnll = +2.447  (base ppl 5.8 -> +reasoning ppl 67.1)

**CONTEXT:**

> LOS ANGELES -- Authorities are searching for a hit-and-run driver who struck and killed a man in a wheelchair in the unincorporated Athens area.

Witnesses told California Highway Patrol officers the driver of a blue Chevrolet Tahoe
truck was going east on El Segundo Boulevard, ran a red light at Avalon Boulevard and struck three people about 2:00PM Friday. Witnesses said the license plate number of the SUV was 5GIY964.

The man in the wheelchair was pronounced dead at the scene.

**GENERATED REASONING:**

1. A hit-and-run incident occurred in the Athens area where a driver struck and killed a man in a wheelchair.
2. The incident involved a driver of a blue Chevrolet Tahoe truck running a red light at Avalon Boulevard while traveling east on El Segundo Boulevard.
3. This action directly led to the collision with three individuals, one of whom was in a wheelchair and subsequently died at the scene.
4. The provided license plate number 5GIY964 is a critical piece of evidence for identifying the vehicle and potentially locating the driver responsible for the fatal accident.
5. Given the severity of the crime and the availability of witness statements and a potential identifier like the license plate, authorities are likely conducting a thorough investigation to apprehend the driver.

**'CONTINUATION' it was scored against:**

> The other two victims were expected to be okay.

The Tahoe was last seen going south on Avalon.

---

### 6. Δnll = +2.355  (base ppl 24.4 -> +reasoning ppl 257.0)

**CONTEXT:**

> ESX 4 provisioning issue (35 Views)
Occasional Visitor
Scott Ramnitz
Posts: 1
Registered: ‎06-14-2010
Message 1 of 1 (35 Views)

ESX 4 provisioning issue

I just can't seem to get ESX 4 host provisioning working. I get past the first and second phase and the server boots the "HPSA ESX 4 Assisted Installer." However I lose my network connectivity after this.
Going into the alternate shell I can see that only "download portgroup" is being created a port group, nothing else.
How do you get this phase to create the correct vswitches? I've tried adding those commands to the kickstart with no luck.

**GENERATED REASONING:**

1. The user is attempting to provision an ESX 4 host, which involves setting up virtual switches (vSwitches) and network configurations necessary for communication within the VMware environment.
2. During the provisioning process, the system reaches a stage where it is supposed to automatically configure these network settings, including creating appropriate vSwitches and port groups.
3. The issue arises when, despite progressing through initial phases, the network connectivity is lost post the booting of the "HPSA ESX 4 Assisted Installer," indicating a failure in the automatic configuration of the networking components.
4. Upon investigation using the alternate shell, it's observed that only the "download portgroup" has been created, suggesting that the script or automated process intended to set up additional vSwitches and port groups did not execute correctly.
5. The user has attempted to manually specify the creation of these vSwitches by modifying the kickstart file, a practice commonly used to automate installations in Linux environments, yet this adjustment did not resolve the issue, implying that the problem may lie deeper within the provisioning script or the environment 

**'CONTINUATION' it was scored against:**

> Is anyone doing this?

---

### 7. Δnll = +1.901  (base ppl 5.2 -> +reasoning ppl 34.8)

**CONTEXT:**

> Alerts   |   Russia

Head of Russian television station shot dead

New York, April 18, 2003—Russian journalist Dmitry Shvets, head of the independent television station TV-21 Northwestern Broadcasting in the northern Russian city of Murmansk, was shot dead today outside of the station’s offices. The motive is unclear.

Police have launched an investigation, but no details were available.

**GENERATED REASONING:**

1. A journalist named Dmitry Shvets, who was the head of an independent television station in Murmansk, Russia, was found dead after being shot.
2. The shooting occurred outside the television station's offices, indicating it may be related to his work or position.
3. Since the motive for the shooting remains unclear, it suggests the case involves complex circumstances or possible multiple motives.
4. Authorities have started an investigation, implying they recognize the potential severity and complexity of the situation involving a media figure.

**'CONTINUATION' it was scored against:**

> CPJ will continue to monitor the case.


Like this article? Support our work

---

### 8. Δnll = +1.854  (base ppl 25.7 -> +reasoning ppl 164.3)

**CONTEXT:**

> White House Suddenly Decides Fake News Is a Bad Thing

Whenever you see a photograph of the president making a major address from inside the White House, it's really a picture of him saying "peas and carrots, peas and carrots" after the speech while photographers get their shots. Not anymore, though! For some reason, the White House has decided to stop participating in that particular form of fake-news manufacturing.

**GENERATED REASONING:**

1. The passage implies that previously, photographs of the president giving a major address at the White House were taken after the actual speech, with the president repeating "peas and carrots" to refocus his expression.
2. This practice allowed for staged and potentially misleading imagery, contributing to what the author calls "fake news."
3. The White House has now ceased this practice, suggesting an acknowledgment that such staged photographs could misrepresent reality.
4. By stopping this form of photo-taking, the White House aims to reduce the production of misleading or manipulated media, aligning with a stance against "fake news."

**'CONTINUATION' it was scored against:**

> It will continue to fake other news events, though. [AP, photo via Getty]

---

### 9. Δnll = +1.820  (base ppl 6.8 -> +reasoning ppl 42.0)

**CONTEXT:**

> You use the atg.commerce.pricing.TaxProcessorTaxCalculator class if you are setting up a site that uses third-party software to handle tax calculation. This class consults a TaxProcessor (an implementation of the interface) to determine how much tax to charge for an order.

The TaxProcessorTaxCalculator component is located in the ATG Control Center at atg/commerce/pricing/calculators.

**GENERATED REASONING:**

1. The passage mentions using the atg.commerce.pricing.TaxProcessorTaxCalculator class for sites that require third-party software to manage tax calculations.
2. To use this class effectively, one must understand that it interacts with a TaxProcessor, which is an implementation of a specific interface designed to calculate taxes.
3. The interaction between the TaxProcessorTaxCalculator and the TaxProcessor allows for dynamic tax calculation based on the order details provided by the third-party software.
4. Since the TaxProcessorTaxCalculator is part of the ATG Control Center, users can access and configure it within this environment, specifically under the path atg/commerce/pricing/calculators.
5. Therefore, setting up tax calculation through third-party software involves configuring the TaxProcessorTaxCalculator within the ATG Control Center to ensure accurate tax determination for orders.

**'CONTINUATION' it was scored against:**

> TaxProcessorTaxCalculator class has the following properties:

---

### 10. Δnll = +1.813  (base ppl 3.5 -> +reasoning ppl 21.2)

**CONTEXT:**

> "... we had all tried enough times to pass and kick a ball, we had on our separate rock-strewn sandlots taken enough lumps and bruises, to know that we were viewing something truly fine, something that only comes with years of toil, something very like art." -- Frederick Exley, A Fan's Notes.

"Serious sport has nothing to do with fair play.

**GENERATED REASONING:**

1. The author describes the experience of playing sports on makeshift fields, implying a deep understanding of the effort required to excel.
2. This background allows the players to recognize and appreciate exceptional skill when they see it.
3. The second sentence contrasts the appreciation for skill with the idea that serious competition is not about fairness.
4. It suggests that in high-stakes games, the focus is on winning through any means rather than maintaining equitable play.

**'CONTINUATION' it was scored against:**

> It is bound up with hatred, jealousy, boastfulness, and disregard of all the rules." -- George Orwell, Shooting an Elephant.

---

### 11. Δnll = +1.674  (base ppl 121.2 -> +reasoning ppl 646.3)

**CONTEXT:**

> Peepin' The Stars: The 2007 Bentley Continental GTC

Yeah baby, the Bentley Continental GTC's been Convertible-ized for 2007. This fan fav of Ukrainian mafia now comes built to allow what remains of their hair to blow in the wind, while they snort a line of blow off of the chest of the hungarian hooker in the passenger seat. This incarnation again boasts the 6.0 liter W12 that'll more than likely rip the plugs out of their fake-baked scalps with its blistering-fast top speed of 195 mph.

**GENERATED REASONING:**

1. The Bentley Continental GTC has been converted into a convertible model for 2007, which allows the driver and passengers to have an open-air driving experience.
2. The passage implies that a certain demographic, here referred to as "Ukrainian mafia," is a target audience for this vehicle, suggesting they are likely to purchase it.
3. The car's top speed of 195 mph is emphasized, implying that high-speed driving is a desirable feature for this demographic.
4. The mention of "snort[ing] a line of blow" and the presence of a "hungarian hooker" indicates the implication that drug use and involvement with sex workers are activities associated with this demographic while using the vehicle.
5. The combination of luxurious, fast driving experiences with these activities suggests that the passage is making a commentary on the perceived lifestyle and behaviors of the target demographic.

**'CONTINUATION' it was scored against:**

> Ah, the joys of narco-terrorism.

MTM's Tuner Bentley Continental [internal]

---

### 12. Δnll = +1.638  (base ppl 43.3 -> +reasoning ppl 222.5)

**CONTEXT:**

> Watch It

On DVD: TBD | On Blu-ray: TBD

The 27th Day


A locked-in-the-fifties science fiction film, The 27th Day begins with five different people from five different countries suddenly disappearing from view. They have been gently abducted by the agent (Arnold Moss) of a faraway dying planet, who gives each of the five earthlings a "killing capsule" that will destroy everything on Earth and allow the residents of the alien planet to re-colonize the planet--but which will be ineffective if not used after 27 days.

**GENERATED REASONING:**

1. The passage describes a scenario in which five individuals from various countries are abducted by an alien agent.
2. These individuals are given a "killing capsule" that has the capability to destroy all life on Earth, enabling the aliens to recolonize it.
3. The effectiveness of the killing capsule is time-limited, specifically to 27 days, after which it becomes ineffective.
4. This plot device suggests a race against time for both the abductees and possibly other characters in the story to either prevent the destruction or ensure the capsule's use before the deadline.
5. The narrative implies a conflict between preserving Earth and the survival needs of the alien species facing extinction on their own planet.

**'CONTINUATION' it was scored against:**

> In typical Cold War fashion, the...more

---

## The few cases where reasoning HELPED (Δnll < 0)

### helped 1. Δnll = -0.025  (base ppl 2.8 -> +reasoning ppl 2.7)

**CONTEXT:**

> ED 413 - Field Study IV
(3 cr.) Students gain experience in a middle school or high school setting. Participants discuss the teacher's responsibility to communicate with parents and community agencies to promote K-12 students' learning and well-being. Also, students develop problem-solving skills to ensure that every student learns.

**GENERATED REASONING:**

1. The course ED 413 is designed for students to gain practical experience in middle or high school settings.
2. Effective teaching involves more than just classroom instruction; it requires communication between teachers, parents, and community agencies to support student success comprehensively.
3. To enhance student learning and well-being, teachers must collaborate with external stakeholders to address various needs that impact academic performance.
4. Developing problem-solving skills is essential for teachers to identify and implement strategies that cater to diverse learning needs, ensuring educational outcomes are met for all students.
5. By engaging in discussions about responsibilities and honing their problem-solving abilities, participants in ED 413 are better prepared to create inclusive and supportive learning environments.

**CONTINUATION:**

> This course requires 20 hours of fieldwork over 10 weeks. Prerequisite: ED 412; Every Year, Spring

Spring 2016

---


---

## ROOT CAUSE (confirmed 2026-07-06 by reading the actual records)

The two worst cases are **contiguous** (context+continuation = the full doc, no gap) — but the "continuation"
is a **trailing scrap**, because the split (`first ". " after 65% of chars`) lands near the end of short docs:
- RADAR: context 623 chars ends "...how to make use of your radar system." → continuation = `"[Via MAKE]"` (10 chars).
- OBAMA: context 481 chars ends "...spend your way out of a debt crisis." → continuation = `"Obama the failure."` (18 chars).

Distribution over all 3000 32B docs (continuation length):
- **21.9% empty**, 22.8% ≤20 chars, 25.0% ≤50 chars, 33.6% ≤100, 60.1% ≤200; median 160, mean 184.

**Conclusion:** the negative completeness result is largely an **artifact of a broken split** (plus web
boilerplate), NOT evidence about the reasoning or the completeness idea. The reasoning itself is good; the
"answer key" was a scrap for ~a quarter of docs. FIX: split so the continuation is a substantial multi-sentence
chunk (min length, split earlier, use longer docs), and drop docs whose continuation is boilerplate — then rerun.
