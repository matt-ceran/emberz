## Business value

**In short:** Americans 60+ reported $7.75B in fraud losses to the FBI in 2025, and older adults' median loss is far higher on scams that start with a phone call ($3,000) than on social media ($599). Emberz guards one phone number for about **$3.64 per household per month** in line and model costs (assumptions below). For a family paying an illustrative ~$10/month ($120 a year), it pays for itself if it stops **one median phone-fraud loss ($3,000) per 25 household-years**. Once a GPU carries real traffic, our fine-tuned 4B guardian's model cost is about **13.5x lower than Qwen3.8 Max and 22.5x lower than Kimi K3** (2.5x and 3.6x lower once the phone line is included). How many scams it actually catches comes from our eval: {{catch_before_ask}} at {{false_alarms_per_100}} false alarms per 100 benign calls.

### Who uses it and why it matters

- **Adult children of an aging parent** (the main buyer). People 60+ filed 201,266 complaints to the FBI in 2025 and reported **$7.75B** in losses, up 59%. Their average loss was $38,500, and 12,444 of them lost more than $100k [1]. The two call-center scam types, tech support and government impersonation, cost them **$1.45B**, about 19% of their losses [1]. The FTC estimates the true 2025 cost of fraud to older adults at **$13.3B-$119.5B** once unreported cases are counted [2].
- **Why phone in particular.** For older adults, the median loss on scams that start with a phone call is **$3,000**, against $599 for scams that start on social media. People 80+ lose more on phone scams than on any other contact method [2]. Government impostors took 41% of older adults' phone-fraud losses [2]. FTC reports put 2025 impostor losses at **$3.5B** for all ages: about $1B to business impostors (banks lead) and $920M to government impostors [3].
- **AI voices are already being used.** The FBI logged 22,364 complaints mentioning AI in 2025 ($893M in losses), including 3,143 from people 60+ ($352M). Victims reported losing over $5M to grandparent or "distress" scams, a type in which cloned voices of loved ones are used [1]. The FCC ruled that AI-generated voices count as "artificial" under the TCPA, and it named voice clones posing as a family member as a harm [4].
- **Parents of kids with a first phone** (a second market; we don't put a dollar figure on it). NCMEC received 1.4M online-enticement reports in 2025, up 158%. These include 800+ cases where the offender traveled to meet the child, and 50,000+ financial-sextortion reports [5][6]. These are reports about online platforms, not voice calls. Emberz covers only the call part ("Are your parents home? What street?"), and we don't count that toward the dollar value.

### What exists and how we differ

| Product | Price | What it doesn't do (that Emberz does) |
|---|---|---|
| Google/Samsung Scam Detection | Built into the phone, no fee listed (Pixel 6+ in the US, Pixel 9+ in some other countries; Galaxy S26 in US English) [7][18] | Runs on the device and warns only the phone's user [18]; on the Galaxy S26 it is off for saved contacts [7]. Neither page mentions caretaker alerts, caretaker rules, a kids mode or an AI-voice check |
| Carrier add-ons (Verizon Call Filter Plus, T-Mobile Scam Shield Premium) | $3.99-$4 per line per month [8][9] | Caller ID, spam lookup and blocking by number or category. The pages describe no analysis of what is said on the call |
| Aura Family | $32 per month billed annually, $50 billed monthly; includes AI spam call protection [10] | Checks unknown calls before they ring through, plus identity monitoring. The page describes no listening once a call is connected and no family alerts about calls |
| Bark Premium | $14 per month or $99 per year [11] | Covers kids' texts, email and social apps; phone calls aren't listed |
| Truecaller Premium (AI Call Scanner), Hiya AI Voice Detection | Truecaller Premium $9.99 per month on the US App Store; Hiya's price isn't published [12][13] | They check whether the voice is synthetic; neither describes checking what the caller is asking for |
| OnGuardAI | Subscription, price not published [14] | It texts the user and a family member a list of red flags and keeps no transcript. The launch article mentions no AI-voice check, no family-written rules and no kids mode; it is "mainly designed for seniors" [14] |

### Cost model (per protected call-minute)

**Assumptions:**
- **Tokens:** each scored turn sends about **1,500 input tokens** (rules plus the recent transcript window) and returns about **100 output tokens** of guardian JSON.
- **Turns:** about **10 scored turns per call-minute**. That makes 15k input and 1k output tokens per minute.
- **GPU:** an H100 costs $8/h, billed per second [15]. We assume one H100 serving the 4B LoRA can carry **100 calls at once** within our latency target ({{calls_per_h100}}, to be measured), with **50% average utilization**, so about 50 calls at once on average.
- **Phone line:** Telnyx list prices [16]. The inbound leg costs $0.0052/min ($0.002 Voice API + $0.0032 SIP inbound), the leg to the protected phone $0.007/min ($0.002 + $0.005 SIP outbound), two conference participants $0.004/min and the media stream $0.0035/min, for **$0.0197/min** in total. Deepfake detection is $0.01 per call and the number is $1 per month.
- **Not counted:** speech-to-text, which runs locally on CPU (37 ms per 160 ms audio chunk on the Mac, PLAN 2b), the caretaker's monitor leg, which is only used after an alert, and any text-to-speech fee for the coach line's spoken tips.

| | **Ours: Qwen3-4B + LoRA, on-demand H100** | B5 Qwen3.8 Max, serverless ($2/$6 per 1M) | B4 Kimi K3, serverless ($3/$15 per 1M) |
|---|---|---|---|
| Guardian $/call-minute | **$0.0027** at 50 calls at once on average. It is $0.133 with one call on the GPU and $0.013 with 10 | $0.036 (reasoning on, about 700 output tokens a turn: $0.072) | $0.060 |
| Telnyx line $/call-minute | $0.0197 | $0.0197 | $0.0197 |
| **Total $/call-minute** | **$0.022** | $0.056 | $0.080 |
| Per household-month (100 call-minutes, 40 calls, 1 number) | **$3.64** | $6.97 | $9.37 |
| Measured guardian cost (from the eval) | {{cost_per_min_ours}} | {{cost_per_min_b5}} | {{cost_per_min_b4}} |

**Where each option wins:**
- **At low traffic, serverless is cheaper.** A dedicated H100 beats Qwen3.8 Max only once it carries more than about **3.7 calls at once** on average, and Kimi K3 at more than about 2.2. During the hackathon, with one call at a time, the frontier models are cheaper per minute.
- **At scale, the small model is cheaper.** One H100 at the assumed load handles about 2.2M call-minutes a month, or roughly 21,000 households.
- **Caching would cut the serverless prices.** Both serverless models offer cached input ($0.25 and $0.30 per 1M), which would lower their cost if the rules prefix is reused. We report list prices.

### Expected benefit per protected person

Value per year ≈ phone-fraud loss per person × {{catch_before_ask}} × p_heeded. Here p_heeded is the share of caught asks where the coach line or the caretaker actually stops the payment. It has no source, and the illustration below assumes 0.5.

- **Loss per older adult.** The FTC's $13.3B-$119.5B, divided by the 64.6M Americans aged 65+ [17], gives **at most $206-$1,850 per older adult per year**. It is an upper bound because the FTC figure covers everyone 60+, a larger group than the 65+ count we divide by.
- **Share that starts with a call.** Taking the IC3 call-center share of 60+ losses (19%) [1] as a stand-in for phone-led fraud gives **at most $39-$347 per older adult per year**.
- **Value of better catching.** Each 10 points of catch-before-ask is then worth up to **$2-$17 per person-year** at p_heeded = 0.5.

The population average understates the value for real buyers. We expect families to sign up a parent who is 80+ or already being targeted (an assumption). People 80+ report the highest median loss of any age group ($2,000) and lose more to scams that start with a phone call than to any other contact method [2]. So the per-family break-even figure (one prevented $3,000 median loss per 25 household-years) is the more useful number. The other measured inputs are lead time ({{lead_time_turns}} turns between the alert and the ask) and turn-to-coach latency ({{turn_to_coach_p95_ms}} ms p95).

### Adoption barriers (honest list)

- **Consent and notice.** Every call starts with a recorded notice, which is required in all-party-consent states such as California (Penal Code 632.7). The protected adult has to opt in themselves; a caretaker can't consent for a competent adult (PLAN 10). Scammers will hear the notice too, which may deter some of them but also warns them.
- **Giving out a new number.** A guarded number only helps if callers use it. The alternatives each cost something:
  - **Porting the existing number** to the line can take days and needs carrier paperwork.
  - **Conditional call forwarding** depends on the carrier; OnGuardAI says "the economics only work where carriers don't charge for call forwarding" [14].
- **False alarms.** At our target of at most 5 per 100 benign calls, a household getting 40 calls a month would still trigger about 2 caretaker alerts a month. Our controls:
  - "watch" is only logged;
  - only "alert" and above calls the caretaker;
  - the allowed-contact list and the caretaker's written rules keep known callers quiet.
- **Carrier rules.** A forwarded call that shows the original caller's number may get weaker STIR/SHAKEN attestation, so the protected phone could label it as spam (a risk we have not yet checked; it is part of the L1 hour-0 caller-ID test). The FCC treats AI-voice calls as robocalls that need prior consent [4], and that also applies to our automated alert calls, so they go only to caretakers who signed up for them.
- **Minors' privacy.** Under COPPA a recording of a child's voice is personal information, and the amended rule (compliance date April 22, 2026) adds a written data-retention policy and a security program to verifiable parental consent [19]. Our own policy goes further: no voiceprints and deletion after 7 days (PLAN 10).
- **Latency on the coach line.** The tip has to arrive before the person answers the ask. ASR adds about 37 ms per 160 ms chunk, and the full turn-to-coach latency is measured live ({{turn_to_coach_p95_ms}}). Frontier-model latency is measured on the same turns rather than assumed; reasoning mode adds about 600 output tokens per turn (PLAN 7), which costs time on every turn.
- **Trust.** Families have to accept a model listening to their calls. To earn that, we publish the open model, the eval and the failure cases.

Not legal advice.

**Sources** (report year in brackets; all accessed 2026-11-02)
1. FBI IC3, *2025 Internet Crime Report* [2025 data, pub. Apr 2026], elder fraud pp. 44-48, AI pp. 39-40, call-center fraud p. 21: https://www.ic3.gov/AnnualReport/Reports/2025_IC3Report.pdf
2. FTC, *Protecting Older Consumers 2025-2026* [2025 data, pub. Oct 19, 2026], pp. 1, 21-31: https://www.ftc.gov/system/files/ftc_gov/pdf/2026-older-adults-report-10192026.pdf
3. FTC press release, impostor scams [2025 data, pub. Jun 15, 2026]: https://www.ftc.gov/news-events/news/press-releases/2026/06/ftc-data-show-people-reported-losing-3-point-5-billion-imposter-scams-2025 (total fraud of $15.9B in 2025: FTC testimony to the JEC, Mar 2026, https://www.ftc.gov/system/files/ftc_gov/pdf/ftc-testimony-jec-hearing-on-the-rising-scam-economy.pdf)
4. FCC Declaratory Ruling FCC 24-17 [Feb 2024]: https://docs.fcc.gov/public/attachments/FCC-24-17A1.pdf
5. NCMEC CyberTipline data [2025]: https://www.missingkids.org/cybertiplinedata
6. NCMEC, sextortion data [2025, pub. Jun 25, 2026]: https://us.missingkids.org/blog/2026/ncmec-releases-new-sextortion-data-2025
7. Android Authority, Scam Detection on Galaxy S26 [Feb 25, 2026]: https://www.androidauthority.com/google-scam-detection-samsung-galaxy-s26-3643942/
8. Verizon Call Filter FAQ [2026]: https://www.verizon.com/support/call-filter-faqs/
9. T-Mobile Scam Shield [2026]: https://www.t-mobile.com/benefits/scam-shield
10. Aura pricing [2026]: https://www.aura.com/pricing
11. Bark pricing [2026]: https://support.bark.us/en/articles/13460996-pricing
12. Truecaller AI Call Scanner, part of Premium, Android in the US [May 31, 2024]: https://www.speechtechmag.com/Articles/News/Speech-Technology-News/TrueCaller-Launches-AI-Call-Scanner-164270.aspx ; price: US App Store in-app purchases ("Truecaller Premium Monthly $9.99") [2026]: https://apps.apple.com/us/app/truecaller-phone-call-blocker/id448142450
13. Hiya AI Voice Detection [2026]: https://www.hiya.com/products/ai-voice-detection
14. BetaKit, OnGuardAI launch [Aug 17, 2026]: https://betakit.com/onguardai-launches-app-to-flag-scam-calls-in-real-time/
15. Fireworks pricing [2026]: https://fireworks.ai/pricing ; Qwen3.8 Max: https://fireworks.ai/models/fireworks/qwen3p8-max ; Kimi K3: https://fireworks.ai/models/fireworks/kimi-k3
16. Telnyx Call Control pricing [2026]: https://telnyx.com/pricing/call-control
17. U.S. Census Bureau, Vintage 2025 estimates, Table 1 (65+ on July 1, 2025: 64,617,088) [pub. Jun 25, 2026]: https://www.census.gov/newsroom/press-releases/2026/vintage-2025-pop-estimates.html
18. Google, Use Scam Detection (Phone app Help) [2026]: https://support.google.com/phoneapp/answer/15654065?hl=en
19. FTC, COPPA Rule amendments, Federal Register [Apr 22, 2025; compliance date Apr 22, 2026]: https://www.federalregister.gov/documents/2025/04/22/2025-05904/childrens-online-privacy-protection-rule
