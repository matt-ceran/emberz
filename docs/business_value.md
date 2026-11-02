## Business value

**In short:** phone scams cost older Americans billions a year and hit hardest per victim. Emberz guards one phone number for about **$3.64 per household per month** in line and model costs (assumptions below). Paid for at a ~$10/month price, it breaks even if it stops **one median phone-fraud loss ($3,000) per 25 household-years**. Our fine-tuned 4B guardian costs about **13x less per protected minute than Qwen3.8 Max and 22x less than Kimi K3**, once a GPU carries real traffic. How many scams it actually catches comes from our eval: {{catch_before_ask}} at {{false_alarms_per_100}} false alarms per 100 benign calls.

### Who uses it and why it matters

- **Adult children of an aging parent** (the main buyer). People 60+ filed 201,266 complaints to the FBI in 2025 and reported **$7.75B** in losses, up 59%. Their average loss was $38,500, and 12,444 of them lost more than $100k [1]. The two call-center scam types, tech support and government impersonation, cost them **$1.45B**, about 19% of their losses [1]. The FTC estimates the true 2025 cost of fraud to older adults at **$13.3B-$119.5B** once unreported cases are counted [2].
- **Why phone in particular.** Scams that start with a phone call have the highest median loss for older adults: **$3,000**, against $599 for scams that start on social media. People 80+ lose more on phone scams than on any other contact method [2]. Government impostors took 41% of older adults' phone-fraud losses [2]. FTC reports put 2025 impostor losses at **$3.5B** for all ages: about $1B to business impostors (banks lead) and $920M to government impostors [3].
- **AI voices are already being used.** The FBI logged 22,364 complaints mentioning AI in 2025 ($893M in losses), including 3,143 from people 60+ ($352M). Grandparent or "distress" scams that use a cloned voice of a loved one cost victims over $5M [1]. The FCC ruled that AI-generated voices count as "artificial" under the TCPA, and it named voice-cloned family emergencies as a harm [4].
- **Parents of kids with a first phone** (a second market; we don't put a dollar figure on it). NCMEC received 1.4M online-enticement reports in 2025, up 158%. These include 800+ cases where the offender traveled to meet the child, and 50,000+ financial-sextortion reports [5][6]. Most of this happens in messaging apps, not voice calls. Emberz covers only the call part ("Are your parents home? What street?"), and we don't count that toward the dollar value.

### What exists and how we differ

| Product | Price | What it doesn't do (that Emberz does) |
|---|---|---|
| Google/Samsung Scam Detection | Free (Pixel, Galaxy S26; US, English) | It warns only the person on the call, doesn't check calls from saved contacts, and has no caretaker, no rules, no kids mode and no AI-voice check [7] |
| Carrier add-ons (Verizon Call Filter Plus, T-Mobile Scam Shield Premium) | $3.99-$4 per line per month [8][9] | They label and block known spam numbers. They don't listen to the call itself |
| Aura Family | $32 per month billed annually, $50 billed monthly; includes spam call protection [10] | Spam filtering plus identity monitoring. No live call guardian, no caretaker alerts |
| Bark Premium | $14 per month or $99 per year [11] | Monitors kids' texts and apps, not calls |
| Truecaller Premium (AI Call Scanner), Hiya AI Voice Detection | Truecaller ~$9.99 per month (third-party source); Hiya's price isn't published [12][13] | They check whether the voice is synthetic, not what the caller is asking for |
| OnGuardAI | Subscription, price not published [14] | It texts a family member a list of red flags. It keeps no transcript and has no AI-voice check, no caretaker rules and no kids mode |

### Cost model (per protected call-minute)

**Assumptions:**
- **Tokens:** each scored turn sends about **1,500 input tokens** (rules plus the recent transcript window) and returns about **100 output tokens** of guardian JSON.
- **Turns:** about **10 scored turns per call-minute**. That makes 15k input and 1k output tokens per minute.
- **GPU:** an H100 costs $8/h, billed per second [15]. We assume one H100 serving the 4B LoRA can carry **100 calls at once** within our latency target ({{calls_per_h100}}, to be measured), with **50% average utilization**, so about 50 calls at once on average.
- **Phone line:** Telnyx list prices [16]. The inbound leg costs $0.0052/min, the leg to the protected phone $0.007/min, two conference participants $0.004/min and the media stream $0.0035/min, for **$0.0197/min** in total. Deepfake detection is $0.01 per call and the number is $1 per month.
- **Not counted:** speech-to-text, which runs locally on CPU (37 ms per 160 ms audio chunk on the Mac, PLAN 2b), and the caretaker's monitor leg, which is only used after an alert.

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
- **Share that starts with a call.** Taking the call-center share (19%) as a stand-in for phone-led fraud gives **at most $39-$347 per older adult per year**.
- **Value of better catching.** Each 10 points of catch-before-ask is then worth up to **$2-$17 per person-year** at p_heeded = 0.5.

The population average understates the value for real buyers. Families sign up a parent who is 80+, isolated or already being targeted, and that is exactly the group with the highest phone losses [2]. So the per-family break-even figure (one prevented $3,000 median loss per 25 household-years) is the more useful number. The other measured inputs are lead time ({{lead_time_turns}} turns between the alert and the ask) and turn-to-coach latency ({{turn_to_coach_p95_ms}} ms p95).

### Adoption barriers (honest list)

- **Consent and notice.** Every call starts with a recorded notice, which is required in all-party-consent states such as California (Penal Code 632.7). The protected adult has to opt in themselves; a caretaker can't consent for a competent adult (PLAN 10). Scammers will hear the notice too, which may deter some of them but also warns them.
- **Giving out a new number.** A guarded number only helps if callers use it. The alternatives each cost something:
  - **Porting the existing number** to the line takes days and carrier paperwork.
  - **Conditional call forwarding** depends on the carrier; OnGuardAI says "the economics only work where carriers don't charge for call forwarding" [14].
- **False alarms.** At our target of at most 5 per 100 benign calls, a household getting 40 calls a month would still trigger about 2 caretaker alerts a month. Our controls:
  - "watch" is only logged;
  - only "alert" and above calls the caretaker;
  - the allowed-contact list and the caretaker's written rules keep known callers quiet.
- **Carrier rules.** When a forwarded call shows the original caller's number, it can get weaker STIR/SHAKEN attestation, and the protected phone may then label it as spam. The FCC treats AI-voice calls as robocalls that need prior consent [4], and that also applies to any outbound alert calls we automate.
- **Minors' privacy.** Under the amended COPPA rule (compliance date April 22, 2026), a child's voice is personal information. We need verifiable parental consent, written retention and security policies, no voiceprints and 7-day deletion (PLAN 10).
- **Latency on the coach line.** The tip has to arrive before the person answers the ask. ASR adds about 37 ms per 160 ms chunk, and the full turn-to-coach latency is measured live ({{turn_to_coach_p95_ms}}). The serverless frontier models are slower and less predictable here, and reasoning mode adds about 600 output tokens per turn.
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
12. Truecaller AI Call Scanner [May 2024]: https://www.businesswire.com/news/home/20240528009351/en/ ; price from a third-party comparison, https://robokiller.com/blog/top-subscription-based-options-for-blocking-repeat-spam-calls-in-the-us/
13. Hiya AI Voice Detection [2026]: https://www.hiya.com/products/ai-voice-detection
14. BetaKit, OnGuardAI launch [Aug 17, 2026]: https://betakit.com/onguardai-launches-app-to-flag-scam-calls-in-real-time/
15. Fireworks pricing [2026]: https://fireworks.ai/pricing ; Qwen3.8 Max: https://fireworks.ai/models/fireworks/qwen3p8-max ; Kimi K3: https://fireworks.ai/models/fireworks/kimi-k3
16. Telnyx Call Control pricing [2026]: https://telnyx.com/pricing/call-control
17. U.S. Census Bureau, Vintage 2025 estimates, Table 1 (65+ on July 1, 2025: 64,617,088) [pub. Jun 25, 2026]: https://www.census.gov/newsroom/press-releases/2026/vintage-2025-pop-estimates.html
