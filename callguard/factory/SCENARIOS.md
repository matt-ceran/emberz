# Emberz scenario list

Generated from `factory/taxonomy.py` (the source of truth). Each seed is multiplied by ~12 LLM-written variants, the protected person's profile and age, the caretaker's rules, known/unknown caller, AI voice, when the ask lands, how the call ends, speaking styles and call length.

**75 seeds.** Held out of training entirely: `gold_courier_pickup` (tests an unseen scam type).

## Scams (32)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `bank_fraud_otp` | elder, adult | someone claiming to be the bank's fraud department | read back the one-time code |  |
| `bank_safe_account` | elder, adult | a 'bank security officer' | wire savings to a new 'protected' account number |  |
| `irs_back_taxes` | elder, adult | an 'IRS agent' with a badge number | pay with gift cards or at a crypto ATM while staying on the line |  |
| `ssa_suspended` | elder, adult | a 'Social Security Administration' officer | confirm the full Social Security number and bank account |  |
| `medicare_card` | elder | a 'Medicare benefits' representative | read the Medicare number and date of birth |  |
| `jury_duty_warrant` | elder, adult | a 'sheriff's deputy' | pay the bond by prepaid card now or be arrested tonight |  |
| `utility_shutoff` | elder, adult | someone from 'the power company' | pay immediately with a card over the phone or a payment app |  |
| `toll_delivery_fee` | elder, adult, teen | an automated-sounding 'toll services' or 'delivery' line | give card number, expiration and the 3 digits on the back |  |
| `carrier_sim_swap` | adult, elder, teen | someone from 'your phone carrier' | read the account PIN and the code texted to the phone |  |
| `health_insurance` | elder | a 'health insurance provider' representative | give the member ID, Social Security number and a card for the 'renewal' |  |
| `debt_collector_threats` | adult, elder | a 'collections agency' caller | settle today with a debit card or prepaid card |  |
| `gold_courier_pickup` (held out) | elder | a 'federal agent' working with the bank | hand cash or gold to the courier at the door |  |
| `police_relative_arrest` | elder | a 'police officer' then a 'public defender' | get cash for bail that a courier will collect |  |
| `tech_support_remote` | elder, adult | 'computer security' support | install the app and read the access code, then log into online banking |  |
| `tech_refund_overpay` | elder | a 'refund department' agent | buy gift cards to return the overpayment and read the numbers |  |
| `shopping_order_fraud` | elder, adult | an automated 'online store' alert transferring to 'security' | give the login password and verification code |  |
| `account_recovery_code` | teen, adult | someone claiming to be from a social media platform's 'support team' | read the login code that was just texted |  |
| `grandparent_jail` | elder | someone pretending to be a grandchild, crying, with a muffled voice | send bail money by wire or gift cards right now |  |
| `voice_clone_accident` | elder, adult | a cloned AI voice of a family member | send money through a payment app to 'a friend's' account |  |
| `virtual_kidnapping` | adult, elder | a threatening man, with screaming in the background | wire ransom money immediately while staying on the line |  |
| `hospital_parent_kid` | kid, teen | an adult claiming to be a parent's coworker | tell them the door code or come outside to meet them |  |
| `lottery_fee` | elder, adult | a 'sweepstakes prize' official | pay the fee with gift cards or give bank account details for 'deposit' |  |
| `game_currency_kid` | kid, teen | someone saying they're from a popular video game | give the game password, or read the numbers on a parent's card 'to verify age' |  |
| `free_phone_survey` | teen, adult | a 'survey' caller | give address, date of birth and a card for 'shipping' |  |
| `romance_overseas` | elder, adult | an online sweetheart they've never met in person | send money by wire or crypto, and keep it from family who 'wouldn't understand' |  |
| `pig_butchering_crypto` | adult, elder | a friendly stranger who started with a 'wrong number' text weeks ago | deposit savings into a trading app they recommend |  |
| `investment_club` | adult, elder | a 'professor' or 'analyst' from a stock investment club | transfer money into the club's platform today before the window closes |  |
| `fake_charity` | elder, adult | a 'disaster relief' volunteer | give a card number over the phone, no time to mail a check |  |
| `job_reshipping` | teen, adult | a 'remote job recruiter' | give Social Security number and bank login for 'direct deposit setup' |  |
| `relative_asks_door_code` | kid | someone saying they're the child's grandpa | tell him the door code | allowed_contact |
| `relative_asks_address` | kid, teen | someone saying they're an aunt | give the home address | allowed_contact |
| `son_asks_bank_login` | elder | someone saying they're her son | read her banking password and the code texted to her | allowed_contact |

## Giving info away (6)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `kid_home_alone_delivery` | kid | an unknown man claiming to have a delivery | confirm the street address and whether anyone is home |  |
| `kid_school_survey` | kid, teen | a caller running a 'school survey' | give home address and when parents leave for work |  |
| `kid_door_code` | kid | a 'repairman' who says the parents booked him | tell him the door or alarm code |  |
| `elder_overshares_card` | elder | a stranger on a friendly sales call | - |  |
| `teen_overshares_schedule` | teen | a new 'friend' met in an online group | say which house is theirs |  |
| `elder_password_share` | elder | a polite caller about an email problem | email password |  |

## Grooming (tactics only) (7)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `game_friend_gifts` | kid, teen | an adult who says they know the child from an online game | meet at a park or the mall without telling parents |  |
| `talent_scout` | teen | an adult claiming to be a music producer or sports scout | come to a private 'audition' alone and not tell parents yet |  |
| `fake_peer_age` | kid, teen | someone claiming to be 13 who slips about driving and their job | send a selfie or a picture of themselves and turn on video chat, and delete the chat after |  |
| `trusted_adult_secret` | kid | a friendly adult who says they live nearby | find out when the child is home alone |  |
| `isolation_vs_parents` | teen | an older 'friend' the teen talks to often | sneak out to meet tonight |  |
| `secret_phone_gift` | teen, kid | an adult online contact | give home address for the 'gift' and hide it from parents |  |
| `ride_offer` | kid, teen | an adult acquaintance from a community group | get picked up after school without telling parents |  |

## Abuse, threats, swearing (6)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `caller_swears_elder` | elder, adult | an angry stranger | - |  |
| `ex_partner_harassment` | adult | an ex-partner | - |  |
| `prank_profanity_kid` | kid | a prank caller (an older teen) | - |  |
| `teen_bully` | teen | a classmate bully | - |  |
| `extortion_screenshots` | teen, adult | an anonymous blackmailer | pay with gift cards within an hour |  |
| `know_where_you_live` | adult, elder, teen | a menacing stranger | - |  |

## Self-harm (non-graphic) (3)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `teen_self_harm` | teen | a close friend the teen trusts | - |  |
| `friend_self_harm` | teen, kid | the protected child's friend | - |  |
| `elder_despair` | elder | a relative checking in | - |  |

## AI voice (1)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `ai_bank_bot_scam` | elder, adult | an AI voice agent pretending to be the bank | read back the verification code |  |

## Harmless look-alikes (hard negatives) (20)

| Seed | Who | Caller | Ask | Flip |
|---|---|---|---|---|
| `real_bank_alert` | elder, adult | the bank's real fraud team | - |  |
| `doctor_office_dob` | elder, adult | a doctor's office receptionist | - |  |
| `pharmacy_bot` | elder, adult | the pharmacy's automated reminder line | - |  |
| `dentist_bot` | adult, elder, teen | a dental office's AI appointment assistant | - |  |
| `friend_mom_playdate` | kid | a friend's mom calling from an unknown number | - |  |
| `surprise_party_aunt` | kid, teen | an aunt (relative) | - |  |
| `gamer_trash_talk` | teen, kid | a friend from class | - | swearing |
| `hyperbole_teen` | teen | a best friend | - |  |
| `dark_humor_adult` | adult, elder | a coworker or old friend | - |  |
| `grandkid_real_call` | elder | the real grandchild from their known number | - |  |
| `coach_schedule` | kid, teen | the real soccer coach | - |  |
| `school_office` | teen, kid | the school front office | - |  |
| `delivery_driver_real` | adult, elder | a real delivery driver | - |  |
| `isp_callback_real` | adult, elder | the internet provider returning a support ticket they opened | - |  |
| `food_bank_thanks` | elder | a local food bank volunteer | - |  |
| `police_followup_real` | adult, elder | a police officer following up on a report they filed | - |  |
| `elder_old_age_talk` | elder | a sister calling to catch up | - |  |
| `family_money_talk` | elder | her real son | - | money_talk |
| `friend_meetup` | teen | a classmate the teen knows | - | meetups |
| `telemarketer_legit` | adult, elder | a pushy but legit telemarketer selling solar panels | - |  |

## Keyword lexicon (B0 baseline and generator hints)

- **grooming.secrecy**: "our little secret"; "don't tell your parents"; "don't tell your mom"; "just between us"; "keep this between you and me"; "nobody needs to know"; "delete our messages"; "delete the chat"; "you'll get in trouble if you tell"; "it's our thing"
- **grooming.isolation**: "your parents don't get you"; "i'm the only one who understands you"; "talk on another app"; "add me on"; "do you have your own phone"; "use a private account"; "tell your mom you're at a friend's"; "they wouldn't understand"; "you can trust me more than them"
- **grooming.gifts_flattery**: "you're so mature for your age"; "you're not like other kids"; "i'll send you robux"; "i'll send you v-bucks"; "i'll buy you"; "gift card for you"; "you're special"; "you're really talented"; "i'll get you a new phone"
- **grooming.photo_request**: "send me a picture of you"; "send a selfie"; "turn on your camera"; "video call me"; "what do you look like"; "send me a pic"
- **grooming.meeting_request**: "let's meet"; "i can pick you up"; "come alone"; "meet me at the park"; "don't bring anyone"; "sneak out"; "i'll wait for you after school"
- **disclosure.home_alone**: "are your parents home"; "are you home alone"; "when does your mom get home"; "who's there with you"; "is anyone else home"; "what time do your parents leave"
- **disclosure.location_school**: "what school do you go to"; "what street"; "what's your address"; "which grade"; "where do you live"; "what's your teacher's name"; "which house is yours"
- **disclosure.credentials_otp**: "read me the code"; "verification code"; "one-time code"; "your password"; "pin number"; "door code"; "alarm code"; "the code we just sent"; "access code"
- **disclosure.financial_id**: "card number"; "social security"; "routing number"; "account number"; "expiration date"; "three digits on the back"; "medicare number"; "member id"; "date of birth"
- **scam.impersonation_bank_gov**: "fraud department"; "irs"; "warrant"; "suspended"; "badge number"; "safe account"; "bench warrant"; "will be arrested"; "shut off"; "unpaid toll"; "courier"
- **scam.tech_support**: "virus"; "hackers"; "remote access"; "download this app"; "your computer is infected"; "refund"; "anydesk"; "teamviewer"; "screen sharing"
- **scam.family_emergency**: "i'm in jail"; "bail"; "accident"; "don't tell mom"; "lawyer"; "hospital"; "kidnapped"
- **scam.prize**: "you've won"; "sweepstakes"; "processing fee"; "claim your prize"; "free robux"; "free phone"
- **scam.romance_investment**: "guaranteed returns"; "crypto"; "trading platform"; "investment opportunity"; "my love"; "customs fee"; "bitcoin"
- **pressure**: "right now"; "within the hour"; "don't hang up"; "stay on the line"; "act fast"; "today only"; "don't tell anyone"; "gift cards"; "wire transfer"; "bitcoin atm"; "cash pickup"
- **vulgar**: "fuck"; "shit"; "bitch"; "asshole"; "bastard"; "damn you"; "piece of shit"; "screw you"; "dumbass"
- **threat_extortion**: "i know where you live"; "you'll regret"; "or else"; "i'll send these to"; "we have your"; "you'll never see"; "i'll hurt"; "i'm coming over"; "pay or"
- **self_harm**: "kill myself"; "want to die"; "end it all"; "hurt myself"; "not be here anymore"; "no reason to live"; "better off without me"; "cut myself"

## Hard-negative phrases (sprinkled into harmless calls)

"i'm gonna kill you next round"; "you're dead in this game"; "shoot me a text"; "my phone is dead"; "this traffic is killing me"; "don't tell mom about the surprise party"; "what's the wifi password again"; "you're killing it"; "i could die of embarrassment"; "my battery's dying"; "the code is on the fridge"; "my knees are killing me"; "i'll murder you at chess"; "keep it a secret, it's her birthday"; "are your parents home? i want to invite you to the party"; "what school? oh we go there too"

## Caretaker rules pools

- **kid**: Never share our address, the school name or the door code; Tell me right away if someone asks whether we're home; Alert me if an adult asks to meet up or to keep a secret from us; Alert me about any request for photos or video calls; Tell me if someone offers gifts, game currency or money; Never give out my card or any passwords
- **teen**: Tell me if anyone asks to meet up in person; Alert me about any talk of self-harm right away; Normal swearing with friends is fine; Alert me if someone asks for passwords or login codes; Tell me if an adult asks to move to another app or keep it secret; Tell me about any threats or blackmail
- **elder**: Never share codes, passwords or PINs; Alert me about any talk of gift cards, wire transfers, crypto or cash pickups; Tell me if anyone claims to be from a bank, the IRS, Medicare or the police; Don't let anyone talk her into installing apps on her computer; Alert me if anyone asks her to keep a secret from family; Never read out card or Social Security numbers
- **adult**: Never share one-time codes; Alert me about investment or crypto pitches; Tell me about any threats; Alert me if anyone asks for my Social Security number
- **flip rules** (create rule-flip pairs): Tell me if anyone swears at or around my kid; Alert me about any talk of money or bank details, even with family; Tell me if anyone, even a friend, asks to meet up in person; Tell me if anyone asks to install an app or click a link
