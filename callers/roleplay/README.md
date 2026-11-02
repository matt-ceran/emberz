# Role-play guardian calls (V2)

There are 40 blind-written cards in `cards.json`:

| Mix | Cards | What it tests |
|---|---|---|
| `elder_scam` | rp01-rp12 | scams that build to an ask: codes, money, gift cards, account numbers, remote access |
| `kid_stranger` | rp13-rp20 | strangers getting a kid to disclose things or meet up. Tactics only, never sexual content |
| `rule_flip` | rp21-rp28 | 4 pairs that use the same words but a different caller (allowed or unknown) or different rules |
| `benign_hard_negative` | rp29-rp40 | real banks, a doctor's office, a school, a coach, family talking about money, deliveries |

A card gives the scenario, the caller's goal and beats, the protected person's persona and the active rules.
**It never says what the guardian should do.** Labels come later, from the transcript only.

## Before the session

- Run `.venv/bin/python -m callers.roleplay.cards check`.
- Print the cards:
  `.venv/bin/python -m callers.roleplay.cards print --out ~/Harbor\ Share/emberz-cards`
  You get `rpXX_caller.md` and `rpXX_protected.md` for each card, plus `session_setup.json`.
  - Give the caller only their caller card.
  - Give the protected person only their protected card. It doesn't show the caller's goal or ask, so they react naturally.
- People: one caller and one protected person per call. Kid roles are played by adults.
- Put the two people in **different rooms**, or have one of them call from outside. If they're in one room, each voice leaks into the other's microphone and the two tracks stop being clean.
- Phones:
  - The caller dials the guarded number.
  - For `caller_line: unknown` cards, use a phone that is **not** on the allowed list.
  - For `allowed_contact` cards, use the phone registered as that contact (Grandpa, Aunt Sofia, Dan, Leo).
  - The protected person's phone is the protected number the line rings (leg B).
- Line: the server is running, the tunnel is up and the demo app points at it (L1). Every call goes through `CallRecorder`.
- Read the consent text below to everyone, once, and get a yes from each person.

## For each call

1. Set the line's profile, rules and allowed contacts to the card's `session_setup.json` entry. This uses L1's per-call config.
2. Log it: `.venv/bin/python -m callers.log_call roleplay rp07`.
3. The caller dials, the notice plays, the protected phone rings and they talk for the card's length. Speak naturally and don't read the beats aloud. On rule-flip cards, the quoted lines are said word for word.
4. Anyone can say **"pineapple"** to stop the call. Codes and card numbers are always made up.
5. Hang up. Wait about 10 seconds before the next call, so the recording closes.

Shuffle the card order. Don't run the two cards of a rule-flip pair back to back.

## After the session

```sh
.venv/bin/python -m callers.label_sheet match-sources   # ties each recording to its card (source.json)
.venv/bin/python -m callers.label_sheet sheet --all
```

1. Send Matt the `recordings/to_label/` folder only (one `.csv` and one `.md` per call), through the shared folder, never the call folders: those hold `events.jsonl` (model outputs) and `source.json` (which card).
2. Matt labels each turn's risk, category and stage, and marks the ask turn.
3. Matt imports the labels with `label_sheet import`.
4. Commit `recordings/labels/` **before** any model scores the role-play set (S4).

## Consent text (read aloud before the session)

> "These are test calls for a hackathon project called Emberz. Every call to the test number is recorded and
> transcribed: both sides, audio and text. Please use made-up names, codes and numbers, never real ones.
> Recordings stay with Numan and Matt and are deleted within 30 days after the hackathon. We share only
> numbers and charts, never your voice. You can stop any call by saying 'pineapple', and you can ask us to
> delete your calls at any time. Is that okay with you?"
