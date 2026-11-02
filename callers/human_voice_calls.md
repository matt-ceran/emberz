# Human voice calls for V1 (about 50 calls)

**Goal:** real human speech over the same Telnyx path as the AI test callers, recorded on the caller track.
This is the "human" class for the AI-voice detector (V1) and the reply-gap timing cue. The AI class comes from
the 50 Vapi/Retell calls (`callers/agents`). Both classes go through the same phone network, the same Telnyx
leg and the same μ-law codec into `recordings/<call_id>/caller.wav`. The detector then learns AI vs human,
not phone line vs clean audio.

## Who

- **10 callers, 5 calls each.**
  - Numan and Matt.
  - Up to 8 consenting adult friends or family.
  - Aim for a mix of gender, age (include at least 2 people over 60), accent and speaking pace.
- **Phones:**
  - Each caller uses their own mobile.
  - Get at least 2 different carriers and both iPhone and Android.
  - Include 1-2 landline or VoIP calls if anyone has one.
- **Places:**
  - About 60% quiet room.
  - About 20% speakerphone.
  - About 20% outside or in a car (not while driving).
- **Pseudonyms only** (p01-p10). The pseudonym-to-person map stays on paper or in Numan's notes, never in the repo.

## What they say

The content should match the AI calls, so the detector can't tell the classes apart by topic.

- **3 of 5 calls: read-then-improvise.** The caller opens with the first message of one of the 5 agents,
  with a variant's values filled in, then carries on in their own words for 1-2 minutes. Print the scripts with:
  ```sh
  .venv/bin/python -c "from callers.common import *; [print(s['id'], '|', render(s['first_message'], s['variants'][i])) for s in load_specs() for i in range(2)]"
  ```
  Use each of the 5 scripts about equally: about 6 calls per script across all callers.
- **2 of 5 calls: free talk.** Any everyday topic: plans for the weekend, a recipe, asking for a favor.
  These give natural fillers ("um", false starts), which AI agents rarely produce.
- **Start talking within about 1 second of hearing "hello".** V1 uses the first 2-4 s of caller audio, and the AI agents also speak first.
- Someone answers the protected phone and talks back normally, so reply gaps are real conversational gaps.
- Calls last 1-3 minutes. No real personal data; make up names and numbers.

## How each call runs

1. Numan sets the line to a neutral adult profile with no special rules. These calls aren't scored by the guardian.
2. Log the call: `.venv/bin/python -m callers.log_call human p03 --script pharmacy_refill --phone android --place quiet`.
   For free talk, leave out `--script`.
3. The caller dials the guarded number, hears the notice and talks with whoever answers the protected phone.
4. After the session, run `.venv/bin/python -m callers.label_sheet match-sources`. Each recording gets its
   `source.json` (`kind: human`), which V1 uses as the class label. No turn labeling is needed for these calls.

## Consent

Send this in writing (text or email) the day before, and get a written "yes":

> "Hi! For our hackathon project (Emberz, a scam-call guard for families) we need about 5 short phone calls from
> you, around 2 minutes each. You'll call a test number and read a short line, then chat. The calls are
> **recorded and transcribed**: your voice and what you say. We use the recordings only to train and test a
> detector that tells real people from AI voices. We never build a voiceprint, never identify you, and never
> publish your audio; only overall numbers and charts. In our recordings your phone number is stored only as a
> one-way hash (the phone company's call records still show it, as for any call).
> The audio stays with Numan and Matt and is deleted within 30 days after the hackathon, or sooner if you
> ask. Please use made-up names and numbers on the calls. Is that okay? You can say no, or stop at any time."

At the start of every call, the line also plays the "This call is protected by Emberz and may be transcribed"
notice, which covers all-party-consent states (PLAN.md section 10).

## Data handling

- Audio is kept on Numan's Mac and the shared folder, gitignored. Only `recordings/labels/` is committed, and
  these calls have no labels.
- `meta.json` stores phone numbers only as salted hashes (`line/recorder.py`). Set `EMBERZ_NUMBER_SALT` in `.env`
  so the same phone hashes the same way across sessions.
- **Not used:** ElevenLabs output, anyone under 18, and any EmberDial customer audio.
- Delete everything 30 days after the hackathon. The README's teardown checklist (D2) gets a line for this.
