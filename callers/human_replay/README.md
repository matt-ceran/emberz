# callers/human_replay: real human speech over the real phone path

V1's human class (PLAN.md V2 + 2c). We play 60 recordings of real people as real phone calls from the
SignalWire number **+1 304 648 1098** into the guarded Telnyx number **+1 323 516 1282**. The audio goes
through the PSTN, Telnyx and the μ-law codec, the same path as the AI test callers. The line records it in
`recordings/<call_id>/caller.wav`. The detector then learns AI vs human, not clean audio vs phone audio.

| | |
|---|---|
| Clips | 60 clips of 6-12 s (6.1-11.3 s in the current build), **60 distinct speakers**, 30 female / 30 male |
| Sources | 30 LibriSpeech test-clean (read speech) + 30 EdAcc (relaxed conversations, 24 accent labels: Kenyan, Nigerian, Ghanaian, Jamaican, Indian, Indonesian, Irish, Scottish, Southern British and mainstream US English; Chinese, Japanese, Catalan, Italian, Dutch, German, Bulgarian, Lithuanian, Eastern European, Israeli and Latin American speakers; plus generic "European" and "Don't know" labels) |
| Audio | 16 kHz 16-bit mono WAV, active speech at about -20 dBFS, peaks at or below -1 dBFS, 30 ms fades (`--rate 8000` also works) |
| Download | ~890 MB: LibriSpeech test-clean 346 MB, plus 23 EdAcc conversations read with HTTP range requests (542 MB of the 7 GB dataset) |
| Manifest | `callers/human_replay/manifest.csv` (committed, no audio) = `data/human_replay/manifest.csv` |

## Run it

```sh
# 1. Sources (once). EdAcc needs pyarrow, which the shared .venv lacks, so use your own venv:
python3 -m venv /tmp/hr && /tmp/hr/bin/pip install pyarrow numpy soundfile soxr httpx python-dotenv
/tmp/hr/bin/python -m callers.human_replay.build_set download       # ~890 MB into data/human_replay/raw/
#    (LibriSpeech only, no pyarrow: .venv/bin/python -m callers.human_replay.build_set download --sources librispeech)

# 2. Clips + manifest (deterministic: --seed 7)
.venv/bin/python -m callers.human_replay.build_set build
.venv/bin/python -m callers.human_replay.build_set validate

# 3. Calls
.venv/bin/python -m callers.human_replay.place_calls check                  # read-only readiness check
.venv/bin/python -m callers.human_replay.place_calls run                    # DRY RUN: prints the 60 dial payloads
.venv/bin/python -m callers.human_replay.place_calls run --limit 1 --live   # one real call, then listen to caller.wav
.venv/bin/python -m callers.human_replay.place_calls run --live --skip-done  # the rest
```

`--live` uploads any missing clips to `gs://emberz-g1-512122j5/human_replay/`. The bucket stays private:
public access prevention is enforced and objects are deleted after 30 days. It then signs 2-hour V4 URLs as
the `emberz-g1` service account. Then it places one call per clip through SignalWire's Calling API
(`POST /api/calling/calls`, `command: dial`) with this inline SWML:

```json
{"version": "1.0.0", "sections": {"main": [
  {"play": {"url": "silence:2"}}, {"play": {"url": "<signed clip URL>"}}, {"play": {"url": "silence:1"}}, {"hangup": {}}]}}
```

The calls run one at a time. The script waits until SignalWire reports the call ended
(`GET /api/voice/logs/{id}`), then waits 20 s more. If a call is not reported ended in time, it sends
`calling.end` for it. If a dial request fails in transit (timeout), a call may still have been placed, so the
run logs `dial_uncertain` and stops instead of retrying. A run of 60 calls takes about 40 minutes. It stops
after two failed calls in a row, refuses more than `--max-calls` (80), and asks you to type `yes` (or pass `--yes`).

**One-time permission (Numan):** your gcloud account is project owner, but owner does not include
`iam.serviceAccounts.signBlob`. Signing needs this:

```sh
gcloud iam service-accounts add-iam-policy-binding emberz-g1@unified-booster-512122-j5.iam.gserviceaccount.com \
  --project unified-booster-512122-j5 --member=user:info@emberdial.com --role=roles/iam.serviceAccountTokenCreator
```

### Logs and matching to recordings

- `recordings/human_replay_calls.jsonl`: one `{call_id, clip_id, speaker, started_at, source, gender, accent, duration_s, from, to}`
  row when a call is placed, and one `{event: "ended", call_id, clip_id, status}` row when it ends. `call_id` is SignalWire's id.
- `data/callers/calls.jsonl` (the shared ledger) gets `{platform: "signalwire_replay", kind: "human", started_utc, clip_id, speaker, source}`.
  `python -m callers.label_sheet match-sources` then writes `recordings/<line call_id>/source.json` with
  `kind: human`, which V1 uses as the class label. Calls are sequential, so pairing by start time is unambiguous.
- The manifest's `text` column has the reference transcript, so phone-path ASR WER can be measured on these calls for free.

## What the line does on these calls

The line answers, starts the media stream, and plays the Emberz notice to the caller. While the notice
plays, our SWML waits 2 s and then plays the clip. `caller.wav` (the inbound track) holds only the clip,
because the notice is on the outbound track. **The line also rings the protected phone (Numan's cell) on
every call.** Let it ring, or answer it. Don't decline it: a B hangup ends the call and cuts the clip short.
The clip and the hangup come before the line's 30 s ring timeout. Telnyx `deepfake_detection` also runs
on every call, so these 60 calls give the Telnyx baseline's false-alarm rate on human speech.

## Limits (say so in the README and the V1 results)

- **No live reply timing.** A clip plays at a fixed offset after answer and doesn't react to the other side,
  so the reply-gap cue (people ~0.2 s, agents ~1.3-2.4 s) can't be measured on these calls. The 10-20
  live calls from Numan and Matt cover timing (PLAN.md V2).
- **Half is read speech.** LibriSpeech is audiobook reading: fluent, few "um"s, studio or home mics. EdAcc is
  unscripted conversation with fillers, laughter and false starts, but recorded over video calls. The
  partner may be faintly audible, and the audio was already compressed once. So the human class is
  "real people, replayed", not "real people dialing from their own phones".
- **One playback device.** Every call starts from SignalWire's media server, not a handset. There's no handset
  mic, no room acoustics and no mobile codec, so the codec and network conditions are narrower than live calls.
- **Accents:** EdAcc gives 24 accent groups, but each has only 1-2 speakers. LibriSpeech speakers are mostly US.
- These calls aren't scored by the guardian (no risk labels). They feed the human class only.

## Licenses and attribution (paste into the repo README)

No audio is committed. Only the manifest (ids, durations, transcripts) is. Clips are played over phone
calls to our own line. If we ever share the clips themselves, EdAcc-derived clips must stay CC BY-SA 4.0.

> **LibriSpeech** (test-clean) by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur,
> "Librispeech: an ASR corpus based on public domain audio books", ICASSP 2015, https://www.openslr.org/12/,
> licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The audio comes from LibriVox volunteer
> recordings. Changes: we cut 6-12 s clips, resampled them and normalized their level, and played them over phone calls.
>
> **EdAcc: The Edinburgh International Accents of English Corpus** by Ramon Sanabria, Nina Markl, Andrea
> Carmantini, Ondrej Klejch, Peter Bell and Nikolay Bogoychev, University of Edinburgh
> (https://doi.org/10.7488/ds/7914; ICASSP 2023, https://arxiv.org/abs/2303.18110), licensed CC BY-SA 4.0
> (https://creativecommons.org/licenses/by-sa/4.0/), read from the Hugging Face copy `edinburghcstr/edacc`.
> Changes: we cut segments to at most 12 s, resampled them and normalized their level, and played them over phone calls.

Not used: Common Voice (CC0, but "effective October 2025" only through Mozilla Data Collective with an account:
https://huggingface.co/spaces/mozilla-foundation/README), VCTK (CC BY 4.0, but a single ~11 GB zip of 2-4 s sentences),
FLEURS (CC BY 4.0, no speaker ids), VoxPopuli en (CC0, but one 936 MB test parquet), People's Speech
(no speaker metadata).

## Cost (60 calls, each under 1 minute, every leg rounded up to 1 billed minute as a worst case)

| Item | Per call | 60 calls |
|---|---|---|
| SignalWire outbound US: $0.008/min (https://signalwire.com/pricing/voice; the page states no billing increment, so 1 full minute is assumed) | $0.008 | $0.48 |
| Telnyx A leg (inbound): $0.002/min | $0.002 | $0.12 |
| Telnyx media streaming: $0.0035/min | $0.0035 | $0.21 |
| Telnyx `deepfake_detection`: $0.01/call | $0.01 | $0.60 |
| Telnyx B leg (only if the cell answers): $0.002/min | $0.002 | $0.12 |
| Telnyx conference, A + B participants: $0.002/participant-min | $0.004 | $0.24 |
| Cloud Storage: 16 MB for 30 days, ~60 uploads, 16 MB egress | | < $0.01 |
| **Total** | **~$0.03** | **~$1.77, under $2** |

Telnyx rates are PLAN.md section 9's figures (not re-checked here). `max_price_per_minute` is set to $0.05
on every dial, so SignalWire rejects any route that would cost more than that.

## Docs used (checked 2026-11-02)

- Calling API dial with inline `swml`, `caller_id`, `timeout`, `max_price_per_minute`: https://signalwire.com/docs/apis/rest/calls/call-commands
- SWML `play` (`silence:<s>`, HTTP(S) audio URLs) and `hangup`: https://signalwire.com/docs/swml/reference/calling/play, https://signalwire.com/docs/swml/reference/calling/hangup
- Compatibility (LaML) Create a Call: it names an inline `Laml`/`Twiml` parameter but doesn't document it, so we don't use it: https://signalwire.com/docs/compatibility-api/rest/calls/create-a-call
- cXML `<Play>` MIME types (audio/wav): https://signalwire.com/docs/compatibility-api/cxml/reference/voice/play
- Voice logs (call status, read-only): https://signalwire.com/docs/apis/rest/voice-logs/get-voice-log
- Phone numbers list (read-only check): https://signalwire.com/docs/apis/rest/phone-numbers/list-phone-numbers
- V4 signed URLs: https://cloud.google.com/storage/docs/access-control/signed-urls and `gcloud storage sign-url --help` (max 12 h with impersonation)
