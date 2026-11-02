# V1 AI-voice detector: In-the-Wild, unseen speakers, phone path

**Result:** on 2,829 clips from 25 public figures the head never saw in training, after the
simulated phone path (8 kHz G.711 mu-law), with the first 4 s of speech:
**EER 1.27% (95% CI 0.85-1.77%), ROC AUC 0.9991 (0.9985-0.9996)**. At a 1% false-alarm rate
it catches 98.0% of AI clips; at 5% it catches 99.8%.

![ROC](voice_roc.svg)

| Test set (model `w2v-head-v1`) | n | AI / human | EER [95% CI] | AUC [95% CI] | Acc @ P(AI)=0.5 |
|---|---|---|---|---|---|
| `itw_phone_heldout_speakers`: phone path, first 4 s of speech (**headline**) | 2,829 | 1,413 / 1,416 | **1.27%** [0.85, 1.77] | **0.9991** [0.9985, 0.9996] | 98.2% |
| `itw_phone_heldout_speakers/w2s`: phone path, first **2 s** (the live early check) | 2,829 | 1,413 / 1,416 | 1.55% [1.06, 2.05] | 0.9988 [0.9979, 0.9994] | 98.4% |
| `itw_clean_heldout_speakers`: same head, test audio **without** the phone path | 2,829 | 1,413 / 1,416 | 0.92% [0.56, 1.27] | 0.9995 [0.9991, 0.9998] | 98.2% |

CIs are a class-stratified bootstrap over clips (1,000 resamples). Clips from one speaker are
correlated, so we also resampled whole speakers: headline EER CI 0.65-1.71%, AUC 0.9986-0.9998
(`voice_summary.json`). The two CIs are close.

Reading: the phone path costs about 0.35 points of EER compared with clean audio (the CIs
overlap, so the difference is not significant at this n). Scoring at 2 s instead of 4 s costs
about 0.3 points. Both are small, so an early flag at 2 s of speech is viable.

## Method

- **Data:** In-the-Wild (Mueller et al. 2022, arXiv:2203.16263), 31,779 clips, 58 public
  figures, `release_in_the_wild/meta.csv`. We used a balanced subset of 5,998 clips (2,996 AI, 3,002 human),
  sampled in proportion within each speaker so no speaker dominates.
- **Split by speaker, not by clip** (stable hash of the name, `voice.manifest.speaker_split`):
  25 test speakers (2,829 clips), 24 train speakers (2,909 clips), 5 dev speakers (260 clips) used
  only to pick the layer and C. No test speaker appears in train or dev.
- **Phone path:** every train and test clip goes 16 kHz -> 8 kHz -> G.711 mu-law -> 16 kHz
  (`voice.phone.phone_degrade`), the same code and resampler the live line uses on Telnyx audio.
- **Speech gate:** a frame-energy gate (`voice.features.SpeechGate`, shared with live) drops
  silence, then the first 2 s and 4 s of speech are kept. This removes the known ASVspoof
  "leading silence" shortcut.
- **Features:** frozen `facebook/wav2vec2-xls-r-300m`, hidden states of layers 3/6/9/12, mean +
  std pooled (2,048 dims). Extracted on Apple MPS at about 15 clips/s (6k clips in about 7 min).
- **Head:** standardise -> logistic regression -> Platt calibration with balanced weights
  (`voice.train_head`), trained on 2 s and 4 s windows stacked. Dev picked **layer 9, C = 1.0**
  (dev EER 5.5%; the other settings were 5.5-8.0%).
- **Live:** the head is saved to `voice/heads/w2v-head-v1.joblib`, the default path of
  `voice.live.LiveVoiceDetector`. Smoke test: 6 test clips sent as 20 ms mu-law frames through the
  live detector were all scored correctly (human <= 0.006, AI >= 0.97). Scoring one 4 s clip takes
  33 ms on MPS (p50, with another job sharing the GPU).

## Honest caveats

1. **In-domain, not out-of-domain.** The head was trained on In-the-Wild speakers and tested on
   other In-the-Wild speakers. The literature numbers for frozen XLS-R + logistic regression
   trained on ASVspoof and tested on ITW are far worse (Pascu et al. 2024: about 16% mean EER
   across sets for xls-r-300m). Our 1.3% says the method works when the training data looks like
   the test data. It does not mean the detector catches any AI voice.
2. **Celebrity speech from online video, not phone calls.** ITW is speeches, interviews and
   podcasts. The phone path is simulated, and real calls add packet loss, AGC, noise suppression and
   codec switching that we did not model.
3. **Old TTS.** ITW's fakes were collected from the web before 2022. Today's voice agents
   (ElevenLabs, OpenAI realtime, Cartesia, Sesame and similar) are newer and more natural, and none
   are in this set. Expect a real drop on them. The V2 live call set is the test that matters.
4. **Class-correlated recording conditions.** Within ITW, real and fake clips of one person often
   come from different sources (channel, mic, mixing). A detector can learn source traits instead of
   synthesis traits. Splitting by speaker does not rule this out. Clip duration alone separates the
   classes only weakly (logistic regression on duration: test AUC 0.59), so length is not the
   shortcut, but channel could be.
5. **Hardest speakers** (accuracy at 0.5): Dwight Eisenhower 89% (n=47), John Cleese 93% (27),
   Lyndon Johnson 94% (17), Alexandria Ocasio-Cortez 94% (69), FDR 94% (87). Most are old
   archival recordings; a plausible reason is that a heavily degraded archive channel looks like synthesis artefacts (not verified).
6. **Subset.** 6k of the 31.8k clips, for time. ASVspoof 5 was still downloading and is not used.

## Reproduce

```bash
PY=.venv/bin/python
$PY -m voice.itw_fast manifest --dest $PWD/data/voice/itw --out data/voice/feats_fast/itw_subset.csv --per-label 3000 --dev-frac 0.3
D=data/voice/feats_fast; head -1 $D/itw_subset.csv > $D/itw_subset_test.csv; grep ',test,' $D/itw_subset.csv >> $D/itw_subset_test.csv
$PY -m voice.features extract --manifest $D/itw_subset.csv --out-dir $D/phone --layers 3,6,9,12 --windows 2,4 --shard-size 1000
$PY -m voice.features extract --manifest $D/itw_subset_test.csv --out-dir $D/clean_test --no-phone --layers 3,6,9,12 --windows 2,4 --shard-size 1000
$PY -m voice.itw_report --feats $D     # trains the head, writes results/voice_* and voice/heads/w2v-head-v1.joblib
```

Files: `voice_results.jsonl` (ResultsRow), `voice_roc.json` (fpr/tpr/threshold per condition),
`voice_roc.svg` (chart, light and dark), `voice_summary.json` (all metrics, per-speaker accuracy,
head metadata and the dev search).
