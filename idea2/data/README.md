# Data

Built by `data/prepare.py` (run `.venv/bin/python data/prepare.py` from `idea2/`). Rows are
`{"id", "text", "label"}` with `label` the human-readable intent name; `labels.json` holds the
label list in the dataset's own order (this order is the order used in the prompt).

| Task | Source (Hugging Face) | Revision | Licence | train | val | test | labels |
|---|---|---|---|---|---|---|---|
| banking77 | [`PolyAI/banking77`](https://huggingface.co/datasets/PolyAI/banking77) | `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6` | CC BY 4.0 | 9,000 | 1,003 | 3,080 | 77 |
| clinc150 | [`clinc/clinc_oos`](https://huggingface.co/datasets/clinc/clinc_oos), config `plus` | `155b9c710419136e17307b80d0a13e68cd46b4ec` | CC BY 3.0 | 15,250 | 3,100 | 5,500 | 151 (150 + `oos`) |

Papers: Casanueva et al. 2020, "Efficient Intent Detection with Dual Sentence Encoders"
(Banking77); Larson et al. 2019, "An Evaluation Dataset for Intent Classification and
Out-of-Scope Prediction" (CLINC150).

## Splits

- **banking77**: there is no official validation split. `val` is 10% of the official train
  split, stratified by label, seed 0 (min 1 per label); `train` is the other 90%. `test` is the
  official test split, unchanged.
- **clinc150**: official `train` / `validation` / `test` of the `plus` config, unchanged
  (`plus` adds 250 extra out-of-scope training examples). `oos` is the out-of-scope label.

`test` is for final reporting only. `eval/run_eval.py` refuses `--split test` unless
`ALLOW_TEST=1` is set. All selection, prompt work and weight editing use `train` / `val`.

## Quirks

- Banking77 train is imbalanced (val has 4-19 rows per label); the official test is balanced
  (40 per label). So val accuracy weights labels differently from test accuracy.
- CLINC150 val has 20 per in-scope label and 100 `oos`; test has 30 per in-scope label and
  1,000 `oos` (18% of test), so out-of-scope handling matters much more on test.
- Near-duplicates across splits (case/whitespace-normalised exact text): banking77 test∩train 6,
  test∩val 1, 1 duplicate inside test; clinc150 test∩train 2, val∩train 3. Left in place
  (official splits); negligible.
- Banking77 is known to have some label noise and overlapping intents (e.g.
  `card_payment_not_recognised` vs `direct_debit_payment_not_recognised`,
  `failed_transfer` vs `beneficiary_not_allowed`), which caps achievable accuracy.
