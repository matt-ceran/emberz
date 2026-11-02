#!/bin/zsh
# Score systems on the same 50 held-out calls: clean text vs Nemotron-ASR text.
set -u
cd /Users/mattceran/Documents/fireworksamd
S=/private/tmp/claude-501/-Users-mattceran-Documents-fireworksamd/db79b7b9-1daf-471f-bbea-48fddd2891bb/scratchpad/venvs/sherpa/bin/python
V=callguard/.venv/bin/python
V1=tapi:afceran33-zaqel7a5ap/run-4558a95f6aa345deb458372936dd9a13/v1-fin-35494466
V2=tapi:afceran33-zaqel7a5ap/run-d0f99835415b46f5a898cafaa544e005/v2-fin-7ecb9b48
$S callguard/noise/asr_slice.py --n 50 --workers 6 || exit 1
C=callguard/data/calls_test_clean50.jsonl; A=callguard/data/calls_test_asr50.jsonl
# clean 50: reuse cached held-out predictions (no new model calls)
for m in guardian-v1:$V1 guardian-v2:$V2; do n=${m%%:*}; mod=${m#*:}
  $V -I callguard/eval/run_eval.py --model "$mod" --name $n --test $C --test-set clean50 --preds-tag heldout --key-env FIREWORKS_API_KEY_MATT; done
python3 -I callguard/eval/run_eval_rune.py --name b1-rune --test $C --test-set clean50 --preds-tag heldout
python3 -I callguard/eval/run_eval.py --model deepseek --name b5-deepseek-v4p1-flash --test $C --test-set clean50 --preds-tag heldout
# ASR 50: fresh scoring on transcribed text
python3 -I callguard/eval/run_eval_rune.py --name b1-rune --test $A --test-set asr50 --workers 12 &
python3 -I callguard/eval/run_eval.py --model deepseek --name b5-deepseek-v4p1-flash --test $A --test-set asr50 --workers 12 &
$V -I callguard/eval/run_eval.py --model "$V1" --name guardian-v1 --test $A --test-set asr50 --key-env FIREWORKS_API_KEY_MATT --workers 8 &
$V -I callguard/eval/run_eval.py --model "$V2" --name guardian-v2 --test $A --test-set asr50 --key-env FIREWORKS_API_KEY_MATT --workers 8 &
wait
echo ALL_DONE
