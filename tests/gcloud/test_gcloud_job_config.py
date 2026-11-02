"""G1 startup metadata parsing (gcloud/job_config.py) and its shell output."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gcloud"))

import job_config as jc  # noqa: E402

NOISE = {
    "emberz-job": "noise",
    "emberz-bucket": "gs://emberz-g1-512122j5",
    "emberz-run-id": "noise-1102-1400",
    "emberz-model-url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/x.tar.bz2",
    "emberz-model-dir": "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25",
    "emberz-model-sha256": "71111f61b18e1e65e01e369434a5c0434868d2f44892742ae54240600c681209",
}
VOICE = {"emberz-job": "voice", "emberz-bucket": "gs://emberz-g1-512122j5", "emberz-run-id": "voice-1"}


def test_noise_defaults():
    env = jc.parse(NOISE)
    assert env["EMBERZ_JOB"] == "noise"
    assert env["EMBERZ_THREADS"] == "1"
    assert env["EMBERZ_RENDER_WORKERS"] == "0" and env["EMBERZ_ASR_WORKERS"] == "0"
    assert env["EMBERZ_SELF_DELETE"] == "1"
    assert env["EMBERZ_EXTRA_ARGS"] == ""


def test_voice_defaults_and_steps():
    env = jc.parse(VOICE)
    assert env["EMBERZ_VOICE_STEPS"] == "download,manifests,features"
    env = jc.parse({**VOICE, "emberz-voice-steps": "features,head,eval", "emberz-voice-args": "--skip-itw --num-workers 8"})
    assert env["EMBERZ_VOICE_STEPS"] == "features,head,eval"


def test_overrides_are_kept():
    env = jc.parse({**NOISE, "emberz-threads": "2", "emberz-asr-workers": "40",
                    "emberz-extra-args": "--noise-prob 0.7 --loss-prob 0.01", "emberz-self-delete": "0"})
    assert env["EMBERZ_THREADS"] == "2" and env["EMBERZ_ASR_WORKERS"] == "40"
    assert env["EMBERZ_EXTRA_ARGS"] == "--noise-prob 0.7 --loss-prob 0.01"
    assert env["EMBERZ_SELF_DELETE"] == "0"


@pytest.mark.parametrize(
    "patch,msg",
    [
        ({"emberz-job": "train"}, "emberz-job"),
        ({"emberz-bucket": "emberz-g1"}, "emberz-bucket"),
        ({"emberz-bucket": "gs://Bad_Bucket"}, "emberz-bucket"),
        ({"emberz-run-id": "Noise Run"}, "emberz-run-id"),
        ({"emberz-model-sha256": "abc"}, "emberz-model-sha256"),
        ({"emberz-model-url": "http://insecure"}, "emberz-model-url"),
        ({"emberz-threads": "0"}, "emberz-threads"),
        ({"emberz-asr-workers": "-1"}, "emberz-asr-workers"),
        ({"emberz-self-delete": "yes"}, "emberz-self-delete"),
        ({"emberz-extra-args": "--seed 1; rm -rf /"}, "emberz-extra-args"),
        ({"emberz-extra-args": "--seed $(id)"}, "emberz-extra-args"),
        ({"emberz-extra-args": "--seed 'unterminated"}, "emberz-extra-args"),
        ({"emberz-treads": "2"}, "unknown"),
        ({"emberz-voice-steps": "download"}, "unknown"),  # voice key on a noise job
    ],
)
def test_bad_values_fail_loudly(patch, msg):
    with pytest.raises(jc.ConfigError, match=msg):
        jc.parse({**NOISE, **patch})


def test_missing_required_key():
    attrs = dict(NOISE)
    del attrs["emberz-model-sha256"]
    with pytest.raises(jc.ConfigError, match="missing required metadata emberz-model-sha256"):
        jc.parse(attrs)


def test_bad_voice_steps():
    with pytest.raises(jc.ConfigError):
        jc.parse({**VOICE, "emberz-voice-steps": "download,train"})


def test_non_emberz_metadata_is_ignored():
    env = jc.parse({**VOICE, "install-nvidia-driver": "True", "startup-script": "#!/bin/bash"})
    assert "EMBERZ_JOB" in env


def test_shell_output_evaluates_to_the_same_values(tmp_path):
    attrs = {**NOISE, "emberz-extra-args": "--noise-prob 0.7 --snr-min 10"}
    f = tmp_path / "attrs.json"
    f.write_text(json.dumps(attrs))
    out = subprocess.run([sys.executable, str(ROOT / "gcloud" / "job_config.py"), "--from-json", str(f)],
                         capture_output=True, text=True, check=True).stdout
    shown = subprocess.run(["bash", "-c", out + 'printf "%s|%s|%s" "$EMBERZ_JOB" "$EMBERZ_EXTRA_ARGS" "$EMBERZ_RUN_ID"'],
                           capture_output=True, text=True, check=True).stdout
    assert shown == "noise|--noise-prob 0.7 --snr-min 10|noise-1102-1400"


def test_cli_exit_code_on_bad_config(tmp_path):
    f = tmp_path / "attrs.json"
    f.write_text(json.dumps({**NOISE, "emberz-bucket": "nope"}))
    r = subprocess.run([sys.executable, str(ROOT / "gcloud" / "job_config.py"), "--from-json", str(f)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "emberz-bucket" in r.stderr and r.stdout == ""
