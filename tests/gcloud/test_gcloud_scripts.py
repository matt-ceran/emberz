"""Run the G1 shell scripts against a fake `gcloud` on PATH (nothing reaches Google Cloud).

Checks the exact create command (cost stops, spot, metadata that job_config.py accepts),
the code tarball contents, the quota guard, and that teardown deletes only emberz VMs.
"""
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gcloud"))
import job_config as jc  # noqa: E402

FAKE = r'''#!/usr/bin/env python3
import json, os, shutil, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
a = list(args)
if a[:1] == ["--project"]:
    a = a[2:]
gcs = os.environ["FAKE_GCS"]
def local(u):
    return os.path.join(gcs, u[len("gs://"):])
if a[:2] == ["compute", "project-info"]:
    lim = float(os.environ.get("FAKE_CPU_LIMIT", "16"))
    print(json.dumps({"quotas": [{"metric": "CPUS_ALL_REGIONS", "limit": lim, "usage": 4.0},
                                 {"metric": "GPUS_ALL_REGIONS", "limit": 1.0, "usage": 0.0}]}))
elif a[:3] == ["compute", "regions", "describe"]:
    l4 = float(os.environ.get("FAKE_L4_LIMIT", "1"))
    print(json.dumps({"quotas": [{"metric": "CPUS", "limit": 200.0, "usage": 4.0},
                                 {"metric": "N2_CPUS", "limit": 200.0, "usage": 0.0},
                                 {"metric": "SSD_TOTAL_GB", "limit": 300.0, "usage": 100.0},
                                 {"metric": "NVIDIA_L4_GPUS", "limit": l4, "usage": 0.0},
                                 {"metric": "PREEMPTIBLE_NVIDIA_L4_GPUS", "limit": l4, "usage": 0.0}]}))
elif a[:3] == ["storage", "buckets", "describe"] or a[:3] == ["iam", "service-accounts", "describe"]:
    sys.exit(0 if os.environ.get("FAKE_SETUP") == "1" else 1)
elif a[:3] == ["compute", "instances", "list"]:
    flt = " ".join(x for x in a if x.startswith("--filter"))
    if "emberz" in flt:
        for line in os.environ.get("FAKE_VMS", "").split(";"):
            if line:
                print(line.replace(",", "\t"))
elif a[:3] == ["compute", "disks", "list"]:
    pass
elif a[:2] == ["storage", "cp"]:
    src, dst = [x for x in a[2:] if not x.startswith("-")][-2:]
    if dst.startswith("gs://") and os.path.exists(src):
        os.makedirs(os.path.dirname(local(dst)), exist_ok=True)
        shutil.copy(src, local(dst))
elif a[:2] == ["storage", "ls"]:
    sys.exit(1)
'''


@pytest.fixture
def fake(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    g = bin_dir / "gcloud"
    g.write_text(FAKE)
    g.chmod(0o755)
    env = dict(os.environ)
    env.update(PATH=f"{bin_dir}:{env['PATH']}", FAKE_LOG=str(tmp_path / "log.jsonl"),
               FAKE_GCS=str(tmp_path / "gcs"), YES="1", FAKE_SETUP="1", FAKE_CPU_LIMIT="200")
    env.pop("RUN_ID", None)

    def run(script, *args, **extra):
        e = {**env, **extra}
        r = subprocess.run(["bash", str(ROOT / "gcloud" / script), *args], env=e, capture_output=True,
                           text=True, cwd="/", timeout=120)
        calls = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()] \
            if (tmp_path / "log.jsonl").exists() else []
        return r, calls

    run.gcs = tmp_path / "gcs"
    return run


def creates(calls):
    return [c for c in calls if "instances" in c and "create" in c]


def flag(call, name):
    for x in call:
        if x.startswith(f"--{name}="):
            return x.split("=", 1)[1]
    return None


def parse_metadata(value):
    assert value.startswith("^@^")
    return dict(item.split("=", 1) for item in value[3:].split("@"))


def test_launch_noise_command(fake):
    r, calls = fake("launch_noise.sh", str(ROOT / "gcloud/noise_pass/sample_input.jsonl"),
                    RUN_ID="noise-test", NOISE_EXTRA_ARGS="--noise-prob 0.7 --snr-min 10")
    assert r.returncode == 0, r.stderr
    assert "3 calls, 19 turns" in r.stdout
    [c] = creates(calls)
    assert "emberz-noise-test" in c
    assert flag(c, "provisioning-model") == "SPOT"
    assert flag(c, "instance-termination-action") == "DELETE"
    assert flag(c, "max-run-duration") == "90m"
    assert flag(c, "machine-type") == "n2-highcpu-96"
    assert flag(c, "labels") == "emberz=g1,job=noise"
    assert flag(c, "service-account").startswith("emberz-g1@")
    assert flag(c, "metadata-from-file").endswith("gcloud/vm_startup.sh")
    meta = parse_metadata(flag(c, "metadata"))
    env = jc.parse(meta)  # the VM's own validator accepts what the launcher sends
    assert env["EMBERZ_RUN_ID"] == "noise-test"
    assert env["EMBERZ_EXTRA_ARGS"] == "--noise-prob 0.7 --snr-min 10"
    assert env["EMBERZ_MODEL_SHA256"].startswith("71111f61")
    run_dir = fake.gcs / "emberz-g1-512122j5" / "runs" / "noise-test"
    assert (run_dir / "noise_input.jsonl").read_text() == (ROOT / "gcloud/noise_pass/sample_input.jsonl").read_text()
    names = set(tarfile.open(run_dir / "code.tgz").getnames())
    for need in ["contracts/models.py", "gcloud/job_config.py", "gcloud/vm_startup.sh",
                 "gcloud/noise_pass/run_noise.py", "gcloud/noise_pass/noise_lib.py",
                 "gcloud/noise_pass/vm_run.sh", "gcloud/noise_pass/requirements-vm.lock.txt"]:
        assert need in names
    assert not any("__pycache__" in n for n in names)


def test_launch_noise_refuses_without_quota(fake):
    r, calls = fake("launch_noise.sh", str(ROOT / "gcloud/noise_pass/sample_input.jsonl"), FAKE_CPU_LIMIT="16")
    assert r.returncode != 0
    assert "CPUS_ALL_REGIONS has only 12 free" in r.stderr
    assert not creates(calls)
    assert not any(c[2:4] == ["storage", "cp"] for c in calls)  # nothing uploaded either


def test_launch_noise_refuses_without_setup(fake):
    r, calls = fake("launch_noise.sh", str(ROOT / "gcloud/noise_pass/sample_input.jsonl"), FAKE_SETUP="0")
    assert r.returncode != 0 and "setup_bucket.sh" in r.stderr
    assert not creates(calls)


def test_launch_noise_refuses_bad_input(fake, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"call_id": "x"}\n')
    r, calls = fake("launch_noise.sh", str(bad))
    assert r.returncode != 0 and not creates(calls)


def test_launch_voice_command(fake):
    if not (ROOT / "voice" / "pipeline.py").exists():
        pytest.skip("voice/pipeline.py not written yet")
    r, calls = fake("launch_voice.sh", RUN_ID="voice-test", VOICE_STEPS="download,manifests,features,head",
                    VOICE_ARGS="--skip-itw")
    assert r.returncode == 0, r.stderr
    [c] = creates(calls)
    assert flag(c, "machine-type") == "g2-standard-8"
    assert flag(c, "maintenance-policy") == "TERMINATE"
    assert flag(c, "instance-termination-action") == "DELETE"
    assert flag(c, "max-run-duration") == "4h"
    assert flag(c, "image-family").startswith("pytorch-") and flag(c, "image-project") == "deeplearning-platform-release"
    assert flag(c, "boot-disk-size") == "150GB"
    meta = parse_metadata(flag(c, "metadata"))
    env = jc.parse(meta)
    assert env["EMBERZ_VOICE_STEPS"] == "download,manifests,features,head"
    assert env["EMBERZ_VOICE_ARGS"] == "--skip-itw"
    names = set(tarfile.open(fake.gcs / "emberz-g1-512122j5/runs/voice-test/code.tgz").getnames())
    assert {"voice/pipeline.py", "voice/features.py", "voice/requirements-gcp.txt",
            "gcloud/voice_job/vm_run.sh"} <= names


def test_teardown_deletes_only_listed_emberz_vms(fake):
    r, calls = fake("teardown.sh", FAKE_VMS="emberz-noise-1,us-central1-a;emberz-voice-2,us-central1-b")
    assert r.returncode == 0, r.stderr
    deletes = [c for c in calls if "instances" in c and "delete" in c]
    assert [(c[c.index("delete") + 1], c[c.index("--zone") + 1]) for c in deletes] == [
        ("emberz-noise-1", "us-central1-a"), ("emberz-voice-2", "us-central1-b")]
    lists = [c for c in calls if c[2:5] == ["compute", "instances", "list"] and any("--filter" in x for x in c)]
    assert all("name~^emberz- OR labels.emberz=g1" in " ".join(c) for c in lists)
    assert lists
    assert not any(c[2:4] == ["storage", "rm"] for c in calls)  # bucket kept without --bucket


def test_launch_noise_refuses_max_run_too_short_for_small_machine(fake, tmp_path):
    # ~500 calls of ~120 s of speech on n2-standard-8 needs hours, not the default 90 min.
    rows = [l for l in (ROOT / "gcloud/noise_pass/sample_input.jsonl").read_text().splitlines() if l]
    big = tmp_path / "big.jsonl"
    with big.open("w") as f:
        for i in range(500):
            d = json.loads(rows[i % len(rows)])
            d["call_id"] = f"c{i}"
            for t in d["turns"]:
                t["text"] = (t["text"] + " ") * 4
            f.write(json.dumps(d) + "\n")
    r, calls = fake("launch_noise.sh", str(big), NOISE_MACHINE_TYPE="n2-standard-8")
    assert r.returncode != 0 and "NOISE_MAX_RUN" in r.stderr, r.stderr + r.stdout
    assert not creates(calls)
    r, calls = fake("launch_noise.sh", str(big), NOISE_MACHINE_TYPE="n2-standard-8", NOISE_MAX_RUN="1d")
    assert r.returncode == 0, r.stderr
    assert flag(creates(calls)[0], "max-run-duration") == "1d"


def test_launch_noise_rejects_bad_duration(fake):
    r, calls = fake("launch_noise.sh", str(ROOT / "gcloud/noise_pass/sample_input.jsonl"), NOISE_MAX_RUN="90 minutes")
    assert r.returncode != 0 and not creates(calls)


def test_launch_voice_refuses_without_regional_l4(fake):
    if not (ROOT / "voice" / "pipeline.py").exists():
        pytest.skip("voice/pipeline.py not written yet")
    r, calls = fake("launch_voice.sh", FAKE_L4_LIMIT="0")
    assert r.returncode != 0 and "L4" in r.stderr
    assert not creates(calls)


@pytest.mark.parametrize("d,hours", [("90m", 1.5), ("4h", 4.0), ("1h30m", 1.5), ("1d2h", 26.0), ("45s", 0.0125)])
def test_dur_hours(d, hours):
    out = subprocess.run(["bash", "-c", f'source "{ROOT}/gcloud/config.sh"; source "{ROOT}/gcloud/launch_common.sh"; dur_hours {d}'],
                         capture_output=True, text=True, check=True).stdout
    assert abs(float(out) - hours) < 1e-9


@pytest.mark.parametrize("run_id", ["Noise_1", "-x", "a" * 41])
def test_launch_noise_rejects_bad_run_id_before_upload(fake, run_id):
    r, calls = fake("launch_noise.sh", str(ROOT / "gcloud/noise_pass/sample_input.jsonl"), RUN_ID=run_id)
    assert r.returncode != 0 and "RUN_ID" in r.stderr
    assert not creates(calls)
    assert not any(c[2:4] == ["storage", "cp"] for c in calls)
