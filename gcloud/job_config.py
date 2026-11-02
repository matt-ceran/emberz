"""Read and validate the emberz-* instance metadata a G1 VM was launched with.

The launch scripts pass job settings as instance metadata (key=value). The startup script
runs `python3 gcloud/job_config.py --from-metadata` and evals the `export` lines it prints,
so a typo fails loudly at boot instead of halfway through a paid run.
Standard library only: it runs on a bare Debian image before any venv exists.

Metadata server: https://cloud.google.com/compute/docs/metadata/querying-metadata
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import urllib.request

METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/"

_BUCKET = re.compile(r"^gs://[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STEPS = {"download", "manifests", "features", "head", "eval", "aasist"}

# metadata key -> (env name, default or None if required, kind)
COMMON = {
    "emberz-job": ("EMBERZ_JOB", None, "job"),
    "emberz-bucket": ("EMBERZ_BUCKET", None, "bucket"),
    "emberz-run-id": ("EMBERZ_RUN_ID", None, "run_id"),
    "emberz-self-delete": ("EMBERZ_SELF_DELETE", "1", "bool"),
}
NOISE = {
    "emberz-model-url": ("EMBERZ_MODEL_URL", None, "url"),
    "emberz-model-dir": ("EMBERZ_MODEL_DIR", None, "name"),
    "emberz-model-sha256": ("EMBERZ_MODEL_SHA256", None, "hex64"),
    "emberz-threads": ("EMBERZ_THREADS", "1", "int"),
    "emberz-render-workers": ("EMBERZ_RENDER_WORKERS", "0", "int0"),
    "emberz-asr-workers": ("EMBERZ_ASR_WORKERS", "0", "int0"),
    "emberz-extra-args": ("EMBERZ_EXTRA_ARGS", "", "args"),
}
VOICE = {
    "emberz-voice-steps": ("EMBERZ_VOICE_STEPS", "download,manifests,features", "steps"),
    "emberz-voice-args": ("EMBERZ_VOICE_ARGS", "", "args"),
}


class ConfigError(ValueError):
    pass


def _check(key: str, value: str, kind: str) -> str:
    v = value.strip()
    ok = {
        "job": lambda: v in ("noise", "voice"),
        "bucket": lambda: bool(_BUCKET.match(v)),
        "run_id": lambda: bool(_RUN_ID.match(v)),
        "bool": lambda: v in ("0", "1"),
        "url": lambda: v.startswith("https://"),
        "name": lambda: bool(re.match(r"^[A-Za-z0-9._-]+$", v)),
        "hex64": lambda: bool(_HEX64.match(v)),
        "int": lambda: v.isdigit() and int(v) >= 1,
        "int0": lambda: v.isdigit(),
        "steps": lambda: bool(v) and set(v.split(",")) <= _STEPS,
        "args": lambda: _args_ok(v),
    }[kind]()
    if not ok:
        raise ConfigError(f"bad metadata {key}={value!r} (expected {kind})")
    return v


def _args_ok(v: str) -> bool:
    """Extra CLI args: must split cleanly and contain no shell syntax."""
    try:
        parts = shlex.split(v)
    except ValueError:
        return False
    return not any(c in v for c in ";|&`$<>\n") and all(p.startswith("--") or not p.startswith("-") for p in parts)


def parse(attrs: dict[str, str]) -> dict[str, str]:
    """Metadata attributes -> validated env dict. Unknown emberz-* keys are an error (typos)."""
    job = attrs.get("emberz-job")
    if job not in ("noise", "voice"):
        raise ConfigError(f"emberz-job must be noise or voice, got {job!r}")
    spec = {**COMMON, **(NOISE if job == "noise" else VOICE)}
    unknown = sorted(k for k in attrs if k.startswith("emberz-") and k not in spec)
    if unknown:
        raise ConfigError(f"unknown metadata keys for job {job}: {unknown}")
    env = {}
    for key, (name, default, kind) in spec.items():
        if key in attrs:
            env[name] = _check(key, attrs[key], kind)
        elif default is None:
            raise ConfigError(f"missing required metadata {key}")
        else:
            env[name] = default
    return env


def to_shell(env: dict[str, str]) -> str:
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())) + "\n"


def _get(path: str) -> str:
    req = urllib.request.Request(METADATA_URL + path, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.read().decode()


def fetch_metadata() -> tuple[dict[str, str], dict[str, str]]:
    """(emberz-* attributes, instance facts) from the metadata server."""
    attrs = json.loads(_get("attributes/?recursive=true"))
    facts = {"EMBERZ_VM_NAME": _get("name"), "EMBERZ_VM_ZONE": _get("zone").rsplit("/", 1)[-1]}
    return {k: v for k, v in attrs.items() if k.startswith("emberz-")}, facts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-metadata", action="store_true")
    g.add_argument("--from-json", help="a JSON file of attributes (tests, dry runs)")
    a = ap.parse_args(argv)
    try:
        if a.from_metadata:
            attrs, facts = fetch_metadata()
        else:
            with open(a.from_json) as f:
                attrs, facts = json.load(f), {}
        env = parse(attrs)
    except (ConfigError, OSError, ValueError) as e:
        print(f"job_config: {e}", file=sys.stderr)
        return 2
    env.update(facts)
    sys.stdout.write(to_shell(env))
    return 0


if __name__ == "__main__":
    sys.exit(main())
