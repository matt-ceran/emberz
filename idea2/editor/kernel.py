"""The editor's persistent Python kernel (runs ON THE VM). A Jupyter kernel in systemd unit `idea2-kernel`
(user idea2, CPU only: CUDA_VISIBLE_DEVICES is empty, MemoryMax from the unit); variables survive between runs,
like the paper's notebook kernel. kernel_prelude.py is loaded into it once per kernel start.

  python kernel.py run [file.py | -]     run code (default: stdin); prints its output
  python kernel.py start | stop | restart | status

Limits (from /opt/idea2/editor.env): EDITOR_RUN_TIMEOUT seconds per run (default 900; on timeout the run is
interrupted, and the kernel restarted if that fails), EDITOR_MAX_RUNS runs in total (default 300), output cut at
10,000 characters, shown in ASCII. Every run (code + output) is appended to /opt/idea2/work/log.jsonl.
"""
import fcntl
import getpass
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENV_FILE = Path("/opt/idea2/editor.env")
CONF = {}
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            CONF[k.strip()] = v.strip()
WORK = Path(CONF.get("IDEA2_WORK", "/opt/idea2/work"))
KDIR = WORK / "kernel"
CONN = KDIR / "conn.json"
PRELUDE = Path(__file__).with_name("kernel_prelude.py")
TIMEOUT = int(CONF.get("EDITOR_RUN_TIMEOUT", "900"))
MAX_RUNS = int(CONF.get("EDITOR_MAX_RUNS", "300"))
MAX_OUT = 10_000
UNIT = "idea2-kernel"
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def sh(*a):
    return subprocess.run(list(a), capture_output=True, text=True)


def active():
    return sh("systemctl", "is-active", UNIT).stdout.strip() == "active"


def start():
    if active() and CONN.exists():
        return False
    sh("sudo", "-n", "systemctl", "start", UNIT)
    for _ in range(60):
        if CONN.exists() and active():
            time.sleep(1.0)
            return True
        time.sleep(0.5)
    raise SystemExit(f"kernel did not start; see: sudo journalctl -u {UNIT} -n 50")


def stop():
    sh("sudo", "-n", "systemctl", "stop", UNIT)


def client():
    from jupyter_client import BlockingKernelClient
    # ipykernel writes its connection file 0600 as user idea2; read it with sudo (every ssh user has sudo)
    info = json.loads(subprocess.run(["sudo", "-n", "cat", str(CONN)], capture_output=True, text=True,
                                     check=True).stdout)
    info.pop("kernel_name", None)
    c = BlockingKernelClient()
    c.load_connection_info(info)
    c.start_channels()
    c.wait_for_ready(timeout=60)
    return c


def execute(c, code, timeout):
    """Run code; return (text, status) with status ok | error | timeout | died."""
    msg_id = c.execute(code, store_history=False, allow_stdin=False)
    parts, status = [], "ok"
    t_end = time.time() + timeout
    while True:
        left = t_end - time.time()
        if left <= 0:
            return "".join(parts), "timeout"
        try:
            msg = c.get_iopub_msg(timeout=min(5.0, left))
        except Exception:  # noqa: BLE001  (queue.Empty)
            if not active():
                return "".join(parts), "died"
            continue
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        t, ct = msg["msg_type"], msg["content"]
        if t == "stream":
            parts.append(ct["text"])
        elif t in ("execute_result", "display_data"):
            parts.append(ct["data"].get("text/plain", "") + "\n")
        elif t == "error":
            parts.append("\n".join(ANSI.sub("", x) for x in ct["traceback"]) + "\n")
            status = "error"
        elif t == "status" and ct["execution_state"] == "idle":
            return "".join(parts), status


def log(entry):
    entry = {"t": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "user": getpass.getuser(),
             **entry}
    with open(WORK / "log.jsonl", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def counter(bump=False):
    p = KDIR / "runs.json"
    n = json.loads(p.read_text())["runs"] if p.exists() else 0
    if bump:
        n += 1
        p.write_text(json.dumps({"runs": n, "max": MAX_RUNS}))
    return n


def run(code):
    os.umask(0)
    KDIR.mkdir(parents=True, exist_ok=True)
    n = counter()
    if n >= MAX_RUNS:
        print(f"[refused: all {MAX_RUNS} code runs are used]")
        return 1
    notes = []
    if start():
        notes.append("[the kernel was (re)started: variables from earlier runs are gone]")
    c = client()
    try:
        boot, st = execute(c, "try:\n    _IDEA2_PRELUDE\nexcept NameError:\n"
                              f"    exec(open({str(PRELUDE)!r}).read())\n    print('__PRELUDE_LOADED__')", 300)
        if st != "ok":
            print("[prelude failed]\n" + boot)
            return 1
        if "__PRELUDE_LOADED__" in boot and not notes:
            notes.append("[fresh kernel: prelude loaded, no earlier variables]")
        t0 = time.time()
        out, st = execute(c, code, TIMEOUT)
        secs = time.time() - t0
    finally:
        c.stop_channels()
    if st == "timeout":
        sh("sudo", "-n", "systemctl", "kill", "-s", "SIGINT", UNIT)
        c = client()
        _, st2 = execute(c, "None", 20)
        c.stop_channels()
        if st2 == "ok":
            out += f"\n[timed out after {TIMEOUT} s: interrupted; the kernel and its variables are kept]"
        else:
            sh("sudo", "-n", "systemctl", "restart", UNIT)
            out += f"\n[timed out after {TIMEOUT} s: the kernel was restarted, variables are lost]"
    elif st == "died":
        out += ("\n[the kernel died (probably out of memory: the limit is the MemoryMax of idea2-kernel); "
                "it restarts on the next run and variables are lost]")
    k = counter(bump=True)
    text = "\n".join(notes + [out]) if notes else out
    text = text.encode("ascii", "replace").decode()
    if len(text) > MAX_OUT:
        text = text[:MAX_OUT] + f"\n[output cut at {MAX_OUT:,} characters]"
    log({"kind": "python", "run": k, "status": st, "s": round(secs, 2), "code": code, "out": text[:6000]})
    print(text if text.strip() else "(no output)")
    print(f"[run {k}/{MAX_RUNS}, {secs:.1f} s]")
    return 0


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        src = sys.argv[2] if len(sys.argv) > 2 else "-"
        code = sys.stdin.read() if src == "-" else open(src).read()
        sys.exit(run(code))
    if cmd == "start":
        print("started" if start() else "already running")
    elif cmd == "stop":
        stop()
        print("stopped")
    elif cmd == "restart":
        stop()
        start()
        print("restarted (variables are gone)")
    elif cmd == "status":
        print(json.dumps({"active": active(), "runs": counter(), "max_runs": MAX_RUNS, "timeout_s": TIMEOUT}))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
