"""CLI for the forward-pass server (runs ON THE VM; the Mac calls it through ../editor/vm.sh fwd).

  python fwd.py '{"type": "generate", "model": "both", "split": "select", "n": 128}'   # one pass
  python fwd.py - < request.json                                                        # JSON on stdin
  python fwd.py budget | health | reload

The JSON's "type" picks the endpoint: "generate" -> POST /generate, "read" -> POST /read. Prints the server's
JSON reply (summaries only; full outputs are in the files it names).
"""
import json
import os
import sys
import urllib.error
import urllib.request

URL = os.environ.get("EDITOR_URL", "http://127.0.0.1:8100")


def call(method, path, body=None, timeout=7200):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001
            return e.code, {"error": str(e)}
    except urllib.error.URLError as e:
        return 0, {"error": f"forward-pass server not reachable at {URL} ({e.reason}); is the VM in editor mode? "
                            f"(mode.sh status)"}


def compact(x, ind=0):
    """JSON with one line per dict whose values are all scalars (short enough to read on a phone)."""
    pad = " " * ind
    if isinstance(x, dict) and x:
        flat = all(not isinstance(v, dict) and not (isinstance(v, list) and any(isinstance(y, (dict, list)) for y in v))
                   for v in x.values())
        if flat and len(json.dumps(x)) < 160:
            return json.dumps(x, ensure_ascii=True)
        inner = [f'{pad} {json.dumps(k)}: {compact(v, ind + 1)}' for k, v in x.items()]
        return "{\n" + ",\n".join(inner) + "\n" + pad + "}"
    if isinstance(x, list) and x and any(isinstance(v, (dict, list)) for v in x):
        return "[\n" + ",\n".join(pad + " " + compact(v, ind + 1) for v in x) + "\n" + pad + "]"
    return json.dumps(x, ensure_ascii=True)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    a = sys.argv[1]
    if a in ("budget", "health"):
        code, out = call("GET", "/" + a, timeout=30)
    elif a == "reload":
        code, out = call("POST", "/reload_edits", {})
    else:
        req = json.loads(sys.stdin.read() if a == "-" else a)
        kind = req.pop("type", None)
        if kind not in ("generate", "read"):
            sys.exit('the request needs "type": "generate" or "read"')
        code, out = call("POST", "/" + kind, req)
    print(compact(out))
    sys.exit(0 if code == 200 else 1)


if __name__ == "__main__":
    main()
