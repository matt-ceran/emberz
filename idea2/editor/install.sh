#!/bin/bash
# Installs the editor harness on the VM (run as root by `vm.sh deploy`, from the staging dir it copied).
# Idempotent. Does NOT start or stop anything (the GPU belongs to whichever mode is running).
set -euo pipefail
SRC=$(cd "$(dirname "$0")" && pwd)
E=/opt/idea2/editor
id idea2 >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin idea2
mkdir -p "$E" "$E/splits" /opt/idea2/work /opt/idea2/work/edited /opt/idea2/work/kernel
chmod 0777 /opt/idea2/work /opt/idea2/work/edited /opt/idea2/work/kernel
install -m 0644 "$SRC"/*.py "$SRC"/*.md "$E"/
install -m 0755 "$SRC"/mode.sh "$E"/
# only dev and select are copied (vm.sh deploy never sends test.jsonl)
rm -f "$E"/splits/*.jsonl
install -m 0644 "$SRC"/splits/*.jsonl "$E"/splits/ 2>/dev/null || true
if [ -e "$E/splits/test.jsonl" ]; then echo "refusing: test split must not be on the VM"; rm -f "$E/splits/test.jsonl"; exit 1; fi
if [ ! -e /opt/idea2/editor.env ]; then
  cat >/opt/idea2/editor.env <<'ENV'
# Read by idea2-editor and idea2-kernel (systemd) and by kernel.py. Change, then: mode.sh editor (restarts it)
EDITOR_BUDGET=50
EDITOR_MAX_RUNS=300
EDITOR_RUN_TIMEOUT=900
EDITOR_ALLOWED_SPLITS=dev,select
EDITOR_MAX_GEN=1024
EDITOR_GEN_BATCH=128
EDITOR_KV_TOKENS=24000
ENV
fi
chmod 0644 /opt/idea2/editor.env
install -m 0644 "$SRC"/units/idea2-editor.service "$SRC"/units/idea2-kernel.service /etc/systemd/system/
touch /opt/idea2/editor.log /opt/idea2/kernel.log /opt/idea2/work/log.jsonl
chown idea2 /opt/idea2/editor.log /opt/idea2/kernel.log
chmod 0644 /opt/idea2/editor.log /opt/idea2/kernel.log
chmod 0666 /opt/idea2/work/log.jsonl
systemctl daemon-reload
echo "installed: $(ls "$E" | tr '\n' ' ')"
echo "splits: $(cd "$E/splits" && wc -l *.jsonl 2>/dev/null | tr '\n' ' ')"
