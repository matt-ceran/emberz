# Shared by launch_noise.sh and launch_voice.sh (sourced after config.sh).

die() { echo "error: $*" >&2; exit 1; }

# The VM is named emberz-<run-id>: GCE names are lowercase letters, digits and dashes (job_config.py
# applies the same rule on the VM). Checked before anything is uploaded.
check_run_id() {
  [[ "$1" =~ ^[a-z0-9][a-z0-9-]{0,39}$ ]] || die "RUN_ID '$1' must be 1-40 chars of a-z, 0-9 and '-', starting with a letter or digit"
}

# Refuse to launch when the project cannot run the VM (quota, bucket, service account),
# so we never pay for a VM that fails at boot.
preflight() {
  local machine=$1 gpus=${2:-0}
  local need; need=$(vcpus_of "$machine")
  gc storage buckets describe "$EMBERZ_BUCKET" --format="value(name)" >/dev/null 2>&1 \
    || die "bucket $EMBERZ_BUCKET missing: run gcloud/setup_bucket.sh first"
  gc iam service-accounts describe "$EMBERZ_SA" --format="value(email)" >/dev/null 2>&1 \
    || die "service account $EMBERZ_SA missing: run gcloud/setup_bucket.sh first"
  local free
  free=$(gc compute project-info describe --format=json | python3 -c '
import json, sys
q = {x["metric"]: x for x in json.load(sys.stdin)["quotas"]}
c, g = q["CPUS_ALL_REGIONS"], q.get("GPUS_ALL_REGIONS", {"limit": 0, "usage": 0})
print(int(c["limit"] - c["usage"]), int(g["limit"] - g["usage"]))')
  local free_cpu=${free% *} free_gpu=${free#* }
  [ "$free_cpu" -ge "$need" ] || die "$machine needs $need vCPUs but CPUS_ALL_REGIONS has only $free_cpu free.
  Ask for a quota increase (IAM & Admin > Quotas: compute.googleapis.com/cpus_all_regions) or pick
  a smaller machine, e.g. NOISE_MACHINE_TYPE=n2-standard-8 NOISE_MAX_RUN=6h (see gcloud/README.md)."
  [ "$free_gpu" -ge "$gpus" ] || die "needs $gpus GPU(s) but GPUS_ALL_REGIONS has only $free_gpu free"
  # Regional quotas: CPUS (G2 has no family metric, it counts here), N2_CPUS for n2-*,
  # L4 (spot uses PREEMPTIBLE_NVIDIA_L4_GPUS when its limit is > 0), SSD_TOTAL_GB (pd-balanced).
  local disk_gb=${3:-0} prov=${4:-STANDARD}
  gc compute regions describe "$EMBERZ_REGION" --format=json | python3 -c '
import json, sys
need_cpu, gpus, disk, family, prov = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
q = {x["metric"]: x for x in json.load(sys.stdin)["quotas"]}
def free(m):
    return q[m]["limit"] - q[m]["usage"] if m in q else None
want = [("CPUS", need_cpu), ("SSD_TOTAL_GB", disk)]
if family == "n2":
    want.append(("N2_CPUS", need_cpu))
if gpus:
    spot = prov == "SPOT" and (free("PREEMPTIBLE_NVIDIA_L4_GPUS") or 0) > 0
    want.append(("PREEMPTIBLE_NVIDIA_L4_GPUS" if spot else "NVIDIA_L4_GPUS", gpus))
bad = [f"{m}: need {n}, free {free(m):.0f}" for m, n in want if free(m) is not None and free(m) < n]
if bad:
    sys.exit("regional quota short in this region: " + "; ".join(bad))
' "$need" "$gpus" "$disk_gb" "${machine%%-*}" "$prov" || die "quota check failed (see above)"
  local running
  running=$(gc compute instances list --filter="name~^${EMBERZ_PREFIX}" --format="value(name)")
  [ -z "$running" ] || echo "note: emberz VMs already running: $running"
}

# Tar only what the VM needs. Usage: pack_code out.tgz path...
pack_code() {
  local out=$1; shift
  (cd "$REPO_ROOT" && tar --exclude='__pycache__' --exclude='*.pyc' -czf "$out" "$@")
}

confirm() {
  [ "${YES:-0}" = 1 ] && return 0
  read -r -p "$1 Type yes: " ok
  [ "$ok" = "yes" ] || die "aborted"
}

# price per hour for machine + provisioning, from config.sh (empty when unknown)
price_of() {
  local key; key="PRICE_$(echo "$1" | tr 'a-z-' 'A-Z_')_$2"
  echo "${!key:-}"
}

# hours in a --max-run-duration value ("90m", "2h", "1h30m", "1d2h3m4s"; gcloud's format)
dur_hours() {
  python3 -c "
import re, sys
d = sys.argv[1]
m = re.fullmatch(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?', d)
if not d or not m:
    sys.exit('bad duration ' + repr(d) + ' (use e.g. 90m, 2h, 1h30m)')
dd, hh, mm, ss = (int(x or 0) for x in m.groups())
print(dd * 24 + hh + mm / 60 + ss / 3600)
" "$1"
}

# worst-case cost = price/h x max-run hours
worst_case() {
  local h; h=$(dur_hours "$2") || return 1
  python3 -c "
import sys
p, d, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
print(f'\${float(p) * h:.2f} at most ({d} x \${float(p):.4f}/h)' if p else f'unknown price for this machine; {d} max')
" "$1" "$2" "$h"
}
