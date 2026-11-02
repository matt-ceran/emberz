#!/bin/bash
# Create (or start) the Idea 2 GPU VM, idempotently, then wait until vLLM answers on the VM.
#   ./launch.sh            create if missing, start if stopped, no-op if running
#   NO_WAIT=1 ./launch.sh  return right after create/start
# First boot installs vLLM and downloads Qwen3-8B (~10-15 min). Later boots: vLLM up in ~2-3 min.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh

[[ "$IDEA2_VM" == "$IDEA2_PREFIX"* ]] || die "IDEA2_VM must start with $IDEA2_PREFIX"

st=$(vm_status || true)
case "$st" in
  RUNNING)
    echo "$IDEA2_VM already RUNNING in $IDEA2_ZONE" ;;
  TERMINATED|STOPPED|SUSPENDED)
    echo "starting stopped VM $IDEA2_VM (max run $IDEA2_MAX_RUN from now, then it stops again)"
    if [ "$st" = SUSPENDED ]; then gc compute instances resume "$IDEA2_VM" --zone "$IDEA2_ZONE"
    else gc compute instances start "$IDEA2_VM" --zone "$IDEA2_ZONE"; fi ;;
  STOPPING|PROVISIONING|STAGING|REPAIRING)
    die "$IDEA2_VM is $st; wait a minute and rerun" ;;
  "")
    python3 -c "import sys; p,d,h=map(float,sys.argv[1:]); print(f'cost: \${p:.4f}/h running + \${d:.4f}/h disk; one max-run window ($IDEA2_MAX_RUN) = \${p*h:.2f} at most')" \
      "$PRICE_VM_PER_H" "$PRICE_DISK_PER_H" "${IDEA2_MAX_RUN%h}"
    for z in $IDEA2_ZONES; do
      echo "creating $IDEA2_VM in $z: $IDEA2_MACHINE_TYPE + L4, $IDEA2_PROVISIONING, ${IDEA2_DISK_GB} GB $IDEA2_DISK_TYPE"
      if gc compute instances create "$IDEA2_VM" --zone "$z" \
          --machine-type "$IDEA2_MACHINE_TYPE" --accelerator "$IDEA2_ACCELERATOR" \
          --provisioning-model "$IDEA2_PROVISIONING" --maintenance-policy TERMINATE \
          --max-run-duration "$IDEA2_MAX_RUN" --instance-termination-action STOP \
          --image-family "$IDEA2_IMAGE_FAMILY" --image-project "$IDEA2_IMAGE_PROJECT" \
          --boot-disk-size "${IDEA2_DISK_GB}GB" --boot-disk-type "$IDEA2_DISK_TYPE" \
          --labels "$IDEA2_LABEL" --no-service-account --no-scopes \
          --metadata-from-file startup-script=vm_setup.sh 2>"${TMPDIR:-/tmp}/idea2-create.err"; then
        IDEA2_ZONE=$z; break
      fi
      grep -q -i "STOCKOUT\|resource_availability\|does not have enough resources" "${TMPDIR:-/tmp}/idea2-create.err" \
        || { cat "${TMPDIR:-/tmp}/idea2-create.err" >&2; die "create failed in $z"; }
      echo "  L4 stockout in $z, trying the next zone"
    done
    [ -n "${IDEA2_ZONE:-}" ] || die "no L4 capacity in: $IDEA2_ZONES (retry later, or IDEA2_ZONES='us-east4-a ...' with matching quota)" ;;
  *) die "unexpected VM status '$st'" ;;
esac

[ "${NO_WAIT:-0}" = 1 ] && exit 0
echo "waiting for 'idea2-setup: READY' on the serial console (first boot ~10-15 min)..."
start=$(date +%s)
while :; do
  out=$(gc compute instances get-serial-port-output "$IDEA2_VM" --zone "$IDEA2_ZONE" 2>/dev/null \
        | grep -a "idea2-setup:" | sed 's/.*idea2-setup: /  /' || true)
  last=$(echo "$out" | tail -n 1)
  [ -n "$last" ] && echo "[$(( ($(date +%s) - start) / 60 ))m] $last"
  # only lines from this boot: READY/FAILED after the latest "boot" line
  tail_boot=$(echo "$out" | awk '/ boot /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}')
  echo "$tail_boot" | grep -q "READY" && { echo "vLLM is up. Next: ./tunnel.sh"; exit 0; }
  echo "$tail_boot" | grep -q "FAILED" && die "setup failed; ./ssh.sh then: tail -n 100 /opt/idea2/setup.log /opt/idea2/vllm.log"
  [ $(( $(date +%s) - start )) -gt 2400 ] && die "timed out after 40 min; check ./ssh.sh"
  sleep 30
done
