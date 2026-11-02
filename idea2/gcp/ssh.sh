#!/bin/bash
# ssh into the Idea 2 VM. Extra args go to gcloud compute ssh, e.g.:
#   ./ssh.sh                                  interactive shell
#   ./ssh.sh --command 'nvidia-smi'           one command
#   ./ssh.sh --command 'tail -f /opt/idea2/vllm.log'
# Copy files: source config.sh; gc compute scp ./adapter -r "$IDEA2_VM":/opt/idea2/adapters/ --zone "$IDEA2_ZONE"
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
exec gcloud --project "$IDEA2_PROJECT" compute ssh "$IDEA2_VM" --zone "$IDEA2_ZONE" "$@"
