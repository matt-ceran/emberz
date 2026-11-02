# Numan's ledger

One line per paid action. Fireworks account B ($50, Auto Reload off), Google Cloud G1 (cap $25), Telnyx.
At every session end: `firectl deployment list`, `gcloud/check.sh`, Telnyx usage.

## G1: Google Cloud (cap $25)

Prices (USD/h, us-central1, spot / on-demand): n2-highcpu-96 2.0642 / 3.4414; n2-standard-96
2.7963 / 4.6614; g2-standard-8 + 1 L4 0.5108 / 0.8536 (gcloud/config.sh). Cost = wall hours x $/h
(+ a few cents of storage).

| Date | Run id | Job | Machine | Spot? | Calls / items | Start (UTC) | End (UTC) | Wall | Est. cost | check.sh clean after? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |

G1 running total: **$0.00** of $25.

Session-end checks:

| Date | check.sh: no emberz VMs/disks | Bucket size | By |
|---|---|---|---|
| | | | |
