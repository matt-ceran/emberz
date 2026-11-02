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

## 2026-11-02 15:11 PST: first full live end-to-end call (Matt's emberz-guardian-v2)
SignalWire AI scam caller -> Telnyx guarded number -> conference with protected person (real phone).
11 turns transcribed live. Urgent alert on turn 4 (scam.impersonation_bank_gov) and turn 10
(disclosure.credentials_otp). Private coach to protected person on turns 4, 7, 11: trigger->sent
702-753 ms, trigger->ack 1.6-1.8 s. Caretaker phoned at alert, pressed a key, joined in monitor mode,
whispered the second alert. Call 98 s; 0 brain errors. Fix that made it work: RTP timestamps were
fed to ASR as ms (commit 3c9f725). Guardian: reasoning_effort=none (reasoning otherwise truncates JSON).
