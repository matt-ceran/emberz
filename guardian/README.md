# guardian/: Matt's frozen guardian prompt, bridged into the live line

- `render.py` is an exact copy of `callguard/guardian/render.py`, the prompt the fine-tuned guardian was trained on (`PROMPT_SHA` 806908cfba7d). Do not edit it.
- `prompt.py` is what `line/guardian_client.py` imports (`guardian.prompt.render_messages(req)`). It converts a `GuardianRequest` into the training input and fails loudly if the prompt drifts.
- Optional: the training prompt includes "Earlier in this call" (the guardian's own verdict just before the 8-turn window). Set `req.earlier = {"risk", "category", "evidence"}` (dict key or attribute) when the brain tracks it. Without it, calls longer than 8 turns lose that context.

Contract changes (additive, so existing values still validate): `Stage` gains `"none"`, `Category` gains `"custom_rule"`, and `GuardianResponse.source` gains `"none"`. The tuned model emits these on safe turns and on caretaker-rule breaks. Without them, 1,107 of 1,722 real v2 verdicts (64%) failed validation.
