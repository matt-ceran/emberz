"""ElevenLabs Agents test callers for the two BENIGN personas only (PLAN.md 2c).

ElevenLabs' use policy (https://elevenlabs.io/use-policy) bans fraud and impersonation with no
stated exception for testing, so no scam persona is ever sent to ElevenLabs. `agents.check_persona`
enforces an allowlist; every payload builder calls it.
"""
