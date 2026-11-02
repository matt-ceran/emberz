"""Timing and text cues for "is the caller an AI agent?", from transcript turns only.

Three explainable cues, each a number in [0, 1] (1 = agent-like):

1. Reply gap. People answer after ~0.2 s and often overlap; voice agents (ASR end-of-turn
   + LLM + TTS) answer after ~1.3-2.4 s (PLAN.md section V1). The gap we measure at the
   line server also contains the caller's network round trip, so the human anchor is set
   to 0.5 s, not 0.2 s. Median over the caller's replies; 0 at or below `human_gap_ms`,
   1 at or above `agent_gap_ms`, linear in between. Overlaps (negative gaps) count as 0.
2. Missing disfluencies. People say "um"/"uh" a few times per 100 words; TTS agents
   almost never do. Weak cue: the ASR may drop fillers too, so it only counts after
   `min_words` caller words and has a small weight.
3. Self-identification. "I'm an AI assistant", "virtual assistant", "automated call", ...
   Lawful AI callers must identify themselves (47 CFR 64.1200(b)), so this is strong:
   it lifts the score to at least 0.95 by itself.

score = weighted mean of the gap cue (0.75) and, once there are `min_words` caller
words, the disfluency cue (0.25); then max(score, 0.95) if a self-ID phrase was heard.
Nothing is emitted until there are at least `min_gaps` caller replies or a self-ID
phrase: the disfluency cue never scores on its own (an ASR that drops "um"s would make
every fluent human read as 1.0, and line/brain.py keeps the max over sources).
Negated phrases ("this is not an automated call", "not an AI assistant") do not count.
Duplicate turn numbers (a re-delivered `turn` event) are ignored.

Thresholds are starting values; calibrate them on V2 human calls over the real path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from contracts.models import Turn, VoiceSignal

SELF_ID = re.compile(
    r"\b(?:"
    r"i\s*'?\s*a?m\s+an?\s+(?:ai|a\.i\.|artificial|automated|virtual|digital)(?!\w)"
    # no "system": human scam scripts say "our automated/AI system flagged your account"
    r"|(?:ai|a\.i\.|virtual|digital|automated)\s+(?:assistant|agent|receptionist)"
    r"|this\s+is\s+an?\s+(?:automated|recorded|pre-?recorded)"
    r"|automated\s+(?:call|voice)"  # "this is an automated message/reminder" is matched above
    r"|(?:uses?|using|powered\s+by)\s+artificial\s+intelligence"
    r"|recorded\s+message"
    r")",
    re.IGNORECASE,
)
FILLERS = re.compile(r"\b(um+|uh+|erm+|er|hmm+|mm+|uh-huh|you know|i mean)\b", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z']+")
NEGATION = re.compile(r"(?:\bnot|\bnever|n't|\bnot\s+really)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)


def find_self_id(text: str) -> Optional[str]:
    """First self-identification phrase in `text` that is not negated just before it."""
    for m in SELF_ID.finditer(text):
        if not NEGATION.search(text[max(0, m.start() - 40): m.start()]):
            return m.group(0)
    return None


def _ramp(x: float, lo: float, hi: float) -> float:
    return 0.0 if x <= lo else 1.0 if x >= hi else (x - lo) / (hi - lo)


@dataclass
class TimingScorer:
    call_id: str
    speaker: str = "caller"           # whose turns we judge
    human_gap_ms: float = 500.0
    agent_gap_ms: float = 1300.0
    human_fillers_per_100w: float = 2.0  # at or above this rate: disfluency cue = 0
    min_gaps: int = 2
    min_words: int = 30
    w_gap: float = 0.75
    w_disfl: float = 0.25
    emit_delta: float = 0.05          # only emit when the score moved this much

    gaps: list[int] = field(default_factory=list)
    words: int = 0
    fillers: int = 0
    self_id: Optional[str] = None
    _prev: Optional[Turn] = None
    _last_emitted: Optional[float] = None
    _seen: set = field(default_factory=set)

    def update(self, turn: Turn) -> Optional[VoiceSignal]:
        if turn.turn in self._seen:  # duplicate delivery of the same turn
            return None
        self._seen.add(turn.turn)
        prev, self._prev = self._prev, turn
        if turn.speaker != self.speaker:
            return None
        gap = turn.gap_ms
        if gap is None and prev is not None and prev.speaker != turn.speaker:
            gap = turn.t_start_ms - prev.t_end_ms
        if gap is not None and (prev is None or prev.speaker != turn.speaker):
            self.gaps.append(int(gap))
        self.words += len(WORD.findall(turn.text))
        self.fillers += len(FILLERS.findall(turn.text))
        if self.self_id is None:
            self.self_id = find_self_id(turn.text)

        score = self.score()
        if score is None:
            return None
        if self._last_emitted is not None and abs(score - self._last_emitted) < self.emit_delta:
            return None
        self._last_emitted = score
        return VoiceSignal(call_id=self.call_id, at_ms=turn.t_end_ms, ai_score=round(score, 4), model="timing")

    def cues(self) -> dict:
        gap = disfl = None
        if len(self.gaps) >= self.min_gaps:
            med = sorted(self.gaps)[len(self.gaps) // 2]
            gap = _ramp(med, self.human_gap_ms, self.agent_gap_ms)
        if gap is not None and self.words >= self.min_words:  # never scores alone (see module doc)
            rate = 100.0 * self.fillers / self.words
            disfl = 1.0 - min(rate / self.human_fillers_per_100w, 1.0)
        return {"gap": gap, "disfluency": disfl, "self_id": self.self_id,
                "median_gap_ms": sorted(self.gaps)[len(self.gaps) // 2] if self.gaps else None,
                "n_gaps": len(self.gaps), "words": self.words, "fillers": self.fillers}

    def score(self) -> Optional[float]:
        c = self.cues()
        parts = [(c["gap"], self.w_gap), (c["disfluency"], self.w_disfl)]
        parts = [(v, w) for v, w in parts if v is not None]
        s = sum(v * w for v, w in parts) / sum(w for _, w in parts) if parts else None
        if self.self_id:
            s = max(s or 0.0, 0.95)
        if s is None:
            return None
        return min(max(s, 0.0), 1.0)
