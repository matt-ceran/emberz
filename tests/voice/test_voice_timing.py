"""TimingScorer: reply gaps, self-identification, disfluencies."""
from __future__ import annotations

from contracts.models import Turn
from voice.timing import SELF_ID, TimingScorer


def call(gaps_ms, caller_texts=None, protected_text="okay sure", start=0):
    """Alternating protected/caller turns; caller turn i starts gaps_ms[i] after the protected turn ends."""
    turns, t, n = [], start, 1
    for i, g in enumerate(gaps_ms):
        turns.append(Turn(call_id="c1", turn=n, speaker="protected", text=protected_text, t_start_ms=t, t_end_ms=t + 800))
        t += 800
        n += 1
        text = (caller_texts or ["so what I need from you today is the account number"] * len(gaps_ms))[i]
        s = max(t + g, 0)
        turns.append(Turn(call_id="c1", turn=n, speaker="caller", text=text, t_start_ms=s, t_end_ms=s + 2000, gap_ms=g))
        t = s + 2000
        n += 1
    return turns


def run(turns, **kw):
    ts = TimingScorer("c1", **kw)
    sigs = [s for s in (ts.update(t) for t in turns) if s is not None]
    return ts, sigs


def test_nothing_before_two_gaps():
    ts, sigs = run(call([1500]))
    assert sigs == [] and ts.score() is None


def test_human_gaps_low_agent_gaps_high():
    chatty = ["um so yeah I uh I was calling about the the thing you know from last week"] * 5
    human, _ = run(call([150, 250, -100, 300, 200], chatty))
    agent, sigs = run(call([1600, 1900, 2200, 1400, 1800]))
    assert human.score() < 0.15
    assert agent.score() > 0.85
    assert sigs and sigs[-1].model == "timing" and sigs[-1].call_id == "c1"
    assert 0.0 <= sigs[-1].ai_score <= 1.0


def test_middle_gap_is_middle_score():
    ts, _ = run(call([900, 900, 900]))
    assert ts.cues()["gap"] == 0.5
    # gap 0.5 (weight 0.75) + no fillers in 30+ words, disfluency 1.0 (weight 0.25)
    assert abs(ts.score() - 0.625) < 1e-9


def test_self_id_alone_is_enough():
    turns = [Turn(call_id="c1", turn=1, speaker="caller", text="Hi, I'm an AI assistant calling from Riverside Pharmacy.",
                  t_start_ms=0, t_end_ms=3000)]
    ts, sigs = run(turns)
    assert sigs and sigs[0].ai_score >= 0.95
    assert ts.cues()["self_id"].lower().startswith("i'm an ai")


def test_self_id_phrases():
    yes = ["I am an A.I. calling about your order", "this is an automated message", "your virtual assistant here",
           "I'm a digital agent", "this call uses artificial intelligence", "Im an AI"]
    no = ["I'm a nurse at the clinic", "she said assistant manager", "I am an adult", "automatically renewed"]
    assert all(SELF_ID.search(s) for s in yes)
    assert not any(SELF_ID.search(s) for s in no)


def test_fillers_lower_the_score():
    fluent = ["so what I need from you today is the account number and the routing number please"] * 4
    disfluent = ["um so uh what I need from you today is um the account number and uh the routing number"] * 4
    a, _ = run(call([900, 900, 900, 900], fluent))
    b, _ = run(call([900, 900, 900, 900], disfluent))
    assert a.cues()["disfluency"] == 1.0
    assert b.cues()["disfluency"] == 0.0
    assert a.score() > b.score()


def test_gap_computed_when_missing():
    turns = [
        Turn(call_id="c1", turn=1, speaker="protected", text="hello", t_start_ms=0, t_end_ms=1000),
        Turn(call_id="c1", turn=2, speaker="caller", text="hi there", t_start_ms=2800, t_end_ms=4000),
        Turn(call_id="c1", turn=3, speaker="protected", text="yes", t_start_ms=4100, t_end_ms=4500),
        Turn(call_id="c1", turn=4, speaker="caller", text="great", t_start_ms=6300, t_end_ms=7000),
    ]
    ts, _ = run(turns)
    assert ts.gaps == [1800, 1800]
    assert ts.score() == 1.0


def test_emits_only_on_change():
    ts, sigs = run(call([1600, 1600, 1600, 1600, 1600]))
    assert len(sigs) == 1  # same score every turn after the second gap


# ---- reviewer additions
def test_disfluency_alone_never_scores():
    # One long fluent caller monologue (no replies yet): 40+ words, zero fillers, no gaps.
    turns = [Turn(call_id="c1", turn=1, speaker="caller",
                  text="hello this is Mark from the billing office and I am calling about your account because "
                       "we noticed several charges that we need to verify with you today before the end of business",
                  t_start_ms=0, t_end_ms=9000)]
    ts, sigs = run(turns)
    assert ts.words >= 30 and sigs == [] and ts.score() is None


def test_negated_self_id_does_not_count():
    from voice.timing import find_self_id

    assert find_self_id("Don't worry, this is not an automated call, I'm a real person") is None
    assert find_self_id("I'm not some AI assistant, ma'am") is None
    assert find_self_id("this isn't a recorded message") is None
    assert find_self_id("No, I'm an AI assistant from the pharmacy") is not None
    assert find_self_id("Hi, this is an automated reminder") is not None
    assert not SELF_ID.search("artificial intelligence is everywhere these days")
    assert find_self_id("our automated system flagged a charge, and our AI system locked the card") is None
    assert find_self_id("we sent you an automated message yesterday") is None


def test_duplicate_turn_is_ignored():
    turns = call([1600, 1600])
    ts = TimingScorer("c1")
    for t in turns + turns:  # every turn re-delivered
        ts.update(t)
    assert ts.gaps == [1600, 1600]
    assert ts.words == sum(len(t.text.split()) for t in turns if t.speaker == "caller")
