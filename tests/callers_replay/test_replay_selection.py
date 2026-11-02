"""Selection balance for the human-replay clip set (callers/human_replay/build_set.py)."""
from collections import Counter

import pytest

from callers.human_replay.build_set import Candidate, candidates_from, select_clips, selection_stats

WORDS = "one two three four five six seven eight"


def make_pool(n_libri=40, n_edacc=46, clips_each=5, edacc_unknown=1):
    rows = []
    for i in range(n_libri):
        for j in range(clips_each):
            rows.append({"source": "librispeech", "speaker": f"librispeech:{i}", "gender": "female" if i % 2 else "male",
                         "accent": "", "path": f"/x/l{i}_{j}.flac", "duration": 6.5 + j, "text": WORDS})
    accents = ["Kenyan", "Irish", "Scottish", "Indian", "Spanish", "Chinese", "Nigerian", "US"]
    for i in range(n_edacc):
        g = "unknown" if i < edacc_unknown else ("female" if i % 3 else "male")
        for j in range(clips_each):
            rows.append({"source": "edacc", "speaker": f"edacc:C{i:02d}", "gender": g, "accent": accents[i % len(accents)],
                         "path": f"/x/e{i}_{j}.wav", "duration": 6.2 + j * 1.5, "text": WORDS})
    return candidates_from(rows)


def test_sixty_clips_distinct_speakers_and_balance():
    clips = select_clips(make_pool(), n=60, seed=7)
    st = selection_stats(clips)
    assert st["clips"] == 60
    assert st["speakers"] == 60                       # one clip per speaker when there are enough speakers
    assert st["by_source"] == {"edacc": 30, "librispeech": 30}
    per = Counter((c.source, c.gender) for c in clips)
    for s in ("edacc", "librispeech"):
        assert abs(per[(s, "female")] - per[(s, "male")]) <= 1
    assert per[("edacc", "unknown")] == 0              # unknown gender only as filler
    assert all(6.0 <= c.clip_duration <= 12.0 for c in clips)


def test_deterministic_and_seed_sensitive():
    pool = make_pool()
    a = [c.path for c in select_clips(pool, n=60, seed=7)]
    assert a == [c.path for c in select_clips(list(reversed(pool)), n=60, seed=7)]
    assert a != [c.path for c in select_clips(pool, n=60, seed=8)]


def test_accent_round_robin_covers_all_accents():
    clips = select_clips(make_pool(), n=60, seed=1)
    assert len({c.accent for c in clips if c.source == "edacc"}) == 8


def test_gender_shortfall_filled_from_other_gender():
    # only 5 male LibriSpeech speakers: the source quota is still met with female speakers
    rows = [{"source": "librispeech", "speaker": f"librispeech:{i}", "gender": "male" if i < 5 else "female",
             "accent": "", "path": f"/x/{i}.flac", "duration": 8.0, "text": WORDS} for i in range(40)]
    clips = select_clips(candidates_from(rows), n=30, seed=3)
    per = Counter(c.gender for c in clips)
    assert per["male"] == 5 and per["female"] == 25 and len({c.speaker for c in clips}) == 30


def test_second_clip_per_speaker_only_when_speakers_run_out():
    rows = [{"source": "librispeech", "speaker": f"librispeech:{i}", "gender": "female" if i % 2 else "male",
             "accent": "", "path": f"/x/{i}_{j}.flac", "duration": 8.0, "text": WORDS}
            for i in range(10) for j in range(3)]
    clips = select_clips(candidates_from(rows), n=25, seed=0)
    assert len(clips) == 25
    assert len({c.path for c in clips}) == 25          # never the same segment twice
    assert max(Counter(c.speaker for c in clips).values()) == 3
    assert min(Counter(c.speaker for c in clips).values()) == 2


def test_source_quota_moves_to_other_source_when_short():
    pool = make_pool(n_libri=40, n_edacc=4, clips_each=1)
    clips = select_clips(pool, n=40, seed=2)
    st = selection_stats(clips)
    assert st["by_source"]["edacc"] == 4 and st["by_source"]["librispeech"] == 36


def test_candidates_filter():
    rows = [
        {"source": "edacc", "speaker": "edacc:a", "duration": 5.9, "text": WORDS, "path": "a"},          # too short
        {"source": "edacc", "speaker": "edacc:a", "duration": 31.0, "text": WORDS, "path": "b"},         # too long to cut
        {"source": "edacc", "speaker": "edacc:a", "duration": 8.0, "text": "<LAUGH> <BREATH> yes", "path": "c"},  # markers
        {"source": "edacc", "speaker": "edacc:a", "duration": 8.0, "text": WORDS, "path": "d"},
        {"source": "edacc", "speaker": "edacc:a", "duration": 20.0, "text": WORDS, "path": "e"},          # cut to 12 s
    ]
    got = candidates_from(rows)
    assert [c.path for c in got] == ["d", "e"]
    assert got[1].clip_duration == 12.0
    assert got[0].gender == "unknown"


def test_interleaved_call_order_mixes_sources_early():
    clips = select_clips(make_pool(), n=60, seed=7)
    first8 = Counter((c.source, c.gender) for c in clips[:8])
    assert len(first8) == 4


@pytest.mark.parametrize("n", [0, 1, 7])
def test_small_n(n):
    assert len(select_clips(make_pool(), n=n, seed=7)) == n
