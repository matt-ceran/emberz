"""Media WebSocket: Telnyx stream frames -> brain.on_media(track, payload_bytes, ts_ms)."""
import base64
import json

import pytest
from starlette.websockets import WebSocketDisconnect

from callflow_helpers import A_CCID, CALL_ID
from line.server import TrackReorder


def _media(track, chunk, payload: bytes, ts):
    return json.dumps({"event": "media", "sequence_number": str(chunk + 2), "stream_id": "st-1",
                       "media": {"track": track, "chunk": str(chunk), "timestamp": str(ts),
                                 "payload": base64.b64encode(payload).decode()}})


def test_media_frames_reach_the_brain_in_order(line):
    line.inbound()
    with line.client.websocket_connect(f"/media/{CALL_ID}") as ws:
        ws.send_text(json.dumps({"event": "connected", "version": "1.0.0"}))
        ws.send_text(json.dumps({"event": "start", "sequence_number": "1", "stream_id": "st-1", "start": {
            "call_control_id": A_CCID, "media_format": {"encoding": "PCMU", "sample_rate": 8000, "channels": 1}}}))
        # Chunks 1..4 per track; inbound 2 and 3 arrive swapped; outbound 2 duplicated.
        frames = [("inbound", 1, b"\x01" * 160, 0), ("outbound", 1, b"\x81" * 160, 0),
                  ("inbound", 3, b"\x03" * 160, 40), ("inbound", 2, b"\x02" * 160, 20),
                  ("outbound", 2, b"\x82" * 160, 20), ("outbound", 2, b"\x82" * 160, 20),
                  ("inbound", 4, b"\x04" * 160, 60), ("outbound", 3, b"\x83" * 160, 40)]
        for f in frames:
            ws.send_text(_media(*f))
        ws.send_text(json.dumps({"event": "stop", "sequence_number": "99", "stream_id": "st-1",
                                 "stop": {"call_control_id": A_CCID}}))
    brain = line.brains[0]
    inbound = [(p[0], ts) for t, p, ts in brain.media if t == "inbound"]
    outbound = [(p[0], ts) for t, p, ts in brain.media if t == "outbound"]
    assert inbound == [(1, 0), (2, 20), (3, 40), (4, 60)]
    assert outbound == [(0x81, 0), (0x82, 20), (0x83, 40)]
    assert all(len(p) == 160 for _, p, _ in brain.media)
    start = [p for e, p in brain.events if e == "media.start"][0]
    assert start["media_format"]["encoding"] == "PCMU" and start["stream_id"] == "st-1"
    assert line.flow.sessions[CALL_ID].stream_id == "st-1"


def test_media_socket_for_unknown_call_is_closed(line):
    with pytest.raises(WebSocketDisconnect) as e:
        with line.client.websocket_connect("/media/c_nope") as ws:
            ws.receive_text()
    assert e.value.code == 1008


def test_disconnect_without_stop_flushes_buffer(line):
    line.inbound()
    with line.client.websocket_connect(f"/media/{CALL_ID}") as ws:
        ws.send_text(_media("inbound", 1, b"\x01" * 160, 0))
        ws.send_text(_media("inbound", 2, b"\x02" * 160, 20))
    assert [p[0] for _, p, _ in line.brains[0].media] == [1, 2]


def test_reorder_buffer_drops_late_frames():
    r = TrackReorder(depth=1)
    assert r.push(1, b"a", 0) == []
    assert r.push(2, b"b", 20) == [(b"a", 0)]
    assert r.push(4, b"d", 60) == [(b"b", 20)]
    assert r.push(1, b"late", 0) == []          # older than what was emitted
    assert r.push(3, b"c", 40) == [(b"c", 40)]
    assert r.flush() == [(b"d", 60)]
