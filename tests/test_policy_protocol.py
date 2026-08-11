"""The policy server's wire format, checkpoint resolution and latency maths.

Sim-free and torch-free: ``scripts/policy_server.py`` keeps its top-level
imports to the standard library precisely so the framing can be exercised (and
shared with the Isaac-side client) in an interpreter that has neither lerobot
nor a GPU. Everything here runs under `~/isaaclab-env` with the rest of the
suite.
"""

from __future__ import annotations

import json
import socket
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import policy_server as ps  # noqa: E402


@pytest.fixture
def pair():
    """A connected socket pair standing in for client and server."""
    left, right = socket.socketpair()
    try:
        yield left, right
    finally:
        left.close()
        right.close()


# --- Framing -----------------------------------------------------------------


def test_request_round_trips_exactly(pair):
    client, server = pair
    joints = [0.1, -0.25, 1.5, -0.0, 3.14159265358979, -2.7]
    jpeg = bytes(range(256)) * 7
    client.sendall(ps.pack_request(joints, jpeg))
    received_joints, received_jpeg = ps.recv_request(server)
    assert received_joints == joints
    assert received_jpeg == jpeg


def test_request_header_declares_the_jpeg_length(pair):
    client, server = pair
    jpeg = b"\xff\xd8not-really-a-jpeg\xff\xd9"
    client.sendall(ps.pack_request([0.0] * 6, jpeg))
    (header_len,) = ps.LENGTH_STRUCT.unpack(ps.read_exactly(server, 4))
    header = json.loads(ps.read_exactly(server, header_len).decode())
    assert header["jpeg_len"] == len(jpeg)
    assert header["joint_pos"] == [0.0] * 6
    assert ps.read_exactly(server, header["jpeg_len"]) == jpeg


def test_two_requests_do_not_bleed_into_each_other(pair):
    """Length prefixes, not delimiters: back-to-back frames must stay separate."""
    client, server = pair
    client.sendall(ps.pack_request([1.0] * 6, b"first") + ps.pack_request([2.0] * 6, b"second!"))
    assert ps.recv_request(server) == ([1.0] * 6, b"first")
    assert ps.recv_request(server) == ([2.0] * 6, b"second!")


def test_reply_round_trips_float_values_exactly(pair):
    """JSON decimal literals must not lose a bit of the float32 the policy emitted."""
    client, server = pair
    actions = [[0.1, -1 / 3, 1e-8, 123456.78125, -0.0, 2.5]] * 3
    server.sendall(ps.pack_reply(actions, server_ms=12.5))
    reply = ps.recv_reply(client)
    assert reply["actions"] == actions
    assert reply["server_ms"] == 12.5


def test_error_reply_raises_rather_than_looking_like_a_chunk(pair):
    """A failed inference must never be mistaken for a chunk of zeros."""
    client, server = pair
    server.sendall(ps.pack_error("ValueError: expected 6 joint positions, got 3"))
    with pytest.raises(RuntimeError, match="expected 6 joint positions"):
        ps.recv_reply(client)


def test_truncated_stream_raises_instead_of_short_reading(pair):
    client, server = pair
    client.sendall(ps.LENGTH_STRUCT.pack(64) + b"only-a-few-bytes")
    client.close()
    with pytest.raises(ConnectionError):
        ps.recv_request(server)


def test_absurd_header_length_is_refused_before_allocating(pair):
    client, server = pair
    client.sendall(ps.LENGTH_STRUCT.pack(ps.MAX_HEADER_BYTES + 1))
    with pytest.raises(ValueError, match="header claims"):
        ps.recv_request(server)


def test_absurd_jpeg_length_is_refused_before_allocating(pair):
    client, server = pair
    header = json.dumps({"joint_pos": [0.0] * 6, "jpeg_len": ps.MAX_JPEG_BYTES + 1}).encode()
    client.sendall(ps.LENGTH_STRUCT.pack(len(header)) + header)
    with pytest.raises(ValueError, match="JPEG"):
        ps.recv_request(server)


def test_length_prefix_is_four_bytes_big_endian():
    """The wire format is a contract with the other venv; pin its shape."""
    assert ps.LENGTH_STRUCT.size == 4
    assert ps.LENGTH_STRUCT.pack(1) == b"\x00\x00\x00\x01"
    assert ps.LENGTH_STRUCT is not struct.Struct("<I")


@pytest.mark.parametrize("script", ["policy_server.py", "eval_policy.py"])
def test_no_pickle_anywhere_on_the_wire(script):
    """The plan's security constraint, pinned so a refactor cannot reintroduce it.

    Deliberately a check on *code*, not on the word: both files discuss why
    there is no pickle here, and a naive substring search would fail on their
    own rationale.
    """
    source = (REPO_ROOT / "scripts" / script).read_text(encoding="utf-8")
    for forbidden in ("import pickle", "pickle.loads", "pickle.dumps", "cPickle", "marshal."):
        assert forbidden not in source


# --- Latency -----------------------------------------------------------------


def test_percentile_is_nearest_rank():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert ps.percentile(values, 0.5) == 5.0
    assert ps.percentile(values, 0.95) == 10.0
    assert ps.percentile(values, 0.0) == 1.0
    assert ps.percentile([], 0.5) is None


def test_percentile_reports_a_latency_that_happened():
    """Nearest-rank never interpolates, so every quoted number is a real sample."""
    values = [0.01, 0.02, 0.05]
    for fraction in (0.1, 0.5, 0.9, 1.0):
        assert ps.percentile(values, fraction) in values


def test_latency_summary_shape():
    summary = ps.latency_summary([0.010, 0.020, 0.030, 0.040])
    assert summary["requests"] == 4
    assert summary["p50_ms"] == pytest.approx(20.0)
    assert summary["p95_ms"] == pytest.approx(40.0)
    assert summary["max_ms"] == pytest.approx(40.0)
    assert summary["mean_ms"] == pytest.approx(25.0)


def test_latency_summary_of_nothing_is_not_zero():
    summary = ps.latency_summary([])
    assert summary["requests"] == 0
    assert summary["p50_ms"] is None


# --- Checkpoint resolution ----------------------------------------------------


def _fake_pretrained(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    return directory


def test_resolve_accepts_the_pretrained_directory_itself(tmp_path):
    pretrained = _fake_pretrained(tmp_path / "00002000" / "pretrained_model")
    assert ps.resolve_pretrained(pretrained) == pretrained.resolve()


def test_resolve_accepts_a_checkpoint_directory(tmp_path):
    pretrained = _fake_pretrained(tmp_path / "00002000" / "pretrained_model")
    assert ps.resolve_pretrained(tmp_path / "00002000") == pretrained.resolve()


def test_resolve_accepts_a_run_directory_and_prefers_best(tmp_path):
    best = _fake_pretrained(tmp_path / "checkpoints" / "00001500" / "pretrained_model")
    _fake_pretrained(tmp_path / "checkpoints" / "00002000" / "pretrained_model")
    (tmp_path / "checkpoints" / "best").symlink_to("00001500")
    (tmp_path / "checkpoints" / "last").symlink_to("00002000")
    assert ps.resolve_pretrained(tmp_path) == best.resolve()


def test_resolve_explains_itself_when_there_is_nothing_there(tmp_path):
    with pytest.raises(SystemExit, match="no checkpoint under"):
        ps.resolve_pretrained(tmp_path)


def test_digest_of_a_missing_weights_file_is_none(tmp_path):
    assert ps.checkpoint_digest(_fake_pretrained(tmp_path / "pretrained_model")) is None


def test_digest_is_the_sha256_of_the_weights(tmp_path):
    import hashlib

    pretrained = _fake_pretrained(tmp_path / "pretrained_model")
    payload = b"not really safetensors"
    (pretrained / "model.safetensors").write_bytes(payload)
    assert ps.checkpoint_digest(pretrained) == hashlib.sha256(payload).hexdigest()
