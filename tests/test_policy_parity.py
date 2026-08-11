"""The parity gate: the socket must not change the policy's answer.

One fixed observation — a deterministic joint vector and a deterministic JPEG —
goes down three paths and must come back identical to 1e-6:

1. **over the socket**, through a real ``scripts/policy_server.py`` process;
2. **through ``PolicyRunner`` directly**, in-process, no wire;
3. **through lerobot's API directly** (``tests/parity_reference.py``), written
   without reference to the server's code.

(1) vs (2) isolates the transport: JSON float literals, the length framing, the
JPEG bytes surviving the socket. (1) vs (3) isolates the preprocessing: if the
server ever stops matching what the training pipeline fed the policy — a
forgotten ``/255``, an HWC/CHW slip, a missing ``eval()`` — this is where it
shows, before an eval run spends an hour producing plausible nonsense.

CI-safe: skips cleanly when there is no ``.venv-lerobot`` or no checkpoint,
which is the state of a fresh clone. Run it for real against a trained
checkpoint — the two subprocesses take about thirty seconds.
"""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import policy_server as ps  # noqa: E402

LEROBOT_PYTHON = REPO_ROOT / ".venv-lerobot" / "bin" / "python"
ATOL = 1e-6

JOINT_POS = [0.1234, -0.5678, 0.9012, 1.3456, -0.7890, 0.2468]
"""A fixed, unremarkable joint vector: nothing here should depend on it."""


def _checkpoints() -> list[Path]:
    """Every checkpoint under ``runs/train/``, newest run first."""
    root = REPO_ROOT / "runs" / "train"
    if not root.is_dir():
        return []
    found = []
    for run in sorted(root.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
        for which in ("best", "last"):
            candidate = run / "checkpoints" / which
            if (candidate / "pretrained_model" / "config.json").is_file():
                found.append(candidate.resolve())
                break
    return found


def _fixed_jpeg() -> bytes:
    """A deterministic JPEG at the wire resolution.

    Structured rather than random noise: a gradient plus a block compresses the
    way a real frame does, so the decoder is exercised on plausible input, and
    ``numpy``'s seeded generator makes the bytes identical on every machine.
    """
    numpy = pytest.importorskip("numpy")
    pil = pytest.importorskip("PIL.Image")

    rows = numpy.linspace(0, 255, ps.FRAME_HEIGHT, dtype=numpy.uint8)[:, None]
    columns = numpy.linspace(0, 255, ps.FRAME_WIDTH, dtype=numpy.uint8)[None, :]
    frame = numpy.zeros((ps.FRAME_HEIGHT, ps.FRAME_WIDTH, 3), dtype=numpy.uint8)
    frame[..., 0] = rows
    frame[..., 1] = columns
    frame[..., 2] = (rows.astype(numpy.uint16) + columns) // 2
    frame[80:160, 100:220] = (200, 30, 40)
    buffer = io.BytesIO()
    pil.fromarray(frame).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    """The checkpoint under test: ``$PARITY_CKPT``, else the newest trained one."""
    if not LEROBOT_PYTHON.is_file():
        pytest.skip(".venv-lerobot is not built in this checkout")
    override = os.environ.get("PARITY_CKPT")
    if override:
        return ps.resolve_pretrained(Path(override)).parent
    found = _checkpoints()
    if not found:
        pytest.skip("no trained checkpoint under runs/train/")
    return found[0]


@pytest.fixture(scope="module")
def fixture_input(tmp_path_factory) -> tuple[Path, bytes]:
    jpeg = _fixed_jpeg()
    path = tmp_path_factory.mktemp("parity") / "frame.jpg"
    path.write_bytes(jpeg)
    return path, jpeg


@pytest.fixture(scope="module")
def direct(checkpoint, fixture_input) -> dict:
    """Both no-socket paths, computed once in the lerobot interpreter."""
    jpeg_path, _ = fixture_input
    out = jpeg_path.with_name("direct.json")
    result = subprocess.run(
        [
            str(LEROBOT_PYTHON),
            str(REPO_ROOT / "tests" / "parity_reference.py"),
            "--ckpt", str(checkpoint),
            "--jpeg", str(jpeg_path),
            "--joints", ",".join(repr(value) for value in JOINT_POS),
            "--out", str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"parity_reference.py failed:\n{result.stdout}\n{result.stderr}")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def over_socket(checkpoint, fixture_input, tmp_path_factory) -> list[list[float]]:
    """The same observation, answered by a real server process over a real socket."""
    _, jpeg = fixture_input
    # AF_UNIX allows ~107 bytes of path, and pytest's tmp dirs are long; put the
    # socket somewhere short and deterministic instead.
    socket_path = REPO_ROOT / "runs" / "eval" / "tmp" / f"parity-{os.getpid()}.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = subprocess.Popen(
        [
            str(LEROBOT_PYTHON),
            str(REPO_ROOT / "scripts" / "policy_server.py"),
            "--ckpt", str(checkpoint),
            "--socket", str(socket_path),
            "--warmup",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 600
        connection = None
        while time.time() < deadline and connection is None:
            if server.poll() is not None:
                pytest.fail(f"policy server exited early:\n{server.stdout.read()}")
            try:
                candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                candidate.connect(str(socket_path))
                connection = candidate
            except OSError:
                candidate.close()
                time.sleep(0.5)
        if connection is None:
            pytest.fail("the policy server never started listening")
        with connection:
            connection.sendall(ps.pack_request(JOINT_POS, jpeg))
            reply = ps.recv_reply(connection)
        return reply["actions"]
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a wedged server
            server.kill()
        socket_path.unlink(missing_ok=True)


def _max_delta(left, right) -> float:
    assert len(left) == len(right), f"chunk length {len(left)} vs {len(right)}"
    return max(
        abs(a - b)
        for row_a, row_b in zip(left, right, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )


def test_socket_matches_the_servers_own_direct_call(over_socket, direct):
    """The transport must be lossless: same code, one of them behind a socket."""
    delta = _max_delta(over_socket, direct["server_direct"])
    print(f"\nparity socket vs PolicyRunner direct: max abs delta {delta:.3e}")
    assert delta <= ATOL


def test_socket_matches_an_independent_lerobot_call(over_socket, direct):
    """The server's preprocessing must be lerobot's, not its own invention."""
    delta = _max_delta(over_socket, direct["reference"])
    print(f"\nparity socket vs independent lerobot reference: max abs delta {delta:.3e}")
    assert delta <= ATOL


def test_the_chunk_has_the_shape_the_client_will_ensemble(over_socket, checkpoint):
    config = json.loads(
        (checkpoint / "pretrained_model" / "config.json").read_text(encoding="utf-8")
    )
    assert len(over_socket) == config["chunk_size"]
    assert {len(row) for row in over_socket} == {6}


def test_every_action_is_a_finite_number(over_socket):
    """A NaN would drive the arm into a joint stop and read as 'the policy is bad'."""
    import math

    assert all(math.isfinite(value) for row in over_socket for value in row)
