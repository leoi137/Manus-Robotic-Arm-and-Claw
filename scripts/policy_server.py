"""Serve a trained ACT checkpoint over a unix socket, on the CPU. `.venv-lerobot`.

.. code-block:: bash

    ./.venv-lerobot/bin/python scripts/policy_server.py \
        --ckpt runs/train/<run>/checkpoints/best --warmup

    # the Isaac client, in the other interpreter, in another shell
    ~/isaaclab-env/bin/python scripts/eval_policy.py --ckpt-run <run> --episodes 10

**Why a socket at all.** Isaac Sim and lerobot cannot share a process — they
disagree on numpy, and this machine allows one GPU process at a time. So the
policy runs here, on the CPU, and the simulator keeps the card to itself. That
is the plan's ceiling contract, not a performance preference.

**The wire format, exactly.** No pickle anywhere: a pickle on a socket is a
remote-code-execution primitive, and nothing here needs one.

.. code-block:: text

    request   uint32 big-endian  length of the JSON header
              JSON header        {"joint_pos": [6 floats], "jpeg_len": int}
              raw bytes          jpeg_len bytes of JPEG, 320x240 RGB

    reply     uint32 big-endian  length of the JSON body
              JSON body          {"actions": [[6 floats] x K], "server_ms": float}
                                 or {"error": "..."} — never a silent zero

Only ``actions`` is load-bearing; ``server_ms`` is the server's own view of its
latency, so a report can separate inference cost from IPC cost. Floats cross as
JSON decimal literals, which round-trip float32 exactly at 17 significant
digits — the parity test pins that rather than trusting it.

**The server owns normalization.** Both processor pipelines are loaded from the
checkpoint directory, so the statistics applied at inference are the ones
``scripts/train_act.py`` baked in at save time — the client never sees a mean,
a standard deviation, or a feature name, and cannot get them wrong. Image
preprocessing mirrors what the training pipeline fed the policy, grounded in
``lerobot.datasets.video_utils``: decode to ``uint8`` (H, W, C), permute to
(C, H, W), cast to float32 and divide by 255, then hand it to lerobot's own
normalizer step. Nothing is invented here.

Top-level imports are stdlib only, on purpose: the framing helpers are shared
verbatim with the Isaac-side client (which cannot import torch), and the
protocol tests run in either interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

LENGTH_STRUCT = struct.Struct(">I")
"""Both length prefixes: 4 bytes, big-endian, unsigned."""

MAX_HEADER_BYTES = 1 << 16
MAX_JPEG_BYTES = 1 << 24
"""Sanity ceilings so a corrupt length cannot ask for a gigabyte allocation."""

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
"""The resolution the whole chain is specified at (``scripts/gen_dataset.py``)."""

DEFAULT_SOCKET = REPO_ROOT / "runs" / "eval" / "tmp" / "policy.sock"
"""Default socket path: inside the (gitignored) run tree, not a shared /tmp name."""

PRETRAINED_DIR = "pretrained_model"
"""lerobot's name for the weights-and-processors half of a checkpoint."""


# --- Wire format -----------------------------------------------------------------
# Pure stdlib, importable from both interpreters. Every function here is total:
# it either returns a well-formed value or raises, and never half-reads a stream.


def pack_request(joint_pos: Any, jpeg: bytes) -> bytes:
    """Frame one observation for the wire."""
    header = json.dumps(
        {"joint_pos": [float(value) for value in joint_pos], "jpeg_len": len(jpeg)}
    ).encode("utf-8")
    return LENGTH_STRUCT.pack(len(header)) + header + jpeg


def pack_reply(actions: Any, server_ms: float | None = None) -> bytes:
    """Frame one action chunk for the wire."""
    body: dict[str, Any] = {"actions": [[float(value) for value in row] for row in actions]}
    if server_ms is not None:
        body["server_ms"] = float(server_ms)
    return _frame_json(body)


def pack_error(message: str) -> bytes:
    """Frame an error the client must not mistake for a chunk of zeros."""
    return _frame_json({"error": str(message)})


def _frame_json(body: dict[str, Any]) -> bytes:
    payload = json.dumps(body).encode("utf-8")
    return LENGTH_STRUCT.pack(len(payload)) + payload


def read_exactly(stream: Any, count: int) -> bytes:
    """Read exactly `count` bytes from a socket-like object.

    Raises:
        ConnectionError: If the peer closed before `count` bytes arrived. A
            short read is never returned — a truncated JPEG that decoded to
            garbage would be far worse than a dropped connection.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        block = stream.recv(remaining)
        if not block:
            raise ConnectionError(f"peer closed with {remaining} of {count} bytes outstanding")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def recv_request(stream: Any) -> tuple[list[float], bytes]:
    """Read one request. Returns the joint positions and the raw JPEG bytes."""
    (header_len,) = LENGTH_STRUCT.unpack(read_exactly(stream, LENGTH_STRUCT.size))
    if header_len > MAX_HEADER_BYTES:
        raise ValueError(f"header claims {header_len} bytes (max {MAX_HEADER_BYTES})")
    header = json.loads(read_exactly(stream, header_len).decode("utf-8"))
    joint_pos = [float(value) for value in header["joint_pos"]]
    jpeg_len = int(header["jpeg_len"])
    if jpeg_len > MAX_JPEG_BYTES:
        raise ValueError(f"header claims a {jpeg_len}-byte JPEG (max {MAX_JPEG_BYTES})")
    return joint_pos, read_exactly(stream, jpeg_len)


def recv_reply(stream: Any) -> dict[str, Any]:
    """Read one reply body. Raises RuntimeError if the server reported an error."""
    (length,) = LENGTH_STRUCT.unpack(read_exactly(stream, LENGTH_STRUCT.size))
    body = json.loads(read_exactly(stream, length).decode("utf-8"))
    if "error" in body:
        raise RuntimeError(f"policy server: {body['error']}")
    return body


# --- Checkpoint resolution -------------------------------------------------------


def resolve_pretrained(path: Path) -> Path:
    """Find the ``pretrained_model`` directory `path` refers to.

    Accepts, in order: the directory itself, a checkpoint directory containing
    it, or a run directory whose ``checkpoints/best`` (else ``checkpoints/last``)
    resolves to one. Being liberal here costs eight lines and saves the operator
    from remembering which of three plausible paths the server wanted.
    """
    path = Path(path)
    candidates = [
        path,
        path / PRETRAINED_DIR,
        path / "checkpoints" / "best" / PRETRAINED_DIR,
        path / "checkpoints" / "last" / PRETRAINED_DIR,
    ]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise SystemExit(
        f"no checkpoint under {path}; looked for config.json in "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def checkpoint_digest(pretrained: Path) -> str | None:
    """SHA-256 of the weights file, for the eval run's provenance."""
    import hashlib

    weights = pretrained / "model.safetensors"
    if not weights.is_file():
        return None
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- The policy ------------------------------------------------------------------


class PolicyRunner:
    """A loaded checkpoint, ready to turn one observation into one action chunk.

    Separated from the socket so the parity test can call the identical code
    path directly, with no wire in between: whatever difference the test finds
    is then attributable to the transport alone.
    """

    def __init__(self, pretrained: Path, device: str = "cpu") -> None:
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        self.pretrained = Path(pretrained)
        self.device = device

        # The checkpoint's own config, retargeted at this device. Everything
        # else about it -- the feature spec, the chunk length, the backbone --
        # is the trainer's, not this file's.
        self.config = ACTConfig.from_pretrained(self.pretrained)
        self.config.device = device
        self.policy = ACTPolicy.from_pretrained(self.pretrained, config=self.config)
        self.policy.to(device)
        self.policy.eval()
        # Both pipelines come from the checkpoint, so the statistics are the
        # ones training used. Only the device is overridden: the trainer saved
        # them pointing at cuda and this process must not touch the card.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.config,
            pretrained_path=str(self.pretrained),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": device}},
        )
        self.image_key = next(iter(self.config.image_features))
        self.state_key = "observation.state"
        self.chunk_size = int(self.config.chunk_size)
        self.action_dim = int(self.config.output_features["action"].shape[0])

    def decode_image(self, jpeg: bytes) -> Any:
        """JPEG bytes to the float tensor the training pipeline fed the policy.

        The conversion is transcribed from
        ``lerobot.datasets.video_utils.decode_video_frames_torchcodec``, which
        is what produced every image ACT was trained on:
        ``(frames / 255.0).type(torch.float32)`` over a (C, H, W) uint8 tensor.
        Frames that arrive at the wrong size are resampled with the same PIL
        LANCZOS filter ``scripts/gen_dataset.py`` used to make the dataset, so
        a client that forgets to downscale is merely slow, not wrong.
        """
        import io

        import numpy as np
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg))
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (FRAME_WIDTH, FRAME_HEIGHT):
            image = image.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS)
        hwc = np.asarray(image, dtype=np.uint8)
        chw = self.torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1)))
        return (chw / 255.0).type(self.torch.float32)

    def observation(self, joint_pos: Any, jpeg: bytes) -> dict[str, Any]:
        """The unbatched observation dict, before lerobot's preprocessing.

        Unbatched on purpose: the pipeline's own ``AddBatchDimension`` step is
        what training used, so letting it do the work here keeps one less thing
        that could differ between the two paths.
        """
        state = self.torch.tensor([float(value) for value in joint_pos], dtype=self.torch.float32)
        if state.shape[0] != self.action_dim:
            raise ValueError(f"expected {self.action_dim} joint positions, got {state.shape[0]}")
        return {self.state_key: state, self.image_key: self.decode_image(jpeg)}

    def __call__(self, joint_pos: Any, jpeg: bytes) -> list[list[float]]:
        """One observation in, one (K, action_dim) chunk of joint targets out."""
        with self.torch.inference_mode():
            batch = self.preprocessor(self.observation(joint_pos, jpeg))
            chunk = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(chunk)
        return [[float(value) for value in row] for row in actions[0]]

    def warmup(self) -> float:
        """One dummy inference, so the first real request is not the slow one.

        Lazily built graphs, cuDNN/oneDNN algorithm selection and the first
        allocation of every intermediate all happen on request one; on a CPU
        server that is a multiple of the steady-state latency, and it would
        otherwise land in the middle of an episode.
        """
        import io

        import numpy as np
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)).save(
            buffer, format="JPEG", quality=92
        )
        started = time.perf_counter()
        self([0.0] * self.action_dim, buffer.getvalue())
        return time.perf_counter() - started


# --- Latency ---------------------------------------------------------------------


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile of `values` (0.5 = median). None when empty.

    Nearest-rank rather than interpolated: every reported number is then a
    latency that actually happened, which is the honest thing to quote for a
    control loop.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return ordered[rank - 1]


def latency_summary(samples: list[float]) -> dict[str, Any]:
    """p50/p95 and friends over per-request seconds, reported in milliseconds."""
    return {
        "requests": len(samples),
        "p50_ms": (percentile(samples, 0.50) or 0.0) * 1e3 if samples else None,
        "p95_ms": (percentile(samples, 0.95) or 0.0) * 1e3 if samples else None,
        "max_ms": max(samples) * 1e3 if samples else None,
        "mean_ms": (sum(samples) / len(samples)) * 1e3 if samples else None,
    }


# --- The server ------------------------------------------------------------------


def serve(runner: PolicyRunner, socket_path: Path, log_every: int) -> dict[str, Any]:
    """Accept connections and answer requests until interrupted.

    One connection at a time and one request at a time: the client is a single
    simulation loop, and a thread pool would buy nothing but a way for two
    inferences to fight over the same CPU.
    """
    samples: list[float] = []
    # sockaddr_un.sun_path is 108 bytes on Linux, and bind() reports the
    # overflow as a bare "AF_UNIX path too long" with no hint about the limit.
    if len(str(socket_path).encode()) >= 104:
        raise SystemExit(
            f"socket path is {len(str(socket_path))} bytes; AF_UNIX allows ~107. "
            f"Use something shorter, e.g. {DEFAULT_SOCKET}"
        )
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    # A leftover socket file from a killed server would make bind() fail with
    # EADDRINUSE even though nothing is listening.
    if socket_path.exists() or socket_path.is_symlink():
        socket_path.unlink()

    stopping = {"now": False}

    def stop(number: int, _frame: Any) -> None:
        stopping["now"] = True
        print(f"\n{signal.Signals(number).name}: shutting down", flush=True)
        raise KeyboardInterrupt

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, stop)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(socket_path))
        server.listen(1)
        print(
            f"listening on {socket_path} "
            f"(chunk {runner.chunk_size}, device {runner.device})",
            flush=True,
        )
        while not stopping["now"]:
            connection, _ = server.accept()
            print("client connected", flush=True)
            with connection:
                _serve_connection(runner, connection, samples, log_every)
            print(f"client disconnected after {len(samples)} requests", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
    return latency_summary(samples)


def _serve_connection(
    runner: PolicyRunner, connection: Any, samples: list[float], log_every: int
) -> None:
    """Answer every request on one connection until the client hangs up."""
    while True:
        try:
            joint_pos, jpeg = recv_request(connection)
        except (ConnectionError, OSError):
            return
        started = time.perf_counter()
        try:
            actions = runner(joint_pos, jpeg)
        except Exception as error:  # noqa: BLE001 - a dead episode beats a wrong action
            print(f"request failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            connection.sendall(pack_error(f"{type(error).__name__}: {error}"))
            continue
        elapsed = time.perf_counter() - started
        samples.append(elapsed)
        connection.sendall(pack_reply(actions, server_ms=elapsed * 1e3))
        if log_every and len(samples) % log_every == 0:
            summary = latency_summary(samples[-log_every:])
            print(
                f"  {len(samples)} requests, last {log_every}: "
                f"p50 {summary['p50_ms']:.1f} ms, p95 {summary['p95_ms']:.1f} ms",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="checkpoint directory (or a run directory: checkpoints/best is used)",
    )
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET, help="unix socket path")
    parser.add_argument(
        "--device", default="cpu", help="torch device; cpu by the plan's GPU contract"
    )
    parser.add_argument(
        "--warmup", action="store_true", help="one dummy inference before listening"
    )
    parser.add_argument(
        "--log-every", type=int, default=100, help="requests between latency lines"
    )
    parser.add_argument(
        "--stats-path", type=Path, default=None, help="write the latency summary here on exit"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load a checkpoint and serve it. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    if args.device != "cpu":
        # Not forbidden, but the plan's contract is one GPU process machine-wide
        # and during an eval that process is Isaac. Say so rather than quietly
        # taking the card.
        print(
            f"WARNING: --device {args.device} contradicts the shared-GPU contract "
            "(during eval the only GPU process is Isaac)",
            file=sys.stderr,
        )

    pretrained = resolve_pretrained(args.ckpt)
    started = time.perf_counter()
    runner = PolicyRunner(pretrained, device=args.device)
    print(
        f"loaded {pretrained} in {time.perf_counter() - started:.1f} s "
        f"(sha256 {(checkpoint_digest(pretrained) or 'unknown')[:12]}, "
        f"chunk {runner.chunk_size}, action dim {runner.action_dim})"
    )
    if args.warmup:
        print(f"warmup inference: {runner.warmup() * 1e3:.0f} ms")

    summary = serve(runner, args.socket, args.log_every)
    print(
        f"latency over {summary['requests']} requests: "
        + (
            f"p50 {summary['p50_ms']:.1f} ms, p95 {summary['p95_ms']:.1f} ms, "
            f"max {summary['max_ms']:.1f} ms"
            if summary["requests"]
            else "no requests served"
        )
    )
    if args.stats_path:
        args.stats_path.parent.mkdir(parents=True, exist_ok=True)
        args.stats_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(pretrained),
                    "checkpoint_sha256": checkpoint_digest(pretrained),
                    "device": args.device,
                    "chunk_size": runner.chunk_size,
                    "latency": summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
