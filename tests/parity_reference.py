"""Reference inference for the parity test. Runs in `.venv-lerobot`, not pytest.

Given a checkpoint and a fixed observation, this prints the action chunk twice:

``reference``
    Built here, straight from lerobot's own API — ``ACTPolicy.from_pretrained``,
    ``make_pre_post_processors``, and the uint8 → CHW → ``/255`` float
    conversion transcribed from ``lerobot.datasets.video_utils``. It does not
    import ``scripts/policy_server.py`` at all, so if the server's
    preprocessing has drifted from what the training pipeline fed the policy,
    the two disagree.

``server_direct``
    ``policy_server.PolicyRunner`` called in-process, with no socket. Compared
    against the socket result, this isolates the transport: any difference is
    the wire's, not the model's.

Not named ``test_*`` on purpose — pytest must not collect it, because the
interpreter running the suite has no lerobot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def reference_chunk(pretrained: Path, joint_pos: list[float], jpeg: bytes) -> list[list[float]]:
    """The action chunk, computed from lerobot's API without the server's code."""
    import io

    import numpy as np
    import torch
    from PIL import Image

    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = ACTConfig.from_pretrained(pretrained)
    config.device = "cpu"
    policy = ACTPolicy.from_pretrained(pretrained, config=config)
    policy.to("cpu")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(pretrained),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    hwc = np.asarray(image, dtype=np.uint8)
    chw = torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1)))
    observation = {
        "observation.state": torch.tensor(joint_pos, dtype=torch.float32),
        next(iter(config.image_features)): (chw / 255.0).type(torch.float32),
    }
    with torch.inference_mode():
        chunk = postprocessor(policy.predict_action_chunk(preprocessor(observation)))
    return [[float(value) for value in row] for row in chunk[0]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--jpeg", required=True, type=Path)
    parser.add_argument("--joints", required=True, help="comma-separated joint positions")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    import policy_server

    pretrained = policy_server.resolve_pretrained(args.ckpt)
    joint_pos = [float(value) for value in args.joints.split(",")]
    jpeg = args.jpeg.read_bytes()

    runner = policy_server.PolicyRunner(pretrained, device="cpu")
    payload = {
        "pretrained": str(pretrained),
        "server_direct": runner(joint_pos, jpeg),
        "reference": reference_chunk(pretrained, joint_pos, jpeg),
    }
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
