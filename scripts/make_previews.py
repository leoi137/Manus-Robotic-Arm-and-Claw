"""Turn a raw dataset into something a human can actually look at.

Two artefacts, for two different readers:

``datasets/raw/<name>/preview.mp4``
    A contact sheet: 16 evenly spaced episodes tiled 4x4 and played together,
    each labelled with its attempt index. This is the QC view — sixteen wrist
    POVs side by side make a mis-aimed grasp, a frozen camera or a lighting
    draw gone wrong obvious in one watch. It lives next to the ledger it
    describes and is **not** tracked (``datasets/raw/**`` is gitignored; the
    script checks that with ``git check-ignore`` and says so).

``media/datasets/<name>.gif``
    A small tracked preview: three episodes side by side, real-time at 10 fps,
    kept under 3 MB so it can live in git and render in ``DATASETS.md``.

.. code-block:: bash

    ~/isaaclab-env/bin/python scripts/make_previews.py --dataset grasp_cube_dev

Sim-free — it reads episodes through :func:`manus.recorder.load_episode` and
needs only numpy, Pillow and imageio, so it runs in either interpreter.

Two encoding details are load-bearing. **GIF timing is quantised to
centiseconds**: the format stores a frame delay in hundredths of a second, so
only a duration that is a whole number of centiseconds survives the round trip.
10 fps is exactly 10 cs, which is why the GIF is built at 10 fps (every third
30 Hz frame) rather than at some rate that would silently drift. And **one
global palette** is built for the whole GIF rather than one per frame: it keeps
the colours from crawling between frames and roughly halves the file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the src-layout package importable without installing it.
sys.path.insert(0, str(REPO_ROOT / "src"))

from manus import recorder  # noqa: E402

GRID_EPISODES = 16
"""Episodes in the contact sheet (4x4)."""

GRID_TILE = (160, 120)
"""Tile size (width, height) in the contact sheet: a quarter of a 320x240 frame."""

GRID_FPS = 30
"""Contact-sheet playback rate: one frame per control step, so it runs real time."""

GIF_EPISODES = 3
"""Episodes side by side in the tracked GIF."""

GIF_FPS = 10
"""GIF playback rate. 10 fps is exactly 10 centiseconds a frame — see the module
docstring on GIF timing quantisation."""

GIF_MAX_BYTES = 3 * 1024 * 1024
"""Ceiling for the tracked GIF: 3 MB."""

GIF_LADDER: tuple[tuple[tuple[int, int], int], ...] = (
    ((320, 240), 128),
    ((320, 240), 64),
    ((256, 192), 64),
    ((208, 156), 48),
)
"""``(panel size, palette colours)`` tried in order until the GIF fits.

The first rung is the specified preview (three 320x240 panels); every rung
below it trades a little fidelity for bytes. Which rung was used is printed and
returned, because "it fit" is only meaningful alongside what it cost.
"""


def _label(image: Image.Image, text: str) -> None:
    """Burn a small outlined caption into the top-left corner, in place."""
    draw = ImageDraw.Draw(image)
    for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((4 + offset[0], 3 + offset[1]), text, fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))


def _evenly_spaced(count: int, wanted: int) -> list[int]:
    """`wanted` indices spread over ``range(count)``, first and last included."""
    if count <= wanted:
        return list(range(count))
    return [int(round(index)) for index in np.linspace(0, count - 1, wanted)]


def success_episodes(dataset_dir: Path) -> list[Path]:
    """Every successful episode of a dataset, in attempt order.

    Read from the ledger rather than from a glob so the previews show the
    episodes the manifest counts, in the order the converter will write them.
    """
    paths = []
    for record in recorder.read_attempts(dataset_dir):
        if record["outcome"] == recorder.SUCCESS and record.get("episode_file"):
            path = dataset_dir / record["episode_file"]
            if path.is_file():
                paths.append(path)
    return paths


def _tile(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize one RGB frame to `size` (width, height)."""
    if (frame.shape[1], frame.shape[0]) == size:
        return frame
    return np.asarray(Image.fromarray(frame).resize(size, Image.LANCZOS), dtype=np.uint8)


# --- Contact sheet -------------------------------------------------------------


def contact_sheet(
    episodes: list[recorder.Episode], path: Path, *, tile: tuple[int, int], fps: int
) -> dict[str, Any]:
    """Write the 4x4 contact-sheet video and return what it contains.

    Episodes differ in length, so a tile that has run out holds its last frame:
    freezing is honest (the grasp is over) and keeps every tile in sync with the
    clock rather than with its own index.
    """
    import imageio.v2 as imageio

    columns = int(np.ceil(np.sqrt(len(episodes))))
    rows = int(np.ceil(len(episodes) / columns))
    width, height = tile
    longest = max(len(episode) for episode in episodes)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    try:
        for step in range(longest):
            canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
            for index, episode in enumerate(episodes):
                frame = _tile(episode.frame(min(step, len(episode) - 1)), tile)
                image = Image.fromarray(frame)
                _label(image, f"#{episode.attempt_index}")
                row, column = divmod(index, columns)
                canvas[
                    row * height : (row + 1) * height, column * width : (column + 1) * width
                ] = np.asarray(image)
            writer.append_data(canvas)
    finally:
        writer.close()

    return {
        "path": str(path),
        "episodes": [episode.attempt_index for episode in episodes],
        "grid": [rows, columns],
        "frames": longest,
        "size": [rows * height, columns * width],
        "fps": fps,
        "bytes": path.stat().st_size,
    }


# --- Tracked GIF ----------------------------------------------------------------


def _gif_frames(
    episodes: list[recorder.Episode], panel: tuple[int, int], stride: int
) -> list[Image.Image]:
    """Compose the side-by-side RGB frames of the GIF."""
    width, height = panel
    longest = max(len(episode) for episode in episodes)
    frames = []
    for step in range(0, longest, stride):
        canvas = Image.new("RGB", (width * len(episodes), height))
        for index, episode in enumerate(episodes):
            image = Image.fromarray(
                _tile(episode.frame(min(step, len(episode) - 1)), panel)
            )
            _label(image, f"#{episode.attempt_index}")
            canvas.paste(image, (index * width, 0))
        frames.append(canvas)
    return frames


def _write_gif(frames: list[Image.Image], path: Path, colours: int, fps: int) -> None:
    """Write `frames` as a GIF with one global palette and exact frame timing."""
    # One palette for the whole animation, derived from a vertical strip of
    # evenly spaced frames so it covers the episode's whole colour range rather
    # than just its first frame.
    sample = Image.new("RGB", (frames[0].width, frames[0].height * min(8, len(frames))))
    for slot, index in enumerate(_evenly_spaced(len(frames), min(8, len(frames)))):
        sample.paste(frames[index], (0, slot * frames[0].height))
    palette = sample.convert("P", palette=Image.ADAPTIVE, colors=colours)

    quantised = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(
        path,
        save_all=True,
        append_images=quantised[1:],
        # Milliseconds in, centiseconds on disk: 100 ms is exactly 10 cs, so the
        # written delay is the requested one and the GIF plays at 10 fps.
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )


def tracked_gif(
    episodes: list[recorder.Episode], path: Path, *, fps: int, source_fps: int, max_bytes: int
) -> dict[str, Any]:
    """Write the tracked GIF, shrinking it down the ladder until it fits.

    Returns the description of what was written, including the rung used and
    the frame delay as the file actually records it — the one number a GIF is
    routinely wrong about.
    """
    stride = max(1, round(source_fps / fps))
    attempts = []
    for panel, colours in GIF_LADDER:
        frames = _gif_frames(episodes, panel, stride)
        _write_gif(frames, path, colours, fps)
        size = path.stat().st_size
        attempts.append({"panel": list(panel), "colours": colours, "bytes": size})
        print(
            f"  gif rung {panel[0]}x{panel[1]} x{colours} colours: "
            f"{size / 1e6:.2f} MB ({'fits' if size <= max_bytes else 'too big'})"
        )
        if size <= max_bytes:
            break
    else:
        raise SystemExit(
            f"{path}: still {path.stat().st_size / 1e6:.2f} MB at the smallest rung, "
            f"over the {max_bytes / 1e6:.0f} MB ceiling"
        )

    with Image.open(path) as written:
        recorded_delay = written.info.get("duration")
        frame_count = getattr(written, "n_frames", 1)
        size_hw = [written.height, written.width]
    return {
        "path": str(path),
        "episodes": [episode.attempt_index for episode in episodes],
        "frames": frame_count,
        "size": size_hw,
        "fps": fps,
        "stride": stride,
        "frame_delay_ms": recorded_delay,
        "seconds": frame_count / fps,
        "bytes": path.stat().st_size,
        "ladder": attempts,
    }


# --- Entry point -------------------------------------------------------------------


def is_ignored(path: Path) -> bool | None:
    """Whether git ignores `path`; None when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)], cwd=REPO_ROOT, check=False
        )
    except OSError:
        return None
    return None if result.returncode not in (0, 1) else result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Build both previews for one dataset. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="dataset name, e.g. grasp_cube_dev")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--grid-episodes", type=int, default=GRID_EPISODES)
    parser.add_argument("--gif-episodes", type=int, default=GIF_EPISODES)
    parser.add_argument("--gif-fps", type=int, default=GIF_FPS)
    parser.add_argument(
        "--max-gif-mb", type=float, default=GIF_MAX_BYTES / 1e6, help="tracked GIF ceiling"
    )
    parser.add_argument(
        "--dump-frames",
        type=Path,
        default=None,
        help="write one decoded frame of each artefact here, for eyeballing",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.root / "datasets" / "raw" / args.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"no such dataset: {dataset_dir}")
    paths = success_episodes(dataset_dir)
    if not paths:
        raise SystemExit(f"{dataset_dir}: no successful episodes to preview")
    print(f"{args.dataset}: {len(paths)} successful episodes")

    grid_paths = [paths[index] for index in _evenly_spaced(len(paths), args.grid_episodes)]
    sheet = contact_sheet(
        [recorder.load_episode(path) for path in grid_paths],
        dataset_dir / "preview.mp4",
        tile=GRID_TILE,
        fps=GRID_FPS,
    )
    ignored = is_ignored(Path(sheet["path"]))
    print(
        f"wrote {sheet['path']}: {sheet['grid'][0]}x{sheet['grid'][1]} grid, "
        f"{sheet['frames']} frames, {sheet['bytes'] / 1e6:.1f} MB, "
        f"gitignored={ignored}"
    )
    if ignored is False:
        print("WARN: the contact sheet is NOT gitignored — check the datasets/raw/** rules")

    gif_paths = [paths[index] for index in _evenly_spaced(len(paths), args.gif_episodes)]
    gif = tracked_gif(
        [recorder.load_episode(path) for path in gif_paths],
        args.root / "media" / "datasets" / f"{args.dataset}.gif",
        fps=args.gif_fps,
        source_fps=recorder.CONTROL_HZ,
        max_bytes=int(args.max_gif_mb * 1e6),
    )
    print(
        f"wrote {gif['path']}: {gif['frames']} frames of {gif['size'][1]}x{gif['size'][0]}, "
        f"{gif['frame_delay_ms']} ms/frame = {gif['seconds']:.1f} s, "
        f"{gif['bytes'] / 1e6:.2f} MB, episodes {gif['episodes']}"
    )
    if gif["frame_delay_ms"] != round(1000 / args.gif_fps):
        print(
            f"WARN: the GIF records {gif['frame_delay_ms']} ms a frame, not "
            f"{round(1000 / args.gif_fps)} ms — timing was quantised"
        )

    if args.dump_frames:
        import imageio.v2 as imageio

        args.dump_frames.mkdir(parents=True, exist_ok=True)
        video = imageio.get_reader(sheet["path"])
        try:
            middle = sheet["frames"] // 2
            for index, frame in enumerate(video):
                if index == middle:
                    Image.fromarray(frame).save(args.dump_frames / f"{args.dataset}_sheet.png")
                    break
        finally:
            video.close()
        with Image.open(gif["path"]) as written:
            written.seek(min(gif["frames"] - 1, gif["frames"] // 2))
            written.convert("RGB").save(args.dump_frames / f"{args.dataset}_gif.png")
        print(f"dumped a frame of each into {args.dump_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
