"""Shared fixtures: synthetic video so ingestion tests never need real film."""

from __future__ import annotations

import numpy as np
import pytest


def synthetic_frames(n: int, width: int = 160, height: int = 96, *, green: bool = True, seed_index: int = 0):
    """Frames whose content encodes the frame index, so a decoded frame can be identified.

    Green frames look like a field (turf plus white yard lines); grey ones look like a menu.
    A moving white square encodes the index by position; a brightness step encodes it too.
    """
    for i in range(n):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        if green:
            img[..., 1] = 120
            img[..., 0] = 40
            img[..., 2] = 40
            for x in range(0, width, 20):
                img[:, x:x + 2, :] = 230
        else:
            img[...] = 90
        k = seed_index + i
        x0 = (k * 7) % max(1, width - 12)
        y0 = (k * 3) % max(1, height - 12)
        img[y0:y0 + 12, x0:x0 + 12, :] = 255
        yield img


@pytest.fixture
def make_video(tmp_path):
    """Factory: write a synthetic mp4 and return its path plus the parameters used."""
    def _make(name: str = "clip.mp4", *, n: int = 50, fps: float = 25.0, width: int = 160, height: int = 96,
              segments: list[tuple[int, bool]] | None = None):
        from formation_zero.ingest.encode import write_frames

        def gen():
            if segments:
                offset = 0
                for count, green in segments:
                    yield from synthetic_frames(count, width, height, green=green, seed_index=offset)
                    offset += count
            else:
                yield from synthetic_frames(n, width, height)
        path = tmp_path / name
        total = write_frames(path, gen(), fps)
        return path, {"frames": total, "fps": fps, "width": width, "height": height}
    return _make
