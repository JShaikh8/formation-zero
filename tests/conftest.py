"""Shared fixtures: synthetic video so ingestion tests never need real film."""

from __future__ import annotations

import numpy as np
import pytest


def synthetic_frames(n: int, width: int = 160, height: int = 96, *, green: bool = True, seed_index: int = 0,
                     kind: str | None = None):
    """Frames of three kinds, so a decoded frame can be identified and a recording can be faked.

    film   (green=True): bright turf with white yard lines and a square that moves every frame.
    menu   (green=False): dim, mostly static, with a big jump every 3 seconds (a scroll).
    frozen: one bright frame repeated.
    """
    kind = kind or ("film" if green else "menu")
    rng = np.random.default_rng(seed_index)
    base_menu = np.full((height, width, 3), 40, dtype=np.uint8)
    for i in range(n):
        k = seed_index + i
        if kind in ("film", "film2"):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            if kind == "film":
                img[..., 0] = 90; img[..., 1] = 160; img[..., 2] = 90
            else:                                   # a different take: same brightness band, other colours
                img[..., 0] = 175; img[..., 1] = 175; img[..., 2] = 120
            for x in range(0, width, 20):
                img[:, x:x + 2, :] = 235
            x0 = (k * 7) % max(1, width - 12); y0 = (k * 3) % max(1, height - 12)
            img[y0:y0 + 12, x0:x0 + 12, :] = 255
            img = (img.astype(np.int16) + rng.integers(-6, 7, img.shape)).clip(0, 255).astype(np.uint8)
        elif kind == "frozen":
            img = np.full((height, width, 3), 130, dtype=np.uint8)
        else:
            img = base_menu.copy()
            if (i // 75) % 2 == 1:          # every 3 s at 25 fps the "page" changes
                img[...] = 70
            img[10:22, 10:22, :] = 120
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
                for seg in segments:
                    count, green = seg[0], seg[1]
                    kind = seg[2] if len(seg) > 2 else None
                    yield from synthetic_frames(count, width, height, green=green, seed_index=offset, kind=kind)
                    offset += count
            else:
                yield from synthetic_frames(n, width, height)
        path = tmp_path / name
        total = write_frames(path, gen(), fps)
        return path, {"frames": total, "fps": fps, "width": width, "height": height}
    return _make
