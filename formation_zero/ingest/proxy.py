"""Make a small, cheap, frame-aligned copy of a film file.

    fz-proxy source.mp4 proxy.mp4 --height 540

Every cheap pass (shots, film window, snap search, registration search) runs on the proxy; an
index found there addresses the same frame in the full-resolution source because the fps and
frame count are identical. A keyframe every second makes seeking inside it exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from formation_zero.ingest.encode import VideoWriter
from formation_zero.ingest.sources import FileSource


def _resize_rgb(image: np.ndarray, height: int) -> np.ndarray:
    """Nearest-neighbour downscale without OpenCV: exact enough for a proxy, dependency-free."""
    h, w = image.shape[:2]
    if h <= height:
        return image
    scale = height / h
    new_w = max(2, int(round(w * scale)))
    ys = (np.arange(height) / scale).astype(int).clip(0, h - 1)
    xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
    return image[ys][:, xs]


def make_proxy(src: str | Path, dst: str | Path, *, height: int = 540, start: int = 0,
               end: int | None = None) -> dict:
    """Encode `src` frames [start, end) into `dst` at `height` pixels tall. Returns a summary."""
    with FileSource(src) as source:
        writer = None
        n = 0
        for frame in source.iter_frames(start, end):
            small = _resize_rgb(frame.image, height)
            if writer is None:
                writer = VideoWriter(dst, source.fps, small.shape[1], small.shape[0])
            writer.write(small)
            n += 1
        if writer is not None:
            writer.close()
        return {"src": str(src), "dst": str(dst), "fps": source.fps, "frames": n,
                "height": height, "first_frame": start}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--height", type=int, default=540)
    a = ap.parse_args(argv)
    r = make_proxy(a.src, a.dst, height=a.height)
    print(f"{r['frames']} frames at {r['fps']:g} fps, {r['height']}p -> {r['dst']}")
    return 0


__all__ = ["make_proxy"]

if __name__ == "__main__":
    raise SystemExit(main())
