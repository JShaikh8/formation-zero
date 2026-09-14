"""Write video from frames: the one encoder used by the proxy and the clip cutter.

H.264 with a keyframe every `gop` frames so seeking inside the output is cheap and exact. The
output keeps the source fps, so frame numbers stay aligned between a proxy and its source.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Iterable

import numpy as np


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int, *,
                 gop: int | None = None, crf: int = 23, preset: str = "veryfast"):
        import av

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._container = av.open(str(self.path), mode="w")
        rate = Fraction(fps).limit_denominator(1000)
        self._stream = self._container.add_stream("libx264", rate=rate)
        self._stream.width = int(width) - (int(width) % 2)     # x264 needs even dimensions
        self._stream.height = int(height) - (int(height) % 2)
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {"crf": str(crf), "preset": preset, "g": str(gop or max(1, int(round(fps)))),
                                "keyint_min": str(gop or max(1, int(round(fps)))), "sc_threshold": "0"}
        self.count = 0

    def write(self, image: np.ndarray) -> None:
        import av

        h, w = self._stream.height, self._stream.width
        if image.shape[0] != h or image.shape[1] != w:
            image = image[:h, :w]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.count += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_frames(path: str | Path, frames: Iterable[np.ndarray], fps: float, **kw) -> int:
    it = iter(frames)
    first = next(it, None)
    if first is None:
        raise ValueError("no frames to write")
    with VideoWriter(path, fps, first.shape[1], first.shape[0], **kw) as w:
        w.write(first)
        for img in it:
            w.write(img)
        return w.count


__all__ = ["VideoWriter", "write_frames"]
