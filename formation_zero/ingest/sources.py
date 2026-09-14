"""Frame sources: one interface, two implementations (file, stream).

A `FrameSource` yields `Frame`s in order. `FileSource` knows its length and can start at any
frame; `StreamSource` has no end and starts at "now". Nothing downstream should branch on which
it has, which is the whole point: the live pass (blueprint 11.2) is the same pipeline fed by a
stream.

Frames are RGB uint8 arrays of shape (height, width, 3). Frame indices are the media's own frame
numbers, so an index found on a proxy rendition addresses the same frame in the source file.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One decoded frame."""

    index: int            # media frame number, 0-based
    time_s: float         # presentation time in seconds
    image: np.ndarray     # RGB, (h, w, 3), uint8

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@runtime_checkable
class FrameSource(Protocol):
    """What every stage reads from."""

    fps: float
    width: int
    height: int
    frames: int | None    # None for a stream

    def iter_frames(self, start: int = 0, end: int | None = None, step: int = 1) -> Iterator[Frame]:
        """Yield frames with index in [start, end), every `step`-th one. End is ignored by streams."""
        ...

    def close(self) -> None: ...


def _open_container(path):
    import av  # optional [video] dependency, kept off the module import path

    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    return container, stream


class FileSource:
    """A video file with a known length, decoded sequentially from any start frame.

    Seeks to the nearest keyframe before `start` and discards frames up to it, which is far
    faster than seeking per frame and keeps decode order exact.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._container, self._stream = _open_container(self.path)
        rate = self._stream.average_rate or self._stream.guessed_rate or Fraction(25, 1)
        self.fps = float(rate)
        self.width = int(self._stream.codec_context.width)
        self.height = int(self._stream.codec_context.height)
        self.frames: int | None = int(self._stream.frames) or None
        if self.frames is None and self._stream.duration is not None:
            self.frames = int(round(float(self._stream.duration * self._stream.time_base) * self.fps))
        self._time_base = self._stream.time_base

    def iter_frames(self, start: int = 0, end: int | None = None, step: int = 1) -> Iterator[Frame]:
        if step < 1:
            raise ValueError("step must be >= 1")
        if start > 0:
            self._container.seek(int(start / self.fps / self._time_base), stream=self._stream, backward=True)
        else:
            self._container.seek(0, stream=self._stream, backward=True)
        for packet_frame in self._container.decode(self._stream):
            if packet_frame.pts is None:
                continue
            time_s = float(packet_frame.pts * self._time_base)
            index = int(round(time_s * self.fps))
            if index < start:
                continue
            if end is not None and index >= end:
                break
            if (index - start) % step:
                continue
            yield Frame(index=index, time_s=time_s, image=packet_frame.to_ndarray(format="rgb24"))

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> "FileSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class StreamSource:
    """A live source (RTSP/HTTP URL or capture device). No length, no seeking, starts at now.

    Frame indices count from the first frame read, so a live play record's frames are relative
    to when the pipeline started listening; the post pass re-anchors them to the film.
    """

    def __init__(self, url: str, fps_hint: float | None = None):
        self.url = url
        self._container, self._stream = _open_container(url)
        rate = self._stream.average_rate or self._stream.guessed_rate
        self.fps = float(rate) if rate else float(fps_hint or 25.0)
        self.width = int(self._stream.codec_context.width)
        self.height = int(self._stream.codec_context.height)
        self.frames = None
        self._time_base = self._stream.time_base

    def iter_frames(self, start: int = 0, end: int | None = None, step: int = 1) -> Iterator[Frame]:
        index = -1
        t0 = None
        for packet_frame in self._container.decode(self._stream):
            index += 1
            if index < start or (index - start) % step:
                continue
            pts = float(packet_frame.pts * self._time_base) if packet_frame.pts is not None else index / self.fps
            t0 = pts if t0 is None else t0
            yield Frame(index=index, time_s=pts - t0, image=packet_frame.to_ndarray(format="rgb24"))

    def close(self) -> None:
        self._container.close()


def open_source(ref: str | Path, *, fps_hint: float | None = None) -> FrameSource:
    """A file path becomes a FileSource; anything with a scheme (rtsp://, http://) a StreamSource."""
    s = str(ref)
    if "://" in s and not s.startswith("file://"):
        return StreamSource(s, fps_hint=fps_hint)
    return FileSource(s.removeprefix("file://"))


__all__ = ["Frame", "FrameSource", "FileSource", "StreamSource", "open_source"]
