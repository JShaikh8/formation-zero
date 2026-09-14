"""Probe a video once and write what every later stage needs beside it.

    fz-probe data/raw/2025/wk20/LA_at_CHI/source/film.mp4

writes `film.mp4.probe.json` with fps, frame count, duration, size, codec and the keyframe
times, so no stage re-opens the file just to ask.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    path: str
    fps: float
    frames: int | None
    duration_s: float | None
    width: int
    height: int
    codec: str
    keyframes_s: list[float]

    def sidecar(self) -> Path:
        return Path(self.path).with_name(Path(self.path).name + ".probe.json")


def probe(path: str | Path, *, keyframes: bool = True) -> Probe:
    import av

    path = Path(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 25)
        tb = stream.time_base
        duration = float(stream.duration * tb) if stream.duration is not None else (
            float(container.duration / 1_000_000) if container.duration else None)
        frames = int(stream.frames) or (int(round(duration * fps)) if duration else None)
        keys: list[float] = []
        if keyframes:
            for packet in container.demux(stream):
                if packet.is_keyframe and packet.pts is not None:
                    keys.append(round(float(packet.pts * tb), 3))
        return Probe(path=str(path), fps=fps, frames=frames, duration_s=duration,
                     width=int(stream.codec_context.width), height=int(stream.codec_context.height),
                     codec=stream.codec_context.name, keyframes_s=keys)


def write_sidecar(p: Probe) -> Path:
    out = p.sidecar()
    out.write_text(json.dumps(asdict(p), indent=1))
    return out


def read_sidecar(path: str | Path) -> Probe | None:
    side = Path(path).with_name(Path(path).name + ".probe.json")
    if not side.exists():
        return None
    return Probe(**json.loads(side.read_text()))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--no-keyframes", action="store_true", help="skip the keyframe scan (fast)")
    a = ap.parse_args(argv)
    p = probe(a.video, keyframes=not a.no_keyframes)
    out = write_sidecar(p)
    print(f"{p.width}x{p.height} {p.codec} {p.fps:g} fps, {p.frames} frames, "
          f"{(p.duration_s or 0) / 60:.1f} min, {len(p.keyframes_s)} keyframes -> {out}")
    return 0


__all__ = ["Probe", "probe", "write_sidecar", "read_sidecar"]

if __name__ == "__main__":
    raise SystemExit(main())
