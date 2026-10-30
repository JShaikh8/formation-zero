"""Draw tracked boxes and ids on a take's frames and write an MP4: the milestone 3 check by eye.

Colours say team when teams are known (`team_of`: track id -> 0 or 1), else label. A track keeps
its number, so an id switch shows as a number changing on one player.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEAM_COLOURS = {0: (255, 210, 63), 1: (63, 208, 255)}
LABEL_COLOURS = {"player": (255, 255, 255), "referee": (255, 93, 115)}


def draw(image: np.ndarray, boxes: pd.DataFrame, team_of: dict[int, int] | None = None) -> np.ndarray:
    """One frame with its boxes: rows of track_id, label, x1, y1, x2, y2."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(image)
    d = ImageDraw.Draw(im)
    width = max(2, image.shape[1] // 640)
    font = ImageFont.load_default(size=max(12, image.shape[1] // 64))   # legible after scaling to 1280
    for r in boxes.itertuples():
        team = (team_of or {}).get(int(r.track_id), -1)
        colour = LABEL_COLOURS["referee"] if r.label == "referee" else TEAM_COLOURS.get(team, LABEL_COLOURS["player"])
        d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=colour, width=width)
        x0, y0, x1, y1 = d.textbbox((r.x1, r.y1), str(int(r.track_id)), font=font, anchor="lb")
        d.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(0, 0, 0))
        d.text((r.x1, r.y1), str(int(r.track_id)), fill=colour, font=font, anchor="lb")
    return np.asarray(im)


def render(source, start: int, end: int, boxes: pd.DataFrame, dst: str | Path, *,
           team_of: dict[int, int] | None = None, max_width: int = 1280) -> int:
    """Frames start..end of `source` (a FileSource) with their boxes, scaled to max_width, to dst."""
    from PIL import Image

    from formation_zero.ingest.encode import VideoWriter

    scale = min(1.0, max_width / source.width)
    w, h = int(source.width * scale), int(source.height * scale)
    by_frame = dict(tuple(boxes.groupby("frame"))) if len(boxes) else {}
    with VideoWriter(dst, source.fps, w, h) as out:
        for f in source.iter_frames(start, end):
            img = draw(f.image, by_frame.get(f.index, boxes.iloc[:0]), team_of)
            if scale < 1:
                img = np.asarray(Image.fromarray(img).resize((w, h)))
            out.write(img)
        return out.count


__all__ = ["draw", "render"]
