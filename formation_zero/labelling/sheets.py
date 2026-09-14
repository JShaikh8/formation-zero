"""Contact sheets: many thumbnails on one image, each with an id, for one-glance review.

    fz-sheets views --season 2026 --week 1 --away SF --home LA      # one thumbnail per take

Writes data/derived/review/<game_key>/views_NN.png. Labels are written back as the existing
time-keyed ground-truth format (data/groundtruth/<game_key>.shot_views.json), so nothing
downstream changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def thumb(image: np.ndarray, w: int, h: int) -> np.ndarray:
    H, W = image.shape[:2]
    ys = (np.arange(h) * H / h).astype(int).clip(0, H - 1)
    xs = (np.arange(w) * W / w).astype(int).clip(0, W - 1)
    return image[ys][:, xs]


def sheet(tiles: list[tuple[str, np.ndarray]], cols: int = 6, tw: int = 213, th: int = 120) -> "Image":
    from PIL import Image, ImageDraw

    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for k, (_, img) in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb(img, tw, th)
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    for k, (label, _) in enumerate(tiles):
        r, c = divmod(k, cols)
        x, y = c * tw + 3, r * th + 2
        d.rectangle([x - 2, y - 1, x + 8 * len(label) + 4, y + 12], fill=(0, 0, 0))
        d.text((x, y), label, fill=(255, 230, 80))
    return im


def views_sheets(paths, *, per_sheet: int = 30, cols: int = 6) -> list[Path]:
    """One mid-take thumbnail per camera take, 30 per sheet."""
    import pandas as pd

    from formation_zero.ingest.sources import FileSource

    proxy = paths.proxy_path()
    meta = json.loads(proxy.with_name(proxy.name + ".json").read_text())
    off = int(meta.get("first_frame", 0))
    shots = pd.read_parquet(paths.shots_path)
    out_dir = paths.root / "derived" / "review" / paths.game_key
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with FileSource(proxy) as src:
        for start in range(0, len(shots), per_sheet):
            chunk = shots.iloc[start:start + per_sheet]
            tiles = []
            for r in chunk.itertuples():
                mid = int((r.start_frame + r.end_frame) // 2) - off
                fr = next(src.iter_frames(max(0, mid), max(0, mid) + 1), None)
                if fr is None:
                    continue
                tiles.append((f"{r.shot_index}", fr.image))
            p = out_dir / f"views_{start // per_sheet:02d}.png"
            sheet(tiles, cols=cols).save(p)
            written.append(p)
    return written


def write_view_labels(paths, labels: dict[int, str], *, source: str = "human") -> Path:
    """{shot_index: 'SIDELINE'|'ENDZONE'} -> the time-keyed ground-truth file (merging with any existing)."""
    import pandas as pd

    shots = pd.read_parquet(paths.shots_path).set_index("shot_index")
    out = paths.groundtruth_dir / f"{paths.game_key}.shot_views.json"
    existing = json.loads(out.read_text()) if out.exists() else {"game_key": paths.game_key, "labels": []}
    keep = [l for l in existing["labels"] if l.get("shot_index") not in labels]
    for idx, view in sorted(labels.items()):
        r = shots.loc[idx]
        keep.append({"shot_index": int(idx), "start_s": float(r.start_s), "end_s": float(r.end_s),
                     "view": view, "confident": True, "source": source})
    existing["labels"] = sorted(keep, key=lambda l: l["start_s"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(existing, indent=1))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=["views"])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True); ap.add_argument("--home", required=True)
    ap.add_argument("--data-root", default="data"); ap.add_argument("--per-sheet", type=int, default=30)
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    files = views_sheets(paths, per_sheet=a.per_sheet)
    print(f"{len(files)} sheets -> {files[0].parent}")
    return 0


__all__ = ["thumb", "sheet", "views_sheets", "write_view_labels"]

if __name__ == "__main__":
    raise SystemExit(main())
