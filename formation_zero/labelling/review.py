"""Approve the teacher's boxes one keystroke per frame (D-015): an HTML contact sheet, then an import.

    fz-review build  --season 2025 --week 20 --away LA --home CHI --detections <teacher.json>
    fz-review import --season 2025 --week 20 --away LA --home CHI --detections <teacher.json> --decisions <export.json>

`build` writes review.html into the game's frames folder on the film drive, beside the sampled
JPEGs, so the page loads them by relative path with no server. Frames come least confident first:
the teacher's weakest frames are where review time pays. Keys: k keep, d drop, f flag (look again
with masks), u undo, arrows move. Decisions live in the browser until Export downloads them.

`import` writes the kept frames' boxes to data/groundtruth/<game>.detections.json, keyed by source
frame number and time like the other ground truth, and lists the flagged frames for a second look.

Teacher detections are teacher-agnostic JSON:
    {"teacher": "sam3", "frames": {"<frame>": [{"label": "player", "box": [x1, y1, x2, y2], "score": 0.9}, ...]}}
with boxes in pixels of the extracted frame.
"""

from __future__ import annotations

import json
from pathlib import Path

DECISIONS = {"k": "keep", "d": "drop", "f": "flag"}
COLORS = {"player": "#ffd23f", "referee": "#ff5d73", "football": "#3fd0ff"}


def frame_order(detections: dict) -> list[int]:
    """Frames by their least confident box, weakest first; frames with no boxes come first of all."""
    def weakest(boxes):
        return min((b.get("score", 1.0) for b in boxes), default=-1.0)
    return sorted((int(f) for f in detections["frames"]), key=lambda f: (weakest(detections["frames"][str(f)]), f))


def build_page(game_key: str, detections: dict, frames: list[int]) -> str:
    """The self-contained review page: data inline, images by relative path."""
    data = {"game": game_key, "teacher": detections.get("teacher", "?"), "colors": COLORS,
            "frames": [{"frame": f, "img": f"{f:06d}.jpg", "boxes": detections["frames"][str(f)]} for f in frames]}
    return _PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def import_decisions(game_key: str, detections: dict, decisions: dict, *, fps: float = 25.0) -> dict:
    """Kept frames with their boxes, flagged frames listed; dropped and undecided frames left out."""
    kept, flagged = [], []
    for f, d in sorted(((int(k), v) for k, v in decisions["decisions"].items())):
        if d == "keep":
            kept.append({"frame": f, "time_s": round(f / fps, 3), "boxes": detections["frames"][str(f)]})
        elif d == "flag":
            flagged.append(f)
    return {"game_key": game_key, "teacher": detections.get("teacher"), "source": "human review",
            "keyed_by": "source frame number (frame-aligned with every rendition)",
            "reviewed": len(decisions["decisions"]), "kept": kept, "flagged": flagged}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from formation_zero.data.layout import GamePaths

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["build", "import"])
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int, required=True)
    ap.add_argument("--away", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--detections", required=True, help="teacher detections JSON")
    ap.add_argument("--decisions", help="the file Export downloaded (import only)")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)

    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    detections = json.loads(Path(a.detections).read_text())
    if a.action == "build":
        page = paths.frames_dir / "review.html"
        missing = [f for f in detections["frames"] if not (paths.frames_dir / f"{int(f):06d}.jpg").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} frames not extracted yet (fz-sample --extract), e.g. {missing[:3]}")
        page.write_text(build_page(paths.game_key, detections, frame_order(detections)))
        print(f"{len(detections['frames'])} frames -> {page}")
    else:
        if not a.decisions:
            raise SystemExit("--decisions is required for import")
        out = paths.groundtruth_dir / f"{paths.game_key}.detections.json"
        result = import_decisions(paths.game_key, detections, json.loads(Path(a.decisions).read_text()))
        out.write_text(json.dumps(result, indent=1))
        print(f"{len(result['kept'])} kept, {len(result['flagged'])} flagged -> {out}")
    return 0


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Teacher Review</title>
<style>
:root { --bg:#f4f5f1; --ink:#16211b; --muted:#6b7a70; --card:#fff; --line:#d8ddd6; --keep:#2f9e57; --drop:#c8483f; --flag:#d9a21b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#121714; --ink:#e7ece8; --muted:#93a198; --card:#1b221d; --line:#2c362f; } }
:root[data-theme="dark"] { --bg:#121714; --ink:#e7ece8; --muted:#93a198; --card:#1b221d; --line:#2c362f; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 -apple-system, system-ui, sans-serif; }
header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; z-index:2; }
header b { font-size:16px; } .keys { color:var(--muted); } button { font:inherit; padding:4px 10px; }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:10px; padding:12px 16px; }
.tile { background:var(--card); border:3px solid var(--line); border-radius:6px; overflow:hidden; }
.tile.cur { border-color:var(--ink); } .tile.keep { border-color:var(--keep); } .tile.drop { border-color:var(--drop); opacity:.55; } .tile.flag { border-color:var(--flag); }
.tile.cur.keep, .tile.cur.drop, .tile.cur.flag { outline:3px solid var(--ink); }
canvas { width:100%; display:block; } .cap { padding:4px 8px; color:var(--muted); font-size:12px; display:flex; justify-content:space-between; }
</style></head><body>
<header><b>Teacher review</b><span id="meta"></span><span id="count"></span>
<span class="keys">k keep · d drop · f flag · u undo · arrows move</span><button id="export">Export</button></header>
<div id="grid"></div>
<script>
const D = __DATA__;
const KEY = "fz-review-" + D.game + "-" + D.teacher;
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const hist = []; let cur = Math.max(0, D.frames.findIndex(f => !(f.frame in dec)));
const grid = document.getElementById("grid");
document.getElementById("meta").textContent = D.game + " · teacher " + D.teacher + " · " + D.frames.length + " frames, least confident first";
const tiles = D.frames.map((f, i) => {
  const t = document.createElement("div"); t.className = "tile";
  const c = document.createElement("canvas"); const cap = document.createElement("div"); cap.className = "cap";
  const low = f.boxes.length ? Math.min(...f.boxes.map(b => b.score ?? 1)) : null;
  cap.innerHTML = "<span>frame " + f.frame + "</span><span>" + f.boxes.length + " boxes" + (low !== null ? " · weakest " + low.toFixed(2) : "") + "</span>";
  t.append(c, cap); grid.append(t); t.onclick = () => { cur = i; paint(); };
  const img = new Image(); img.loading = "lazy"; img.src = f.img;
  img.onload = () => { c.width = img.naturalWidth; c.height = img.naturalHeight; const g = c.getContext("2d"); g.drawImage(img, 0, 0);
    g.lineWidth = Math.max(2, img.naturalWidth / 640);
    for (const b of f.boxes) { g.strokeStyle = D.colors[b.label] || "#fff"; const [x1, y1, x2, y2] = b.box; g.strokeRect(x1, y1, x2 - x1, y2 - y1); } };
  return t; });
function save() { try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) {} }
function paint() {
  tiles.forEach((t, i) => { const d = dec[D.frames[i].frame]; t.className = "tile" + (d ? " " + d : "") + (i === cur ? " cur" : ""); });
  tiles[cur].scrollIntoView({ block: "nearest" });
  const n = Object.keys(dec).length; document.getElementById("count").textContent = n + " / " + D.frames.length + " decided";
}
const NAMES = { k: "keep", d: "drop", f: "flag" };
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey) return;
  const cols = Math.max(1, Math.round(grid.clientWidth / tiles[0].clientWidth));
  if (NAMES[e.key]) { const fr = D.frames[cur].frame; hist.push([fr, dec[fr]]); dec[fr] = NAMES[e.key]; save(); cur = Math.min(D.frames.length - 1, cur + 1); }
  else if (e.key === "u" && hist.length) { const [fr, prev] = hist.pop(); if (prev) dec[fr] = prev; else delete dec[fr]; save(); cur = D.frames.findIndex(f => f.frame === fr); }
  else if (e.key === "ArrowRight") cur = Math.min(D.frames.length - 1, cur + 1);
  else if (e.key === "ArrowLeft") cur = Math.max(0, cur - 1);
  else if (e.key === "ArrowDown") cur = Math.min(D.frames.length - 1, cur + cols);
  else if (e.key === "ArrowUp") cur = Math.max(0, cur - cols);
  else return;
  e.preventDefault(); paint();
});
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify({ game: D.game, teacher: D.teacher, decisions: dec }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = D.game + ".review.json"; a.click();
};
paint();
</script></body></html>
"""

__all__ = ["COLORS", "DECISIONS", "build_page", "frame_order", "import_decisions"]


if __name__ == "__main__":
    raise SystemExit(main())
