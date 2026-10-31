"""Approve the teacher's boxes one keystroke per frame (D-015): an HTML contact sheet, then an import.

    fz-review build  --season 2025 --week 20 --away LA --home CHI --detections <teacher.json>
    fz-review import --season 2025 --week 20 --away LA --home CHI --detections <teacher.json> --decisions <export.json>

`build` writes review.html into the game's frames folder on the film drive, beside the sampled
JPEGs, so the page loads them by relative path with no server. Frames come least confident first:
the teacher's weakest frames are where review time pays; once a held-out student has scored
the frames (`--order`), the most disputed come first instead. Keys: k keep, d drop, f flag (look again
with masks), u undo, arrows move, space enlarges the current frame. In the enlarged frame a click
removes one box (and a second click restores it), so a frame that is right but for a bench player
the teacher boxed can still be kept. With `--suggest` (`labelling.field_filter`), boxes whose feet
are off the field start out removed, and a click restores any the field mask got wrong. Decisions live in the browser until Export downloads them.
Frames load as they scroll into view; 2,000 full-size frames at once would exhaust the tab.

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


def frame_order(detections: dict, disputed: dict[str, int] | None = None) -> list[int]:
    """Frames by their least confident box, weakest first; frames with no boxes come first of all.
    With `disputed` (frame -> boxes a held-out student disagreed on, `labelling.train.heldout`),
    the most disputed frames come first and confidence only breaks ties."""
    def weakest(boxes):
        return min((b.get("score", 1.0) for b in boxes), default=-1.0)
    d = disputed or {}
    return sorted((int(f) for f in detections["frames"]),
                  key=lambda f: (-d.get(str(f), 0), weakest(detections["frames"][str(f)]), f))


def build_page(game_key: str, detections: dict, frames: list[int], order: str = "least confident first",
               suggestions: dict[str, list[int]] | None = None) -> str:
    """The self-contained review page: data inline, images by relative path. `suggestions` are box
    indices per frame that start out removed until the reviewer restores them."""
    sug = suggestions or {}
    data = {"game": game_key, "teacher": detections.get("teacher", "?"), "colors": COLORS, "order": order,
            "frames": [{"frame": f, "img": f"{f:06d}.jpg", "boxes": detections["frames"][str(f)],
                        **({"suggest": sug[str(f)]} if str(f) in sug else {})} for f in frames]}
    return _PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))


def import_decisions(game_key: str, detections: dict, decisions: dict, *, fps: float = 25.0) -> dict:
    """Kept frames with their boxes less any the reviewer removed, flagged and dropped frames listed;
    undecided frames left out. `removed` maps a frame to indices into its teacher box list."""
    removed = decisions.get("removed", {})
    kept, flagged, dropped, n_removed = [], [], [], 0
    for f, d in sorted(((int(k), v) for k, v in decisions["decisions"].items())):
        if d == "keep":
            teacher, gone = detections["frames"][str(f)], set(removed.get(str(f), []))
            boxes = [b for j, b in enumerate(teacher) if j not in gone]
            n_removed += len(teacher) - len(boxes)
            kept.append({"frame": f, "time_s": round(f / fps, 3), "boxes": boxes})
        elif d == "flag":
            flagged.append(f)
        elif d == "drop":
            dropped.append(f)
    return {"game_key": game_key, "teacher": detections.get("teacher"), "source": "human review",
            "keyed_by": "source frame number (frame-aligned with every rendition)",
            "reviewed": len(decisions["decisions"]), "boxes_removed": n_removed, "kept": kept, "flagged": flagged,
            "dropped": dropped}


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
    ap.add_argument("--order", help="held-out disagreement JSON: most disputed frames first (build only)")
    ap.add_argument("--suggest", help="field-mask suggestions JSON: those boxes start removed (build only)")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args(argv)

    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home)
    detections = json.loads(Path(a.detections).read_text())
    if a.action == "build":
        page = paths.frames_dir / "review.html"
        missing = [f for f in detections["frames"] if not (paths.frames_dir / f"{int(f):06d}.jpg").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} frames not extracted yet (fz-sample --extract), e.g. {missing[:3]}")
        disputed = json.loads(Path(a.order).read_text())["scores"] if a.order else None
        suggestions = None
        if a.suggest:
            sug = json.loads(Path(a.suggest).read_text())
            if sug["teacher"] != detections.get("teacher"):
                raise SystemExit("suggestions were made for another detections file (box indices would not match)")
            suggestions = sug["suggestions"]
        page.write_text(build_page(paths.game_key, detections, frame_order(detections, disputed),
                                   "most disputed first" if disputed else "least confident first", suggestions))
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
.pic { position:relative; aspect-ratio:16 / 9; background:#000; }
.pic img, .pic svg { position:absolute; inset:0; width:100%; height:100%; display:block; object-fit:contain; }
.pic rect { fill:transparent; stroke-width:2; vector-effect:non-scaling-stroke; pointer-events:none; }
.pic rect.gone { stroke:#fff !important; stroke-width:1; stroke-dasharray:4 4; opacity:.75; }
.cap { padding:4px 8px; color:var(--muted); font-size:12px; display:flex; justify-content:space-between; }
#big { position:fixed; inset:0; background:rgba(0,0,0,.93); display:none; flex-direction:column; z-index:5; }
#big.on { display:flex; }
#big .pic { flex:1; aspect-ratio:auto; background:none; }
#big rect { stroke-width:3; pointer-events:all; cursor:pointer; }
#big .bar { color:#d8ddd6; padding:8px 16px; display:flex; gap:16px; justify-content:space-between; flex-wrap:wrap; }
</style></head><body>
<header><b>Teacher review</b><span id="meta"></span><span id="count"></span>
<span class="keys">k keep · d drop · f flag · u undo · arrows move · space enlarge, then click a box to remove or restore it (dashed = removed)</span><button id="export">Export</button></header>
<div id="grid"></div>
<div id="big"><div class="bar"><span id="bigcap"></span><span>click a box to remove it, again to restore · k keep · d drop · f flag · u undo · space or esc close</span></div></div>
<script>
const D = __DATA__;
const KEY = "fz-review-" + D.game + "-" + D.teacher;
const NS = "http://www.w3.org/2000/svg";
let st = { decisions: {}, removed: {} };
try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); if (s && s.decisions) st = { decisions: s.decisions, removed: s.removed || {} }; } catch (e) {}
const dec = st.decisions, rm = st.removed, hist = [], touched = new Set(Object.keys(rm).map(Number));
for (const f of D.frames) if (f.suggest && !(f.frame in rm) && !(f.frame in dec)) rm[f.frame] = f.suggest.slice();   // off the field: start removed
let cur = Math.max(0, D.frames.findIndex(f => !(f.frame in dec))), bigOn = false;
const grid = document.getElementById("grid"), big = document.getElementById("big");
document.getElementById("meta").textContent = D.game + " · teacher " + D.teacher + " · " + D.frames.length + " frames, " + D.order;
const area = b => (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]);
const at = fr => D.frames.findIndex(f => f.frame === fr);

// A frame with its boxes as SVG over the image: the browser loads and frees the images as they scroll.
function pic(f, eager, onBox) {
  const p = document.createElement("div"); p.className = "pic";
  const img = document.createElement("img"); img.loading = eager ? "eager" : "lazy"; img.decoding = "async"; img.alt = "frame " + f.frame; img.src = f.img;
  const svg = document.createElementNS(NS, "svg"); svg.style.display = "none";   // until the image gives its size
  for (const j of f.boxes.map((b, j) => j).sort((a, b) => area(f.boxes[b]) - area(f.boxes[a]))) {   // small boxes on top, so clickable
    const b = f.boxes[j], r = document.createElementNS(NS, "rect");
    r.setAttribute("x", b.box[0]); r.setAttribute("y", b.box[1]); r.setAttribute("width", b.box[2] - b.box[0]); r.setAttribute("height", b.box[3] - b.box[1]);
    r.style.stroke = D.colors[b.label] || "#fff"; r.dataset.j = j;
    if (onBox) r.addEventListener("click", e => { e.stopPropagation(); onBox(j); });
    svg.append(r);
  }
  img.onload = () => { svg.setAttribute("viewBox", "0 0 " + img.naturalWidth + " " + img.naturalHeight); svg.style.display = ""; };
  p.append(img, svg); mark(p, f.frame); return p;
}
function mark(p, fr) { const gone = new Set(rm[fr] || []); p.querySelectorAll("rect").forEach(r => r.classList.toggle("gone", gone.has(+r.dataset.j))); }
function caption(i) {
  const f = D.frames[i], n = (rm[f.frame] || []).length;
  const low = f.boxes.length ? Math.min(...f.boxes.map(b => b.score ?? 1)) : null;
  return "frame " + f.frame + " · " + f.boxes.length + " boxes" + (n ? " · " + n + " removed" : "") + (low !== null ? " · weakest " + low.toFixed(2) : "");
}
const tiles = D.frames.map((f, i) => {
  const t = document.createElement("div"); t.className = "tile";
  const cap = document.createElement("div"); cap.className = "cap"; cap.textContent = caption(i);
  t.append(pic(f, false, null), cap); grid.append(t);
  t.onclick = () => { cur = i; paint(); };
  t.ondblclick = () => { cur = i; bigOn = true; paint(); };
  return t;
});
function save() {   // removals are saved only for frames looked at, so a rebuilt page's new suggestions reach the rest
  const kept = Object.fromEntries(Object.entries(rm).filter(([fr]) => touched.has(+fr) || fr in dec));
  try { localStorage.setItem(KEY, JSON.stringify({ decisions: dec, removed: kept })); } catch (e) {}
}
function flip(fr, j) {
  const s = new Set(rm[fr] || []); if (s.has(j)) s.delete(j); else s.add(j);
  rm[fr] = [...s].sort((a, b) => a - b); touched.add(fr);
  save(); const i = at(fr); mark(tiles[i], fr); tiles[i].querySelector(".cap").textContent = caption(i);
  if (bigOn && i === cur) { mark(big, fr); document.getElementById("bigcap").textContent = caption(i); }
}
function paint() {
  tiles.forEach((t, i) => { const d = dec[D.frames[i].frame]; t.className = "tile" + (d ? " " + d : "") + (i === cur ? " cur" : ""); });
  const n = Object.keys(dec).length; document.getElementById("count").textContent = n + " / " + D.frames.length + " decided";
  big.classList.toggle("on", bigOn);
  const old = big.querySelector(".pic"); if (old) old.remove();
  if (bigOn) {
    const f = D.frames[cur];
    big.prepend(pic(f, true, j => { hist.push({ fr: f.frame, box: j }); flip(f.frame, j); }));
    document.getElementById("bigcap").textContent = (dec[f.frame] ? dec[f.frame] + " · " : "") + caption(cur);
  } else tiles[cur].scrollIntoView({ block: "nearest" });
}
const NAMES = { k: "keep", d: "drop", f: "flag" };
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const cols = bigOn ? 1 : Math.max(1, Math.round(grid.clientWidth / tiles[0].clientWidth));
  if (NAMES[e.key]) { const fr = D.frames[cur].frame; hist.push({ fr, prev: dec[fr] || null }); dec[fr] = NAMES[e.key]; save(); cur = Math.min(D.frames.length - 1, cur + 1); }
  else if (e.key === "u" && hist.length) { const h = hist.pop(); if ("box" in h) flip(h.fr, h.box); else { if (h.prev) dec[h.fr] = h.prev; else delete dec[h.fr]; save(); } cur = at(h.fr); }
  else if (e.key === " " || e.key === "Enter") bigOn = !bigOn;
  else if (e.key === "Escape" && bigOn) bigOn = false;
  else if (e.key === "ArrowRight") cur = Math.min(D.frames.length - 1, cur + 1);
  else if (e.key === "ArrowLeft") cur = Math.max(0, cur - 1);
  else if (e.key === "ArrowDown") cur = Math.min(D.frames.length - 1, cur + cols);
  else if (e.key === "ArrowUp") cur = Math.max(0, cur - cols);
  else return;
  e.preventDefault(); paint();
});
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify({ game: D.game, teacher: D.teacher, decisions: dec, removed: rm }, null, 1)], { type: "application/json" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = D.game + ".review.json"; a.click();
};
paint();
</script></body></html>
"""

__all__ = ["COLORS", "DECISIONS", "build_page", "frame_order", "import_decisions"]


if __name__ == "__main__":
    raise SystemExit(main())
