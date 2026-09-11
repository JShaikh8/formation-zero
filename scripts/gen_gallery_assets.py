"""Generate the tracker gallery's SVG images from real project data.

Three images, written to site/assets/:

    field.svg            the NFL field model, drawn from gridiron.configs.nfl_field
    shots-histogram.svg  take durations and the tilt signal for the demo game's 419 shots
    play-timeline.svg    the 184 plays of the demo game on the film's clock

Pure Python: no matplotlib. Each SVG paints its own paper background and uses explicit fills,
so it reads the same inside a light or dark page. Run from the repo root with the project venv:

    .venv/bin/python scripts/gen_gallery_assets.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from gridiron.configs.nfl_field import (
    END_ZONE_DEPTH_YD,
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    HASH_BOTTOM_Y,
    HASH_TOP_Y,
    absolute_x_to_yard_number,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site" / "assets"
SHOTS = ROOT / "data" / "derived" / "shots" / "2025_wk20_LA-CHI.parquet"
PLAYS = ROOT / "data" / "derived" / "shots" / "2025_wk20_LA-CHI.plays.parquet"

PAPER = "#F3F5F0"
INK = "#16211B"
MUTED = "#6B7A70"
RULE = "#D9DFD8"
TURF = "#2E7D4F"
TURF_DARK = "#1F5A39"
CHALK = "#F4F6F2"
BLUE = "#2C5BD1"
YELLOW = "#F2C94C"
FONT = "'Barlow Condensed', 'Arial Narrow', 'Helvetica Neue', Arial, sans-serif"
BODY = "'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif"
MONO = "'IBM Plex Mono', Menlo, Consolas, monospace"


def svg_open(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{title}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{PAPER}"/>',
    ]


def text(x: float, y: float, s: str, *, size: int = 14, fill: str = INK, anchor: str = "start",
         family: str = BODY, weight: str = "400", extra: str = "") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{s}</text>')


# ------------------------------------------------------------------ field ---------------

def field_svg() -> str:
    scale = 9.0                      # px per yard
    pad = 30
    w = int(FIELD_LENGTH_YD * scale) + 2 * pad
    h = int(FIELD_WIDTH_YD * scale) + 2 * pad + 44
    top = pad + 44
    parts = svg_open(w, h, "NFL field model in yards")
    parts.append(text(pad, 30, "NFL field model", size=26, family=FONT, weight="700"))
    parts.append(text(pad + 175, 30, "every landmark in real yards, drawn from the code",
                      size=13, fill=MUTED))

    def fx(x_yd: float) -> float:
        return pad + x_yd * scale

    def fy(y_yd: float) -> float:
        # field y runs bottom sideline -> top; SVG y runs down, so flip.
        return top + (FIELD_WIDTH_YD - y_yd) * scale

    parts.append(f'<rect x="{fx(0):.1f}" y="{fy(FIELD_WIDTH_YD):.1f}" '
                 f'width="{FIELD_LENGTH_YD * scale:.1f}" height="{FIELD_WIDTH_YD * scale:.1f}" '
                 f'fill="{TURF}"/>')
    for x0 in (0.0, FIELD_LENGTH_YD - END_ZONE_DEPTH_YD):
        parts.append(f'<rect x="{fx(x0):.1f}" y="{fy(FIELD_WIDTH_YD):.1f}" '
                     f'width="{END_ZONE_DEPTH_YD * scale:.1f}" '
                     f'height="{FIELD_WIDTH_YD * scale:.1f}" fill="{TURF_DARK}"/>')
    # Yard lines every 5, goal lines heavier.
    x = END_ZONE_DEPTH_YD
    while x <= FIELD_LENGTH_YD - END_ZONE_DEPTH_YD + 1e-9:
        heavy = x in (END_ZONE_DEPTH_YD, FIELD_LENGTH_YD - END_ZONE_DEPTH_YD)
        parts.append(f'<line x1="{fx(x):.1f}" y1="{fy(0):.1f}" x2="{fx(x):.1f}" '
                     f'y2="{fy(FIELD_WIDTH_YD):.1f}" stroke="{CHALK}" '
                     f'stroke-width="{2.4 if heavy else 1.2}"/>')
        x += 5
    # Hash ticks every yard along both hash rows.
    for yd in range(int(END_ZONE_DEPTH_YD) + 1, int(FIELD_LENGTH_YD - END_ZONE_DEPTH_YD)):
        for hy in (HASH_BOTTOM_Y, HASH_TOP_Y):
            parts.append(f'<line x1="{fx(yd):.1f}" y1="{fy(hy) - 3:.1f}" x2="{fx(yd):.1f}" '
                         f'y2="{fy(hy) + 3:.1f}" stroke="{CHALK}" stroke-width="1" opacity="0.9"/>')
    # Yard numbers every 10, both sidelines, as painted.
    for yd in range(20, 101, 10):
        n = absolute_x_to_yard_number(float(yd))
        label = f"{n:02d}" if n else "G"
        parts.append(text(fx(yd), fy(8), label, size=22, fill=CHALK, anchor="middle",
                          family=FONT, weight="700"))
        parts.append(text(fx(yd), fy(FIELD_WIDTH_YD - 8) + 8, label, size=22, fill=CHALK,
                          anchor="middle", family=FONT, weight="700",
                          extra=f'transform="rotate(180 {fx(yd):.1f} {fy(FIELD_WIDTH_YD - 8) + 4:.1f})"'))
    # Dimension callouts.
    parts.append(text(fx(60), fy(FIELD_WIDTH_YD) - 8, "120 yd including two 10-yd end zones",
                      size=12, fill=MUTED, anchor="middle", family=MONO))
    parts.append(text(fx(FIELD_LENGTH_YD) + 6, fy(FIELD_WIDTH_YD / 2) + 4, "53.3 yd",
                      size=12, fill=MUTED, family=MONO,
                      extra=f'transform="rotate(90 {fx(FIELD_LENGTH_YD) + 6:.1f} {fy(FIELD_WIDTH_YD / 2) + 4:.1f})"'))
    parts.append(text(fx(112), fy(HASH_TOP_Y) - 6, "hash 23.58 yd from sideline",
                      size=11, fill=CHALK, anchor="end", family=MONO))
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------ histograms ----------

def histogram_panel(parts: list[str], values, *, x0: float, y0: float, w: float, h: float,
                    lo: float, hi: float, step: float, title: str, unit: str,
                    color: str) -> None:
    bins = int(math.ceil((hi - lo) / step))
    counts = [0] * bins
    for v in values:
        if v != v:
            continue
        i = min(bins - 1, max(0, int((v - lo) / step)))
        counts[i] += 1
    peak = max(counts) or 1
    parts.append(text(x0, y0 - 12, title, size=18, family=FONT, weight="700"))
    plot_top = y0 + 6
    plot_h = h - 40
    base = plot_top + plot_h
    # faint grid + y labels at 4 levels
    for k in range(0, 5):
        yy = base - plot_h * k / 4
        parts.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x0 + w:.1f}" y2="{yy:.1f}" '
                     f'stroke="{RULE}" stroke-width="1"/>')
        parts.append(text(x0 - 6, yy + 4, f"{int(round(peak * k / 4))}", size=11, fill=MUTED,
                          anchor="end", family=MONO))
    bw = w / bins
    for i, c in enumerate(counts):
        bh = plot_h * c / peak
        parts.append(f'<rect x="{x0 + i * bw + 1:.1f}" y="{base - bh:.1f}" '
                     f'width="{max(bw - 2, 1):.1f}" height="{bh:.1f}" fill="{color}"/>')
    # x labels every few bins
    every = max(1, bins // 6)
    for i in range(0, bins + 1, every):
        xx = x0 + i * bw
        parts.append(text(xx, base + 16, f"{lo + i * step:g}", size=11, fill=MUTED,
                          anchor="middle", family=MONO))
    parts.append(text(x0 + w, base + 32, unit, size=11, fill=MUTED, anchor="end"))
    parts.append(f'<line x1="{x0:.1f}" y1="{base:.1f}" x2="{x0 + w:.1f}" y2="{base:.1f}" '
                 f'stroke="{INK}" stroke-width="1"/>')


def shots_svg() -> str:
    s = pd.read_parquet(SHOTS)
    w, h = 1000, 380
    parts = svg_open(w, h, "Shot durations and tilt for the demo game")
    parts.append(text(30, 34, f"{len(s)} camera takes, Rams at Bears 2025 wk 20",
                      size=26, family=FONT, weight="700"))
    parts.append(text(30, 56, "left: how long each take lasts.  right: dominant edge angle per "
                      "take, the signal that separates the two camera angles.",
                      size=13, fill=MUTED))
    histogram_panel(parts, s["duration_s"].tolist(), x0=70, y0=110, w=400, h=230,
                    lo=0, hi=48, step=2, title="Take duration", unit="seconds", color=TURF)
    histogram_panel(parts, s["tilt"].tolist(), x0=560, y0=110, w=400, h=230,
                    lo=0, hi=90, step=3, title="Tilt", unit="degrees", color=BLUE)
    med = float(s["duration_s"].median())
    parts.append(text(470, 330, f"median {med:.1f}s", size=12, fill=MUTED, anchor="end",
                      family=MONO))
    parts.append(text(960, 330, "sideline ≈ 88°   end zone ≈ 47°", size=12, fill=MUTED,
                      anchor="end", family=MONO))
    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------------ timeline ------------

PLAY_COLORS = {
    "pass": BLUE, "run": TURF, "punt": "#8A9790", "kickoff": "#7A5BC7",
    "field_goal": YELLOW, "extra_point": "#D9A82F",
}


def timeline_svg() -> str:
    p = pd.read_parquet(PLAYS).sort_values("start_s")
    w, h = 1000, 300
    parts = svg_open(w, h, "Play timeline for the demo game")
    parts.append(text(30, 34, f"{len(p)} plays on the film's clock", size=26, family=FONT,
                      weight="700"))
    parts.append(text(30, 56, "each bar is one play (both camera angles), placed where it sits "
                      "in the 96-minute recording", size=13, fill=MUTED))
    x0, x1 = 40.0, 960.0
    t0, t1 = float(p["start_s"].min()), float(p["end_s"].max())

    def tx(t: float) -> float:
        return x0 + (t - t0) / (t1 - t0) * (x1 - x0)

    # Two rows: a lane per angle to make the alternation visible.
    lanes = {"sideline": 120.0, "endzone": 160.0}
    for name, yy in lanes.items():
        parts.append(text(x0 - 6, yy + 18, name, size=12, fill=MUTED, anchor="end",
                          family=MONO))
    for r in p.itertuples():
        color = PLAY_COLORS.get(r.play_type, MUTED)
        for angle, yy in lanes.items():
            a, b = getattr(r, f"{angle}_start_s"), getattr(r, f"{angle}_end_s")
            parts.append(f'<rect x="{tx(a):.1f}" y="{yy:.1f}" width="{max(tx(b) - tx(a), 1):.1f}" '
                         f'height="24" fill="{color}" opacity="0.92"/>')
    # Time axis in minutes of media.
    base = 200.0
    parts.append(f'<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="{INK}" stroke-width="1"/>')
    for m in range(int(t0 // 600) * 10, int(t1 // 60) + 1, 10):
        t = m * 60
        if t < t0 or t > t1:
            continue
        parts.append(f'<line x1="{tx(t):.1f}" y1="{base}" x2="{tx(t):.1f}" y2="{base + 6}" '
                     f'stroke="{INK}" stroke-width="1"/>')
        parts.append(text(tx(t), base + 20, f"{m}:00", size=11, fill=MUTED, anchor="middle",
                          family=MONO))
    parts.append(text(x1, base + 38, "media time (min)", size=11, fill=MUTED, anchor="end"))
    # Legend with counts.
    counts = p["play_type"].value_counts().to_dict()
    lx = x0
    for kind, color in PLAY_COLORS.items():
        if kind not in counts:
            continue
        parts.append(f'<rect x="{lx:.1f}" y="262" width="14" height="14" fill="{color}"/>')
        label = f"{kind.replace('_', ' ')} {counts[kind]}"
        parts.append(text(lx + 20, 274, label, size=13))
        lx += 24 + 8 * len(label) + 20
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "field.svg").write_text(field_svg())
    (OUT / "shots-histogram.svg").write_text(shots_svg())
    (OUT / "play-timeline.svg").write_text(timeline_svg())
    for name in ("field.svg", "shots-histogram.svg", "play-timeline.svg"):
        print(f"wrote {OUT / name} ({(OUT / name).stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
