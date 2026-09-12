"""Hand-authored play records for the film-room demo.

This is NOT pipeline output. Three real plays from Rams at Bears (2025 Divisional round) are
re-enacted from their official play-by-play descriptions: the down, distance, yard line,
personnel, formation, who threw, who caught, who tackled, and where the ball ended up are
official. Every player's path between those facts is authored by hand here to look like real
football, so the demo can show what a finished play record and its views will be. Every field
that came from a hand carries `source: "human"`, and every record carries `demo: true`.

Output: site/demo/data/records.js  ->  window.FZ_DEMO_RECORDS = { play_uid: record, ... }

Coordinates: field frame from formation_zero.configs.nfl_field (x 0..120 along the field,
y 0..53.3 across, yards). All three plays attack toward +x. From the offense's point of view,
"left" is +y and "right" is -y. Directions are degrees, 0 = toward the top sideline (+y),
clockwise, matching the Big Data Bowl convention.

Run:  .venv/bin/python scripts/demo/synth.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "demo" / "data" / "records.js"

FPS = 25
GOAL_X = 110.0          # right goal line
HASH_L, MID, HASH_R = 29.75, 26.667, 23.583   # y of left hash, middle, right hash (offense facing +x)

LA = {"abbr": "LA", "name": "Los Angeles Rams"}
CHI = {"abbr": "CHI", "name": "Chicago Bears"}


# ------------------------------------------------------------------ motion primitives ------

def ease(t: float) -> float:
    """Smoothstep: no velocity discontinuity at waypoints."""
    return t * t * (3 - 2 * t)


@dataclass
class Mover:
    """An entity with waypoints (time_s, x, y). Positions are interpolated per frame."""

    id: str
    team: str                  # "LA" | "CHI" | "OFF"
    side: str                  # "offense" | "defense" | "official"
    jersey: str
    name: str | None
    position: str              # roster position
    role: str                  # assigned role at the snap
    face: float                # resting orientation, degrees
    wps: list[tuple[float, float, float]] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def at(self, t: float) -> tuple[float, float]:
        w = self.wps
        if t <= w[0][0]:
            return w[0][1], w[0][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(w, w[1:]):
            if t0 <= t <= t1:
                u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
                return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
        return w[-1][1], w[-1][2]

    def sample(self, frames: int) -> list[list[float]]:
        out = []
        prev = None
        for f in range(frames):
            x, y = self.at(f / FPS)
            if prev is not None:
                dx, dy = x - prev[0], y - prev[1]
                d = math.degrees(math.atan2(dx, dy)) % 360 if math.hypot(dx, dy) > 0.02 else self.face
            else:
                d = self.face
            out.append([round(x, 2), round(y, 2), round(d, 1)])
            prev = (x, y)
        return out


def rel(los_x: float, ball_y: float, lateral: float, depth: float) -> tuple[float, float]:
    """Snap-relative (lateral: +left/-right, depth: +behind LOS) -> absolute (x, y), attacking +x."""
    return los_x - depth, ball_y + lateral


def ball_flight(p0, p1, t0, t1, peak_z=7.0, frames=None):
    """Parabolic pass from p0 at t0 to p1 at t1: per-frame [x, y, z]."""
    out = {}
    for f in range(int(t0 * FPS), int(t1 * FPS) + 1):
        u = (f / FPS - t0) / (t1 - t0)
        x = p0[0] + (p1[0] - p0[0]) * u
        y = p0[1] + (p1[1] - p0[1]) * u
        z = 1.8 + peak_z * 4 * u * (1 - u)
        out[f] = [round(x, 2), round(y, 2), round(z, 2)]
    return out


# ------------------------------------------------------------------ record assembly --------

def summary(track: list[list[float]]) -> dict:
    dist = 0.0
    vmax = 0.0
    for a, b in zip(track, track[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        dist += d
        vmax = max(vmax, d * FPS)
    return {"distance_yds": round(dist, 1), "max_speed_yds_s": round(vmax, 1),
            "frames_tracked": len(track), "frames_lost": 0}


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def metrics_for(play: dict, movers: list[Mover], tracks: dict, ball: dict, frames: int) -> tuple[dict, dict]:
    """Separation, closeness and pursuit, read from the same distances on both sides.

    Offense skill players: distance to the nearest defender per frame (separation).
    Defenders: distance to their assignment (man) or nearest receiver (zone), and to whoever
    has the ball (pursuit). Quarterback: nearest rusher over the dropback (pressure).
    Summaries go in the record; the viewer draws the per-frame curves from the tracking table.
    """
    snap_f, whistle_f = play["snap_frame"], play["whistle_frame"]
    live = range(snap_f, whistle_f + 1)
    release = play["live"]["ball"]["release_frame"]
    arrival = play["live"]["ball"]["arrival_frame"]
    by_id = {m.id: m for m in movers}
    offense = [m.id for m in movers if m.side == "offense"]
    skill = [m.id for m in movers if m.side == "offense" and (m.extra.get("route_detail") or m.extra.get("ball_carrier"))]
    receivers = [m.id for m in movers if m.side == "offense" and m.extra.get("route_detail")]
    defenders = [m.id for m in movers if m.side == "defense"]
    rushers = [d for d in defenders if (by_id[d].extra.get("pass_rush") or {}).get("rushed")]
    qb = next((m.id for m in movers if m.extra.get("qb")), None)
    out: dict = {}

    def nearest(f: int, src: str, pool: list[str]):
        best, bd = None, 1e9
        for pid in pool:
            d = dist(tracks[src][f], tracks[pid][f])
            if d < bd:
                best, bd = pid, d
        return best, bd

    for pid in skill:
        sep = [nearest(f, pid, defenders)[1] for f in live]
        m = {"kind": "separation",
             "min_live_yds": round(min(sep), 1), "mean_live_yds": round(sum(sep) / len(sep), 1),
             "min_at_frame": snap_f + sep.index(min(sep))}
        if release is not None:
            n, d = nearest(release, pid, defenders)
            m["at_throw_yds"], m["nearest_at_throw"] = round(d, 1), n
        if arrival is not None:
            n, d = nearest(arrival, pid, defenders)
            m["at_catch_yds"], m["nearest_at_catch"] = round(d, 1), n
        out[pid] = m

    for did in defenders:
        cov = by_id[did].extra.get("coverage") or {}
        assignment = cov.get("on")
        pool = [assignment] if assignment else receivers
        close = [nearest(f, did, pool)[1] for f in live] if pool else []
        m = {"kind": "coverage" if pool else "front"}
        if close:
            m.update({"assignment": assignment, "mean_to_assignment_yds": round(sum(close) / len(close), 1),
                      "min_to_assignment_yds": round(min(close), 1)})
            if release is not None:
                m["to_assignment_at_throw_yds"] = round(nearest(release, did, pool)[1], 1)
            if arrival is not None:
                target = play["live"]["events"]
                catcher = next((e.get("track_id") for e in target if e["type"] == "pass_outcome_caught"), None)
                if catcher:
                    m["nearest_at_catch"] = nearest(arrival, catcher, defenders)[0] == did
                    m["to_catch_yds"] = round(dist(tracks[did][arrival], tracks[catcher][arrival]), 1)
        # Pursuit: distance to whoever has the ball, each live frame.
        carriers = [ball["carrier"](f) for f in live]
        pur = [dist(tracks[did][f], tracks[c][f]) for f, c in zip(live, carriers)]
        first_close = next((f for f, d in zip(live, pur) if d <= 2.0), None)
        # Closing speed only between frames with the same carrier: a handoff or catch moves
        # the ball, not the defender, and would otherwise read as an impossible burst.
        closing = [(pur[i] - pur[i + 1]) * FPS for i in range(len(pur) - 1) if carriers[i] == carriers[i + 1]]
        m["pursuit"] = {"min_to_carrier_yds": round(min(pur), 1), "at_frame": snap_f + pur.index(min(pur)),
                        "first_within_2yds_frame": first_close,
                        "closing_speed_max_yds_s": round(max(0.0, max(closing)) if closing else 0.0, 1)}
        out[did] = m

    qb_pressure = None
    if qb and rushers:
        end = release if release is not None else min(whistle_f, snap_f + 3 * FPS)
        span = list(range(snap_f, end + 1))
        near = [nearest(f, qb, rushers)[1] for f in span]
        first = next((f for f, d in zip(span, near) if d <= 2.5), None)
        qb_pressure = {"nearest_rusher_min_yds": round(min(near), 1), "at_frame": snap_f + near.index(min(near)),
                       "time_to_pressure_s": round((first - snap_f) / FPS, 2) if first is not None else None,
                       "rushers_within_3yds_at_throw": (sum(dist(tracks[qb][release], tracks[r][release]) <= 3.0 for r in rushers)
                                                       if release is not None else None)}
        out[qb] = {"kind": "passer"}
    return out, qb_pressure


def build(play: dict, movers: list[Mover], ball: dict, frames: int) -> dict:
    los_x, ball_y, snap_f = play["los_x"], play["ball_y"], play["snap_frame"]
    tracks = {m.id: m.sample(frames) for m in movers}
    metrics, qb_pressure = metrics_for(play, movers, tracks, ball, frames)
    if qb_pressure is not None:
        play["live"]["qb"]["pressure"] = qb_pressure
    players = []
    for m in movers:
        tr = tracks[m.id]
        pre = tr[max(0, snap_f - 60)]
        snap = tr[snap_f]
        block = {
            "track_id": m.id, "team": m.team, "side": m.side,
            "gsis_id": None, "jersey": m.jersey, "name": m.name, "position_roster": m.position,
            "role": m.role,
            "alignment_pre_snap": {"lateral": round(pre[1] - ball_y, 1), "depth": round(los_x - pre[0], 1),
                                   "x": pre[0], "y": pre[1], "frame": max(0, snap_f - 60)},
            "alignment_at_snap": {"lateral": round(snap[1] - ball_y, 1), "depth": round(los_x - snap[0], 1),
                                  "x": snap[0], "y": snap[1], "frame": snap_f},
            "direction_at_snap_deg": snap[2], "orientation_at_snap_deg": m.face,
            "summary": summary(tr),
            "metrics": metrics.get(m.id),
            "source": "human", "confidence": 1.0 if m.name or m.side == "official" else 0.55,
        }
        block.update(m.extra)
        players.append(block)

    ball_track = []
    for f in range(frames):
        if f in ball["flight"]:
            ball_track.append(ball["flight"][f])
        else:
            carrier = ball["carrier"](f)
            x, y, _ = tracks[carrier][f]
            ball_track.append([x, y, 1.0])

    record = {
        "schema_version": "1.0",
        "demo": True,
        "demo_note": "Hand-made example of a finished play record. The situation, personnel, formation label, "
                     "who threw, who caught, who tackled, and the result are official. Every path between "
                     "those facts was authored by hand to show what the pipeline will produce.",
        "play_uid": play["uid"],
        "game": {"game_key": "2025_wk20_LA-CHI", "season": 2025, "week": 20, "away": "LA", "home": "CHI",
                 "nfl_game_id": "2025_20_LA_CHI", "nfl_play_id": play["nfl_play_id"]},
        "situation": play["situation"],
        "official": play["official"],
        "film": {"fps": FPS,
                 "sideline": {"clip": f"raw/2025/wk20/LA_at_CHI/sideline/{play['uid'][-4:]}.mp4", "frames": frames, "url": None},
                 "endzone": {"clip": f"raw/2025/wk20/LA_at_CHI/endzone/{play['uid'][-4:]}.mp4", "frames": frames, "url": None},
                 "video_note": "Clips are frame-locked to the recreation. The url is filled with whatever footage the client is licensed to load; none is published here.",
                 "snap_frame": snap_f, "whistle_frame": play["whistle_frame"],
                 "phases": {"pre_snap": [0, snap_f - 1], "live": [snap_f, play["whistle_frame"]],
                            "post_snap": [play["whistle_frame"] + 1, frames - 1]}},
        "personnel": play["personnel"],
        "pre_snap": play["pre_snap"],
        "at_snap": {"frame": snap_f, "ball": {"x": los_x, "y": ball_y}, "in_motion": play["in_motion"]},
        "live": play["live"],
        "post_snap": play["post_snap"],
        "players": players,
        "tracking": {"fps": FPS, "frames": frames, "snap_frame": snap_f,
                     "ids": [m.id for m in movers], "xy": tracks, "ball": ball_track},
        "quality": {"registration_error_px": None, "tracks_fragmented": 0, "identity_unresolved": 0,
                    "router": [], "overall_confidence": 1.0, "note": "hand-made example"},
        "generated": {"pipeline_version": "demo", "at": "2026-09-11"},
    }
    return record


# ------------------------------------------------------------------ shared casts -----------

def rams_oline(los, by, depth=0.7):
    return [
        Mover("LA77", "LA", "offense", "77", "Alaric Jackson", "T", "LT", 90, [(0, *rel(los, by, 2.6, depth))]),
        Mover("LA73", "LA", "offense", "73", "Steve Avila", "G", "LG", 90, [(0, *rel(los, by, 1.3, depth))]),
        Mover("LA65", "LA", "offense", "65", "Coleman Shelton", "C", "C", 90, [(0, *rel(los, by, 0.0, depth))]),
        Mover("LA69", "LA", "offense", "69", "Kevin Dotson", "G", "RG", 90, [(0, *rel(los, by, -1.3, depth))]),
        Mover("LA79", "LA", "offense", "79", "Rob Havenstein", "T", "RT", 90, [(0, *rel(los, by, -2.6, depth))]),
    ]


def bears_oline(los, by, depth=0.7):
    return [
        Mover("CHI70", "CHI", "offense", "70", "Braxton Jones", "T", "LT", 90, [(0, *rel(los, by, 2.6, depth))]),
        Mover("CHI62", "CHI", "offense", "62", "Joe Thuney", "G", "LG", 90, [(0, *rel(los, by, 1.3, depth))]),
        Mover("CHI63", "CHI", "offense", "63", "Drew Dalman", "C", "C", 90, [(0, *rel(los, by, 0.0, depth))]),
        Mover("CHI73", "CHI", "offense", "73", "Jonah Jackson", "G", "RG", 90, [(0, *rel(los, by, -1.3, depth))]),
        Mover("CHI58", "CHI", "offense", "58", "Darnell Wright", "T", "RT", 90, [(0, *rel(los, by, -2.6, depth))]),
    ]


def officials(los, by, snap_t, end_t):
    r = Mover("OFF-R", "OFF", "official", "R", None, "R", "OFFICIAL", 90,
              [(0, *rel(los, by, -6, 14)), (snap_t + 1, *rel(los, by, -8, 12)), (end_t, *rel(los, by, -8, 6))])
    u = Mover("OFF-U", "OFF", "official", "U", None, "U", "OFFICIAL", 270,
              [(0, *rel(los, by, 3, -6)), (snap_t + 1.5, *rel(los, by, 3, -1)), (end_t, *rel(los, by, 3, -4))])
    return [r, u]


def pass_set(oline: list[Mover], los, by, snap_t, hold_t, depth_gain=1.8):
    """Offensive line pass protection: kick back and hold."""
    for m in oline:
        x, y = m.wps[0][1], m.wps[0][2]
        lat = y - by
        m.wps = [(0, x, y), (snap_t, x, y), (snap_t + 0.5, x - depth_gain, y + (0.4 if lat > 0 else -0.4 if lat < 0 else 0)),
                 (hold_t, x - depth_gain - 0.6, y + (0.6 if lat > 0 else -0.6 if lat < 0 else 0))]
        m.extra["block"] = {"assignment": "pass_pro"}


def run_block(oline: list[Mover], los, by, snap_t, drive_t, push=1.6, lean=0.0):
    for m in oline:
        x, y = m.wps[0][1], m.wps[0][2]
        m.wps = [(0, x, y), (snap_t, x, y), (drive_t, x + push, y + lean)]
        m.extra["block"] = {"assignment": "run_block"}


# ------------------------------------------------------------------ play 1: p001 ----------

def play_001() -> dict:
    """CHI 1st & 10 at CHI 37: Williams play-action, DJ Moore shallow cross, 34 yards after the catch."""
    los, by = 47.0, MID
    snap_t, throw_t, catch_t, contact_t, tackle_t, whistle_t, end_t = 3.0, 4.8, 5.3, 7.4, 8.0, 8.5, 10.0
    frames = int(end_t * FPS)

    ol = bears_oline(los, by)
    pass_set(ol, los, by, snap_t, tackle_t, depth_gain=1.2)
    # Play-action: line sells the run for half a second first.
    for m in ol:
        x, y = m.wps[0][1], m.wps[0][2]
        m.wps = [(0, x, y), (snap_t, x, y), (snap_t + 0.4, x + 0.8, y), (snap_t + 1.0, x - 1.0, y), (tackle_t, x - 1.6, y)]

    qb = Mover("CHI18", "CHI", "offense", "18", "Caleb Williams", "QB", "QB", 90, [
        (0, *rel(los, by, 0, 1.4)), (snap_t, *rel(los, by, 0, 1.4)),
        (snap_t + 0.5, *rel(los, by, 0.8, 4.5)),      # turn, fake to the back
        (snap_t + 1.2, *rel(los, by, -3.5, 7.5)),     # boot right
        (throw_t, *rel(los, by, -6.0, 7.0)),
        (throw_t + 0.6, *rel(los, by, -7.0, 6.5)), (end_t, *rel(los, by, -7.0, 6.5))],
        extra={"qb": True})
    rb = Mover("CHI4", "CHI", "offense", "4", "D'Andre Swift", "RB", "RB", 90, [
        (0, *rel(los, by, 0, 7.0)), (snap_t, *rel(los, by, 0, 7.0)),
        (snap_t + 0.5, *rel(los, by, 1.5, 4.0)),      # takes the fake toward the left A gap
        (snap_t + 1.3, *rel(los, by, 6.0, 1.5)),
        (throw_t + 0.5, *rel(los, by, 12.0, -2.0)), (end_t, *rel(los, by, 14.0, -3.0))],
        extra={"route": "flat", "route_detail": {"route": "flat", "tree": 1, "break_frame": int((snap_t + 1.3) * FPS),
                                                 "depth_at_break_yds": -1.5, "break_direction": "left", "targeted": False}})
    te_kmet = Mover("CHI85", "CHI", "offense", "85", "Cole Kmet", "TE", "TE", 90, [
        (0, *rel(los, by, -4.0, 0.7)), (snap_t, *rel(los, by, -4.0, 0.7)),
        (snap_t + 0.6, *rel(los, by, -4.5, -1.0)),    # sells the block on the edge
        (snap_t + 2.0, *rel(los, by, -6.0, -9.0)),
        (throw_t + 0.8, *rel(los, by, -12.0, -16.0)), (end_t, *rel(los, by, -13.0, -18.0))],
        extra={"route": "corner", "route_detail": {"route": "corner", "tree": 7, "break_frame": int((snap_t + 2.0) * FPS),
                                                   "depth_at_break_yds": 9.0, "break_direction": "right", "targeted": False}})
    te_love = Mover("CHI84", "CHI", "offense", "84", "Colston Loveland", "TE", "TE", 90, [
        (0, *rel(los, by, 4.5, 1.2)), (1.0, *rel(los, by, 4.5, 1.2)),
        (2.2, *rel(los, by, -6.0, 1.2)),              # shift: wing left to wing right, set before the snap
        (snap_t, *rel(los, by, -6.0, 1.2)),
        (snap_t + 0.6, *rel(los, by, -6.5, 0.2)),     # arc block on the edge, then release
        (snap_t + 1.4, *rel(los, by, -8.0, -1.0)), (end_t, *rel(los, by, -9.0, -2.0))],
        extra={"block": {"assignment": "arc_block", "engaged_with": "LA8"}})
    moore = Mover("CHI2", "CHI", "offense", "2", "DJ Moore", "WR", "WR", 90, [
        (0, *rel(los, by, 8.0, 1.0)), (snap_t, *rel(los, by, 8.0, 1.0)),
        (snap_t + 0.7, *rel(los, by, 6.0, -3.0)),     # stem, then across shallow
        (throw_t, *rel(los, by, -5.0, -5.0)),
        (catch_t, *rel(los, by, -9.0, -6.0)),         # catch at CHI 43 (x=53), right of the hash
        (6.2, 60.0, 11.0), (7.0, 70.0, 6.5), (contact_t, 76.0, 5.0),
        (tackle_t, 81.0, 4.2), (end_t, 81.5, 4.0)],
        extra={"route": "cross", "route_detail": {"route": "cross", "tree": None, "break_frame": int((snap_t + 0.7) * FPS),
                                                  "depth_at_break_yds": 3.0, "break_direction": "right", "targeted": True,
                                                  "catch_frame": int(catch_t * FPS)}})
    odunze = Mover("CHI15", "CHI", "offense", "15", "Rome Odunze", "WR", "WR", 90, [
        (0, *rel(los, by, 18.0, 0.8)), (snap_t, *rel(los, by, 18.0, 0.8)),
        (throw_t + 1.0, *rel(los, by, 17.0, -22.0)), (end_t, *rel(los, by, 16.0, -32.0))],
        extra={"route": "go", "route_detail": {"route": "go", "tree": 9, "break_frame": None,
                                               "depth_at_break_yds": None, "break_direction": None, "targeted": False}})

    # Rams 3-4, Cover 3 with a strong rotation. They bite on the play fake.
    d = []
    d.append(Mover("LA95", "LA", "defense", "95", "Poona Ford", "NT", "NT", 270, [
        (0, *rel(los, by, 0.3, -1.2)), (snap_t, *rel(los, by, 0.3, -1.2)), (snap_t + 1.0, *rel(los, by, 0.3, -0.2)), (end_t, *rel(los, by, 0.5, 1.0))],
        extra={"technique": "0", "stance": "3pt", "pass_rush": {"rushed": True, "path": "interior", "pressure": False}}))
    d.append(Mover("LA55", "LA", "defense", "55", "Braden Fiske", "DE", "DT", 270, [
        (0, *rel(los, by, 2.4, -1.2)), (snap_t, *rel(los, by, 2.4, -1.2)), (snap_t + 1.0, *rel(los, by, 2.6, 0.0)), (end_t, *rel(los, by, 2.5, 1.5))],
        extra={"technique": "4i", "stance": "3pt", "pass_rush": {"rushed": True, "path": "interior", "pressure": False}}))
    d.append(Mover("LA91", "LA", "defense", "91", "Kobie Turner", "DT", "DT", 270, [
        (0, *rel(los, by, -1.8, -1.2)), (snap_t, *rel(los, by, -1.8, -1.2)), (snap_t + 1.0, *rel(los, by, -2.0, 0.2)), (end_t, *rel(los, by, -2.5, 2.0))],
        extra={"technique": "3", "stance": "3pt", "pass_rush": {"rushed": True, "path": "interior", "pressure": False}}))
    d.append(Mover("LA0", "LA", "defense", "0", "Byron Young", "OLB", "EDGE", 270, [
        (0, *rel(los, by, 4.8, -1.0)), (snap_t, *rel(los, by, 4.8, -1.0)), (snap_t + 0.6, *rel(los, by, 5.5, 1.5)), (throw_t, *rel(los, by, 2.0, 5.0)), (end_t, *rel(los, by, 0.0, 6.0))],
        extra={"technique": "7", "stance": "2pt", "pass_rush": {"rushed": True, "path": "edge_left", "pressure": False}}))
    d.append(Mover("LA8", "LA", "defense", "8", "Jared Verse", "OLB", "EDGE", 270, [
        (0, *rel(los, by, -5.5, -1.0)), (snap_t, *rel(los, by, -5.5, -1.0)), (snap_t + 0.6, *rel(los, by, -5.0, 0.5)),
        (snap_t + 1.4, *rel(los, by, -7.0, 2.5)), (throw_t, *rel(los, by, -8.0, 5.5)), (end_t, *rel(los, by, -8.5, 6.0))],
        extra={"technique": "9", "stance": "2pt", "pass_rush": {"rushed": True, "path": "edge_right", "pressure": True}}))
    d.append(Mover("LA53", "LA", "defense", "53", "Nate Landman", "ILB", "LB", 270, [
        (0, *rel(los, by, 1.5, -4.5)), (snap_t, *rel(los, by, 1.5, -4.5)), (snap_t + 0.5, *rel(los, by, 2.0, -2.5)),   # bites
        (throw_t, *rel(los, by, 3.0, -8.0)), (tackle_t, *rel(los, by, -20.0, -22.0)), (end_t, *rel(los, by, -22.0, -26.0))],
        extra={"coverage": {"assignment": "hook_left", "man_zone": "zone"}}))
    d.append(Mover("LA44", "LA", "defense", "44", "Omar Speights", "ILB", "LB", 270, [
        (0, *rel(los, by, -1.5, -4.5)), (snap_t, *rel(los, by, -1.5, -4.5)), (snap_t + 0.5, *rel(los, by, -1.0, -2.5)),
        (throw_t, *rel(los, by, -4.0, -7.0)), (catch_t + 0.4, *rel(los, by, -8.0, -7.0)), (tackle_t, *rel(los, by, -19.0, -30.0)), (end_t, *rel(los, by, -20.0, -33.0))],
        extra={"coverage": {"assignment": "hook_right", "man_zone": "zone"}}))
    d.append(Mover("LA31", "LA", "defense", "31", "Darious Williams", "CB", "CB", 270, [
        (0, *rel(los, by, 17.0, -7.0)), (snap_t, *rel(los, by, 17.0, -7.0)), (throw_t + 1.0, *rel(los, by, 15.0, -24.0)), (end_t, *rel(los, by, 14.0, -34.0))],
        extra={"coverage": {"assignment": "deep_third_left", "man_zone": "zone"}}))
    d.append(Mover("LA25", "LA", "defense", "25", "Roger McCreary", "CB", "CB", 270, [
        (0, *rel(los, by, -13.0, -6.0)), (snap_t, *rel(los, by, -13.0, -6.0)), (throw_t, *rel(los, by, -15.0, -14.0)),
        (catch_t + 0.3, *rel(los, by, -15.5, -13.0)), (6.6, 66.0, 9.5), (contact_t, 76.5, 6.0), (tackle_t, 81.0, 5.0), (end_t, 81.5, 5.2)],
        extra={"coverage": {"assignment": "deep_third_right", "man_zone": "zone"}}))
    d.append(Mover("LA26", "LA", "defense", "26", "Kamren Kinchens", "S", "FS", 270, [
        (0, *rel(los, by, 0.5, -14.0)), (snap_t, *rel(los, by, 0.5, -14.0)), (throw_t, *rel(los, by, 0.0, -20.0)),
        (tackle_t, *rel(los, by, -16.0, -34.0)), (end_t, *rel(los, by, -17.0, -35.0))],
        extra={"coverage": {"assignment": "deep_middle", "man_zone": "zone"}}))
    d.append(Mover("LA3", "LA", "defense", "3", "Kam Curl", "S", "SS", 270, [
        (0, *rel(los, by, -6.0, -9.0)), (snap_t, *rel(los, by, -6.0, -9.0)), (snap_t + 0.6, *rel(los, by, -4.0, -5.0)),   # bites on the fake
        (throw_t, *rel(los, by, -4.0, -3.0)), (catch_t + 0.5, *rel(los, by, -7.0, -5.5)), (tackle_t, *rel(los, by, -18.0, -30.0)), (end_t, *rel(los, by, -19.0, -32.0))],
        extra={"coverage": {"assignment": "curl_flat_right", "man_zone": "zone"}}))

    movers = ol + [qb, rb, te_kmet, te_love, moore, odunze] + d + officials(los, by, snap_t, end_t)
    throw_from = qb.at(throw_t)
    catch_at = moore.at(catch_t)
    flight = ball_flight(throw_from, catch_at, throw_t, catch_t, peak_z=3.0)

    def carrier(f):
        t = f / FPS
        if t < snap_t + 0.5:
            return "CHI18"
        if t < catch_t:
            return "CHI18"
        return "CHI2"

    play = {
        "uid": "2025_wk20_LA-CHI_p001", "nfl_play_id": 40, "los_x": los, "ball_y": by,
        "snap_frame": int(snap_t * FPS), "whistle_frame": int(whistle_t * FPS),
        "situation": {"quarter": 1, "clock": "14:53", "down": 1, "distance": 10, "possession": "CHI",
                      "yardline_100": 63, "yrdln": "CHI 37", "los_x": los, "first_down_x": los + 10,
                      "direction_of_attack": "right", "hash": "M", "score": {"away": 0, "home": 0}},
        "official": {"play_type": "pass",
                     "description": "18-C.Williams pass short right to 2-D.Moore to LA 29 for 34 yards (25-R.McCreary).",
                     "yards_gained": 34, "first_down": True, "touchdown": False, "turnover": False,
                     "players_involved": [{"jersey": "18", "team": "CHI", "role": "passer", "name": "C.Williams"},
                                          {"jersey": "2", "team": "CHI", "role": "receiver", "name": "D.Moore"},
                                          {"jersey": "25", "team": "LA", "role": "tackler", "name": "R.McCreary"}],
                     "labels": {"offense_formation": "UNDER CENTER", "offense_personnel": "12",
                                "defense_personnel": "3-4 base", "source": "nflverse"}},
        "personnel": {"offense": {"grouping": "12", "rb": 1, "te": 2, "wr": 2, "source": "human", "confidence": 1.0},
                      "defense": {"package": "base", "dl": 3, "lb": 4, "db": 4, "source": "human", "confidence": 1.0}},
        "pre_snap": {
            "formation_initial": {"frame": 20, "name": "ace_wing_left", "strength": "left", "qb": "under_center",
                                  "backfield": "single_back", "source": "human", "confidence": 1.0},
            "shifts": [{"track_id": "CHI84", "from_slot": "WGL", "to_slot": "WGR", "start_frame": 25, "end_frame": 55}],
            "motions": [],
            "formation_at_snap": {"frame": int(snap_t * FPS), "name": "ace_wing_right", "strength": "right",
                                  "qb": "under_center", "backfield": "single_back", "source": "human", "confidence": 1.0},
            "defense": {"front": "odd", "shell": "1-high", "box_count": 7, "safeties_depth_yds": [14.0, 9.0],
                        "press_corners": [False, False], "source": "human", "confidence": 1.0}},
        "in_motion": [],
        "live": {
            "play_family": "play_action", "play_action": True, "rpo": False,
            "qb": {"track_id": "CHI18", "drop": "rollout_right", "time_to_throw_s": round(throw_t - snap_t, 1),
                   "pocket": "moved", "scramble": False},
            "routes": [
                {"track_id": "CHI2", "route": "cross", "tree": None, "break_frame": int((snap_t + 0.7) * FPS), "depth_at_break_yds": 3.0,
                 "break_direction": "right", "targeted": True, "catch_frame": int(catch_t * FPS), "source": "human", "confidence": 1.0},
                {"track_id": "CHI85", "route": "corner", "tree": 7, "break_frame": int((snap_t + 2.0) * FPS), "depth_at_break_yds": 9.0,
                 "break_direction": "right", "targeted": False, "source": "human", "confidence": 1.0},
                {"track_id": "CHI15", "route": "go", "tree": 9, "break_frame": None, "depth_at_break_yds": None,
                 "break_direction": None, "targeted": False, "source": "human", "confidence": 1.0},
                {"track_id": "CHI4", "route": "flat", "tree": 1, "break_frame": int((snap_t + 1.3) * FPS), "depth_at_break_yds": -1.5,
                 "break_direction": "left", "targeted": False, "source": "human", "confidence": 1.0}],
            "run": None,
            "blocks": [{"track_id": "CHI84", "assignment": "arc_block", "engaged_with": "LA8", "engaged_frames": [int((snap_t + 0.5) * FPS), int((snap_t + 1.3) * FPS)]}],
            "defense": {"coverage": {"family": "cover_3", "man_zone": "zone", "rotation": "strong", "source": "human", "confidence": 0.72},
                        "pass_rush": [{"track_id": "LA8", "rushed": True, "path": "edge_right"}, {"track_id": "LA0", "rushed": True, "path": "edge_left"},
                                      {"track_id": "LA95", "rushed": True, "path": "interior"}, {"track_id": "LA55", "rushed": True, "path": "interior"},
                                      {"track_id": "LA91", "rushed": True, "path": "interior"}],
                        "blitz": False},
            "ball": {"release_frame": int(throw_t * FPS), "arrival_frame": int(catch_t * FPS), "air_yards": 6, "outcome": "complete"},
            "events": [{"frame": int(snap_t * FPS), "type": "snap"},
                       {"frame": int((snap_t + 0.5) * FPS), "type": "play_fake", "track_id": "CHI4"},
                       {"frame": int(throw_t * FPS), "type": "pass_forward", "track_id": "CHI18"},
                       {"frame": int(catch_t * FPS), "type": "pass_outcome_caught", "track_id": "CHI2"},
                       {"frame": int(contact_t * FPS), "type": "first_contact", "track_id": "CHI2", "by": ["LA25"]},
                       {"frame": int(tackle_t * FPS), "type": "tackle", "track_id": "CHI2", "by": ["LA25"]},
                       {"frame": int(whistle_t * FPS), "type": "whistle"}]},
        "post_snap": {"result": "complete", "yards_gained": 34, "yards_after_catch": 28,
                      "ball_spot": {"x": 81.0, "y": 4.2}, "first_down": True,
                      "tackle": {"frame": int(tackle_t * FPS), "x": 81.0, "y": 4.2, "by": ["LA25"]},
                      "out_of_bounds": False, "penalty": None, "end_frame": frames - 1},
    }
    return build(play, movers, {"flight": flight, "carrier": carrier}, frames)


# ------------------------------------------------------------------ play 2: p019 ----------

def play_019() -> dict:
    """LA 2nd & 4 at CHI 4: Kyren Williams off right tackle for the touchdown."""
    los, by = 106.0, MID
    snap_t, hand_t, contact_t, td_t, whistle_t, end_t = 3.0, 3.6, 4.6, 5.0, 5.6, 7.5
    frames = int(end_t * FPS)

    ol = rams_oline(los, by)
    run_block(ol, los, by, snap_t, td_t, push=1.4)
    ol[3].wps[-1] = (td_t, *rel(los, by, -1.6, -1.4))        # RG on the 3-tech
    ol[4].wps[-1] = (td_t, *rel(los, by, -3.2, -1.8))        # RT kicks the edge out
    higbee = Mover("LA89", "LA", "offense", "89", "Tyler Higbee", "TE", "TE", 90, [
        (0, *rel(los, by, -4.0, 0.7)), (snap_t, *rel(los, by, -4.0, 0.7)), (td_t, *rel(los, by, -5.5, -1.6))],
        extra={"block": {"assignment": "run_block", "engaged_with": "CHI98"}})
    qb = Mover("LA9", "LA", "offense", "9", "Matthew Stafford", "QB", "QB", 90, [
        (0, *rel(los, by, 0, 1.4)), (snap_t, *rel(los, by, 0, 1.4)), (hand_t, *rel(los, by, -1.0, 4.5)),
        (hand_t + 0.6, *rel(los, by, 1.5, 6.0)), (end_t, *rel(los, by, 1.5, 6.0))], extra={"qb": True})
    rb = Mover("LA23", "LA", "offense", "23", "Kyren Williams", "RB", "RB", 90, [
        (0, *rel(los, by, 0, 6.8)), (snap_t, *rel(los, by, 0, 6.8)), (hand_t, *rel(los, by, -1.0, 4.4)),
        (4.2, *rel(los, by, -3.2, 1.0)), (contact_t, *rel(los, by, -4.5, -2.5)), (td_t, *rel(los, by, -5.5, -4.3)),
        (5.5, *rel(los, by, -6.5, -6.5)), (end_t, *rel(los, by, -7.0, -7.0))],
        extra={"ball_carrier": True})
    nacua = Mover("LA12", "LA", "offense", "12", "Puka Nacua", "WR", "WR", 90, [
        (0, *rel(los, by, -9.0, 0.8)), (snap_t, *rel(los, by, -9.0, 0.8)), (td_t, *rel(los, by, -10.0, -2.5)), (end_t, *rel(los, by, -10.0, -3.0))],
        extra={"block": {"assignment": "stalk_block", "engaged_with": "CHI29"}})
    adams = Mover("LA17", "LA", "offense", "17", "Davante Adams", "WR", "WR", 90, [
        (0, *rel(los, by, 14.0, 0.8)), (snap_t, *rel(los, by, 14.0, 0.8)), (td_t, *rel(los, by, 13.0, -3.0)), (end_t, *rel(los, by, 13.0, -3.5))],
        extra={"block": {"assignment": "stalk_block", "engaged_with": "CHI1"}})
    whit = Mover("LA88", "LA", "offense", "88", "Jordan Whittington", "WR", "WR", 90, [
        (0, *rel(los, by, 7.0, 1.0)), (snap_t, *rel(los, by, 7.0, 1.0)), (td_t, *rel(los, by, 5.5, -2.5)), (end_t, *rel(los, by, 5.5, -3.0))],
        extra={"block": {"assignment": "stalk_block", "engaged_with": "CHI6"}})

    d = []
    d.append(Mover("CHI98", "CHI", "defense", "98", "Montez Sweat", "DE", "EDGE", 270, [
        (0, *rel(los, by, -5.2, -1.0)), (snap_t, *rel(los, by, -5.2, -1.0)), (contact_t, *rel(los, by, -6.5, -1.8)), (end_t, *rel(los, by, -7.0, -2.5))],
        extra={"technique": "7", "stance": "3pt"}))
    d.append(Mover("CHI54", "CHI", "defense", "54", "Dayo Odeyingbo", "DE", "EDGE", 270, [
        (0, *rel(los, by, 3.6, -1.0)), (snap_t, *rel(los, by, 3.6, -1.0)), (contact_t, *rel(los, by, 2.5, -0.5)), (end_t, *rel(los, by, 0.5, -1.5))],
        extra={"technique": "5", "stance": "3pt"}))
    d.append(Mover("CHI97", "CHI", "defense", "97", "Grady Jarrett", "DT", "DT", 270, [
        (0, *rel(los, by, -1.8, -1.1)), (snap_t, *rel(los, by, -1.8, -1.1)), (contact_t, *rel(los, by, -2.4, -0.3)), (end_t, *rel(los, by, -3.0, -1.5))],
        extra={"technique": "3", "stance": "3pt"}))
    d.append(Mover("CHI99", "CHI", "defense", "99", "Gervon Dexter", "DT", "DT", 270, [
        (0, *rel(los, by, 0.8, -1.1)), (snap_t, *rel(los, by, 0.8, -1.1)), (contact_t, *rel(los, by, 0.4, -0.2)), (end_t, *rel(los, by, -1.5, -1.5))],
        extra={"technique": "1", "stance": "3pt"}))
    d.append(Mover("CHI53", "CHI", "defense", "53", "T.J. Edwards", "LB", "LB", 270, [
        (0, *rel(los, by, 0.0, -3.5)), (snap_t, *rel(los, by, 0.0, -3.5)), (4.2, *rel(los, by, -2.0, -2.5)), (contact_t, *rel(los, by, -4.0, -2.8)), (td_t, *rel(los, by, -5.0, -4.0)), (end_t, *rel(los, by, -6.0, -6.0))],
        extra={"coverage": {"assignment": "run_fit_B_right"}}))
    d.append(Mover("CHI49", "CHI", "defense", "49", "Tremaine Edmunds", "LB", "LB", 270, [
        (0, *rel(los, by, -3.0, -3.6)), (snap_t, *rel(los, by, -3.0, -3.6)), (4.2, *rel(los, by, -4.5, -2.8)), (contact_t, *rel(los, by, -4.8, -2.6)), (td_t, *rel(los, by, -5.6, -4.2)), (end_t, *rel(los, by, -6.5, -6.5))],
        extra={"coverage": {"assignment": "run_fit_C_right"}}))
    d.append(Mover("CHI9", "CHI", "defense", "9", "Jaquan Brisker", "S", "SS", 270, [
        (0, *rel(los, by, 4.0, -3.6)), (snap_t, *rel(los, by, 4.0, -3.6)), (4.2, *rel(los, by, 1.0, -2.5)), (contact_t, *rel(los, by, -2.5, -2.8)), (td_t, *rel(los, by, -4.5, -4.5)), (end_t, *rel(los, by, -5.5, -6.5))],
        extra={"coverage": {"assignment": "box_safety"}}))
    d.append(Mover("CHI1", "CHI", "defense", "1", "Jaylon Johnson", "CB", "CB", 270, [
        (0, *rel(los, by, 14.0, -2.0)), (snap_t, *rel(los, by, 14.0, -2.0)), (td_t, *rel(los, by, 12.5, -3.5)), (end_t, *rel(los, by, 11.0, -4.5))],
        extra={"coverage": {"assignment": "man", "man_zone": "man"}}))
    d.append(Mover("CHI29", "CHI", "defense", "29", "Tyrique Stevenson", "CB", "CB", 270, [
        (0, *rel(los, by, -9.5, -2.0)), (snap_t, *rel(los, by, -9.5, -2.0)), (contact_t, *rel(los, by, -9.0, -2.8)), (td_t, *rel(los, by, -7.5, -4.0)), (end_t, *rel(los, by, -7.5, -5.0))],
        extra={"coverage": {"assignment": "man", "man_zone": "man"}}))
    d.append(Mover("CHI6", "CHI", "defense", "6", "Kyler Gordon", "CB", "NB", 270, [
        (0, *rel(los, by, 7.0, -3.0)), (snap_t, *rel(los, by, 7.0, -3.0)), (td_t, *rel(los, by, 5.0, -3.2)), (end_t, *rel(los, by, 3.0, -4.5))],
        extra={"coverage": {"assignment": "man", "man_zone": "man"}}))
    d.append(Mover("CHI35", "CHI", "defense", "35", "C.J. Gardner-Johnson", "S", "FS", 270, [
        (0, *rel(los, by, 0.0, -8.0)), (snap_t, *rel(los, by, 0.0, -8.0)), (contact_t, *rel(los, by, -3.5, -5.5)), (td_t, *rel(los, by, -5.0, -5.0)), (end_t, *rel(los, by, -6.0, -6.0))],
        extra={"coverage": {"assignment": "deep_middle", "man_zone": "zone"}}))

    movers = ol + [higbee, qb, rb, nacua, adams, whit] + d + officials(los, by, snap_t, end_t)

    def carrier(f):
        return "LA9" if f / FPS < hand_t else "LA23"

    play = {
        "uid": "2025_wk20_LA-CHI_p019", "nfl_play_id": 480, "los_x": los, "ball_y": by,
        "snap_frame": int(snap_t * FPS), "whistle_frame": int(whistle_t * FPS),
        "situation": {"quarter": 1, "clock": "6:24", "down": 2, "distance": 4, "possession": "LA",
                      "yardline_100": 4, "yrdln": "CHI 4", "los_x": los, "first_down_x": GOAL_X,
                      "direction_of_attack": "right", "hash": "M", "score": {"away": 0, "home": 0}},
        "official": {"play_type": "run", "description": "23-K.Williams right tackle for 4 yards, TOUCHDOWN.",
                     "yards_gained": 4, "first_down": True, "touchdown": True, "turnover": False,
                     "players_involved": [{"jersey": "23", "team": "LA", "role": "rusher", "name": "K.Williams"}],
                     "labels": {"offense_formation": "UNDER CENTER", "offense_personnel": "11",
                                "defense_personnel": "nickel", "source": "nflverse"}},
        "personnel": {"offense": {"grouping": "11", "rb": 1, "te": 1, "wr": 3, "source": "human", "confidence": 1.0},
                      "defense": {"package": "nickel", "dl": 4, "lb": 2, "db": 5, "source": "human", "confidence": 1.0}},
        "pre_snap": {
            "formation_initial": {"frame": 20, "name": "i_form_tight_right", "strength": "right", "qb": "under_center",
                                  "backfield": "dot", "source": "human", "confidence": 1.0},
            "shifts": [], "motions": [],
            "formation_at_snap": {"frame": int(snap_t * FPS), "name": "i_form_tight_right", "strength": "right",
                                  "qb": "under_center", "backfield": "dot", "source": "human", "confidence": 1.0},
            "defense": {"front": "even", "shell": "1-high", "box_count": 7, "safeties_depth_yds": [8.0, 3.6],
                        "press_corners": [True, True], "source": "human", "confidence": 1.0}},
        "in_motion": [],
        "live": {
            "play_family": "run", "play_action": False, "rpo": False,
            "qb": {"track_id": "LA9", "drop": "none", "time_to_throw_s": None, "pocket": None, "scramble": False},
            "routes": [],
            "run": {"ball_carrier": "LA23", "gap": "C-right", "scheme": "duo", "handoff_frame": int(hand_t * FPS),
                    "source": "human", "confidence": 1.0},
            "blocks": [{"track_id": "LA79", "assignment": "run_block", "engaged_with": "CHI98", "engaged_frames": [int(snap_t * FPS) + 5, int(td_t * FPS)]},
                       {"track_id": "LA89", "assignment": "run_block", "engaged_with": "CHI98", "engaged_frames": [int(snap_t * FPS) + 6, int(td_t * FPS)]},
                       {"track_id": "LA69", "assignment": "run_block", "engaged_with": "CHI97", "engaged_frames": [int(snap_t * FPS) + 5, int(td_t * FPS)]}],
            "defense": {"coverage": {"family": "cover_1", "man_zone": "man", "rotation": None, "source": "human", "confidence": 1.0},
                        "pass_rush": [], "blitz": False},
            "ball": {"release_frame": None, "arrival_frame": None, "air_yards": None, "outcome": None},
            "events": [{"frame": int(snap_t * FPS), "type": "snap"},
                       {"frame": int(hand_t * FPS), "type": "handoff", "track_id": "LA23"},
                       {"frame": int(contact_t * FPS), "type": "first_contact", "track_id": "LA23", "by": ["CHI49", "CHI9"]},
                       {"frame": int(td_t * FPS), "type": "touchdown", "track_id": "LA23"},
                       {"frame": int(whistle_t * FPS), "type": "whistle"}]},
        "post_snap": {"result": "touchdown", "yards_gained": 4, "yards_after_catch": None,
                      "ball_spot": {"x": 110.0, "y": round(by - 4.3, 1)}, "first_down": True,
                      "tackle": None, "out_of_bounds": False, "penalty": None, "end_frame": frames - 1},
    }
    return build(play, movers, {"flight": {}, "carrier": carrier}, frames)


# ------------------------------------------------------------------ play 3: p062 ----------

def play_062() -> dict:
    """LA 2nd & 3 at LA 27, no huddle: jet motion by Nacua, Stafford deep right to Higbee for 27."""
    los, by = 37.0, HASH_R
    snap_t, throw_t, catch_t, contact_t, tackle_t, whistle_t, end_t = 3.0, 5.2, 6.4, 7.1, 7.4, 7.9, 9.0
    frames = int(end_t * FPS)

    ol = rams_oline(los, by)
    pass_set(ol, los, by, snap_t, tackle_t, depth_gain=2.2)
    qb = Mover("LA9", "LA", "offense", "9", "Matthew Stafford", "QB", "QB", 90, [
        (0, *rel(los, by, 0, 5.0)), (snap_t, *rel(los, by, 0, 5.0)), (snap_t + 0.3, *rel(los, by, 0.5, 5.5)),   # fake to the jet
        (snap_t + 1.2, *rel(los, by, 0.5, 8.0)), (throw_t, *rel(los, by, 1.0, 8.5)), (end_t, *rel(los, by, 1.0, 8.5))],
        extra={"qb": True})
    rb = Mover("LA23", "LA", "offense", "23", "Kyren Williams", "RB", "RB", 90, [
        (0, *rel(los, by, 1.8, 5.0)), (snap_t, *rel(los, by, 1.8, 5.0)), (snap_t + 0.8, *rel(los, by, 3.5, 5.5)),
        (throw_t, *rel(los, by, 4.0, 6.0)), (end_t, *rel(los, by, 4.0, 6.0))],
        extra={"block": {"assignment": "pass_pro", "engaged_with": "CHI54"}})
    nacua = Mover("LA12", "LA", "offense", "12", "Puka Nacua", "WR", "WR", 90, [
        (0, *rel(los, by, 9.0, 1.0)), (1.4, *rel(los, by, 9.0, 1.0)),
        (2.2, *rel(los, by, 4.0, 2.2)), (snap_t, *rel(los, by, -3.0, 2.4)),      # jet motion, still moving at the snap
        (snap_t + 0.5, *rel(los, by, -8.0, 1.0)), (snap_t + 1.4, *rel(los, by, -14.0, -3.0)),
        (throw_t, *rel(los, by, -17.0, -6.0)), (end_t, *rel(los, by, -18.0, -8.0))],
        extra={"motion": {"type": "jet", "direction": "right", "start_frame": 35, "end_frame": int(snap_t * FPS), "at_snap": "in_motion"},
               "route": "swing", "route_detail": {"route": "swing", "tree": None, "break_frame": int((snap_t + 0.5) * FPS),
                                                  "depth_at_break_yds": -1.0, "break_direction": "right", "targeted": False}})
    higbee = Mover("LA89", "LA", "offense", "89", "Tyler Higbee", "TE", "TE", 90, [
        (0, *rel(los, by, -4.2, 0.7)), (snap_t, *rel(los, by, -4.2, 0.7)), (snap_t + 1.5, *rel(los, by, -4.5, -10.0)),
        (snap_t + 2.4, *rel(los, by, -9.0, -17.0)),                                   # bends to the corner
        (catch_t, *rel(los, by, -16.5, -22.0)), (contact_t, 62.5, 9.0), (tackle_t, 64.0, 8.0), (end_t, 64.5, 7.8)],
        extra={"route": "corner", "route_detail": {"route": "corner", "tree": 7, "break_frame": int((snap_t + 1.5) * FPS),
                                                   "depth_at_break_yds": 10.0, "break_direction": "right", "targeted": True,
                                                   "catch_frame": int(catch_t * FPS)}})
    adams = Mover("LA17", "LA", "offense", "17", "Davante Adams", "WR", "WR", 90, [
        (0, *rel(los, by, 18.0, 0.8)), (snap_t, *rel(los, by, 18.0, 0.8)), (snap_t + 1.7, *rel(los, by, 17.5, -12.0)),
        (throw_t + 0.6, *rel(los, by, 9.0, -13.0)), (end_t, *rel(los, by, 4.0, -13.5))],
        extra={"route": "dig", "route_detail": {"route": "dig", "tree": 6, "break_frame": int((snap_t + 1.7) * FPS),
                                                "depth_at_break_yds": 12.0, "break_direction": "right", "targeted": False}})
    whit = Mover("LA88", "LA", "offense", "88", "Jordan Whittington", "WR", "WR", 90, [
        (0, *rel(los, by, -16.0, 0.8)), (snap_t, *rel(los, by, -16.0, 0.8)), (throw_t + 0.8, *rel(los, by, -17.5, -26.0)), (end_t, *rel(los, by, -18.0, -34.0))],
        extra={"route": "go", "route_detail": {"route": "go", "tree": 9, "break_frame": None, "depth_at_break_yds": None,
                                               "break_direction": None, "targeted": False}})

    d = []
    d.append(Mover("CHI98", "CHI", "defense", "98", "Montez Sweat", "DE", "EDGE", 270, [
        (0, *rel(los, by, -5.2, -1.0)), (snap_t, *rel(los, by, -5.2, -1.0)), (snap_t + 0.7, *rel(los, by, -6.0, 1.5)), (throw_t, *rel(los, by, -3.5, 6.5)), (end_t, *rel(los, by, -2.5, 8.0))],
        extra={"technique": "7", "stance": "3pt", "pass_rush": {"rushed": True, "path": "edge_right", "pressure": True}}))
    d.append(Mover("CHI54", "CHI", "defense", "54", "Dayo Odeyingbo", "DE", "EDGE", 270, [
        (0, *rel(los, by, 4.5, -1.0)), (snap_t, *rel(los, by, 4.5, -1.0)), (snap_t + 0.7, *rel(los, by, 5.0, 1.5)), (throw_t, *rel(los, by, 4.5, 4.5)), (end_t, *rel(los, by, 4.0, 5.5))],
        extra={"technique": "5", "stance": "3pt", "pass_rush": {"rushed": True, "path": "edge_left", "pressure": False}}))
    d.append(Mover("CHI97", "CHI", "defense", "97", "Grady Jarrett", "DT", "DT", 270, [
        (0, *rel(los, by, -1.8, -1.1)), (snap_t, *rel(los, by, -1.8, -1.1)), (throw_t, *rel(los, by, -2.2, 1.8)), (end_t, *rel(los, by, -2.5, 2.5))],
        extra={"technique": "3", "stance": "3pt", "pass_rush": {"rushed": True, "path": "interior", "pressure": False}}))
    d.append(Mover("CHI99", "CHI", "defense", "99", "Gervon Dexter", "DT", "DT", 270, [
        (0, *rel(los, by, 0.8, -1.1)), (snap_t, *rel(los, by, 0.8, -1.1)), (throw_t, *rel(los, by, 1.2, 1.5)), (end_t, *rel(los, by, 1.5, 2.2))],
        extra={"technique": "1", "stance": "3pt", "pass_rush": {"rushed": True, "path": "interior", "pressure": False}}))
    d.append(Mover("CHI53", "CHI", "defense", "53", "T.J. Edwards", "LB", "LB", 270, [
        (0, *rel(los, by, 0.5, -4.5)), (snap_t, *rel(los, by, 0.5, -4.5)), (snap_t + 0.8, *rel(los, by, 2.5, -3.5)), (throw_t, *rel(los, by, 3.5, -4.5)), (end_t, *rel(los, by, 2.0, -5.0))],
        extra={"coverage": {"assignment": "man_on_RB", "man_zone": "man"}}))
    d.append(Mover("CHI1", "CHI", "defense", "1", "Jaylon Johnson", "CB", "CB", 270, [
        (0, *rel(los, by, 17.0, -6.0)), (snap_t, *rel(los, by, 17.0, -6.0)), (snap_t + 1.9, *rel(los, by, 17.0, -13.5)), (throw_t + 0.6, *rel(los, by, 10.0, -14.5)), (end_t, *rel(los, by, 5.0, -15.0))],
        extra={"coverage": {"assignment": "man", "man_zone": "man", "on": "LA17"}}))
    d.append(Mover("CHI29", "CHI", "defense", "29", "Tyrique Stevenson", "CB", "CB", 270, [
        (0, *rel(los, by, -16.0, -6.0)), (snap_t, *rel(los, by, -16.0, -6.0)), (throw_t + 0.8, *rel(los, by, -17.0, -27.0)), (end_t, *rel(los, by, -17.5, -35.0))],
        extra={"coverage": {"assignment": "man", "man_zone": "man", "on": "LA88"}}))
    d.append(Mover("CHI6", "CHI", "defense", "6", "Kyler Gordon", "CB", "NB", 270, [
        (0, *rel(los, by, 9.0, -4.0)), (1.4, *rel(los, by, 9.0, -4.0)), (2.2, *rel(los, by, 4.5, -4.0)),
        (snap_t, *rel(los, by, -2.5, -4.0)),                                          # travels with the jet motion: man
        (snap_t + 0.6, *rel(los, by, -8.0, -3.0)), (throw_t, *rel(los, by, -15.0, -4.0)), (end_t, *rel(los, by, -16.0, -6.0))],
        extra={"coverage": {"assignment": "man", "man_zone": "man", "on": "LA12"}}))
    d.append(Mover("CHI20", "CHI", "defense", "20", None, "CB", "CB", 270, [
        (0, *rel(los, by, -6.0, -6.0)), (snap_t, *rel(los, by, -6.0, -6.0)), (snap_t + 1.5, *rel(los, by, -3.0, -9.0)),
        (snap_t + 2.4, *rel(los, by, -6.5, -15.5)), (catch_t, *rel(los, by, -14.0, -20.5)), (tackle_t, 63.0, 9.5), (end_t, 63.5, 9.0)],
        extra={"coverage": {"assignment": "man", "man_zone": "man", "on": "LA89"}}))
    d.append(Mover("CHI9", "CHI", "defense", "9", "Jaquan Brisker", "S", "SS", 270, [
        (0, *rel(los, by, -8.0, -10.0)), (snap_t, *rel(los, by, -8.0, -10.0)), (throw_t, *rel(los, by, -6.0, -12.0)), (tackle_t, 60.0, 14.0), (end_t, 61.0, 12.0)],
        extra={"coverage": {"assignment": "robber", "man_zone": "zone"}}))
    d.append(Mover("CHI35", "CHI", "defense", "35", "C.J. Gardner-Johnson", "S", "FS", 270, [
        (0, *rel(los, by, 2.0, -14.0)), (snap_t, *rel(los, by, 2.0, -14.0)), (throw_t, *rel(los, by, -2.0, -20.0)),
        (catch_t, *rel(los, by, -13.0, -24.0)), (contact_t, 62.0, 9.8), (tackle_t, 64.0, 8.5), (end_t, 64.5, 8.5)],
        extra={"coverage": {"assignment": "deep_middle", "man_zone": "zone"}}))

    movers = ol + [qb, rb, nacua, higbee, adams, whit] + d + officials(los, by, snap_t, end_t)
    flight = ball_flight(qb.at(throw_t), higbee.at(catch_t), throw_t, catch_t, peak_z=9.0)

    def carrier(f):
        return "LA9" if f / FPS < catch_t else "LA89"

    play = {
        "uid": "2025_wk20_LA-CHI_p062", "nfl_play_id": 1640, "los_x": los, "ball_y": by,
        "snap_frame": int(snap_t * FPS), "whistle_frame": int(whistle_t * FPS),
        "situation": {"quarter": 2, "clock": "0:45", "down": 2, "distance": 3, "possession": "LA",
                      "yardline_100": 73, "yrdln": "LA 27", "los_x": los, "first_down_x": los + 3,
                      "direction_of_attack": "right", "hash": "R", "score": {"away": 7, "home": 7}},
        "official": {"play_type": "pass",
                     "description": "(No Huddle, Shotgun) 9-M.Stafford pass deep right to 89-T.Higbee to CHI 46 for 27 yards (35-C.Gardner-Johnson).",
                     "yards_gained": 27, "first_down": True, "touchdown": False, "turnover": False,
                     "players_involved": [{"jersey": "9", "team": "LA", "role": "passer", "name": "M.Stafford"},
                                          {"jersey": "89", "team": "LA", "role": "receiver", "name": "T.Higbee"},
                                          {"jersey": "35", "team": "CHI", "role": "tackler", "name": "C.Gardner-Johnson"}],
                     "labels": {"offense_formation": "SHOTGUN", "offense_personnel": "11",
                                "defense_personnel": "dime", "source": "nflverse"}},
        "personnel": {"offense": {"grouping": "11", "rb": 1, "te": 1, "wr": 3, "source": "human", "confidence": 1.0},
                      "defense": {"package": "dime", "dl": 4, "lb": 1, "db": 6, "source": "human", "confidence": 1.0}},
        "pre_snap": {
            "formation_initial": {"frame": 20, "name": "gun_2x2_slot_left", "strength": "left", "qb": "shotgun",
                                  "backfield": "offset_left", "source": "human", "confidence": 1.0},
            "shifts": [],
            "motions": [{"track_id": "LA12", "type": "jet", "direction": "right", "start_frame": 35, "end_frame": int(snap_t * FPS),
                         "at_snap": "in_motion", "speed_at_snap_yds_s": 7.8}],
            "formation_at_snap": {"frame": int(snap_t * FPS), "name": "gun_3x1_right_jet", "strength": "right",
                                  "qb": "shotgun", "backfield": "offset_left", "source": "human", "confidence": 1.0},
            "defense": {"front": "even", "shell": "1-high", "box_count": 5, "safeties_depth_yds": [14.0, 10.0],
                        "press_corners": [False, False], "source": "human", "confidence": 1.0}},
        "in_motion": ["LA12"],
        "live": {
            "play_family": "dropback", "play_action": False, "rpo": False,
            "qb": {"track_id": "LA9", "drop": "3-step", "time_to_throw_s": round(throw_t - snap_t, 1),
                   "pocket": "clean", "scramble": False},
            "routes": [
                {"track_id": "LA89", "route": "corner", "tree": 7, "break_frame": int((snap_t + 1.5) * FPS), "depth_at_break_yds": 10.0,
                 "break_direction": "right", "targeted": True, "catch_frame": int(catch_t * FPS), "source": "human", "confidence": 1.0},
                {"track_id": "LA17", "route": "dig", "tree": 6, "break_frame": int((snap_t + 1.7) * FPS), "depth_at_break_yds": 12.0,
                 "break_direction": "right", "targeted": False, "source": "human", "confidence": 1.0},
                {"track_id": "LA88", "route": "go", "tree": 9, "break_frame": None, "depth_at_break_yds": None,
                 "break_direction": None, "targeted": False, "source": "human", "confidence": 1.0},
                {"track_id": "LA12", "route": "swing", "tree": None, "break_frame": int((snap_t + 0.5) * FPS), "depth_at_break_yds": -1.0,
                 "break_direction": "right", "targeted": False, "source": "human", "confidence": 1.0}],
            "run": None,
            "blocks": [{"track_id": "LA23", "assignment": "pass_pro", "engaged_with": "CHI54", "engaged_frames": [int((snap_t + 0.8) * FPS), int(throw_t * FPS)]}],
            "defense": {"coverage": {"family": "cover_1", "man_zone": "man", "rotation": None, "source": "human", "confidence": 0.8},
                        "pass_rush": [{"track_id": "CHI98", "rushed": True, "path": "edge_right"}, {"track_id": "CHI54", "rushed": True, "path": "edge_left"},
                                      {"track_id": "CHI97", "rushed": True, "path": "interior"}, {"track_id": "CHI99", "rushed": True, "path": "interior"}],
                        "blitz": False},
            "ball": {"release_frame": int(throw_t * FPS), "arrival_frame": int(catch_t * FPS), "air_yards": 22, "outcome": "complete"},
            "events": [{"frame": int(snap_t * FPS), "type": "snap"},
                       {"frame": int(throw_t * FPS), "type": "pass_forward", "track_id": "LA9"},
                       {"frame": int(catch_t * FPS), "type": "pass_outcome_caught", "track_id": "LA89"},
                       {"frame": int(contact_t * FPS), "type": "first_contact", "track_id": "LA89", "by": ["CHI35"]},
                       {"frame": int(tackle_t * FPS), "type": "tackle", "track_id": "LA89", "by": ["CHI35"]},
                       {"frame": int(whistle_t * FPS), "type": "whistle"}]},
        "post_snap": {"result": "complete", "yards_gained": 27, "yards_after_catch": 5,
                      "ball_spot": {"x": 64.0, "y": 8.0}, "first_down": True,
                      "tackle": {"frame": int(tackle_t * FPS), "x": 64.0, "y": 8.0, "by": ["CHI35"]},
                      "out_of_bounds": False, "penalty": None, "end_frame": frames - 1},
    }
    return build(play, movers, {"flight": flight, "carrier": carrier}, frames)


def main() -> int:
    records = {r["play_uid"]: r for r in (play_001(), play_019(), play_062())}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text("// Generated by scripts/demo/synth.py. Hand-made example records, not pipeline output.\n"
                   f"window.FZ_DEMO_RECORDS = {payload};\n")
    for uid, r in records.items():
        print(f"{uid}: {r['tracking']['frames']} frames, {len(r['players'])} entities, "
              f"{len(r['live']['events'])} events")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
