"""Pull play-by-play (PBP) for one game from nflverse via nfl_data_py.

This is the **labels + alignment backbone** of the whole system:
  - down, distance, line-of-scrimmage (`yardline_100`) for every play,
  - personnel / formation (from participation data) as ground truth,
  - a `play_index` over filmable plays (kicks and kneels included) to align segmented film clips against (Nth film play
    -> Nth scrimmage play), since clean coaches film has no on-screen clock to match on.

The LOS yard line is what later anchors field registration to a known landmark.

CLI:
    fz-pull-pbp --season 2024 --week 5 --away BUF --home NYJ --out data
"""

from __future__ import annotations

import argparse

from formation_zero.ids import game_key, nflverse_game_id, play_uid

# Plays that typically appear in offensive All-22 coaches film, in order.
from formation_zero.data.playtypes import FILMABLE_PLAY_TYPES, SCRIMMAGE_PLAY_TYPES  # noqa: E402

# Tidy output columns (subset of what's available is kept gracefully).
_BASE_COLS = [
    "play_id", "qtr", "down", "ydstogo", "yardline_100", "yrdln", "side_of_field",
    "posteam", "defteam", "play_type", "epa", "desc",
    # analytics / efficiency + situational (kept only if present in this season's PBP)
    "yards_gained", "success", "wpa", "first_down", "goal_to_go",
    "shotgun", "no_huddle", "qb_dropback", "qb_scramble", "play_action",
    "pass_location", "pass_length", "air_yards", "yards_after_catch",
    "run_location", "run_gap", "sack", "interception", "touchdown",
]
_PARTICIPATION_COLS = [
    "offense_formation", "offense_personnel", "defense_personnel", "defenders_in_box",
]


def pull_game_pbp(season: int, week: int, away: str, home: str,
                  include_participation: bool = True):
    """Return a tidy per-play DataFrame for one game. Raises with guidance if not found."""
    try:
        import nfl_data_py as nfl
    except ImportError as e:  # pragma: no cover - environment guidance
        raise SystemExit(
            "nfl_data_py is not installed. Run:  pip install -e .  (from ~/projects/formation_zero)"
        ) from e
    import pandas as pd

    gid = nflverse_game_id(season, week, away, home)

    pbp = nfl.import_pbp_data([season], downcast=True, cache=False)
    game = pbp[pbp["game_id"] == gid].copy()
    if game.empty:
        avail = sorted(pbp.loc[pbp["week"] == week, "game_id"].unique())
        raise SystemExit(
            f"No PBP rows for game_id {gid!r}.\n"
            f"Games in {season} week {week}:\n  " + "\n  ".join(avail)
        )

    game = game.sort_values("play_id").reset_index(drop=True)

    # Modern nfl_data_py ships personnel/formation directly in PBP. Only fall back to the
    # (older) participation feed if those columns are absent.
    if include_participation and "offense_personnel" not in game.columns:
        game = _join_participation(nfl, season, game, pd)

    keep = [c for c in (_BASE_COLS + _PARTICIPATION_COLS) if c in game.columns]
    tidy = game[keep].copy()

    tidy.insert(0, "game_key", game_key(season, week, away, home))
    tidy["is_scrimmage"] = tidy["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
    tidy["is_filmable"] = tidy["play_type"].isin(FILMABLE_PLAY_TYPES)

    tidy = assign_play_index(tidy, gk)

    return tidy


def _join_participation(nfl, season: int, game, pd):
    """Best-effort join of the older participation personnel/formation feed onto the game frame."""
    import_part = getattr(nfl, "import_participation_data", None)
    if import_part is None:
        return game  # removed in modern nfl_data_py; personnel comes from PBP instead
    try:
        part = import_part([season])
    except Exception:  # data may be unavailable for this season
        return game
    cols = ["play_id"] + [c for c in _PARTICIPATION_COLS if c in part.columns]
    gid = game["game_id"].iloc[0] if "game_id" in game.columns else None
    if "game_id" in part.columns and gid is not None:
        part = part[part["game_id"] == gid]
    part = part[[c for c in cols if c in part.columns]].drop_duplicates("play_id")
    return game.merge(part, on="play_id", how="left")


def write_game_pbp(season: int, week: int, away: str, home: str, out: str = "data") -> str:
    """Pull + write data/pbp/<game_key>.parquet. Returns the output path."""
    from formation_zero.data.layout import GamePaths

    tidy = pull_game_pbp(season, week, away, home)
    paths = GamePaths(out, season, week, away, home).ensure_dirs()
    tidy.to_parquet(paths.pbp_path, index=False)

    n_scrim = int(tidy["is_scrimmage"].sum())
    print(f"Wrote {paths.pbp_path}")
    print(f"  {len(tidy)} plays total, {n_scrim} scrimmage plays (your film should align to these).")
    cols = tidy.columns.tolist()
    if "offense_personnel" in cols and tidy["offense_personnel"].notna().any():
        print("  personnel/formation present — usable as ground truth for the intelligence layer.")
    else:
        print("  note: no personnel/formation columns for this season; down/distance/LOS still present.")
    return str(paths.pbp_path)



def assign_play_index(tidy, gk: str):
    """Number every FILMABLE play in order (kicks and kneels included) and derive play_uid.

    The film index pairs the Nth film play with the Nth filmable row, so the two must count the
    same things. Administrative rows keep a null index and no uid. Idempotent: re-running on a
    table that already has the columns recomputes them.
    """
    if "is_filmable" not in tidy.columns:
        tidy["is_filmable"] = tidy["play_type"].isin(FILMABLE_PLAY_TYPES)
    idx = 0
    play_index: list = []
    uids: list = []
    for filmable in tidy["is_filmable"]:
        if bool(filmable):
            idx += 1
            play_index.append(idx)
            uids.append(play_uid(gk, idx))
        else:
            play_index.append(None)
            uids.append(None)
    for col in ("play_uid", "play_index"):
        if col in tidy.columns:
            tidy = tidy.drop(columns=[col])
    tidy.insert(1, "play_uid", uids)
    tidy.insert(2, "play_index", play_index)
    return tidy

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pull one game's play-by-play from nflverse.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--away", required=True, help="away team abbrev, e.g. BUF")
    p.add_argument("--home", required=True, help="home team abbrev, e.g. NYJ")
    p.add_argument("--out", default="data", help="data root (default: data)")
    args = p.parse_args(argv)
    write_game_pbp(args.season, args.week, args.away, args.home, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
