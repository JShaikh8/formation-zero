"""fz-nfl: pull a game from api.nfl.com into data/pbp/<game_key>.parquet.

    fz-nfl games --season 2025 --week 20            # list the week's games and slugs
    fz-nfl pull  --season 2025 --week 20 --away LA --home CHI

Regular-season weeks 1-18 map to type REG; later weeks to POST with the week renumbered from 1.
That mapping is an assumption until checked against a real weekly payload (story GRD-20).
"""

from __future__ import annotations

import argparse
import json
import sys

from formation_zero.data.nflapi.client import NflApi, NotConfigured
from formation_zero.data.nflapi.game_detail import adapt_game_detail, to_frame
from formation_zero.data.nflapi.teams import abbreviation_from_logo


def season_type_and_week(week: int) -> tuple[str, int]:
    return ("REG", week) if week <= 18 else ("POST", week - 18)


def find_slug(api: NflApi, season: int, week: int, away: str, home: str) -> str | None:
    stype, w = season_type_and_week(week)
    games = api.weekly_games(season, stype, w)
    for g in games if isinstance(games, list) else []:
        h = abbreviation_from_logo((g.get("homeTeam") or {}).get("currentLogo"))
        a = abbreviation_from_logo((g.get("awayTeam") or {}).get("currentLogo"))
        if h == home.upper() and a == away.upper():
            return next((e.get("id") for e in (g.get("externalIds") or []) if e.get("source") == "slug"), None)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games"); g.add_argument("--season", type=int, required=True); g.add_argument("--week", type=int, required=True)
    p = sub.add_parser("pull")
    for k in ("--season", "--week"):
        p.add_argument(k, type=int, required=True)
    p.add_argument("--away", required=True); p.add_argument("--home", required=True); p.add_argument("--out", default="data")
    a = ap.parse_args(argv)

    from pathlib import Path
    api = NflApi(cache_dir=Path(a.out if a.cmd == "pull" else "data") / "pbp" / "nfl_raw")
    if not api.is_configured():
        print("api.nfl.com is not configured on this machine: set NFL_API_CLIENT_ID, NFL_API_CLIENT_KEY, "
              "NFL_API_CLIENT_SECRET, NFL_API_DEVICE_ID in .env (org-issued). Nothing pulled.", file=sys.stderr)
        return 2
    try:
        if a.cmd == "games":
            stype, w = season_type_and_week(a.week)
            for gm in api.weekly_games(a.season, stype, w):
                slug = next((e.get("id") for e in (gm.get("externalIds") or []) if e.get("source") == "slug"), "?")
                print(f"{abbreviation_from_logo((gm.get('awayTeam') or {}).get('currentLogo')):>3} at "
                      f"{abbreviation_from_logo((gm.get('homeTeam') or {}).get('currentLogo')):<3}  {slug}")
            return 0
        slug = find_slug(api, a.season, a.week, a.away, a.home)
        if not slug:
            print("game not found in the weekly slate", file=sys.stderr)
            return 1
        from formation_zero.ids import game_key
        gk = game_key(a.season, a.week, a.away, a.home)
        detail = api.game_detail(slug, cache_key=gk)
        adapted = adapt_game_detail(detail, season=a.season, week=a.week)
        df = to_frame(adapted)
        out = Path(a.out) / "pbp" / f"{gk}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.drop(columns=["players"]).to_parquet(out, index=False)
        (out.with_suffix(".players.json")).write_text(json.dumps(
            {r["play_uid"]: r["players"] for r in adapted["plays"] if r["play_uid"]}, indent=1))
        print(f"Wrote {out}: {len(df)} rows, {int(df['play_index'].notna().sum())} filmable plays, "
              f"{int(df['is_scrimmage'].sum())} scrimmage plays")
        return 0
    except NotConfigured as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
