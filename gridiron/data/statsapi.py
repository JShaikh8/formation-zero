"""STATS Perform API (api.stats.com) client — schedule lookup + full event play-by-play.

An alternative PBP source to nflverse (see gridiron.data.pbp). Two reasons it matters here:

  1. nflverse participation (personnel/formation) is FTN-sourced, post-season only, and has no
     2025+ data — so nflverse alone can't label recent film.
  2. The STATS feed is per-*event*, keyed by an ``eventId`` resolved from the season schedule,
     and ships box + drive charts alongside the play list.

Auth
----
``sig = sha256(api_key + api_secret + <unix seconds>)``, sent as ``&sig=``. Credentials come
from ``.env`` at the repo root (gitignored). If no secret is set, requests go out with
``api_key`` only, which some deployments accept.

Stdlib only — no new dependencies.

CLI::

    gridiron-stats events --season 2025 --team CHI     # find the eventId
    gridiron-stats pbp --event-id 2879591              # fetch + cache raw JSON
    gridiron-stats pbp --event-id 2879591 --show       # ... and summarize its shape
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://api.stats.com/v1/stats/football/nfl"

# Walk depth cap when scanning an unknown response envelope for event records.
_MAX_WALK_DEPTH = 12


# ---- credentials ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Repo root (parent of the `gridiron` package)."""
    return Path(__file__).resolve().parents[2]


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Parse a .env file into a dict and populate os.environ (without clobbering existing vars).

    Deliberately minimal: `KEY=value` lines, `#` comments, optional surrounding quotes.
    """
    env_path = Path(path) if path else repo_root() / ".env"
    found: dict[str, str] = {}
    if not env_path.exists():
        return found
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        found[key] = value
        os.environ.setdefault(key, value)
    return found


def credentials() -> tuple[str, str]:
    """Return (api_key, api_secret). Raises with guidance if the key is missing."""
    load_env()
    key = os.environ.get("STATS_API_KEY", "").strip()
    secret = os.environ.get("STATS_API_SECRET", "").strip()
    if not key:
        raise SystemExit(
            f"STATS_API_KEY is empty. Add it to {repo_root() / '.env'}:\n"
            "    STATS_API_KEY=...\n"
            "    STATS_API_SECRET=..."
        )
    return key, secret


def signature(api_key: str, api_secret: str, now: int | None = None) -> str:
    """sha256(api_key + api_secret + unix_seconds) — the `sig` query parameter."""
    ts = int(time.time()) if now is None else int(now)
    return hashlib.sha256(f"{api_key}{api_secret}{ts}".encode()).hexdigest()


# ---- transport -----------------------------------------------------------------------------


def build_url(path: str, params: dict | None = None) -> str:
    """Absolute URL for an API path, with auth params appended."""
    api_key, api_secret = credentials()
    query = {"accept": "json", **(params or {}), "api_key": api_key}
    if api_secret:
        query["sig"] = signature(api_key, api_secret)
    return f"{BASE_URL}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"


def get(path: str, params: dict | None = None, timeout: int = 60) -> dict:
    """GET an API path and return parsed JSON. Surfaces HTTP errors with the response body."""
    url = build_url(path, params)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # pragma: no cover - network
        body = e.read().decode("utf-8", "replace")[:500]
        hint = ""
        if e.code in (401, 403):
            hint = "\nCheck STATS_API_KEY/STATS_API_SECRET; a stale clock also breaks `sig`."
        raise SystemExit(f"HTTP {e.code} for {path}: {body}{hint}") from e
    except urllib.error.URLError as e:  # pragma: no cover - network
        raise SystemExit(f"Network error for {path}: {e.reason}") from e


# ---- endpoints -----------------------------------------------------------------------------


def fetch_schedule(season: int) -> dict:
    """Full season schedule (raw envelope)."""
    return get("events/", {"season": season})


def fetch_teams(season: int) -> dict:
    """Team registry for a season (raw envelope): conferences -> divisions -> teams."""
    return get("teams/", {"season": season})


def fetch_event(event_id: int | str, *, box: bool = True, pbp: bool = True,
                drive_charts: bool = True, linescore: bool = True) -> dict:
    """One event with play-by-play (raw envelope)."""
    flags = {"box": box, "pbp": pbp, "driveCharts": drive_charts, "linescore": linescore}
    return get(f"events/{event_id}", {k: str(v).lower() for k, v in flags.items()})


# ---- envelope: the real STATS shape ---------------------------------------------------------
#
#   apiResults[0].league.season.eventType[] -> {eventTypeId, name, events[]}
#
# eventTypeId: 0 = Preseason, 1 = Regular Season, 2 = Postseason. This matters: preseason and
# regular season BOTH number weeks from 1, so (season, week) alone does not identify a game.

EVENT_TYPES = {0: "preseason", 1: "regular", 2: "postseason"}


def season_node(payload: dict) -> dict:
    """The `season` node of a response, or {} if the envelope differs."""
    try:
        return payload["apiResults"][0]["league"]["season"]
    except (KeyError, IndexError, TypeError):
        return {}


def iter_events(payload: dict):
    """Yield (event_type_id, event_type_name, event) for every event in a schedule payload.

    Falls back to a structural scan if the documented path is absent, so an envelope change
    degrades to "unlabelled events" rather than "no events".
    """
    groups = season_node(payload).get("eventType") or []
    if not groups:
        for event in extract_events(payload):
            yield event.get("eventTypeId"), None, event
        return
    for group in groups:
        type_id, type_name = group.get("eventTypeId"), group.get("name")
        for event in group.get("events") or []:
            yield type_id, type_name, event


# ---- envelope walking ----------------------------------------------------------------------
#
# The STATS envelope nests events under apiResults[].league.season.eventType[].events[], but the
# exact shape varies by sport/endpoint. Rather than hard-code that path, find event records by
# structure: any dict carrying an "eventId". Robust to envelope changes, and lets us inspect a
# real payload before committing to a normalizer.


def _walk(node, depth: int = 0):
    """Yield every dict nested anywhere in a JSON structure."""
    if depth > _MAX_WALK_DEPTH:
        return
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value, depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, depth + 1)


def extract_events(payload: dict) -> list[dict]:
    """Every event-shaped record (has an `eventId`) in a response, de-duplicated by id."""
    events: dict = {}
    for node in _walk(payload):
        if "eventId" in node and node["eventId"] not in events:
            events[node["eventId"]] = node
    return list(events.values())


def event_team_ids(event: dict) -> set[int]:
    """teamIds participating in an event. `teamId` is the join key — abbreviations are not."""
    ids = set()
    for team in event.get("teams") or []:
        if isinstance(team, dict) and team.get("teamId") is not None:
            ids.add(int(team["teamId"]))
    return ids


def home_away(event: dict) -> tuple[dict | None, dict | None]:
    """(home_team, away_team) records, read from `teamLocationType.name`."""
    home = away = None
    for team in event.get("teams") or []:
        name = (team.get("teamLocationType") or {}).get("name")
        if name == "home":
            home = team
        elif name == "away":
            away = team
    return home, away


def start_date(event: dict, date_type: str = "UTC") -> str:
    """Start date string for a given dateType ('UTC', 'Local', 'Eastern'), else any available."""
    dates = event.get("startDate")
    if isinstance(dates, str):
        return dates
    if not isinstance(dates, list):
        return ""
    for entry in dates:
        if isinstance(entry, dict) and entry.get("dateType") == date_type:
            return entry.get("full") or ""
    for entry in dates:
        if isinstance(entry, dict) and entry.get("full"):
            return entry["full"]
    return ""


def find_events(payload: dict, *, team_id: int | None = None, week: int | None = None,
                event_type_id: int | None = None) -> list[dict]:
    """Schedule events filtered by teamId, week, and/or eventTypeId.

    `event_type_id` is usually required alongside `week`: preseason and regular season both
    number weeks from 1.
    """
    hits = []
    for type_id, _type_name, event in iter_events(payload):
        if event_type_id is not None and type_id != event_type_id:
            continue
        if week is not None and event.get("week") != week:
            continue
        if team_id is not None and int(team_id) not in event_team_ids(event):
            continue
        hits.append(event)
    return hits


def describe_event(event: dict, registry=None) -> str:
    """One-line summary of an event, formatted `away @ home` with ids."""
    home, away = home_away(event)

    def label(team):
        if not team:
            return "?"
        tid = team.get("teamId")
        abbr = team.get("abbreviation") or (registry.abbreviation(tid) if registry else None)
        return f"{abbr}({tid})" if abbr else str(tid)

    title = event.get("eventTitle") or ""
    return (
        f"eventId={event.get('eventId')}  wk{event.get('week')}  "
        f"{label(away)} @ {label(home)}  {start_date(event)}  {title}"
    ).rstrip()


# ---- caching -------------------------------------------------------------------------------


def raw_dir(data_root: str | Path = "data") -> Path:
    """Where raw STATS payloads are cached (uninterpreted, so we can re-parse later)."""
    return Path(data_root) / "pbp" / "stats_raw"


def save_raw(payload: dict, name: str, data_root: str | Path = "data") -> Path:
    """Write a raw payload to data/pbp/stats_raw/<name>.json. Returns the path."""
    out_dir = raw_dir(data_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def summarize_shape(node, prefix: str = "", depth: int = 0, max_depth: int = 4) -> list[str]:
    """Outline a JSON payload's structure — key paths, types, and list lengths.

    Used to inspect a real response before writing a normalizer against a guessed schema.
    """
    lines: list[str] = []
    if depth > max_depth:
        return lines
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                lines.append(f"{path}  (dict, {len(value)} keys)")
                lines += summarize_shape(value, path, depth + 1, max_depth)
            elif isinstance(value, list):
                lines.append(f"{path}[]  (list, {len(value)} items)")
                if value:
                    lines += summarize_shape(value[0], f"{path}[0]", depth + 1, max_depth)
            else:
                lines.append(f"{path}  ({type(value).__name__})")
    elif isinstance(node, list) and node:
        lines += summarize_shape(node[0], f"{prefix}[0]", depth + 1, max_depth)
    return lines


# ---- CLI -----------------------------------------------------------------------------------


_EVENT_TYPE_ARG = {"preseason": 0, "regular": 1, "postseason": 2}


def _cmd_events(args) -> int:
    from gridiron.data.teams import load_registry

    registry = load_registry(args.season, args.out)
    team_id = registry.resolve_id(args.team) if args.team else None
    payload = fetch_schedule(args.season)
    save_raw(payload, f"schedule_{args.season}", args.out)

    type_id = _EVENT_TYPE_ARG.get(args.type) if args.type else None
    events = find_events(payload, team_id=team_id, week=args.week, event_type_id=type_id)
    label = f" for teamId {team_id}" if team_id else ""
    print(f"{len(events)} event(s){label}")
    for event in events[: args.limit]:
        print("  " + describe_event(event, registry))
    if not events:
        print("\nNothing matched. Note preseason and regular season both number weeks from 1 —")
        print("pass --type regular|postseason to disambiguate.")
    return 0


def _cmd_pbp(args) -> int:
    from gridiron.data.stats_pbp import normalize_event
    from gridiron.data.teams import load_registry

    payload = fetch_event(args.event_id)
    raw_path = save_raw(payload, f"event_{args.event_id}", args.out)
    print(f"Cached raw payload: {raw_path}")

    if args.show:
        print("\nPayload shape:")
        for line in summarize_shape(payload)[:120]:
            print("  " + line)
        return 0

    tidy = normalize_event(payload, load_registry(args.season, args.out))
    if tidy.empty:
        print("No plays in payload — was it fetched with pbp=true?")
        return 1

    game = tidy["game_key"].iloc[0]
    out_path = Path(args.out) / "pbp" / f"{game}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `players` holds nested records; parquet keeps them as a list of structs.
    tidy.to_parquet(out_path, index=False)
    n_scrimmage = int(tidy["is_scrimmage"].sum())
    print(f"Wrote {out_path}")
    print(f"  {len(tidy)} rows, {n_scrimmage} scrimmage plays (film should align to these).")
    print("  no personnel/formation/EPA in this feed — join nflverse for those.")
    return 0


def _cmd_shape(args) -> int:
    payload = json.loads(Path(args.file).read_text())
    for line in summarize_shape(payload)[: args.limit]:
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="STATS Perform API client (schedule + PBP).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared by every subcommand so `--out` can follow the subcommand, where users expect it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="data", help="data root (default: data)")

    p_events = sub.add_parser("events", parents=[common],
                              help="fetch a season schedule and find eventIds")
    p_events.add_argument("--season", type=int, required=True)
    p_events.add_argument("--team", help="team id (326), abbrev (CHI), or nickname (Bears)")
    p_events.add_argument("--week", type=int)
    p_events.add_argument("--type", choices=sorted(_EVENT_TYPE_ARG),
                          help="preseason and regular season both start at week 1")
    p_events.add_argument("--limit", type=int, default=40)
    p_events.set_defaults(func=_cmd_events)

    p_pbp = sub.add_parser("pbp", parents=[common],
                           help="fetch one event's PBP and write the tidy parquet")
    p_pbp.add_argument("--event-id", required=True)
    p_pbp.add_argument("--season", type=int, default=2025, help="for the team registry")
    p_pbp.add_argument("--show", action="store_true",
                       help="print the payload's structure instead of writing parquet")
    p_pbp.set_defaults(func=_cmd_pbp)

    p_shape = sub.add_parser("shape", parents=[common],
                             help="outline the structure of a cached payload")
    p_shape.add_argument("--file", required=True)
    p_shape.add_argument("--limit", type=int, default=200)
    p_shape.set_defaults(func=_cmd_shape)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
