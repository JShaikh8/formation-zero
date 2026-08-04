"""Tests for the STATS Perform client's pure logic — no network, no credentials."""

import hashlib

import pytest

from gridiron.data import statsapi


def test_signature_matches_sha256_of_key_secret_timestamp():
    expected = hashlib.sha256(b"KEYSECRET1700000000").hexdigest()
    assert statsapi.signature("KEY", "SECRET", now=1700000000) == expected


def test_signature_changes_with_timestamp():
    a = statsapi.signature("KEY", "SECRET", now=1700000000)
    b = statsapi.signature("KEY", "SECRET", now=1700000001)
    assert a != b


def test_load_env_parses_comments_quotes_and_blanks(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "STATS_API_KEY=abc123",
                "  STATS_API_SECRET = 'sh h'  ",
                'QUOTED="dq"',
                "MALFORMED_NO_EQUALS",
            ]
        )
    )
    found = statsapi.load_env(env)
    assert found["STATS_API_KEY"] == "abc123"
    assert found["STATS_API_SECRET"] == "sh h"
    assert found["QUOTED"] == "dq"
    assert "MALFORMED_NO_EQUALS" not in found


def test_load_env_missing_file_is_not_an_error(tmp_path):
    assert statsapi.load_env(tmp_path / "nope.env") == {}


def test_credentials_raises_when_key_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("STATS_API_KEY", raising=False)
    monkeypatch.setattr(statsapi, "repo_root", lambda: tmp_path)
    with pytest.raises(SystemExit, match="STATS_API_KEY"):
        statsapi.credentials()


def test_build_url_includes_sig_when_secret_present(monkeypatch):
    monkeypatch.setattr(statsapi, "credentials", lambda: ("K", "S"))
    url = statsapi.build_url("events/123", {"pbp": "true"})
    assert url.startswith("https://api.stats.com/v1/stats/football/nfl/events/123?")
    assert "api_key=K" in url and "sig=" in url and "pbp=true" in url


def test_build_url_omits_sig_when_no_secret(monkeypatch):
    monkeypatch.setattr(statsapi, "credentials", lambda: ("K", ""))
    assert "sig=" not in statsapi.build_url("events/")


def _event(event_id, week, home_id, away_id, **extra):
    return {
        "eventId": event_id,
        "week": week,
        "startDate": [
            {"full": "2026-01-18T18:30:00", "dateType": "Local"},
            {"full": "2026-01-18T23:30:00", "dateType": "UTC"},
        ],
        "teams": [
            {"teamId": home_id, "teamLocationType": {"name": "home"}},
            {"teamId": away_id, "teamLocationType": {"name": "away"}},
        ],
        **extra,
    }


# The envelope STATS actually returns: preseason and regular season BOTH start at week 1.
_SCHEDULE = {
    "apiResults": [
        {
            "league": {
                "season": {
                    "season": 2025,
                    "eventType": [
                        {
                            "eventTypeId": 0,
                            "name": "Preseason",
                            "events": [_event(1000, 1, 326, 343)],
                        },
                        {
                            "eventTypeId": 1,
                            "name": "Regular Season",
                            "events": [_event(2000, 1, 339, 324)],
                        },
                        {
                            "eventTypeId": 2,
                            "name": "Postseason",
                            "events": [
                                _event(2879591, 20, 326, 343,
                                       eventTitle="NFC Divisional Playoff")
                            ],
                        },
                    ],
                }
            }
        }
    ]
}


def test_iter_events_labels_each_event_with_its_type():
    seen = {(tid, name, e["eventId"]) for tid, name, e in statsapi.iter_events(_SCHEDULE)}
    assert (0, "Preseason", 1000) in seen
    assert (2, "Postseason", 2879591) in seen


def test_iter_events_falls_back_to_structural_scan():
    """An unrecognized envelope should still yield events, just unlabelled."""
    events = list(statsapi.iter_events({"weird": {"events": [{"eventId": 42}]}}))
    assert [e["eventId"] for _t, _n, e in events] == [42]


def test_week_alone_is_ambiguous_across_event_types():
    """Preseason wk1 and regular-season wk1 both exist — event_type_id disambiguates."""
    assert len(statsapi.find_events(_SCHEDULE, week=1)) == 2
    assert len(statsapi.find_events(_SCHEDULE, week=1, event_type_id=1)) == 1


def test_find_events_filters_by_team_id():
    hits = statsapi.find_events(_SCHEDULE, team_id=326, event_type_id=2)
    assert [e["eventId"] for e in hits] == [2879591]
    assert statsapi.find_events(_SCHEDULE, team_id=361) == []


def test_event_team_ids():
    event = statsapi.find_events(_SCHEDULE, team_id=326, event_type_id=2)[0]
    assert statsapi.event_team_ids(event) == {326, 343}


def test_home_away_reads_location_type():
    event = statsapi.find_events(_SCHEDULE, team_id=326, event_type_id=2)[0]
    home, away = statsapi.home_away(event)
    assert home["teamId"] == 326
    assert away["teamId"] == 343


def test_start_date_selects_requested_date_type():
    event = statsapi.find_events(_SCHEDULE, team_id=326, event_type_id=2)[0]
    assert statsapi.start_date(event, "UTC") == "2026-01-18T23:30:00"
    assert statsapi.start_date(event, "Local") == "2026-01-18T18:30:00"


def test_start_date_missing_is_empty_string():
    assert statsapi.start_date({}) == ""


def test_describe_event_shows_away_at_home():
    event = statsapi.find_events(_SCHEDULE, team_id=326, event_type_id=2)[0]
    out = statsapi.describe_event(event)
    assert "eventId=2879591" in out and "wk20" in out
    assert "343" in out.split(" @ ")[0] and "326" in out.split(" @ ")[1]


def test_describe_event_survives_missing_fields():
    assert "eventId=1" in statsapi.describe_event({"eventId": 1})


def test_extract_events_dedupes_repeated_ids():
    payload = {"a": _SCHEDULE, "b": _SCHEDULE}
    assert len(statsapi.extract_events(payload)) == 3


def test_summarize_shape_reports_types_and_list_lengths():
    lines = statsapi.summarize_shape({"n": 1, "items": [{"x": "a"}], "d": {"k": True}})
    joined = "\n".join(lines)
    assert "n  (int)" in joined
    assert "items[]  (list, 1 items)" in joined
    assert "items[0].x  (str)" in joined
    assert "d  (dict, 1 keys)" in joined


def test_save_raw_writes_json(tmp_path):
    path = statsapi.save_raw({"ok": True}, "event_1", tmp_path)
    assert path.exists()
    assert path.parent == tmp_path / "pbp" / "stats_raw"


def test_walk_is_depth_bounded():
    node = cur = {}
    for _ in range(50):
        cur["next"] = {}
        cur = cur["next"]
    assert len(list(statsapi._walk(node))) <= statsapi._MAX_WALK_DEPTH + 1
