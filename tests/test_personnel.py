"""Unit tests for the personnel/formation parser."""

from formation_zero.intelligence.personnel import (
    defensive_front,
    offensive_grouping,
    parse_personnel,
    summarize_play,
)


def test_parse_ol_inclusive_string():
    d = parse_personnel("1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR")
    assert d == {"C": 1, "G": 2, "QB": 1, "RB": 1, "T": 2, "TE": 1, "WR": 3}


def test_parse_skill_only_string():
    assert parse_personnel("1 RB, 2 TE, 2 WR") == {"RB": 1, "TE": 2, "WR": 2}


def test_offensive_groupings():
    assert offensive_grouping(parse_personnel("1 RB, 1 TE, 3 WR")) == "11"
    assert offensive_grouping(parse_personnel("1 RB, 2 TE, 2 WR")) == "12"
    assert offensive_grouping(parse_personnel("2 RB, 1 TE, 2 WR")) == "21"
    # A fullback counts as a back (21 personnel).
    assert offensive_grouping(parse_personnel("1 RB, 1 FB, 1 TE, 2 WR")) == "21"
    # Empty backfield, 1 TE -> 01.
    assert offensive_grouping(parse_personnel("1 TE, 4 WR")) == "01"


def test_defensive_front_nickel():
    front = defensive_front(parse_personnel("2 CB, 2 DE, 2 DT, 1 FS, 1 ILB, 1 OLB, 2 SS"))
    assert front == {"dl": 4, "lb": 2, "db": 5, "front": "4-2-5", "package": "nickel"}


def test_defensive_packages_by_db_count():
    assert defensive_front(parse_personnel("4 DL, 3 LB, 4 DB"))["package"] == "base"
    assert defensive_front(parse_personnel("4 DL, 2 LB, 5 DB"))["package"] == "nickel"
    assert defensive_front(parse_personnel("4 DL, 1 LB, 6 DB"))["package"] == "dime"


def test_summarize_play():
    s = summarize_play(
        "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR",
        "2 CB, 2 DE, 2 DT, 1 FS, 1 ILB, 1 OLB, 2 SS",
        "SHOTGUN",
    )
    assert s["off_grouping"] == "11"
    assert s["off_formation"] == "SHOTGUN"
    assert s["def_package"] == "nickel"
    assert s["def_front"] == "4-2-5"
