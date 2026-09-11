"""Geometry checks for the NFL field model — no external deps, run with pytest."""

from formation_zero.configs.nfl_field import (
    FIELD_LENGTH_YD,
    FIELD_WIDTH_YD,
    NFLFieldConfiguration,
    absolute_x_to_yard_number,
)


def test_dimensions():
    assert FIELD_LENGTH_YD == 120.0
    assert abs(FIELD_WIDTH_YD - 53.333) < 0.01  # 160 ft


def test_hash_marks_symmetric():
    cfg = NFLFieldConfiguration()
    _, y_bot = cfg.point(60, "hash_b")
    _, y_top = cfg.point(60, "hash_t")
    # 70'9" from each sideline -> 23.583 and 29.75; symmetric about midfield width.
    assert abs(y_bot - 23.583) < 0.01
    assert abs(y_top - 29.75) < 0.01
    assert abs((y_bot + y_top) - FIELD_WIDTH_YD) < 1e-6


def test_fifty_yard_line_at_x60():
    # The 50-yard line sits at absolute x = 60 (10 yd end zone + 50 yd).
    cfg = NFLFieldConfiguration()
    x, _ = cfg.point(60, "sideline_b")
    assert x == 60.0
    assert absolute_x_to_yard_number(60.0) == 50


def test_yard_numbers_count_down_from_fifty():
    assert absolute_x_to_yard_number(10.0) == 0    # left goal line
    assert absolute_x_to_yard_number(110.0) == 0   # right goal line
    assert absolute_x_to_yard_number(35.0) == 25
    assert absolute_x_to_yard_number(85.0) == 25


def test_los_anchor_kills_ambiguity():
    cfg = NFLFieldConfiguration()
    # Ball on opponent's 35 (yardline_100 = 35) -> absolute x = 10 + (100 - 35) = 75.
    assert cfg.line_of_scrimmage_x(35) == 75.0
    # Own 20 (yardline_100 = 80) -> x = 30.
    assert cfg.line_of_scrimmage_x(80) == 30.0


def test_vertices_grid_is_regular():
    cfg = NFLFieldConfiguration()
    # 4 lateral lines x (2 end lines + 21 lines from goal-to-goal every 5 yd) = 4 * 23 = 92.
    assert len(cfg.vertices) == len(cfg.labels)
    assert len(cfg.vertices) == 4 * 23
