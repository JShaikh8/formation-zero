"""The teacher-review sheet: ordering, the page's inline data, and importing decisions."""

import json

from formation_zero.labelling.review import build_page, frame_order, import_decisions

DET = {"teacher": "test", "frames": {
    "100": [{"label": "player", "box": [1, 2, 30, 60], "score": 0.9}],
    "200": [{"label": "player", "box": [5, 5, 20, 50], "score": 0.4}, {"label": "football", "box": [9, 9, 12, 12], "score": 0.95}],
    "300": [],
}}


def test_weakest_frames_come_first_and_empty_frames_before_them():
    assert frame_order(DET) == [300, 200, 100]


def test_page_carries_frames_by_relative_path():
    page = build_page("g", DET, frame_order(DET))
    data = json.loads(page.split("const D = ", 1)[1].split(";\n", 1)[0])
    assert [f["img"] for f in data["frames"]] == ["000300.jpg", "000200.jpg", "000100.jpg"]
    assert data["frames"][1]["boxes"][1]["label"] == "football"


def test_import_keeps_kept_frames_lists_flagged_and_drops_the_rest():
    out = import_decisions("g", DET, {"decisions": {"100": "keep", "200": "flag", "300": "drop"}})
    assert [k["frame"] for k in out["kept"]] == [100]
    assert out["kept"][0]["time_s"] == 4.0 and out["kept"][0]["boxes"] == DET["frames"]["100"]
    assert out["flagged"] == [200] and out["reviewed"] == 3


def test_import_leaves_out_boxes_the_reviewer_removed():
    out = import_decisions("g", DET, {"decisions": {"200": "keep", "100": "drop"}, "removed": {"200": [0], "100": [0]}})
    assert out["kept"][0]["boxes"] == [DET["frames"]["200"][1]]
    assert out["boxes_removed"] == 1                     # removals on a dropped frame do not count


def test_disputed_frames_come_first_and_confidence_breaks_ties():
    assert frame_order(DET, {"100": 3, "300": 3}) == [300, 100, 200]
