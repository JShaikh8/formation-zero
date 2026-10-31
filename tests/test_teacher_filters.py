"""The teacher's geometry filters (numpy only: SAM 3 itself is not needed to test them)."""

from formation_zero.labelling.teacher_sam3 import apply_filters, drop_duplicates, drop_giant_balls


def box(label, x1, y1, x2, y2, score=0.9):
    return {"label": label, "box": [x1, y1, x2, y2], "score": score}


def test_a_near_copy_of_a_better_box_goes_but_a_neighbour_stays():
    boxes = [box("player", 0, 0, 10, 30, 0.9), box("player", 1, 0, 11, 30, 0.6), box("player", 12, 0, 22, 30, 0.8)]
    kept, dropped = drop_duplicates(boxes)
    assert dropped == 1 and sorted(b["score"] for b in kept) == [0.8, 0.9]


def test_duplicates_are_judged_within_a_label():
    boxes = [box("player", 0, 0, 10, 30), box("referee", 0, 0, 10, 30, 0.7)]
    assert drop_duplicates(boxes)[1] == 0


def test_a_ball_as_big_as_a_player_is_a_logo():
    boxes = [box("player", 0, 0, 10, 60), box("player", 20, 0, 30, 60), box("football", 40, 0, 70, 40, 0.6),
             box("football", 80, 10, 86, 14, 0.4)]
    kept, dropped = drop_giant_balls(boxes)
    assert dropped == 1 and [b["box"] for b in kept if b["label"] == "football"] == [[80, 10, 86, 14]]


def test_a_long_thin_ball_is_the_shield_even_in_a_close_shot():
    boxes = [box("player", 0, 0, 30, 240), box("football", 40, 0, 260, 78, 0.5), box("football", 0, 100, 30, 120, 0.5)]
    kept, dropped = drop_giant_balls(boxes)
    assert dropped == 1 and [b["box"] for b in kept if b["label"] == "football"] == [[0, 100, 30, 120]]


def test_without_players_only_the_shape_test_applies():
    kept, dropped = drop_giant_balls([box("football", 0, 0, 30, 10), box("football", 0, 0, 12, 10)])
    assert dropped == 1 and kept[0]["box"] == [0, 0, 12, 10]


def test_filters_report_what_they_dropped():
    _, dropped = apply_filters([box("player", 0, 0, 10, 60), box("player", 0, 0, 10, 61, 0.5)])
    assert dropped == {"duplicates": 1, "giant_balls": 0}


def test_detections_file_cuts_weak_balls_and_names_the_cut():
    from formation_zero.labelling.teacher_sam3 import detections_file

    rows = [{"frame": 7, "boxes": [box("player", 0, 0, 10, 60, 0.55), box("football", 1, 1, 3, 3, 0.35),
                                   box("football", 5, 5, 7, 7, 0.45)]}]
    out = detections_file(rows)
    assert [b["score"] for b in out["frames"]["7"]] == [0.55, 0.45]
    assert "football>0.4" in out["teacher"]


def test_the_frame_size_names_the_teacher_when_the_log_has_it():
    from formation_zero.labelling.teacher_sam3 import detections_file

    old = detections_file([{"frame": 1, "boxes": []}])["teacher"]
    new = detections_file([{"frame": 1, "size": [1920, 1080], "boxes": []}])["teacher"]
    assert "x" not in old.split(",")[0] and new.split(",")[0].endswith("batched 1920x1080")


def test_detections_file_reapplies_the_ball_rules_to_old_logs():
    from formation_zero.labelling.teacher_sam3 import detections_file

    rows = [{"frame": 7, "boxes": [box("player", 0, 0, 30, 240), box("football", 40, 0, 260, 78, 0.5)]}]
    assert [b["label"] for b in detections_file(rows)["frames"]["7"]] == ["player"]
