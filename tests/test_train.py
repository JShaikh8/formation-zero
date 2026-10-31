"""Student training data: YOLO label lines, the split by play, export, and teacher-student disagreement."""

import json

from PIL import Image

from formation_zero.data.layout import GamePaths
from formation_zero.labelling.train import disagreement, export_dataset, val_play, yolo_lines


def test_yolo_lines_normalise_and_clip():
    lines = yolo_lines([{"label": "referee", "box": [90, 10, 110, 50]}, {"label": "cone", "box": [0, 0, 5, 5]}], 100, 100)
    assert lines == ["1 0.950000 0.300000 0.100000 0.400000"]


def test_split_is_by_play_and_stable():
    assert val_play("g_p001") == val_play("g_p001")
    share = sum(val_play(f"g_p{i:03d}") for i in range(1000)) / 1000
    assert 0.1 < share < 0.2


def test_export_keeps_a_plays_frames_together(tmp_path):
    paths = GamePaths(tmp_path / "data", 2025, 1, "AA", "BB", film_root=tmp_path / "film")
    paths.frames_dir.mkdir(parents=True)
    paths.labels_dir.mkdir(parents=True)
    paths.groundtruth_dir.mkdir(parents=True)
    samples, kept = [], []
    for i in range(40):
        frame, uid = 100 + i, f"g_p{i // 2:03d}"           # two frames per play
        Image.new("RGB", (64, 36)).save(paths.frames_dir / f"{frame:06d}.jpg")
        samples.append({"frame": frame, "play_uid": uid})
        kept.append({"frame": frame, "boxes": [{"label": "player", "box": [1, 1, 10, 20]}]})
    paths.samples_path.write_text(json.dumps({"samples": samples}))
    (paths.groundtruth_dir / f"{paths.game_key}.detections.json").write_text(json.dumps({"kept": kept, "flagged": []}))
    counts = export_dataset([paths], tmp_path / "ds")
    assert counts["train"] + counts["val"] == 40 and counts["val"] % 2 == 0
    labels = sorted((tmp_path / "ds" / "labels").rglob("*.txt"))
    assert len(labels) == 40 and labels[0].read_text().startswith("0 ")
    assert (tmp_path / "ds" / "data.yaml").read_text().count(":") >= 5


def test_disagreement_counts_unmatched_boxes_per_class():
    teacher = [{"label": "player", "box": [0, 0, 10, 30]}, {"label": "player", "box": [50, 0, 60, 30]}]
    student = [{"label": "player", "box": [1, 0, 11, 30]}, {"label": "referee", "box": [50, 0, 60, 30]}]
    assert disagreement(teacher, student) == 2        # one player missed, one extra referee
    assert disagreement(teacher, teacher) == 0


def test_round_zero_exports_every_teacher_frame_and_leaves_out_the_ball(tmp_path):
    paths = GamePaths(tmp_path / "data", 2025, 1, "AA", "BB", film_root=tmp_path / "film")
    for d in (paths.frames_dir, paths.labels_dir):
        d.mkdir(parents=True)
    Image.new("RGB", (64, 36)).save(paths.frames_dir / "000100.jpg")
    paths.samples_path.write_text(json.dumps({"samples": [{"frame": 100, "play_uid": "g_p001"}]}))
    boxes = [{"label": "player", "box": [1, 1, 10, 20], "score": 0.9}, {"label": "football", "box": [2, 2, 4, 4], "score": 0.8}]
    (paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").write_text(json.dumps({"frames": {"100": boxes}}))
    assert export_dataset([paths], tmp_path / "ds", source="approved") == {"train": 0, "val": 0}
    counts = export_dataset([paths], tmp_path / "ds", source="teacher")
    assert counts["train"] + counts["val"] == 1
    (label,) = (tmp_path / "ds" / "labels").rglob("*.txt")
    assert label.read_text().splitlines() == ["0 0.085938 0.291667 0.140625 0.527778"]


def test_folds_split_plays_in_half_independently_of_validation():
    from formation_zero.labelling.train import fold_of

    uids = [f"g_p{i:03d}" for i in range(1000)]
    assert 0.45 < sum(fold_of(u) for u in uids) / 1000 < 0.55
    both = sum(fold_of(u) == 1 and val_play(u) for u in uids) / sum(val_play(u) for u in uids)
    assert 0.35 < both < 0.65


def test_next_sheet_skips_every_reviewed_frame_and_ignores_boxes_off_the_field(tmp_path, monkeypatch):
    from formation_zero.labelling import train

    paths = GamePaths(tmp_path / "data", 2025, 1, "AA", "BB", film_root=tmp_path / "film")
    for d in (paths.labels_dir, paths.groundtruth_dir):
        d.mkdir(parents=True)
    on, bench = {"label": "player", "box": [0, 0, 10, 30]}, {"label": "player", "box": [50, 0, 60, 30]}
    frames = {str(f): [on, bench] for f in (1, 2, 3, 4, 5)}
    (paths.labels_dir / f"{paths.game_key}.teacher_sam3.json").write_text(json.dumps({"teacher": "t", "frames": frames}))
    (paths.groundtruth_dir / f"{paths.game_key}.detections.json").write_text(
        json.dumps({"kept": [{"frame": 1, "boxes": []}], "flagged": [2], "dropped": [3]}))
    (paths.labels_dir / f"{paths.game_key}.field_suggestions.json").write_text(json.dumps({"suggestions": {"4": [1]}}))
    asked = {}

    def scored(paths, weights, frames, *, imgsz, device, removed):
        asked.update(frames=frames, removed=removed)
        return {"4": 0, "5": 1}

    monkeypatch.setattr(train, "score_frames", scored)
    sheet = json.loads(train.next_sheet(paths, tmp_path / "w.pt").read_text())
    assert asked == {"frames": ["4", "5"], "removed": {"4": [1]}}
    assert sheet["frames"] == {"5": [on, bench]}        # teacher boxes stay whole: suggestion indices point into them
