"""Ingestion on synthetic video: sources, probe, proxy, clips, film window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from formation_zero.ingest import FileSource, Frame, FrameSource, open_source
from formation_zero.ingest import clips, filmwindow, probe, proxy


def test_file_source_reports_shape_and_length(make_video):
    path, meta = make_video(n=40)
    with FileSource(path) as src:
        assert isinstance(src, FrameSource)
        assert src.fps == pytest.approx(25.0)
        assert (src.width, src.height) == (160, 96)
        assert src.frames == 40


def test_file_source_iterates_in_order_with_index_and_time(make_video):
    path, _ = make_video(n=30)
    with FileSource(path) as src:
        frames = list(src.iter_frames())
    assert [f.index for f in frames] == list(range(30))
    assert frames[10].time_s == pytest.approx(10 / 25, abs=1e-3)
    assert all(isinstance(f, Frame) and f.image.shape == (96, 160, 3) for f in frames)


def test_file_source_start_end_step(make_video):
    path, _ = make_video(n=60)
    with FileSource(path) as src:
        idx = [f.index for f in src.iter_frames(start=17, end=41, step=3)]
    assert idx == list(range(17, 41, 3))


def test_open_source_picks_file_for_paths(make_video):
    path, _ = make_video(n=5)
    src = open_source(path)
    assert isinstance(src, FileSource)
    src.close()


def test_probe_writes_sidecar(make_video):
    path, meta = make_video(n=50)
    p = probe.probe(path)
    assert p.frames == 50 and p.fps == pytest.approx(25.0) and (p.width, p.height) == (160, 96)
    assert p.keyframes_s and p.keyframes_s[0] == 0.0
    side = probe.write_sidecar(p)
    assert side.exists() and probe.read_sidecar(path).frames == 50


def test_proxy_is_frame_aligned_and_smaller(make_video, tmp_path):
    path, meta = make_video(n=50)
    out = tmp_path / "proxy.mp4"
    r = proxy.make_proxy(path, out, height=48)
    assert r["frames"] == 50
    with FileSource(out) as src:
        assert src.frames == 50 and src.fps == pytest.approx(25.0) and src.height == 48


def test_clip_cut_keeps_source_frame_numbers(make_video, tmp_path):
    path, _ = make_video(n=100)
    with FileSource(path) as src:
        row = clips.cut(src, tmp_path / "p001.mp4", start_s=1.0, end_s=2.0, pad_s=0.4)
    assert row["source_start_frame"] == 15 and row["frames"] == 45   # 0.6 s to 2.4 s at 25 fps
    with FileSource(row["clip"]) as c:
        assert c.frames == 45


def test_cut_game_writes_both_angles(make_video, tmp_path):
    from formation_zero.data.layout import GamePaths

    path, _ = make_video(n=100)
    paths = GamePaths(tmp_path / "data", 2025, 20, "LA", "CHI").ensure_dirs()
    plays = pd.DataFrame([{"play_uid": "2025_wk20_LA-CHI_p001", "play_index": 1,
                           "sideline_start_s": 0.2, "sideline_end_s": 1.0,
                           "endzone_start_s": 1.4, "endzone_end_s": 2.2}])
    rows = clips.cut_game(path, plays, paths, pad_s=0.2)
    assert {r["angle"] for r in rows} == {"sideline", "endzone"}
    assert all((tmp_path / "data").joinpath(r["clip"]).exists() or __import__("pathlib").Path(r["clip"]).exists() for r in rows)
    assert paths.clip_path("sideline", 1).exists() and paths.clip_path("endzone", 1).exists()


def test_fieldness_separates_turf_from_menu():
    green = np.zeros((20, 20, 3), dtype=np.uint8); green[..., 1] = 120; green[..., 0] = 40; green[..., 2] = 40
    grey = np.full((20, 20, 3), 90, dtype=np.uint8)
    assert filmwindow.fieldness(green) > 0.9
    assert filmwindow.fieldness(grey) == 0.0


def test_longest_run_tolerates_short_gaps():
    flags = [False, True, True, False, True, True, True, False, False, False, True]
    assert filmwindow.longest_run(flags, max_gap=1) == (1, 6)
    assert filmwindow.longest_run(flags, max_gap=0) == (4, 6)
    assert filmwindow.longest_run([False, False], max_gap=1) is None


def test_film_window_found_between_menu_and_frozen_tail(make_video):
    # 10 s menu, 40 s film, 10 s frozen frame, at 25 fps; 5-second windows.
    path, _ = make_video(n=0, segments=[(250, False), (1000, True), (250, False, "frozen")])
    with FileSource(path) as src:
        w = filmwindow.find(src, step_s=1.0, window_s=5.0, max_gap_s=5.0, min_duration_s=20.0)
    assert w is not None
    assert w.start_s == pytest.approx(10.0, abs=5.0)
    assert w.end_s == pytest.approx(50.0, abs=5.0)
    assert w.start_frame % 25 == 0


def test_film_window_none_when_nothing_moves(make_video):
    path, _ = make_video(n=0, segments=[(500, False, "frozen")])
    with FileSource(path) as src:
        assert filmwindow.find(src, step_s=1.0, window_s=5.0, min_duration_s=5.0) is None


def test_film_root_separates_big_files_from_the_data_root(tmp_path, monkeypatch):
    from formation_zero.data.layout import GamePaths
    from formation_zero.ingest import register

    monkeypatch.setenv("FZ_FILM_ROOT", str(tmp_path / "ssd"))
    paths = GamePaths(tmp_path / "data", 2025, 20, "LA", "CHI")
    assert paths.raw_dir == tmp_path / "ssd" / "2025" / "wk20" / "LA_at_CHI"
    assert paths.pbp_path.parent == tmp_path / "data" / "pbp"
    video = tmp_path / "download.mp4"; video.write_bytes(b"x")
    dst = register.add(paths, video)
    assert dst.is_symlink() and dst.resolve() == video.resolve()
    assert paths.source_dir.parent == paths.raw_dir


def test_unmounted_volume_is_refused(tmp_path):
    from formation_zero.data.layout import FilmRootNotMounted, GamePaths
    import pytest as _pytest

    paths = GamePaths(tmp_path / "data", 2025, 20, "LA", "CHI", film_root="/Volumes/NoSuchDrive_fz/formation-zero")
    assert not paths.film_root_available()
    with _pytest.raises(FilmRootNotMounted):
        paths.ensure_dirs(film=True)
    paths.ensure_dirs(film=False)          # small things still work without the drive
    assert paths.pbp_path.parent.exists()
