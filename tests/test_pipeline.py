"""The runner: skip when unchanged, rerun when an input or config changes, force, dry-run."""

from __future__ import annotations

from formation_zero.data.layout import GamePaths
from formation_zero.pipeline import Runner, StageContext, registry, stage


def _paths(tmp_path):
    return GamePaths(tmp_path / "data", 2025, 20, "LA", "CHI").ensure_dirs()


def test_runner_skips_unchanged_and_reruns_on_input_change(tmp_path):
    calls = []

    @stage("t_double", inputs=("in/{game_key}.txt",), outputs=("out/{game_key}.txt",), config_keys=("k",))
    def double(ctx: StageContext):
        src = ctx.fmt("in/{game_key}.txt"); dst = ctx.fmt("out/{game_key}.txt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text() * 2)
        calls.append(ctx.paths.game_key)

    paths = _paths(tmp_path)
    src = paths.root / "in" / f"{paths.game_key}.txt"; src.parent.mkdir(parents=True); src.write_text("a")
    runner = Runner(paths, {"k": 1})
    assert [r.action for r in runner.run("t_double")] == ["ran"]
    assert [r.action for r in runner.run("t_double")] == ["skipped"]
    src.write_text("b")
    assert [r.action for r in runner.run("t_double")] == ["ran"]
    assert len(calls) == 2
    # config change reruns; force reruns; dry-run only reports
    assert [r.action for r in Runner(paths, {"k": 2}).run("t_double")] == ["ran"]
    assert [r.action for r in runner.run("t_double", force=True, dry_run=True)] == ["would_run"]
    assert (paths.root / "derived" / "manifests" / f"{paths.game_key}.json").exists()
    registry.pop("t_double")


def test_missing_input_is_reported_not_raised(tmp_path):
    @stage("t_needs", inputs=("nope/{game_key}.txt",))
    def needs(ctx):  # pragma: no cover - never runs
        raise AssertionError
    paths = _paths(tmp_path)
    r = Runner(paths).run("t_needs")
    assert r[0].action == "missing_input" and "nope" in r[0].reason
    registry.pop("t_needs")


def test_per_play_units(tmp_path):
    seen = []

    @stage("t_play", per="play", outputs=("derived/x/{play_uid}.txt",))
    def per_play(ctx):
        p = ctx.fmt("derived/x/{play_uid}.txt"); p.parent.mkdir(parents=True, exist_ok=True); p.write_text("ok")
        seen.append(ctx.play_index)
    paths = _paths(tmp_path)
    plays = [("2025_wk20_LA-CHI_p001", 1), ("2025_wk20_LA-CHI_p002", 2)]
    r = Runner(paths).run("t_play", plays=plays)
    assert [x.action for x in r] == ["ran", "ran"] and seen == [1, 2]
    assert [x.action for x in Runner(paths).run("t_play", plays=plays)] == ["skipped", "skipped"]
    registry.pop("t_play")


def test_builtin_records_and_export_run_through_the_runner(tmp_path):
    import shutil

    import formation_zero.pipeline.builtin  # noqa: F401
    paths = _paths(tmp_path)
    shutil.copy("data/pbp/2025_wk20_LA-CHI.parquet", paths.pbp_path)
    runner = Runner(paths)
    assert [r.action for r in runner.run("records")] == ["ran"]
    assert len(list(paths.plays_dir.glob("*.json"))) == 184
    assert [r.action for r in runner.run("export")] == ["ran"]
    assert paths.export_path.exists()
    assert [r.action for r in runner.run("export")] == ["skipped"]
