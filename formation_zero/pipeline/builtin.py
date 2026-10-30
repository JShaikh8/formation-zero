"""The milestone-1 stages, wired to the runner. Each is a thin call into the module that does the work."""

from __future__ import annotations

from formation_zero.pipeline.stages import StageContext, stage

SOURCE = "raw/{season}/wk{week:02d}/{away}_at_{home}/source/film.mp4"


def _src(ctx: StageContext):
    return ctx.paths.source_dir / ctx.config.get("source_name", "film.mp4")


@stage("probe", inputs=(), outputs=(), per="game", config_keys=("source_name",))
def probe_stage(ctx: StageContext) -> None:
    """Probe the source film and write its sidecar."""
    from formation_zero.ingest.probe import probe, write_sidecar

    write_sidecar(probe(_src(ctx)))


@stage("filmwindow", per="game", config_keys=("source_name", "filmwindow_step_s", "filmwindow_window_s", "filmwindow_min_s"))
def filmwindow_stage(ctx: StageContext) -> None:
    """Find the coaches-film window inside the source recording; write it beside the source."""
    import json

    from formation_zero.ingest.filmwindow import find
    from formation_zero.ingest.sources import FileSource

    src = _src(ctx)
    with FileSource(src) as source:
        w = find(source, step_s=float(ctx.config.get("filmwindow_step_s", 1.0)),
                 window_s=float(ctx.config.get("filmwindow_window_s", 30.0)),
                 min_duration_s=float(ctx.config.get("filmwindow_min_s", 60.0)))
    out = src.with_name(src.name + ".filmwindow.json")
    out.write_text(json.dumps(None if w is None else w.__dict__, indent=1))


def _window(ctx: StageContext) -> tuple[int, int | None]:
    """Frame bounds of the film window if it has been found, else the whole file."""
    import json

    src = _src(ctx)
    wf = src.with_name(src.name + ".filmwindow.json")
    if wf.exists():
        w = json.loads(wf.read_text())
        if w:
            return int(w["start_frame"]), int(w["end_frame"])
    return 0, None


@stage("proxy", per="game", config_keys=("source_name", "proxy_height"))
def proxy_stage(ctx: StageContext) -> None:
    """540p frame-aligned rendition of the film window (or the whole file if no window is known)."""
    import json

    from formation_zero.ingest.proxy import make_proxy

    start, end = _window(ctx)
    dst = ctx.paths.proxy_path(ctx.config.get("source_name", "film.mp4"))
    r = make_proxy(_src(ctx), dst, height=int(ctx.config.get("proxy_height", 540)), start=start, end=end)
    dst.with_name(dst.name + ".json").write_text(json.dumps(r, indent=1))


@stage("records", inputs=("pbp/{game_key}.parquet",), outputs=(), per="game")
def records_stage(ctx: StageContext) -> None:
    """One play record per filmable play from official data alone."""
    import json

    import pandas as pd

    from formation_zero.data.records import _game_from_key, build_records, write_records

    pbp = pd.read_parquet(ctx.paths.pbp_path)
    players_path = ctx.paths.pbp_path.with_suffix(".players.json")
    players = json.loads(players_path.read_text()) if players_path.exists() else {}
    game = _game_from_key(ctx.paths.game_key, ctx.paths.pbp_path)
    write_records(build_records(pbp, game, players), ctx.paths.plays_dir)


@stage("export", inputs=("pbp/{game_key}.parquet",), outputs=("derived/exports/{game_key}.jsonl",), per="game")
def export_stage(ctx: StageContext) -> None:
    """All of a game's play records as JSONL."""
    from formation_zero.data.records import export_jsonl

    export_jsonl(ctx.paths.plays_dir, ctx.paths.export_path, ctx.paths.game_key)


@stage("shots", per="game", config_keys=("source_name", "shots_cut_threshold", "shots_cut_mode"))
def shots_stage(ctx: StageContext) -> None:
    """Cut the film window into camera takes, on the proxy; writes derived/shots/<game>.parquet."""
    import json

    from formation_zero.perception.shots import scan, to_frame, video_fps

    src = _src(ctx)
    proxy = ctx.paths.proxy_path(ctx.config.get("source_name", "film.mp4"))
    meta = json.loads(proxy.with_name(proxy.name + ".json").read_text())
    fps = video_fps(proxy)
    # The proxy holds only the film window, so scan all of it; report times in SOURCE seconds.
    offset_s = meta.get("first_frame", 0) / fps
    shots = scan(proxy, 0.0, meta["frames"] / fps, cut_threshold=float(ctx.config.get("shots_cut_threshold", 15.0)),
                 mode=str(ctx.config.get("shots_cut_mode", "spike")))
    table = to_frame(shots, ctx.paths.game_key, fps)
    for col in ("start_s", "end_s"):
        table[col] = (table[col] + offset_s).round(3)
    for col in ("start_frame", "end_frame"):
        table[col] = table[col] + int(meta.get("first_frame", 0))
    ctx.paths.shots_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(ctx.paths.shots_path, index=False)


@stage("dense", per="game", config_keys=("source_name",))
def dense_stage(ctx: StageContext) -> None:
    """Camera-angle evidence every 0.4 s across the film window, from the proxy; writes derived/shots/<game>.dense.npz."""
    import json

    import numpy as np

    from formation_zero.perception.views import dense_samples

    proxy = ctx.paths.proxy_path(ctx.config.get("source_name", "film.mp4"))
    meta = json.loads(proxy.with_name(proxy.name + ".json").read_text())
    frames, diffs, thumbs, orient = dense_samples(proxy, int(meta.get("first_frame", 0)))
    ctx.paths.shots_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ctx.paths.dense_path, frames=frames, diffs=diffs, thumbs=thumbs, orient=orient,
                        fps=float(meta["fps"]))


@stage("play_index", per="game",
       inputs=("pbp/{game_key}.parquet", "derived/shots/{game_key}.parquet", "derived/shots/{game_key}.dense.npz",
               "groundtruth/{game_key}.shot_views.json"))
def play_index_stage(ctx: StageContext) -> None:
    """Align the film's takes to the plays on film and join them to the play-by-play.

    Writes derived/shots/<game>.plays.parquet (one row per counted play on film) and a report beside
    it (<game>.plays.report.json): which optional rows the film kept, nullified snaps with their film,
    and anything the alignment could not explain. Needs this game's own view labels.
    """
    import json

    import numpy as np
    import pandas as pd

    from formation_zero.perception.play_index import build_from_dense

    labels = json.loads((ctx.paths.groundtruth_dir / f"{ctx.paths.game_key}.shot_views.json").read_text())["labels"]
    with np.load(ctx.paths.dense_path) as dense:
        index, report = build_from_dense(dict(dense), pd.read_parquet(ctx.paths.shots_path),
                                         pd.read_parquet(ctx.paths.pbp_path), labels)
    index.to_parquet(ctx.paths.play_index_path, index=False)
    ctx.paths.play_index_path.with_name(f"{ctx.paths.game_key}.plays.report.json").write_text(
        json.dumps(report, indent=1, default=str))
