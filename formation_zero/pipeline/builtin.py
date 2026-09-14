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


@stage("proxy", per="game", config_keys=("source_name", "proxy_height"))
def proxy_stage(ctx: StageContext) -> None:
    """540p frame-aligned rendition of the source film."""
    from formation_zero.ingest.proxy import make_proxy

    make_proxy(_src(ctx), ctx.paths.proxy_path(ctx.config.get("source_name", "film.mp4")),
               height=int(ctx.config.get("proxy_height", 540)))


@stage("filmwindow", per="game", config_keys=("source_name", "filmwindow_step_s"))
def filmwindow_stage(ctx: StageContext) -> None:
    """Find the coaches-film window inside the recording and write it beside the proxy."""
    import json

    from formation_zero.ingest.filmwindow import find
    from formation_zero.ingest.sources import FileSource

    proxy = ctx.paths.proxy_path(ctx.config.get("source_name", "film.mp4"))
    with FileSource(proxy) as src:
        w = find(src, step_s=float(ctx.config.get("filmwindow_step_s", 2.0)))
    out = proxy.with_suffix(".filmwindow.json")
    out.write_text(json.dumps(None if w is None else w.__dict__, indent=1))


@stage("records", inputs=("pbp/{game_key}.parquet",), outputs=(), per="game")
def records_stage(ctx: StageContext) -> None:
    """One play record per filmable play from official data alone."""
    import json

    import pandas as pd

    from formation_zero.data.records import build_records, write_records
    from formation_zero.ids import parse_game_key

    pbp = pd.read_parquet(ctx.paths.pbp_path)
    players_path = ctx.paths.pbp_path.with_suffix(".players.json")
    players = json.loads(players_path.read_text()) if players_path.exists() else {}
    game = {"game_key": ctx.paths.game_key, **parse_game_key(ctx.paths.game_key)}
    write_records(build_records(pbp, game, players), ctx.paths.plays_dir)


@stage("export", inputs=("pbp/{game_key}.parquet",), outputs=("derived/exports/{game_key}.jsonl",), per="game")
def export_stage(ctx: StageContext) -> None:
    """All of a game's play records as JSONL."""
    from formation_zero.data.records import export_jsonl

    export_jsonl(ctx.paths.plays_dir, ctx.paths.export_path, ctx.paths.game_key)
