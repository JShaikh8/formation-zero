"""Register a film file for a game without copying it: fz-film.

    fz-film add --season 2025 --week 20 --away LA --home CHI /Volumes/MySSD/downloads/rams-bears.mp4
    fz-film add ... --copy      # copy instead of linking
    fz-film show --season 2025 --week 20 --away LA --home CHI

By default the file is symlinked into <film root>/<season>/wk<NN>/<AWAY>_at_<HOME>/source/film.mp4,
so a download that already lives on the external drive is not duplicated. The film root itself
comes from FZ_FILM_ROOT (environment or the repo .env), which is where the drive is named once.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from formation_zero.data.layout import FilmRootNotMounted, GamePaths


def add(paths: GamePaths, video: Path, *, mode: str = "link", name: str = "film.mp4") -> Path:
    video = video.expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    paths.ensure_dirs(film=True)
    dst = paths.source_dir / name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(video, dst)
    elif mode == "move":
        shutil.move(str(video), dst)
    else:
        dst.symlink_to(video)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("add", "show"):
        sp = sub.add_parser(name)
        for k in ("--season", "--week"):
            sp.add_argument(k, type=int, required=True)
        sp.add_argument("--away", required=True); sp.add_argument("--home", required=True)
        sp.add_argument("--data-root", default="data"); sp.add_argument("--film-root")
        if name == "add":
            sp.add_argument("video")
            sp.add_argument("--copy", action="store_true"); sp.add_argument("--move", action="store_true")
            sp.add_argument("--name", default="film.mp4")
    a = ap.parse_args(argv)
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home, film_root=a.film_root)
    try:
        if a.cmd == "add":
            dst = add(paths, Path(a.video), mode="copy" if a.copy else "move" if a.move else "link", name=a.name)
            kind = "copied to" if a.copy else "moved to" if a.move else "linked at"
            print(f"{kind} {dst}")
            return 0
        print(f"film root: {paths.film_root}  ({'mounted' if paths.film_root_available() else 'NOT MOUNTED'})")
        print(f"source:    {paths.source_dir}")
        for f in sorted(paths.source_dir.glob("*")) if paths.source_dir.exists() else []:
            target = f" -> {f.resolve()}" if f.is_symlink() else ""
            size = f.stat().st_size / 1e9 if f.exists() else 0
            print(f"  {f.name}  {size:.2f} GB{target}")
        return 0
    except FilmRootNotMounted as e:
        print(str(e), file=sys.stderr)
        return 2


__all__ = ["add"]

if __name__ == "__main__":
    raise SystemExit(main())
