"""Filesystem layout for a game's raw footage, PBP, and derived artifacts.

    data/
      raw/<season>/wk<NN>/<AWAY>_at_<HOME>/
        source/            continuous coaches-film file(s), immutable
        sideline/ pNNN.mp4
        endzone/  pNNN.mp4
      pbp/<game_key>.parquet
      derived/
        shots/<game_key>.parquet   camera cuts in the continuous film
        tracking/<play_uid>.parquet
        labels/<play_uid>.json
      plays.parquet
"""

from __future__ import annotations

from pathlib import Path

from gridiron.ids import game_key


class GamePaths:
    """Resolves all paths for one game under a data root. Creating it makes the dirs."""

    def __init__(self, data_root: str | Path, season: int, week: int, away: str, home: str):
        self.root = Path(data_root)
        self.season, self.week = season, week
        self.away, self.home = away.upper(), home.upper()
        self.game_key = game_key(season, week, self.away, self.home)

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw" / f"{self.season}" / f"wk{self.week:02d}" / f"{self.away}_at_{self.home}"

    @property
    def source_dir(self) -> Path:
        return self.raw_dir / "source"

    def angle_dir(self, angle: str) -> Path:
        if angle not in ("sideline", "endzone"):
            raise ValueError(f"angle must be 'sideline' or 'endzone', got {angle!r}")
        return self.raw_dir / angle

    @property
    def pbp_path(self) -> Path:
        return self.root / "pbp" / f"{self.game_key}.parquet"

    @property
    def shots_dir(self) -> Path:
        return self.root / "derived" / "shots"

    @property
    def shots_path(self) -> Path:
        """Cut boundaries for this game's film — see `gridiron.perception.shots`."""
        return self.shots_dir / f"{self.game_key}.parquet"

    @property
    def play_index_path(self) -> Path:
        """Film time <-> play-by-play join — see `gridiron.perception.play_index`."""
        return self.root / "derived" / "shots" / f"{self.game_key}.plays.parquet"

    @property
    def groundtruth_dir(self) -> Path:
        """Hand-made labels. Versioned, unlike everything else under data/."""
        return self.root / "groundtruth"

    @property
    def tracking_dir(self) -> Path:
        return self.root / "derived" / "tracking"

    @property
    def labels_dir(self) -> Path:
        return self.root / "derived" / "labels"

    @property
    def plays_index_path(self) -> Path:
        return self.root / "plays.parquet"

    def ensure_dirs(self) -> "GamePaths":
        for d in (self.source_dir, self.angle_dir("sideline"), self.angle_dir("endzone"),
                  self.pbp_path.parent, self.shots_dir, self.tracking_dir, self.labels_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self
