"""Build the queryable play store: PBP parquet + intelligence enrichment -> DuckDB `plays` table.

This is the data behind the chat layer's semantic layer. It loads every game's PBP, enriches
each scrimmage play with the parsed personnel/formation labels, and exposes a DuckDB table the
structured query tool runs against.
"""

from __future__ import annotations

from pathlib import Path

from formation_zero.intelligence.personnel import summarize_play

# Columns added by the intelligence layer (parallel to summarize_play()).
ENRICHED_COLS = [
    "off_grouping", "off_formation", "wr", "te", "rb",
    "def_front", "def_package", "def_dl", "def_lb", "def_db",
]


def enrich_plays(df):
    """Return scrimmage plays with personnel/formation labels added."""
    import pandas as pd  # noqa: F401  (ensures pandas present)

    scrim = df[df["is_scrimmage"]].copy()
    enriched = scrim.apply(
        lambda r: summarize_play(
            r.get("offense_personnel"), r.get("defense_personnel"), r.get("offense_formation")
        ),
        axis=1,
        result_type="expand",
    )
    out = scrim.reset_index(drop=True)
    for col in ENRICHED_COLS:
        out[col] = enriched[col].values
    return out


def load_plays(data_root: str | Path = "data", con=None):
    """Load all PBP parquet under <data_root>/pbp into a DuckDB `plays` table. Returns the connection."""
    import duckdb
    import pandas as pd

    pbp_dir = Path(data_root) / "pbp"
    paths = sorted(pbp_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"No PBP parquet in {pbp_dir}. Run e.g.:\n"
            f"  fz-pull-pbp --season 2023 --week 1 --away DET --home KC --out {data_root}"
        )
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    plays = enrich_plays(df)

    con = con or duckdb.connect()
    con.register("_plays_df", plays)
    con.execute("CREATE OR REPLACE TABLE plays AS SELECT * FROM _plays_df")
    con.unregister("_plays_df")
    return con
