"""Merge the free data's labels into the NFL API play table, joined by play id.

The NFL API carries the plays, the situation and who touched each play; nflverse carries the
formation, personnel and efficiency labels the API does not. Both use the GSIS play id, so the
join is exact. The free description keeps jersey numbers, so it is preferred for `desc`; the
API's own text is kept beside it.
"""

from __future__ import annotations

LABEL_COLUMNS = ("offense_formation", "offense_personnel", "defense_personnel", "defenders_in_box",
                 "epa", "wpa", "success", "first_down", "shotgun", "no_huddle", "qb_dropback", "qb_scramble",
                 "pass_location", "pass_length", "air_yards", "yards_after_catch", "run_location", "run_gap",
                 "sack", "interception", "touchdown", "goal_to_go")


def merge_labels(api, nflverse):
    """api: the adapted NFL API table; nflverse: the free puller's table. Returns a new table."""
    import pandas as pd

    nv = nflverse[nflverse["play_id"].notna()].copy()
    nv["nfl_play_id"] = nv["play_id"].astype(int)
    cols = [c for c in LABEL_COLUMNS if c in nv.columns]
    take = nv[["nfl_play_id", "desc", *cols]].rename(columns={"desc": "desc_numbered"})
    out = api.copy()
    out["nfl_play_id"] = out["nfl_play_id"].astype("Int64")
    take["nfl_play_id"] = take["nfl_play_id"].astype("Int64")
    for c in cols:
        if c in out.columns:
            out = out.drop(columns=[c])
    out = out.merge(take, on="nfl_play_id", how="left")
    out["desc_nfl"] = out["desc"]
    out["desc"] = out["desc_numbered"].where(out["desc_numbered"].notna(), out["desc"])
    out = out.drop(columns=["desc_numbered"])
    out["labels_source"] = "nflverse"
    out.loc[out[cols[0]].isna() if cols else slice(None), "labels_source"] = None
    return out


__all__ = ["merge_labels", "LABEL_COLUMNS"]
