"""Closed vocabularies for the play record (blueprint section 9.3). Clients filter on these."""

from __future__ import annotations

PHASES = ("pre_snap", "live", "post_snap")
MOTION_TYPES = ("jet", "orbit", "across", "short", "return", "shift_only")
MOTION_AT_SNAP = ("in_motion", "set")
QB_ALIGNMENTS = ("under_center", "shotgun", "pistol")
SHELLS = ("0-high", "1-high", "2-high")
FRONTS = ("over", "under", "bear", "odd", "even", "wide_9")
PLAY_FAMILIES = ("dropback", "play_action", "rpo", "screen", "run", "scramble", "trick", "kick")
DROPS = ("1-step", "3-step", "5-step", "7-step", "rollout_left", "rollout_right", "none")
ROUTES = ("flat", "slant", "comeback", "curl", "out", "dig", "corner", "post", "go", "seam", "wheel",
          "screen", "swing", "angle", "whip", "stick", "hitch", "cross", "block")
RUN_GAPS = tuple(f"{g}-{s}" for g in "ABCD" for s in ("left", "right"))
RUN_SCHEMES = ("inside_zone", "outside_zone", "power", "counter", "duo", "draw", "trap", "sweep", "qb_design")
COVERAGES = ("cover_0", "cover_1", "cover_2", "cover_2_man", "cover_3", "cover_4", "cover_6", "unknown")
EVENTS = ("snap", "handoff", "play_fake", "pass_forward", "pass_outcome_caught", "pass_outcome_incomplete",
          "pass_outcome_interception", "first_contact", "tackle", "out_of_bounds", "touchdown", "fumble",
          "qb_sack", "qb_scramble", "lateral", "whistle")
ROLES = ("QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT", "EDGE", "DT", "NT", "LB", "CB", "NB", "S",
         "K", "P", "LS", "OFFICIAL")
TECHNIQUES = ("0", "1", "2i", "2", "3", "4i", "4", "5", "7", "9", "wide_9")
STANCES = ("2pt", "3pt", "4pt")
SOURCES = ("official", "nflverse", "cv", "hosted_model", "human")
PASSES = ("official", "live", "post")
REVIEW_STATES = ("unreviewed", "reviewed", "corrected", "flagged")
PLAY_TYPES = ("pass", "run", "punt", "field_goal", "extra_point", "kickoff", "qb_kneel", "qb_spike", "no_play")

ALL = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, tuple)}

__all__ = list(ALL) + ["ALL"]
