"""Which play types are snaps, and which put anything on film. One definition, shared."""

import re

SCRIMMAGE_PLAY_TYPES = frozenset({"pass", "run", "qb_kneel", "qb_spike"})

# Kicks, punts and kneels are filmed; timeouts, the two-minute warning and dead-ball penalties
# are not. These are the plays that count and get a play_uid.
FILMABLE_PLAY_TYPES = ("pass", "run", "punt", "field_goal", "extra_point", "kickoff", "qb_kneel", "qb_spike")

# A play wiped out by a penalty is still on film: the snap happened. Its row is `no_play`, and its
# description starts with the play ("(Shotgun) M.Stafford pass incomplete ... PENALTY on ...").
# A pre-snap foul has nothing before the penalty ("PENALTY on X, False Start ... - No Play.").
# Leading tags such as "(4:22)" and "(No Huddle, Shotgun)" are not the play.
_LEADING_TAGS = re.compile(r"^(\s*\([^)]*\))+\s*")


def snapped(desc) -> bool:
    """True when a no_play description shows a snapped play before its penalty."""
    text = _LEADING_TAGS.sub("", str(desc or "")).lstrip()
    return bool(text) and not text.upper().startswith("PENALTY")


def on_film(play_type, desc) -> bool:
    """Whether a play-by-play row put a play on film: every counted play, plus nullified snaps.

    The film index has to count what the film shows. Nullified snaps do not get a play_uid (the
    official count leaves them out), but leaving them out of the film count shifts every pairing
    after the first one.
    """
    return play_type in FILMABLE_PLAY_TYPES or (play_type == "no_play" and snapped(desc))


def maybe_on_film(play_type, desc) -> bool:
    """A row that is on film in some games and not others: a pre-snap foul, or a kneel-down.

    A pre-snap foul has the teams lined up and the flag before the snap; the Rams-Bears film keeps
    a shotgun false start. A kneel-down to end a half is often cut; the same film drops the one
    that ends regulation. The play list cannot say which survived the edit, so the aligner treats
    these rows as optional and lets the film decide.
    """
    if play_type == "qb_kneel":
        return True
    return play_type == "no_play" and not snapped(desc) and "PENALTY" in str(desc or "").upper()


__all__ = ["FILMABLE_PLAY_TYPES", "SCRIMMAGE_PLAY_TYPES", "maybe_on_film", "on_film", "snapped"]
