"""Which play types are snaps, and which put anything on film. One definition, shared."""

SCRIMMAGE_PLAY_TYPES = frozenset({"pass", "run", "qb_kneel", "qb_spike"})

# Kicks, punts and kneels are filmed; timeouts, the two-minute warning and dead-ball penalties
# are not. This set is what the play index counts, so it must match what the film contains.
FILMABLE_PLAY_TYPES = ("pass", "run", "punt", "field_goal", "extra_point", "kickoff", "qb_kneel", "qb_spike")

__all__ = ["SCRIMMAGE_PLAY_TYPES", "FILMABLE_PLAY_TYPES"]
