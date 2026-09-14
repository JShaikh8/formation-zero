"""The NFL's own API (api.nfl.com): the play-by-play and identity backbone (blueprint section 5).

Ported from the owner's other project. Credentials must be issued to us and live in the
environment (or a repo-root .env); nothing here works without them, and everything here says so
instead of failing quietly. Raw payloads are cached to disk before any adaptation.
"""

from formation_zero.data.nflapi.client import NflApi, is_configured
from formation_zero.data.nflapi.game_detail import adapt_game_detail

__all__ = ["NflApi", "is_configured", "adapt_game_detail"]
