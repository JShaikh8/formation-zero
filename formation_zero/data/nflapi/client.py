"""Bearer-token client for api.nfl.com, with the same discipline as the reference port.

- The token is cached and refreshed 60 s before its stated expiry (15 min if unstated).
- A 401 on a data request is retried once with a freshly minted token. That separates a stale
  cached token (fixed by re-minting) from dead credentials (the re-mint itself fails and says so).
- A token minted in the last 30 s is never re-minted for a 401: one endpoint that refuses an
  unauthorised scope would otherwise cost every other call a fresh exchange.
- Raw payloads are cached to disk under data/pbp/nfl_raw/ before anything reads them.

The HTTP function is injectable so all of this is testable without the network.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from formation_zero.data.nflapi.config import NflApiConfig

EXPIRY_SKEW_S = 60.0
DEFAULT_TTL_S = 15 * 60.0
FRESH_TOKEN_S = 30.0

# (method, url, headers, body) -> (status, body_text)
HttpFn = Callable[[str, str, dict, bytes | None], tuple[int, str]]


def _urllib_http(method: str, url: str, headers: dict, body: bytes | None) -> tuple[int, str]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


class NotConfigured(RuntimeError):
    pass


class NflApi:
    def __init__(self, config: NflApiConfig | None = None, *, http: HttpFn = _urllib_http,
                 cache_dir: str | Path | None = None, clock: Callable[[], float] = time.time):
        self.config = config or NflApiConfig.from_env()
        self._http = http
        self._clock = clock
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._token: str | None = None
        self._expires_at = 0.0
        self._minted_at = 0.0

    # ---- auth -------------------------------------------------------------------------------
    def is_configured(self) -> bool:
        return self.config.is_configured()

    def _mint(self) -> str:
        c = self.config
        if not c.is_configured():
            raise NotConfigured("api.nfl.com is not configured: set NFL_API_CLIENT_ID, NFL_API_CLIENT_KEY, "
                                "NFL_API_CLIENT_SECRET and NFL_API_DEVICE_ID (org-issued credentials).")
        if c.uses_identity_v3:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            body = json.dumps({"clientId": c.client_id, "clientKey": c.client_key,
                               "clientSecret": c.client_secret, "deviceId": c.device_id}).encode()
        else:
            headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            form = {"grant_type": c.grant_type or "client_credentials", "client_id": c.client_id,
                    "client_key": c.client_key, "client_secret": c.client_secret}
            if c.device_id:
                form["device_id"] = c.device_id
            body = urllib.parse.urlencode(form).encode()
        status, text = self._http("POST", c.token_url, headers, body)
        if status != 200:
            raise RuntimeError(f"NFL token exchange failed: {status}")
        data = json.loads(text)
        token = data.get("access_token") or data.get("accessToken")
        if not token:
            raise RuntimeError("NFL token response missing access_token")
        ttl = data.get("expires_in") or data.get("expiresIn")
        now = self._clock()
        self._token = token
        self._expires_at = now + (float(ttl) if isinstance(ttl, (int, float)) and ttl > 0 else DEFAULT_TTL_S)
        self._minted_at = now
        return token

    def token(self) -> str:
        if self._token and self._clock() < self._expires_at - EXPIRY_SKEW_S:
            return self._token
        return self._mint()

    def reset_token(self) -> None:
        self._token, self._expires_at, self._minted_at = None, 0.0, 0.0

    # ---- requests ---------------------------------------------------------------------------
    def get(self, path: str, *, cache_name: str | None = None) -> dict | list:
        url = path if path.startswith("http") else f"{self.config.base_url}{path}"
        status, text = self._send(url, self.token())
        if status == 401 and self._clock() - self._minted_at >= FRESH_TOKEN_S:
            self.reset_token()
            status, text = self._send(url, self.token())
        if status != 200:
            raise RuntimeError(f"NFL API request failed: {status} {path}")
        payload = json.loads(text)
        if cache_name and self.cache_dir:
            out = self.cache_dir / cache_name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=1))
        return payload

    def _send(self, url: str, token: str) -> tuple[int, str]:
        return self._http("GET", url, {"Authorization": f"Bearer {token}", "Accept": "application/json"}, None)

    # ---- endpoints --------------------------------------------------------------------------
    def weekly_games(self, season: int, season_type: str, week: int) -> list:
        q = urllib.parse.urlencode({"includeDriveChart": "false", "includeReplays": "false",
                                    "includeStandings": "false", "includeTaggedVideos": "false",
                                    "season": season, "type": season_type, "week": week})
        return self.get(f"/football/v2/experience/weekly-game-details?{q}",
                        cache_name=f"weekly/{season}_{season_type}_{week:02d}.json")

    def game_detail(self, slug: str, *, cache_key: str | None = None) -> dict:
        return self.get(f"/experience/v1/gamedetailsbyslug/{urllib.parse.quote(slug)}?includeReplays=false",
                        cache_name=f"{cache_key or slug}/gamedetail.json")

    def teams(self, season: int) -> list | dict:
        return self.get(f"/experience/v1/teams?season={season}", cache_name=f"teams/{season}.json")

    def rosters(self, season: int, team_guid: str) -> dict:
        return self.get(f"/football/v2/rosters?season={season}&teamId={urllib.parse.quote(team_guid)}",
                        cache_name=f"rosters/{season}_{team_guid}.json")


def is_configured() -> bool:
    return NflApiConfig.from_env().is_configured()


__all__ = ["NflApi", "NotConfigured", "is_configured", "EXPIRY_SKEW_S", "DEFAULT_TTL_S", "FRESH_TOKEN_S"]
