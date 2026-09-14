"""Where the NFL API credentials come from. Never from nfl.com's web bundle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_KEYS = ("NFL_API_BASE_URL", "NFL_API_TOKEN_URL", "NFL_API_GRANT_TYPE",
            "NFL_API_CLIENT_ID", "NFL_API_CLIENT_KEY", "NFL_API_CLIENT_SECRET", "NFL_API_DEVICE_ID")


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Read KEY=VALUE lines from a .env file without a dependency. Environment wins over the file."""
    path = Path(path) if path else Path.cwd() / ".env"
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    for k in ENV_KEYS:
        if os.environ.get(k):
            values[k] = os.environ[k]
    return values


@dataclass(frozen=True)
class NflApiConfig:
    base_url: str = "https://api.nfl.com"
    token_url: str = "https://api.nfl.com/identity/v3/token"
    grant_type: str = "client_credentials"
    client_id: str = ""
    client_key: str = ""
    client_secret: str = ""
    device_id: str = ""

    @classmethod
    def from_env(cls, dotenv: str | Path | None = None) -> "NflApiConfig":
        v = load_dotenv(dotenv)
        return cls(base_url=v.get("NFL_API_BASE_URL", cls.base_url),
                   token_url=v.get("NFL_API_TOKEN_URL", cls.token_url),
                   grant_type=v.get("NFL_API_GRANT_TYPE", cls.grant_type),
                   client_id=v.get("NFL_API_CLIENT_ID", ""), client_key=v.get("NFL_API_CLIENT_KEY", ""),
                   client_secret=v.get("NFL_API_CLIENT_SECRET", ""), device_id=v.get("NFL_API_DEVICE_ID", ""))

    @property
    def uses_identity_v3(self) -> bool:
        return "/identity/v3/token" in self.token_url

    def is_configured(self) -> bool:
        if self.uses_identity_v3:
            return bool(self.client_id and self.client_key and self.client_secret and self.device_id)
        return bool(self.token_url and self.client_key and self.client_secret)


__all__ = ["NflApiConfig", "load_dotenv", "ENV_KEYS"]
