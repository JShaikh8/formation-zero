"""Per-game manifest: what produced each artifact, from which inputs, under which config.

    data/derived/manifests/<game_key>.json

Hashing: small files by content (sha256); large files (video) by size, mtime and the first and
last megabyte, because hashing a 6 GB film to decide whether to skip a stage defeats the point.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

LARGE_FILE_BYTES = 64 * 1024 * 1024


def file_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    st = path.stat()
    h = hashlib.sha256()
    if st.st_size <= LARGE_FILE_BYTES:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()[:24]
    h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    with path.open("rb") as fh:
        h.update(fh.read(1 << 20))
        fh.seek(-(1 << 20), 2)
        h.update(fh.read(1 << 20))
    return "large:" + h.hexdigest()[:24]


def config_hash(config: dict) -> str:
    return "cfg:" + hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {"units": {}}

    def key(self, stage: str, unit: str) -> str:
        return f"{stage}::{unit}"

    def get(self, stage: str, unit: str) -> dict | None:
        return self.data["units"].get(self.key(stage, unit))

    def record(self, stage: str, unit: str, *, inputs: dict, config: str, outputs: list[str], version: str) -> None:
        self.data["units"][self.key(stage, unit)] = {
            "inputs": inputs, "config": config, "outputs": outputs, "version": version,
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n")


__all__ = ["Manifest", "file_fingerprint", "config_hash", "LARGE_FILE_BYTES"]
