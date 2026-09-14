"""The play record contract (blueprint section 9): JSON Schema v1 and a validator."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("play_record.schema.json")
SCHEMA_VERSION = "1.0"


@lru_cache(maxsize=1)
def play_record_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate_play_record(record: dict) -> list[str]:
    """Return a list of problems (empty means valid). Uses jsonschema when installed."""
    try:
        import jsonschema
    except ImportError:  # keep the core importable without the optional validator
        return _minimal_check(record)
    v = jsonschema.Draft202012Validator(play_record_schema())
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in v.iter_errors(record)]


def _minimal_check(record: dict) -> list[str]:
    problems = []
    for k in ("schema_version", "play_uid", "game", "situation", "official", "review", "generated"):
        if k not in record:
            problems.append(f"<root>: missing {k}")
    return problems


__all__ = ["SCHEMA_PATH", "SCHEMA_VERSION", "play_record_schema", "validate_play_record"]
