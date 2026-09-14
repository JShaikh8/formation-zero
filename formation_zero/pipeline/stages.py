"""Stage definitions. A stage is a function plus declared inputs, outputs and config keys.

Inputs and outputs are path templates resolved against a `StageContext`: `{game_key}`,
`{play_uid}`, `{play_index:03d}`, `{angle}`, plus anything in the context's `vars`. A stage
runs per game (`per="game"`) or per play (`per="play"`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from formation_zero.data.layout import GamePaths


@dataclass
class StageContext:
    paths: GamePaths
    config: dict
    play_uid: str | None = None
    play_index: int | None = None
    vars: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.paths.root

    def fmt(self, template: str) -> Path:
        values = {"game_key": self.paths.game_key, "play_uid": self.play_uid or "",
                  "play_index": self.play_index if self.play_index is not None else 0, **self.vars}
        return self.root / template.format(**values)


@dataclass
class Stage:
    name: str
    fn: Callable[[StageContext], None]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    per: str = "game"              # "game" | "play"
    description: str = ""

    def resolve(self, ctx: StageContext, templates) -> list[Path]:
        return [ctx.fmt(t) for t in templates]


registry: dict[str, Stage] = {}


def stage(name: str, *, inputs=(), outputs=(), config_keys=(), per="game", description=""):
    """Decorator: register a stage function."""
    def wrap(fn):
        registry[name] = Stage(name=name, fn=fn, inputs=tuple(inputs), outputs=tuple(outputs),
                               config_keys=tuple(config_keys), per=per, description=description or (fn.__doc__ or "").strip())
        return fn
    return wrap


__all__ = ["Stage", "StageContext", "registry", "stage"]
