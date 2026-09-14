"""Run stages for a game, skipping units whose inputs, config and outputs are unchanged.

    fz run probe proxy --season 2025 --week 20 --away LA --home CHI --dry-run

A unit is one game (per="game") or one play (per="play"). Before running a unit, the runner
fingerprints its inputs and config; if the manifest has the same fingerprints and every output
exists, the unit is skipped. `--force` reruns regardless. `--dry-run` reports what would run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from formation_zero.data.layout import GamePaths
from formation_zero.pipeline.manifest import Manifest, config_hash, file_fingerprint
from formation_zero.pipeline.stages import Stage, StageContext, registry

PIPELINE_VERSION = "0.1.0"


@dataclass
class UnitResult:
    stage: str
    unit: str
    action: str          # "ran" | "skipped" | "would_run" | "missing_input"
    reason: str = ""


class Runner:
    def __init__(self, paths: GamePaths, config: dict | None = None, *, version: str = PIPELINE_VERSION):
        self.paths = paths
        self.config = config or {}
        self.version = version
        self.manifest = Manifest(paths.root / "derived" / "manifests" / f"{paths.game_key}.json")

    def _units(self, st: Stage, plays) -> list[tuple[str, StageContext]]:
        if st.per == "game":
            return [("game", StageContext(self.paths, self.config))]
        return [(uid, StageContext(self.paths, self.config, play_uid=uid, play_index=idx)) for uid, idx in plays]

    def run(self, stage_name: str, *, plays=(), force: bool = False, dry_run: bool = False) -> list[UnitResult]:
        st = registry[stage_name]
        cfg = {k: self.config.get(k) for k in st.config_keys}
        chash = config_hash(cfg)
        results = []
        for unit, ctx in self._units(st, plays):
            inputs = {str(p.relative_to(self.paths.root)): file_fingerprint(p) for p in st.resolve(ctx, st.inputs)}
            outputs = st.resolve(ctx, st.outputs)
            if any(v is None for v in inputs.values()):
                missing = [k for k, v in inputs.items() if v is None]
                results.append(UnitResult(st.name, unit, "missing_input", ", ".join(missing)))
                continue
            prev = self.manifest.get(st.name, unit)
            unchanged = (prev is not None and prev.get("inputs") == inputs and prev.get("config") == chash
                         and prev.get("version") == self.version and all(p.exists() for p in outputs))
            if unchanged and not force:
                results.append(UnitResult(st.name, unit, "skipped", "inputs, config and outputs unchanged"))
                continue
            if dry_run:
                results.append(UnitResult(st.name, unit, "would_run", "forced" if force else "changed or never run"))
                continue
            st.fn(ctx)
            self.manifest.record(st.name, unit, inputs=inputs, config=chash,
                                 outputs=[str(p.relative_to(self.paths.root)) for p in outputs], version=self.version)
            self.manifest.save()
            results.append(UnitResult(st.name, unit, "ran"))
        return results


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    import formation_zero.pipeline.builtin  # noqa: F401  (registers the built-in stages)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stages", nargs="+", help="stage names in order, or 'list'")
    for k in ("--season", "--week"):
        ap.add_argument(k, type=int)
    ap.add_argument("--away"); ap.add_argument("--home"); ap.add_argument("--data-root", default="data")
    ap.add_argument("--config", help="JSON file of stage config")
    ap.add_argument("--force", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.stages == ["list"]:
        for name, st in registry.items():
            print(f"{name:<12} per {st.per:<5} {st.description.splitlines()[0] if st.description else ''}")
        return 0
    if not all([a.season, a.week, a.away, a.home]):
        ap.error("--season, --week, --away and --home are required")
    paths = GamePaths(a.data_root, a.season, a.week, a.away, a.home).ensure_dirs(film=False)
    config = json.loads(Path(a.config).read_text()) if a.config else {}
    runner = Runner(paths, config)
    plays = []
    if paths.play_index_path.exists():
        import pandas as pd
        pi = pd.read_parquet(paths.play_index_path)
        plays = [(r.play_uid, int(r.play_index)) for r in pi.itertuples() if isinstance(r.play_uid, str) and r.play_index == r.play_index]
    for name in a.stages:
        if name not in registry:
            print(f"unknown stage {name!r}; try 'fz run list'")
            return 2
        results = runner.run(name, plays=plays, force=a.force, dry_run=a.dry_run)
        counts = {}
        for r in results:
            counts[r.action] = counts.get(r.action, 0) + 1
        detail = "; ".join(f"{r.unit}: {r.reason}" for r in results if r.action == "missing_input")[:200]
        print(f"{name}: " + ", ".join(f"{v} {k}" for k, v in counts.items()) + (f"  ({detail})" if detail else ""))
    return 0


__all__ = ["Runner", "UnitResult", "PIPELINE_VERSION"]

if __name__ == "__main__":
    raise SystemExit(main())
