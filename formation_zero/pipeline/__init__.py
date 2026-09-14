"""The stage graph: every stage declares inputs, outputs and config; the runner skips what is unchanged."""

from formation_zero.pipeline.stages import Stage, StageContext, registry, stage
from formation_zero.pipeline.runner import Runner

__all__ = ["Stage", "StageContext", "Runner", "registry", "stage"]
