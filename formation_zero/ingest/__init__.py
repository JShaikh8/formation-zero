"""Video ingestion: frame sources, probing, proxy renditions, per-play clips, film window.

Everything downstream consumes `Frame`s from a `FrameSource` and does not know whether they
came from a file with a known length or a live stream with no end (blueprint section 11.2).
"""

from formation_zero.ingest.sources import FileSource, Frame, FrameSource, StreamSource, open_source

__all__ = ["Frame", "FrameSource", "FileSource", "StreamSource", "open_source"]
