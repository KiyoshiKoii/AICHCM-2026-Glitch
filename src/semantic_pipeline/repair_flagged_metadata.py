"""Compatibility CLI wrapper for :mod:`semantic_pipeline.gemini.repair`."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # Support ``python src/semantic_pipeline/...``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantic_pipeline.gemini.repair import *  # noqa: F403
from semantic_pipeline.gemini.repair import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
