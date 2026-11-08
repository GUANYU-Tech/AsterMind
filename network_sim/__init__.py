"""Expose the src-layout network package from the repository root."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "network_sim")]

from .api import *
from .api import __all__
