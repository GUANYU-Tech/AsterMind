"""Configuration-driven serving, compute and physical communication integration."""

from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCE = str(WORKSPACE / "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
