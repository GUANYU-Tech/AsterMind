"""Manifest loading helpers for OrbitServe experiments."""

from __future__ import annotations

import json
from pathlib import Path

from .models import ExperimentSpec


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ExperimentSpec.from_dict(data, base_dir=str(manifest_path.parent))
