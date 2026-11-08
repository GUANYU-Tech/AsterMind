"""Select the Mid-Layer source root before loading its src.* modules."""

import os
from pathlib import Path
import sys

DEFAULT_ML = Path(__file__).resolve().parent / "ArchPrism" / "midlayer" / "upstream"


def activate(root=None):
    root = Path(root or os.environ.get("ORBITServe_MIDLAYER_ROOT", DEFAULT_ML)).resolve()
    if not (root / "src/serving/execution.py").is_file():
        raise ValueError(f"Missing Mid-Layer snapshot: {root}")
    loaded = sys.modules.get("src.kernel.hardware_spec")
    if loaded is not None and Path(loaded.__file__).resolve().parents[2] != root:
        raise RuntimeError("Mid-Layer is already loaded; start a new process to switch snapshots")
    os.environ["ORBITServe_MIDLAYER_ROOT"] = str(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
