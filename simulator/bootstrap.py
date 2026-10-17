"""Select one immutable M-L snapshot before any native imports."""
from pathlib import Path
import os
import sys

WORKSPACE = Path(__file__).resolve().parents[1]
INTEGRATION = WORKSPACE / "integration"
DEFAULT_ML = INTEGRATION / "upstream/midlayer-6fe5734-canonical"


def activate(root=None):
    root = Path(root or os.environ.get("ORBITServe_MIDLAYER_ROOT", DEFAULT_ML)).resolve()
    if not (root / "src/serving/execution.py").is_file():
        raise ValueError(f"Missing M-L snapshot: {root}")
    if "src.kernel.hardware_spec" in sys.modules:
        loaded = Path(sys.modules["src.kernel.hardware_spec"].__file__).resolve().parents[2]
        if loaded != root:
            raise RuntimeError("M-L snapshot is already loaded; start a new process to switch snapshots")
    os.environ["ORBITServe_MIDLAYER_ROOT"] = str(root)
    for path in (WORKSPACE / "src", root, INTEGRATION):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
