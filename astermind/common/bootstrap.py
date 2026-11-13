"""Select one immutable M-L snapshot before any native imports."""

import os
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_ML = WORKSPACE / "astermind/engine/compute/upstream"


def activate(root=None):
    root = Path(
        root
        or os.environ.get("ORBITSERVE_MIDLAYER_ROOT")
        or os.environ.get("ORBITServe_MIDLAYER_ROOT", DEFAULT_ML)
    ).resolve()
    if not (root / "src/serving/execution.py").is_file():
        raise ValueError(f"Missing M-L snapshot: {root}")
    marker=root/'.orbitserve_snapshot.json'
    expected='2e66b9c300efce1d80de158bd519e00e6e95ec797c9262a14b94cf3a25ae251e'
    if not marker.is_file() or json.loads(marker.read_text()).get('source_config_tree_sha256')!=expected:
        raise ValueError('Use the pinned Communication_backend native snapshot; old compute snapshots are retired')
    if "src.kernel.hardware_spec" in sys.modules:
        loaded = (
            Path(sys.modules["src.kernel.hardware_spec"].__file__).resolve().parents[2]
        )
        if loaded != root:
            raise RuntimeError(
                "M-L snapshot is already loaded; start a new process to switch snapshots"
            )
    os.environ["ORBITServe_MIDLAYER_ROOT"] = str(root)
    for path in (root,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
