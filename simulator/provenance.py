"""Capture local source identities without touching another development tree."""
import hashlib
import json
from pathlib import Path

from .bootstrap import WORKSPACE, DEFAULT_ML


def capture():
    paths = [WORKSPACE/"integration"/name for name in ("midlayer_cell.py", "latest_vllm_backend.py",
        "native_cell_storage.py", "cell_storage.py", "cell_execution.py", "network_batch_executor.py", "cell_communication.py")]
    paths += [DEFAULT_ML/"src/serving/execution.py",
              DEFAULT_ML/"src/serving/kernel_lowering/lower_workload.py"]
    manifest = DEFAULT_ML/".orbitserve_snapshot.json"
    if manifest.is_file():
        paths.append(manifest)
    paths += [p for p in (WORKSPACE/"simulator").rglob("*") if p.is_file()
              and p.suffix in (".py", ".cc", ".csv", ".toml", ".json", ".yaml") and "runs" not in p.parts]
    return dict(default_midlayer_commit="6fe573479c37a931744a6442e7a4b3503eec02e6",
                source_sha256={str(p.relative_to(WORKSPACE)).replace("\\", "/"):
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})


if __name__ == "__main__":
    output = WORKSPACE/"simulator/runs/provenance.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(capture(), indent=2), encoding="utf-8")
    print(output)
