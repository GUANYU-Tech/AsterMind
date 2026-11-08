"""Capture local source identities without touching another development tree."""
import hashlib
import json
from .bootstrap import WORKSPACE, DEFAULT_ML


def capture():
    root = WORKSPACE / "src/astermind"
    paths = [path for path in root.rglob("*.py") if DEFAULT_ML not in path.parents]
    paths += [DEFAULT_ML/"src/serving/execution.py",
              DEFAULT_ML/"src/serving/kernel_lowering/lower_workload.py",
              DEFAULT_ML/"src/serving/backend/vllm.py",
              DEFAULT_ML/"src/serving/backend/vllm_scheduler_process.py"]
    manifest = DEFAULT_ML/".orbitserve_snapshot.json"
    paths.append(manifest)
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))
    paths += [p for p in (WORKSPACE/"simulator").rglob("*") if p.is_file()
              and p.suffix in (".py", ".cc", ".csv", ".toml", ".json", ".yaml") and "runs" not in p.parts]
    runtime = WORKSPACE / "src/orbitserve/runtime"
    paths.extend((runtime / "timeline.py", runtime / "engine/cell_events.py",
                  runtime / "engine/network_events.py"))
    network = WORKSPACE / "src/network_sim"
    paths += [p for p in network.rglob("*") if p.is_file() and p.suffix in (".py", ".cc", ".hh")
              and not {"build", "runs"} & set(p.relative_to(network).parts)]
    return dict(default_midlayer_source=snapshot["source"],
                default_midlayer_commit=snapshot["source_commit"],
                default_midlayer_source_config_sha256=snapshot["source_config_tree_sha256"],
                source_sha256={str(p.relative_to(WORKSPACE)).replace("\\", "/"):
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})


if __name__ == "__main__":
    output = WORKSPACE/"simulator/runs/provenance.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(capture(), indent=2), encoding="utf-8")
    print(output)
