"""Publish and reuse complete, backend-independent physical datasets."""

import hashlib
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

from .network_inputs import snapshot_network_input

SCHEMA = "orbitserve.physical_dataset.v1"
PHYSICS_VERSION = "sample_hold_directional_causal_v1"


def input_fingerprint(concrete):
    value = deepcopy(concrete)
    value.get("network", {}).pop("transport_model", None)
    value.pop("route_probes", None)
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_dataset(root, concrete, *, source_input=None):
    root = Path(root)
    files = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and (
            p.parts[len(root.parts)] in {"orbit_trace", "frames"}
            or p.name
            in {
                "network_result.json",
                "materialized_input.json",
                "physical_state.sqlite",
                "node_positions.jsonl.gz",
                "link_attributes.jsonl.gz",
                "link_events.jsonl.gz",
                "astermind_network.jsonl.gz",
            }
        )
    ]
    value = {
        "schema_version": SCHEMA,
        "physics_version": PHYSICS_VERSION,
        "input_sha256": input_fingerprint(concrete),
        "source_sha256": input_fingerprint(
            source_input if source_input is not None else concrete
        ),
        "files": {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)},
    }
    temporary = root / "physical_dataset.json.tmp"
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(root / "physical_dataset.json")
    return value


def load_dataset(reference, concrete=None):
    root = Path(reference)
    if root.is_file():
        root = root.parent
    index = root / "physical_dataset.json"
    if not index.is_file():
        raise ValueError("完整物理数据尚未就绪，请在第一步完成物理预计算")
    value = json.loads(index.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != SCHEMA
        or value.get("physics_version") != PHYSICS_VERSION
    ):
        raise ValueError("物理数据版本已过期，请在第一步重新生成")
    if concrete is not None and value["input_sha256"] != input_fingerprint(concrete):
        raise ValueError("物理配置与数据包不匹配，请返回第一步重新生成")
    files = value["files"]
    if (
        not {
            "physical_state.sqlite",
            "network_result.json",
            "materialized_input.json",
            "orbit_trace/trace_manifest.json",
            "node_positions.jsonl.gz",
            "link_attributes.jsonl.gz",
            "link_events.jsonl.gz",
            "astermind_network.jsonl.gz",
        }
        <= files.keys()
    ):
        raise ValueError("incomplete physical dataset")
    for name, expected in files.items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or not path.is_file()
            or digest(path) != expected
        ):
            raise ValueError(f"physical dataset missing or corrupt: {name}")
    stored = json.loads((root / "materialized_input.json").read_text(encoding="utf-8"))
    if input_fingerprint(stored) != value["input_sha256"]:
        raise ValueError("physical dataset input identity mismatch")
    result = json.loads((root / "network_result.json").read_text(encoding="utf-8"))
    from .network import TERMINAL_ALLOCATION_MODEL

    if (
        result.get("provenance", {}).get("terminal_allocation_model")
        != TERMINAL_ALLOCATION_MODEL
        or result.get("provenance", {}).get("between_samples") != "sample_hold"
    ):
        raise ValueError("物理模型语义已过期，请在第一步重新生成")
    expected_frames = result["artifacts"]["frames"]
    if any(frame["artifact"] not in files for frame in expected_frames):
        raise ValueError("incomplete physical dataset frames")
    return root.resolve(), value, result


def reuse_dataset(reference, destination, concrete, *, base_dir="."):
    source, manifest, result = load_dataset(Path(base_dir) / reference, concrete)
    target = Path(destination)
    if source != target.resolve():
        target.mkdir(parents=True, exist_ok=True)
        # Share immutable bytes on the same filesystem; portable copy fallback.
        for name in [*manifest["files"], "physical_dataset.json"]:
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                if digest(output) != digest(source / name):
                    raise ValueError(
                        "output already contains a different physical dataset"
                    )
                continue
            try:
                os.link(source / name, output)
            except OSError:
                shutil.copy2(source / name, output)
    result = deepcopy(result)
    result["provenance"]["physical_dataset"] = {
        "input_sha256": manifest["input_sha256"],
        "source": str(source),
        "reused": True,
    }
    return result


def validate_dataset_input(reference, physical, base_dir="."):
    root, manifest, result = load_dataset(Path(base_dir) / reference)
    requested = input_fingerprint(snapshot_network_input(physical, base_dir))
    if requested not in {manifest.get("source_sha256"), manifest["input_sha256"]}:
        raise ValueError("物理配置与数据包不匹配，请返回第一步重新生成")
    return root, manifest, result


def dataset_physical_input(reference, physical, base_dir="."):
    root, _, _ = validate_dataset_input(reference, physical, base_dir)
    concrete = json.loads(
        (root / "materialized_input.json").read_text(encoding="utf-8")
    )
    # Transport is a Runtime choice and is deliberately absent from identity.
    if "transport_model" in physical["network"]:
        concrete["network"]["transport_model"] = physical["network"]["transport_model"]
    return concrete
