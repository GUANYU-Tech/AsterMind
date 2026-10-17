"""Build the NVIDIA H200 grid-block-only GEMM launch selection.

The source is the H200 ``kcampaign`` launch records of the DSR1 router op
(``scripts/kernel_dataset/nvidia_h200/kc_deepseek_r1_decode_runner.py --op router``:
``torch.mm(x_bf16, W_bf16.T, out_dtype=torch.float32)`` above the DSV3 router
kernel's 16-token domain).  Each record states the cuBLASLt main kernel's
grid and block and the separate ``cublasLt::splitKreduce_kernel`` launch; it
states neither the tile nor the split count, so those columns stay null.
Only launch geometry is read: the records' durations are never consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from dataset_pipeline.launch_data.json_validators import (
    sha256_bytes,
    sorted_json_sha256,
)
from src.kernel.launch_config.gemm_launch_selection import (
    GEMM_BOOLEAN_ENCODING,
    GEMM_LAUNCH_ARTIFACT_TYPE,
    GEMM_LAUNCH_COMPRESSION,
    GEMM_LAUNCH_CSV_FILENAME,
    GEMM_LAUNCH_MANIFEST_FILENAME,
    GEMM_NULL_ENCODING,
    compress_gemm_launch_csv,
    gemm_launch_csv_fields,
    gemm_launch_fields,
    gemm_launch_hardware_contract_sha256,
    gemm_launch_schema_version,
    load_gemm_launch_hardware_contract,
    load_gemm_launch_selection_files,
)
from src.kernel.launch_config.gemm_workload import (
    GEMM_LAUNCH_SELECTION_KEY_FIELDS,
    GEMM_SEMANTIC_CONTEXT_FIELDS,
)


HARDWARE_ID = "nvidia_h200"
SOFTWARE_REVISION = "cuda_12.9"
GENERATED_BY = "scripts/launch_data/nvidia_h200_launch_generator.py"
CONSUMPTION_SCOPE = (
    "sglang_moe_router_cublas_bf16_fp32_only: captured under torch "
    "2.11.0+cu129 / vLLM 0.24.0+cu129, not the SGLang v0.5.19 cu130 image; "
    "every prediction consuming a row carries the limitation "
    "launch_observed_on_cuda_12_9_not_serving_cuda_13_0")
_ROOT = Path(__file__).resolve().parents[2]
_CASE_PATTERN = re.compile(r"DSR1_ROUTER_M(\d+)_H(\d+)_E(\d+)")
_MAIN_KERNEL_PATTERN = re.compile(r"nvjet_[A-Za-z0-9_]+_splitK_[A-Z]+")
# cublasLt::splitKreduce_kernel<32, 16, int, float(partial), ...>: the fourth
# template argument is the partial (compute) type.
_REDUCER_PREFIX = (
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, ")
_REDUCER_BLOCK = (32, 16, 1)


def _window_geometry(record: Mapping[str, object], context: str):
    launches = record["launches"]
    if not isinstance(launches, list) or len(launches) != 2:
        raise ValueError(
            f"{context} must record exactly the main GEMM and its reducer")
    main, reducer = launches
    if _MAIN_KERNEL_PATTERN.fullmatch(main["name"]) is None:
        raise ValueError(f"{context} main launch is not a split-K nvjet GEMM")
    if not reducer["name"].startswith(_REDUCER_PREFIX):
        raise ValueError(
            f"{context} second launch is not the FP32-partial split-K reducer")
    main_grid = tuple(main["grid"])
    main_block = tuple(main["block"])
    reducer_grid = tuple(reducer["grid"])
    reducer_block = tuple(reducer["block"])
    if (len(main_grid) != 3 or main_block[1:] != (1, 1)
            or reducer_block != _REDUCER_BLOCK or reducer_grid[2] != 1):
        raise ValueError(f"{context} launch dimensions differ from the contract")
    return {
        "main_name": main["name"],
        "grid": list(main_grid),
        "block_threads": main_block[0],
        "reducer_grid": list(reducer_grid[:2]),
        "reducer_block": list(reducer_block[:2]),
    }


def expand_router_evidence(capture_dir: Path) -> list[dict[str, object]]:
    """Return every split-K router case with window-identical geometry."""

    cases = []
    runtimes = set()
    for case_dir in sorted(Path(capture_dir).iterdir()):
        match = _CASE_PATTERN.fullmatch(case_dir.name)
        if match is None:
            continue
        m, hidden, experts = (int(value) for value in match.groups())
        windows = sorted(case_dir.glob("window_*.launch.json"))
        if not windows:
            raise ValueError(f"{case_dir} has no launch windows")
        records = [json.loads(path.read_text(encoding="utf-8"))
                   for path in windows]
        expected_args = [
            "--op", "router", "--tokens", str(m), "--hidden", str(hidden),
            "--experts", str(experts)]
        for path, record in zip(windows, records):
            if record["runner_args"][:len(expected_args)] != expected_args:
                raise ValueError(f"{path} runner arguments differ from its case")
            telemetry = record["telemetry"]
            if (telemetry["hardware_id"] != HARDWARE_ID
                    or telemetry["clock_accepted"] is not True):
                raise ValueError(f"{path} is not an accepted H200 window")
            runtimes.add(tuple(sorted(telemetry["environment"].items())))
        if records[0]["launches"][0]["name"].startswith(
                "void router_gemm_kernel_float_output"):
            # M <= 16: the source-selected DSV3 router kernel, not cuBLAS.
            continue
        geometries = [
            _window_geometry(record, str(path))
            for path, record in zip(windows, records)]
        if any(geometry != geometries[0] for geometry in geometries[1:]):
            raise ValueError(f"{case_dir} windows disagree on launch geometry")
        cases.append({
            "case_id": case_dir.name,
            "m": m, "n": experts, "k": hidden,
            "windows": len(windows),
            "window_sha256": [
                sha256_bytes(path.read_bytes()) for path in windows],
            **geometries[0],
        })
    if len(runtimes) != 1:
        raise ValueError(f"capture mixes runtimes: {sorted(runtimes)}")
    runtime = dict(next(iter(runtimes)))
    if runtime["torch_cuda_version"] != "12.9":
        raise ValueError("router capture runtime is not CUDA 12.9")
    if not cases:
        raise ValueError(f"{capture_dir} contains no split-K router case")
    return [{"capture_runtime": runtime}, *cases]


def expanded_evidence_sha256(expanded: Sequence[object]) -> str:
    return sorted_json_sha256(list(expanded))


def _row(case: Mapping[str, object]) -> list[str]:
    key = [
        "bf16", "gemm", "1", str(case["m"]), str(case["n"]), str(case["k"]),
        "row_major", "row_major", "row_major", "false", "true", "none",
        "false",
    ]
    launch = ["", "", "", "", "", "fp32_external"]
    grid = case["grid"]
    observed = [
        "grid_block_only", *(str(value) for value in grid),
        str(case["block_threads"]),
        *(str(value) for value in case["reducer_grid"]),
        *(str(value) for value in case["reducer_block"]),
        "fp32", "fp32",
    ]
    return key + launch + observed


def build_launch_selection(
        capture_dir: Path, output_dir: Path,
        capture_baseline_commit: str) -> int:
    expanded = expand_router_evidence(capture_dir)
    runtime, cases = expanded[0]["capture_runtime"], expanded[1:]
    rows = sorted(
        (_row(case) for case in cases),
        key=lambda row: tuple(int(v) if v.isdecimal() else v
                              for v in row[:13]))
    header = list(gemm_launch_csv_fields(HARDWARE_ID))
    csv_bytes = (
        "\n".join(",".join(row) for row in [header, *rows]) + "\n"
    ).encode("utf-8")
    compressed = compress_gemm_launch_csv(csv_bytes)
    keys = [json.dumps(row[:13], separators=(",", ":")) for row in rows]
    contract = load_gemm_launch_hardware_contract(
        _ROOT / "configs" / "hardware" / f"{HARDWARE_ID}.yaml",
        hardware_id=HARDWARE_ID)
    evidence_sha256 = expanded_evidence_sha256(expanded)
    raw_sha256 = sorted_json_sha256(
        [sha for case in cases for sha in case["window_sha256"]])
    manifest = {
        "schema_version": gemm_launch_schema_version(HARDWARE_ID),
        "artifact_type": GEMM_LAUNCH_ARTIFACT_TYPE,
        "hardware_id": HARDWARE_ID,
        "hardware_revision": {
            "config_path": f"configs/hardware/{HARDWARE_ID}.yaml",
            "launch_contract": contract,
            "launch_contract_sha256": (
                gemm_launch_hardware_contract_sha256(contract)),
        },
        "software_revision": SOFTWARE_REVISION,
        "generated_by": GENERATED_BY,
        "evidence_sha256": evidence_sha256,
        "evidence_sources": [
            {"kind": "expanded_evidence", "sha256": evidence_sha256},
            {"kind": "kcampaign_launch_windows", "sha256": raw_sha256},
        ],
        "capture_baseline_commit": capture_baseline_commit,
        "domain_sha256": sha256_bytes("\n".join(keys).encode("utf-8")),
        "csv_sha256": sha256_bytes(csv_bytes),
        "csv_compressed_sha256": sha256_bytes(compressed),
        "compression": GEMM_LAUNCH_COMPRESSION,
        "row_count": len(rows),
        "selection_key_fields": list(GEMM_LAUNCH_SELECTION_KEY_FIELDS),
        "launch_fields": list(gemm_launch_fields(HARDWARE_ID)),
        "semantic_context_fields": list(GEMM_SEMANTIC_CONTEXT_FIELDS),
        "boolean_encoding": GEMM_BOOLEAN_ENCODING,
        "null_encoding": GEMM_NULL_ENCODING,
        "duration_consumed": False,
        "capture_runtime": runtime,
        "consumption_scope": CONSUMPTION_SCOPE,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / GEMM_LAUNCH_CSV_FILENAME
    manifest_path = output_dir / GEMM_LAUNCH_MANIFEST_FILENAME
    csv_path.write_bytes(compressed)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    loaded = load_gemm_launch_selection_files(
        csv_path, manifest_path, hardware_id=HARDWARE_ID,
        software_revision=SOFTWARE_REVISION)
    return len(loaded)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "launch_data" / HARDWARE_ID)
    parser.add_argument("--capture-baseline-commit", required=True)
    args = parser.parse_args(argv)
    count = build_launch_selection(
        args.capture_dir, args.output_dir, args.capture_baseline_commit)
    print(json.dumps({"gemm": count}, sort_keys=True))
    return 0
