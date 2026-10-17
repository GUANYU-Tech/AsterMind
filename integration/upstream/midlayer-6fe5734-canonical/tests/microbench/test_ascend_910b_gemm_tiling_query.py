"""Pure contracts for rank-two Ascend GEMM launch-only capture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "microbench/ascend_910b/operators/gemm/tiling_query_runner.py"


def _runner_module():
    spec = importlib.util.spec_from_file_location("rank2_tiling_query", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operation(*, input_rank: int, batch: int = 1) -> dict[str, int]:
    return {
        "input_rank": input_rank,
        "b": batch,
        "m": 2220,
        "n": 4096,
        "k": 12288,
    }


def test_rank_two_layout_uses_real_rank_two_lhs_and_output() -> None:
    runner = _runner_module()

    layout = runner._operation_layout(_operation(input_rank=2))

    assert layout == {
        "input_rank": 2,
        "lhs": {
            "kind": "row_major",
            "view_shape": [2220, 12288],
            "strides": [12288, 1],
            "storage_shape": [2220, 12288],
        },
        "rhs": {
            "kind": "transposed_view",
            "view_shape": [12288, 4096],
            "strides": [1, 12288],
            "storage_shape": [4096, 12288],
        },
        "output": {
            "kind": "row_major",
            "view_shape": [2220, 4096],
            "strides": [4096, 1],
            "storage_shape": [2220, 4096],
        },
    }


def test_rank_three_layout_remains_an_explicit_comparison_mode() -> None:
    runner = _runner_module()

    layout = runner._operation_layout(_operation(input_rank=3))

    assert layout["input_rank"] == 3
    assert layout["lhs"]["view_shape"] == [1, 2220, 12288]
    assert layout["output"]["view_shape"] == [1, 2220, 4096]
    assert layout["rhs"]["view_shape"] == [12288, 4096]
    assert layout["rhs"]["storage_shape"] == [4096, 12288]


def test_rank_two_rejects_batched_tensor_shape() -> None:
    runner = _runner_module()

    with pytest.raises(ValueError, match="rank-two operation requires B=1"):
        runner._operation_layout(_operation(input_rank=2, batch=2))


def test_rank_two_helper_metadata_requires_explicit_safe_rank(tmp_path: Path) -> None:
    runner = _runner_module()
    metadata = tmp_path / "helper_metadata.txt"
    metadata.write_text(
        "schema_version=6\ninput_rank=2\nbatch_size=1\n"
        "workspace_size_bytes=0\nexecute_api_called=true\n"
        "device_kernel_submitted=false\nduration_consumed=false\n"
        "executor_lifecycle=consumed_by_aclnnMatmul\n"
        "process_peak_rss_kib=1\n",
        encoding="utf-8",
    )

    assert runner.parse_helper_metadata(metadata)["input_rank"] == 2

    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace("input_rank=2", "input_rank=4"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported input rank"):
        runner.parse_helper_metadata(metadata)


def test_direct_selection_domain_is_complete_and_unique() -> None:
    runner = _runner_module()

    plan = runner.load_bulk_launch_selection(
        ROOT / "launch_data/ascend_910b/gemm_launch_selection.csv.zst", 0, ROOT)

    assert plan["operation_count"] == 140544
    assert plan["operations"][0]["operation_id"] == "operation_000000"
    assert plan["operations"][-1]["operation_id"] == "operation_140543"
    assert plan["launch_selection_domain"]["physical_operation_count"] == 140544
    assert plan["launch_selection_domain"]["physical_domain_sha256"] == (
        "28cfa6222847684c337ea730f47637465782a8549638a4d70d0a56da0c2c569f")


def test_kernel_dataset_probe_flattens_batch_into_rank_two_contract() -> None:
    runner = _runner_module()

    plan = runner.load_kernel_dataset_probe(
        ROOT / "kernel_dataset/ascend_910b/gemm.json.zst", 0, ROOT)

    assert plan["case_count"] == 221
    operation = next(
        item for item in plan["operations"]
        if item["operation_id"] == "GEMM_B32_M1_K5120_N1536")
    assert operation == {
        "operation_id": "GEMM_B32_M1_K5120_N1536",
        "b": 1,
        "m": 32,
        "n": 1536,
        "k": 5120,
        "dtype": "bf16",
        "cube_math_type": 0,
        "logical_shape": {"b": 32, "m": 1, "n": 1536, "k": 5120},
        "workload_keys": [],
    }


def test_inline_capture_never_retains_temporary_artifact_paths() -> None:
    runner = _runner_module()
    result = {
        "dry_launch": {
            "launch": {
                "host_args": {"path": "/tmp/host_args.bin"},
                "placeholders": {"path": "/tmp/placeholders.bin"},
                "configuration": {"path": "/tmp/config.bin"},
                "metadata_path": "/tmp/launch.meta",
            },
        },
    }

    normalized = runner._inline_capture_paths(result)

    assert normalized["capture_storage"] == "inline_result_json"
    assert normalized["dry_launch"]["launch"]["metadata_path"] == (
        "embedded_in_result_json")
    assert all(
        normalized["dry_launch"]["launch"][field]["path"]
        == "embedded_in_result_json"
        for field in ("host_args", "placeholders", "configuration"))
