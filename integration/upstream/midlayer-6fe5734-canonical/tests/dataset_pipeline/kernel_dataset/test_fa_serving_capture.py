"""Contracts for the frozen cross-hardware FA serving capture adapter."""

from __future__ import annotations

import json
from pathlib import Path

from dataset_pipeline.kernel_dataset.fa_serving_capture import (
    build_capture_plan,
    logical_identity_sha256,
    validate_runner_capture_metadata,
)


def _loaded_manifest():
    workload = {
        "B": 2,
        "D": 128,
        "HKV": 8,
        "HQ": 32,
        "KV_LEN": 17,
        "KV_LENS": [17, 9],
        "Q_LEN": 1,
        "Q_LENS": [1, 1],
        "causal": False,
        "dtype": "bf16",
        "kv_dist": "explicit",
        "kv_layout": "paged_varlen",
        "split_policy": "serving",
    }
    case = {
        "case_id": "",
        "logical_identity_sha256": "",
        "phase": "decode",
        "workload": workload,
    }
    identity = logical_identity_sha256(case)
    case["logical_identity_sha256"] = identity
    case["case_id"] = (
        "FA_SERVING_HQ32_HKV8_D128_DECODE_B2_QSUM2_QMAX1_KVMAX17_"
        f"{identity[:12]}")
    return {
        "path": Path("synthetic_capture_manifest.json"),
        "sha256": "a" * 64,
        "manifest": {"cases": [case]},
    }


def test_capture_plan_preserves_workload_and_hardware_contracts():
    loaded = _loaded_manifest()
    for hardware_id, page_size, requested_splits in (
            ("nvidia_a800", 16, 1),
            ("metax_c500", 16, 1),
            ("ascend_910b", 128, None)):
        plan = build_capture_plan(loaded, hardware_id=hardware_id)
        assert plan["case_count"] == 1
        for entry in plan["plans"]:
            metadata = entry["capture_metadata"]
            assert metadata["hardware_id"] == hardware_id
            assert metadata["kv_page_size_tokens"] == page_size
            assert metadata["requested_num_splits"] == requested_splits
            assert metadata["workload"]["Q_LENS"]
            assert metadata["workload"]["KV_LENS"]


def test_runner_capture_metadata_requires_exact_unmodified_workload(tmp_path):
    loaded = _loaded_manifest()
    plan = build_capture_plan(loaded, hardware_id="metax_c500")
    metadata = plan["plans"][0]["capture_metadata"]
    path = tmp_path / "capture_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    assert validate_runner_capture_metadata(
        path,
        hardware_id="metax_c500",
        actual_workload=metadata["workload"],
    ) == metadata

    rewritten = dict(metadata["workload"])
    rewritten["KV_LENS"] = list(rewritten["KV_LENS"])
    rewritten["KV_LENS"][0] += 1
    try:
        validate_runner_capture_metadata(
            path,
            hardware_id="metax_c500",
            actual_workload=rewritten,
        )
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("rewritten runner input must fail closed")
