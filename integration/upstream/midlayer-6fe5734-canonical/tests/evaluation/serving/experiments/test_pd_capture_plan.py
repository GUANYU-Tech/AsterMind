import json

import pytest

from evaluation.serving.experiments import pd_capture_plan


def _engine():
    return {
        "async_scheduling": True,
        "compilation_level": 3,
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "cpu_offload_gb": 0,
        "cudagraph_metrics": False,
        "dtype": "bfloat16",
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "graph_mode": "full_and_piecewise_cuda_graph",
        "kv_block_size_tokens": 16,
        "max_concurrent_batches": 2,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "pipeline_parallel_size": 1,
        "safetensors_load_strategy": None,
        "safetensors_prefetch_num_threads": None,
        "scheduler_policy": "fcfs",
        "seed": 42,
        "shutdown_timeout_seconds": 0,
        "tensor_parallel_size": 1,
        "vllm_engine_generation": "v1",
    }


def _case():
    identity = {
        "case_id": "tp1_qps4",
        "hardware_id": "nvidia_a800",
        "model_id": "qwen3-8b",
        "raw_model_id": "Qwen3-8B",
        "model_revision": None,
        "source_kind": "sharegpt",
        "tokenizer_revision": None,
    }
    return {
        "identity": identity,
        "requests_row_count": 128,
        "requests_sha256": "requests-sha",
        "runtime": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "seed": 42,
            "target_qps": 4.0,
            "runtime_version": "0.11.0",
            "capture_enable_prefix_caching": True,
        },
    }


def _ground_truth(case, model_path):
    return {
        "schema_version": 1,
        "status": "success",
        "workload": {
            "identity": case["identity"],
            "request_count": 128,
            "expected_qps": 4.0,
            "seed": 42,
            "requests_sha256": "requests-sha",
        },
        "server": {
            "model_path": str(model_path),
            "client_vllm_version": "0.11.0",
            "scheduler_trace": {"enabled": True},
            "profiler": {"requested": False},
            "engine": _engine(),
        },
    }


def test_colocated_execution_settings_are_loaded_not_pd_defaulted(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    case = _case()
    source_path = tmp_path / "ground_truth.json"
    source_path.write_text(json.dumps(_ground_truth(case, model_path)))

    settings = pd_capture_plan._colocated_execution_settings(
        colocated_ground_truth=source_path, case=case, model=model_path
    )

    assert settings["gpu_memory_utilization"] == 0.8
    assert settings["max_model_len"] == 8192
    assert settings["max_num_batched_tokens"] == 8192
    assert settings["max_num_seqs"] == 256
    assert settings["kv_block_size_tokens"] == 16


def test_colocated_execution_settings_reject_request_mismatch(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    case = _case()
    source = _ground_truth(case, model_path)
    source["workload"]["requests_sha256"] = "different"
    source_path = tmp_path / "ground_truth.json"
    source_path.write_text(json.dumps(source))

    with pytest.raises(ValueError, match="requests differ"):
        pd_capture_plan._colocated_execution_settings(
            colocated_ground_truth=source_path, case=case, model=model_path
        )
