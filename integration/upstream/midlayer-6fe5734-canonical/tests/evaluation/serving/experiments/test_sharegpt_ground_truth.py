"""Direct contracts for the ShareGPT ground-truth server plan."""

from argparse import Namespace
import json
from pathlib import Path

import pytest

from evaluation.serving.experiments import sharegpt_ground_truth


def test_c500_scheduler_trace_exports_executor_hardware_contract(tmp_path, monkeypatch):
    for key in sharegpt_ground_truth.C500_DEVICE_EVENT_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    args = Namespace(
        devices=[1],
        port=18956,
        seed=42,
        gpu_memory_utilization=0.8,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
        kv_block_size_tokens=128,
        max_dispatch_lag_ms=100.0, arrival_probe_per_1000_tokens_ms=0.0,
        model=str(tmp_path),
        scheduler_trace_output=tmp_path / "scheduler_steps.jsonl",
        scheduler_request_add_output=None,
        a800_compilation_counter_output=None,
    )
    case = {
        "identity": {"hardware_id": "metax_c500", "model_id": "qwen3-4b"},
        "requests_row_count": 128,
        "runtime": {
            "runtime_version": "0.24.0",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "source_commit": "vllm=0.24.0;core=test",
        },
    }

    plan = sharegpt_ground_truth._server_plan(args, case)

    assert plan["scheduler_trace"]["enabled"] is True
    assert plan["environment"]["MIDLAYER_SCHEDULER_TRACE_HARDWARE_ID"] == (
        "metax_c500"
    )
    assert plan["environment"]["MIDLAYER_SCHEDULER_TRACE_WORKER_RANK_COUNT"] == "1"


def test_a800_async_executor_log_validation_is_hardware_scoped():
    assert sharegpt_ground_truth._requires_a800_async_scheduling_validation(
        "nvidia_a800", True)
    assert not sharegpt_ground_truth._requires_a800_async_scheduling_validation(
        "metax_c500", True)
    assert not sharegpt_ground_truth._requires_a800_async_scheduling_validation(
        "ascend_910b", True)
    assert not sharegpt_ground_truth._requires_a800_async_scheduling_validation(
        "nvidia_a800", False)


def test_vllm_server_process_group_uses_the_reserved_port(tmp_path):
    proc_root = tmp_path / "proc"
    target = proc_root / "101"
    target.mkdir(parents=True)
    (target / "cmdline").write_bytes(
        b"python\0-m\0vllm.entrypoints.cli.main\0serve\0model\0"
        b"--port\0" b"18153\0"
    )
    (target / "stat").write_text("101 (python) S 1 77 77 0 0 0\n")
    unrelated = proc_root / "102"
    unrelated.mkdir()
    (unrelated / "cmdline").write_bytes(
        b"python\0-m\0vllm.entrypoints.cli.main\0serve\0other\0"
        b"--port\0" b"18154\0"
    )
    (unrelated / "stat").write_text("102 (python) S 1 88 88 0 0 0\n")

    assert sharegpt_ground_truth._vllm_server_process_group_for_port(
        18153, proc_root=proc_root) == 77


def test_vllm_server_process_group_rejects_ambiguous_port(tmp_path):
    proc_root = tmp_path / "proc"
    for pid, group in ((101, 77), (102, 88)):
        process_path = proc_root / str(pid)
        process_path.mkdir(parents=True)
        (process_path / "cmdline").write_bytes(
            b"python\0-m\0vllm.entrypoints.cli.main\0serve\0model\0"
            b"--port\0" b"18153\0"
        )
        (process_path / "stat").write_text(
            f"{pid} (python) S 1 {group} {group} 0 0 0\n")

    with pytest.raises(RuntimeError, match="exactly one live vLLM server"):
        sharegpt_ground_truth._vllm_server_process_group_for_port(
            18153, proc_root=proc_root)


def test_ascend_step_timestamp_is_forwarded_only_for_ascend(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDLAYER_910B_MSPROF_STEP_TIMESTAMP", "1")
    args = Namespace(
        devices=[2], port=18956, seed=42, gpu_memory_utilization=0.8,
        max_model_len=8192, max_num_batched_tokens=8192, max_num_seqs=256,
        kv_block_size_tokens=16, max_dispatch_lag_ms=100.0, arrival_probe_per_1000_tokens_ms=0.0, model=str(tmp_path),
        scheduler_trace_output=tmp_path / "scheduler_steps.jsonl",
        scheduler_request_add_output=None, a800_compilation_counter_output=None,
    )
    case = {
        "identity": {"hardware_id": "ascend_910b", "model_id": "qwen3-4b"},
        "requests_row_count": 128,
        "runtime": {
            "runtime_version": "0.18.0", "dtype": "bfloat16",
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "expert_parallel_size": 1, "source_commit": "vllm=0.18.0;core=test",
        },
    }

    plan = sharegpt_ground_truth._server_plan(args, case)

    assert plan["environment"]["MIDLAYER_910B_MSPROF_STEP_TIMESTAMP"] == "1"


def test_ascend_mstx_step_mark_is_forwarded_only_for_ascend(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDLAYER_910B_MSPROFTX_STEP_MARK", "1")
    args = Namespace(
        devices=[2], port=18956, seed=42, gpu_memory_utilization=0.8,
        max_model_len=8192, max_num_batched_tokens=8192, max_num_seqs=256,
        kv_block_size_tokens=16, max_dispatch_lag_ms=100.0, arrival_probe_per_1000_tokens_ms=0.0, model=str(tmp_path),
        scheduler_trace_output=tmp_path / "scheduler_steps.jsonl",
        scheduler_request_add_output=None, a800_compilation_counter_output=None,
    )
    case = {
        "identity": {"hardware_id": "ascend_910b", "model_id": "qwen3-4b"},
        "requests_row_count": 128,
        "runtime": {
            "runtime_version": "0.18.0", "dtype": "bfloat16",
            "tensor_parallel_size": 1, "pipeline_parallel_size": 1,
            "expert_parallel_size": 1, "source_commit": "vllm=0.18.0;core=test",
        },
    }

    plan = sharegpt_ground_truth._server_plan(args, case)

    assert plan["environment"]["MIDLAYER_910B_MSPROFTX_STEP_MARK"] == "1"


def test_scheduler_runtime_depth_accepts_vllm_config_evidence(tmp_path):
    trace_path = tmp_path / "scheduler_steps.jsonl"
    Path(f"{trace_path}.runtime.json").write_text(json.dumps({
        "schema_version": 1,
        "scheduler_pipeline_depth": 2,
        "scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_vllm_config_field",
            "source_field": "vllm_config.max_concurrent_batches",
            "configured_depth": 2,
            "executor_class": "vllm.v1.executor.uniproc_executor.UniProcExecutor",
            "pipeline_parallel_size": 1,
            "runtime_version": "0.24.0",
            "source_identity": {
                "module_file": "/opt/conda/lib/python3.12/site-packages/vllm/config/vllm.py",
                "module_name": "vllm.v1.executor.uniproc_executor",
            },
        },
    }))

    metadata = sharegpt_ground_truth._scheduler_runtime_depth_metadata(trace_path)

    assert metadata["scheduler_pipeline_depth"] == 2
    assert metadata["scheduler_pipeline_depth_evidence"]["configured_depth"] == 2


def test_scheduler_runtime_depth_rejects_mismatched_vllm_config_depth(tmp_path):
    trace_path = tmp_path / "scheduler_steps.jsonl"
    Path(f"{trace_path}.runtime.json").write_text(json.dumps({
        "schema_version": 1,
        "scheduler_pipeline_depth": 2,
        "scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_vllm_config_field",
            "source_field": "vllm_config.max_concurrent_batches",
            "configured_depth": 3,
            "executor_class": "vllm.v1.executor.uniproc_executor.UniProcExecutor",
            "pipeline_parallel_size": 1,
            "runtime_version": "0.24.0",
            "source_identity": {
                "module_file": "/opt/conda/lib/python3.12/site-packages/vllm/config/vllm.py",
                "module_name": "vllm.v1.executor.uniproc_executor",
            },
        },
    }))

    with pytest.raises(ValueError, match="configured depth"):
        sharegpt_ground_truth._scheduler_runtime_depth_metadata(trace_path)


def test_scheduler_trace_wait_accepts_published_output(tmp_path):
    trace_path = tmp_path / "scheduler_steps.jsonl"
    trace_path.write_text("{}\n")

    sharegpt_ground_truth._wait_for_scheduler_trace_output(trace_path)


def _write_worker_dispatch_trace(
    tmp_path: Path,
    worker_envelopes: list[list[tuple[int, int]]],
    *,
    executor_class: str = (
        "vllm.v1.executor.multiproc_executor.MultiprocExecutor"
    ),
) -> Path:
    """Write a vLLM 0.11 worker-dispatch Scheduler trace with its sidecars.

    ``worker_envelopes[i][rank]`` is the (start, end) worker envelope of step
    ``i``; step ``i``'s executor RPC envelope is ``[100*i + 10, 100*i + 50]``.
    """

    trace_path = tmp_path / "scheduler_steps.raw.jsonl"
    Path(f"{trace_path}.runtime.json").write_text(json.dumps({
        "schema_version": 1,
        "scheduler_pipeline_depth": 2,
        "scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_executor_property",
            "source_field": "max_concurrent_batches",
            "executor_class": executor_class,
            "pipeline_parallel_size": 1,
            "runtime_version": "0.11.0",
            "source_identity": (
                "vllm=0.11.0;core=4573563d...cfbedd;executor=ad938d34...e0cd16"
            ),
        },
    }))
    Path(f"{trace_path}.clock.json").write_text(json.dumps({
        "schema_version": 1,
        "clock_domain": "shared_host_time.perf_counter_ns",
        "host_boot_id": "boot-id",
        "trace_origin_monotonic_ns": 1000,
    }))
    Path(f"{trace_path}.warmup.json").write_text(json.dumps({
        "max_scheduled_request_count": 8,
        "mixed_prefill_decode_step_count": 1,
    }))
    rows = []
    for index, envelopes in enumerate(worker_envelopes):
        base = 100 * index
        step_id = f"vllm:{index}"
        rows.append({
            "step_index": index,
            "step_id": step_id,
            "cumulative_completed_requests": index + 1,
            "cumulative_scheduled_tokens": 4 * (index + 1),
            "scheduled_tokens": 4,
            "scheduled_requests": [{"scheduled_tokens": 4}],
            "prefill_tokens": 4,
            "decode_tokens": 0,
            "schedule_start_ns": base,
            "schedule_end_ns": base + 5,
            "device_execution_start_ns": base + 10,
            "device_execution_end_ns": base + 50,
            "scheduler_update_start_ns": base + 60,
            "scheduler_update_end_ns": base + 70,
            "device_execution_clock": "time.perf_counter_ns",
            "device_execution_source": "host_monotonic_executor_rpc_envelope",
            "in_flight_step_count": 1,
            "rank_observations": [
                {
                    "schema_version": 1,
                    "source": "vllm_v1_worker_cudagraph_dispatch",
                    "rank": rank,
                    "trace_step_id": step_id,
                    "scheduled_tokens": 4,
                    "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
                    "runtime_cudagraph_mode": "FULL",
                    "dispatch_input_key": {
                        "num_tokens": 4, "uniform_decode": False},
                    "selected_graph_key": {
                        "num_tokens": 4, "uniform_decode": False},
                    "graph_fallback_reason": None,
                    "num_unpadded_tokens": 4,
                    "padded_total_tokens": 4,
                    "padding_tokens": 0,
                    "worker_execution_start_ns": base + start,
                    "worker_execution_end_ns": base + end,
                    "worker_execution_clock": "time.perf_counter_ns",
                    "worker_execution_source": (
                        "vllm_v1_worker_execute_model_envelope"),
                }
                for rank, (start, end) in enumerate(envelopes)
            ],
        })
    trace_path.write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return trace_path


def test_worker_dispatch_summary_records_late_non_reply_rank_end(tmp_path):
    # vLLM 0.11 MultiprocExecutor waits only for output rank 0, so rank 1 may
    # return from execute_model after the executor RPC envelope closed.
    trace_path = _write_worker_dispatch_trace(tmp_path, [
        [(12, 40), (13, 41)],
        [(12, 40), (14, 53)],
        [(12, 49), (13, 57)],
    ])

    summary = sharegpt_ground_truth._scheduler_trace_summary(trace_path, 3)

    assert summary["worker_ranks"] == [0, 1]
    assert summary["worker_execution_envelope"] == {
        "executor_reply_rank": 0,
        "non_reply_rank_end_after_executor_end_count": 2,
        "non_reply_rank_end_after_executor_end_max_ns": 7,
        "non_reply_rank_end_after_executor_end": [
            {"step_id": "vllm:1", "rank": 1,
             "worker_end_after_executor_end_ns": 3},
            {"step_id": "vllm:2", "rank": 1,
             "worker_end_after_executor_end_ns": 7},
        ],
    }


def test_worker_dispatch_summary_records_no_overhang_as_none(tmp_path):
    trace_path = _write_worker_dispatch_trace(tmp_path, [
        [(12, 40), (13, 50)],
    ])

    summary = sharegpt_ground_truth._scheduler_trace_summary(trace_path, 1)

    assert summary["worker_execution_envelope"] == {
        "executor_reply_rank": 0,
        "non_reply_rank_end_after_executor_end_count": 0,
        "non_reply_rank_end_after_executor_end_max_ns": None,
        "non_reply_rank_end_after_executor_end": [],
    }


@pytest.mark.parametrize("envelopes", [
    # The reply rank's worker cannot finish after its own reply arrived.
    [(12, 51), (13, 41)],
    # No rank may start before the executor submitted the step.
    [(12, 40), (9, 41)],
    # A late non-reply end still requires a causal worker envelope.
    [(12, 40), (45, 44)],
])
def test_worker_dispatch_summary_rejects_uncontained_envelopes(
    tmp_path, envelopes
):
    trace_path = _write_worker_dispatch_trace(tmp_path, [
        [(12, 40), (13, 41)],
        envelopes,
    ])
    with pytest.raises(ValueError, match="outside executor envelope"):
        sharegpt_ground_truth._scheduler_trace_summary(trace_path, 2)


def test_worker_dispatch_summary_rejects_unaudited_multi_rank_executor(
    tmp_path,
):
    trace_path = _write_worker_dispatch_trace(
        tmp_path,
        [[(12, 40), (13, 41)]],
        executor_class="vllm.v1.executor.ray_distributed_executor.RayDistributedExecutor",
    )

    with pytest.raises(ValueError, match="unaudited executor reply contract"):
        sharegpt_ground_truth._scheduler_trace_summary(trace_path, 1)


def test_reserve_client_fd_table_grows_fd_table_before_replay():
    import subprocess
    import sys

    # Fresh interpreter: the table only grows once per process.
    code = (
        "from evaluation.serving.experiments import sharegpt_ground_truth as gt\n"
        "def fd_size():\n"
        "    for line in open('/proc/self/status'):\n"
        "        if line.startswith('FDSize:'):\n"
        "            return int(line.split()[1])\n"
        "    raise RuntimeError('no FDSize')\n"
        "before = fd_size()\n"
        "gt._reserve_client_fd_table()\n"
        "after = fd_size()\n"
        "assert before < 1024, before\n"
        "assert after >= gt.CLIENT_FD_TABLE_RESERVE + 1, after\n"
        "print(before, after)\n"
    )
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, check=True,
                               cwd=str(Path(__file__).resolve().parents[4]))
    before, after = map(int, completed.stdout.split())
    assert after > before
