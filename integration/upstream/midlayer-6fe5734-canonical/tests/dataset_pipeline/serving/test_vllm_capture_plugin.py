"""Contract checks for the opt-in C500 vLLM raw capture plugin."""

from __future__ import annotations

import json

import pytest

from dataset_pipeline.serving_dataset import vllm_capture_plugin


class _SchedulerOutput:
    num_scheduled_tokens = {"req_0": 4}
    total_num_scheduled_tokens = 4
    finished_req_ids = {"req_0"}


class _Scheduler:
    def schedule(self):
        return _SchedulerOutput()


def test_capture_plugin_is_inert_without_both_output_paths(monkeypatch):
    monkeypatch.delenv("CUDA_SIDECAR_PATH", raising=False)
    monkeypatch.delenv("CUDA_OPERATOR_TRACE_PATH", raising=False)

    assert vllm_capture_plugin.install_from_environment() is False


def test_scheduler_capture_writes_raw_adapter_record(monkeypatch, tmp_path):
    scheduler_path = tmp_path / "scheduler.jsonl"
    monkeypatch.setenv("CUDA_SIDECAR_PATH", str(scheduler_path))
    monkeypatch.setenv("CUDA_OPERATOR_TRACE_PATH", str(tmp_path / "operators.jsonl"))

    vllm_capture_plugin._patch_scheduler(_Scheduler)
    assert _Scheduler().schedule() is not None

    rows = [json.loads(line) for line in scheduler_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["iter_id"] == 0
    assert rows[0]["scheduled"] == {"req_0": 4}
    assert rows[0]["total_tokens"] == 4
    assert rows[0]["finished"] == ["req_0"]


def test_capture_plugin_rejects_incomplete_output_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_SIDECAR_PATH", str(tmp_path / "scheduler.jsonl"))
    monkeypatch.delenv("CUDA_OPERATOR_TRACE_PATH", raising=False)

    with pytest.raises(RuntimeError, match="CUDA_OPERATOR_TRACE_PATH"):
        vllm_capture_plugin._required_path("CUDA_OPERATOR_TRACE_PATH")


def test_capture_plugin_uses_plain_python_for_spawned_engine(monkeypatch, tmp_path):
    wrapper = tmp_path / "spawn.sh"
    wrapper.write_text("#!/bin/sh\n")
    wrapper.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    monkeypatch.setenv("MIDLAYER_C500_CAPTURE_SPAWN_WRAPPER", str(wrapper))
    monkeypatch.setenv("MIDLAYER_C500_CAPTURE_SPAWN_PYTHON", str(python))
    monkeypatch.setenv("MIDLAYER_C500_CAPTURE_SPAWN_PYTHONPATH", "bootstrap")
    configured = []
    monkeypatch.setattr(
        vllm_capture_plugin.multiprocessing,
        "set_executable",
        configured.append,
    )

    vllm_capture_plugin._configure_spawn_bootstrap()

    assert configured == [str(wrapper.resolve())]


def test_operator_writer_flushes_on_capture_shutdown(tmp_path):
    path = tmp_path / "operators.jsonl"

    vllm_capture_plugin._append_jsonl(path, {"step_id": 0}, flush=False)
    vllm_capture_plugin._close_writers()

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"step_id": 0}
    ]
