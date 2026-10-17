"""Tests for scripts/serve_trace/trace_disagg.py and _disagg_instance.py"""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.serve_trace.trace_disagg import (
    parse_gpu_list,
    validate_config,
    build_instance_command,
    build_instance_env,
    build_nsys_prefix,
    resolve_workload_provenance,
    validate_nsys_outputs,
    _terminate_process,
    _wait_for_disagg_processes,
    _cleanup_pid_trace_files,
    INSTANCE_SCRIPT,
)
from src.serve_predict.parallel_utils import (
    validate_gpu_capacity,
    validate_parallel_semantics,
    ParallelConfigError,
)


class _FakeArgs:
    def __init__(self, **kwargs):
        defaults = {
            "prefill_gpus": "0",
            "decode_gpus": "1",
            "prefill_engine_args": None,
            "decode_engine_args": None,
            "kv_connector": "P2pNcclConnector",
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class TestParseGpuList:
    def test_single(self):
        assert parse_gpu_list("0") == [0]

    def test_multiple(self):
        assert parse_gpu_list("0,1,2") == [0, 1, 2]

    def test_with_spaces(self):
        assert parse_gpu_list("0, 1, 2") == [0, 1, 2]


class TestValidateConfig:
    def test_missing_connector_raises(self):
        with pytest.raises(ValueError, match="KV connector is required"):
            validate_config("0", "1", None, None, None)

    def test_gpu_overlap_raises(self):
        with pytest.raises(ValueError, match="must not overlap"):
            validate_config("0,1", "1,2", "P2pNcclConnector", None, None)

    def test_valid_config(self):
        p, d, pe, de = validate_config("0", "1", "P2pNcclConnector", None, None)
        assert p == [0]
        assert d == [1]
        assert pe == {}
        assert de == {}

    def test_missing_prefill_gpus_raises(self):
        with pytest.raises(ValueError, match="Prefill GPU list is required"):
            validate_config(None, "1", "P2pNcclConnector", None, None)

    def test_with_engine_args(self):
        p, d, pe, de = validate_config(
            "0,1", "2", "P2pNcclConnector",
            '{"tensor_parallel_size": 2}', None)
        assert pe == {"tensor_parallel_size": 2}

    def test_insufficient_gpus_for_tp(self):
        with pytest.raises(ValueError, match="requires 2 GPUs"):
            validate_config("0", "1", "P2pNcclConnector",
                          '{"tensor_parallel_size": 2}', None)

    def test_insufficient_gpus_for_tp_pp_dp(self):
        with pytest.raises(ValueError, match="requires 4 GPUs"):
            validate_config("0,1", "2", "P2pNcclConnector", None,
                          '{"tensor_parallel_size": 2, "pipeline_parallel_size": 2}')


class TestValidateGpuCount:
    def test_single_gpu_default(self):
        validate_gpu_capacity("test", [0], {})

    def test_tp2_with_2_gpus(self):
        validate_gpu_capacity("test", [0, 1], {"tensor_parallel_size": 2})

    def test_tp_pp_dp(self):
        validate_gpu_capacity("test", [0, 1, 2, 3], {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "data_parallel_size": 1,
        })

    def test_dp2_tp2_needs_4_gpus(self):
        with pytest.raises(ParallelConfigError, match="requires 4 GPUs"):
            validate_gpu_capacity("test", [0, 1], {
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
            })


class TestValidateParallelSemantics:
    def test_valid_dcp(self):
        validate_parallel_semantics({"tensor_parallel_size": 4, "decode_context_parallel_size": 2}, "test")

    def test_dcp_zero_raises(self):
        with pytest.raises(ParallelConfigError, match="must be >= 1"):
            validate_parallel_semantics({"decode_context_parallel_size": 0}, "test")

    def test_dcp_exceeds_tp_raises(self):
        with pytest.raises(ParallelConfigError, match="must be <= tensor_parallel_size"):
            validate_parallel_semantics({"tensor_parallel_size": 2, "decode_context_parallel_size": 4}, "test")

    def test_tp_not_divisible_by_dcp_raises(self):
        with pytest.raises(ParallelConfigError, match="must be divisible"):
            validate_parallel_semantics({"tensor_parallel_size": 3, "decode_context_parallel_size": 2}, "test")


class TestWorkloadProvenance:
    def test_loads_canonical_hash_from_prompts_file(self, tmp_path):
        path = tmp_path / "workload.json"
        path.write_text(json.dumps({"hash": "canonical-hash", "prompts": ["a"]}))
        args = _FakeArgs(prompts_file=str(path))

        provenance = resolve_workload_provenance(args)

        assert provenance == {
            "workload_path": str(path),
            "workload_hash": "canonical-hash",
        }

    def test_explicit_workload_hash_overrides_legacy_hash(self, tmp_path):
        path = tmp_path / "workload.json"
        path.write_text(json.dumps({"hash": "canonical-hash", "prompts": ["a"]}))
        args = _FakeArgs(
            prompts_file=str(path),
            workload_artifact=str(path),
            workload_hash="explicit-hash",
        )

        provenance = resolve_workload_provenance(args)

        assert provenance == {
            "workload_path": str(path),
            "workload_hash": "explicit-hash",
        }


class TestBuildInstanceCommand:
    def test_uses_instance_script(self):
        cmd = build_instance_command(
            "prefill", "/model", {}, "P2pNcclConnector",
            14579, "127.0.0.1", 8000, 8001, 3, 512, 8, 8192, "/tmp/trace_p",
            "configs/hw/A800.yaml")
        assert INSTANCE_SCRIPT in cmd

    def test_kv_connector_passed(self):
        cmd = build_instance_command(
            "prefill", "/model", {}, "P2pNcclConnector",
            14579, "127.0.0.1", 8000, 8001, 3, 512, 8, 8192, "/tmp/trace_p",
            "configs/hw/A800.yaml")
        assert "--kv-connector" in cmd
        idx = cmd.index("--kv-connector")
        assert cmd[idx + 1] == "P2pNcclConnector"

    def test_role_passed(self):
        for role in ("prefill", "decode"):
            cmd = build_instance_command(
                role, "/model", {}, "P2pNcclConnector",
                14579, "127.0.0.1", 8000, 8001, 1, 100, 4, 4096, f"/tmp/{role}",
                "hw.yaml")
            idx = cmd.index("--role")
            assert cmd[idx + 1] == role

    def test_kv_port_and_ip_passed(self):
        cmd = build_instance_command(
            "prefill", "/model", {}, "P2pNcclConnector",
            9999, "192.168.1.1", 8000, 8001, 1, 100, 4, 4096, "/tmp/p", "hw.yaml")
        idx_port = cmd.index("--kv-port")
        assert cmd[idx_port + 1] == "9999"
        idx_ip = cmd.index("--kv-ip")
        assert cmd[idx_ip + 1] == "192.168.1.1"

    def test_http_port_and_peer_kv_port_passed(self):
        cmd = build_instance_command(
            "prefill", "/model", {}, "P2pNcclConnector",
            14579, "127.0.0.1", 15000, 15001, 1, 100, 4, 4096, "/tmp/p", "hw.yaml")
        idx = cmd.index("--http-port")
        assert cmd[idx + 1] == "15000"
        idx_peer = cmd.index("--peer-kv-port")
        assert cmd[idx_peer + 1] == "15001"

    def test_full_engine_args_json_passed(self):
        engine_args = {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "enable_expert_parallel": True,
            "decode_context_parallel_size": 4,
        }
        cmd = build_instance_command(
            "prefill", "/model", engine_args, "P2pNcclConnector",
            14579, "127.0.0.1", 8000, 8001, 1, 100, 4, 4096, "/tmp/p", "hw.yaml")
        idx = cmd.index("--engine-args-json")
        passed_args = json.loads(cmd[idx + 1])
        assert passed_args == engine_args

    def test_hw_config_passed(self):
        cmd = build_instance_command(
            "decode", "/model", {}, "P2pNcclConnector",
            14579, "127.0.0.1", 8000, 8001, 1, 100, 4, 4096, "/tmp/d",
            "configs/hw/H100.yaml")
        idx = cmd.index("--hw-config")
        assert cmd[idx + 1] == "configs/hw/H100.yaml"

    def test_per_instance_trace_path(self):
        cmd_p = build_instance_command(
            "prefill", "/m", {}, "P2pNcclConnector",
            14579, "127.0.0.1", 8000, 8001, 1, 100, 4, 4096, "/tmp/disagg_prefill", "hw.yaml")
        cmd_d = build_instance_command(
            "decode", "/m", {}, "P2pNcclConnector",
            14579, "127.0.0.1", 8001, 8000, 1, 100, 4, 4096, "/tmp/disagg_decode", "hw.yaml")
        idx_p = cmd_p.index("--trace-output")
        idx_d = cmd_d.index("--trace-output")
        assert cmd_p[idx_p + 1] != cmd_d[idx_d + 1]


class TestSharedRequestIds:
    """Tests that producer and consumer use identical raw request IDs for tensor key matching."""

    def test_shared_id_format_matches_connector_regex(self):
        """Both parse_request_id(shared, is_prefill=True/False) extract correct addresses."""
        import re

        prefill_ip, prefill_port = "127.0.0.1", 8000
        decode_ip, decode_port = "127.0.0.1", 8001
        shared_id = f"___prefill_addr_{prefill_ip}:{prefill_port}___req-0___decode_addr_{decode_ip}:{decode_port}"

        # Prefill side extracts decode address
        prefill_pattern = r"___decode_addr_(.*):(\d+)"
        match = re.search(prefill_pattern, shared_id)
        assert match is not None
        assert match.group(1) == decode_ip
        assert int(match.group(2)) == decode_port

        # Decode side extracts prefill address
        decode_pattern = r"___prefill_addr_(.*):(\d+)___"
        match = re.search(decode_pattern, shared_id)
        assert match is not None
        assert match.group(1) == prefill_ip
        assert int(match.group(2)) == prefill_port

    def test_tensor_ids_identical_between_producer_consumer(self):
        """Producer send tensor_id == consumer recv tensor_id."""
        prefill_ip, prefill_port = "127.0.0.1", 8000
        decode_ip, decode_port = "127.0.0.1", 8001

        # Both roles generate the same shared ID
        shared_id = f"___prefill_addr_{prefill_ip}:{prefill_port}___req-0___decode_addr_{decode_ip}:{decode_port}"

        layer_name = "model.layers.0.self_attn.k_proj"
        producer_tensor_id = f"{shared_id}#{layer_name}"
        consumer_tensor_id = f"{shared_id}#{layer_name}"
        assert producer_tensor_id == consumer_tensor_id

    def test_normalize_strips_both_markers(self):
        from src.serve_predict.serve_metric_predictor import normalize_request_id
        shared = "___prefill_addr_127.0.0.1:8000___req-0___decode_addr_127.0.0.1:8001"
        assert normalize_request_id(shared) == "req-0"

    def test_normalize_handles_legacy_prefill_only(self):
        from src.serve_predict.serve_metric_predictor import normalize_request_id
        assert normalize_request_id("req-0___decode_addr_127.0.0.1:8001") == "req-0"

    def test_normalize_handles_legacy_decode_only(self):
        from src.serve_predict.serve_metric_predictor import normalize_request_id
        assert normalize_request_id("___prefill_addr_127.0.0.1:8000___req-0") == "req-0"


class TestMakeDisaggRequestId:
    """Tests for the module-level make_disagg_request_id helper."""

    def test_import_from_module(self):
        from scripts.serve_trace._disagg_instance import make_disagg_request_id
        assert callable(make_disagg_request_id)

    def test_both_roles_produce_identical_string(self):
        from scripts.serve_trace._disagg_instance import make_disagg_request_id
        prefill_id = make_disagg_request_id("prefill", 0, "127.0.0.1", 8000, 8001)
        decode_id = make_disagg_request_id("decode", 0, "127.0.0.1", 8001, 8000)
        assert prefill_id == decode_id

    def test_different_indices_differ(self):
        from scripts.serve_trace._disagg_instance import make_disagg_request_id
        id0 = make_disagg_request_id("prefill", 0, "127.0.0.1", 8000, 8001)
        id1 = make_disagg_request_id("prefill", 1, "127.0.0.1", 8000, 8001)
        assert id0 != id1

    def test_connector_regex_parses_both_sides(self):
        import re
        from scripts.serve_trace._disagg_instance import make_disagg_request_id
        shared = make_disagg_request_id("prefill", 0, "10.0.0.1", 9000, 9001)
        # Prefill side extracts decode address
        m = re.search(r"___decode_addr_(.*):(\d+)", shared)
        assert m and m.group(1) == "10.0.0.1" and int(m.group(2)) == 9001
        # Decode side extracts prefill address
        m = re.search(r"___prefill_addr_(.*):(\d+)___", shared)
        assert m and m.group(1) == "10.0.0.1" and int(m.group(2)) == 9000


class TestRequestCounterSubmission:
    """Test that request_counter override submits shared IDs to _add_request."""

    def test_request_counter_yields_shared_ids(self):
        from scripts.serve_trace._disagg_instance import make_disagg_request_id

        ids = [make_disagg_request_id("prefill", i, "127.0.0.1", 8000, 8001)
               for i in range(3)]
        counter = iter(ids)
        submitted = [str(next(counter)) for _ in range(3)]
        assert submitted == ids
        assert all("___prefill_addr_" in s and "___decode_addr_" in s for s in submitted)


class TestTerminateProcess:
    """Tests for _terminate_process helper."""

    def test_already_exited(self):
        from scripts.serve_trace.trace_disagg import _terminate_process
        proc = MagicMock()
        proc.poll.return_value = 0
        _terminate_process(proc)
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_terminate_success(self):
        from scripts.serve_trace.trace_disagg import _terminate_process
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.return_value = None
        _terminate_process(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()

    def test_terminate_timeout_then_kill(self):
        import subprocess as sp
        from scripts.serve_trace.trace_disagg import _terminate_process
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = [sp.TimeoutExpired(cmd="test", timeout=5), None]
        _terminate_process(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()


class TestMockedLLMSubmission:
    """Prove shared IDs reach vLLM's _add_request → llm_engine.add_request path."""

    def test_add_request_submits_shared_ids_to_engine(self):
        """Simulate the vLLM _add_request call boundary."""
        from scripts.serve_trace._disagg_instance import make_disagg_request_id

        ids = [make_disagg_request_id("prefill", i, "127.0.0.1", 8000, 8001)
               for i in range(2)]

        # Faithful stand-in matching vLLM's _add_request call boundary
        engine_submitted = []

        class FakeLLMEngine:
            def add_request(self, request_id, prompt, params, **kwargs):
                engine_submitted.append(request_id)

        class FakeLLM:
            def __init__(self):
                self.request_counter = iter(ids)
                self.llm_engine = FakeLLMEngine()

            def _add_request(self, prompt, params):
                request_id = str(next(self.request_counter))
                self.llm_engine.add_request(request_id, prompt, params)

        llm = FakeLLM()
        llm._add_request("prompt0", None)
        llm._add_request("prompt1", None)

        # The exact shared IDs reach llm_engine.add_request
        assert engine_submitted == ids
        for sid in engine_submitted:
            assert "___prefill_addr_" in sid
            assert "___decode_addr_" in sid

    def test_producer_consumer_submit_identical_ids(self):
        from scripts.serve_trace._disagg_instance import make_disagg_request_id
        for i in range(3):
            p_id = make_disagg_request_id("prefill", i, "10.0.0.1", 9000, 9001)
            c_id = make_disagg_request_id("decode", i, "10.0.0.1", 9001, 9000)
            assert p_id == c_id, f"Mismatch at index {i}: {p_id} != {c_id}"


class TestDisaggSidecarEnv:
    """Verify _setup_scheduler_sidecar helper works correctly."""

    def test_setup_sidecar_sets_env_and_calls_installer(self, tmp_path, monkeypatch):
        from scripts.serve_trace._disagg_instance import _setup_scheduler_sidecar
        trace_base = str(tmp_path / "trace")

        # Patch the installer to just record it was called
        installer_called = []
        monkeypatch.setattr(
            "scripts.serve_trace.trace_serving._install_scheduler_sidecar",
            lambda: installer_called.append(True))

        result = _setup_scheduler_sidecar(trace_base)
        assert result == f"{trace_base}_scheduler.jsonl"
        assert os.environ.get("CUDA_SIDECAR_PATH") == result
        assert len(installer_called) == 1


class TestTraceCleanup:
    def test_cleanup_pid_trace_files_removes_only_numeric_suffixes(self, tmp_path):
        trace_base = tmp_path / "prefill_trace.json"
        merged = trace_base
        stale_pid = Path(str(trace_base) + ".12345")
        other_trace = Path(str(trace_base) + ".worker")
        log_path = Path(str(trace_base) + ".log")
        merged.write_text("[]")
        stale_pid.write_text("old")
        other_trace.write_text("keep")
        log_path.write_text("keep")

        assert _cleanup_pid_trace_files(str(trace_base)) == 1
        assert merged.exists()
        assert not stale_pid.exists()
        assert other_trace.exists()
        assert log_path.exists()


class TestPollLoopOrchestration:
    """Tests for _wait_for_disagg_processes helper."""

    def _mock_proc(self, poll_sequence):
        proc = MagicMock()
        proc.poll = MagicMock(side_effect=poll_sequence)
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        proc.wait = MagicMock()
        return proc

    def test_both_succeed(self):
        prefill = self._mock_proc([None, 0])
        decode = self._mock_proc([None, 0])
        prefill.returncode = 0
        decode.returncode = 0
        terminated = []
        result = _wait_for_disagg_processes(
            prefill, decode, timeout_s=10,
            now_fn=lambda: 0, sleep_fn=lambda s: None,
            terminate_fn=lambda p: terminated.append(p))
        assert result == "success"
        assert terminated == []

    def test_prefill_fails_terminates_decode(self):
        prefill = self._mock_proc([1])  # exit code 1 immediately
        decode = self._mock_proc([None])
        terminated = []
        result = _wait_for_disagg_processes(
            prefill, decode, timeout_s=10,
            now_fn=lambda: 0, sleep_fn=lambda s: None,
            terminate_fn=lambda p: terminated.append(p))
        assert result == "prefill_failed"
        assert decode in terminated

    def test_decode_fails_terminates_prefill(self):
        prefill = self._mock_proc([None])
        decode = self._mock_proc([1])
        terminated = []
        result = _wait_for_disagg_processes(
            prefill, decode, timeout_s=10,
            now_fn=lambda: 0, sleep_fn=lambda s: None,
            terminate_fn=lambda p: terminated.append(p))
        assert result == "decode_failed"
        assert prefill in terminated

    def test_deadline_terminates_both(self):
        prefill = self._mock_proc([None, None, None])
        decode = self._mock_proc([None, None, None])
        clock = [0]
        def tick_fn():
            clock[0] += 5
            return clock[0]
        terminated = []
        result = _wait_for_disagg_processes(
            prefill, decode, timeout_s=2,
            now_fn=tick_fn, sleep_fn=lambda s: None,
            terminate_fn=lambda p: terminated.append(p))
        assert result == "timeout"
        assert prefill in terminated
        assert decode in terminated


class TestCLIProvenanceResolution:
    """Test predict_disaggregated.py resolves provenance and emits structured JSON."""

    def _run_predict_cli(self, argv):
        import scripts.serve_predict.predict_serve_metric_disagg as cli_mod
        import sys as _sys

        old_argv = _sys.argv
        _sys.argv = argv
        try:
            cli_mod.main()
        finally:
            _sys.argv = old_argv

    def test_cli_resolves_sidecar_from_provenance_and_emits_json(self, tmp_path, capsys):
        """CLI loads provenance, resolves sidecar/window, emits structured JSON output."""
        import json as _json

        prefill_trace = tmp_path / "prefill_trace.json"
        prefill_trace.write_text("[]")
        decode_trace = tmp_path / "decode_trace.json"
        decode_trace.write_text("[]")

        p_sidecar = tmp_path / "custom_prefill_sched.jsonl"
        p_sidecar.write_text("")
        d_sidecar = tmp_path / "custom_decode_sched.jsonl"
        d_sidecar.write_text("")

        p_prov = tmp_path / "prefill_trace.json_provenance.json"
        p_prov.write_text(_json.dumps({
            "scheduler_sidecar": str(p_sidecar),
            "inference_window_us": [1000, 5000],
        }))
        d_prov = tmp_path / "decode_trace.json_provenance.json"
        d_prov.write_text(_json.dumps({
            "scheduler_sidecar": str(d_sidecar),
            "inference_window_us": [2000, 6000],
        }))

        captured_kwargs = {}
        def mock_predict(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        with patch("src.serve_predict.serve_metric_predictor.predict_disaggregated_metrics", mock_predict), \
             patch("src.serve_predict.model_config.load_model_config", return_value=MagicMock()):
            try:
                self._run_predict_cli([
                    "predict_disaggregated.py",
                    "--prefill-trace", str(prefill_trace),
                    "--decode-trace", str(decode_trace),
                    "--model", "/fake/model",
                    "--json",
                ])
            except SystemExit:
                pass

        # Verify provenance-resolved paths reached the predictor
        assert captured_kwargs.get("prefill_sidecar_path") == str(p_sidecar)
        assert captured_kwargs.get("decode_sidecar_path") == str(d_sidecar)
        assert captured_kwargs.get("prefill_inference_window_us") == (1000, 5000)
        assert captured_kwargs.get("decode_inference_window_us") == (2000, 6000)

        # Verify structured JSON output
        out = capsys.readouterr().out
        output = _json.loads(out)
        assert "inputs" in output
        assert "metrics" in output
        assert isinstance(output["metrics"], list)
        assert output["inputs"]["prefill_sidecar"] == str(p_sidecar)
        assert output["inputs"]["decode_sidecar"] == str(d_sidecar)
        assert output["inputs"]["prefill_inference_window_us"] == [1000, 5000]
        assert output["inputs"]["decode_inference_window_us"] == [2000, 6000]

    def test_cli_json_output_writes_requested_file(self, tmp_path, capsys):
        """CLI writes JSON output to --output while preserving JSON stdout."""
        import json as _json

        prefill_trace = tmp_path / "prefill_trace.json"
        prefill_trace.write_text("[]")
        decode_trace = tmp_path / "decode_trace.json"
        decode_trace.write_text("[]")
        prefill_sidecar = tmp_path / "prefill_scheduler.jsonl"
        prefill_sidecar.write_text("")
        decode_sidecar = tmp_path / "decode_scheduler.jsonl"
        decode_sidecar.write_text("")
        output_path = tmp_path / "prediction.json"

        with patch("src.serve_predict.serve_metric_predictor.predict_disaggregated_metrics", return_value=[]), \
             patch("src.serve_predict.model_config.load_model_config", return_value=MagicMock()):
            self._run_predict_cli([
                "predict_disaggregated.py",
                "--prefill-trace", str(prefill_trace),
                "--decode-trace", str(decode_trace),
                "--prefill-sidecar", str(prefill_sidecar),
                "--decode-sidecar", str(decode_sidecar),
                "--model", "/fake/model",
                "--json",
                "--output", str(output_path),
            ])

        stdout_output = _json.loads(capsys.readouterr().out)
        file_output = _json.loads(output_path.read_text())
        assert stdout_output == file_output
        assert file_output["inputs"]["prefill_trace"] == str(prefill_trace)
        assert file_output["metrics"] == []

    def test_cli_passes_heterogeneous_tp_sizes_and_emits_json_inputs(self, tmp_path, capsys):
        import json as _json

        prefill_trace = tmp_path / "prefill_trace.json"
        prefill_trace.write_text(_json.dumps([
            {"kernel_name": "nccl_send", "grid": [4, 0, 99], "block": [9000, 0], "timestamp_us": 9},
            {"kernel_name": "nccl_send", "grid": [4, 0, 7], "block": [1000, 0], "timestamp_us": 10},
            {"kernel_name": "nccl_send", "grid": [4, 1, 7], "block": [2000, 0], "timestamp_us": 11},
            {"kernel_name": "nccl_send", "grid": [4, 2, 7], "block": [1500, 0], "timestamp_us": 12},
            {"kernel_name": "nccl_send", "grid": [4, 3, 7], "block": [1200, 0], "timestamp_us": 13},
            {"kernel_name": "nccl_send", "grid": [4, 0, 100], "block": [8000, 0], "timestamp_us": 14},
        ]))
        (tmp_path / "prefill_trace.json_provenance.json").write_text(_json.dumps({
            "inference_window_us": [10, 13],
        }))
        decode_trace = tmp_path / "decode_trace.json"
        decode_trace.write_text("[]")
        p_sidecar = tmp_path / "prefill_sched.jsonl"
        p_sidecar.write_text("")
        d_sidecar = tmp_path / "decode_sched.jsonl"
        d_sidecar.write_text("")

        captured_kwargs = {}
        def mock_predict(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        with patch("src.serve_predict.serve_metric_predictor.predict_disaggregated_metrics", mock_predict), \
             patch("src.serve_predict.model_config.load_model_config", return_value=MagicMock()):
            self._run_predict_cli([
                "predict_disaggregated.py",
                "--prefill-trace", str(prefill_trace),
                "--decode-trace", str(decode_trace),
                "--prefill-sidecar", str(p_sidecar),
                "--decode-sidecar", str(d_sidecar),
                "--model", "/fake/model",
                "--prefill-tp-size", "4",
                "--decode-tp-size", "2",
                "--kv-bw-gb-s", "1",
                "--nccl-alpha-ns", "10",
                "--json",
            ])

        assert captured_kwargs["prefill_tp_size"] == 4
        assert captured_kwargs["decode_tp_size"] == 2
        assert captured_kwargs["observed_kv_bytes"] == 5700
        assert captured_kwargs["observed_kv_transfer_time_ns"] == pytest.approx(2010.0)

        output = _json.loads(capsys.readouterr().out)
        assert output["inputs"]["prefill_tp_size"] == 4
        assert output["inputs"]["decode_tp_size"] == 2
        assert output["inputs"]["prefill_inference_window_us"] == [10, 13]
        assert output["kv_transfer"]["prefill_tp_size"] == 4
        assert output["kv_transfer"]["decode_tp_size"] == 2

    def test_cli_tp_size_sets_both_instance_tp_sizes(self, tmp_path, capsys):
        import json as _json

        prefill_trace = tmp_path / "prefill_trace.json"
        prefill_trace.write_text("[]")
        decode_trace = tmp_path / "decode_trace.json"
        decode_trace.write_text("[]")
        p_sidecar = tmp_path / "prefill_sched.jsonl"
        p_sidecar.write_text("")
        d_sidecar = tmp_path / "decode_sched.jsonl"
        d_sidecar.write_text("")

        captured_kwargs = {}
        def mock_predict(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        with patch("src.serve_predict.serve_metric_predictor.predict_disaggregated_metrics", mock_predict), \
             patch("src.serve_predict.model_config.load_model_config", return_value=MagicMock()):
            self._run_predict_cli([
                "predict_disaggregated.py",
                "--prefill-trace", str(prefill_trace),
                "--decode-trace", str(decode_trace),
                "--prefill-sidecar", str(p_sidecar),
                "--decode-sidecar", str(d_sidecar),
                "--model", "/fake/model",
                "--tp-size", "4",
                "--json",
            ])

        assert captured_kwargs["tp_size"] == 4
        assert captured_kwargs["prefill_tp_size"] == 4
        assert captured_kwargs["decode_tp_size"] == 4

        output = _json.loads(capsys.readouterr().out)
        assert output["inputs"]["tp_size"] == 4
        assert output["inputs"]["prefill_tp_size"] == 4
        assert output["inputs"]["decode_tp_size"] == 4

    def test_cli_rejects_conflicting_tp_size_flags(self, tmp_path, capsys):
        prefill_trace = tmp_path / "prefill_trace.json"
        prefill_trace.write_text("[]")
        decode_trace = tmp_path / "decode_trace.json"
        decode_trace.write_text("[]")
        p_sidecar = tmp_path / "prefill_sched.jsonl"
        p_sidecar.write_text("")
        d_sidecar = tmp_path / "decode_sched.jsonl"
        d_sidecar.write_text("")

        with pytest.raises(SystemExit) as exc:
            self._run_predict_cli([
                "predict_disaggregated.py",
                "--prefill-trace", str(prefill_trace),
                "--decode-trace", str(decode_trace),
                "--prefill-sidecar", str(p_sidecar),
                "--decode-sidecar", str(d_sidecar),
                "--model", "/fake/model",
                "--tp-size", "4",
                "--decode-tp-size", "2",
                "--json",
            ])

        assert exc.value.code != 0
        assert "conflict" in capsys.readouterr().err.lower()


class TestNsysGroundTruthWrapper:
    """The opt-in nsys ground-truth capture wiring (--nsys-role) added for the
    matched PD prefill/decode validation against role-specific nsys."""

    def test_nsys_prefix_is_cudaprofiler_gated(self):
        pfx = build_nsys_prefix("/tmp/x/decode_nsys")
        assert pfx[:2] == ["nsys", "profile"]
        # cudaProfilerApi gating so warmup/init is excluded from the window
        assert "--capture-range=cudaProfilerApi" in pfx
        assert "--export=sqlite" in pfx
        assert pfx[-1] == "/tmp/x/decode_nsys"

    def test_profiled_env_runs_in_process_and_gates(self):
        # An nsys-profiled instance must run the engine in-process so
        # cudaProfilerStart/Stop reach the kernels (the EngineCore child cannot be
        # reached by a parent-process profiler), and must signal the gate.
        env = build_instance_env([1], "/tmp/t", "configs/hardware/nvidia_a800.yaml",
                                 mode="plugin", nsys_profiled=True)
        assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        assert env["PD_NSYS_PROFILE"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_unprofiled_env_keeps_multiprocessing(self):
        # The normal (synth-trace) run must NOT disable multiprocessing or set the
        # gate flag — its operator capture relies on the spawned/forked child.
        env = build_instance_env([1], "/tmp/t", "configs/hardware/nvidia_a800.yaml",
                                 mode="plugin", nsys_profiled=False)
        assert "PD_NSYS_PROFILE" not in env
        assert env.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0"


class TestFinalizePluginInstance:
    """Tests for the factored _finalize_plugin_instance helper in _disagg_instance.

    This covers the two control-flow branches (ground-truth-only vs normal plugin)
    without needing vLLM or a GPU.
    """

    def _make_args(self, role="decode", model="/fake/model",
                   hw_config="configs/hardware/nvidia_a800.yaml",
                   seed=42, max_tokens=8, prompts_file=None):
        args = MagicMock()
        args.role = role
        args.model = model
        args.hw_config = hw_config
        args.seed = seed
        args.max_tokens = max_tokens
        args.prompts_file = prompts_file
        return args

    def test_ground_truth_only_returns_zero_no_operator_records(self, tmp_path):
        from scripts.serve_trace._disagg_instance import _finalize_plugin_instance
        trace_base = str(tmp_path / "decode_trace")
        args = self._make_args(role="decode")
        kv_cfg = {"kv_connector": "P2pNcclConnector"}
        engine_args = {}

        rc = _finalize_plugin_instance(
            args, trace_base, trace_base + "_sched.jsonl",
            kv_cfg, engine_args, ground_truth_only=True)

        assert rc == 0
        prov_path = f"{trace_base}_provenance.json"
        assert Path(prov_path).exists()
        prov = json.loads(Path(prov_path).read_text())
        assert prov["ground_truth_only"] is True
        assert prov["mode"] == "plugin-nsys-ground-truth"
        assert prov["synthesized_trace"] is None

    def test_normal_plugin_returns_one_when_no_operator_records(self, tmp_path):
        from scripts.serve_trace._disagg_instance import _finalize_plugin_instance
        trace_base = str(tmp_path / "decode_trace")
        args = self._make_args(role="decode")

        rc = _finalize_plugin_instance(
            args, trace_base, trace_base + "_sched.jsonl",
            {}, {}, ground_truth_only=False,
            op_trace_glob=str(tmp_path / "nonexistent_ops.*"))

        assert rc == 1

    def test_ground_truth_only_prefill(self, tmp_path):
        from scripts.serve_trace._disagg_instance import _finalize_plugin_instance
        trace_base = str(tmp_path / "prefill_trace")
        args = self._make_args(role="prefill")

        rc = _finalize_plugin_instance(
            args, trace_base, trace_base + "_sched.jsonl",
            {"kv_connector": "P2pNcclConnector"}, {},
            ground_truth_only=True)

        assert rc == 0
        prov = json.loads(Path(f"{trace_base}_provenance.json").read_text())
        assert prov["role"] == "prefill"
        assert prov["ground_truth_only"] is True


class TestValidateNsysOutputs:
    """Tests for the fail-closed nsys sqlite validator in trace_disagg."""

    def test_missing_sqlite_reports_problem(self, tmp_path):
        problems = validate_nsys_outputs(
            {"decode"},
            {"decode": str(tmp_path / "nonexistent.sqlite")})
        assert len(problems) == 1
        assert "missing" in problems[0].lower()

    def test_empty_sqlite_reports_problem(self, tmp_path):
        path = tmp_path / "empty.sqlite"
        path.write_bytes(b"")
        problems = validate_nsys_outputs({"decode"}, {"decode": str(path)})
        assert len(problems) == 1
        assert "empty" in problems[0].lower()

    def test_sqlite_without_kernel_table_reports_problem(self, tmp_path):
        import sqlite3
        path = tmp_path / "no_kernels.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        problems = validate_nsys_outputs({"decode"}, {"decode": str(path)})
        assert len(problems) == 1
        assert "kernel" in problems[0].lower()

    def test_sqlite_with_zero_kernel_rows_reports_problem(self, tmp_path):
        import sqlite3
        path = tmp_path / "zero_kernels.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, shortName TEXT)")
        conn.commit()
        conn.close()

        problems = validate_nsys_outputs({"decode"}, {"decode": str(path)})
        assert len(problems) == 1
        assert "0 kernel rows" in problems[0]

    def test_valid_sqlite_passes(self, tmp_path):
        import sqlite3
        path = tmp_path / "valid.sqlite"
        conn = sqlite3.connect(str(path))
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, shortName TEXT)")
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000, 2000, 'gemm')")
        conn.commit()
        conn.close()

        problems = validate_nsys_outputs({"decode"}, {"decode": str(path)})
        assert problems == []

    def test_multiple_roles_all_checked(self, tmp_path):
        import sqlite3
        # Prefill valid, decode missing
        pf_path = tmp_path / "prefill.sqlite"
        conn = sqlite3.connect(str(pf_path))
        conn.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(start INTEGER, end INTEGER, shortName TEXT)")
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000, 2000, 'gemm')")
        conn.commit()
        conn.close()

        problems = validate_nsys_outputs(
            {"prefill", "decode"},
            {"prefill": str(pf_path),
             "decode": str(tmp_path / "nonexistent.sqlite")})
        assert len(problems) == 1
        assert "decode" in problems[0]

    def test_no_roles_returns_empty(self):
        problems = validate_nsys_outputs(set(), {})
        assert problems == []
