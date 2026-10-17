"""Tests for src/serve_predict/serve_metric_predictor.py"""
import json
from unittest.mock import patch, MagicMock

import pytest

from src.serve_predict.serve_metric_predictor import (
    ServingPredictor,
    PerRequestMetrics,
    SchedulerStep,
    load_scheduler_sidecar,
    _segment_by_iteration,
    _segment_by_scheduler_steps,
    _infer_iteration_phase,
)
from src.serve_predict.predict_kernel_time_from_trace import KernelAnnotation


def _make_kernel(name, ts_us, grid=(1, 1, 1), block=(256, 1, 1), shared_mem=0):
    return {
        "kernel_name": name,
        "timestamp_us": ts_us,
        "grid": list(grid),
        "block": list(block),
        "shared_mem": shared_mem,
    }


class TestSchedulerSidecar:
    def test_load_sidecar(self, tmp_path):
        sidecar = tmp_path / "sched.jsonl"
        records = [
            {"iter_id": 0, "ts_us": 1000000, "scheduled": {"req-0": 512}, "total_tokens": 512, "finished": []},
            {"iter_id": 1, "ts_us": 1001000, "scheduled": {"req-0": 1, "req-1": 1}, "total_tokens": 2, "finished": []},
            {"iter_id": 2, "ts_us": 1002000, "scheduled": {"req-0": 1, "req-1": 1}, "total_tokens": 2, "finished": ["req-0"]},
        ]
        sidecar.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        steps = load_scheduler_sidecar(str(sidecar))
        assert len(steps) == 3
        assert steps[0].iter_id == 0
        assert steps[0].scheduled == {"req-0": 512}
        assert steps[2].finished == ["req-0"]

    def test_empty_sidecar_raises(self, tmp_path):
        sidecar = tmp_path / "empty.jsonl"
        sidecar.write_text("")
        steps = load_scheduler_sidecar(str(sidecar))
        assert steps == []


class TestSegmentation:
    def test_basic_segmentation(self):
        entries = [
            _make_kernel("k1", 1000),
            _make_kernel("k2", 1010),
            _make_kernel("k3", 3000),  # gap > 1000us
            _make_kernel("k4", 3010),
        ]
        iters = _segment_by_iteration(entries, gap_threshold_us=1000.0)
        assert len(iters) == 2
        assert len(iters[0]) == 2
        assert len(iters[1]) == 2

    def test_empty_trace(self):
        assert _segment_by_iteration([]) == []

    def test_scheduler_step_segmentation_requires_each_step_covered(self):
        entries = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("flash_fwd_kernel", 1100),
            _make_kernel("flash_fwd_kernel", 1200),
        ]
        steps = [
            SchedulerStep(iter_id=0, ts_us=1000, scheduled={"r0": 128}, total_tokens=128, finished=[]),
            SchedulerStep(iter_id=1, ts_us=1100, scheduled={"r0": 1}, total_tokens=1, finished=[]),
            SchedulerStep(iter_id=2, ts_us=1200, scheduled={"r0": 1}, total_tokens=1, finished=[]),
        ]

        segmented = _segment_by_scheduler_steps(entries, steps)
        assert [len(it) for it in segmented] == [1, 1, 1]

        missing = _segment_by_scheduler_steps(entries[:2], steps)
        assert missing is None


class TestIterationPhase:
    def test_prefill(self):
        entries = [_make_kernel("flash_fwd_kernel", 100)]
        assert _infer_iteration_phase(entries) == "prefill"

    def test_decode(self):
        entries = [_make_kernel("internal_gemvx_kernel", 100)]
        assert _infer_iteration_phase(entries) == "decode"

    def test_mixed(self):
        entries = [
            _make_kernel("flash_fwd_kernel", 100),
            _make_kernel("gemv2T_kernel", 200),
        ]
        assert _infer_iteration_phase(entries) == "mixed"


def _stub_annotate(predicted_ns=1000000.0, cpu_overhead_ns=0.0):
    """Return a mock annotate_trace that returns fixed-latency annotations."""
    def _mock_annotate(entries, seq_len, batch_size, tp_size):
        return [KernelAnnotation(
            index=0,
            kernel_name="stub",
            grid=[1, 1, 1],
            block=[256, 1, 1],
            shared_mem=0,
            category="GEMM",
            predicted_ns=predicted_ns,
            cpu_overhead_ns=cpu_overhead_ns,
        )]
    return _mock_annotate


class TestPredictorContract:
    """Tests for the actual predict_per_request_metrics() contract."""

    def _setup_files(self, tmp_path, trace_entries, sidecar_records):
        """Write trace JSON and sidecar JSONL to tmp_path."""
        trace_path = str(tmp_path / "trace.json")
        sidecar_path = str(tmp_path / "trace.json_scheduler.jsonl")
        with open(trace_path, "w") as f:
            json.dump(trace_entries, f)
        with open(sidecar_path, "w") as f:
            for rec in sidecar_records:
                f.write(json.dumps(rec) + "\n")
        return trace_path, sidecar_path

    def _make_predictor(self, annotate_fn):
        """Create a ServingPredictor with stubbed TracePredictor."""
        sp = object.__new__(ServingPredictor)
        sp.tp_size = 1
        sp.gap_threshold_us = 1000.0
        sp.predictor = MagicMock()
        sp.predictor.annotate_trace = annotate_fn
        return sp

    def test_single_request_ttft_includes_chunked_prefill_gap(self, tmp_path):
        """Single request TTFT includes chunked prefill iteration times and gaps."""
        # 2 prefill iterations (chunked), each predicted at 5ms, plus a 1.99ms gap.
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("s16816gemm_kernel", 1010),
            # gap
            _make_kernel("flash_fwd_kernel", 3000),
            _make_kernel("s16816gemm_kernel", 3010),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 256}, "total_tokens": 256, "finished": []},
            {"iter_id": 1, "ts_us": 3000, "scheduled": {"req-0": 256}, "total_tokens": 256, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=5_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 1
        m = metrics[0]
        assert m.request_id == "req-0"
        assert m.ttft_ms == pytest.approx(11.99, abs=0.001)
        assert m.num_prefill_tokens == 512
        assert m.prefill_iterations == 2

    def test_multi_request_decode_full_iteration_time(self, tmp_path):
        """Each request in a shared decode iteration gets the full iteration and its gap."""
        # Iter 0: all 3 requests prefill first
        # Iter 1: all 3 requests decode (1 token each)
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            # gap
            _make_kernel("internal_gemvx_kernel", 5000),
            _make_kernel("s16816gemm_kernel", 5010),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"r0": 100, "r1": 100, "r2": 100}, "total_tokens": 300, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"r0": 1, "r1": 1, "r2": 1}, "total_tokens": 3, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=2_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 3
        for m in metrics:
            # Each request waits through the 4ms scheduler gap and full 2ms decode iteration.
            assert m.tpot_ms == pytest.approx(6.0, abs=0.001)
            assert m.decode_iterations == 1

    def test_prefill_plus_decode_e2e(self, tmp_path):
        """e2e = prefill time + decode time, including the gap before decode."""
        trace = [
            # Prefill iteration: 3ms
            _make_kernel("flash_fwd_kernel", 1000),
            # gap
            # Decode iteration: 1ms
            _make_kernel("internal_gemvx_kernel", 5000),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 100}, "total_tokens": 100, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        call_count = [0]
        def _alternating_annotate(entries, seq_len, batch_size, tp_size):
            ns = 3_000_000.0 if call_count[0] == 0 else 1_000_000.0
            call_count[0] += 1
            return [KernelAnnotation(
                index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
                shared_mem=0, category="GEMM", predicted_ns=ns, cpu_overhead_ns=0.0,
            )]

        sp = self._make_predictor(_alternating_annotate)
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 1
        m = metrics[0]
        assert m.ttft_ms == pytest.approx(3.0, abs=0.001)
        assert m.tpot_ms == pytest.approx(5.0, abs=0.001)
        assert m.e2e_ms == pytest.approx(8.0, abs=0.001)
        assert m.start_time_ms == pytest.approx(0.0, abs=0.001)
        assert m.end_time_ms == pytest.approx(8.0, abs=0.001)

    def test_metrics_include_relative_timing_window_for_throughput(self, tmp_path):
        """Per-request windows expose enough timing evidence for comparison throughput."""
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("internal_gemvx_kernel", 5000),
            _make_kernel("s16816gemm_kernel", 5010),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"r0": 100, "r1": 100}, "total_tokens": 200, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"r0": 1, "r1": 1}, "total_tokens": 2, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=2_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 2
        for m in metrics:
            assert m.start_time_ms == pytest.approx(0.0, abs=0.001)
            assert m.end_time_ms == pytest.approx(8.0, abs=0.001)

    def test_predict_serving_metrics_json_serializes_timing_window(
        self, tmp_path, monkeypatch, capsys
    ):
        import scripts.serve_predict.predict_serve_metric as cli

        trace = tmp_path / "trace.json"
        sidecar = tmp_path / "trace.json_scheduler.jsonl"
        trace.write_text("[]")
        sidecar.write_text("{}\n")

        class FakePredictor:
            def __init__(self, **kwargs):
                pass

            def predict_per_request_metrics(self, **kwargs):
                return [
                    PerRequestMetrics(
                        request_id="req-0",
                        ttft_ms=3.0,
                        tpot_ms=5.0,
                        e2e_ms=8.0,
                        start_time_ms=0.0,
                        end_time_ms=4.0,
                        num_prefill_tokens=100,
                        num_decode_tokens=1,
                        prefill_iterations=1,
                        decode_iterations=1,
                    )
                ]

        monkeypatch.setattr("src.serve_predict.serve_metric_predictor.ServingPredictor", FakePredictor)
        monkeypatch.setattr("src.serve_predict.predict_kernel_time_from_trace.load_model_config", lambda model: object())

        cli.main(["--trace", str(trace), "--scheduler", str(sidecar), "--json"])

        output = json.loads(capsys.readouterr().out)
        assert output[0]["start_time_ms"] == 0.0
        assert output[0]["end_time_ms"] == 4.0

    def test_iteration_count_mismatch_raises(self, tmp_path):
        """Fewer trace iterations than sidecar steps raises ValueError."""
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
        ]
        # 2 sidecar steps for 1 trace iteration (under-supply)
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 100}, "total_tokens": 100, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate())
        with pytest.raises(ValueError, match="Iteration count mismatch"):
            sp.predict_per_request_metrics(trace_path, sidecar_path)

    def test_sidecar_timestamp_fallback_splits_coarse_trace_segments(self, tmp_path):
        """Scheduler timestamps split continuous serving steps merged by gap segmentation."""
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("flash_fwd_kernel", 1100),
            _make_kernel("flash_fwd_kernel", 1200),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 128}, "total_tokens": 128, "finished": []},
            {"iter_id": 1, "ts_us": 1100, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
            {"iter_id": 2, "ts_us": 1200, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": ["req-0"]},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=1_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 1
        assert metrics[0].prefill_iterations == 1
        assert metrics[0].decode_iterations == 2
        assert metrics[0].num_decode_tokens == 2

    def test_empty_sidecar_raises(self, tmp_path):
        """Empty sidecar after filtering raises ValueError."""
        trace = [_make_kernel("flash_fwd_kernel", 1000)]
        sidecar = []
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate())
        with pytest.raises(ValueError, match="Empty scheduler sidecar"):
            sp.predict_per_request_metrics(trace_path, sidecar_path)

    def test_zero_token_steps_filtered_before_alignment(self, tmp_path):
        """Zero-token sidecar steps are filtered and don't cause mismatch."""
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 500, "scheduled": {}, "total_tokens": 0, "finished": []},
            {"iter_id": 1, "ts_us": 1000, "scheduled": {"req-0": 100}, "total_tokens": 100, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=1_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)
        assert len(metrics) == 1
        assert metrics[0].ttft_ms == pytest.approx(1.0, abs=0.001)

    def test_mixed_iteration_per_request_classification(self, tmp_path):
        """In a mixed iteration, old_req (already past first step, 1 tok) = decode,
        new_req (first appearance, N tok) = prefill."""
        trace = [
            # Iteration 0: old_req prefills (first step)
            _make_kernel("flash_fwd_kernel", 1000),
            # gap
            # Iteration 1: mixed - old_req decodes 1 token, new_req prefills 256
            _make_kernel("flash_fwd_kernel", 5000),
            _make_kernel("internal_gemvx_kernel", 5010),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"old_req": 100}, "total_tokens": 100, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"old_req": 1, "new_req": 256},
             "total_tokens": 257, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=4_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        metrics_by_id = {m.request_id: m for m in metrics}
        # old_req: iter0=prefill(100tok), iter1=decode(1tok)
        assert metrics_by_id["old_req"].num_prefill_tokens == 100
        assert metrics_by_id["old_req"].num_decode_tokens == 1
        assert metrics_by_id["old_req"].decode_time_ms == pytest.approx(8.0, abs=0.001)
        # new_req: iter1=prefill(256tok, first step)
        assert metrics_by_id["new_req"].num_prefill_tokens == 256
        assert metrics_by_id["new_req"].num_decode_tokens == 0
        assert metrics_by_id["new_req"].ttft_ms == pytest.approx(4.0, abs=0.001)

    def test_single_token_prompt_first_step_is_prefill(self, tmp_path):
        """A request's first scheduled step is always prefill, even with tok_count=1."""
        trace = [
            _make_kernel("internal_gemvx_kernel", 1000),
            # gap
            _make_kernel("internal_gemvx_kernel", 5000),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=2_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 1
        m = metrics[0]
        # First step: prefill (even though tok_count=1)
        assert m.num_prefill_tokens == 1
        assert m.prefill_time_ms == pytest.approx(2.0, abs=0.001)
        # Second step: decode
        assert m.num_decode_tokens == 1
        assert m.decode_time_ms == pytest.approx(6.0, abs=0.001)
        assert m.ttft_ms == pytest.approx(2.0, abs=0.001)

    def test_component_fields_populated(self, tmp_path):
        """PerRequestMetrics exposes prefill_time_ms and decode_time_ms."""
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("internal_gemvx_kernel", 5000),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 100}, "total_tokens": 100, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        call_count = [0]
        def _alt(entries, seq_len, batch_size, tp_size):
            ns = 3_000_000.0 if call_count[0] == 0 else 1_000_000.0
            call_count[0] += 1
            return [KernelAnnotation(
                index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
                shared_mem=0, category="GEMM", predicted_ns=ns, cpu_overhead_ns=0.0,
            )]

        sp = self._make_predictor(_alt)
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)
        m = metrics[0]
        assert m.prefill_time_ms == pytest.approx(3.0, abs=0.001)
        assert m.decode_time_ms == pytest.approx(5.0, abs=0.001)

    def test_prefill_iterations_include_inter_iteration_gap(self, tmp_path):
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("s16816gemm_kernel", 1010),
            _make_kernel("flash_fwd_kernel", 3010),
            _make_kernel("s16816gemm_kernel", 3020),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 128}, "total_tokens": 128, "finished": []},
            {"iter_id": 1, "ts_us": 3010, "scheduled": {"req-0": 128}, "total_tokens": 128, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=5_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        assert len(metrics) == 1
        m = metrics[0]
        assert m.prefill_iterations == 2
        assert m.prefill_time_ms == pytest.approx(12.0, abs=0.001)
        assert m.ttft_ms == pytest.approx(12.0, abs=0.001)
        assert m.e2e_ms == pytest.approx(12.0, abs=0.001)

    def test_decode_iterations_include_repeated_scheduler_gaps(self, tmp_path):
        trace = [_make_kernel("internal_gemvx_kernel", 1000 + i * 110)
                 for i in range(100)]
        sidecar = [
            {"iter_id": i, "ts_us": 1000 + i * 110,
             "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []}
            for i in range(100)
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=1_000_000.0))
        sp.gap_threshold_us = 50.0
        metrics = sp.predict_per_request_metrics(
            trace_path, sidecar_path, first_step_is_prefill=False)

        assert len(metrics) == 1
        m = metrics[0]
        expected_decode_ms = 100.0 + 99 * 0.11
        assert m.num_decode_tokens == 100
        assert m.decode_time_ms == pytest.approx(expected_decode_ms, abs=0.001)
        assert m.tpot_ms == pytest.approx(expected_decode_ms / 100, abs=0.001)
        assert m.e2e_ms == pytest.approx(expected_decode_ms, abs=0.001)

    def test_shared_decode_iteration_charges_gap_only_to_existing_request(self, tmp_path):
        trace = [
            _make_kernel("internal_gemvx_kernel", 1000),
            _make_kernel("internal_gemvx_kernel", 3000),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"r0": 1}, "total_tokens": 1, "finished": []},
            {"iter_id": 1, "ts_us": 3000, "scheduled": {"r0": 1, "r1": 1}, "total_tokens": 2, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=1_000_000.0))
        metrics = sp.predict_per_request_metrics(
            trace_path, sidecar_path, first_step_is_prefill=False)
        by_id = {m.request_id: m for m in metrics}

        assert by_id["r0"].decode_time_ms == pytest.approx(4.0, abs=0.001)
        assert by_id["r0"].e2e_ms == pytest.approx(4.0, abs=0.001)
        assert by_id["r1"].decode_time_ms == pytest.approx(1.0, abs=0.001)
        assert by_id["r1"].e2e_ms == pytest.approx(1.0, abs=0.001)

    def test_single_step_request_does_not_receive_unrelated_gap(self, tmp_path):
        trace = [
            _make_kernel("flash_fwd_kernel", 1000),
            _make_kernel("flash_fwd_kernel", 5000),
        ]
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"r0": 64}, "total_tokens": 64, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"r1": 64}, "total_tokens": 64, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)

        sp = self._make_predictor(_stub_annotate(predicted_ns=2_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)

        by_id = {m.request_id: m for m in metrics}
        assert by_id["r0"].prefill_time_ms == pytest.approx(2.0, abs=0.001)
        assert by_id["r0"].e2e_ms == pytest.approx(2.0, abs=0.001)
        assert by_id["r1"].prefill_time_ms == pytest.approx(2.0, abs=0.001)
        assert by_id["r1"].e2e_ms == pytest.approx(2.0, abs=0.001)


class TestDisaggregatedMetrics:
    """Tests for predict_disaggregated_metrics()."""

    def _setup_instance(self, tmp_path, prefix, trace_entries, sidecar_records):
        trace_path = str(tmp_path / f"{prefix}_trace.json")
        sidecar_path = str(tmp_path / f"{prefix}_sched.jsonl")
        with open(trace_path, "w") as f:
            json.dump(trace_entries, f)
        with open(sidecar_path, "w") as f:
            for rec in sidecar_records:
                f.write(json.dumps(rec) + "\n")
        return trace_path, sidecar_path

    @patch("src.serve_predict.serve_metric_predictor.TracePredictor")
    def test_inference_window_filters_out_of_window_rows(self, MockTP, tmp_path):
        """Extra sidecar rows outside inference_window_us are filtered."""
        from src.serve_predict.serve_metric_predictor import predict_disaggregated_metrics
        from src.serving.workload.model_config import ModelConfig

        # Prefill: 2 iterations, window covers only the first
        prefill_trace = [_make_kernel("flash_fwd_kernel", 1000)]
        prefill_sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 100}, "total_tokens": 100, "finished": []},
            {"iter_id": 1, "ts_us": 9000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]
        # Decode: 1 iteration in window
        decode_trace = [_make_kernel("internal_gemvx_kernel", 2000)]
        decode_sidecar = [
            {"iter_id": 0, "ts_us": 2000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
            {"iter_id": 1, "ts_us": 9000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]

        p_trace, p_sidecar = self._setup_instance(tmp_path, "prefill", prefill_trace, prefill_sidecar)
        d_trace, d_sidecar = self._setup_instance(tmp_path, "decode", decode_trace, decode_sidecar)

        # Mock TracePredictor.annotate_trace to return fixed 1ms per iteration
        mock_instance = MagicMock()
        mock_instance.annotate_trace.return_value = [KernelAnnotation(
            index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
            shared_mem=0, category="GEMM", predicted_ns=1_000_000.0, cpu_overhead_ns=0.0,
        )]
        MockTP.return_value = mock_instance

        mc = ModelConfig(hidden_size=2048, intermediate_size=5504,
                        num_attention_heads=16, num_key_value_heads=4,
                        num_hidden_layers=24, head_dim=128, vocab_size=32000)

        metrics = predict_disaggregated_metrics(
            p_trace, p_sidecar, "configs/hardware/nvidia_a800.yaml",
            d_trace, d_sidecar, "configs/hardware/nvidia_a800.yaml",
            mc, kv_transfer_bw_bytes_s=50e9,
            prefill_inference_window_us=(500, 5000),
            decode_inference_window_us=(1500, 5000),
        )
        # With windows: prefill has 1 row (ts=1000 in [500,5000]),
        # decode has 1 row (ts=2000 in [1500,5000]).
        # Each traces has 1 kernel → 1 iteration. Alignment should be 1:1.
        assert len(metrics) == 1
        m = metrics[0]
        assert m.prefill_time_ms == pytest.approx(1.0, abs=0.01)

    @patch("src.serve_predict.serve_metric_predictor.TracePredictor")
    def test_heterogeneous_tp_sizes_reach_prefill_and_decode_predictors(self, MockTP, tmp_path):
        from src.serve_predict.serve_metric_predictor import predict_disaggregated_metrics
        from src.serving.workload.model_config import ModelConfig

        prefill_trace = [_make_kernel("flash_fwd_kernel", 1000)]
        prefill_sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 8}, "total_tokens": 8, "finished": []},
        ]
        decode_trace = [_make_kernel("internal_gemvx_kernel", 2000)]
        decode_sidecar = [
            {"iter_id": 0, "ts_us": 2000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]

        p_trace, p_sidecar = self._setup_instance(tmp_path, "prefill", prefill_trace, prefill_sidecar)
        d_trace, d_sidecar = self._setup_instance(tmp_path, "decode", decode_trace, decode_sidecar)

        prefill_tp = MagicMock()
        decode_tp = MagicMock()
        for mock_tp in (prefill_tp, decode_tp):
            mock_tp.annotate_trace.return_value = [KernelAnnotation(
                index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
                shared_mem=0, category="GEMM", predicted_ns=1_000_000.0,
                cpu_overhead_ns=0.0,
            )]
        MockTP.side_effect = [prefill_tp, decode_tp]

        mc = ModelConfig(hidden_size=2048, intermediate_size=5504,
                        num_attention_heads=16, num_key_value_heads=4,
                        num_hidden_layers=1, head_dim=1, vocab_size=32000)

        metrics = predict_disaggregated_metrics(
            p_trace, p_sidecar, "configs/hardware/nvidia_a800.yaml",
            d_trace, d_sidecar, "configs/hardware/nvidia_a800.yaml",
            mc, kv_transfer_bw_bytes_s=1e9, nccl_alpha_ns=0.0,
            prefill_tp_size=4, decode_tp_size=2)

        assert prefill_tp.annotate_trace.call_args.kwargs["tp_size"] == 4
        assert decode_tp.annotate_trace.call_args.kwargs["tp_size"] == 2
        assert len(metrics) == 1
        expected_kv_bytes = 1 * 2 * (4 // 4) * 1 * 8 * 2
        assert metrics[0].kv_transfer_time_ms == pytest.approx(
            expected_kv_bytes / 1e6)

    @patch("src.serve_predict.serve_metric_predictor.TracePredictor")
    def test_rejects_prefill_decode_request_mismatch(self, MockTP, tmp_path):
        from src.serve_predict.serve_metric_predictor import predict_disaggregated_metrics
        from src.serving.workload.model_config import ModelConfig

        prefill_trace = [_make_kernel("flash_fwd_kernel", 1000)]
        prefill_sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-prefill": 8}, "total_tokens": 8, "finished": []},
        ]
        decode_trace = [_make_kernel("internal_gemvx_kernel", 2000)]
        decode_sidecar = [
            {"iter_id": 0, "ts_us": 2000, "scheduled": {"req-decode": 1}, "total_tokens": 1, "finished": []},
        ]

        p_trace, p_sidecar = self._setup_instance(tmp_path, "prefill", prefill_trace, prefill_sidecar)
        d_trace, d_sidecar = self._setup_instance(tmp_path, "decode", decode_trace, decode_sidecar)

        mock_instance = MagicMock()
        mock_instance.annotate_trace.return_value = [KernelAnnotation(
            index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
            shared_mem=0, category="GEMM", predicted_ns=1_000_000.0,
            cpu_overhead_ns=0.0,
        )]
        MockTP.return_value = mock_instance

        mc = ModelConfig(hidden_size=2048, intermediate_size=5504,
                        num_attention_heads=16, num_key_value_heads=4,
                        num_hidden_layers=1, head_dim=1, vocab_size=32000)

        with pytest.raises(ValueError, match="missing decode: req-prefill.*missing prefill: req-decode"):
            predict_disaggregated_metrics(
                p_trace, p_sidecar, "configs/hardware/nvidia_a800.yaml",
                d_trace, d_sidecar, "configs/hardware/nvidia_a800.yaml",
                mc, kv_transfer_bw_bytes_s=1e9)

    @patch("src.serve_predict.serve_metric_predictor.TracePredictor")
    def test_observed_kv_latency_uses_grouped_tp_send_time(self, MockTP, tmp_path):
        from src.serve_predict.serve_metric_predictor import predict_disaggregated_metrics
        from src.serving.workload.model_config import ModelConfig

        prefill_trace = [_make_kernel("flash_fwd_kernel", 1000)]
        prefill_sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"req-0": 8}, "total_tokens": 8, "finished": []},
        ]
        decode_trace = [_make_kernel("internal_gemvx_kernel", 2000)]
        decode_sidecar = [
            {"iter_id": 0, "ts_us": 2000, "scheduled": {"req-0": 1}, "total_tokens": 1, "finished": []},
        ]

        p_trace, p_sidecar = self._setup_instance(tmp_path, "prefill", prefill_trace, prefill_sidecar)
        d_trace, d_sidecar = self._setup_instance(tmp_path, "decode", decode_trace, decode_sidecar)

        mock_instance = MagicMock()
        mock_instance.annotate_trace.return_value = [KernelAnnotation(
            index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
            shared_mem=0, category="GEMM", predicted_ns=1_000_000.0,
            cpu_overhead_ns=0.0,
        )]
        MockTP.return_value = mock_instance

        mc = ModelConfig(hidden_size=2048, intermediate_size=5504,
                        num_attention_heads=16, num_key_value_heads=4,
                        num_hidden_layers=1, head_dim=1, vocab_size=32000)

        metrics = predict_disaggregated_metrics(
            p_trace, p_sidecar, "configs/hardware/nvidia_a800.yaml",
            d_trace, d_sidecar, "configs/hardware/nvidia_a800.yaml",
            mc, kv_transfer_bw_bytes_s=1e9, nccl_alpha_ns=10.0,
            prefill_tp_size=4, decode_tp_size=2,
            observed_kv_bytes=5700,
            observed_kv_transfer_time_ns=2010.0)

        assert len(metrics) == 1
        assert metrics[0].kv_transfer_time_ms == pytest.approx(2010.0 / 1e6)


class TestAlignmentRegression:
    """Tests for trace/sidecar alignment with warmup iterations."""

    def _setup_files(self, tmp_path, trace_entries, sidecar_records):
        trace_path = str(tmp_path / "trace.json")
        sidecar_path = str(tmp_path / "trace.json_scheduler.jsonl")
        with open(trace_path, "w") as f:
            json.dump(trace_entries, f)
        with open(sidecar_path, "w") as f:
            for rec in sidecar_records:
                f.write(json.dumps(rec) + "\n")
        return trace_path, sidecar_path

    def _make_predictor(self, annotate_fn):
        sp = object.__new__(ServingPredictor)
        sp.tp_size = 1
        sp.gap_threshold_us = 1000.0
        sp.predictor = MagicMock()
        sp.predictor.annotate_trace = annotate_fn
        return sp

    def test_warmup_burst_dropped_prefill_kept(self, tmp_path):
        """2-kernel warmup, then 50-kernel prefill, then 30-kernel decode.
        2 active sidecar steps → keeps prefill (50) + decode (30), drops warmup (2)."""
        trace = (
            [_make_kernel("unknown_kernel", 100), _make_kernel("unknown_kernel", 110)] +
            # gap
            [_make_kernel("flash_fwd_kernel", 5000 + i) for i in range(50)] +
            # gap
            [_make_kernel("internal_gemvx_kernel", 10000 + i) for i in range(30)]
        )
        sidecar = [
            {"iter_id": 0, "ts_us": 5000, "scheduled": {"r0": 64}, "total_tokens": 64, "finished": []},
            {"iter_id": 1, "ts_us": 10000, "scheduled": {"r0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)
        sp = self._make_predictor(_stub_annotate(predicted_ns=5_000_000.0))
        metrics = sp.predict_per_request_metrics(trace_path, sidecar_path)
        assert len(metrics) == 1
        # Prefill iteration had 50 kernels; decode had 30. Both kept.
        assert metrics[0].num_prefill_tokens == 64
        assert metrics[0].prefill_time_ms == pytest.approx(5.0, abs=0.01)

    def test_extra_large_known_phase_raises(self, tmp_path):
        """Extra large known-phase iterations raise ValueError (cannot safely align)."""
        trace = (
            [_make_kernel("flash_fwd_kernel", 1000 + i) for i in range(50)] +
            [_make_kernel("flash_fwd_kernel", 5000 + i) for i in range(50)] +
            [_make_kernel("flash_fwd_kernel", 10000 + i) for i in range(50)]
        )
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"r0": 64}, "total_tokens": 64, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)
        sp = self._make_predictor(_stub_annotate())
        with pytest.raises(ValueError, match="Cannot safely align"):
            sp.predict_per_request_metrics(trace_path, sidecar_path)

    def test_small_known_phase_raises(self, tmp_path):
        """A small known-phase iteration (5-kernel flash_fwd) is NOT dropped,
        so it causes a mismatch with fewer sidecar steps."""
        trace = (
            [_make_kernel("flash_fwd_kernel", 1000 + i) for i in range(5)] +
            [_make_kernel("flash_fwd_kernel", 5000 + i) for i in range(50)]
        )
        sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"r0": 64}, "total_tokens": 64, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)
        sp = self._make_predictor(_stub_annotate())
        with pytest.raises(ValueError, match="Cannot safely align"):
            sp.predict_per_request_metrics(trace_path, sidecar_path)

    def test_decode_side_leading_sidecar_row_trimmed(self, tmp_path):
        """Decode-side disagg: scheduler fires an active step on KV-receive
        (minimal GPU work that gets dropped as unknown), then again for the
        actual decode iteration. With first_step_is_prefill=False the leading
        sidecar row whose ts precedes the first retained iteration is trimmed
        so 2 sidecar rows align with 1 retained model iteration.

        Mirrors /tmp/disagg_r22_decode shape: leading active row at ts ~8.4M,
        a long gap, then 1 retained mixed iteration starting at ts ~10.07M.
        """
        first_iter_start_us = 10_067_711
        trace = (
            # Tiny kernel at the leading sidecar timestamp - dropped as unknown
            [_make_kernel("unknown_kernel", 8_369_359)] +
            # Real decode model iteration (mixed phase)
            [_make_kernel("flash_fwd_kernel", first_iter_start_us + i) for i in range(40)] +
            [_make_kernel("internal_gemvx_kernel", first_iter_start_us + 50 + i) for i in range(40)]
        )
        sidecar = [
            # Leading row: KV-receive scheduler step
            {"iter_id": 0, "ts_us": 8_368_341, "scheduled": {"r0": 1}, "total_tokens": 1, "finished": []},
            # Actual decode iteration
            {"iter_id": 1, "ts_us": 10_106_718, "scheduled": {"r0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)
        sp = self._make_predictor(_stub_annotate(predicted_ns=5_000_000.0))
        metrics = sp.predict_per_request_metrics(
            trace_path, sidecar_path, first_step_is_prefill=False
        )
        assert len(metrics) == 1
        # Decode-side: first_step_is_prefill=False, single 1-token step is decode
        assert metrics[0].num_decode_tokens == 1
        assert metrics[0].num_prefill_tokens == 0
        assert metrics[0].decode_time_ms == pytest.approx(5.0, abs=0.01)

    def test_decode_side_strict_alignment_when_no_leading_row(self, tmp_path):
        """Colocated path (first_step_is_prefill=True) does NOT trim leading
        sidecar rows even when sidecar count exceeds retained iterations."""
        trace = (
            # Tiny unknown leading iter - dropped by warmup filter
            [_make_kernel("unknown_kernel", 100)] +
            [_make_kernel("flash_fwd_kernel", 5000 + i) for i in range(40)]
        )
        sidecar = [
            {"iter_id": 0, "ts_us": 100, "scheduled": {"r0": 1}, "total_tokens": 1, "finished": []},
            {"iter_id": 1, "ts_us": 5000, "scheduled": {"r0": 64}, "total_tokens": 64, "finished": []},
            {"iter_id": 2, "ts_us": 10000, "scheduled": {"r0": 1}, "total_tokens": 1, "finished": []},
        ]
        trace_path, sidecar_path = self._setup_files(tmp_path, trace, sidecar)
        sp = self._make_predictor(_stub_annotate())
        # Colocated path: 3 sidecar > 2 iters > 1 kept. Strict mismatch.
        with pytest.raises(ValueError, match="Cannot safely align"):
            sp.predict_per_request_metrics(
                trace_path, sidecar_path, first_step_is_prefill=True
            )


class TestProportionalAllocation:
    """Test proportional observed KV byte allocation for variable prompts."""

    @patch("src.serve_predict.serve_metric_predictor.TracePredictor")
    def test_longer_prompt_gets_more_kv_time(self, MockTP, tmp_path):
        from src.serve_predict.serve_metric_predictor import predict_disaggregated_metrics
        from src.serving.workload.model_config import ModelConfig

        # Two requests with different prompt lengths
        prefill_trace = [_make_kernel("flash_fwd_kernel", 1000 + i) for i in range(10)]
        prefill_sidecar = [
            {"iter_id": 0, "ts_us": 1000, "scheduled": {"short": 32, "long": 128}, "total_tokens": 160, "finished": []},
        ]
        decode_trace = [_make_kernel("internal_gemvx_kernel", 5000 + i) for i in range(10)]
        decode_sidecar = [
            {"iter_id": 0, "ts_us": 5000, "scheduled": {"short": 1, "long": 1}, "total_tokens": 2, "finished": []},
        ]

        for prefix, entries, records in [("p", prefill_trace, prefill_sidecar), ("d", decode_trace, decode_sidecar)]:
            with open(str(tmp_path / f"{prefix}_trace.json"), "w") as f:
                json.dump(entries, f)
            with open(str(tmp_path / f"{prefix}_sched.jsonl"), "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        mock_instance = MagicMock()
        mock_instance.annotate_trace.return_value = [KernelAnnotation(
            index=0, kernel_name="stub", grid=[1,1,1], block=[256,1,1],
            shared_mem=0, category="GEMM", predicted_ns=1_000_000.0, cpu_overhead_ns=0.0,
        )]
        MockTP.return_value = mock_instance

        mc = ModelConfig(hidden_size=2048, intermediate_size=5504,
                        num_attention_heads=16, num_key_value_heads=4,
                        num_hidden_layers=24, head_dim=128, vocab_size=32000)

        metrics = predict_disaggregated_metrics(
            str(tmp_path / "p_trace.json"), str(tmp_path / "p_sched.jsonl"), "configs/hardware/nvidia_a800.yaml",
            str(tmp_path / "d_trace.json"), str(tmp_path / "d_sched.jsonl"), "configs/hardware/nvidia_a800.yaml",
            mc, kv_transfer_bw_bytes_s=50e9, observed_kv_bytes=1000000)

        by_id = {m.request_id: m for m in metrics}
        # Longer prompt (128 tokens) gets more KV transfer time than shorter (32)
        assert by_id["long"].kv_transfer_time_ms > by_id["short"].kv_transfer_time_ms
        # Check proportionality: long has 4x the tokens of short
        # With alpha, the ratio won't be exactly 4x, but long should be > 3x short
        ratio = by_id["long"].kv_transfer_time_ms / by_id["short"].kv_transfer_time_ms
        assert ratio > 2.5  # 128/32 = 4x tokens, so KV time should be roughly 4x
        # Verify aggregate is preserved: total KV bytes inferred from times
        # kv_time = alpha + bytes / bw → bytes = (kv_time - alpha) * bw
        bw_per_ns = 50e9 / 1e9  # bytes per ns
        alpha_ms = 1000.0 / 1e6  # alpha in ms
        total_inferred = sum(
            (m.kv_transfer_time_ms - alpha_ms) * bw_per_ns * 1e6  # ms → ns → bytes
            for m in metrics
        )
        assert abs(total_inferred - 1000000) < 10  # within rounding
