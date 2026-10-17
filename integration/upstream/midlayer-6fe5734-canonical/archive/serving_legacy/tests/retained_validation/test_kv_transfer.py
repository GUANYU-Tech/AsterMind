"""Tests for src/serve_trace/kv_transfer.py"""
import pytest

from src.kernel.validation.legacy_ops.comm import (
    compute_expected_kv_bytes,
    predict_kv_transfer_ns as predict_kv_transfer_time,
)
from src.serve_trace.kv_transfer import (
    parse_kv_transfers,
    predict_kv_transfer,
    KVTransferOp,
    KVTransferPrediction,
)


def _make_nccl_entry(op_type, bytes_val, nranks=2, rank=0, op_id=0, ts_us=1000):
    """Create a synthetic nccl_send/nccl_recv trace entry."""
    return {
        "kernel_name": f"nccl_{op_type}",
        "grid": [nranks, rank, op_id],
        "block": [bytes_val & 0xFFFFFFFF, bytes_val >> 32, 0],
        "shared_mem": 0,
        "timestamp_us": ts_us,
    }


class TestParseKVTransfers:
    def test_extracts_send_recv(self):
        trace = [
            _make_nccl_entry("send", 1024, ts_us=100),
            {"kernel_name": "s16816gemm", "grid": [1, 1, 1], "block": [256, 1, 1],
             "shared_mem": 0, "timestamp_us": 200},
            _make_nccl_entry("recv", 1024, rank=1, ts_us=300),
        ]
        ops = parse_kv_transfers(trace)
        assert len(ops) == 2
        assert ops[0].op_type == "send"
        assert ops[0].bytes == 1024
        assert ops[1].op_type == "recv"

    def test_large_bytes_encoding(self):
        large_bytes = 5 * (1 << 32) + 42  # > 4GB
        trace = [_make_nccl_entry("send", large_bytes)]
        ops = parse_kv_transfers(trace)
        assert ops[0].bytes == large_bytes

    def test_empty_trace_returns_empty(self):
        trace = [{"kernel_name": "gemm", "grid": [1, 1, 1], "block": [1, 1, 1],
                  "shared_mem": 0, "timestamp_us": 0}]
        assert parse_kv_transfers(trace) == []

    def test_metadata_decoded(self):
        trace = [_make_nccl_entry("send", 2048, nranks=4, rank=2, op_id=7, ts_us=500)]
        ops = parse_kv_transfers(trace)
        assert ops[0].nranks == 4
        assert ops[0].rank == 2
        assert ops[0].op_id == 7
        assert ops[0].timestamp_us == 500


class TestExpectedKVBytes:
    def test_basic_computation(self):
        # 24 layers, 8 kv_heads, head_dim=128, prompt_len=512, fp16, tp=1
        expected = 24 * 2 * 8 * 128 * 512 * 2
        assert compute_expected_kv_bytes(24, 8, 128, 512, 2, 1) == expected

    def test_tp_sharding(self):
        # With TP=2, each rank has half the kv_heads
        full = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        half = compute_expected_kv_bytes(24, 8, 128, 512, 2, 2)
        assert half == full // 2

    def test_longer_prompt_more_bytes(self):
        short = compute_expected_kv_bytes(24, 8, 128, 256, 2, 1)
        long = compute_expected_kv_bytes(24, 8, 128, 1024, 2, 1)
        assert long == short * 4


class TestPredictKVTransferTime:
    def test_basic_prediction(self):
        # 1GB at 50 GB/s + 1000ns alpha
        bytes_val = 1_000_000_000
        bw = 50e9  # 50 GB/s
        alpha = 1000.0  # 1us
        t = predict_kv_transfer_time(bytes_val, alpha, bw)
        # Expected: 1000 + 1e9 / (50e9/1e9) = 1000 + 1e9/50 = 1000 + 20_000_000
        expected = alpha + bytes_val / (bw / 1e9)
        assert t == pytest.approx(expected)

    def test_zero_bytes(self):
        t = predict_kv_transfer_time(0, 1000.0, 50e9)
        assert t == 1000.0  # just alpha


class TestPredictKVTransfer:
    def test_with_send_ops(self):
        trace = [
            _make_nccl_entry("send", 1000, ts_us=100),
            _make_nccl_entry("send", 2000, ts_us=200),
        ]
        pred = predict_kv_transfer(
            trace, num_layers=1, num_kv_heads=1, head_dim=1, prompt_len=1,
            dtype_bytes=2, tp_size=1, nccl_alpha_ns=500.0, nvlink_bw_bytes_s=50e9)
        assert pred.total_bytes == 3000
        assert pred.transfer_time_ns > 500.0

    def test_tp_ranks_with_same_op_id_are_parallel(self):
        trace = [
            _make_nccl_entry("send", 1000, nranks=4, rank=0, op_id=7, ts_us=100),
            _make_nccl_entry("send", 2000, nranks=4, rank=1, op_id=7, ts_us=101),
            _make_nccl_entry("send", 1500, nranks=4, rank=2, op_id=7, ts_us=102),
            _make_nccl_entry("send", 1200, nranks=4, rank=3, op_id=7, ts_us=103),
        ]
        pred = predict_kv_transfer(
            trace, num_layers=1, num_kv_heads=4, head_dim=1, prompt_len=1,
            dtype_bytes=2, tp_size=4, nccl_alpha_ns=10.0,
            nvlink_bw_bytes_s=1e12, max_overhead_ratio=1000.0)

        assert pred.total_bytes == 5700
        assert pred.transfer_time_ns == pytest.approx(10.0 + 2000 / 1000.0)

    def test_distinct_send_op_ids_charge_alpha_per_group(self):
        trace = [
            _make_nccl_entry("send", 1000, nranks=2, rank=0, op_id=0, ts_us=100),
            _make_nccl_entry("send", 2000, nranks=2, rank=1, op_id=0, ts_us=101),
            _make_nccl_entry("send", 3000, nranks=2, rank=0, op_id=1, ts_us=200),
            _make_nccl_entry("send", 4000, nranks=2, rank=1, op_id=1, ts_us=201),
        ]
        pred = predict_kv_transfer(
            trace, num_layers=1, num_kv_heads=2, head_dim=1, prompt_len=1,
            dtype_bytes=2, tp_size=2, nccl_alpha_ns=10.0,
            nvlink_bw_bytes_s=1e12, max_overhead_ratio=1000.0)

        assert pred.total_bytes == 10_000
        expected_ns = (10.0 + 2000 / 1000.0) + (10.0 + 4000 / 1000.0)
        assert pred.transfer_time_ns == pytest.approx(expected_ns)

    def test_no_send_ops_warns(self, capsys):
        trace = [_make_nccl_entry("recv", 1000)]
        pred = predict_kv_transfer(
            trace, num_layers=24, num_kv_heads=8, head_dim=128, prompt_len=512,
            dtype_bytes=2, tp_size=1, nccl_alpha_ns=500.0, nvlink_bw_bytes_s=50e9)
        assert pred.total_bytes == 0
        assert pred.transfer_time_ns == 0.0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "No nccl_send" in captured.err

    def test_byte_mismatch_ratio(self):
        # Expected = 24 * 2 * 8 * 128 * 512 * 2 = 50_331_648
        expected = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        # Observed: exact match
        trace = [_make_nccl_entry("send", expected)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 512, 2, 1, 500.0, 50e9)
        assert pred.byte_mismatch_ratio == pytest.approx(0.0)

    def test_byte_mismatch_nonzero(self):
        expected = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        # Observed: half the expected
        trace = [_make_nccl_entry("send", expected // 2)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 512, 2, 1, 500.0, 50e9)
        assert pred.byte_mismatch_ratio == pytest.approx(0.5, abs=0.01)

    def test_multi_request_expected_bytes(self):
        """3 requests: total expected = 3x per-request bytes."""
        per_req = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        total = per_req * 3
        trace = [_make_nccl_entry("send", total)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 512, 2, 1, 500.0, 50e9, num_requests=3)
        assert pred.expected_bytes == total
        assert pred.byte_mismatch_ratio == pytest.approx(0.0)

    def test_multi_request_mismatch_when_single(self):
        """Observed = 3x expected is accepted as block overhead (within 4x default)."""
        per_req = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        trace = [_make_nccl_entry("send", per_req * 3)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 512, 2, 1, 500.0, 50e9, num_requests=1)
        assert pred.byte_mismatch_ratio == pytest.approx(0.0)

    def test_excessive_over_transfer_detected(self):
        """Observed = 10x expected exceeds max_overhead_ratio and reports mismatch."""
        per_req = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        trace = [_make_nccl_entry("send", per_req * 10)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 512, 2, 1, 500.0, 50e9, num_requests=1)
        assert pred.byte_mismatch_ratio > 0.0

    def test_under_transfer_reports_mismatch(self):
        """If observed < expected, mismatch ratio is positive (missing data)."""
        expected = compute_expected_kv_bytes(24, 8, 128, 512, 2, 1)
        trace = [_make_nccl_entry("send", expected // 2)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 512, 2, 1, 500.0, 50e9)
        assert pred.byte_mismatch_ratio == pytest.approx(0.5, abs=0.01)

    def test_prompt_lens_exact_sum(self):
        """Non-uniform prompt_lens uses exact per-request sum, not floor(avg)*N."""
        e31 = compute_expected_kv_bytes(24, 8, 128, 31, 2, 1)
        e32 = compute_expected_kv_bytes(24, 8, 128, 32, 2, 1)
        exact_total = e31 + e32
        floor_avg_total = compute_expected_kv_bytes(24, 8, 128, 31, 2, 1) * 2
        assert exact_total != floor_avg_total

        trace = [_make_nccl_entry("send", exact_total)]
        pred = predict_kv_transfer(
            trace, 24, 8, 128, 31, 2, 1, 500.0, 50e9,
            num_requests=2, prompt_lens=[31, 32])
        assert pred.expected_bytes == exact_total
        assert pred.byte_mismatch_ratio == pytest.approx(0.0)
