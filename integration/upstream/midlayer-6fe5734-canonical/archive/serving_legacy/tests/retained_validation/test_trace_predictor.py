"""Test for P54(a): content-driven GEMM dimension inference."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.serve_predict.predict_kernel_time_from_trace import TracePredictor
from src.serving.workload.model_config import ModelConfig
from src.serving.validation.kernel_classifier import classify_kernel


def _make_predictor():
    return TracePredictor(
        "configs/hardware/nvidia_a800.yaml",
        ModelConfig(
            hidden_size=2048,
            intermediate_size=5632,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=24,
            head_dim=128,
            vocab_size=151936,
            dtype_bytes=2,
            num_experts=0,
            num_experts_per_tok=2,
        ),
    )


def _make_moe_predictor():
    """Predictor with MoE config (128 experts, 8 active per token)."""
    return TracePredictor(
        "configs/hardware/nvidia_a800.yaml",
        ModelConfig(
            hidden_size=2048,
            intermediate_size=5632,
            num_attention_heads=16,
            num_key_value_heads=8,
            num_hidden_layers=24,
            head_dim=128,
            vocab_size=151936,
            dtype_bytes=2,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
        ),
    )


def _gemm_entry(kernel_name, grid_x, grid_y=1, grid_z=1, block_x=128):
    return {
        "kernel_name": kernel_name,
        "grid": [grid_x, grid_y, grid_z],
        "block": [block_x, 1, 1],
        "shared_mem": 0,
        "category": "GEMM",
        "timestamp_us": None,
    }


class TestGemmPositionMatching:
    def test_qkv_proj_matched_by_grid(self):
        """qkv_proj has a distinctive large N; should be identified by grid."""
        pred = _make_predictor()
        # Qwen 2048 model: qkv_proj N = (16 + 2*8)*128 = 4096
        # M=1 (decode), tile_M=64, tile_N=64
        # grid_x = ceil(1/64) * ceil(4096/64) = 1 * 64 = 64
        entry = _gemm_entry("ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=64)
        pos, M, N, K = pred._match_gemm_position(entry, seq_len=1, batch_size=1, tp_size=1)
        assert pos == "qkv_proj"
        assert N == 4096  # (16 + 2*8) * 128

    def test_unknown_grid_raises_value_error(self):
        """A kernel whose grid_x is clearly nonsensical raises ValueError."""
        pred = _make_predictor()
        # grid_x=9999 is >> max(lm_head expected ~2374); ratio >> 4x for all candidates
        entry = _gemm_entry("ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=99999)
        with pytest.raises(ValueError, match="does not match any expected"):
            pred._match_gemm_position(entry, seq_len=1, batch_size=1, tp_size=1)

    def test_gate_up_proj_larger_than_o_proj(self):
        """gate_up_proj has N=2*I, larger than o_proj N=H; should distinguish them."""
        pred = _make_predictor()
        # gate_up_proj N = 2*5632=11264, tile_N=64 → grid_x = ceil(1/64)*ceil(11264/64) = 176
        entry_gate = _gemm_entry("ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=176)
        pos_gate, _, N_gate, _ = pred._match_gemm_position(entry_gate, seq_len=1, batch_size=1, tp_size=1)
        assert pos_gate == "gate_up_proj"
        assert N_gate == 11264  # 2 * 5632


class TestBuildOpAttr:
    """Verify _build_op_attr produces correct keys for all supported categories."""

    def _make_entry(self, kernel_name="ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8"):
        return {
            "kernel_name": kernel_name,
            "grid": [4, 1, 1],
            "block": [128, 1, 1],
            "shared_mem": 0,
        }

    def test_gemm_op_attr_keys(self):
        pred = _make_predictor()
        # qkv_proj for Qwen 2048: N=(16+2*8)*128=4096, tile=64x64
        # grid_x = ceil(128/64) * ceil(4096/64) = 2 * 64 = 128
        entry = _gemm_entry("ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)
        op_attr = pred._build_op_attr(entry, "GEMM", seq_len=128, batch_size=1, tp_size=1)
        assert "B" in op_attr, "GEMM op_attr missing B"
        assert "M" in op_attr, "GEMM op_attr missing M"
        assert "K" in op_attr
        assert "N" in op_attr

    def test_flashattn_op_attr_keys(self):
        pred = _make_predictor()
        entry = self._make_entry("flash_attn_fwd_kernel")
        op_attr = pred._build_op_attr(entry, "FLASHATTN", seq_len=128, batch_size=1, tp_size=1)
        assert "B" in op_attr, "FA op_attr missing B"
        assert "S" in op_attr
        assert "D" in op_attr
        assert "SP" in op_attr

    def test_fusedmoe_op_attr_keys(self):
        pred = _make_moe_predictor()
        entry = self._make_entry("moe_gemm_kernel")
        op_attr = pred._build_op_attr(entry, "FUSEDMOE", seq_len=128, batch_size=1, tp_size=1)
        assert "TOPK" in op_attr, "FUSEDMOE op_attr missing TOPK (not E_TOP)"
        assert "moe_tiling" in op_attr, "FUSEDMOE op_attr missing moe_tiling"
        assert "E" in op_attr

    def test_norm_op_attr_keys(self):
        pred = _make_predictor()
        # Use a non-GEMM kernel name so _build_op_attr doesn't try to match dims
        entry = self._make_entry("rms_norm_kernel")
        # _build_op_attr for NORM doesn't use grid matching — safe with any entry
        op_attr = pred._build_op_attr(entry, "NORM", seq_len=128, batch_size=1, tp_size=1)
        assert "batch" in op_attr, "NORM op_attr should use 'batch' key"
        assert "hidden" in op_attr, "NORM op_attr should use 'hidden' key"

    def test_activation_op_attr_keys(self):
        pred = _make_predictor()
        entry = self._make_entry("silu_and_mul")
        op_attr = pred._build_op_attr(entry, "ACTIVATION", seq_len=128, batch_size=1, tp_size=1)
        assert "batch" in op_attr, "ACTIVATION op_attr should use 'batch' key"
        assert "intermediate" in op_attr, "ACTIVATION op_attr should use 'intermediate' key"

    def test_unsupported_cat_returns_none(self):
        pred = _make_predictor()
        entry = self._make_entry()
        op_attr = pred._build_op_attr(entry, "UNKNOWN_CAT", seq_len=128, batch_size=1, tp_size=1)
        assert op_attr is None

    def test_elementwise_op_attr_keys(self):
        pred = _make_predictor()
        entry = self._make_entry("vectorized_elementwise_kernel")
        op_attr = pred._build_op_attr(entry, "ELEMENTWISE", seq_len=128, batch_size=1, tp_size=1)
        assert "element_count" in op_attr, "ELEMENTWISE op_attr should use 'element_count'"
        assert "dtype_bytes" in op_attr
        # element_count = grid×block×4 (fp16 vectorized load width)
        grid = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
        block = entry["block"][0] * entry["block"][1] * entry["block"][2]
        assert op_attr["element_count"] == grid * block * 4

    def test_elementwise_predicts_above_launch_overhead(self):
        """Large ELEMENTWISE kernel must use the memory-bound model, not the
        constant launch_overhead_ns default."""
        pred = _make_predictor()
        entry = {
            "kernel_name": "vectorized_elementwise_kernel",
            "grid": [32768, 1, 1],
            "block": [256, 1, 1],
            "shared_mem": 0,
            "timestamp_us": 0.0,
        }
        ann = pred.annotate_trace([entry], seq_len=1, batch_size=1, tp_size=1)[0]
        assert ann.category == "ELEMENTWISE"
        assert ann.predicted_ns > pred.launch_overhead_ns


class TestAnnotateTraceDiff:
    """Verify annotate_trace_diff schema validation and coverage."""

    def test_missing_kernel_name_raises(self):
        pred = _make_predictor()
        trace = [{"grid": [1, 1, 1], "block": [32, 1, 1], "shared_mem": 0}]
        with pytest.raises(ValueError, match="kernel_name"):
            pred.annotate_trace_diff(trace, seq_len=128)

    def test_missing_grid_raises(self):
        pred = _make_predictor()
        trace = [{"kernel_name": "test", "block": [32, 1, 1], "shared_mem": 0}]
        with pytest.raises(ValueError, match="grid"):
            pred.annotate_trace_diff(trace, seq_len=128)

    def test_valid_minimal_trace(self):
        pred = _make_predictor()
        # A minimal GEMM entry whose grid_x=4 doesn't match any layer projection
        # for seq_len=128. annotate_trace_diff logs a warning and excludes it
        # (fail-loud: error is printed, not silently dropped).
        trace = [{
            "kernel_name": "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8",
            "grid": [4, 1, 1],
            "block": [128, 1, 1],
            "shared_mem": 20480,
        }]
        preds = pred.annotate_trace_diff(trace, seq_len=128)
        assert isinstance(preds, list)
        assert len(preds) == 0  # kernel excluded due to dim mismatch
        coverage = pred.get_diff_coverage()
        assert isinstance(coverage, dict)
        assert coverage.get("TC_GEMM", 0) + coverage.get("GEMM", 0) >= 1

    def test_unsupported_category_in_coverage(self):
        pred = _make_predictor()
        trace = [{
            "kernel_name": "nccl_allreduce_kernel",
            "grid": [2, 1, 1],
            "block": [256, 1, 1],
            "shared_mem": 0,
        }]
        preds = pred.annotate_trace_diff(trace, seq_len=128)
        coverage = pred.get_diff_coverage()
        assert "NCCL" in coverage
        assert coverage["NCCL"] == 1

    def test_return_type_is_list_of_predictions(self):
        pred = _make_predictor()
        trace = [{
            "kernel_name": "nccl_allreduce_kernel",
            "grid": [2, 1, 1],
            "block": [256, 1, 1],
            "shared_mem": 0,
        }]
        result = pred.annotate_trace_diff(trace, seq_len=128)
        assert isinstance(result, list), "annotate_trace_diff must return List"
        assert len(result) == 0  # NCCL is unsupported
        coverage = pred.get_diff_coverage()
        assert coverage.get("NCCL", 0) == 1

    def test_gemm_trace_produces_gradient(self):
        from src.dse.diff_formula import aggregate_gradients
        pred = _make_predictor()
        # qkv_proj: grid_x=128 for Qwen 2048 model with seq_len=128
        trace = [_gemm_entry(
            "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)]
        result = pred.annotate_trace_diff(trace, seq_len=128)
        assert isinstance(result, list)
        assert len(result) >= 1, "Expected at least 1 DiffKernelPrediction"
        agg = aggregate_gradients(result)
        assert "grad_capacity_scaled" in agg
        assert agg["t_total_s"] > 0
        # Verify gradient keys exist for the bottleneck resource
        assert len(agg["grad_capacity_scaled"]) > 0


class TestDseIntegration:
    """Verify e2e_dse.py functions work on a fixture trace."""

    def test_predict_with_scales_changes_latency(self):
        from scripts.dse.run_dse import _predict_with_scales
        from src.kernel.hardware_spec import HWSpec
        pred = _make_predictor()
        hw = HWSpec("configs/hardware/nvidia_a800.yaml")
        trace = [_gemm_entry(
            "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)]
        t_base, _ = _predict_with_scales(pred, trace, 128, hw, 1.0, 1.0, 1, 1)
        t_2x, _ = _predict_with_scales(pred, trace, 128, hw, 2.0, 2.0, 1, 1)
        t_half, _ = _predict_with_scales(pred, trace, 128, hw, 0.5, 0.5, 1, 1)
        assert t_half > t_base > t_2x, \
            f"Scaling broken: 0.5x={t_half:.0f} > base={t_base:.0f} > 2x={t_2x:.0f}"

    def test_gradient_sign_matches_finite_diff(self):
        from scripts.dse.run_dse import _predict_with_scales, _compute_scale_gradients
        from src.kernel.hardware_spec import HWSpec
        pred = _make_predictor()
        hw = HWSpec("configs/hardware/nvidia_a800.yaml")
        trace = [_gemm_entry(
            "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)]
        eps = 0.01
        # Check compute_scale: analytic sign must match finite-diff sign
        t_lo, _ = _predict_with_scales(pred, trace, 128, hw, 1.0, 1.0 - eps, 1, 1)
        t_hi, _ = _predict_with_scales(pred, trace, 128, hw, 1.0, 1.0 + eps, 1, 1)
        fd_compute = (t_hi - t_lo) / (2 * eps)
        _, agg = _predict_with_scales(pred, trace, 128, hw, 1.0, 1.0, 1, 1)
        _, analytic_comp = _compute_scale_gradients(agg, hw, 1.0, 1.0)
        analytic_comp_ns = analytic_comp * 1e9
        if abs(analytic_comp) > 1e-20:
            assert analytic_comp_ns * fd_compute > 0, \
                f"Compute gradient sign mismatch: analytic={analytic_comp_ns:.1f}ns, fd={fd_compute:.1f}ns"
        # Check dram_scale: analytic sign must match finite-diff sign
        t_lo_d, _ = _predict_with_scales(pred, trace, 128, hw, 1.0 - eps, 1.0, 1, 1)
        t_hi_d, _ = _predict_with_scales(pred, trace, 128, hw, 1.0 + eps, 1.0, 1, 1)
        fd_dram = (t_hi_d - t_lo_d) / (2 * eps)
        analytic_dram, _ = _compute_scale_gradients(agg, hw, 1.0, 1.0)
        analytic_dram_ns = analytic_dram * 1e9
        if abs(analytic_dram) > 1e-20:
            assert analytic_dram_ns * fd_dram > 0, \
                f"DRAM gradient sign mismatch: analytic={analytic_dram_ns:.1f}ns, fd={fd_dram:.1f}ns"

    def test_grid_search_produces_optimum(self):
        from scripts.dse.run_dse import grid_search, _predict_with_scales
        from src.kernel.hardware_spec import HWSpec
        pred = _make_predictor()
        hw = HWSpec("configs/hardware/nvidia_a800.yaml")
        trace = [_gemm_entry(
            "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)]
        best_t, dram_s, comp_s = grid_search(
            pred, trace, 128, hw, 1, 1, grid_size=3, lo=0.5, hi=2.0)
        # Baseline should be within [0.5, 2.0] range
        t_base, _ = _predict_with_scales(pred, trace, 128, hw, 1.0, 1.0, 1, 1)
        assert best_t <= t_base, \
            f"Grid search best ({best_t:.0f}ns) should be <= baseline ({t_base:.0f}ns)"

    def test_gradient_descent_converges_within_grid_tolerance(self):
        from scripts.dse.run_dse import gradient_descent, grid_search, _predict_with_scales
        from src.kernel.hardware_spec import HWSpec
        pred = _make_predictor()
        hw = HWSpec("configs/hardware/nvidia_a800.yaml")
        # Use a single GEMM entry with a large grid — this produces a
        # compute-bound or L2-bound kernel where gradient descent finds
        # the same optimum as grid search (both should converge to the
        # upper bound since higher capacity = lower latency, but the
        # objective is bounded by launch overhead).
        trace = [_gemm_entry(
            "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)]
        t_grad, dram_g, comp_g, traj = gradient_descent(
            pred, trace, 128, hw, 1, 1, max_steps=100, step_size=0.05,
            lo=0.5, hi=2.0)
        t_grid, dram_gs, comp_gs = grid_search(
            pred, trace, 128, hw, 1, 1, grid_size=5, lo=0.5, hi=2.0)
        t_base, _ = _predict_with_scales(pred, trace, 128, hw, 1.0, 1.0, 1, 1)
        assert t_grad > 0 and t_grid > 0, "Predictions should be positive"
        assert t_grad <= t_base, f"gradient ({t_grad:.0f}ns) <= baseline ({t_base:.0f}ns)"
        assert len(traj) >= 1, "Should have at least 1 trajectory step"
        if t_grid > 0:
            gap_pct = abs(t_grad - t_grid) / t_grid * 100
            assert gap_pct <= 5.0, \
                f"Gradient descent ({t_grad:.0f}ns) should be within 5% of grid ({t_grid:.0f}ns), got {gap_pct:.1f}%"

    def test_zero_scales_rejected(self):
        from scripts.dse.run_dse import _predict_with_scales
        from src.kernel.hardware_spec import HWSpec
        import pytest as _pytest
        pred = _make_predictor()
        hw = HWSpec("configs/hardware/nvidia_a800.yaml")
        trace = [_gemm_entry(
            "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8", grid_x=128)]
        with _pytest.raises(ValueError, match="dram_bw_scale"):
            _predict_with_scales(pred, trace, 128, hw, 0.0, 1.0, 1, 1)
        with _pytest.raises(ValueError, match="compute_scale"):
            _predict_with_scales(pred, trace, 128, hw, 1.0, 0.0, 1, 1)

    def test_cli_scale_args_change_baseline(self):
        import subprocess, sys, tempfile, json, os
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            output_path = os.path.join(tmpdir, "result.json")
            sidecar_path = trace_path + "_provenance.json"
            with open(trace_path, "w") as f:
                json.dump([_gemm_entry(
                    "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8",
                    grid_x=128)], f)
            with open(sidecar_path, "w") as f:
                json.dump({}, f)
            # Non-default scales should produce different output
            result = subprocess.run(
                [sys.executable, "scripts/dse/run_dse.py",
                 "--trace", trace_path,
                 "--hw-config", "configs/hardware/nvidia_a800.yaml",
                 "--seq-len", "128", "--mode", "gradient", "--steps", "2",
                 "--dram-bw-scale", "2.0", "--compute-scale", "2.0",
                 "--output", output_path],
                capture_output=True, text=True)
            assert result.returncode == 0
            with open(output_path) as f:
                results = json.load(f)
            assert results["baseline_t_gpu_ns"] > 0

    def test_missing_trace_args_rejected(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/dse/run_dse.py"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, \
            "e2e_dse.py without --trace should exit non-zero"

    def test_provenance_json_keys(self):
        """Verify provenance sidecar and result JSON contain required keys."""
        import tempfile, json, os
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.json")
            sidecar_path = trace_path + "_provenance.json"
            output_path = os.path.join(tmpdir, "result.json")
            # Write fixture trace
            with open(trace_path, "w") as f:
                json.dump([_gemm_entry(
                    "ampere_fp16_s16816gemm_fp16_64x64x32_tn_align8",
                    grid_x=128)], f)
            # Write provenance sidecar
            with open(sidecar_path, "w") as f:
                json.dump({"capture_command": "test_capture_cmd"}, f)
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "scripts/dse/run_dse.py",
                 "--trace", trace_path,
                 "--hw-config", "configs/hardware/nvidia_a800.yaml",
                 "--seq-len", "128",
                 "--mode", "gradient",
                 "--steps", "2",
                 "--output", output_path],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, \
                f"e2e_dse.py failed: {result.stderr[:500]}"
            with open(output_path) as f:
                results = json.load(f)
            assert "dev_commit" in results
            assert "archive_commit" in results
            assert "trace_path" in results
            assert "hw_config_path" in results
            assert "model_path" in results
            assert "vllm_version" in results
            assert "capture_command" in results
            assert results["capture_command"] == "test_capture_cmd"
            # Verify dev_commit matches git HEAD
            import subprocess as _sp
            try:
                head = _sp.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
                assert results["dev_commit"] == head, \
                    f"dev_commit {results['dev_commit']} != HEAD {head}"
            except FileNotFoundError:
                pass  # git unavailable, skip check


class TestDseGradientCorrectness:
    """Verify AC-3: compute_scale gradients for all SM-clocked resources."""

    @pytest.fixture(scope="class")
    def hw(self):
        from src.kernel.hardware_spec import HWSpec
        return HWSpec("configs/hardware/nvidia_a800.yaml")

    @pytest.fixture(scope="class")
    def formula(self, hw):
        from src.dse.diff_formula import DiffPerformanceFormula
        return DiffPerformanceFormula(hw)

    @staticmethod
    def _make_inp(hw, demands, is_nt=True, active_units=None, l1_bypass=False):
        from src.kernel.validation.legacy_kernel_base import PredictionInput
        base = {r: 0.0 for r in ['compute_tensor', 'compute_fma', 'compute_sfu',
                                  'compute_alu', 'l2_cache', 'dram',
                                  'smem', 'l1_cache', 'xbar']}
        base.update(demands)
        return PredictionInput(demand_per_resource=base, eta=1.0,
            active_units=active_units or hw.num_compute_units,
            sm_freq_hz=hw.nominal_sm_freq_hz,
            dram_freq_hz=hw.nominal_dram_freq_hz,
            l1_bypass=l1_bypass, duration_ns=0,
            grid_size=hw.num_compute_units, kernel_name='test',
            is_non_tensor=is_nt, pipeline_floor_ns=0, dram_bw_eff=1.0)

    def _finite_diff_scale_gradients(self, hw, formula, demands, *,
                                     active_units=None, l1_bypass=False,
                                     eps=0.01):
        inp_hi = self._make_inp(
            hw, demands, True, active_units=active_units, l1_bypass=l1_bypass)
        inp_hi.dram_freq_hz = hw.nominal_dram_freq_hz * (1 + eps)
        inp_lo = self._make_inp(
            hw, demands, True, active_units=active_units, l1_bypass=l1_bypass)
        inp_lo.dram_freq_hz = hw.nominal_dram_freq_hz * (1 - eps)
        fd_dram = (formula.predict_kernel(inp_hi).t_total_ns -
                   formula.predict_kernel(inp_lo).t_total_ns) / (2 * eps)

        inp_hi = self._make_inp(
            hw, demands, True, active_units=active_units, l1_bypass=l1_bypass)
        inp_hi.sm_freq_hz = hw.nominal_sm_freq_hz * (1 + eps)
        inp_lo = self._make_inp(
            hw, demands, True, active_units=active_units, l1_bypass=l1_bypass)
        inp_lo.sm_freq_hz = hw.nominal_sm_freq_hz * (1 - eps)
        fd_compute = (formula.predict_kernel(inp_hi).t_total_ns -
                      formula.predict_kernel(inp_lo).t_total_ns) / (2 * eps)
        return fd_dram, fd_compute

    def _check_gradient(self, hw, formula, demands, is_nt, msg):
        from scripts.dse.run_dse import _compute_scale_gradients
        from src.dse.diff_formula import aggregate_gradients
        inp = self._make_inp(hw, demands, is_nt)
        pred = formula.predict_kernel(inp)
        agg = aggregate_gradients([pred])
        _, gc = _compute_scale_gradients(agg, hw, 1.0, 1.0)
        eps = 0.01
        inp_hi = self._make_inp(hw, demands, is_nt)
        inp_hi.sm_freq_hz = hw.nominal_sm_freq_hz * (1 + eps)
        inp_lo = self._make_inp(hw, demands, is_nt)
        inp_lo.sm_freq_hz = hw.nominal_sm_freq_hz * (1 - eps)
        fd = (formula.predict_kernel(inp_hi).t_total_ns -
              formula.predict_kernel(inp_lo).t_total_ns) / (2 * eps)
        analytic_ns = gc * 1e9
        assert analytic_ns * fd > 0, \
            f"{msg}: analytic={analytic_ns:.1f}ns, fd={fd:.1f}ns (sign mismatch)"
        assert abs(analytic_ns) > 1e-10, \
            f"{msg}: analytic gradient should be non-zero"

    def test_l2_cache_gradient(self, hw, formula):
        self._check_gradient(hw, formula,
            {'l2_cache': 1e10}, True, "L2-bound")

    def test_compute_fma_gradient(self, hw, formula):
        self._check_gradient(hw, formula,
            {'compute_fma': 1e10}, True, "FMA-bound")

    def test_compute_alu_gradient(self, hw, formula):
        self._check_gradient(hw, formula,
            {'compute_alu': 1e10}, True, "ALU-bound")

    def test_compute_sfu_gradient(self, hw, formula):
        self._check_gradient(hw, formula,
            {'compute_sfu': 1e10}, True, "SFU-bound")

    def test_dram_gradient(self, hw, formula):
        from scripts.dse.run_dse import _compute_scale_gradients
        from src.dse.diff_formula import aggregate_gradients
        demands = {'dram': 1e10}
        inp = self._make_inp(hw, demands, True)
        pred = formula.predict_kernel(inp)
        agg = aggregate_gradients([pred])
        gd, _ = _compute_scale_gradients(agg, hw, 1.0, 1.0)
        eps = 0.01
        inp_hi = self._make_inp(hw, demands, True)
        inp_hi.dram_freq_hz = hw.nominal_dram_freq_hz * (1 + eps)
        inp_lo = self._make_inp(hw, demands, True)
        inp_lo.dram_freq_hz = hw.nominal_dram_freq_hz * (1 - eps)
        fd = (formula.predict_kernel(inp_hi).t_total_ns -
              formula.predict_kernel(inp_lo).t_total_ns) / (2 * eps)
        analytic_ns = gd * 1e9
        assert analytic_ns * fd > 0, \
            f"DRAM-bound: analytic={analytic_ns:.1f}ns, fd={fd:.1f}ns (sign mismatch)"
        assert abs(analytic_ns) > 1e-10, \
            f"DRAM-bound: analytic gradient should be non-zero"

    def test_unsaturated_dram_gradient_routes_to_compute_scale(self, hw, formula):
        from scripts.dse.run_dse import _compute_scale_gradients
        from src.dse.diff_formula import aggregate_gradients
        demands = {'dram': 1e10}
        active_units = max(1, int(hw.sm_saturate_bw // 4))
        inp = self._make_inp(hw, demands, True, active_units=active_units)
        pred = formula.predict_kernel(inp)
        assert pred.bottleneck_group == "dram"

        agg = aggregate_gradients([pred])
        gd, gc = _compute_scale_gradients(agg, hw, 1.0, 1.0)
        fd_dram, fd_compute = self._finite_diff_scale_gradients(
            hw, formula, demands, active_units=active_units)

        analytic_dram_ns = gd * 1e9
        analytic_compute_ns = gc * 1e9
        assert abs(fd_dram) < 1e-6
        assert abs(analytic_dram_ns) < 1e-6
        assert analytic_compute_ns == pytest.approx(fd_compute, rel=0.02, abs=1e-6)
        assert analytic_compute_ns < 0

    def test_saturated_dram_gradient_matches_dram_scale_finite_diff(self, hw, formula):
        from scripts.dse.run_dse import _compute_scale_gradients
        from src.dse.diff_formula import aggregate_gradients
        demands = {'dram': 1e10}
        inp = self._make_inp(hw, demands, True, active_units=hw.num_compute_units)
        pred = formula.predict_kernel(inp)
        assert pred.bottleneck_group == "dram"

        agg = aggregate_gradients([pred])
        gd, gc = _compute_scale_gradients(agg, hw, 1.0, 1.0)
        fd_dram, fd_compute = self._finite_diff_scale_gradients(
            hw, formula, demands, active_units=hw.num_compute_units)

        analytic_dram_ns = gd * 1e9
        assert analytic_dram_ns == pytest.approx(fd_dram, rel=0.02, abs=1e-6)
        assert analytic_dram_ns < 0
        assert abs(fd_compute) < 1e-6
        assert abs(gc * 1e9) < 1e-6


class TestCapturedTraceCoverage:
    """Verify AC-2: annotate_trace_diff on captured vLLM trace fixture."""

    def test_fixture_trace_coverage_above_90_pct(self):
        import json, os
        from src.kernel.validation.dataset_io import get_emulation_output_dir
        fixture_path = os.path.join(get_emulation_output_dir(), "eager_trace.json")
        if not os.path.exists(fixture_path):
            pytest.skip("trace fixture not available — run serve_trace/trace_eager.py first")
        with open(fixture_path) as f:
            trace = json.load(f)
        pred = _make_predictor()
        preds = pred.annotate_trace_diff(trace, seq_len=1, batch_size=1, tp_size=1)
        coverage = pred.get_diff_coverage()
        total = len(trace)
        excluded = sum(coverage.values())
        modelled = total - excluded
        coverage_pct = modelled / total * 100
        # AC-2: >=90% of GEMM/FA/MOE/NORM focus set on captured vLLM 0.11 trace.
        # ELEMENTWISE (752 entries) has no midlayer predictor — excluded from gate.
        # With vLLM 0.11 FA parser and GEMV fallback, focus set reaches 100%.
        focus_cats = {'TC_GEMM', 'SPLITK', 'GEMV', 'GEMM', 'FLASH_ATTN',
                       'FUSEDMOE', 'NORM', 'FLASHATTN'}
        focus_excluded = sum(v for k, v in coverage.items() if k in focus_cats)
        focus_total = sum(1 for e in trace
                          if classify_kernel(e) in focus_cats)
        focus_pct = (focus_total - focus_excluded) / focus_total * 100
        assert focus_pct >= 90.0, \
            f"Focus coverage {focus_pct:.1f}% < 90%: excluded={focus_excluded}/{focus_total}"
        assert len(preds) >= 1, "Should produce at least 1 DiffKernelPrediction"


class TestReduceGatherPredictors:
    """Verify ReducePredictor and GatherPredictor behavior."""

    @pytest.fixture
    def hw(self):
        from src.kernel.hardware_spec import HWSpec
        return HWSpec("configs/hardware/nvidia_a800.yaml")

    @pytest.fixture
    def synthetic_ncu(self):
        from src.serve_predict.predict_kernel_time_from_trace import TracePredictor
        return TracePredictor.build_synthetic_ncu(
            {"kernel_name": "reduce_test", "grid": [1, 1, 1],
             "block": [128, 1, 1], "shared_mem": 0},
            {}, 128,
            sm_freq_hz=1.13e9, dram_freq_hz=1.593e9,
        )

    def test_reduce_zero_elements_raises(self, hw, synthetic_ncu):
        from src.kernel.validation.legacy_kernel_base import create_predictor
        r = create_predictor("REDUCE")
        with pytest.raises(ValueError, match="element_count"):
            r.forward(synthetic_ncu, hw, {"element_count": 0})

    def test_gather_zero_elements_raises(self, hw, synthetic_ncu):
        from src.kernel.validation.legacy_kernel_base import create_predictor
        g = create_predictor("GATHER")
        with pytest.raises(ValueError, match="element_count"):
            g.forward(synthetic_ncu, hw, {"element_count": 0})

    def test_reduce_forward_within_roofline(self, hw, synthetic_ncu):
        from src.kernel.validation.legacy_kernel_base import create_predictor
        from src.kernel.formula import PerformanceFormula
        r = create_predictor("REDUCE")
        formula = PerformanceFormula(hw)
        inp = r.forward(synthetic_ncu, hw, {"element_count": 1000000})
        pred = formula.predict_kernel(inp)
        assert pred.t_total_ns > 0

    def test_gather_scattered_dram_eff(self, hw, synthetic_ncu):
        from src.kernel.validation.legacy_kernel_base import create_predictor
        g = create_predictor("GATHER")
        inp = g.forward(synthetic_ncu, hw, {
            "element_count": 1000000, "scattered": True})
        # Verify dram_bw_eff is set to the hardware scattered efficiency
        assert inp.dram_bw_eff == hw.dram_efficiency_scattered
