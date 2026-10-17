"""ModelSimulator: end-to-end GPU kernel latency predictor for LLM inference.

Generates a synthetic transformer kernel trace from model config + hardware spec
+ workload parameters (batch, seq_len, n_decode), then runs each kernel through
the midlayer forward predictors to produce per-kernel latency estimates.

Two public APIs:
  predict()      → (t_e2e_gpu_ns, per_category_breakdown)
  predict_diff() → (t_e2e_gpu_ns, agg_grad)  with analytical gradients

Both paths collect per-kernel DiffKernelPrediction objects for gradient flow.
`predict()` discards the gradient objects after summing; `predict_diff()` returns
the aggregated gradient dict from aggregate_gradients().

Scope: GPU kernel sum only (no CPU framework overhead).  The returned
t_e2e_gpu_ns is the differentiable quantity for hardware DSE.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from src.serve_predict.model_config import ModelConfig, load_model_config
from src.serve_trace.shape_inference import infer_gemm_dimensions
from src.kernel.hw_spec import HWSpec
from src.dse.diff_formula import DiffPerformanceFormula, aggregate_gradients
from src.kernel.kernel_base import create_predictor


# ---------------------------------------------------------------------------
# Synthetic kernel entry
# ---------------------------------------------------------------------------

@dataclass
class SyntheticKernelEntry:
    """A single synthetic GPU kernel entry for the prediction pipeline.

    Matches the dict format expected by TracePredictor.build_synthetic_ncu().
    """
    kernel_name: str          # matches classify_kernel() patterns in trace_predictor
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    shared_mem: int
    op_type: str              # "GEMM", "FLASHATTN", "NORM", etc.
    op_attr: Dict             # predictor-specific shape attributes


# ---------------------------------------------------------------------------
# ModelSimulator
# ---------------------------------------------------------------------------

class ModelSimulator:
    """Predicts end-to-end GPU kernel latency from model config + hardware spec.

    No real GPU required — all predictions come from the midlayer forward
    performance model applied to a synthetically generated kernel trace.
    """

    def __init__(self, hw_config_path: str):
        """
        Args:
            hw_config_path: path to hw_configs/*.yaml
        """
        from pathlib import Path
        if not Path(hw_config_path).exists():
            raise FileNotFoundError(
                f"Hardware config not found: {hw_config_path}")
        self._hw_config_path = hw_config_path
        self._hw = HWSpec(hw_config_path)

    # ------------------------------------------------------------------
    # Synthetic trace generation
    # ------------------------------------------------------------------

    def generate_trace(self, model_config: ModelConfig,
                       batch: int = 1,
                       seq_len: int = 128,
                       n_decode: int = 0,
                       tp_size: int = 1) -> List[SyntheticKernelEntry]:
        """Generate a synthetic kernel trace for a full LLM forward pass.

        Covers one prefill step (batch × seq_len tokens) followed by n_decode
        decode steps (batch × 1 token each).  Per-layer kernel types:
          GEMM   (qkv_proj, o_proj, gate_up_proj, down_proj)
          FLASH  (multi-head self-attention)
          NORM   (pre-attention RMSNorm, pre-FFN RMSNorm)
          ROTARY (Q/K rotary embeddings)
          ACTIVATION (SiLU gate)
          KV_CACHE   (KV append)

        For MoE layers (num_experts > 0): FUSEDMOE replaces gate_up + down +
        ACTIVATION; a REDUCE kernel is added for expert routing softmax, and
        GATHER kernels wrap the routed token dispatch.

        Args:
            model_config: parsed ModelConfig (from load_model_config())
            batch: batch size
            seq_len: prefill sequence length
            n_decode: number of decode steps
            tp_size: tensor parallelism factor

        Returns:
            list of SyntheticKernelEntry in execution order

        Raises:
            ValueError: for invalid model_config or tp_size
        """
        if model_config.num_hidden_layers <= 0:
            raise ValueError(
                f"model_config.num_hidden_layers must be > 0, got "
                f"{model_config.num_hidden_layers}")
        if model_config.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"tp_size={tp_size} does not divide num_attention_heads="
                f"{model_config.num_attention_heads}. "
                f"tp_size must divide num_attention_heads for attention head sharding.")

        trace: List[SyntheticKernelEntry] = []
        is_moe = model_config.num_experts > 0

        # Prefill pass
        for _ in range(model_config.num_hidden_layers):
            trace.extend(
                self._layer_kernels(model_config, batch, seq_len, tp_size,
                                    is_decode=False, is_moe=is_moe))
        # lm_head (prefill only): large TC_GEMM over vocab dimension
        # For decode, cuBLAS dispatches internal_gemvx (M=1 → GEMV, not TC_GEMM)
        trace.extend(self._make_gemm(model_config, "lm_head", batch, seq_len, tp_size))

        # Decode passes
        for _ in range(n_decode):
            for _ in range(model_config.num_hidden_layers):
                trace.extend(
                    self._layer_kernels(model_config, batch, 1, tp_size,
                                        is_decode=True, is_moe=is_moe))
            # lm_head for each decode step (M=batch → GEMV for batch=1)
            trace.extend(self._make_gemm(model_config, "lm_head", batch, 1, tp_size))

        return trace

    def _layer_kernels(self, m: ModelConfig, batch: int, seq_len: int,
                       tp_size: int, is_decode: bool,
                       is_moe: bool) -> List[SyntheticKernelEntry]:
        """Generate all kernels for one transformer layer."""
        kernels: List[SyntheticKernelEntry] = []

        # ── NORM 1 (pre-attention RMSNorm) ──
        kernels.append(self._make_norm(m, batch, seq_len, tp_size, "norm_1"))

        # ── QKV projection ──
        kernels.extend(self._make_gemm(m, "qkv_proj", batch, seq_len, tp_size))

        # ── Rotary embeddings ──
        kernels.append(self._make_rotary(m, batch, seq_len, tp_size))

        # ── KV cache append ──
        kernels.append(self._make_kv_cache(m, batch, seq_len, tp_size))

        # ── FlashAttention ──
        kernels.extend(self._make_flash_attn(m, batch, seq_len, tp_size))

        # ── O projection ──
        kernels.extend(self._make_gemm(m, "o_proj", batch, seq_len, tp_size))

        # ── NORM 2 (pre-FFN RMSNorm) ──
        kernels.append(self._make_norm(m, batch, seq_len, tp_size, "norm_2"))

        # ── FFN block ──
        if is_moe:
            kernels.extend(
                self._make_moe_ffn(m, batch, seq_len, tp_size))
        else:
            # Dense FFN: gate_up (GEMM) → SiLU (ACTIVATION) → down (GEMM)
            kernels.extend(self._make_gemm(m, "gate_up_proj", batch, seq_len, tp_size))
            kernels.append(self._make_activation(m, batch, seq_len, tp_size))
            kernels.extend(self._make_gemm(m, "down_proj", batch, seq_len, tp_size))

        return kernels

    # ------------------------------------------------------------------
    # Per-kernel factory helpers
    # ------------------------------------------------------------------

    def _synthetic_ncu(self, entry: SyntheticKernelEntry) -> dict:
        """Convert SyntheticKernelEntry to TracePredictor.build_synthetic_ncu() format."""
        from src.serve_predict.predict_kernel_time_from_trace import TracePredictor
        grid_size = entry.grid[0] * entry.grid[1] * entry.grid[2]
        return TracePredictor.build_synthetic_ncu(
            {"kernel_name": entry.kernel_name,
             "grid": list(entry.grid),
             "block": list(entry.block),
             "shared_mem": entry.shared_mem},
            entry.op_attr,
            grid_size,
            sm_freq_hz=self._hw.nominal_sm_freq_hz,
            dram_freq_hz=self._hw.nominal_dram_freq_hz,
        )

    def _make_norm(self, m: ModelConfig, batch: int, seq_len: int,
                   tp_size: int, suffix: str) -> SyntheticKernelEntry:
        tokens = batch * seq_len
        op_attr = {
            "batch": tokens,
            "hidden": m.hidden_size,
            "dtype_bytes": m.dtype_bytes,
        }
        grid_x = math.ceil(tokens * m.hidden_size / 256)
        return SyntheticKernelEntry(
            kernel_name=f"rms_norm_kernel_{suffix}",
            grid=(max(1, grid_x), 1, 1),
            block=(256, 1, 1),
            shared_mem=0,
            op_type="NORM",
            op_attr=op_attr,
        )

    def _make_rotary(self, m: ModelConfig, batch: int, seq_len: int,
                     tp_size: int) -> SyntheticKernelEntry:
        num_heads = (m.num_attention_heads + m.num_key_value_heads) // tp_size
        op_attr = {
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": m.head_dim,
            "dtype_bytes": m.dtype_bytes,
        }
        tokens = batch * seq_len
        grid_x = math.ceil(tokens * num_heads / 128)
        return SyntheticKernelEntry(
            kernel_name="rotary_embedding_kernel",
            grid=(max(1, grid_x), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="ROTARY",
            op_attr=op_attr,
        )

    def _make_kv_cache(self, m: ModelConfig, batch: int, seq_len: int,
                       tp_size: int) -> SyntheticKernelEntry:
        num_kv_heads = m.num_key_value_heads // tp_size
        op_attr = {
            "seq_len": seq_len,
            "num_kv_heads": num_kv_heads,
            "head_dim": m.head_dim,
            "dtype_bytes": m.dtype_bytes,
        }
        tokens = batch * seq_len
        grid_x = math.ceil(tokens * num_kv_heads / 128)
        return SyntheticKernelEntry(
            kernel_name="reshape_and_cache_kernel",
            grid=(max(1, grid_x), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="KV_CACHE",
            op_attr=op_attr,
        )

    def _make_activation(self, m: ModelConfig, batch: int, seq_len: int,
                          tp_size: int) -> SyntheticKernelEntry:
        tokens = batch * seq_len
        intermediate = m.intermediate_size // tp_size
        op_attr = {
            "batch": tokens,
            "intermediate": intermediate,
            "dtype_bytes": m.dtype_bytes,
        }
        grid_x = math.ceil(tokens * intermediate / 256)
        return SyntheticKernelEntry(
            kernel_name="act_and_mul_kernel",
            grid=(max(1, grid_x), 1, 1),
            block=(256, 1, 1),
            shared_mem=0,
            op_type="ACTIVATION",
            op_attr=op_attr,
        )

    def _make_flash_attn(self, m: ModelConfig, batch: int, seq_len: int,
                         tp_size: int) -> List[SyntheticKernelEntry]:
        hq = m.num_attention_heads // tp_size
        hkv = m.num_key_value_heads // tp_size
        op_attr = {
            "B": batch,
            "S": seq_len,
            "D": m.head_dim,
            "HQ": hq,
            "HKV": hkv,
            "SP": 1,
        }
        # Tile sizes: kBlockM=128, kBlockN=64 for head_dim=128
        kBlockM, kBlockN = 128, 64
        # Grid: per-head per-kBlockM-tile × batch
        grid_x = hq * math.ceil(seq_len / kBlockM) * batch
        smem = 2 * kBlockM * m.head_dim * m.dtype_bytes

        # Use the full template kernel name that _parse_fa_template() can parse
        dtype_str = "cutlass::half_t" if m.dtype_bytes == 2 else "cutlass::bfloat16_t"
        kernel_name = (
            f"void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<"
            f"(int){kBlockM}, (int){kBlockN}, (int){m.head_dim}, (int)4, "
            f"(bool)0, (bool)0, {dtype_str}>, (bool)1>(flash::Flash_fwd_params)"
        )
        kernels = [SyntheticKernelEntry(
            kernel_name=kernel_name,
            grid=(max(1, grid_x), 1, 1),
            block=(128, 1, 1),
            shared_mem=smem,
            op_type="FLASHATTN",
            op_attr=op_attr,
        )]
        return kernels

    def _make_gemm(self, m: ModelConfig, layer_position: str,
                   batch: int, seq_len: int,
                   tp_size: int) -> List[SyntheticKernelEntry]:
        """Build GEMM entry (+ splitKreduce if applicable).

        For decode (seq_len=1), uses internal_gemvx kernel names with a small
        grid so the GEMM predictor's launch-overhead floor dominates, matching
        the actual cuBLAS GEMV dispatch pattern for M=1 decode tokens.
        """
        M, N, K = infer_gemm_dimensions(m, layer_position, seq_len, batch, tp_size)
        op_attr = {"B": batch, "M": M, "N": N, "K": K}
        is_decode = (seq_len == 1)

        if is_decode and M <= 2 * batch:
            # Decode path: cuBLAS dispatches internal_gemvx for small M
            # Grid per cuBLAS emulation: (ceil(N/4), 1, 1), block=(32, 4, 1)
            grid_x = max(1, math.ceil(N / 4))
            kernel_name = f"internal_gemvx_kernel_{layer_position}"
            return [SyntheticKernelEntry(
                kernel_name=kernel_name,
                grid=(grid_x, 1, 1),
                block=(32, 4, 1),
                shared_mem=0,
                op_type="GEMM",
                op_attr=op_attr,
            )]

        tile_M, tile_N = (64, 64) if M <= 64 else (128, 128)
        grid_x = math.ceil(M / tile_M) * math.ceil(N / tile_N)
        kernel_name = (
            f"ampere_fp16_s16816gemm_fp16_{tile_M}x{tile_N}_"
            f"ldg8_f2f_stages_32x5_tn_{layer_position}"
        )

        kernels = [SyntheticKernelEntry(
            kernel_name=kernel_name,
            grid=(max(1, grid_x), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="GEMM",
            op_attr=op_attr,
        )]

        # Emit splitKreduce when K is large and M×N is small (decode regime)
        if K > 2048 and M * N < 8192:
            reduce_grid_x = max(1, math.ceil(M / 32))
            kernels.append(SyntheticKernelEntry(
                kernel_name=f"splitKreduce_kernel_{layer_position}",
                grid=(reduce_grid_x, 1, 1),
                block=(32, 16, 1),
                shared_mem=0,
                op_type="GEMM",
                op_attr=op_attr,
            ))

        return kernels

    def _make_moe_ffn(self, m: ModelConfig, batch: int, seq_len: int,
                      tp_size: int) -> List[SyntheticKernelEntry]:
        """MoE FFN: expert routing REDUCE, token GATHER, fused GEMM, output GATHER."""
        tokens = batch * seq_len
        int_size = m.moe_intermediate_size or m.intermediate_size
        num_experts = m.num_experts
        num_experts_per_tok = m.num_experts_per_tok or 1
        routed_tokens = tokens * num_experts_per_tok

        kernels: List[SyntheticKernelEntry] = []

        # Router logits softmax (REDUCE over expert dimension)
        reduce_elements = tokens * num_experts
        kernels.append(SyntheticKernelEntry(
            kernel_name="at::native::reduce_kernel_moe_router",
            grid=(max(1, math.ceil(tokens / 128)), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="REDUCE",
            op_attr={
                "element_count": reduce_elements,
                "dtype_bytes": m.dtype_bytes,
                "reduction_factor": 6,
            },
        ))

        # Token permutation gather (route tokens to expert buckets)
        kernels.append(SyntheticKernelEntry(
            kernel_name="gather_moe_dispatch",
            grid=(max(1, math.ceil(routed_tokens / 128)), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="GATHER",
            op_attr={
                "element_count": routed_tokens * m.hidden_size,
                "dtype_bytes": m.dtype_bytes,
                "scattered": True,
                "index_count": routed_tokens,
            },
        ))

        # FusedMoE GEMM (grouped expert GEMM: gate+up → SiLU → down)
        # Build a properly-structured moe_tiling dict that _match_moe_projection
        # can parse. Each projection spec contains BM, BN, K (inner), N (outer),
        # grid (expected grid_size), GSM, EM.
        bm, bn = 64, 64
        # w1 (gate+up): tokens×hidden → (2*int_size) per expert, E experts
        # Grid: num_experts × ceil(int_size/BN) [upstream convention per fused_moe.py]
        w1_k = m.hidden_size
        w1_n = 2 * int_size  # fused gate+up output per expert
        w1_n_tiles = math.ceil(int_size / bn)  # vLLM splits by int_size not 2*int_size
        w1_grid = num_experts * w1_n_tiles
        w2_k = int_size  # down proj
        w2_n = m.hidden_size
        w2_n_tiles = math.ceil(w2_n / bn)
        w2_grid = num_experts * w2_n_tiles
        moe_tiling = {
            "w1": {"BM": bm, "BN": bn, "BK": 16, "K": w1_k, "N": w1_n,
                   "grid": w1_grid, "GSM": 1, "EM": num_experts},
            "w2": {"BM": bm, "BN": bn, "BK": 16, "K": w2_k, "N": w2_n,
                   "grid": w2_grid, "GSM": 1, "EM": num_experts},
        }
        moe_op_attr = {
            "TOK": tokens,
            "H": m.hidden_size,
            "INT": int_size,
            "TOPK": num_experts_per_tok,
            "E": num_experts,
            "moe_tiling": moe_tiling,
            "tok_per_expert": max(1, routed_tokens // num_experts),
        }
        kernels.append(SyntheticKernelEntry(
            kernel_name="fused_moe_kernel_grouped",
            grid=(max(1, w1_grid), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="FUSEDMOE",
            op_attr=moe_op_attr,
        ))

        # Output gather (collect expert outputs back to token order)
        kernels.append(SyntheticKernelEntry(
            kernel_name="gather_moe_combine",
            grid=(max(1, math.ceil(routed_tokens / 128)), 1, 1),
            block=(128, 1, 1),
            shared_mem=0,
            op_type="GATHER",
            op_attr={
                "element_count": routed_tokens * m.hidden_size,
                "dtype_bytes": m.dtype_bytes,
                "scattered": True,
                "index_count": routed_tokens,
            },
        ))

        return kernels

    # ------------------------------------------------------------------
    # Prediction core
    # ------------------------------------------------------------------

    def _predict_kernels(self, model_config: ModelConfig,
                         hw: HWSpec,
                         batch: int, seq_len: int,
                         n_decode: int, tp_size: int):
        """Run all kernels through DiffPerformanceFormula.

        Returns list of DiffKernelPrediction objects.
        """
        from src.serve_predict.predict_kernel_time_from_trace import TracePredictor
        trace = self.generate_trace(model_config, batch, seq_len, n_decode, tp_size)
        formula = DiffPerformanceFormula(hw)
        predictions = []
        for entry in trace:
            predictor = create_predictor(entry.op_type)
            kdata = self._synthetic_ncu(entry)
            inp = predictor.forward(kdata, hw, entry.op_attr)
            pred = formula.predict_kernel(inp)
            predictions.append((entry, pred))
        return predictions

    def predict(self, model_config: ModelConfig,
                batch: int = 1, seq_len: int = 128,
                n_decode: int = 0, tp_size: int = 1
                ) -> Tuple[float, Dict[str, float]]:
        """Predict GPU kernel sum latency.

        Args:
            model_config: parsed ModelConfig
            batch: batch size
            seq_len: prefill sequence length
            n_decode: number of decode steps
            tp_size: tensor parallelism factor

        Returns:
            (t_e2e_gpu_ns, per_category_breakdown)
            where t_e2e_gpu_ns is the GPU kernel sum in nanoseconds (no CPU overhead),
            and per_category_breakdown maps op_type to total_ns for that category.

        Raises:
            FileNotFoundError: if hw_config_path is invalid (caught in __init__)
            ValueError: for invalid model_config or tp_size
        """
        predictions = self._predict_kernels(
            model_config, self._hw, batch, seq_len, n_decode, tp_size)

        t_e2e_gpu_ns = 0.0
        breakdown: Dict[str, float] = {}
        for entry, pred in predictions:
            ns = pred.t_total_ns
            t_e2e_gpu_ns += ns
            breakdown[entry.op_type] = breakdown.get(entry.op_type, 0.0) + ns

        return t_e2e_gpu_ns, breakdown

    def predict_diff(self, model_config: ModelConfig,
                     batch: int = 1, seq_len: int = 128,
                     n_decode: int = 0, tp_size: int = 1
                     ) -> Tuple[float, Dict]:
        """Predict GPU kernel sum latency with analytical gradients.

        Args:
            model_config: parsed ModelConfig
            batch: batch size
            seq_len: prefill sequence length
            n_decode: number of decode steps
            tp_size: tensor parallelism factor

        Returns:
            (t_e2e_gpu_ns, agg_grad)
            where agg_grad is the output of aggregate_gradients() containing
            grad_demand[role], grad_capacity[role], grad_eta, etc.

        Notes:
            agg_grad["grad_capacity"]["dram"] < 0 for DRAM-bound workloads
            (more DRAM BW → less time).
        """
        predictions_with_entries = self._predict_kernels(
            model_config, self._hw, batch, seq_len, n_decode, tp_size)

        diff_preds = [pred for _, pred in predictions_with_entries]
        t_e2e_gpu_ns = sum(p.t_total_ns for p in diff_preds)
        agg_grad = aggregate_gradients(diff_preds)

        return t_e2e_gpu_ns, agg_grad

    def predict_diff_with_scaled_hw(self, model_config: ModelConfig,
                                     hw_config_path: str,
                                     dram_bw_scale: float = 1.0,
                                     compute_scale: float = 1.0,
                                     batch: int = 1, seq_len: int = 128,
                                     n_decode: int = 0,
                                     tp_size: int = 1) -> Tuple[float, Dict]:
        """predict_diff() with scaled hardware parameters for DSE.

        Scales are applied by multiplying the effective DRAM and SM clock
        frequencies, which scales the corresponding resource capacities.

        Args:
            dram_bw_scale: multiplier for DRAM bandwidth (>1 → more BW)
            compute_scale: multiplier for compute throughput (>1 → more FLOPS)

        Raises:
            ValueError: if dram_bw_scale <= 0 or compute_scale <= 0
        """
        if dram_bw_scale <= 0:
            raise ValueError(
                f"dram_bw_scale must be > 0, got {dram_bw_scale}")
        if compute_scale <= 0:
            raise ValueError(
                f"compute_scale must be > 0, got {compute_scale}")

        hw = HWSpec(hw_config_path)
        # Scale effective frequencies to modify resource capacities
        # dram_bw_scale → eff_dram_freq = nominal × scale
        # compute_scale → eff_sm_freq = nominal × scale
        eff_dram_freq = hw.nominal_dram_freq_hz * dram_bw_scale
        eff_sm_freq = hw.nominal_sm_freq_hz * compute_scale

        trace = self.generate_trace(model_config, batch, seq_len, n_decode, tp_size)
        formula = DiffPerformanceFormula(hw)
        diff_preds = []
        for entry in trace:
            predictor = create_predictor(entry.op_type)
            kdata = self._synthetic_ncu(entry)
            inp = predictor.forward(kdata, hw, entry.op_attr)
            # Override frequencies to apply scaling
            from dataclasses import replace as _replace
            inp_scaled = _replace(inp,
                                   sm_freq_hz=eff_sm_freq,
                                   dram_freq_hz=eff_dram_freq)
            pred = formula.predict_kernel(inp_scaled)
            diff_preds.append(pred)

        t_e2e_gpu_ns = sum(p.t_total_ns for p in diff_preds)
        agg_grad = aggregate_gradients(diff_preds)
        return t_e2e_gpu_ns, agg_grad


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_simulator(hw_config_path: str) -> ModelSimulator:
    """Create a ModelSimulator for the given hardware config."""
    return ModelSimulator(hw_config_path)
