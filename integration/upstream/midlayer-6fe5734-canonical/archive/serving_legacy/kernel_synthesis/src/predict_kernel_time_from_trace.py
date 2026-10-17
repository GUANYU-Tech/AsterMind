"""
Trace Predictor: predict per-kernel latency from emulation trace + model config.

P18.1: Offline trace annotation with predicted latencies.

Takes a kernel trace (from CUDA emulation) and model configuration,
classifies each kernel, infers dimensions, and predicts latency using
the Midlayer forward model.

Architecture:
  trace entry (name, grid, block, shared_mem)
       ↓
  KernelClassifier → (op_type, dimensions)
       ↓
  Midlayer Forward Predictor → predicted_latency_ns
       ↓
  Aggregation → total prefill time, per-token decode time
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import yaml

from src.serve_predict.model_config import ModelConfig, load_model_config
from src.serve_trace.classifier import classify_kernel, map_category, resolve_category


@dataclass
class KernelAnnotation:
    """Annotated kernel with predicted latency and CPU overhead."""
    index: int
    kernel_name: str
    grid: list
    block: list
    shared_mem: int
    category: str       # GEMM, FLASHATTN, NORM, ROTARY, ACTIVATION, ELEMENTWISE, KV_CACHE, COLLECTIVE, CUSTOM_COMM, SAMPLER, OTHER
    predicted_ns: float
    cpu_overhead_ns: float = 0.0  # CPU framework overhead (wall-clock gap from previous kernel)
    dimensions: dict = None       # inferred M/N/K or S/D/HQ etc.


class TracePredictor:
    """Predict per-kernel latency from trace + model config.

    Supports two prediction modes:
      - "midlayer": full forward model via PerformanceFormula (default)
      - "roofline": simple max(compute, memory) baseline
    """

    VALID_MODES = ("midlayer", "roofline", "pipeweave")

    def __init__(self, hw_config_path: str, model_config: ModelConfig,
                 mode: str = "midlayer", lut=None, moe_tiling_bm_bn_bk=None,
                 pipeweave_model_hw=None):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown prediction mode '{mode}', "
                f"expected one of {self.VALID_MODES}")
        self.mode = mode
        self._pipeweave_model_hw = pipeweave_model_hw
        self.model = model_config
        self._hw_config_path = hw_config_path
        self._is_moe = model_config.num_experts > 0
        self._lut = lut
        # Override MOE tiling (BM, BN, BK) — for G4 real-grid prediction
        self._moe_bm_bn_bk = moe_tiling_bm_bn_bk

        # Load HWSpec eagerly — single source of truth for hardware parameters.
        from src.kernel.hardware_spec import HWSpec
        self._hw_spec = HWSpec(hw_config_path)
        self.clock_hz = self._hw_spec.nominal_sm_freq_hz
        self.dram_clock_hz = self._hw_spec.nominal_dram_freq_hz
        self.launch_overhead_ns = self._hw_spec.dispatch_overhead_ns
        # MetaX mcblas (C500) vs NVIDIA cuBLAS — same gating as
        # AnalyticTraceGenerator._is_maca. The position-matcher dispatch must
        # pass maca=True on C500 so it reproduces mcblas/b16gemvt kernel names
        # (else it emits NVIDIA GEMV/TC names and asserts on C500 shapes like
        # lm_head N=151936 not in the A800 _TILE_LUT).
        try:
            _mem_sub = self._hw_spec.memory_subsystem
        except AttributeError:
            _mem_sub = {}
        self._is_maca = float(_mem_sub.get("mcblas_splitk_reduce_fixed_ns", 0.0)) > 0

        # Lazy-initialized midlayer diff formula (separate from non-diff formula)
        self._midlayer_diff_formula = None

        # Midlayer HWSpec + PerformanceFormula (lazy: constructed on first use)
        self._midlayer_hw = None
        self._midlayer_formula = None

    def _get_midlayer(self):
        """Lazy-construct midlayer PerformanceFormula and predictor cache."""
        if self._midlayer_hw is None:
            from src.kernel.formula import PerformanceFormula
            from src.kernel.validation.legacy_kernel_base import create_predictor
            self._midlayer_hw = self._hw_spec
            self._midlayer_formula = PerformanceFormula(self._midlayer_hw)
            self._midlayer_predictor = {
                "GEMM": create_predictor("GEMM"),
                "FLASHATTN": create_predictor("FLASHATTN"),
                "FUSEDMOE": create_predictor("FUSEDMOE"),
                "NORM": create_predictor("NORM"),
                "ROTARY": create_predictor("ROTARY"),
                "ACTIVATION": create_predictor("ACTIVATION"),
                "KV_CACHE": create_predictor("KV_CACHE"),
                "REDUCE": create_predictor("REDUCE"),
                "GATHER": create_predictor("GATHER"),
                "ELEMENTWISE": create_predictor("ELEMENTWISE"),
                "COLLECTIVE": create_predictor("COLLECTIVE"),
                "CUSTOM_COMM": create_predictor("CUSTOM_COMM"),
                "MEMCPY": create_predictor("MEMCPY"),
                "MOE_SORT": create_predictor("MOE_SORT"),
                "MOE_HISTOGRAM": create_predictor("MOE_HISTOGRAM"),
            }
        return self._midlayer_hw, self._midlayer_formula, self._midlayer_predictor

    def _get_midlayer_diff(self):
        """Lazy-initialize the differentiable midlayer formula.

        Reuses the non-diff formula's HWSpec and predictor cache; only replaces
        PerformanceFormula with DiffPerformanceFormula.
        """
        if self._midlayer_diff_formula is None:
            _, _, _ = self._get_midlayer()  # ensure HWSpec + predictors are loaded
            from src.dse.diff_formula import DiffPerformanceFormula
            self._midlayer_diff_formula = DiffPerformanceFormula(self._midlayer_hw)
        return self._midlayer_hw, self._midlayer_diff_formula, self._midlayer_predictor

    @staticmethod
    def build_synthetic_ncu(entry: dict, op_attr: dict,
                             grid_size: int, sm_freq_hz: float,
                             dram_freq_hz: float, duration_ns: float = 0.0,
                             op_type: str = None) -> dict:
        """Build a minimal synthetic NCU-style kdata from a trace entry + op_attr.

        Constructs the nested Sections/Metadata structure that extract_ncu_metrics()
        expects. Measurement fields (BW, cycles, hit rates) are set to neutral defaults
        since forward mode derives demand from op_attr, not NCU measurements.
        Occupancy fields are intentionally set to 100% / max — the occ_ratio correction
        in compute_wave_eta and fa_serial_floor_ns is computed analytically from
        regs_per_thread + block_size + grid_size + hw_spec, not from NCU measurements.

        op_type: when set (e.g. "cube" for an Ascend GroupedMatmul), stamped into
        Metadata."Op Type" so extract_ncu_metrics yields a non-None op_type and the
        predictor takes its Ascend branch. Omitted (None) on GPU → op_type stays
        None and behavior is unchanged.
        """
        grid_x, grid_y, grid_z = entry["grid"]
        block_flat = entry["block"][0] * entry["block"][1] * entry["block"][2]
        grid_str = f"({grid_x}, {grid_y}, {grid_z})"
        shared_mem = entry["shared_mem"]
        # Phase 2 launch-tile: carried top-level so extract_ncu_metrics surfaces
        # it as ncu["tile"] for the time-modeling layer to read (no name regex).
        # None when the emitter hasn't filled it (legacy) → gemm.py name-parse.
        tile = entry.get("tile")

        metadata = {
            "Kernel Name": entry["kernel_name"],
            "Grid Size": grid_str,
        }
        if op_type is not None:
            metadata["Op Type"] = op_type

        return {
            "Metadata": metadata,
            "Sections": {
                "GPU Speed Of Light Throughput": {"Header": {
                    "Duration [nsecond]": str(duration_ns),
                    "SM Frequency [cycle/second]": str(sm_freq_hz),
                    "DRAM Frequency [cycle/second]": str(dram_freq_hz),
                    "Elapsed Cycles [cycle]": "1000",
                    "SM Active Cycles [cycle]": "1000",
                }},
                "Compute Workload Analysis": {
                    "Pipe Utilization (% of active cycles)": {
                        "Tensor (All) [%]": "100",
                        "FMA [%]": "0",
                        "ALU [%]": "0",
                    },
                    "Pipe Utilization (% of peak instructions executed)": {
                        "XU [%]": "0",
                    },
                },
                "Memory Workload Analysis Chart": {"Memory Chart": {
                    "l1tex__m_xbar2l1tex_read_bytes.sum [byte]": "0",
                    "l1tex__m_l1tex2xbar_write_bytes.sum [byte]": "0",
                    "dram__bytes_read.sum [byte]": "0",
                    "dram__bytes_write.sum [byte]": "0",
                    "l1tex__t_sector_hit_rate.pct [%]": "0",
                    "lts__t_sector_hit_rate.pct [%]": "0",
                    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum [byte]": "0",
                }},
                "Launch Statistics": {"Header": {
                    "Block Size": str(block_flat),
                    "Registers Per Thread [register/thread]": "128",
                    "Dynamic Shared Memory Per Block [byte/block]": str(shared_mem),
                    "Static Shared Memory Per Block [byte/block]": "0",
                    "Driver Shared Memory Per Block [byte/block]": "0",
                    "Grid Size": grid_str,
                    "Waves Per SM": "1",
                    "Theoretical Occupancy [%]": "100",
                    "Achieved Occupancy [%]": "100",
                }},
                "Occupancy": {"Header": {
                    "Block Limit SM [block]": "32",
                    "Block Limit Registers [block]": "4",
                    "Block Limit Shared Mem [block]": "4" if shared_mem > 0 else "32",
                    "Block Limit Warps [block]": "8",
                    "Theoretical Active Warps per SM [warp]": "32",
                    "Theoretical Occupancy [%]": "100",
                    "Achieved Occupancy [%]": "100",
                    "Achieved Active Warps Per SM [warp]": "32",
                }},
                "Memory Workload Analysis Tables": {"Device Memory": {
                    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed [%]": "0",
                    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed [%]": "0",
                }},
            },
            "tile": tile,
        }

    def _predict_kernel(self, entry: dict, cat: str, seq_len: int,
                         batch_size: int, tp_size: int) -> float:
        """Predict latency for a single kernel using self.mode.

        Calls predictor.predict(mode=self.mode): either the full midlayer
        forward model or the naive roofline baseline.
        Raises on failure — no silent fallback.
        """
        # MoE routing signature: fused_moe kernels share an identical grid
        # across EP ranks (grid is derived from total tokens, not per-rank
        # routing), so the routing histogram must enter the cache key or every
        # rank collides on rank0's cached value. None for non-MoE kernels.
        routing_sig = entry.get("tok_per_expert")
        # Fixed-grid GEMM disambiguation (Ascend/CANN). On GPU, distinct GEMM
        # shapes get distinct kernel names (ampere_..._64x64) or grids (grid
        # encodes the tile count), so (kernel_name, grid, block) uniquely keys
        # the shape. On Ascend, ALL dense projections (qkv/o/gate_up/down) run
        # as the SAME aclnnMatmul_* op on the SAME fixed Block-Dim grid
        # (=num_compute_units), so within one step they collapse to a single
        # LUT key and every projection returns the first one's cached time
        # (~1.8× GEMM under-prediction at high batch, where the shape mix is
        # widest). Fold the carried (M,N,K) into the cache key so each
        # projection keeps its own prediction. Scoped by CANN matmul name so
        # GPU keys (already shape-unique) are unchanged — not a card-name gate.
        if cat in ("GEMM", "SPLITK"):
            _kn = entry.get("kernel_name", "").lower()
            _oa = entry.get("op_attr")
            if (_oa and all(k in _oa for k in ("M", "N", "K"))
                    and any(s in _kn for s in (
                        "aclnnmatmul", "matmulv2", "matmulv3",
                        "batchmatmul", "ppmatmul", "matmulcommon"))):
                _shape = ("mnk", _oa["M"], _oa["N"], _oa["K"])
                routing_sig = (_shape if routing_sig is None
                               else (tuple(routing_sig), _shape))
        # Fixed-grid FA disambiguation (Ascend/CANN). FusedInferAttentionScore
        # runs on the same fixed Block-Dim grid for every step, so distinct KV
        # cache lengths (the per-request decode context, which drives the
        # KV-load floor) collide on one LUT key and every decode FA returns the
        # first step's short-KV time (~4us) → FA under-predicts ~4-9× across the
        # long-context steps. Fold (S, KV) into the cache key. Same collision
        # and fix as the GEMM case above; scoped to the CANN FA op name.
        if cat == "FLASHATTN":
            _oa = entry.get("op_attr")
            if _oa is not None:
                _fa_shape = ("fa", _oa.get("S"), _oa.get("KV"),
                             _oa.get("SP"))
                routing_sig = (_fa_shape if routing_sig is None
                               else (tuple(routing_sig)
                                     if isinstance(routing_sig, (list, tuple))
                                     else routing_sig, _fa_shape))
        if self._lut is not None:
            cached = self._lut.get(
                entry["kernel_name"], entry["grid"], entry["block"],
                entry["shared_mem"], cat, seq_len, batch_size, tp_size,
                routing_sig=routing_sig)
            if cached is not None:
                return cached

        ml_hw, ml_formula, ml_predictors = self._get_midlayer()
        predictor = ml_predictors.get(cat)
        if predictor is None:
            raise ValueError(
                f"No predictor registered for category '{cat}'")

        op_attr = self._build_op_attr(entry, cat, seq_len, batch_size, tp_size)
        if op_attr is None:
            raise ValueError(
                f"Cannot build op_attr for category '{cat}', "
                f"kernel '{entry['kernel_name']}'")

        grid_size = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
        # Ascend GroupedMatmul: stamp Op Type="cube" so extract_ncu_metrics yields
        # op_type and the MoE predictor takes the Ascend single-projection branch
        # (weight/1.35TB·s on the fixed 24-AIC-core grid), NOT the Triton grid-based
        # predict_moe_traffic. Keyed on hw (aic_aiv_separated) + kernel name, not
        # card name. GPU keeps op_type=None → unchanged behavior.
        ascend_grouped = (getattr(ml_hw, "aic_aiv_separated", False)
                          and "grouped_matmul" in entry["kernel_name"].lower())
        # Ascend MoE aux (AIV vector) kernels: stamp op_type="vector" so
        # is_non_tensor → is_nt → _moe_nt_analytical's Ascend branch (measured
        # traffic formulas for SwiGlu/MoeInitRouting/MoeGatingTopK/
        # MoeTokenUnpermute/MoeFinalizeRouting/MoeComputeExpertTokens). Emitted
        # by analytic_mirror._moe_ffn_kernels for aic_aiv_separated hw.
        _kl = entry["kernel_name"].lower()
        ascend_aux = (getattr(ml_hw, "aic_aiv_separated", False) and any(
            s in _kl for s in ("swiglu", "moeinitrouting", "moegatingtopk",
                               "moetokenunpermute", "moefinalizerouting",
                               "moecomputeexperttokens")))
        _synth_op_type = ("vector" if ascend_aux
                          else "cube" if ascend_grouped else None)
        synthetic = self.build_synthetic_ncu(
            entry, op_attr, grid_size,
            sm_freq_hz=self.clock_hz,
            dram_freq_hz=self.dram_clock_hz,
            op_type=_synth_op_type,
        )
        if self.mode == "midlayer":
            predict_mode = "forward"
        elif self.mode == "roofline":
            predict_mode = "roofline"
        elif self.mode == "pipeweave":
            predict_mode = "pipeweave"
        else:
            raise ValueError(f"Unknown prediction mode '{self.mode}'")
        result = predictor.predict(
            synthetic, ml_hw, op_attr,
            formula=ml_formula, mode=predict_mode,
            pipeweave_model_hw=self._pipeweave_model_hw)

        if self._lut is not None:
            self._lut.put(
                entry["kernel_name"], entry["grid"], entry["block"],
                entry["shared_mem"], cat, seq_len, batch_size, tp_size,
                result, routing_sig=routing_sig)

        return result

    def _build_op_attr(self, entry, cat, seq_len, batch_size, tp_size):
        """Build op_attr dict for a given kernel category.

        Returns None for unsupported categories.
        """
        if cat == "GEMM":
            # mcblas split-K reduce (DevDstElementwiseAdd) is a GEMM non-tensor
            # kernel whose duration is an execution floor (dispatch + fixed +
            # per_wave × waves), computed inside GEMMPredictor.forward from the
            # real grid (carried via build_synthetic_ncu). Its traffic is
            # negligible, so feed minimal M=N=K=1 (avoids _match_gemm_position,
            # which only matches real projection GEMMs). The floor formula lives
            # ONLY in GEMMPredictor — no duplicate in the serving layer.
            if ("elementwiseadd" in entry.get("kernel_name", "").lower()
                    or "splitkreduce" in entry.get("kernel_name", "").lower()):
                return {"B": 1, "M": 1, "K": 1, "N": 1}
            # Ascend (vllm-ascend) dense projections (aclnnMatmul_* / MatMulV2 /
            # MatMulV3 / BatchMatMul / PpMatmul) carry their (B,M,N,K) verbatim on
            # the trace entry's op_attr — the recipe stamps it from the operator
            # record (analytic_mirror._proj_gemm ascend branch). Prefer it, exactly
            # as the FLASHATTN branch below prefers the carried FA op_attr. The
            # CANN matmul name has no parseable cuBLAS tile and runs on the fixed
            # 24-AIC-core grid, so _match_gemm_position (which replays the A800
            # cuBLAS-tile dispatch → needs gemm_kernel_lut_a800.json + the cuBLAS
            # grid contract) does NOT apply and would raise. Scoped by kernel NAME
            # (the aclnn/CANN matmul families), not card name; GPU kernel names
            # (ampere_*/cutlass/s16816/gemvx) keep the _match_gemm_position path
            # verbatim — including its lm_head-as-M=1 sampling handling.
            _kn = entry.get("kernel_name", "").lower()
            _is_cann_matmul = any(s in _kn for s in (
                "aclnnmatmul", "matmulv2", "matmulv3", "batchmatmul",
                "ppmatmul", "matmulcommon"))
            carried = entry.get("op_attr")
            if (_is_cann_matmul and carried
                    and all(k in carried for k in ("M", "N", "K"))):
                return {"B": 1, "M": carried["M"], "K": carried["K"],
                        "N": carried["N"]}
            pos, M, N, K = self._match_gemm_position(entry, seq_len, batch_size, tp_size)
            # _match_gemm_position returns M already multiplied by batch_size.
            # Set B=1 so GEMMPredictor doesn't double-count via B * M.
            op = {"B": 1, "M": M, "K": K, "N": N}
            # Phase 2 T-D: carry has_bias + dtype from the emit layer's op_attr
            # (operator_records stamps them) so forward()'s TileStrategy.resolve()
            # can key the EXACT 5-wide lut4 (== _select_gemm_kernel's key). Without
            # this, resolve defaults has_bias=False/dtype=bf16 and name-mode's
            # _select_gemm_kernel gets dtype=None → fp16-default fallback, and the
            # two paths diverge on the 26% prefill shapes gemm_kernel_lut doesn't
            # cover (Qwen3-4B G1 13.03→13.34 spurious regression).
            carried = entry.get("op_attr") or {}
            if "has_bias" in carried:
                op["has_bias"] = carried["has_bias"]
            if "dtype" in carried:
                op["dtype"] = carried["dtype"]
            return op
        elif cat == "FLASHATTN":
            # The recipe stamps the FA op_attr (incl. the per-step KV cache
            # length KV) onto the trace entry. Prefer it verbatim so the
            # reconstructed KV rides through — otherwise decode FA would model
            # the K-loop / KV traffic from query_len (1) and under-predict by
            # the context-length factor.
            carried = entry.get("op_attr")
            if carried and "HQ" in carried:
                op = dict(carried)
                # SP is stamped by the recipe as num_kv_splits (= ceil(KV/kBN)),
                # sourced from nsys. Both the compute kernel (per-CTA K-loop=1,
                # set in flash_attn.forward k_steps) and the combine kernel (its
                # serial reduce floor scales with SP) use it — do NOT override.
                #
                # The recipe stamps S (query len) + KV (attended cache len), but
                # the FA forward model (predict_fa_traffic / _compute_fa_flops /
                # serial K-loop / the Ascend pipeline floor) reads the unified
                # Q_LEN/KV_LEN names. Without them, forward() raises KeyError on
                # EVERY serving FA kernel → the prediction silently falls back to
                # launch-overhead → FA -89% (the whole G1 FLASHATTN gap). Map
                # S→Q_LEN, KV→KV_LEN here so the verified traffic+latency model
                # runs. Offline traces already stamp Q_LEN/KV_LEN (untouched).
                op.setdefault("Q_LEN", op.get("S", 1))
                op.setdefault("KV_LEN", op.get("KV", op.get("S", 1)))
                return op
            # The split-KV combine epilogue reduces SP partial results in a
            # serial chain; its pipeline floor (fa_combine_*) only engages when
            # SP>1. The compute kernel is insensitive to SP here (S/HQ drive it),
            # so use SP=2 for combine kernels and SP=1 otherwise.
            is_combine = "combine" in entry.get("kernel_name", "").lower()
            return {
                "B": batch_size,
                "S": seq_len,
                "D": self.model.head_dim,
                "HQ": self.model.num_attention_heads // tp_size,
                "HKV": self.model.num_key_value_heads // tp_size,
                "SP": 2 if is_combine else 1,
            }
        elif cat == "FUSEDMOE":
            if self.model.num_experts == 0:
                return None  # dense model; no MOE prediction
            int_size = self.model.moe_intermediate_size or self.model.intermediate_size
            E = self.model.num_experts
            topk = self.model.num_experts_per_tok or 1
            # EP: each rank owns a local expert block. The regen
            # (_regenerate_trace_with_step_id) stamps moe_ep_size +
            # moe_local_experts on MoE entries and slices the captured global
            # routing bincount to the local block. ep defaults to 1 for
            # non-EP / un-stamped entries.
            ep = entry.get("moe_ep_size", 1)
            E_local = entry.get("moe_local_experts", E) if ep > 1 else E
            captured = entry.get("tok_per_expert")
            # REAL ROUTING IS MANDATORY — no uniform-distribution fallback.
            # The trace MUST carry the real per-expert bincount captured by
            # _operator_patch._patch_routing_capture (hooked into vLLM's
            # BaseRouter.select_experts on GPU and vllm_ascend's
            # experts_selector.select_experts on Ascend). A missing histogram
            # means the capture hook didn't fire (e.g. an unsupported vLLM /
            # vllm_ascend version, or a non-vLLM MoE path) — predict nothing
            # for this kernel rather than silently fabricating a uniform
            # distribution, which previously under-predicted MoE ~10× and
            # masked the error behind a plausible-looking G1. See
            # docs/moe_tpe_format.md.
            if captured is None:
                import os as _os
                if _os.environ.get("MIDLAYER_DEBUG_MOE_TPE"):
                    import sys as _sys
                    print("[MOE_TPE_DEBUG] offending FUSEDMOE entry: "
                          f"kernel_name={entry.get('kernel_name')!r} "
                          f"raw_cat={classify_kernel(entry)!r} "
                          f"step_id={entry.get('step_id')} "
                          f"grid={entry.get('grid')} "
                          f"moe_ep_size={entry.get('moe_ep_size')} "
                          f"keys={sorted(entry.keys())}", file=_sys.stderr)
                    _sys.stderr.flush()
                raise ValueError(
                    "FUSEDMOE entry has no tok_per_expert — real routing "
                    "histogram is required (uniform fallback is forbidden). "
                    "Re-capture the serving trace with the current "
                    "_operator_patch routing hook (vllm 0.18 BaseRouter."
                    "select_experts / vllm_ascend experts_selector."
                    "select_experts). See docs/moe_tpe_format.md.")
            # Per-expert bincount (already sliced to LOCAL experts by the regen
            # when ep>1). Pad/truncate to E_local.
            tpe_local = list(captured[:E_local]) + [0] * max(0, E_local - len(captured))
            # routed_pairs = this rank's total token-expert pairs.
            routed_pairs = sum(tpe_local)
            # tok_per_expert is PER-EXPERT (the single canonical format — see
            # docs/moe_tpe_format.md). The MoE forward (Ascend GroupedMatmul
            # AND Triton predict_moe_traffic) consumes it as per-expert:
            # rows = sum(tpe), counts = tpe, e_active = count(tpe>0). No
            # cumulative conversion (the 910B dataset was regenerated to
            # per-expert too, so single-op and serving share one contract).
            tok_per_expert = tpe_local
            # Build moe_tiling from grid dimensions and known intermediate geometry.
            # w1: N = 2*INT, w2: N = H (swapped since gate_up is bundled).
            grid_size = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
            tok = batch_size * seq_len
            if self._moe_bm_bn_bk is not None:
                BM, BN, BK = self._moe_bm_bn_bk
                GSM = 1
            else:
                # vLLM 0.11.0 (A800 serving) get_default_config returns a
                # 2-tier flat config, NOT the tiered config from newer vLLM
                # source. Verified by calling get_default_config in the
                # r2r_qwen3_next env: M<=96 → BM=16,BN=32,BK=64,GSM=1;
                # M>=512 → BM=64,BN=64,BK=32,GSM=8. The tiered port (commit
                # 25ab4fe) was based on a newer vLLM source tree that does
                # NOT match the serving vLLM version. Reverted to match.
                # See C500 revert be97d79 for the same issue.
                if tok <= E_local:
                    BM, BN, BK = 16, 32, 64
                    GSM = 1
                else:
                    BM, BN, BK = 64, 64, 32
                    GSM = 8
            w1_N = 2 * int_size
            w2_N = self.model.hidden_size
            n_tiles_w1 = math.ceil(w1_N / BN)
            n_tiles_w2 = math.ceil(w2_N / BN)
            # EM from vLLM formula (matches invoke_fused_moe_triton_kernel):
            #   max_sorted = routed_pairs + E_local*(BM-1)
            #   EM_w1 = min(max_sorted, routed_pairs*BM) if tok < BM else max_sorted
            #   EM_w2 = min(max_sorted, routed_pairs*BM) if routed_pairs < BM else max_sorted
            max_sorted = routed_pairs + E_local * (BM - 1)
            EM_w1 = min(max_sorted, routed_pairs * BM) if tok < BM else max_sorted
            EM_w2 = min(max_sorted, routed_pairs * BM) if routed_pairs < BM else max_sorted
            grid_w1 = math.ceil(EM_w1 / BM) * n_tiles_w1
            grid_w2 = math.ceil(EM_w2 / BM) * n_tiles_w2
            moe_tiling = {
                "w1": {"BM": BM, "BN": BN, "BK": BK, "GSM": GSM,
                        "EM": EM_w1, "K": self.model.hidden_size, "N": w1_N,
                        "grid": grid_w1},
                "w2": {"BM": BM, "BN": BN, "BK": BK, "GSM": GSM,
                        "EM": EM_w2, "K": int_size, "N": w2_N,
                        "grid": grid_w2},
            }
            return {
                "TOK": batch_size * seq_len,
                "H": self.model.hidden_size,
                "INT": int_size,
                "TOPK": self.model.num_experts_per_tok,
                "E": E_local,
                "moe_tiling": moe_tiling,
                "tok_per_expert": tok_per_expert,
            }
        elif cat == "NORM":
            return {
                "batch": batch_size * seq_len,
                "hidden": self.model.hidden_size,
                "dtype_bytes": self.model.dtype_bytes,
            }
        elif cat == "ROTARY":
            return {
                "seq_len": seq_len,
                "num_heads": (self.model.num_attention_heads +
                              self.model.num_key_value_heads) // tp_size,
                "head_dim": self.model.head_dim,
                "dtype_bytes": self.model.dtype_bytes,
            }
        elif cat == "ACTIVATION":
            # act_and_mul: SiLU(gate) * up on the gate_up projection output.
            # MoE (act_and_mul_kernel_moe): operates on the grouped expert output —
            # rows = routed token-expert pairs on THIS rank, width = moe_intermediate
            # (NOT sharded by TP under EP; EP shards the expert count instead).
            # Dense: rows = tokens, width = intermediate_size // tp (column-parallel
            # gate_up output is TP-sharded).
            kname = entry.get("kernel_name", "").lower()
            if self.model.num_experts > 0 and "moe" in kname:
                moe_int = self.model.moe_intermediate_size or self.model.intermediate_size
                captured = entry.get("tok_per_expert")
                if entry.get("moe_ep_size", 1) > 1 and captured is not None:
                    rows = sum(captured)
                else:
                    rows = (batch_size * seq_len
                            * max(1, self.model.num_experts_per_tok))
                return {
                    "batch": rows,
                    "intermediate": moe_int,
                    "dtype_bytes": self.model.dtype_bytes,
                }
            return {
                "batch": batch_size * seq_len,
                "intermediate": self.model.intermediate_size // tp_size,
                "dtype_bytes": self.model.dtype_bytes,
            }
        elif cat == "KV_CACHE":
            return {
                "seq_len": seq_len,
                "num_kv_heads": self.model.num_key_value_heads // tp_size,
                "head_dim": self.model.head_dim,
                "dtype_bytes": self.model.dtype_bytes,
            }
        elif cat == "REDUCE":
            grid_size_local = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
            block_flat = entry["block"][0] * entry["block"][1] * entry["block"][2]
            element_count = grid_size_local * block_flat
            return {
                "element_count": element_count,
                "dtype_bytes": self.model.dtype_bytes,
            }
        elif cat == "GATHER":
            grid_size_local = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
            block_flat = entry["block"][0] * entry["block"][1] * entry["block"][2]
            element_count = grid_size_local * block_flat
            return {
                "element_count": element_count,
                "dtype_bytes": self.model.dtype_bytes,
                "scattered": True,
            }
        elif cat == "ELEMENTWISE":
            # Recipe-emitted framework glue (moe_align Fill, fused_topk div/cast)
            # carries an exact op_attr (element_count + dtype + reads/writes +
            # in-stream floor_ns) onto the trace entry (operator_records preserves
            # it). Prefer it verbatim so the GPU-measured floor rides through.
            carried = entry.get("op_attr")
            if carried and "element_count" in carried:
                return dict(carried)
            # vectorized_elementwise_kernel: each thread processes a vector of
            # elements. element_count ≈ grid×block×4 (×4 = fp16 128-bit vectorized
            # load width). This is an estimate — true element count is not in the
            # trace. Binary-op traffic factor (2 read + 1 write) is the predictor default.
            grid_size_local = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
            block_flat = entry["block"][0] * entry["block"][1] * entry["block"][2]
            element_count = grid_size_local * block_flat * 4
            return {
                "element_count": element_count,
                "dtype_bytes": self.model.dtype_bytes,
            }
        elif cat == "COLLECTIVE":
            # NCCL all-reduce/all-gather: msg_bytes + nranks are carried on the
            # emitted entry's op_attr (operator_records lm_head AllGather,
            # analytic_mirror per-layer AllReduce). Prefer it verbatim — the
            # block/grid here are the kernel's REAL CUDA launch params
            # (block=(nThreads,1,1), grid=(nBlocks,1,1)), NOT the split-encoding
            # below: block[1]=1 (block y-dim) misread as high-32 msg bits would
            # inflate msg to ~4GB and predict ~80ms/call (~447× real).
            carried = entry.get("op_attr")
            if carried and "msg_bytes" in carried:
                return {"msg_bytes": carried["msg_bytes"],
                        "nranks": carried.get("nranks", entry["grid"][0])}
            # Fallback: EP all2all uses the genuine split-encoding
            # (block[0]=msg_low, block[1]=msg_high=0 for <4GB, grid[0]=nranks).
            msg_bytes = entry["block"][0] + (entry["block"][1] << 32)
            return {"msg_bytes": msg_bytes, "nranks": entry["grid"][0]}
        elif cat == "CUSTOM_COMM":
            grid_total = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
            return {"grid_total": grid_total}
        elif cat in ("MEMCPY", "MEMCPY_DTOD", "MEMCPY_DTOH", "MEMCPY_HTOD"):
            # Direction from the kernel name (DTOD/DTOH/HTOD); bytes carried on
            # the emitted entry's op_attr (model-config estimate at emission).
            kname = entry.get("kernel_name", "")
            if "DTOD" in kname:
                direction = "DTOD"
            elif "HTOD" in kname:
                direction = "HTOD"
            else:
                direction = "DTOH"
            src_attr = entry.get("op_attr", {})
            return {"direction": direction, "bytes": src_attr.get("bytes", 0)}
        elif cat in ("MOE_SORT", "MOE_HISTOGRAM"):
            src_attr = entry.get("op_attr", {})
            return {
                "element_count": src_attr.get("element_count", 0),
                "dtype_bytes": src_attr.get("dtype_bytes", 4),
                # radix sort touches keys+values over multiple passes; histogram
                # is a single read+count pass.
                "passes": 4 if cat == "MOE_SORT" else 1,
            }
        return None

    def _predict_with_midlayer_diff(self, entry, cat, seq_len, batch_size, tp_size,
                                  sm_freq_override=None, dram_freq_override=None):
        """Predict latency and return DiffKernelPrediction for DSE.

        Uses DiffPerformanceFormula instead of PerformanceFormula and preserves
        the full gradient-carrying prediction object for downstream DSE
        aggregation.

        Optional freq overrides allow DSE scaling: predictors emit
        PredictionInput with hw.nominal_freq_hz, but callers can override
        sm_freq_hz / dram_freq_hz before DiffPerformanceFormula sees them.

        Raises on failure — no silent fallback.
        """
        ml_hw, ml_formula, ml_predictors = self._get_midlayer_diff()
        predictor = ml_predictors.get(cat)
        if predictor is None:
            raise ValueError(
                f"No diff predictor registered for category '{cat}'")

        op_attr = self._build_op_attr(entry, cat, seq_len, batch_size, tp_size)
        if op_attr is None:
            raise ValueError(
                f"Cannot build op_attr for diff category '{cat}', "
                f"kernel '{entry['kernel_name']}'")

        grid_size = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
        synthetic = self.build_synthetic_ncu(
            entry, op_attr, grid_size,
            sm_freq_hz=self.clock_hz,
            dram_freq_hz=self.dram_clock_hz,
        )
        inp = predictor.forward(synthetic, ml_hw, op_attr)
        # predictor.forward() called directly (not predict()) because DSE needs
        # the full PredictionInput for freq override + gradient-carrying DiffFormula.
        if sm_freq_override is not None:
            inp.sm_freq_hz = sm_freq_override
        if dram_freq_override is not None:
            inp.dram_freq_hz = dram_freq_override
        return ml_formula.predict_kernel(inp)

    def _match_gemm_position_via_dispatch(self, entry: dict, seq_len: int,
                                            batch_size: int, tp_size: int) -> Tuple[str, int, int, int]:
        """Match GEMM position by simulating cuBLAS dispatch for each candidate.

        Used for GEMV and other kernels whose tile dims cannot be parsed from
        the kernel name.  Calls _cublas_gemm_dispatch for each candidate
        position and picks the one whose output matches the observed kernel
        name and grid.
        """
        from src.serve_trace.analytic_mirror import (
            _cublas_gemm_dispatch, _maca_lmhead_dispatch)
        grid_x, grid_y, grid_z = entry["grid"]
        kernel_name = entry["kernel_name"]
        # EP degree for the lm_head (tp,ep) LUT — the regen stamps moe_ep_size.
        ep = entry.get("moe_ep_size", 1)

        positions = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"]
        if self._is_moe:
            positions.append("router")

        best = None
        best_score = float("inf")
        # GEMV kernels (b16gemvt / gemvx / gemv2T) are ALWAYS M=1 (a single-token
        # matrix-vector op: decode batch=1, OR the lm_head sampling the last token
        # of a prefill step). infer_gemm_dimensions returns M=seq*batch for
        # lm_head at prefill (e.g. 729), but the real GEMV is M=1, so dispatching
        # with the infer M would NOT emit the GEMV name -> no position matches.
        # Force M=1 when the observed kernel is a GEMV.
        is_gemv = "gemv" in kernel_name.lower()
        for pos in positions:
            try:
                pos_M, pos_N, pos_K = self.infer_gemm_dimensions(pos, seq_len, batch_size, tp_size)
            except (ValueError, AssertionError):
                continue
            # C500 lm_head: family + sharding vary by (tp, ep); route to the
            # lm_head dispatch (pos_N = full vocab; lm_head is always M=1).
            if self._is_maca and pos == "lm_head":
                cand = _maca_lmhead_dispatch(pos_N, tp_size, ep)
            else:
                if is_gemv:
                    pos_M = 1
                cb_m, cb_n, cb_k = pos_N, pos_M, pos_K
                cand = _cublas_gemm_dispatch(cb_m, cb_n, cb_k, maca=self._is_maca)
            for name, gx, gy, gz, bx, by, _smem in cand:
                if name == kernel_name and gx == grid_x and gy == grid_y:
                    best = (pos, pos_M, pos_N, pos_K)
                    best_score = 0
                    break
                if name == kernel_name:
                    score = abs(gx * gy - grid_x * grid_y)
                    if score < best_score:
                        best_score = score
                        best = (pos, pos_M, pos_N, pos_K)

        if best is None:
            # DIRECT-path fallback (Phase 2 T4/T5): the logical `linear` record
            # already carries the true (M,N,K) on the entry op_attr (recipe
            # _op_to_kernels line ~636), so when the fabricated cutlass/cuBLAS
            # kernel NAME can't be reproduced by re-simulating dispatch across the
            # candidate positions (e.g. the bf16 wmma s161616gemm variant chosen
            # for a small-M MoE projection — emit-side and price-side dispatch
            # diverge), price directly from the carried logical dims instead of
            # raising. Grid-reverse-matching is only needed because the SYNTHESIS
            # path throws away the op's logical identity; the direct path never
            # should have to. Dense is unaffected — its GEMMs grid-match cleanly
            # and never reach this fallback. Position is diagnostic only (pricing
            # uses M/N/K), so tag it "logical".
            carried = entry.get("op_attr")
            if carried and all(k in carried for k in ("M", "N", "K")):
                return ("logical", carried["M"], carried["N"], carried["K"])
            raise ValueError(
                f"GEMM dimension inference: kernel '{kernel_name}' grid=({grid_x},{grid_y},{grid_z}) "
                f"does not match any expected layer projection via dispatch. "
                f"seq_len={seq_len} batch={batch_size} tp={tp_size}"
            )
        return best

    def _match_gemm_position(self, entry: dict, seq_len: int,
                               batch_size: int, tp_size: int) -> Tuple[str, int, int, int]:
        """Content-driven GEMM layer position matching.

        Derives the layer position (qkv_proj / o_proj / gate_up_proj / down_proj / lm_head)
        from the kernel's grid dimensions and the model config, without relying on
        sequential position counting. Raises ValueError if no candidate matches.

        Strategy: for each candidate position, compute expected grid tile count from
        the known M/N/K and tile dims parsed from the kernel name. cuBLAS may use
        1D (grid_x = total tiles, grid_y = 1) or 2D (grid_x = N-tiles, grid_y = M-tiles)
        grids, so we match against grid_x * grid_y (total tiles, excluding split-K in
        grid_z). Pick the candidate whose ratio is closest to 1.0.

        For GEMV kernels (no parseable tile dims), uses cuBLAS dispatch
        simulation to match grid directly.
        """
        from src.kernel.validation.legacy_ops.gemm import _parse_tile_dims as _ptd
        grid_x, grid_y, grid_z = entry["grid"]
        kernel_name = entry["kernel_name"]
        tile_M, tile_N, tile_K = _ptd(kernel_name)

        if tile_M is None or tile_N is None:
            return self._match_gemm_position_via_dispatch(
                entry, seq_len, batch_size, tp_size)

        observed_tiles = grid_x * grid_y

        positions = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head"]
        if self._is_moe:
            positions.append("router")

        M_values = [batch_size * seq_len]
        if batch_size > 1:
            M_values.append(batch_size)
        # vLLM computes lm_head only for sample positions (M=1 per request),
        # not all tokens — try M=1 so lm_head can match during prefill.
        if batch_size * seq_len > 1:
            M_values.append(1)
        # The grid itself implies an M (token) count: grid_y is the M-tile axis
        # (dispatch grid_y = ceil(tokens / tile_M)), so grid_y * tile_M is the
        # M the kernel actually ran with. vLLM's lm_head / sampled-projection M
        # (number of sampled positions, chunked-prefill sample counts) is often
        # none of the values above — without this candidate the matcher falls to
        # the dispatch fallback and raises (e.g. MoE TP4 lm_head grid_y=2 →
        # M=128 not in [seq*batch, batch, 1]). The grid-implied M always matches
        # observed_tiles exactly (ratio 1.0) for the position whose N fits
        # grid_x, so it is only selected when it is in fact the right M.
        if tile_M and tile_M > 0 and grid_y > 0:
            M_values.append(grid_y * tile_M)

        best_overall = None

        for m_val in M_values:
            candidates = {}
            for pos in positions:
                try:
                    pos_M, pos_N, pos_K = self.infer_gemm_dimensions(
                        pos, 1, m_val, tp_size)
                except (ValueError, AssertionError):
                    continue
                expected_tiles = (
                    ((pos_M + tile_M - 1) // tile_M)
                    * ((pos_N + tile_N - 1) // tile_N)
                )
                if expected_tiles > 0:
                    ratio = observed_tiles / expected_tiles
                    if best_overall is None or abs(ratio - 1.0) < abs(best_overall[1] - 1.0):
                        best_overall = (pos, ratio, pos_M, pos_N, pos_K)

        if best_overall is None:
            raise ValueError(
                f"GEMM dimension inference: no valid candidates for kernel "
                f"'{kernel_name}' grid=({grid_x},{grid_y},{grid_z}) "
                f"seq_len={seq_len} batch={batch_size} tp={tp_size}"
            )

        best_pos, best_ratio, pos_M, pos_N, pos_K = best_overall

        if best_ratio > 4.0 or best_ratio < 0.25:
            M_cur = max(M_values[0], 1)
            if M_cur <= tile_M:
                estimated_n = grid_x * tile_N
                n_fallback = None
                n_fallback_delta = float("inf")
                for m_val in M_values:
                    for pos in positions:
                        try:
                            cM, cN, cK = self.infer_gemm_dimensions(pos, 1, m_val, tp_size)
                        except (ValueError, AssertionError):
                            continue
                        delta = abs(cN - estimated_n)
                        if delta < n_fallback_delta:
                            n_fallback_delta = delta
                            n_fallback = (pos, cM, cN, cK)
                n_tolerance = max(2 * tile_N, 0.5 * estimated_n)
                if n_fallback is not None and n_fallback_delta <= n_tolerance:
                    best_pos, pos_M, pos_N, pos_K = n_fallback
                else:
                    return self._match_gemm_position_via_dispatch(
                        entry, seq_len, batch_size, tp_size)
            else:
                return self._match_gemm_position_via_dispatch(
                    entry, seq_len, batch_size, tp_size)

        return best_pos, pos_M, pos_N, pos_K

    def infer_gemm_dimensions(self, layer_position: str, seq_len: int,
                               batch_size: int = 1, tp_size: int = 1) -> Tuple[int, int, int]:
        """Infer GEMM M/N/K from layer position and model config.

        Delegates to src.serve_trace.shape_inference.infer_gemm_dimensions
        (canonical implementation). Kept here for callers that hold a
        TracePredictor instance.
        """
        from src.serve_trace.shape_inference import infer_gemm_dimensions as _infer
        return _infer(self.model, layer_position, seq_len, batch_size, tp_size)

    @staticmethod
    def _detect_phase_boundaries(trace: List[dict], prompt_len: int) -> List[int]:
        """Detect per-kernel seq_len by finding prefill→decode boundary.

        Uses norm kernel grid[0] (≈ num_tokens) to detect the transition:
        prefill kernels have grid[0] ≈ prompt_len, decode kernels have
        grid[0] ≈ batch_size (typically 1 for single-request, small for batch).

        Returns a list of seq_len values, one per trace entry.
        If no boundary is detected, all entries get prompt_len.
        """
        norm_indices = []
        for i, e in enumerate(trace):
            name = e.get("kernel_name", "")
            if "norm" in name.lower():
                norm_indices.append((i, e["grid"][0]))

        if not norm_indices:
            return [prompt_len] * len(trace)

        boundary_idx = None
        threshold = max(prompt_len * 0.3, 10)
        for j in range(1, len(norm_indices)):
            prev_g = norm_indices[j - 1][1]
            curr_g = norm_indices[j][1]
            if prev_g >= threshold and curr_g < threshold:
                boundary_idx = norm_indices[j][0]
                break

        if boundary_idx is None:
            return [prompt_len] * len(trace)

        seq_lens = []
        for i in range(len(trace)):
            seq_lens.append(prompt_len if i < boundary_idx else 1)
        return seq_lens

    @staticmethod
    def _group_into_steps(trace: List[dict]) -> List[int]:
        """Return step boundary indices (including 0 and len(trace)).

        Uses step_id field when available; falls back to timestamp gaps.
        For real timestamps (large values from perf_counter), uses an
        adaptive gap threshold derived from the median inter-kernel gap.
        """
        n = len(trace)
        if n == 0:
            return [0]

        has_step_id = "step_id" in trace[0]
        if has_step_id:
            bounds = [0]
            for i in range(1, n):
                if trace[i].get("step_id") != trace[i - 1].get("step_id"):
                    bounds.append(i)
            bounds.append(n)
            return bounds

        gaps = []
        for i in range(1, n):
            gaps.append(trace[i]["timestamp_us"] - trace[i - 1]["timestamp_us"])

        if not gaps:
            return [0, n]

        max_ts = max(e["timestamp_us"] for e in trace)
        if max_ts < 1000:
            gap_threshold = 0.5
        else:
            sorted_gaps = sorted(gaps)
            median_gap = sorted_gaps[len(sorted_gaps) // 2]
            gap_threshold = max(median_gap * 10, 5000.0)

        bounds = [0]
        for i, g in enumerate(gaps):
            if g > gap_threshold:
                bounds.append(i + 1)
        bounds.append(n)
        return bounds

    @staticmethod
    def _detect_per_step_dims(trace: List[dict], prompt_len: int,
                               batch_size: int) -> List[Tuple[int, int]]:
        """Detect per-kernel (effective_seq_len, effective_batch) for each trace entry.

        Groups kernels into steps (via step_id or timestamp gaps), then reads
        norm grid[0] as total_tokens per step. Decomposes into (seq_len, batch):
          - total_tokens > prompt_len * 0.3: prefill step → (total_tokens, 1)
          - total_tokens <= batch_size: decode step → (1, total_tokens)
          - otherwise: mixed step → (total_tokens, 1) for GEMM compatibility

        Falls back to _detect_phase_boundaries logic for single-step traces.
        """
        n = len(trace)
        if n == 0:
            return []

        step_bounds = TracePredictor._group_into_steps(trace)

        if len(step_bounds) <= 2:
            seq_lens = TracePredictor._detect_phase_boundaries(trace, prompt_len)
            return [(s, batch_size) for s in seq_lens]

        result = [(prompt_len, batch_size)] * n

        for s_idx in range(len(step_bounds) - 1):
            start, end = step_bounds[s_idx], step_bounds[s_idx + 1]
            norm_g0_vals = []
            for i in range(start, end):
                if "norm" in trace[i].get("kernel_name", "").lower():
                    norm_g0_vals.append(trace[i]["grid"][0])

            if not norm_g0_vals:
                continue

            total_tokens = min(norm_g0_vals)

            threshold = max(prompt_len * 0.3, 10)
            if total_tokens <= batch_size:
                dims = (1, total_tokens)
            elif total_tokens >= threshold:
                if prompt_len > 1 and total_tokens > prompt_len:
                    n_in_step = max(1, round(total_tokens / prompt_len))
                    per_req_s = total_tokens // n_in_step if n_in_step > 0 else total_tokens
                    dims = (per_req_s, n_in_step)
                else:
                    dims = (total_tokens, 1)
            else:
                dims = (total_tokens, 1)

            for i in range(start, end):
                result[i] = dims

        return result

    def annotate_trace(self, trace: List[dict], seq_len: int,
                        batch_size: int = 1, tp_size: int = 1) -> List[KernelAnnotation]:
        """Annotate all trace entries with predicted latencies.

        Args:
            trace: list of trace entries from emulation
            seq_len: prompt sequence length (prefill).  Decode phase
                     kernels are detected automatically and use seq_len=1.
            batch_size: batch size (number of requests)

        Returns:
            List of KernelAnnotation with predicted latencies
        """
        per_kernel_dims = self._detect_per_step_dims(trace, seq_len, batch_size)

        annotations = []
        prev_timestamp_us = None

        for i, entry in enumerate(trace):
            cur_seq, cur_batch = per_kernel_dims[i]
            cat = classify_kernel(entry)
            _raw_cat = cat
            cat = resolve_category(cat, self._is_moe)  # raw cat for dispatch (SPLITK stays SPLITK, not GEMM)
            # resolve_category promotes REDUCE→FUSEDMOE in MoE models to route the
            # moe_sum reduce through MoE pricing — but it also wrongly catches the
            # sampler's greedy ArgMax reduce_kernel (a lm_head/sampler op, NOT a
            # fused-MoE op; it carries no routing histogram), which then raises in
            # _build_op_attr's mandatory-tok_per_expert check. Keep such argmax
            # reduces as their true REDUCE category so they price as a plain
            # reduce. Fix is at the H-layer dispatch site; classifier.py is left
            # untouched (goal §6). Dense is unaffected (no MoE promotion → cat
            # never becomes FUSEDMOE here).
            if (cat == "FUSEDMOE" and _raw_cat == "REDUCE"
                    and "argmax" in entry.get("kernel_name", "").lower()):
                cat = "REDUCE"
            dims = {}
            predicted_ns = self.launch_overhead_ns  # default: launch overhead only

            # Compute CPU overhead from wall-clock timestamp gap.
            # In emulation, all kernel launches return instantly, so the gap
            # between consecutive kernel timestamps = pure CPU framework overhead
            # (scheduler dispatch, Python logic, IPC, etc.).
            cpu_overhead_ns = 0.0
            cur_timestamp_us = entry["timestamp_us"]
            if cur_timestamp_us is not None and prev_timestamp_us is not None:
                gap_us = cur_timestamp_us - prev_timestamp_us
                if gap_us > 0:
                    cpu_overhead_ns = gap_us * 1000.0  # us → ns
            prev_timestamp_us = cur_timestamp_us

            if cat == "GEMM":
                # Ascend CANN matmul (aclnnMatmul_* etc.) carries (M,N,K) on the
                # entry op_attr — prefer it (same rationale as _build_op_attr):
                # _match_gemm_position replays the A800 cuBLAS-tile dispatch which
                # doesn't apply to the fixed-grid CANN kernel. Scoped by kernel
                # name so GPU kernels keep the position-matching path verbatim.
                _kn = entry.get("kernel_name", "").lower()
                _ca = entry.get("op_attr")
                if (any(s in _kn for s in ("aclnnmatmul", "matmulv2", "matmulv3",
                        "batchmatmul", "ppmatmul", "matmulcommon"))
                        and _ca and all(k in _ca for k in ("M", "N", "K"))):
                    M, N, K = _ca["M"], _ca["N"], _ca["K"]
                    dims = {"M": M, "N": N, "K": K, "position": "ascend_matmul"}
                else:
                    pos, M, N, K = self._match_gemm_position(entry, cur_seq, cur_batch, tp_size)
                    dims = {"M": M, "N": N, "K": K, "position": pos}
                predicted_ns = self._predict_kernel(entry, "GEMM", cur_seq, cur_batch, tp_size)

            elif cat == "SPLITK":
                # Both real cuBLAS splitKreduce (A800) and mcblas split-K reduce
                # (C500 DevDstElementwiseAdd) route through GEMMPredictor, which
                # holds the execution-floor model (dispatch + fixed + per_wave ×
                # waves) for DevDst and the traffic model for real reduces. The
                # DevDst minimal-op_attr path lives in _build_op_attr — no floor
                # formula duplicated here.
                dims = {"cat": "SPLITK"}
                predicted_ns = self._predict_kernel(entry, "GEMM", cur_seq, cur_batch, tp_size)

            elif cat == "FLASH_ATTN":
                num_heads_tp = self.model.num_attention_heads // tp_size
                dims = {
                    "seq_len": cur_seq,
                    "num_heads": num_heads_tp,
                    "head_dim": self.model.head_dim,
                }
                predicted_ns = self._predict_kernel(entry, "FLASHATTN", cur_seq, cur_batch, tp_size)

            elif cat == "FUSEDMOE" or (
                    cat.startswith("MOE_")
                    and cat not in ("MOE_SORT", "MOE_HISTOGRAM")):
                dims = {"cat": cat}
                predicted_ns = self._predict_kernel(entry, "FUSEDMOE", cur_seq, cur_batch, tp_size)

            elif cat == "NORM":
                dims = {"seq_len": cur_seq, "hidden_size": self.model.hidden_size}
                predicted_ns = self._predict_kernel(entry, "NORM", cur_seq, cur_batch, tp_size)

            elif cat == "ROTARY":
                num_elements = cur_seq * (self.model.num_attention_heads // tp_size) * self.model.head_dim
                dims = {"num_elements": num_elements}
                predicted_ns = self._predict_kernel(entry, "ROTARY", cur_seq, cur_batch, tp_size)

            elif cat == "ACTIVATION":
                num_elements = cur_seq * (self.model.intermediate_size // tp_size)
                dims = {"num_elements": num_elements}
                predicted_ns = self._predict_kernel(entry, "ACTIVATION", cur_seq, cur_batch, tp_size)

            elif cat == "KV_CACHE":
                num_elements = cur_seq * (self.model.num_key_value_heads // tp_size) * self.model.head_dim * 2
                dims = {"num_elements": num_elements}
                predicted_ns = self._predict_kernel(entry, "KV_CACHE", cur_seq, cur_batch, tp_size)

            elif cat == "REDUCE":
                grid_total = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
                block_total = entry["block"][0] * entry["block"][1] * entry["block"][2]
                num_elements = grid_total * block_total
                dims = {"num_elements": num_elements}
                predicted_ns = self._predict_kernel(entry, "REDUCE", cur_seq, cur_batch, tp_size)

            elif cat == "GATHER":
                grid_total = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
                block_total = entry["block"][0] * entry["block"][1] * entry["block"][2]
                num_elements = grid_total * block_total
                dims = {"num_elements": num_elements}
                predicted_ns = self._predict_kernel(entry, "GATHER", cur_seq, cur_batch, tp_size)

            elif cat == "ELEMENTWISE":
                grid_total = entry["grid"][0] * entry["grid"][1] * entry["grid"][2]
                block_total = entry["block"][0] * entry["block"][1] * entry["block"][2]
                dims = {"num_elements": grid_total * block_total * 4}
                predicted_ns = self._predict_kernel(entry, "ELEMENTWISE", cur_seq, cur_batch, tp_size)

            elif cat == "COLLECTIVE":
                # NCCLPredictor holds the ring all-reduce alpha+bandwidth formula
                # and reads interconnect params from hw. Serving only decodes
                # msg_bytes/nranks (in _build_op_attr) — no formula here.
                dims = {"cat": "COLLECTIVE"}
                predicted_ns = self._predict_kernel(entry, "COLLECTIVE", cur_seq, cur_batch, tp_size)

            elif cat == "CUSTOM_COMM":
                dims = {"cat": "CUSTOM_COMM"}
                predicted_ns = self._predict_kernel(entry, "CUSTOM_COMM", cur_seq, cur_batch, tp_size)

            elif cat in ("MEMCPY_DTOD", "MEMCPY_DTOH", "MEMCPY_HTOD"):
                # MemcpyPredictor holds the bandwidth-latency model (DRAM for
                # DTOD, PCIe for DTOH/HTOD). Serving only supplies direction+bytes.
                dims = {"cat": cat}
                predicted_ns = self._predict_kernel(entry, "MEMCPY", cur_seq, cur_batch, tp_size)

            elif cat == "MOE_SORT":
                dims = {"cat": cat}
                predicted_ns = self._predict_kernel(entry, "MOE_SORT", cur_seq, cur_batch, tp_size)

            elif cat == "MOE_HISTOGRAM":
                dims = {"cat": cat}
                predicted_ns = self._predict_kernel(entry, "MOE_HISTOGRAM", cur_seq, cur_batch, tp_size)

            annotations.append(KernelAnnotation(
                index=i,
                kernel_name=entry["kernel_name"],
                grid=entry["grid"],
                block=entry["block"],
                shared_mem=entry["shared_mem"],
                category=map_category(cat),  # mapped cat for per-category (FLASHATTN, GEMM)
                predicted_ns=predicted_ns,
                cpu_overhead_ns=cpu_overhead_ns,
                dimensions=dims,
            ))

        return annotations

    def annotate_trace_diff(self, trace: List[dict], seq_len: int,
                              batch_size: int = 1, tp_size: int = 1,
                              sm_freq_override=None, dram_freq_override=None):
        """Annotate trace with differentiable predictions using DiffPerformanceFormula.

        Returns List[DiffKernelPrediction] for all midlayer-supported kernels.
        Unsupported kernel categories (ELEMENTWISE, NCCL, TP_COMM, OTHER) are
        excluded with a warning.  Use get_diff_coverage() after calling this
        method to retrieve the per-category exclusion counts.

        Schema validation: entries must have 'kernel_name', 'grid', 'block',
        and 'shared_mem' fields. Ill-formed traces raise ValueError before any
        prediction is attempted.

        Optional sm_freq_override / dram_freq_override allow DSE callers to
        scale hardware capacities without modifying hw_configs.  The predictors
        emit PredictionInput with nominal frequencies; these overrides replace
        sm_freq_hz / dram_freq_hz before DiffPerformanceFormula sees them,
        producing correct scaled gradients for gradient descent.

        Returns:
            List[DiffKernelPrediction] for all successfully predicted kernels
        """
        # Schema validation
        required_fields = ["kernel_name", "grid", "block", "shared_mem"]
        for i, entry in enumerate(trace):
            for field in required_fields:
                if field not in entry:
                    raise ValueError(
                        f"Trace entry {i}: missing required field '{field}'. "
                        f"Supported fields: {required_fields}")

        predictions = []
        coverage = {}

        # Map classify_kernel() categories to midlayer predictor categories.
        # classify_kernel returns fine-grained names (FLASH_ATTN, TC_GEMM, SPLITK,
        # GEMV); midlayer predictors use coarser keys (FLASHATTN, GEMM).
        # map_category applies the shared CATEGORY_MAP (FLASH_ATTN→FLASHATTN,
        # SPLITK→GEMM, GEMV→GEMM). Same function as _nsys_category_map —
        # ensures pred and nsys per-category breakdown never drift.

        for i, entry in enumerate(trace):
            cat = classify_kernel(entry)
            midlayer_cat = map_category(cat)
            if midlayer_cat is not None:
                midlayer_cat = resolve_category(midlayer_cat, self._is_moe)

            if midlayer_cat is not None:
                try:
                    pred = self._predict_with_midlayer_diff(
                        entry, midlayer_cat, seq_len, batch_size, tp_size,
                        sm_freq_override=sm_freq_override,
                        dram_freq_override=dram_freq_override)
                    predictions.append(pred)
                except (ValueError, KeyError, AssertionError) as exc:
                    import sys as _sys
                    print(
                        f"[predict_kernel_time_from_trace] annotate_trace_diff: "
                        f"skipping {cat} kernel '{entry.get('kernel_name', '?')}': {exc}",
                        file=_sys.stderr)
                    coverage[cat] = coverage.get(cat, 0) + 1
            else:
                coverage[cat] = coverage.get(cat, 0) + 1

        # Log coverage summary
        total = len(trace)
        modelled = len(predictions)
        excluded = total - modelled
        if excluded > 0:
            import sys as _sys
            print(f"[predict_kernel_time_from_trace] annotate_trace_diff: "
                  f"{modelled}/{total} kernels modelled ({modelled/total*100:.1f}%), "
                  f"{excluded} excluded: {dict(coverage)}", file=_sys.stderr)

        # Store coverage on instance for retrieval via get_diff_coverage()
        self._diff_coverage = coverage
        return predictions

    def get_diff_coverage(self):
        """Return coverage dict from the most recent annotate_trace_diff() call.

        Returns empty dict if annotate_trace_diff() has not been called.
        """
        return getattr(self, "_diff_coverage", {})

    @staticmethod
    def simulate_gpu_cpu_overlap(annotations: List[KernelAnnotation],
                                 cpu_scale: float = 1.0) -> dict:
        """Event-driven GPU-CPU overlap simulation (design.md §7.4.3).

        Models the real pipelining between CPU dispatch and GPU execution
        in single-stream eager mode (vLLM):

          cpu_ready[i] = cpu_ready[i-1] + cpu_time[i]
          gpu_free[i]  = gpu_end[i-1]
          start[i]     = max(cpu_ready[i], gpu_free[i])
          gpu_end[i]   = start[i] + gpu_predicted[i]

          wall_clock = gpu_end[last]

        Args:
            annotations: KernelAnnotation list with predicted_ns and cpu_overhead_ns.
            cpu_scale: scaling factor for plugin CPU overhead. Plugin mode
                inflates CPU time due to emulation hooks; set < 1.0 to
                compensate.  1.0 = raw plugin timestamps.

        Returns dict with wall_ns, gpu_ns, cpu_ns, gpu_bound_pct, cpu_bound_pct.
        """
        if not annotations:
            return {"wall_ns": 0.0, "gpu_ns": 0.0, "cpu_ns": 0.0,
                    "gpu_bound_pct": 0.0, "cpu_bound_pct": 0.0}

        cpu_ready = 0.0
        gpu_end = 0.0
        gpu_bound_count = 0
        cpu_bound_count = 0

        for ann in annotations:
            cpu_time = ann.cpu_overhead_ns * cpu_scale
            cpu_ready += cpu_time
            gpu_free = gpu_end

            start = max(cpu_ready, gpu_free)
            if gpu_free >= cpu_ready:
                gpu_bound_count += 1
            else:
                cpu_bound_count += 1

            gpu_end = start + ann.predicted_ns

        n = len(annotations)
        gpu_ns = sum(a.predicted_ns for a in annotations)
        cpu_ns = sum(a.cpu_overhead_ns * cpu_scale for a in annotations)

        return {
            "wall_ns": gpu_end,
            "gpu_ns": gpu_ns,
            "cpu_ns": cpu_ns,
            "gpu_bound_pct": gpu_bound_count / n * 100 if n > 0 else 0,
            "cpu_bound_pct": cpu_bound_count / n * 100 if n > 0 else 0,
        }

    def predict_inference_time(self, trace: List[dict], seq_len: int,
                                batch_size: int = 1, tp_size: int = 1,
                                cpu_scale: float = 1.0) -> dict:
        """Predict total inference time from trace.

        Uses event-driven GPU-CPU overlap simulation (design.md §7.4.3):
        wall_clock emerges from max(cpu_ready, gpu_free) per kernel, not
        naive serial sum.

        Args:
            cpu_scale: scaling factor for plugin CPU overhead (default 1.0).
                Plugin mode inflates CPU time due to emulation hooks.

        Returns dict with:
            total_ns: simulated wall-clock time (with overlap)
            gpu_ns: total predicted GPU kernel time
            cpu_ns: total CPU framework overhead (scaled)
            num_kernels: number of kernels in trace
            breakdown: {category: total_gpu_ns} per kernel type
            gpu_bound_pct / cpu_bound_pct: bottleneck distribution
        """
        annotations = self.annotate_trace(trace, seq_len, batch_size, tp_size)

        sim = self.simulate_gpu_cpu_overlap(annotations, cpu_scale=cpu_scale)

        breakdown = {}
        for a in annotations:
            if a.category not in breakdown:
                breakdown[a.category] = 0.0
            breakdown[a.category] += a.predicted_ns

        total_ns = sim["wall_ns"]
        gpu_ns = sim["gpu_ns"]
        cpu_ns = sim["cpu_ns"]

        return {
            "total_ns": total_ns,
            "total_ms": total_ns / 1e6,
            "gpu_ns": gpu_ns,
            "gpu_ms": gpu_ns / 1e6,
            "cpu_ns": cpu_ns,
            "cpu_ms": cpu_ns / 1e6,
            "cpu_pct": cpu_ns / total_ns * 100 if total_ns > 0 else 0,
            "gpu_bound_pct": sim["gpu_bound_pct"],
            "cpu_bound_pct": sim["cpu_bound_pct"],
            "num_kernels": len(annotations),
            "breakdown": breakdown,
            "breakdown_pct": {k: v / gpu_ns * 100 for k, v in breakdown.items()} if gpu_ns > 0 else {},
        }


def main():
    """CLI: predict latency from trace file."""
    import argparse
    parser = argparse.ArgumentParser(description="Predict inference time from emulation trace")
    parser.add_argument("trace", help="Trace JSON file path")
    parser.add_argument("--model", required=True, help="Model path (for config.json)")
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml",
                        help="Hardware config YAML")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--cpu-scale", type=float, default=1.0,
                        help="CPU overhead scaling factor (plugin mode inflates CPU time)")
    args = parser.parse_args()

    # Load trace
    from src.serve_trace import parse_trace
    trace = parse_trace(args.trace)

    # Load configs
    model_config = load_model_config(args.model)
    predictor = TracePredictor(args.hw_config, model_config)

    # Predict
    result = predictor.predict_inference_time(
        trace, args.seq_len, args.batch_size, cpu_scale=args.cpu_scale)

    print(f"Simulated wall-clock: {result['total_ms']:.2f} ms")
    print(f"  GPU kernel time:    {result['gpu_ms']:.2f} ms")
    print(f"  CPU overhead:       {result['cpu_ms']:.2f} ms")
    print(f"  GPU-bound kernels:  {result['gpu_bound_pct']:.1f}%")
    print(f"  CPU-bound kernels:  {result['cpu_bound_pct']:.1f}%")
    print(f"Number of kernels: {result['num_kernels']}")
    print(f"\nBreakdown by category (GPU only):")
    for cat, pct in sorted(result["breakdown_pct"].items(), key=lambda x: -x[1]):
        ns = result["breakdown"][cat]
        print(f"  {cat:15s}: {ns/1e6:8.2f} ms ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
