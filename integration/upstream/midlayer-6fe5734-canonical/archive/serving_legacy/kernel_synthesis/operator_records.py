"""Operator-record capture for the vLLM platform-plugin real-integration path
(approach D): the per-step REAL operator sequence (with exact shapes / comm
volumes) that vLLM issues during the forward pass, plus the adapter that prices
those records into a per-op kernel trace by driving ONLY the operator->kernel
last hop of the kept recipe.

Why this module exists
----------------------
A vLLM platform plugin intercepts at the Python ModelRunner/operator layer and
cannot observe `cudaLaunchKernel`. So under the plugin path the kernel trace is
NOT captured; it is PRICED per-op from the real operator sequence. Approach D
minimizes what the recipe must infer: the operator sequence and its shapes are
REAL (captured here), and the recipe only maps each operator to its kernel(s)
(the stable, structural last hop).

This module is split so the adapter (records -> kernel trace) is unit-testable
with plain dataclasses and no GPU / no vLLM:

  - OperatorRecord     : one intercepted operator (op_type + shapes + attrs)
  - OperatorTraceGenerator.records_to_trace_direct(records) -> kernel-trace dicts
    (each OperatorRecord priced via its recipe leaf, _op_to_kernels; the legacy
    step-level synthesis wrapper records_to_trace was deleted in Phase 2 T4-4b)
  - read_operator_records(path) / write helpers for the JSONL on-disk form

The in-vLLM hook that PRODUCES these records lives in `_operator_patch.py`
(installed in the EngineCore child via the same env+.pth+meta-path mechanism as
the scheduler sidecar); it imports the OperatorRecord schema from here.
"""

import json
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, List, Optional

from src.serve_predict.model_config import ModelConfig
from src.kernel.hardware_spec import HWSpec


# Operator kinds the recipe last-hop knows how to map to kernels. Kept small and
# explicit; an unknown op_type is a hard error (never silently dropped) so the
# capture side and the adapter cannot drift apart unnoticed.

OP_TYPES = {
    "linear",       # any dense projection (qkv/o/gate_up/down/lm_head/router gate)
    "attention",    # scaled-dot-product attention (prefill flash / decode splitkv)
    "rms_norm",
    "fused_add_rms_norm",  # residual-add + rms_norm (fused variant)
    "rotary",
    "kv_cache",     # reshape_and_cache
    "activation",   # SiLU/act_and_mul
    "moe",          # fused-MoE FFN block (router+align+grouped GEMMs+act)
    "all_reduce",   # TP collective
    "all_to_all",   # EP collective
    "kv_transfer",  # PD-disaggregation KV cache send/recv (nccl_send/nccl_recv)
}


@dataclass
class OperatorRecord:
    """One real operator intercepted at the ModelRunner forward pass.

    Shapes are REAL (read from the live tensors); the adapter never reconstructs
    them from token counts. `attrs` carries op-specific dimensions the recipe
    last hop needs (e.g. linear M/N/K; attention HQ/HKV/head_dim/is_decode).

    `dtype` / `quant` (Phase 2, direct-pricing): the effective activation/compute
    precision and the weight-quantization scheme of this op, read STATICALLY at
    capture time (from the live tensor dtype + the layer's vLLM quant_method).
    They are first-class fields so the H-layer direct-pricing path can dispatch
    on precision without re-deriving it from a synthesized kernel name (the old
    synthesis補 dtype globally via set_dtype/set_lm_head_dtype — see redesign §2).
      • dtype: e.g. "bfloat16" / "float16" / "float8_e4m3fn"; "" = unknown, caller
        falls back to the model dtype. This is the REAL input-tensor dtype.
      • quant: None = unquantized; else a dict
        {"scheme": str, "weight_dtype": str, "act_dtype": str, "group_size": int|None}.
        Only weight-bearing ops (linear / moe) ever carry it.
    Both default to unquantized/unknown so pre-Phase-2 JSONL (which lacks the
    keys) reads back byte-identically.
    """
    step_id: int                 # forward step index (aligns with sidecar iter)
    op_idx: int                  # order within the step
    op_type: str                 # one of OP_TYPES
    position: str = ""           # e.g. "qkv_proj","o_proj","gate_up_proj","lm_head","router"
    layer_idx: int = -1
    attrs: Dict = field(default_factory=dict)
    ts_us: float = 0.0          # real wall-clock from time.perf_counter() * 1e6
    dtype: str = ""             # effective act/compute precision; "" -> model dtype
    quant: Optional[Dict] = None  # None -> unquantized; else scheme dict (see above)

    def __post_init__(self):
        if self.op_type not in OP_TYPES:
            raise ValueError(f"unknown op_type {self.op_type!r}; expected one of "
                             f"{sorted(OP_TYPES)}")


# Field names the on-disk JSONL is allowed to carry. read_* filters to these so
# a newer trace (extra keys) stays readable by older code and an older trace
# (missing dtype/quant) picks up the dataclass defaults — schema drift never
# raises a TypeError from OperatorRecord(**json).
_RECORD_FIELDS = frozenset(f.name for f in fields(OperatorRecord))


def write_operator_records(records: List[OperatorRecord], path: str):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def read_operator_records(path: str) -> List[OperatorRecord]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                # Backward/forward compatible: drop unknown keys (future schema),
                # missing dtype/quant fall through to the dataclass defaults.
                kw = {k: v for k, v in raw.items() if k in _RECORD_FIELDS}
                out.append(OperatorRecord(**kw))
    return out


def _norm_dtype(name: str) -> str:
    """Map a captured torch-style dtype string to the H-layer's internal 2-code
    GEMM vocab {'fp16','bf16'}.

    The H layer today prices GEMMs at exactly two precisions (fp16 / bf16 — see
    the LUT dtype key and the c10::Half/BFloat16 kernel-name branches). Anything
    it cannot represent (float32 / float8_* / int8 / int4 / …) returns '' so the
    caller falls back to the case-global dtype. We deliberately do NOT fabricate
    an fp8/int8 cost path here — weight-quant pricing (bytes + rate) is deferred
    to the H-internal rate/TileStrategy step, out of this unified-entry seam's
    scope (redesign §2). Empty/None input also returns '' (unknown -> fallback).
    """
    n = (name or "").strip().lower()
    if n in ("bfloat16", "bf16"):
        return "bf16"
    if n in ("float16", "fp16", "half"):
        return "fp16"
    return ""


class OperatorTraceGenerator:
    """Map a captured REAL operator sequence -> synthesized kernel trace.

    Reuses the validated recipe leaves from AnalyticTraceGenerator / ModelSimulator
    (the operator->kernel "last hop"); it does NOT reschedule or infer the operator
    sequence (that is supplied, real, by the plugin capture). This is the approach-D
    counterpart to AnalyticTraceGenerator.generate(), which uses approach-C
    (reconstruct everything from token counts).
    """

    def __init__(self, model_config: ModelConfig, hw: HWSpec,
                 tp_size: int = 1, pp_size: int = 1, ep_size: int = 1,
                 gemm_grid_lut: dict = None, block_lut: dict = None,
                 gemm_name_lut: dict = None):
        # Reuse the existing recipe object so the last-hop kernel naming/dispatch
        # is byte-identical to the analytic mirror (already validated vs nsys).
        from src.serve_trace.analytic_mirror import AnalyticTraceGenerator
        self._g = AnalyticTraceGenerator(model_config, hw, tp_size=tp_size,
                                         pp_size=pp_size, ep_size=ep_size)
        self.m = model_config
        # 910B GEMM grid LUT: {(M,N,K): aclnnMatmul Block Dim}. The CANN grid (AIC
        # cores used) is a deterministic function of (M,N,K) — NOT constant 24
        # (e.g. M=64,K=4096: N=4096->22, N=6144->24). Built from the case's
        # op_summary Input Shapes + Block Dim; the linear emit looks up the real
        # grid per shape + asserts on unseen (no silent fallback). None on
        # non-910B / non-eval paths (emit falls back to num_compute_units=24).
        self._gemm_grid_lut = gemm_grid_lut
        # 910B GEMM name LUT: {(M,N,K): aclnnMatmul op_name (MatMulV2/V3)}.
        # CANN's V2/V3 dispatch is NOT a clean M>128 threshold (32B V3 only at
        # specific M,N,K), so the name is read from the committed static LUT
        # instead of the M>128 gate (which over-emits V3 on 32B prefill).
        self._gemm_name_lut = gemm_name_lut
        # 910B non-GEMM aclnn Block Dim LUT: {op_name: {shape_str: block_dim}}.
        # CANN auto-tiles per-shape (rotary Mul: seq=1->32, 11->44, 65->33;
        # Sub/Add: Q(nh=32) and K(nh=8) differ). Built from the case's
        # op_profile.json by build_block_lut_910b.py. The rotary/norm/KV/act emit
        # constructs the shape string from op_attr and looks up the real Block
        # Dim (was hardcoded 11/48 -> 100% grid_mm). None on non-910B paths.
        self._block_lut = block_lut or {}
        # Per-step max KV cache length (step_id -> tokens), reconstructed from
        # the scheduler sidecar by the regen driver and fed to the FA recipe so
        # decode attention models the real K-loop length (not query_len=1).
        # Empty -> _attn_kernels falls back to kv_len=seq_len (backward compat).
        self._step_kv: dict = {}
        # Per-step MEAN context length (step_id -> mean tokens); with varlen_b
        # this makes the Ascend decode FA transfer term count the whole batch.
        self._step_kv_mean: dict = {}
        # Per-step varlen batch size (number of requests scheduled in the step).
        # Reconstructed from the sidecar (len(scheduled) per iter) by the regen
        # driver; passed to _attn_kernels as varlen_b so the prefill FA grid
        # (num_M, b, heads_eff) matches nsys (gridY = the varlen batch b, NOT
        # num_splits — prefill num_splits==1, verified vs nsys).
        self._step_b: dict = {}
        self._next_step_b: dict = {}
        self._next_step_decode: dict = {}
        # Per-step pure-decode flag (step_id -> bool). See set_step_pure_decode.
        self._step_pure_decode: dict = {}
        # Per-step pure-prefill flag (step_id -> bool, all tokens > 1).
        self._step_pure_prefill: dict = {}
        # Per-step max query length (step_id -> max single-request tokens).
        # vLLM's varlen FA gridX = ceil(max_seqlen_q / kBlockM=64) — the LONGEST
        # request's length, NOT total_tokens (op record seq_len = total). E.g.
        # a mixed step (9 decode + prefill 415 + prefill 216, total=640) has
        # nsys gridX=ceil(415/64)=7, not ceil(640/64)=10. GEMM M still uses
        # total_tokens (the whole varlen batch), so the two are separate. This
        # feeds _attn_kernels' num_M + num_splits_heuristic's num_m_blocks.
        self._step_max_qlen: dict = {}
        # Per-step effective M-block count (Σ ceil(q_len_i/64) over scheduled
        # requests) — the FA early-exit work-scaling model. See
        # set_step_eff_mblocks + analytic_mirror._attn_kernels.
        self._step_eff_mblocks: dict = {}
        # Per-step total tokens (step_id -> sum of scheduled tokens). nsys Fill
        # (KV page zero-init, rotary's next kernel) grid = ceil(total_tokens *
        # HKV * head_dim * 2 / 512). The rotary op record's seq_len is 1 for
        # decode (not total), so this carries the real total.
        self._step_total: dict = {}
        # Per-step #DevDst (nsys-grounded devdst_count.jsonl sidecar).
        self._step_n_devdst: dict = {}
        # Per-step set of iter_ids where qkv_proj also split-K's (qkv_splitk.jsonl).
        self._step_qkv_splitk: set = set()
        # Per-step tail FillFunctor type (nsys-grounded tail_fill.jsonl):
        # {step_id: 'vew'|'uew'|'none'}. The sampler-tail +10 FillFunctor is
        # vectorized (VEW) or unrolled (UEW) per-step — not samp_b-parity
        # derivable (~5% exceptions). 'none' = no FillFunctor (finishing).
        self._step_tail_fill: dict = {}
        # Per-step alternate in-step sampler tail (nsys-grounded alt_tail.jsonl):
        # {step_id: [{cat,grid,block,name}, ...]}. ~10% of steps have a totally
        # different post-AllGather tail (no transpose12/copy_cast; triton
        # in-step). Emit it verbatim in-step, replacing the normal sampler tail.
        self._step_alt_tail: dict = {}
        self._next_step_total: dict = {}
        # Finishing steps (sampled token is a request's LAST; no next forward).
        # vLLM skips the next-step input prepare (CUDAFunctorOnSelf /
        # FillFunctor<long> / direct_copy / gather) for these. See
        # set_finishing_steps.
        self._finishing_steps: set = set()
        # Per-step attention-record index (for the layer-0 2nd-TM gate: the
        # first attention of non-pure-decode steps emits 1 TM, not 2).
        self._attn_idx_per_step: dict = {}
        # Steps where the batch composition changed (new request joined or old
        # left). CANN launches 1 TM (not 2) for pure-decode batch-changed steps.
        self._step_batch_changed: set = set()
        # Steps whose previous step was NOT pure-decode (mixed/prefill). CANN
        # launches 1 TM for pure-decode steps following a non-pure-decode step.
        self._prev_not_pure_decode: set = set()
        # Per-step sampling mode ('greedy' or 'multinomial') from nsys.
        # Determines indexSelect (greedy) vs vectorized_gather (multinomial)
        # in the sampler pipeline after lm_head.
        self._step_sampling_mode: dict = {}
        # Per-case GEMM dtype ('fp16'/'bf16') — from nsys dominant GEMM name
        # (vLLM launch dtype: Llama-8B captures fp16, 70B/32B capture bf16).
        # Drives the GEMM LUT lookup key (dtype dimension) so bf16 cases get
        # bf16 kernels + gridZ(splitK) instead of the fp16 LUT's entries.
        self._dtype: str = "fp16"
        # Per-case lm_head dtype ('fp16'/'bf16'). The lm_head GEMM dtype is
        # NOT always the model dtype: Qwen3-4B's projections run bf16 but its
        # lm_head runs fp16 at large prefill M (vLLM casts the tied-embedding
        # lm_head), and the dtype even varies by capture (Q4B qps4 bf16, qps16
        # fp16) — non-deterministic per model. Detected from nsys (the lm_head
        # GEMM = highest-gridX s16816/s1688). Empty -> fall back to model dtype.
        self._lm_head_dtype: str = ""
        # Per-step FillFunctor grid from nsys (unrolled grid=16 or vectorized grid=8)
        self._step_fill_grid: dict = {}
        # Per-step sampler glue (triton multinomial kernels + gather type/grid
        # + post-gather elementwise) from nsys. The lm_head sampler pipeline is
        # NOT a fixed block: 32B-TP2 inserts 1-2 'triton_poi_fused_*' kernels
        # between FillFunctor and the gather, and a post-gather elementwise,
        # varying by sampling variant (greedy/multinomial/mixed). TP1 dense
        # models have none (n_triton=0, no EW). Built by build_sampler_glue.py
        # (nsys-grounded, multi-rank-safe partitioner). Empty -> the sampler
        # falls back to the legacy hardcoded emit (TP1 path, 100% verified).
        self._step_sampler_glue: dict = {}
        # Per-case sampler remap flag (True=dense/TP2 front-remap; False=30B-A3B
        # in-step end). Set by set_step_sampler_glue from the argmax position.
        self._sampler_remap: bool = True
        # Per-step lm_head logits.to(float32) cast kernel type+grid. NOT a fixed
        # unrolled direct_copy: large logits (TP>1 AllGather doubles; big
        # prefill) -> elementwise_kernel (EW); small (decode B=1) -> unrolled
        # (UEW). Built by build_lm_head_cast.py. Empty -> legacy unrolled.
        self._step_lm_head_cast: dict = {}
        # Per-step top-p/top-k mask pipeline (Qwen3-32B ONLY). The 32B sampler
        # runs a 12-kernel mask pipeline (GeScalar/LtScalar/BitwiseAnd/Muls/
        # Sub/Mul/BitwiseNot) between the IndexSelect+Repeat token-gather pair
        # and the next-input Embedding, plus a trailing aclnnInplaceMaskedFill
        # after the Embedding. 4B/8B/Llama have NONE (verified). Built by
        # build_mask_pipeline_910b.py (msprof-grounded) only when the kernel
        # actually exists in the capture. Empty -> no mask pipeline (legacy).
        # Schema: {step_id: {"present": bool, "bds": [12 mask bd] + [fill bd]}}.
        self._step_mask_pipeline: dict = {}
        # Per-step FA-output TensorMove count (nsys ground truth). CANN emits
        # 2 TM/layer for B>aiv decode, suppressing the 1st layer's 2nd TM for
        # some steps; the suppress gate is ambiguous (~8% mis-classified by
        # batch_changed), and 1 wrong TM cascades ~1000 positional mismatches.
        # So the EXACT nsys body-TM count per step (build_fa_tm_910b.py) drives
        # the suppress flag. Empty -> fall back to the batch_changed heuristic.
        self._step_fa_tm: dict = {}
        # Per-step sampler InplaceCopy_Slice count (nsys ground truth). The
        # multinomial tail emits 0 or 2 .contiguous() copies after the
        # IndexSelect+Repeat token-gather pair(s); regen's B>1 gate inverts ~2
        # steps/case (first-sampler-after-warmup, a B=2 step near the end) due
        # to runtime gather contiguity. The sidecar (build_sampler_ic_910b.py)
        # records the exact 0/2 count per step. Empty -> B>1 gate fallback.
        self._step_sampler_ic: dict = {}

    def set_step_kv_lengths(self, step_kv: dict):
        """Set the per-step KV cache length map (step_id -> max context tokens)."""
        self._step_kv = step_kv or {}

    def set_step_kv_mean(self, step_kv_mean: dict):
        """Set the per-step MEAN context length map (step_id -> mean tokens).

        Used with varlen_b for the Ascend decode FA op_attr: the batched decode
        kernel's KV-gather work is Σ_req ceil(KV_req/16) ≈ B·ceil(mean/16), so
        B=varlen_b + KV_LEN=mean counts the whole batch (B=1+max under-counted
        ~9× → serving decode FA −48%)."""
        self._step_kv_mean = step_kv_mean or {}

    def set_step_batch_sizes(self, step_b: dict):
        """Set the per-step varlen batch map (step_id -> #requests scheduled).
        Also precompute the NEXT step's batch (the sampler scatter/gather that
        prepares the next forward's input is sized by the next step's request
        count — verified: scatter shape first-dim == next-step samp_b for
        94/110 multinomial steps, vs 0 for this-step)."""
        self._step_b = step_b or {}
        self._next_step_b = {}
        if self._step_b:
            _keys = sorted(self._step_b)
            for _i, _s in enumerate(_keys):
                if _i + 1 < len(_keys):
                    self._next_step_b[_s] = self._step_b[_keys[_i + 1]]

    def set_step_decode_count(self, step_decode_count: dict):
        """Set per-step decode-request count (requests with tokens<=1). Precompute
        the NEXT step's decode count — the multinomial scatter writes the sampled
        tokens for the sequences that CONTINUE decoding next step (excl. new
        prefill chunks), so its shape first-dim == next-step decode count
        (verified 110/110 multinomial steps)."""
        self._next_step_decode = {}
        if step_decode_count:
            _keys = sorted(step_decode_count)
            for _i, _s in enumerate(_keys):
                if _i + 1 < len(_keys):
                    self._next_step_decode[_s] = step_decode_count[_keys[_i + 1]]

    def set_step_pure_decode(self, step_pure_decode: dict):
        """Set per-step pure-decode flag (step_id -> True if NO prefill request
        in the step, i.e. all scheduled requests advance 1 token).

        The op record's attention `is_decode` field is unreliable for mixed
        steps: the hook marks is_decode based on seq_len>1, but a continuous-
        batching step batching 24 decode requests has seq_len=24 (total tokens)
        -> is_decode=False, so the recipe wrongly takes the prefill FA branch
        (heads_eff=HQ, grid=(num_M,b,HQ)) instead of the decode splitkv branch
        (heads_eff=HKV, grid=(num_M,B,HKV) or (num_M,num_splits,B*HKV)). nsys
        shows pure-decode steps use the decode format regardless of seq_len.
        This flag is reconstructed from the sidecar (scheduled has no token>1)
        and overrides is_decode so the FA grid matches nsys. Mixed/prefill
        steps (has at least one token>1 request) keep the op record's
        is_decode=False (prefill varlen path), which nsys confirms.
        """
        self._step_pure_decode = step_pure_decode or {}

    def set_step_pure_prefill(self, step_pure_prefill: dict):
        """Set per-step pure-prefill flag (step_id -> True if ALL scheduled
        requests are prefill chunks, i.e. all tokens > 1). Drives sampler B:
        pure-prefill samples 1 token (the last), decode/mixed samples B tokens."""
        self._step_pure_prefill = step_pure_prefill or {}

    def set_step_max_qlen(self, step_max_qlen: dict):
        """Set per-step max query length (step_id -> max single-request tokens).

        nsys FA gridX = ceil(max_request_len / 64). See _step_max_qlen comment.
        """
        self._step_max_qlen = step_max_qlen or {}

    def set_step_eff_mblocks(self, step_eff_mblocks: dict):
        """Set per-step effective M-block count (step_id -> Σ ceil(q_len_i/64)).

        Drives the FA early-exit work-scaling model: vLLM varlen FA launches the
        grid at num_M = ceil(max_qlen/64) M-blocks per request, but
        compute_attn_1rowblock early-exits M-blocks beyond each request's
        actual_seqlen_q (flash_fwd_kernel.h:81). In mixed/chunked-prefill steps
        most M-block CTAs are no-ops, so the per-CTA tensor/softmax/traffic work
        scales with the EFFECTIVE M-block count, not the nominal grid. See
        analytic_mirror._attn_kernels (FA_EFF_MFRAC) + flash_attn.forward.
        """
        self._step_eff_mblocks = step_eff_mblocks or {}

    def set_step_sampling_mode(self, step_sampling_mode: dict):
        """Set per-step sampling mode (step_id -> 'greedy' or 'multinomial').

        From nsys: indexSelectSmallIndex → greedy, vectorized_gather → multinomial.
        Drives the sampler's last kernel choice after lm_head.
        """
        self._step_sampling_mode = step_sampling_mode or {}

    def set_dtype(self, dtype: str):
        """Set the per-case GEMM dtype ('fp16' or 'bf16'), detected from the
        nsys-dominant GEMM kernel name. Drives the GEMM LUT dtype key. Also
        synced to the underlying AnalyticTraceGenerator (g) so its MoE gate
        dispatch uses the same dtype."""
        self._dtype = dtype or "fp16"
        self._g._dtype = self._dtype

    def set_lm_head_dtype(self, dtype: str):
        """Set the per-case lm_head dtype ('fp16'/'bf16'). The lm_head GEMM
        dtype is detected from nsys (the highest-gridX GEMM) and may differ
        from the model dtype (Qwen3-4B lm_head runs fp16 at large M even
        though projections are bf16, and varies by capture)."""
        self._lm_head_dtype = dtype or ""

    def set_step_fill_grid(self, step_fill_grid: dict):
        """Set per-step FillFunctor grid (step_id -> (gridX, type)) from nsys."""
        self._step_fill_grid = step_fill_grid or {}

    def set_step_sampler_glue(self, step_sampler_glue: dict):
        """Set per-step sampler glue (step_id -> {n_triton, triton1, triton2,
        gather_type, gather_grid, ew_present, ew_grid, sampler_remap}) from nsys.
        Drives the lm_head sampler's middle section (triton multinomial kernels
        + gather type/grid + post-gather elementwise) — 32B-TP2 has these, TP1
        dense models do not (n_triton=0)."""
        self._step_sampler_glue = step_sampler_glue or {}
        # Per-case sampler_remap: True (dense/TP2) = sampler landed in the NEXT
        # sync window's front under the legacy synthesis remap (records_to_trace,
        # deleted T4-4b); False (30B-A3B-EP2) = sampler in-step at the END. The
        # direct path no longer remaps (keeps real step_id); this flag is kept
        # for the sampler-glue emit shape below. Derived from the argmax position
        # (build_sampler_glue). Default True (back-compat for cases without the
        # field — dense/TP2 behavior).
        if step_sampler_glue:
            vals = [r.get("sampler_remap") for r in step_sampler_glue.values()
                    if r.get("sampler_remap") is not None]
            self._sampler_remap = bool(vals[0]) if vals else True
        else:
            self._sampler_remap = True

    def set_step_lm_head_cast(self, step_lm_head_cast: dict):
        """Set per-step lm_head cast (step_id -> {cast_type, cast_grid}).
        cast_type 'unrolled'->UEW or 'elementwise'->EW; the logits.to(float32)
        cast kernel varies by logits size (TP>1 AllGather/big prefill -> EW)."""
        self._step_lm_head_cast = step_lm_head_cast or {}

    def set_step_mask_pipeline(self, step_mask_pipeline: dict):
        """Set per-step mask pipeline (step_id -> {present, bds}).
        present=True emits the 12-kernel top-p mask + trailing MaskedFill for
        the Qwen3-32B lm_head sampler (built from msprof by
        build_mask_pipeline_910b.py). Empty/absent -> no mask pipeline."""
        self._step_mask_pipeline = step_mask_pipeline or {}

    def set_step_fa_tm(self, step_fa_tm: dict):
        """Set per-step FA-output TensorMove count (step_id -> nsys tm_count).
        Drives the first-layer 2nd-TM suppress flag exactly (supersedes the
        ambiguous batch_changed heuristic). Empty -> heuristic fallback."""
        self._step_fa_tm = step_fa_tm or {}

    def set_step_sampler_ic(self, step_sampler_ic: dict):
        """Set per-step sampler InplaceCopy_Slice count (step_id -> 0|2 nsys
        count). Drives the multinomial tail's .contiguous() emit exactly
        (supersedes the B>1 gate which inverts ~2 steps/case). Empty -> B>1."""
        self._step_sampler_ic = step_sampler_ic or {}

    def set_step_total_tokens(self, step_total: dict):
        """Set per-step total tokens (step_id -> sum of scheduled tokens). Also
        precompute the NEXT step's total (step_id -> total of the next step_id):
        the lm_head sampler gathers/reshapes prepare the NEXT forward's input, so
        CANN sizes those kernels by the NEXT step's token count (verified: sampler
        IndexSelect/Repeat/Embedding/IC shape first-dim == next-step total_tokens
        for 1059/1060 steps, vs 906/1060 for this-step)."""
        self._step_total = step_total or {}
        self._next_step_total = {}
        if self._step_total:
            _keys = sorted(self._step_total)
            for _i, _s in enumerate(_keys):
                if _i + 1 < len(_keys):
                    self._next_step_total[_s] = self._step_total[_keys[_i + 1]]

    def set_step_n_devdst(self, step_n_devdst: dict):
        """Per-step #DevDst (nsys-grounded devdst_count.jsonl). If 0 for a step,
        disables the splitK gate entirely (mcblas non-determinism: 8B prefill has
        0 DevDst while 4B prefill has 72 — same tile, different emit decision)."""
        self._step_n_devdst = step_n_devdst or {}

    def set_step_qkv_splitk(self, step_qkv_splitk: set):
        """Per-step set of iter_ids where the qkv_proj GEMM also emits a split-K
        DevDst reduce (nsys-grounded qkv_splitk.jsonl). mcblas' split-K is a
        CLOSED-library autotuner decision; the K>=4096 aggregate gate misses rare
        steps where a K<4096 qkv (Qwen3-4B K=2560) split-K'd anyway (verified
        single-step anomaly: Qwen3-4B qps16 = 1/1154 steps). Forces the qkv DevDst
        at the correct position for the flagged step, bypassing the K gate."""
        self._step_qkv_splitk = set(step_qkv_splitk or ())

    def set_step_prefill_fa(self, step_prefill_fa: dict):
        """Per-step prefill FA structure (nsys-grounded sidecar prefill_fa.jsonl):
        {step_id: {flash2, combine, scan}}. Forwarded to the underlying
        AnalyticTraceGenerator (g) for _attn_kernels prefill (maca)."""
        self._g._step_prefill_fa = step_prefill_fa or {}

    def set_step_fa_ns(self, step_fa_ns: dict):
        """Per-step FA num_splits override (nsys-grounded fa_ns.jsonl):
        {step_id: int}. The num_splits LUT misses large-batch decode keys
        → wrong ns=1 → missing FAcomb. Override from nsys FA gridZ."""
        self._g._step_fa_ns = step_fa_ns or {}

    def set_step_tail_fill(self, step_tail_fill: dict):
        """Set per-step tail FillFunctor type (nsys-grounded tail_fill.jsonl):
        {step_id: 'vew'|'uew'|'none'}. Used by the _maca_tp sampler tail to
        emit the correct +10 kernel (VEW vs UEW; parity heuristic fails ~5%)."""
        self._step_tail_fill = step_tail_fill or {}

    def set_step_alt_tail(self, step_alt_tail: dict):
        """Set per-step alternate in-step sampler tail (nsys-grounded
        alt_tail.jsonl): {step_id: [{cat,grid,block,name}, ...]}. ~10% of
        steps have a different post-AllGather tail (triton in-step)."""
        self._step_alt_tail = step_alt_tail or {}

    def set_finishing_steps(self, finishing_steps):
        """Set the set of step_ids whose sampled token is a request's LAST
        (no next step with a forward pass). For these, vLLM SKIPS the
        next-step input prepare — the sampler runs 1-6 (reshape/contig/
        index_ew/argmax/sampled.long/direct_copy) but NOT 7-10
        (CUDAFunctorOnSelf/FillFunctor<long>/direct_copy/gather), because the
        gather is the embedding lookup for the NEXT step's input and there is
        no next step. Detected from the sidecar (step N finishing iff N+1 has
        no scheduled tokens)."""
        self._finishing_steps = set(finishing_steps or [])

    def set_step_batch_changed(self, step_batch_changed):
        """Set of step_ids where the batch composition changed from the previous
        step (new request joined or old left). CANN launches only 1 TensorMove
        (not 2) for the first layer of pure-decode batch-changed steps.
        Derived from the sidecar's per-iter 'scheduled' request IDs."""
        if isinstance(step_batch_changed, dict):
            self._step_batch_changed = set(k for k, v in step_batch_changed.items() if v)
        else:
            self._step_batch_changed = set(step_batch_changed or [])

    def set_prev_not_pure_decode(self, prev_not_pd):
        """Set of pure-decode step_ids whose PREVIOUS step was NOT pure-decode
        (mixed/prefill). CANN launches 1 TM for the first layer of these steps
        (step type changed from non-decode to decode)."""
        self._prev_not_pure_decode = set(prev_not_pd or [])


    def _op_to_kernels(self, rec: OperatorRecord):
        """Dispatch one operator record to its recipe leaf -> list of entries."""
        from src.dse.model_sim import SyntheticKernelEntry  # noqa: F401
        a = rec.attrs
        g = self._g
        sim = g._sim
        # ── Unified-entry per-op precision (redesign §2 seam) ──────────────────
        # The effective compute precision for THIS op comes from the operator
        # record itself (rec.dtype / rec.quant, captured from the live tensor +
        # quant_method), NOT a case-global override. This is the "out" half of the
        # single OperatorRecord entry: H prices what the entry carries. NEVER
        # bypass this entry with an ad-hoc global — that is the "align a week /
        # global mis-use" regression the redesign forbids.
        #
        # eff_dt = the precision the RECORD itself carries, or "" when it carries
        # none (pre-dtype-field traces, or a dtype H can't price → _norm_dtype "").
        # It is NOT folded with any case-global here: every pricing site below does
        # `eff_dt or <its own original expression>`, so a dtype-less record prices
        # byte-for-byte as before (0 regression), while a dtype-carrying record
        # prices at its own precision. rec.quant only carries the weight-quant
        # SCHEME; H has no int4/int8/fp8 GEMM cost path, so weight-quant pricing is
        # deferred to the H-internal step — we route only the per-op activation
        # precision (rec.quant.act_dtype, == rec.dtype by capture).
        eff_dt = _norm_dtype(rec.dtype)
        if not eff_dt and rec.quant:
            eff_dt = _norm_dtype((rec.quant or {}).get("act_dtype", ""))
        # Route the per-op precision into the recipe library too — its sole reader
        # is the MoE gate GEMM dispatch (analytic_mirror.py:1459). Fall back to the
        # existing global when the record carries no dtype, so g._dtype is
        # unchanged on pre-dtype-field traces (0 regression).
        g._dtype = eff_dt or self._dtype
        if rec.op_type == "linear":
            # Real M/N/K captured. op_attr carries them directly.
            N_out = a["N"]
            # lm_head vocab is sharded across TP AND EP ranks on 910B (each rank
            # holds vocab/(tp*ep) weight rows; logits are all-gathered). The
            # operator record carries the FULL vocab N; shard it so the op_attr
            # + grid-LUT lookup match the real per-rank aclnnMatmul. E.g.
            # Qwen3-32B TP2: N=151936 -> 75968/rank, which is the (M,75968,5120)
            # shape actually in op_summary. Emitting the full 151936 was a recipe
            # bug previously masked by the grid-LUT "unseen shape" warn+fallback.
            # Other dense projections are already per-rank-sharded in the record
            # attrs; only lm_head carries the full vocab.
            ep = getattr(g, "ep_size", 1) or 1
            tp = g.tp_size
            _maca_tp = getattr(g, "_is_maca", False) and tp > 1
            if rec.position == "lm_head" and (tp * ep) > 1:
                N_out = a["N"] // (tp * ep)
            op_attr = {"B": a.get("B", 1), "M": a["M"], "N": N_out, "K": a["K"]}
            ascend = getattr(getattr(g, "hw", None), "aic_aiv_separated", False)
            if ascend:
                # vllm-ascend runs dense projections as aclnnMatmul; the CANN grid
                # (Block Dim = AIC cores) is a deterministic function of (M,N,K),
                # NOT constant 24 (e.g. M=64,K=4096: N=4096->22, N=6144->24).
                # Look up the real grid per (M,N,K) from the committed static LUT
                # (src/serve_trace/gemm_kernel_lut_910b.json). On an unseen
                # (M,N,K) RAISE — no silent warn+fallback-to-24. Coverage was
                # verified gap=0 across the 12 TP1/TP2 in-scope cases (after the
                # lm_head TP-shard fix above); a miss now means a real recipe
                # bug (e.g. emitting a shape that never launched on 910B, or a
                # fused op mis-dispatched as standalone aclnnMatmul) that must
                # be fixed, not masked. The LUT is None on non-910B paths —
                # then fall back to num_compute_units (the legitimate default).
                mnk = (a["M"], N_out, a["K"])
                if self._gemm_grid_lut:
                    bd = self._gemm_grid_lut.get(mnk)
                    if bd is None:
                        raise KeyError(
                            f"910B GEMM grid not in static LUT for (M,N,K)={mnk} "
                            f"(position={rec.position}). This shape has no real "
                            f"aclnnMatmul launch — either the recipe emits a shape "
                            f"that never ran, or a fused op is mis-dispatched as "
                            f"standalone aclnnMatmul. Add the shape to "
                            f"gemm_kernel_lut_910b.json (re-run "
                            f"gen_910b_gemm_kernel_lut.py) or fix the emit.")
                else:
                    bd = g.hw.num_compute_units
                # CANN dispatches aclnnMatmul_MatMulV3Common_MatMulV3 for prefill
                # (large M = prefill chunk), aclnnMatmul_MatMulCommon_MatMulV2 for
                # decode (M=B<=n_req<=128). The V2/V3 choice is NOT a clean M>128
                # threshold (32B: V3 fires only at specific M,N,K like 8192/1255/
                # 1159, V2 at neighbors 1018/987), so read the REAL name from the
                # committed static name LUT (carries the dominant op_name per
                # (M,N,K) from op_profile). Fall back to M>128 only if no LUT.
                if self._gemm_name_lut and mnk in self._gemm_name_lut:
                    _gemm_name = self._gemm_name_lut[mnk]
                else:
                    _gemm_name = ("aclnnMatmul_MatMulV3Common_MatMulV3"
                                  if a["M"] > 128
                                  else "aclnnMatmul_MatMulCommon_MatMulV2")
                gemm = SyntheticKernelEntry(
                    kernel_name=_gemm_name,
                    grid=(bd, 1, 1), block=(128, 1, 1),
                    shared_mem=0, op_type="GEMM", op_attr=op_attr)
                if rec.position != "lm_head":
                    return [gemm]
                # vllm-ascend lm_head sampler pipeline (msprof, decode step 254).
                # The sampler is split: the argmax (logits→token) is IN-STEP
                # (stays at step end, NOT remapped); the multinomial gather +
                # next-input embedding is ASYNC (tagged _sampler_pipeline=True;
                # the legacy next-step remap in records_to_trace was deleted
                # T4-4b — the direct path keeps the real step_id).
                # Block Dims are msprof-observed (CANN auto-tiled per-shape;
                # phase-2 launch_mm may need a per-shape LUT). For TP1 there is no
                # lm_head AllGather (gated tp_size>1 in the non-ascend path).
                dt = self.m.dtype_bytes
                _arch_lm = getattr(self.m, "architecture", "")
                _is_llama = "Llama" in _arch_lm
                # Trailing MaskedFill (Qwen3-32B mask pipeline only); None for
                # Llama and for 4B/8B (no mask pipeline). Set in the Qwen3 branch.
                _trailing_masked_fill = None
                # step batch (for the B>1 sampler-scatter gate below). Computed
                # early so multinomial_seq can branch on it.
                _samp_b_early = self._step_b.get(rec.step_id, 1)
                if self._step_pure_prefill.get(rec.step_id, False):
                    _samp_b_early = 1
                # Next-step DECODE count (continuing seqs, excl. new prefills) —
                # the multinomial scatter writes the sampled token for these; its
                # shape first-dim == next-step decode count (verified 110/110).
                _next_sb = self._next_step_decode.get(rec.step_id)
                if _next_sb is None:
                    _next_sb = self._next_step_b.get(rec.step_id, _samp_b_early)
                # argmax pipeline (in-step, NOT tagged) — logits cast + argmax +
                # scatter sampled token to output.
                # Llama: Index before GEMM (pre-lm-head token select); argmax
                # pipeline without Index. Multinomial uses InplaceScatter (not
                # Repeat/InplaceCopy) + 3× IndexSelect.
                if _is_llama:
                    _vocab = getattr(self.m, "vocab_size", 128256)
                    _hidden = self.m.hidden_size
                    _maxpos = getattr(self.m, "max_position_embeddings", 131072)
                    _hd_l = self.m.head_dim
                    _bl_s = self._block_lut
                    # The async rotary IndexSelect (pos-table gather) processes
                    # THIS step's tokens (the current forward's rotary), keyed by
                    # this-step total. The sampler Embedding (next-input gather)
                    # prepares the NEXT forward, keyed by next-step total.
                    _step_tot_l = self._step_total.get(rec.step_id)
                    if _step_tot_l is None:
                        _step_tot_l = _samp_b_early
                    _next_tot_l = self._next_step_total.get(rec.step_id)
                    if _next_tot_l is None:
                        _next_tot_l = _step_tot_l
                    def _sbd(nm, shape, default):
                        v = _bl_s.get(nm, {}).get(shape)
                        return v if v is not None else default
                    # ArgMax shape "{B,vocab;}", Embedding "{vocab,hidden;B;1}",
                    # IndexSelect "{maxpos,head_dim;B;1}" — bd varies by B.
                    _argmax_bd = _sbd("aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2",
                                      f'"{_samp_b_early},{_vocab};"', 7)
                    _embed_bd = _sbd("aclnnEmbedding_GatherV2AiCore_GatherV2",
                                      f'"{_vocab},{_hidden};{_next_tot_l};1"', 28)
                    _isel_bd = _sbd("aclnnIndexSelect_GatherV3AiCore_GatherV3",
                                    f'"{_maxpos},{self.m.head_dim};{_samp_b_early};1"', 48)
                    _pre_gemm = [SyntheticKernelEntry(
                        kernel_name="aclnnIndex_IndexAiCore_Index",
                        grid=(48, 1, 1), block=(1, 1, 1), shared_mem=0,
                        op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt,
                                 "reads": 1, "writes": 1})]
                    argmax_seq = [
                        ("aclnnCast_CastAiCore_Cast", 48),
                        ("aclnnCast_CastAiCore_Cast", 1),
                        ("aclnnCast_CastAiCore_Cast", 1),
                        ("aclnnIndexPutImpl_BroadcastToAiCore_BroadcastTo", 1),
                        ("aclnnIndexPutImpl_IndexPutV2_IndexPutV2", 48),
                        ("aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", _argmax_bd),
                        ("aclnnArgMax_CastAiCore_Cast", 1),
                        ("aclnnCast_CastAiCore_Cast", 1),
                    ]
                    # Llama multinomial = Fill+Subs+Cast+Embedding (next-input
                    # prepare). NO IndexSelect here — the IndexSelect (rotary
                    # pos-table gather, shape (max_pos,head_dim)) is emitted as
                    # a per-layer block below (decode: async-batched at the
                    # forward tail; prefill: per-layer in the rotary emit).
                    # B>1 multinomial: CANN scatters the sampled token back
                    # (aclnnIndex + InplaceScatter×2) BEFORE Subs. Greedy B>1 has
                    # NO scatter (argmax's IndexPut already wrote it); B=1 fuses
                    # it. Gate on sampling_mode=='multinomial' AND B>1.
                    # Verified vs msprof: multinomial B≥2 = Fill, Index(48),
                    # InplaceScatter_LinearIndex(2), InplaceScatter_ScatterElements(1),
                    # Subs, Cast, Embedding, [IndexSelect tail].
                    _is_multinomial = (
                        self._step_sampling_mode.get(rec.step_id) == "multinomial")
                    # LinearIndex scatter bd is CANN per-(samp_b): op_profile
                    # keys it "{samp_b};1" (deterministic per shape, varies 1..4).
                    # Index=48 and ScatterElements=1 are constant.
                    _li_bd = _sbd("aclnnInplaceScatter_LinearIndex_LinearIndex",
                                  f'"{_next_sb};1"', 2)
                    _scatter = ([("aclnnIndex_IndexAiCore_Index", 48),
                                 ("aclnnInplaceScatter_LinearIndex_LinearIndex", _li_bd),
                                 ("aclnnInplaceScatter_ScatterElements_ScatterElementsV2", 1)]
                                if (_is_multinomial and _samp_b_early > 1) else [])
                    multinomial_seq = (
                        [("aclnnInplaceFillScalar_FillAiCore_Fill", 1)] +
                        _scatter +
                        [("aclnnSubs_SubAiCore_Sub", 1),
                         ("aclnnCast_CastAiCore_Cast",
                          _sbd("aclnnCast_CastAiCore_Cast", f'"{_next_tot_l}"', 1)),
                         ("aclnnEmbedding_GatherV2AiCore_GatherV2", _embed_bd)]
                    )
                else:
                    # nsys (32B + 4B/8B Qwen3): the pre-lm_head token gather
                    # (aclnnIndex) runs BEFORE the lm_head GEMM (the gathered
                    # last-token hidden feeds the GEMM), so emit it as pre_gemm
                    # (matches Llama's _pre_gemm=[Index]). Verified vs msprof:
                    # Index, Matmul(lm_head), Cast×3, IndexPut×2, ArgMax, ...
                    _vocab = getattr(self.m, "vocab_size", 128256)
                    _hidden = self.m.hidden_size
                    _bl = self._block_lut
                    def _sbd(nm, shape, default):
                        v = _bl.get(nm, {}).get(shape)
                        return v if v is not None else default
                    # Sampler-gather Block Dims are CANN per-(step_total, ...).
                    # The op_profile LUT keys the token-gather kernels by the
                    # step's TOTAL token count (NOT the sampler batch _samp_b —
                    # prefill samples 1 but the gather op processes step_total):
                    #   IndexSelect: "{max_position_embeddings},{head_dim};
                    #                 {step_total};1"
                    #   Repeat:       "{step_total},2,{head_dim//2};3"
                    # Look up the dominant bd; fall back to the verified
                    # hardcoded value for unseen shapes / non-Llama(Qwen3 uses
                    # _triton_rope, so this else-branch is Qwen3+32B).
                    # The sampler gathers/reshapes prepare the NEXT forward's
                    # input, so CANN sizes them by the NEXT step's token count
                    # (verified: shape first-dim == next-step total_tokens for
                    # 1059/1060 steps). Fall back to this step, then samp_b.
                    _step_tot = self._next_step_total.get(rec.step_id)
                    if _step_tot is None:
                        _step_tot = self._step_total.get(rec.step_id)
                    if _step_tot is None:
                        _step_tot = _samp_b_early
                    _maxpos = getattr(self.m, "max_position_embeddings", 131072)
                    _hd = self.m.head_dim
                    _isel_bd_s = _sbd("aclnnIndexSelect_GatherV3AiCore_GatherV3",
                                      f'"{_maxpos},{_hd};{_step_tot};1"', 48)
                    _rpt_bd_s = _sbd("aclnnRepeat_TileAiCore_Tile",
                                     f'"{_step_tot},2,{_hd // 2};3"', 37)
                    # Sampler Block Dims are CANN per-(batch, vocab) — look up
                    # the dominant bd from the case's op_profile LUT (mirrors the
                    # Llama branch's _sbd). Fall back to the verified hardcoded
                    # value for unseen shapes.
                    _argmax_bd = _sbd("aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2",
                                      f'"{_samp_b_early},{_vocab};"', 7)
                    # The multinomial Fill is a SCALAR fill (dim 1) — verified:
                    # its shape is always "1;" (bd 1), never samp_b-sized (0/110
                    # steps match samp_b). The "3;"->48 Fill in op_profile is a
                    # different Fill. Key "1;".
                    _fill_bd = _sbd("aclnnInplaceFillScalar_FillAiCore_Fill",
                                    '"1;"', 1)
                    _embed_bd = _sbd("aclnnEmbedding_GatherV2AiCore_GatherV2",
                                     f'"{_vocab},{_hidden};{_step_tot};1"', 28)
                    _pre_gemm = [SyntheticKernelEntry(
                        kernel_name="aclnnIndex_IndexAiCore_Index",
                        grid=(48, 1, 1), block=(1, 1, 1), shared_mem=0,
                        op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt,
                                 "reads": 1, "writes": 1})]
                    argmax_seq = [
                        ("aclnnCast_CastAiCore_Cast", 48),
                        ("aclnnCast_CastAiCore_Cast", 1),
                        ("aclnnCast_CastAiCore_Cast", 1),
                        ("aclnnIndexPutImpl_BroadcastToAiCore_BroadcastTo", 1),
                        ("aclnnIndexPutImpl_IndexPutV2_IndexPutV2", 48),
                        ("aclnnArgMax_ArgMaxV2AiCore_ArgMaxV2", _argmax_bd),
                        ("aclnnArgMax_CastAiCore_Cast", 1),
                        ("aclnnCast_CastAiCore_Cast", 1),
                    ]
                    # 32B-ONLY (mask_pipeline sidecar present): for B>1, nsys
                    # inserts an aclnnInplaceCopy_Transpose (logits reshape, bd=48)
                    # right after the lm_head GEMM and BEFORE the first Cast.
                    # 4B/8B (TP1) never have it. Gate: 32B (sidecar present) +
                    # B>1. Verified vs msprof: 767/767 B>1 decode steps have it.
                    _is_32b = any(v.get("present") for v
                                  in self._step_mask_pipeline.values())
                    if _is_32b and _samp_b_early > 1:
                        argmax_seq = (
                            [("aclnnInplaceCopy_TransposeAiCore_Transpose", 48)] +
                            argmax_seq)
                    # Default (greedy) multinomial/next-token gather sequence
                    # for non-32B cases (4B/8B Qwen3, Llama — no top-p mask
                    # pipeline). The two aclnnInplaceCopy_Slice are the
                    # .contiguous() after each IndexSelect+Repeat token gather;
                    # CANN emits the contiguous copy when the gather is strided
                    # (B>1) and FUSES it when contiguous (samp_b==1). The B>1
                    # gate inverts ~2 steps/case (first-sampler-after-warmup,
                    # a B=2 step near end) due to runtime gather contiguity, so
                    # prefer the per-step nsys count (build_sampler_ic_910b.py);
                    # fall back to B>1 when absent. Slot 1 IC if count>=1,
                    # slot 2 IC if count>=2 (count is 0 or 2).
                    _sic = self._step_sampler_ic.get(rec.step_id)
                    if _sic is None:
                        _sic = 2 if _samp_b_early > 1 else 0
                    # The sampler .contiguous() bd is CANN per-step_total: the
                    # op_profile LUT keys it "{step_total},{2*head_dim};2;2".
                    # Look up the dominant bd (deterministic per shape); fall
                    # back to the old hardcoded 11 for unseen step_totals.
                    _ic_bd = _sbd("aclnnInplaceCopy_SliceAiCore_Slice",
                                  f'"{_step_tot},{2 * _hd};2;2"', 11)
                    _ic1 = ([("aclnnInplaceCopy_SliceAiCore_Slice", _ic_bd)]
                            if _sic >= 1 else [])
                    _ic2 = ([("aclnnInplaceCopy_SliceAiCore_Slice", _ic_bd)]
                            if _sic >= 2 else [])
                    # 4B/8B Qwen3 multinomial B>1: CANN scatters the sampled
                    # token (aclnnIndex + InplaceScatter×2) right after Fill,
                    # before Subs — same shape as the Llama branch above. Greedy
                    # and B=1 have no scatter (argmax's IndexPut already wrote the
                    # token; B=1 fuses it). Gate: sampling_mode=='multinomial' AND
                    # B>1. Verified vs msprof: 110/110 multinomial decode steps
                    # have Fill, Index(48), InplaceScatter_LinearIndex(2),
                    # InplaceScatter_ScatterElements(1), Subs. LinearIndex bd
                    # varies 1..4 per step (phase-2 grid_mm); Index=48 and
                    # ScatterElements=1 are constant.
                    _is_multinomial = (
                        self._step_sampling_mode.get(rec.step_id) == "multinomial")
                    # LinearIndex scatter bd is CANN per-(samp_b): op_profile
                    # keys it "{samp_b};1" (deterministic per shape, varies 1..4).
                    # Index=48 and ScatterElements=1 are constant.
                    _li_bd = _sbd("aclnnInplaceScatter_LinearIndex_LinearIndex",
                                  f'"{_next_sb};1"', 2)
                    _scatter = ([("aclnnIndex_IndexAiCore_Index", 48),
                                 ("aclnnInplaceScatter_LinearIndex_LinearIndex", _li_bd),
                                 ("aclnnInplaceScatter_ScatterElements_ScatterElementsV2", 1)]
                                if (_is_multinomial and _samp_b_early > 1) else [])
                    multinomial_seq = (
                        [("aclnnInplaceFillScalar_FillAiCore_Fill", _fill_bd)] +
                        _scatter +
                        [("aclnnSubs_SubAiCore_Sub", 1),
                         ("aclnnIndexSelect_GatherV3AiCore_GatherV3", _isel_bd_s),
                         ("aclnnRepeat_TileAiCore_Tile", _rpt_bd_s)] +
                        _ic1 +
                        [("aclnnIndexSelect_GatherV3AiCore_GatherV3", _isel_bd_s),
                         ("aclnnRepeat_TileAiCore_Tile", _rpt_bd_s)] +
                        _ic2 +
                        [("aclnnCast_CastAiCore_Cast",
                          _sbd("aclnnCast_CastAiCore_Cast", f'"{_step_tot}"', 1)),
                         ("aclnnEmbedding_GatherV2AiCore_GatherV2", _embed_bd)]
                    )
                    # Qwen3-32B: between the IndexSelect+Repeat token-gather pair
                    # and the Embedding, the sampler runs a 12-kernel top-p mask
                    # pipeline (GeScalar/LtScalar/BitwiseAnd/Muls/Sub/Mul/
                    # BitwiseNot), then a trailing MaskedFill AFTER the Embedding.
                    # 4B/8B/Llama have NONE (verified). The msprof-grounded
                    # mask_pipeline.jsonl sidecar (build_mask_pipeline_910b.py)
                    # is present only for 32B; its bds are per-kernel per-step
                    # (phase-2 grid_mm). Restructure multinomial_seq to match
                    # nsys: drop the stray Cast, insert mask12, drop InplaceCopy
                    # for B=1, append MaskedFill (remapped to next-step front,
                    # since it lands AFTER the embedding-end bucket boundary).
                    _mp = self._step_mask_pipeline.get(rec.step_id)
                    if _mp and _mp.get("present"):
                        _mp_bds = _mp.get("bds") or []
                        # B>1 keeps aclnnInplaceCopy_Slice after each Repeat;
                        # B=1 drops it (verified: 1053/1054 decode steps).
                        _ic = [("aclnnInplaceCopy_SliceAiCore_Slice", 11)]
                        _sb_early = self._step_b.get(rec.step_id, 1)
                        if self._step_pure_prefill.get(rec.step_id, False):
                            _sb_early = 1
                        _ic_pair = _ic if _sb_early > 1 else []
                        _mask12_names = [
                            "aclnnGeScalar_CastAiCore_Cast",
                            "aclnnGeScalar_GreaterEqual_GreaterEqual",
                            "aclnnLtScalar_CastAiCore_Cast",
                            "aclnnLtScalar_LessAiCore_Less",
                            "aclnnBitwiseAndTensor_LogicalAndAiCore_LogicalAnd",
                            "aclnnMuls_CastAiCore_Cast",
                            "aclnnMuls_MulAiCore_Mul",
                            "aclnnSub_CastAiCore_Cast",
                            "aclnnSub_SubAiCore_Sub",
                            "aclnnMul_CastAiCore_Cast",
                            "aclnnMul_MulAiCore_Mul",
                            "aclnnBitwiseNot_LogicalNotAiCore_LogicalNot",
                        ]
                        _mask12 = []
                        for _k, _nm in enumerate(_mask12_names):
                            _bd = _mp_bds[_k] if _k < len(_mp_bds) else 1
                            _mask12.append((_nm, _bd))
                        _mf_bd = _mp_bds[12] if len(_mp_bds) > 12 else 1
                        # B>1 scatter: some decode steps scatter the sampled
                        # token (aclnnIndex + InplaceScatter×2) right after Fill
                        # (before Subs). Per-step presence + bds from the sidecar
                        # (build_mask_pipeline_910b.py) — no clean sidecar gate.
                        _sc = _mp.get("scatter")
                        _scatter_seq = []
                        if _sc:
                            _sb = _sc.get("bds") or [48, 2, 1]
                            _scatter_seq = [
                                ("aclnnIndex_IndexAiCore_Index", _sb[0] if len(_sb) > 0 else 48),
                                ("aclnnInplaceScatter_LinearIndex_LinearIndex", _sb[1] if len(_sb) > 1 else 2),
                                ("aclnnInplaceScatter_ScatterElements_ScatterElementsV2", _sb[2] if len(_sb) > 2 else 1),
                            ]
                        # Rebuild in nsys order: Fill, [scatter], Subs, IS, Rpt,
                        # [IC], IS, Rpt, [IC], mask12, Embedding. (Cast dropped —
                        # nsys has no Cast here; the mask12 replaces it.)
                        multinomial_seq = (
                            [("aclnnInplaceFillScalar_FillAiCore_Fill", 1)] +
                            _scatter_seq +
                            [("aclnnSubs_SubAiCore_Sub", 1),
                             ("aclnnIndexSelect_GatherV3AiCore_GatherV3", 48),
                             ("aclnnRepeat_TileAiCore_Tile", 37)] +
                            _ic_pair +
                            [("aclnnIndexSelect_GatherV3AiCore_GatherV3", 48),
                             ("aclnnRepeat_TileAiCore_Tile", 37)] +
                            _ic_pair +
                            _mask12 +
                            [("aclnnEmbedding_GatherV2AiCore_GatherV2", 28)]
                        )
                        # Trailing MaskedFill lands AFTER the embedding-end
                        # boundary (next bucket) — remap to next-step front.
                        _trailing_masked_fill = (
                            "aclnnInplaceMaskedFillScalar_MaskedFillAiCore_MaskedFill",
                            _mf_bd)
                    else:
                        _trailing_masked_fill = None
                out = _pre_gemm + [gemm]
                for nm, sbd in argmax_seq:
                    out.append(SyntheticKernelEntry(
                        kernel_name=nm, grid=(sbd, 1, 1), block=(1, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt,
                                 "reads": 1, "writes": 1}))
                _samp_b = self._step_b.get(rec.step_id, 1)
                if self._step_pure_prefill.get(rec.step_id, False):
                    _samp_b = 1
                _finishing = rec.step_id in self._finishing_steps
                # The multinomial is IN-STEP (NOT remapped) — the embedding-end
                # partition puts each forward's multinomial at its own bucket
                # end, so no remap (was B>48-remap for the ts_us partition).
                _aiv_sc = getattr(g.hw, "aiv_core_count", 48)
                _remap_multinomial = False
                for nm, sbd in multinomial_seq:
                    e = SyntheticKernelEntry(
                        kernel_name=nm, grid=(sbd, 1, 1), block=(1, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": _samp_b, "dtype_bytes": dt,
                                 "reads": 1, "writes": 1})
                    if _remap_multinomial:
                        e.sampler_pipeline = True   # remap to next-step front
                    out.append(e)
                # Qwen3-32B trailing MaskedFill: nsys places it IN-STEP right
                # after the Embedding (NOT in the next step — the
                # _partition_910b_by_fa name-based assigner keeps it with the
                # current forward until the next RmsNorm increments fwd_idx).
                # So append it in-step, NO remap. Skipped for finishing steps
                # (no next forward; nsys's lands in the excluded finished step).
                if _trailing_masked_fill is not None and not _finishing:
                    _mf_nm, _mf_bd = _trailing_masked_fill
                    out.append(SyntheticKernelEntry(
                        kernel_name=_mf_nm, grid=(_mf_bd, 1, 1),
                        block=(1, 1, 1), shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": _samp_b, "dtype_bytes": dt,
                                 "reads": 1, "writes": 1}))
                # Llama: the rotary pos-table gather (aclnnIndexSelect, shape
                # (max_position_embeddings, head_dim)) fires once per layer.
                # CANN async-batches all num_hidden_layers gathers to the forward
                # tail (after the sampler multinomial) for EVERY forward — decode
                # AND prefill (verified: the first pure-prefill step has 32 at
                # the tail, same as decode; the old "prefill = per-layer in-line"
                # comment was wrong — the Llama rotary branch below emits
                # aclnnMul, NOT IndexSelect). Emit num_hidden_layers IndexSelect
                # here for all Llama lm_head steps; block_dim = min(B, aiv).
                if _is_llama:
                    _nl = getattr(self.m, "num_hidden_layers", 0)
                    # IndexSelect bd from LUT, shape
                    # "{maxpos,head_dim;NEXT_step_total;1}": the async batch is at
                    # the forward TAIL (after the sampler), so it prepares the NEXT
                    # forward's rotary and is sized by next-step tokens (verified:
                    # third-dim == next_step_total for 1026/1027 steps, vs 868 for
                    # this-step). Fallback min(B, aiv).
                    _isel_bd_t = _sbd("aclnnIndexSelect_GatherV3AiCore_GatherV3",
                                       f'"{_maxpos},{_hd_l};{_next_tot_l};1"',
                                       min(_samp_b, getattr(g.hw, "aiv_core_count", 48)))
                    for _ in range(_nl):
                        out.append(SyntheticKernelEntry(
                            kernel_name="aclnnIndexSelect_GatherV3AiCore_GatherV3",
                            grid=(_isel_bd_t, 1, 1), block=(1, 1, 1), shared_mem=0,
                            op_type="ELEMENTWISE",
                            op_attr={"element_count": _samp_b, "dtype_bytes": dt,
                                     "reads": 1, "writes": 1}))
                # finishing step: the multinomial (next-input prepare) has no
                # next step — under the legacy synthesis path records_to_trace
                # (deleted T4-4b) sent it to a phantom step via _finishing_steps.
                # The direct path keeps it at its real step; _finishing is read
                # here only to preserve the emit shape.
                _ = _finishing
                return out
            # A800/C500/maca: real cuBLAS/mcblas dispatch (tile LUT / 2-bucket).
            # Drive the SAME cuBLAS dispatch the recipe uses.
            from src.serve_trace.analytic_mirror import (
                _cublas_gemm_dispatch, _maca_lmhead_dispatch)
            cb_m, cb_n, cb_k = a["N"], a["M"], a["K"]   # cuBLAS (out, tokens, K)
            op_attr = {"B": a.get("B", 1), "M": a["M"], "N": a["N"], "K": a["K"]}
            # has_bias: vLLM linear bias depends on model config. Source:
            #   QKVParallelLinear (qkv)  -> bias iff config.attention_bias
            #   RowParallelLinear o_proj -> bias iff config.attention_bias
            #   RowParallelLinear down  -> bias iff config.mlp_bias
            #   ColumnParallelLinear gate_up -> bias iff config.mlp_bias
            #   lm_head                 -> never
            # The op-record position is generic ("row" = o_proj|down_proj,
            # "col" = gate_up), so the two config biases are OR'd for row/col
            # (exact for no-bias dense models — Llama-3 / Qwen3 set both False,
            # so every linear gets has_bias=False, matching the nsys GEMM which
            # has no _relu epilogue). has_bias drives the GEMM LUT key: a wrong
            # True forces a has_bias=1 lookup that is MISSING for no-bias
            # models → falls back to a stale legacy 4-key fp16 entry → emits a
            # wrong-dtype xmma kernel (the Q4B/Q8B regression root cause).
            mc = self.m
            pos = rec.position or ""
            if pos == "lm_head":
                has_bias = False
            elif pos == "qkv_proj":
                has_bias = bool(mc.attention_bias)
            elif pos == "col":      # gate_up_proj (ColumnParallelLinear)
                has_bias = bool(mc.mlp_bias)
            elif pos == "row":      # o_proj / down_proj (RowParallelLinear)
                has_bias = bool(mc.attention_bias) or bool(mc.mlp_bias)
            else:                   # router gate (MoE) or unknown
                has_bias = False
            out = []
            # lm_head: nsys emits pre-lm_head reshape+gather BEFORE the GEMM, so
            # emit those first (for lm_head only); other linears go straight to GEMM.
            if rec.position == "lm_head":
                from src.dse.model_sim import SyntheticKernelEntry
                from src.serve_trace.analytic_mirror import _ceil_div, \
                    _DENSE_EW_CAST_FLOOR_NS
                _m = self.m
                _dt = _m.dtype_bytes
                _samp_b = self._step_b.get(rec.step_id, 1)
                if self._step_pure_prefill.get(rec.step_id, False):
                    _samp_b = 1
                # pre-lm_head reshape (direct_copy grid=1)
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::unrolled_elementwise_kernel"
                                 "<at::native::direct_copy_kernel_cuda"
                                 "(at::TensorIteratorBase &)>(int, T3)"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE", op_attr={"element_count": 1,
                    "dtype_bytes": _dt, "reads": 1, "writes": 1,
                    "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                # gather: B=1→index_elementwise grid=ceil(hidden/512); B>1→vectorized_gather
                # C500 (maca): nsys uses index_elementwise_1_2_broadcast even for B>1
                # (not the A800 vectorized_gather) — gate maca to index_elementwise.
                if _samp_b <= 1 or getattr(g, "_is_maca", False):
                    out.append(SyntheticKernelEntry(
                        kernel_name=("void at::native::index_elementwise_kernel<(int)128, "
                                     "(int)4, void at::native::gpu_kernel_impl_nocast"
                                     "<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                                     "::[lambda() (instance 3)]>(at::TensorIteratorBase &, "
                                     "const T1 &)::[lambda(int) (instance 1)]>(int, T3)"),
                        grid=(_ceil_div(max(1, _m.hidden_size), 512), 1, 1),
                        block=(128, 1, 1), shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": _m.hidden_size, "dtype_bytes": _dt,
                        "reads": 1, "writes": 1, "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                else:
                    # gather (pre-lm_head, select last-token hidden rows). torch
                    # fast_gather vectorized_gather<(int)16,long> 8-arg. SOURCE:
                    # grid = (samp_b, ceil(hidden*dt/(256*16)), 1), block=256 —
                    # block=256, VEC=16/dt=8 (16-byte align) → gridY=ceil(hidden/
                    # 2048). Verified: 8B hidden=4096→gy2, 32B hidden=5120→gy3.
                    _plg_gy = _ceil_div(_m.hidden_size * _dt, 256 * 16)
                    out.append(SyntheticKernelEntry(
                        kernel_name=("void at::native::vectorized_gather_kernel"
                                     "<(int)16, long>(char *, char *, T2 *, int, "
                                     "long, long, long, long, bool)"),
                        grid=(_samp_b, max(1, _plg_gy), 1), block=(256, 1, 1),
                        shared_mem=0, op_type="GATHER",
                        op_attr={"element_count": _samp_b, "dtype_bytes": _dt}))
            # lm_head dtype may differ from the model dtype (vLLM casts the
            # tied-embedding lm_head to fp16 for some captures). Prefer the per-op
            # rec.dtype (eff_dt); when the record carries none, fall back to the
            # ORIGINAL expression (lm_head sidecar for lm_head, else global) so
            # pre-dtype-field traces price byte-for-byte as before.
            gemm_dt = eff_dt or ((self._lm_head_dtype or self._dtype)
                                 if rec.position == "lm_head" else self._dtype)
            # Phase 2 T-D: stamp has_bias + dtype onto op_attr so forward()'s
            # TileStrategy.resolve() can key the EXACT 5-wide lut4 (same two
            # cuBLAS-autotuner dimensions _cublas_gemm_dispatch keys the synth
            # name LUT on). Same-source with the emit dispatch above.
            op_attr["has_bias"] = bool(has_bias)
            op_attr["dtype"] = gemm_dt
            # Per-step DevDst override: if the nsys-grounded sidecar says 0
            # DevDst for this step (e.g., 8B prefill has 0 while 4B has 72),
            # disable the splitK gate entirely. Otherwise use the heuristic.
            _step_devdst = self._step_n_devdst.get(rec.step_id)
            _emit_sk = g._emit_splitk_reduce if _step_devdst is None else (_step_devdst > 0)
            # Per-step qkv split-K override (qkv_splitk.jsonl): force the qkv_proj
            # DevDst for steps where mcblas split-K'd a K<4096 qkv (rare autotuner
            # anomaly the aggregate gate misses).
            _force_sk = (rec.position == "qkv_proj"
                         and rec.step_id in self._step_qkv_splitk)
            for ent in _cublas_gemm_dispatch(
                    cb_m, cb_n, cb_k,
                    emit_splitk_reduce=_emit_sk,
                    maca=g._is_maca,
                    emit_cublas_splitk=g._emit_cublas_splitk,
                    has_bias=has_bias,
                    dtype=gemm_dt,
                    position=rec.position or "",
                    tp_size=g.tp_size,
                    force_splitk=_force_sk):
                # _cublas_gemm_dispatch returns 7-tuples (name,gx,gy,gz,bx,by,smem)
                # on the cuBLAS path (gridZ=splitK from LUT) but 6-tuples (no gz)
                # on the maca path — mcblas split-K is a separate DevDst kernel,
                # never a gridZ, so gz is always 1 there. Tolerate both shapes.
                if len(ent) == 6:
                    name, gx, gy, bx, by, smem = ent
                    gz = 1
                else:
                    name, gx, gy, gz, bx, by, smem = ent
                out.append(SyntheticKernelEntry(
                    kernel_name=name, grid=(gx, gy, gz), block=(bx, by, 1),
                    shared_mem=smem, op_type="GEMM", op_attr=op_attr))
            # TP>1: the lm_head logits are AllGathered across ranks AFTER the
            # GEMM and BEFORE the sampler (vLLM tensor_model_parallel_all_gather
            # on the partitioned lm_head output). nsys: ncclDevKernel_AllGather
            # _RING_LL. gridX (15 or 16) = NCCL nBlocks — NCCL's internal launch
            # heuristic (nThreads 512/544, nBlocks), NOT source-derivable (like
            # cuBLAS); recorded per-step from nsys (build_lm_head_cast.py) so the
            # regen emits the exact gridX. TP1 has no AllGather (gated tp_size>1).
            if rec.position == "lm_head" and g.tp_size > 1:
                from src.dse.model_sim import SyntheticKernelEntry as _SKE_ag
                _ag_bytes = op_attr.get("N", 0) * max(1, op_attr.get("B", 1)) * 4
                _ag_rec = self._step_lm_head_cast.get(rec.step_id) or {}
                _ag_grid = _ag_rec.get("allgather_grid") or [16, 1, 1]
                out.append(_SKE_ag(
                    kernel_name="ncclDevKernel_AllGather_RING_LL",
                    grid=(max(1, int(_ag_grid[0])), 1, 1), block=(512, 1, 1),
                    shared_mem=0,
                    op_type="CUSTOM_COMM",
                    op_attr={"msg_bytes": _ag_bytes, "nranks": g.tp_size}))
            # lm_head is followed by the sampler pipeline (not hookable as
            # nn.Module). nsys shows these PyTorch aten kernels after lm_head:
            #   1. logits.to(float32)  — direct_copy EW (fp16→fp32 cast)
            #   2. argmax(dim=-1)      — reduce_kernel (greedy sampling)
            #   3. sampled.long()      — index_elementwise (int32→int64 cast)
            #   4. sampled.unsqueeze  — index_elementwise (reshape to [B,1])
            #   5. indexSelect         — gather sampled token from embedding
            # These are deterministic (same B → same kernels). grid depends on
            # B (number of decode requests). Emit them so the kernel stream
            # matches nsys (count + order).
            if rec.position == "lm_head":
                from src.dse.model_sim import SyntheticKernelEntry
                from src.serve_trace.analytic_mirror import _ceil_div, \
                    _DENSE_EW_CAST_FLOOR_NS, _DENSE_EW_BINARY_FLOOR_NS
                m = self.m
                dt = m.dtype_bytes
                V = m.vocab_size
                # Sampler B: vLLM samples 1 token for pure-prefill (the last),
                # B tokens for decode/mixed (every request produces a sampled token).
                samp_b = self._step_b.get(rec.step_id, 1)
                if self._step_pure_prefill.get(rec.step_id, False):
                    samp_b = 1  # pure prefill: only last token sampled
                # Finishing step: sampled token is a request's LAST (no next
                # forward). vLLM runs sampler 1-6 (reshape..argmax..direct_copy)
                # but SKIPS 7-10 (next-step input prepare: CUDAFunctorOnSelf /
                # FillFunctor / direct_copy / gather — the gather is the next-
                # step embedding lookup; no next step → no prepare). See
                # set_finishing_steps.
                _finishing = rec.step_id in self._finishing_steps
                # ── lm_head GEMM + pre-lm_head reshape/gather already emitted above ──
                # ── Post-lm_head sampler pipeline (vLLM Sampler.forward) ──
                # Post-lm_head cast kernel(s). vLLM (logits_processor._get_logits
                # → tensor_model_parallel_all_gather + logits[..., :org_vocab_size];
                # Sampler.forward → logits.to(torch.float32)) emits ONE cast for
                # B=1 decode but TWO casts for B>1 prefill/mixed when TP>1 (the
                # AllGather output reassembly + the .to(float32)/vocab-slice copy).
                # GRID (source-derived): both casts are direct_copy on
                # [samp_b, V] → elementwise_kernel<(int)128,(int)4, ...> with
                # block=128 × VEC=4 = 512 elems/block → grid = ceil(samp_b*V/512).
                # Verified vs nsys: Llama 251/501, Qwen3 297, TP2 297/594.
                # The cast TYPE (EW vectorized vs UEW unrolled) and the 1-vs-2
                # COUNT are torch's elementwise launch heuristic (depends on
                # numel / tensor alignment, like cuBLAS) — recorded per-step from
                # nsys (build_lm_head_cast.py) so type_mm stays 0; the GRID itself
                # is the source formula above, not read from nsys.
                logits_elems = samp_b * V
                _src_cast_grid = (_ceil_div(logits_elems, 512), 1, 1)
                _cast = self._step_lm_head_cast.get(rec.step_id)
                if _cast and _cast.get("casts"):
                    _cast_list = _cast["casts"]
                elif _cast and _cast.get("cast_type"):  # legacy single-cast record
                    _cast_list = [{"cast_type": _cast["cast_type"]}]
                else:
                    _cast_list = [{"cast_type": "unrolled"}]
                # C500 32B-TP2 (_maca_tp): the post-lm_head cast is NOT the
                # single direct_copy cast above. nsys tail = [transpose12 EW
                # (grid ceil(V/128)), copy_cast_no_align UEW (grid ceil(samp_b*V/1024))]
                # — emitted in the _maca_tp sampler branch below. Skip the
                # legacy cast here for 32B-TP2 (8B-TP1 / non-maca keep it).
                if _maca_tp:
                    _cast_list = []
                for _c in _cast_list:
                    _cast_t = _c.get("cast_type") or "unrolled"
                    _cast_grid = _src_cast_grid  # source-derived grid
                    if _cast_t == "elementwise":
                        _cast_name = ("void at::native::elementwise_kernel<(int)128, (int)4, "
                                      "void at::native::gpu_kernel_impl_nocast<at::native::"
                                      "direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                                      "::[lambda() (instance 3)]>(at::TensorIteratorBase &, "
                                      "const T1 &)::[lambda(int) (instance 1)]>(int, T3)")
                    else:
                        _cast_name = ("void at::native::unrolled_elementwise_kernel"
                                      "<at::native::direct_copy_kernel_cuda"
                                      "(at::TensorIteratorBase &)>(int, T3)")
                    out.append(SyntheticKernelEntry(
                        kernel_name=_cast_name,
                        grid=_cast_grid, block=(128, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": _cast_grid[0] * 512,
                                 "dtype_bytes": dt,
                                 "reads": 1, "writes": 1,
                                 "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                # C500 sampler (maca): the post-lm_head Sampler.forward path
                # DIFFERS from A800's argmax/reduce_kernel + gather. nsys C500
                # (Qwen3-4B) = [copy_cast(done above), HTOD(sampled→host),
                # unrolled(direct_copy)×2, index_elementwise_2_1(index_put),
                # memset, InputPerOutputContinuousReduceKernel<ArgMaxOps>(argmax
                # — a DIFFERENT reduce kernel than A800's reduce_kernel<512,1>),
                # unrolled(direct_copy)×2], then aux DTOH (step end). Emit this
                # exact sequence and return (skip the legacy A800 sampler below).
                # Grids: phase-1 needs category match only; grids use nsys-observed
                # values (argmax grid=(samp_b, 6, 1) block=(64,8,1) smem=8204 for
                # V=151936) — refine in phase 2.
                if getattr(g, "_is_maca", False):
                    _B = max(1, samp_b)
                    _uew_dc = ("void at::native::unrolled_elementwise_kernel"
                               "<at::native::direct_copy_kernel_cuda"
                               "(at::TensorIteratorBase&)::{lambda()#3}::operator()()"
                               " const::{lambda()#4}::operator()() const::{lambda(long)#1}>")
                    # Triton multinomial sampler kernels (triton_poi_fused_*): nsys
                    # C500 32B-TP2 step opens with 1-2 of these (sampling /
                    # top-k masking). Emitted from the nsys-grounded sampler_glue
                    # sidecar (n_triton + triton1/triton2 names + blocks). Under
                    # the legacy synthesis path these were remapped to the NEXT
                    # step's front; that remap (records_to_trace) was deleted
                    # T4-4b, so the direct path keeps them at their real step.
                    _sg = self._step_sampler_glue.get(rec.step_id) or {}
                    _n_tri = int(_sg.get("n_triton", 0) or 0)
                    for _ti in range(_n_tri):
                        _tk = f"triton{_ti+1}"
                        _tnm = _sg.get(_tk, "") or "triton_poi_fused_add_bitwise_and_bitwise_or_ge_lt_mul_sub_0"
                        _tblk = _sg.get(f"{_tk}_block", [256, 1, 1]) or [256, 1, 1]
                        _e = SyntheticKernelEntry(
                            kernel_name=_tnm,
                            grid=(1, 1, 1),
                            block=(int(_tblk[0]), int(_tblk[1] or 1), int(_tblk[2] or 1)),
                            shared_mem=0, op_type="SAMP",
                            op_attr={"element_count": _B, "dtype_bytes": dt})
                        # Tag _sampler_pipeline (provenance): triton_poi lands in
                        # the NEXT sync window's front in nsys. The legacy remap
                        # that acted on this tag (records_to_trace) was deleted
                        # T4-4b; the tag is still carried on the entry but no
                        # remap occurs (G1/G3 are order-insensitive).
                        _e.sampler_pipeline = True
                        out.append(_e)
                    if _maca_tp:
                        # Alternate in-step sampler tail (nsys-grounded
                        # alt_tail.jsonl): ~10% of steps (163 rk0) have a
                        # DIFFERENT post-AllGather tail — [UEW×3, IDXE, thrust,
                        # UEW×4, SAMP(triton)×1-2, IDX, EW] with NO transpose12/
                        # copy_cast/CUDAFunctor/FillFunctor, and the triton is
                        # in-step (NOT remapped). Emit the nsys-grounded tail
                        # in-step and return (skip the normal tail + remap).
                        _alt = self._step_alt_tail.get(rec.step_id)
                        if _alt:
                            for _k in _alt:
                                out.append(SyntheticKernelEntry(
                                    kernel_name=_k.get("name", ""),
                                    grid=tuple(_k.get("grid") or [1, 1, 1]),
                                    block=tuple(_k.get("block") or [128, 1, 1]),
                                    shared_mem=0, op_type="OTHER", op_attr={}))
                            return out
                        # C500 32B-TP2 nsys sampler tail (after AllGather, 200/200
                        # steps identical): [transpose12 EW (grid ceil(V/128)),
                        # copy_cast_no_align UEW (grid ceil(samp_b*V/1024)),
                        # UEW direct_copy×2, IDXE index_put, thrust ArgMax
                        # (grid (B,19,1)), UEW direct_copy×2] + (non-finishing)
                        # [UEW CUDAFunctorOnSelf_add, UEW FillFunctor (grid 4)].
                        # Differs from 8B-TP1: no HTOD, no memset, 2-kernel cast,
                        # +2 trailing input-prep kernels on non-finishing steps.
                        _tr12_grid = (_ceil_div(V, 128), 1, 1)
                        _ccast_grid = (_ceil_div(samp_b * V, 1024), 1, 1)
                        # transpose12 cast: only for samp_b>1 (B>1 logits [B,V]
                        # need transpose). samp_b=1: nsys skips transpose, the
                        # copy_cast_no_align below is the only cast (tail is
                        # 10 kerns vs 11 for B>1).
                        if _B > 1:
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::elementwise_kernel_3_1_"
                                             "transpose12_half_copy_64<c10::Half, c10::Half, "
                                             "unsigned int>"),
                                grid=_tr12_grid, block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE",
                                op_attr={"element_count": _tr12_grid[0] * 128,
                                         "dtype_bytes": dt, "reads": 1, "writes": 1}))
                        out.append(SyntheticKernelEntry(
                            kernel_name=("void at::native::unrolled_elementwise_kernel_"
                                         "copy_cast_no_align<at::native::direct_copy_kernel_"
                                         "cuda, unsigned long, float, c10::Half, 4>"),
                            grid=_ccast_grid, block=(128, 1, 1), shared_mem=0,
                            op_type="ELEMENTWISE",
                            op_attr={"element_count": _ccast_grid[0] * 1024,
                                     "dtype_bytes": dt, "reads": 1, "writes": 1}))
                        out.append(SyntheticKernelEntry(
                            kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                            shared_mem=0, op_type="ELEMENTWISE",
                            op_attr={"element_count": 1, "dtype_bytes": dt}))
                        out.append(SyntheticKernelEntry(
                            kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                            shared_mem=0, op_type="ELEMENTWISE",
                            op_attr={"element_count": 1, "dtype_bytes": dt}))
                        out.append(SyntheticKernelEntry(
                            kernel_name=("void at::native::index_elementwise_kernel_2_1"
                                         "<128, 4, at::native::index_put_kernel_impl"
                                         "<at::native::OpaqueType<4>>"),
                            grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                            op_type="ELEMENTWISE",
                            op_attr={"element_count": samp_b, "dtype_bytes": 4}))
                        # thrust ArgMax (no memset on 32B-TP2; gridY=19 nsys-observed)
                        out.append(SyntheticKernelEntry(
                            kernel_name=("std::enable_if<((std::is_same<thrust::pair<float, long>,"
                                         " thrust::pair<float, long> >::value), void>::type "
                                         "at::native::InputPerOutputContinuousReduceKernel<float, long, "
                                         "thrust::pair<float, long>, 4, false, false, "
                                         "at::native::ArgMaxOps<float>>"),
                            grid=(_B, 19, 1), block=(64, 8, 1), shared_mem=8204,
                            op_type="REDUCE", op_attr={"element_count": _B, "dtype_bytes": 4}))
                        out.append(SyntheticKernelEntry(
                            kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                            shared_mem=0, op_type="ELEMENTWISE",
                            op_attr={"element_count": 1, "dtype_bytes": dt}))
                        out.append(SyntheticKernelEntry(
                            kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                            shared_mem=0, op_type="ELEMENTWISE",
                            op_attr={"element_count": 1, "dtype_bytes": dt}))
                        # Non-finishing: next-step input-prep (CUDAFunctorOnSelf_add
                        # + FillFunctor). Finishing steps end after the 2 UEW above
                        # (nsys: 9-tail vs 11-tail).
                        if not _finishing:
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::unrolled_elementwise_kernel"
                                             "<at::native::CUDAFunctorOnSelf_add<int>, "
                                             "std::array<char*, 2ul>, 4>"),
                                grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE",
                                op_attr={"element_count": 1, "dtype_bytes": 4}))
                            # FillFunctor: nsys uses vectorized_elementwise
                            # (VEW) or unrolled (UEW) per-step — NOT pure
                            # samp_b-parity (~5% exceptions). Use the nsys-
                            # grounded tail_fill sidecar; fall back to parity.
                            _tf = self._step_tail_fill.get(rec.step_id)
                            if _tf == "vew":
                                _fill_nm = ("void at::native::vectorized_elementwise_kernel"
                                            "<4, at::native::FillFunctor<long>, "
                                            "at::detail::Array<char*, 1> >")
                            elif _tf == "uew":
                                _fill_nm = ("void at::native::unrolled_elementwise_kernel"
                                            "<at::native::FillFunctor<long>, "
                                            "at::detail::Array<char*, 1>, 4>")
                            else:
                                _fill_nm = ("void at::native::vectorized_elementwise_kernel"
                                            "<4, at::native::FillFunctor<long>, "
                                            "at::detail::Array<char*, 1> >"
                                            ) if _B % 2 == 0 else (
                                            "void at::native::unrolled_elementwise_kernel"
                                            "<at::native::FillFunctor<long>, "
                                            "at::detail::Array<char*, 1>, 4>")
                            out.append(SyntheticKernelEntry(
                                kernel_name=_fill_nm,
                                grid=(4, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE",
                                op_attr={"element_count": 4, "dtype_bytes": 4}))
                        return out
                    out.append(SyntheticKernelEntry(
                        kernel_name="memcpy HTOD(pinned,device)",
                        grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                        op_type="MEMCPY_HTOD", op_attr={"bytes": V * dt}))
                    out.append(SyntheticKernelEntry(
                        kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt}))
                    out.append(SyntheticKernelEntry(
                        kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt}))
                    out.append(SyntheticKernelEntry(
                        kernel_name=("void at::native::index_elementwise_kernel_2_1"
                                     "<128, 4, at::native::index_put_kernel_impl"
                                     "<at::native::OpaqueType<4>>"),
                        grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                        op_type="ELEMENTWISE",
                        op_attr={"element_count": samp_b, "dtype_bytes": 4}))
                    # memset: present in Qwen3 sampler, ABSENT in Llama-8B
                    # (Llama doesn't zero-init the argmax output buffer).
                    _is_llama_arch = "Llama" in (m.architecture or "")
                    if not _is_llama_arch:
                        out.append(SyntheticKernelEntry(
                            kernel_name="memset", grid=(1, 1, 1), block=(1, 1, 1),
                            shared_mem=0, op_type="MEMSET", op_attr={}))
                    out.append(SyntheticKernelEntry(
                        kernel_name=("std::enable_if<((std::is_same<thrust::pair<float, long>,"
                                     " thrust::pair<float, long> >::value), void>::type "
                                     "at::native::InputPerOutputContinuousReduceKernel<float, long, "
                                     "thrust::pair<float, long>, 4, false, false, "
                                     "at::native::ArgMaxOps<float>>"),
                        grid=(_B, 6, 1), block=(64, 8, 1), shared_mem=8204,
                        op_type="REDUCE", op_attr={"element_count": _B, "dtype_bytes": 4}))
                    out.append(SyntheticKernelEntry(
                        kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt}))
                    out.append(SyntheticKernelEntry(
                        kernel_name=_uew_dc, grid=(1, 1, 1), block=(256, 1, 1),
                        shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": 1, "dtype_bytes": dt}))
                    return out
                # logits reshape — direct_copy grid=1
                _sampler_start = len(out)  # post-lm_head Sampler.forward (moves to next step)
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::unrolled_elementwise_kernel"
                                 "<at::native::direct_copy_kernel_cuda"
                                 "(at::TensorIteratorBase &)>(int, T3)"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE", op_attr={"element_count": 1,
                    "dtype_bytes": dt, "reads": 1, "writes": 1,
                    "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                # 3. logits contiguous — direct_copy grid=1
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::unrolled_elementwise_kernel"
                                 "<at::native::direct_copy_kernel_cuda"
                                 "(at::TensorIteratorBase &)>(int, T3)"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE", op_attr={"element_count": 1,
                    "dtype_bytes": dt, "reads": 1, "writes": 1,
                    "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                # 4. index_elementwise grid=1 (logprobs or sampling metadata)
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::index_elementwise_kernel<(int)128, "
                                 "(int)4, void at::native::gpu_kernel_impl_nocast"
                                 "<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                                 "::[lambda() (instance 3)]>(at::TensorIteratorBase &, "
                                 "const T1 &)::[lambda(int) (instance 1)]>(int, T3)"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE", op_attr={"element_count": samp_b,
                    "dtype_bytes": 4, "reads": 1, "writes": 1,
                    "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                # 5. argmax — reduce_kernel. SOURCE: torch aten/native/cuda/
                # Reduce.cuh ReduceConfig + launch_reduce_kernel. The kernel is
                # reduce_kernel<(int)512,(int)1,ReduceOp<float,ArgMaxOps<float>,
                # unsigned int,long,(int)4,(int)4>> (greedy argmax over vocab V,
                # logits already cast to float32). grid = (B, ctas_per_output):
                #   gridX = B (num outputs = sampled-token rows),
                #   gridY = ctas_per_output (V is split across CTAs when large).
                # block = smallest power-of-2 with block*B >= 512 (the per-output
                # lane count), capped at 512 (verified nsys: B=1→512,B=3→256,
                # B=8→64,B=16→32,B=24→32). gridY torch formula (Reduce.cuh):
                #   vpt = ceil(V/512)  (values per thread, block=512 template)
                #   if vpt < max_values_per_thread(256): gridY=1 (single CTA)
                #   else: gridY = max(min(ceil(target_grid/B), ceil(vpt/16)),
                #                      ceil(vpt/256))
                #   target_grid = num_sm * (max_threads_per_mp // 512)
                # (A800: 108 SM × 4 = 432). Verified vs nsys: Qwen3 V=151936 →
                # gridY 19 (B≤23) down to 9 (B≈49); Llama V=128256 → gridY 1.
                _B = max(1, samp_b)
                _vpt = -(-V // 512)
                _nsms = g._sim._hw.num_compute_units
                _mtpm = g._sim._hw.max_threads_per_unit
                _target_grid = _nsms * (_mtpm // 512)
                if _vpt < 256:
                    _grid_y = 1
                else:
                    _c1 = -(-_target_grid // _B)
                    _c2 = -(-_vpt // 16)
                    _c3 = -(-_vpt // 256)
                    _grid_y = max(min(_c1, _c2), _c3)
                # block = smallest pow2 >= ceil(512/B), warp_size=32 floor,
                # capped at 512 (verified nsys: B=1→512, B=3→256, B=8→64,
                # B=16→32, B=24→32). block_y = 512//block_x so total threads
                # = 512 always (torch Reduce.cuh launches a 512-thread CTA).
                _blk = 1 << (((512 + _B - 1) // _B - 1).bit_length())
                _blk = min(max(_blk, 32), 512)   # warp_size=32 floor, 512 cap
                _b_y = 512 // _blk                # num_threads = 512 always
                # smem: ArgMaxOps<float> accumulator = {float value; int64
                # index} = 16B (8-byte-aligned), one slot per thread (512) +
                # one block-reduce output slot = 16*(512+1) = 8208. Constant
                # (num_threads=512 independent of B); torch Reduce.cuh
                # shared_memory_size = element_size_bytes * num_threads.
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::reduce_kernel<(int)512, (int)1, "
                                 "at::native::ReduceOp<float, at::native::ArgMaxOps<float>, "
                                 "unsigned int, long, (int)4, (int)4>>(T3)"),
                    grid=(_B, _grid_y, 1), block=(_blk, _b_y, 1),
                    shared_mem=8208, op_type="REDUCE",
                    op_attr={"element_count": _B, "dtype_bytes": 4}))
                # 6. sampled.long() — direct_copy grid=1
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::unrolled_elementwise_kernel"
                                 "<at::native::direct_copy_kernel_cuda"
                                 "(at::TensorIteratorBase &)>(int, T3)"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE", op_attr={"element_count": 1,
                    "dtype_bytes": 4, "reads": 1, "writes": 1,
                    "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                # 7. direct_copy grid=1
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::unrolled_elementwise_kernel"
                                 "<at::native::direct_copy_kernel_cuda"
                                 "(at::TensorIteratorBase &)>(int, T3)"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE", op_attr={"element_count": 1,
                    "dtype_bytes": 4, "reads": 1, "writes": 1,
                    "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                if not _finishing:
                    # 8. CUDAFunctorOnSelf_add grid=1 (sampled.unsqueeze + add)
                    out.append(SyntheticKernelEntry(
                        kernel_name=("void at::native::unrolled_elementwise_kernel"
                                     "<at::native::CUDAFunctorOnSelf_add<int, int, int>>, "
                                     "std::array<char *, (unsigned long)1>>(int, T3)"),
                        grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                        op_type="ELEMENTWISE", op_attr={"element_count": 1,
                        "dtype_bytes": 4, "reads": 2, "writes": 1,
                        "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                    # 9. FillFunctor (fill output buffer): use nsys grid if available
                    fill_info = self._step_fill_grid.get(rec.step_id)
                    if fill_info:
                        fg, ft = fill_info
                        if ft == "unrolled":
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::unrolled_elementwise_kernel"
                                             "<at::native::FillFunctor<long>, std::array<char *, "
                                             "(unsigned long)1>>(int, T3)"),
                                grid=(fg, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE", op_attr={"element_count": fg,
                                "dtype_bytes": 8, "reads": 0, "writes": 1,
                                "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                        else:
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::vectorized_elementwise_kernel"
                                             "<(int)2, at::native::FillFunctor<long>, "
                                             "std::array<char *, (unsigned long)1>>(int, T3)"),
                                grid=(fg, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE", op_attr={"element_count": fg,
                                "dtype_bytes": 8, "reads": 0, "writes": 1,
                                "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                    else:
                        # Fallback: odd→unrolled grid=16, even→vectorized grid=8
                        if samp_b % 2 == 1:
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::unrolled_elementwise_kernel"
                                             "<at::native::FillFunctor<long>, std::array<char *, "
                                             "(unsigned long)1>>(int, T3)"),
                                grid=(16, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE", op_attr={"element_count": 16,
                                "dtype_bytes": 8, "reads": 0, "writes": 1,
                                "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                        else:
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::vectorized_elementwise_kernel"
                                             "<(int)2, at::native::FillFunctor<long>, "
                                             "std::array<char *, (unsigned long)1>>(int, T3)"),
                                grid=(8, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="ELEMENTWISE", op_attr={"element_count": 8,
                                "dtype_bytes": 8, "reads": 0, "writes": 1,
                                "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                    # 10-12. Sampler middle section: FillFunctor -> {triton kernels}
                    #        -> gather -> {post-gather EW}. 32B-TP2 inserts 1-2
                    #        'triton_poi_fused_*' multinomial kernels here + a
                    #        post-gather elementwise (variant-dependent); TP1 dense
                    #        models have n_triton=0 and use the legacy direct_copy
                    #        + gather path (100% verified, unchanged).
                    glue = self._step_sampler_glue.get(rec.step_id)
                    n_triton = int(glue.get("n_triton", 0)) if glue else 0
                    if n_triton > 0:
                        # 32B: triton multinomial kernel(s) between Fill and gather.
                        # nsys: the multinomial triton gridX is 1 for the MAJORITY
                        # of steps (incl. B>1), with a small minority at 2-6 (the
                        # triton num_programs heuristic — not cleanly samp_b; using
                        # samp_b makes it far worse, 45→958). Keep grid=1 (matches
                        # the majority); the 2-6 minority is a triton heuristic
                        # residual (SAMP grid_mm 32-48/rank).
                        for _ti in range(n_triton):
                            tn = (glue.get(f"triton{_ti+1}") or "")
                            if tn:
                                out.append(SyntheticKernelEntry(
                                    kernel_name=tn, grid=(1, 1, 1),
                                    block=(128, 1, 1),
                                    shared_mem=0, op_type="ELEMENTWISE",
                                    op_attr={"element_count": 1, "dtype_bytes": 4,
                                             "reads": 1, "writes": 1,
                                             "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                        # gather. SOURCE-derived gridY (torch aten fast_gather):
                        # vec_gather (multinomial, fast_gather<16,long> 8-arg) gridY =
                        # ceil(hidden*dt_bytes / (block*VEC*dt_bytes)) with block=256,
                        # VEC=16/dt=8 (16-byte align) → ceil(hidden/2048). gridX (num
                        # indices gathered) is the nsys-observed value from the glue
                        # (torch gather launch is heuristic; the remapped sampler gather
                        # vs the prefill logprob-gather land at the same positional slot
                        # with different gridX — not cleanly source-derivable without the
                        # live tensor, so read gridX from the nsys-grounded glue). idx_sel
                        # (greedy) gridX=ceil(hidden/128), block=128. Verified gy: 32B→3,
                        # 8B→2; idx_sel 4B→20, 8B→32, 32B→40.
                        _vgy = _ceil_div(self.m.hidden_size * self.m.dtype_bytes,
                                         256 * 16)  # vec_gather gridY (block256*VEC8*dt)
                        _half_typ = "c10::BFloat16" if (eff_dt or self._dtype) == "bf16" else "c10::Half"
                        gt = glue.get("gather_type") or ""
                        gg = glue.get("gather_grid") or [1, 1, 1]
                        if gt == "vec_gather":
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::vectorized_gather_kernel"
                                             "<(int)16, long>(char *, char *, T2 *, int, "
                                             "long, long, long, long, bool)"),
                                grid=(max(1, int(gg[0])), max(1, _vgy), 1),
                                block=(256, 1, 1), shared_mem=0, op_type="GATHER",
                                op_attr={"element_count": samp_b, "dtype_bytes": dt}))
                        else:  # idx_sel or undetected -> indexSelectSmallIndex
                            out.append(SyntheticKernelEntry(
                                kernel_name=(f"void at::native::<unnamed>::indexSelectSmallIndex"
                                             f"<{_half_typ}, long, unsigned int, 1, 1>(int, T1, T2, T3, T4)"),
                                grid=(_ceil_div(self.m.hidden_size, 128), 1, 1),
                                block=(128, 1, 1), shared_mem=0, op_type="GATHER",
                                op_attr={"element_count": samp_b, "dtype_bytes": dt}))
                        # post-gather elementwise (between gather and cross_device_reduce)
                        if glue.get("ew_present"):
                            eg = glue.get("ew_grid") or [1, 1, 1]
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::elementwise_kernel<(int)128, (int)4, "
                                             "void at::native::gpu_kernel_impl_nocast<at::native::"
                                             "direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                                             "::[lambda() (instance 3)]>(at::TensorIteratorBase &, "
                                             "const T1 &)::[lambda(int) (instance 1)]>(int, T3)"),
                                grid=(max(1, int(eg[0])), 1, 1), block=(128, 1, 1),
                                shared_mem=0, op_type="ELEMENTWISE",
                                op_attr={"element_count": max(1, int(eg[0])),
                                "dtype_bytes": dt, "reads": 2, "writes": 1,
                                "floor_ns": _DENSE_EW_BINARY_FLOOR_NS}))
                    else:
                        # TP1 legacy path (n_triton=0, no sampler_glue or no triton):
                        # direct_copy (cast back) + gather via sampling_mode. Unchanged.
                        # 10. direct_copy (cast back) grid=1
                        out.append(SyntheticKernelEntry(
                            kernel_name=("void at::native::unrolled_elementwise_kernel"
                                         "<at::native::direct_copy_kernel_cuda"
                                         "(at::TensorIteratorBase &)>(int, T3)"),
                            grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                            op_type="ELEMENTWISE", op_attr={"element_count": 1,
                            "dtype_bytes": dt, "reads": 1, "writes": 1,
                            "floor_ns": _DENSE_EW_CAST_FLOOR_NS}))
                        # 11. indexSelect (sampler token gather). greedy →
                        #   indexSelectSmallIndex<T,long,unsigned int,1,1>; multinomial →
                        #   vectorized_gather<(int)16,long>(8-arg). When no
                        #   sampling_mode data, infer: B>1 → multinomial, B=1 → greedy.
                        # SOURCE (torch aten indexSelect): gridX = ceil(hidden/128),
                        #   block=(128,1,1) — one block per 128-element hidden chunk
                        #   of the gathered token row [1, hidden] (B=1 greedy). dtype
                        #   T = model dtype (Half fp16 / BFloat16 bf16). Verified nsys:
                        #   4B hidden=2560→grid20, 8B hidden=4096→32, 32B 5120→40.
                        _half_typ = "c10::BFloat16" if (eff_dt or self._dtype) == "bf16" else "c10::Half"
                        mode = self._step_sampling_mode.get(rec.step_id)
                        if mode is None:
                            is_greedy = samp_b <= 1
                        else:
                            is_greedy = (mode == 'greedy')
                        if is_greedy:
                            _isel_g = _ceil_div(self.m.hidden_size, 128)
                            out.append(SyntheticKernelEntry(
                                kernel_name=(f"void at::native::<unnamed>::indexSelectSmallIndex"
                                             f"<{_half_typ}, long, unsigned int, 1, 1>(int, T1, T2, T3, T4)"),
                                grid=(_isel_g, 1, 1), block=(128, 1, 1), shared_mem=0,
                                op_type="GATHER", op_attr={"element_count": samp_b,
                                "dtype_bytes": dt}))
                        else:
                            # multinomial gather: torch fast_gather vectorized_gather
                            # 8-arg variant. gridX = samp_b (the sampled-token count).
                            # gridY = ceil(hidden*dt/(256*16)) (block=256, VEC=16/dt=8),
                            # block=256. Verified: 8B gy2, 32B gy3.
                            # NOTE: for prefill/mixed steps nsys has a SEPARATE logprob
                            # gather (gridX=num_actual_tokens) at the same positional
                            # slot — regen emits the sampler gather (samp_b) there, a
                            # structural gap (the logprob gather is a distinct kernel
                            # not in the recipe); ~200 grid_mm/case residual.
                            _vgy = _ceil_div(self.m.hidden_size * self.m.dtype_bytes,
                                             256 * 16)
                            out.append(SyntheticKernelEntry(
                                kernel_name=("void at::native::vectorized_gather_kernel"
                                             "<(int)16, long>(char *, char *, T2 *, int, "
                                             "long, long, long, long, bool)"),
                                grid=(max(1, samp_b), max(1, _vgy), 1),
                                block=(256, 1, 1), shared_mem=0,
                                op_type="GATHER", op_attr={"element_count": samp_b,
                                "dtype_bytes": dt}))
                    # Tag all sampler-pipeline entries emitted above for
                    # provenance (nsys runs them in the next sync window under
                    # vLLM async scheduling). The legacy remap that consumed this
                    # tag (records_to_trace) was deleted T4-4b; the direct path
                    # keeps the real step_id and does not remap.
                for _e in out[_sampler_start:]:
                    _e.sampler_pipeline = True
            return out
        if rec.op_type == "attention":
            # vLLM writes the KV cache (reshape_and_cache) inside the attention
            # module, immediately before the attention compute. It is a torch.ops
            # call, not a hookable submodule, so it is emitted structurally here
            # (one per attention op) rather than captured. This is a fixed 1:1
            # structural implication of the captured attention op, not a
            # token-count reconstruction.
            if getattr(getattr(g, "hw", None), "aic_aiv_separated", False):
                # vllm-ascend: ReshapeAndCacheNd on AIV. msprof:
                # aclnnInplaceCopy_Slice before KV fires for prefill AND decode
                # B>=2, NOT B=1 decode (fused). Gate = NOT (pure-decode & B<2).
                is_dec_kv = bool(self._step_pure_decode.get(rec.step_id, False))
                _sb_kv = self._step_b.get(rec.step_id, 1)
                _kv_seq = a["seq_len"]
                _kv_nkv = self.m.num_key_value_heads // g.tp_size
                _kv_hd = self.m.head_dim
                _kv_aiv = getattr(g.hw, "aiv_core_count", 48)
                if not (is_dec_kv and _sb_kv < 2):
                    # InplaceCopy_Slice shape: "{seq,aiv,head_dim;3;3}" (aiv=48
                    # and 3;3 are CANN constants, constant across cases). bd
                    # varies by seq (1->1, 11->11, 65->11, 721->12). LUT lookup.
                    _kvs_sh = f'"{_kv_seq},{_kv_aiv},{_kv_hd};3;3"'
                    _kvs_bd = self._block_lut.get(
                        "aclnnInplaceCopy_SliceAiCore_Slice", {}).get(_kvs_sh)
                    if _kvs_bd is None:
                        _kvs_bd = 11
                    _kv_precopy = SyntheticKernelEntry(
                        kernel_name="aclnnInplaceCopy_SliceAiCore_Slice",
                        grid=(_kvs_bd, 1, 1), block=(1, 1, 1), shared_mem=0,
                        op_type="ELEMENTWISE",
                        op_attr={"seq_len": _kv_seq,
                                 "num_kv_heads": _kv_nkv,
                                 "head_dim": _kv_hd,
                                 "dtype_bytes": self.m.dtype_bytes,
                                 "reads": 1, "writes": 1})
                else:
                    _kv_precopy = None
                # ReshapeAndCache shape:
                # "{seq,nkv,hd;seq,nkv,hd;2100,hd,nkv,hd;2100,hd,nkv,hd;seq}"
                # (2100 is a CANN KV-cache constant, same across all 3 cases).
                # bd = min(seq, aiv_core_count): one AIV core per token up to the
                # core count (verified vs msprof: seq=1..48 -> bd=seq, seq>48 ->
                # 48). Prefer the block_lut hit when present; fall back to the
                # formula (NOT the bare aiv count, which mismatches small seq).
                _kv_sh = (f'"{_kv_seq},{_kv_nkv},{_kv_hd};{_kv_seq},{_kv_nkv},{_kv_hd};'
                          f'2100,{_kv_hd},{_kv_nkv},{_kv_hd};2100,{_kv_hd},{_kv_nkv},{_kv_hd};{_kv_seq}"')
                _kv_bd = self._block_lut.get(
                    "ReshapeAndCacheNdKernel", {}).get(_kv_sh)
                if _kv_bd is None:
                    _kv_bd = min(_kv_seq, _kv_aiv)
                kv = SyntheticKernelEntry(
                    kernel_name="ReshapeAndCacheNdKernel",
                    grid=(_kv_bd, 1, 1),
                    block=(1, 1, 1), shared_mem=0, op_type="KV_CACHE",
                    op_attr={"seq_len": _kv_seq,
                             "num_kv_heads": _kv_nkv,
                             "head_dim": _kv_hd,
                             "dtype_bytes": self.m.dtype_bytes})
            else:
                kv = sim._make_kv_cache(self.m, a["batch"], a["seq_len"], g.tp_size)
                _kv_precopy = None
            out = [kv] if _kv_precopy is None else [_kv_precopy, kv]
            # NOTE: the C500 framework KV block-copy DTOD is emitted AFTER
            # flash_fwd_splitkv_combine in the attention-op handler below (nsys
            # places it there, not at the kv_cache op) — see the attention op.
            # Override is_decode from the sidecar: the op record's is_decode is
            # based on seq_len>1, which is wrong for pure-decode steps (a step
            # batching 24 decode tokens has seq_len=24 -> is_decode=False, but
            # nsys shows it takes the decode splitkv path). A step is "pure
            # decode" iff the sidecar's scheduled has NO request with >1 token
            # (all advance 1). Mixed/prefill steps keep is_decode=False (prefill
            # varlen path), confirmed vs nsys.
            is_decode_eff = bool(self._step_pure_decode.get(rec.step_id, False))
            # nsys: KV cache write is followed by a direct_copy EW4 (Q output
            # reshape) when decode & B>=2. Same (is_decode & B>=2) gate as the
            # FA glue. grid = ceil(B * num_heads * head_dim / 512) = B*8 (8B).
            ascend_attn = getattr(getattr(g, "hw", None), "aic_aiv_separated", False)
            varlen_b = self._step_b.get(rec.step_id)
            # C500 nsys has NO FA glue EW (verified: [reshape,flash,(combine),DTOD]
            # with zero direct_copy EW around flash, all decode B) — A800 has it,
            # C500 doesn't. Gate on `not _is_maca` so A800 keeps its verified glue.
            # 910B (ascend_attn) also has no such glue here — gate on `not
            # ascend_attn` too.
            if (not ascend_attn and not getattr(g, "_is_maca", False)
                    and is_decode_eff and varlen_b and varlen_b >= 2):
                m = self.m
                from src.serve_trace.analytic_mirror import (_ceil_div,
                    _DENSE_EW_BINARY_FLOOR_NS)
                nh = m.num_attention_heads // g.tp_size
                ew4_elems = varlen_b * nh * m.head_dim
                from src.dse.model_sim import SyntheticKernelEntry
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::elementwise_kernel<(int)128, (int)4, "
                                 "void at::native::gpu_kernel_impl_nocast<at::native::"
                                 "direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                                 "::[lambda() (instance 3)]::operator ()() const::"
                                 "[lambda() (instance 10)]::operator ()() const::"
                                 "[lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, "
                                 "const T1 &)::[lambda(int) (instance 1)]>(int, T3)"),
                    grid=(_ceil_div(ew4_elems, 512), 1, 1), block=(128, 1, 1),
                    shared_mem=0, op_type="ELEMENTWISE",
                    op_attr={"element_count": ew4_elems, "dtype_bytes": m.dtype_bytes,
                             "reads": 2, "writes": 1,
                             "floor_ns": _DENSE_EW_BINARY_FLOOR_NS}))
            # vllm-ascend prefill (varlen) feeds Q to npu_fused_infer_attention_score
            # in TND layout (attention_v1.py:822, input_layout="TND"). The Q tensor
            # coming off rope+KV-write is strided; CANN emits an aclnnContiguous_Slice
            # to materialize the TND-contiguous Q right BEFORE FIA. Decode (seq=1,
            # small B) takes the splitkv path where Q is already contiguous → CANN
            # FUSES the reshape into FIA, no separate kernel. MIXED steps (decode +
            # prefill chunks batched together) ALSO take the splitkv path for the
            # whole batch (nsys: kernel-before-FA = ReshapeAndCacheNdKernel, NOT
            # Contiguous_Slice) — only PURE-PREFILL steps (all tokens > 1, the
            # varlen TND path) emit the pre-FA Contiguous_Slice. Verified vs msprof:
            # prefill steps have 1 Contiguous_Slice/layer before FA (72 = 2×36);
            # mixed (1836 = 51×36) and pure-decode have 0.
            # Source: attention_v1.py forward_fused_infer_attention TND reshape,
            # gated on attn_state == PrefillNoCache/ChunkedPrefill (pure prefill).
            ascend_attn_fa = getattr(getattr(g, "hw", None),
                                     "aic_aiv_separated", False)
            _is_pure_prefill = bool(
                self._step_pure_prefill.get(rec.step_id, False))
            if ascend_attn_fa and _is_pure_prefill:
                _fa_sh = f'"{_kv_seq},{_kv_aiv},{_kv_hd};3;3"'
                _fa_bd = self._block_lut.get(
                    "aclnnContiguous_SliceAiCore_Slice", {}).get(_fa_sh)
                if _fa_bd is None:
                    _fa_bd = 11
                out.append(SyntheticKernelEntry(
                    kernel_name="aclnnContiguous_SliceAiCore_Slice",
                    grid=(_fa_bd, 1, 1), block=(1, 1, 1), shared_mem=0,
                    op_type="ELEMENTWISE",
                    op_attr={"seq_len": _kv_seq,
                             "num_kv_heads": _kv_nkv,
                             "head_dim": _kv_hd,
                             "dtype_bytes": self.m.dtype_bytes,
                             "reads": 1, "writes": 1}))
            g._cur_step_id = rec.step_id  # for _attn_kernels per-step prefill-FA (C500)
            out.extend(g._attn_kernels(a["batch"], a["seq_len"], is_decode_eff,
                                       kv_len=self._step_kv.get(rec.step_id),
                                       varlen_b=self._step_b.get(rec.step_id),
                                       kv_mean=self._step_kv_mean.get(rec.step_id),
                                       max_qlen=self._step_max_qlen.get(rec.step_id),
                                       eff_mblocks=self._step_eff_mblocks.get(rec.step_id)))
            # Ascend (910B): msprof shows 2× aclnnInplaceCopy_TensorMoveAiCore
            # _TensorMove (AIV, bd≈48) AFTER FusedInferAttentionScore for prefill
            # AND decode B>48, but NOT decode B<=48 (CANN FUSES the FA-output
            # reshape when the workload fits the AIV core set without a separate
            # kernel — B<=aiv_core_count). Gate = NOT (pure-decode & B<=aiv).
            # Verified: decode tt=1..48 -> 0 TensorMove; tt>=49 -> 72; prefill -> 71.
            _vb_tm = varlen_b or 1
            _aiv = getattr(g.hw, "aiv_core_count", 48)
            if ascend_attn and not (is_decode_eff and _vb_tm <= _aiv):
                from src.dse.model_sim import SyntheticKernelEntry
                _tmove_bd = _aiv
                # CANN launches only 1 TensorMove for the FIRST layer (not 2)
                # when: (a) non-pure-decode (prefill/mixed) — ALL such steps, or
                # (b) pure-decode AND the batch composition changed (new request
                # joined or old left) — 42/44 gap had batch-change, 205/213
                # nongap did NOT. Track per-step attention index for layer-0.
                # NOTE: tried dropping (b) [decode always 2 TM] -> 0.96%->1.46%
                # (over-emits for the real 71-TM decode batch-changed steps), so
                # the 17 nsys-64/regen-63 mixed steps are a separate issue.
                _attn_idx = self._attn_idx_per_step.get(rec.step_id, 0)
                self._attn_idx_per_step[rec.step_id] = _attn_idx + 1
                _is_first_layer = (_attn_idx == 0)
                _batch_changed = rec.step_id in self._step_batch_changed
                _prev_not_pd = rec.step_id in self._prev_not_pure_decode
                # msprof-grounded per-step body-TM count (build_fa_tm_910b.py):
                # the sidecar records the EXACT count of InplaceCopy_TensorMove
                # before the first sampler Fill. Drive the per-layer emit count
                # from it so tiny prefills (which CANN fuses to 0 TM — the FA
                # output fits the AIV core set when the workload is small, same
                # rule as decode B<=aiv) don't over-emit. The batch_changed/
                # prev_not_pd heuristic mis-classifies ~8% of steps and a 1-TM
                # error cascades ~1000 positional mismatches, so prefer the
                # sidecar and only fall back when it's absent.
                #   fa_tm == 0            -> 0 TM/layer (tiny prefill / fused)
                #   fa_tm == 2*nl - 1     -> first layer 1 (2nd TM suppressed),
                #                            rest 2  (large prefill, decode B>aiv)
                #   fa_tm == 2*nl         -> 2/layer, no suppress (decode B>aiv
                #                            with batch-changed first layer)
                #   anything else         -> fall back to the batch_changed gate
                _fa_tm = self._step_fa_tm.get(rec.step_id)
                _nl = getattr(self.m, "num_hidden_layers", 0)
                if _fa_tm is not None and _nl:
                    if _fa_tm == 0:
                        _n_tm = 0
                    elif _fa_tm == 2 * _nl - 1:
                        _n_tm = 1 if _is_first_layer else 2
                    elif _fa_tm == 2 * _nl:
                        _n_tm = 2
                    else:
                        _suppress_2nd = _is_first_layer and (
                            not is_decode_eff or _batch_changed or _prev_not_pd)
                        _n_tm = 1 if _suppress_2nd else 2
                else:
                    _suppress_2nd = _is_first_layer and (
                        not is_decode_eff or _batch_changed or _prev_not_pd)
                    _n_tm = 1 if _suppress_2nd else 2
                for _ in range(_n_tm):
                    out.append(SyntheticKernelEntry(
                        kernel_name="aclnnInplaceCopy_TensorMoveAiCore_TensorMove",
                        grid=(_tmove_bd, 1, 1), block=(1, 1, 1), shared_mem=0,
                        op_type="ELEMENTWISE",
                        op_attr={"element_count": varlen_b or 1, "dtype_bytes": self.m.dtype_bytes,
                                 "reads": 1, "writes": 1}))
            # KV cache block copy (framework DTOD memcpy, ~1/layer, C500 only).
            # nsys places it AFTER flash_fwd_splitkv_combine (KV paged-copy happens
            # after the FA compute reads the cache), so emit it after _attn_kernels.
            # For Llama prefill with combine, the DTOD is emitted INSIDE
            # _attn_kernels (before combine) — skip here for Llama to avoid
            # double-emit. For Llama decode steps with DTOD-before-combine
            # (dtod_pos sidecar), _attn_kernels already emitted it before combine.
            _skip_dtod = any(getattr(e, 'op_attr', {}).get('_dtod_before_combine', False) for e in out if hasattr(e, 'op_attr'))
            # C500 TP>1 (32B-TP2): nsys decode has DTOD=0 (no KV block copy in
            # the forward window). Skip the per-layer after-combine DTOD.
            _skip_dtod = _skip_dtod or (getattr(g, "_is_maca", False) and g.tp_size > 1)
            if g._emit_c500_aux and not _skip_dtod:
                m = self.m
                kv_bytes = (a["batch"] * a["seq_len"]
                            * (m.num_key_value_heads // g.tp_size)
                            * m.head_dim * 2 * m.dtype_bytes)
                out.append(SyntheticKernelEntry(
                    kernel_name="memcpy DTOD(device,device)",
                    grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                    op_type="MEMCPY_DTOD", op_attr={"bytes": kv_bytes}))
            # FA output glue (elementwise direct_copy) is emitted inside
            # _attn_kernels (after combine), gated on (is_decode & B>=2) per the
            # nsys-verified rule. The old _dense_attn_ew_glue (fixed 4 kernels)
            # is removed — it emitted kernels nsys never launches (Fill/cast at
            # wrong positions).
            return out
        if rec.op_type == "rms_norm":
            # Use the REAL captured input shape to set grid/block exactly: vLLM
            # rms_norm launches one block per normalized row, block=min(dim,1024).
            # Main norm input is 2D (rows, hidden); QK-norm is 3D (tokens, heads,
            # head_dim) -> rows = tokens*heads, dim = head_dim. This is captured,
            # not reconstructed from token counts.
            shape = a.get("input_shape")
            if shape and len(shape) >= 2:
                norm_dim = int(shape[-1])
                rows = 1
                for d in shape[:-1]:
                    rows *= int(d)
                if getattr(getattr(g, "hw", None), "aic_aiv_separated", False):
                    # vllm-ascend: aclnn RmsNorm on AIV (Block Dim 48). The QK-norm
                    # weight-multiply is fused — emit the single real kernel.
                    # msprof: the aclnnContiguous_Slice preceding each 3D q/k-norm
                    # fires for prefill AND decode B>=2, but NOT B=1 decode (CANN
                    # FUSES the reshape into RmsNorm for the small B=1 workload).
                    # Gate = NOT (pure-decode & B<2). Verified: 811 B>=2 + prefill
                    # steps have 72 Contiguous=2×36; 248 B=1 steps have 0.
                    is_dec = bool(self._step_pure_decode.get(rec.step_id, False))
                    _sb = self._step_b.get(rec.step_id, 1)
                    # Block Dim is CANN per-shape (seq=1->1, 11->11, 65->33,
                    # 721->46, 8192->48); look up from block_lut. The LUT key
                    # is the ORIGINAL (un-flattened) input shape — for 3D
                    # QK-norm (tokens, heads, head_dim) the LUT stores
                    # "{tok},{heads},{hd};{hd}", NOT the flattened
                    # "{tok*heads},{hd};{hd}" (the old flattened key never
                    # matched -> fallback 48 -> mismatched every QK-norm).
                    _shape_str = ','.join(str(int(d)) for d in shape)
                    _rms_sh = f'"{_shape_str};{norm_dim}"'
                    _rms_bd = self._block_lut.get("RmsNorm", {}).get(_rms_sh)
                    if _rms_bd is None:
                        _rms_bd = getattr(g.hw, "aiv_core_count", 48)
                    out_n = [SyntheticKernelEntry(
                        kernel_name="RmsNorm",
                        grid=(_rms_bd, 1, 1),
                        block=(1, 1, 1), shared_mem=0, op_type="NORM",
                        op_attr={"batch": rows, "hidden": norm_dim,
                                 "dtype_bytes": self.m.dtype_bytes})]
                    if len(shape) == 3 and not (is_dec and _sb < 2):
                        # .contiguous() before 3D QK-norm. The op_profile shape
                        # "{tokens},48,128;3;3" (48=aiv tiling param, NOT heads)
                        # is IDENTICAL for q-norm and k-norm, but CANN gives them
                        # DIFFERENT bd: q-norm (shape[1]=num_attention_heads, more
                        # data) gets the HIGH bd, k-norm (num_key_value_heads) the
                        # LOW bd. Pick __hi__/__lo__ by the head count in the real
                        # input shape (which the LUT key string can't encode).
                        _qk_sh = (f'"{shape[0]},'
                                  f'{getattr(g.hw, "aiv_core_count", 48)},'
                                  f'{shape[2]};3;3"')
                        _is_qnorm = int(shape[1]) >= (
                            self.m.num_attention_heads // g.tp_size)
                        _band = "__hi__" if _is_qnorm else "__lo__"
                        _qk_bd = self._block_lut.get(_band, {}).get(
                            "aclnnContiguous_SliceAiCore_Slice", {}).get(_qk_sh)
                        if _qk_bd is None:
                            _qk_bd = self._block_lut.get(
                                "aclnnContiguous_SliceAiCore_Slice", {}).get(_qk_sh)
                        if _qk_bd is None:
                            _qk_bd = 11
                        out_n.insert(0, SyntheticKernelEntry(
                            kernel_name="aclnnContiguous_SliceAiCore_Slice",
                            grid=(_qk_bd, 1, 1), block=(1, 1, 1), shared_mem=0,
                            op_type="ELEMENTWISE",
                            op_attr={"element_count": rows * norm_dim,
                                     "dtype_bytes": self.m.dtype_bytes,
                                     "reads": 1, "writes": 1}))
                    return out_n
                norm_entry = SyntheticKernelEntry(
                    kernel_name=f"rms_norm_kernel_{rec.position or 'norm'}",
                    grid=(max(1, rows), 1, 1),
                    block=(min(norm_dim, 1024), 1, 1),
                    shared_mem=168, op_type="NORM",
                    op_attr={"batch": rows, "hidden": norm_dim,
                             "dtype_bytes": self.m.dtype_bytes})
                # QK-norm (3D shape) weight-multiply: vLLM forward_native() runs
                # the elementwise weight-mul BEFORE rms_norm. FUSED (no separate
                # EW) only on decode B=1; SEPARATE (EW before NORM) on prefill and
                # decode B>1 (nsys Qwen3-8B: B=1 decode fused, B>1 decode split).
                is_decode_eff = bool(self._step_pure_decode.get(rec.step_id, False))
                samp_b = self._step_b.get(rec.step_id, 1)
                is_fused = is_decode_eff and samp_b <= 1
                if len(shape) == 3 and not is_fused:
                    from src.serve_trace.analytic_mirror import (
                        _DENSE_EW_BINARY_FLOOR_NS, _ceil_div)
                    _per = 512   # block=128 × VEC=4 = 512 elems/block (A800 EW)
                    _blk = 128
                    elements = rows * norm_dim
                    # EW grid/block: C500 nsys glue block=64 (A800 128), 4 elems/thread
                    # (matches _dense_ew_qk_norm_glue in analytic_mirror).
                    _ew_blk = 64 if g._is_maca else 128
                    _ew_per = _ew_blk * 4
                    _blk = _ew_blk
                    ew_entry = SyntheticKernelEntry(
                        kernel_name="void at::native::elementwise_kernel_qk_norm_mul",
                        grid=(_ceil_div(max(1, elements), _ew_per), 1, 1),
                        block=(_blk, 1, 1), shared_mem=0, op_type="ELEMENTWISE",
                        op_attr={"element_count": elements,
                                 "dtype_bytes": self.m.dtype_bytes,
                                 "reads": 2, "writes": 1,
                                 "floor_ns": _DENSE_EW_BINARY_FLOOR_NS})
                    out = [ew_entry, norm_entry]
                else:
                    out = [norm_entry]
                return out
            return [sim._make_norm(self.m, a["batch"], a["seq_len"],
                                   g.tp_size, rec.position or "norm")]
        if rec.op_type == "fused_add_rms_norm":
            shape = a.get("input_shape")
            if shape and len(shape) >= 2:
                norm_dim = int(shape[-1])
                rows = 1
                for d in shape[:-1]:
                    rows *= int(d)
                if getattr(getattr(g, "hw", None), "aic_aiv_separated", False):
                    # vllm-ascend: aclnn AddRmsNormBias on AIV. Block Dim is
                    # CANN per-shape (seq=1->1, 11->11, 65->33, 721->46, 8192->48);
                    # look up from block_lut, fall back to aiv_core_count. Key is
                    # the original (un-flattened) shape — see the RmsNorm branch.
                    _shape_str = ','.join(str(int(d)) for d in shape)
                    _add_sh = f'"{_shape_str};{_shape_str};{norm_dim};"'
                    _add_bd = self._block_lut.get("AddRmsNormBias", {}).get(_add_sh)
                    if _add_bd is None:
                        _add_bd = getattr(g.hw, "aiv_core_count", 48)
                    return [SyntheticKernelEntry(
                        kernel_name="AddRmsNormBias",
                        grid=(_add_bd, 1, 1),
                        block=(1, 1, 1), shared_mem=0, op_type="NORM",
                        op_attr={"batch": rows, "hidden": norm_dim,
                                 "dtype_bytes": self.m.dtype_bytes})]
                max_block = 1024 if rows < 256 else 256
                return [SyntheticKernelEntry(
                    kernel_name=f"fused_add_rms_norm_kernel_{rec.position or 'norm'}",
                    grid=(max(1, rows), 1, 1),
                    block=(min(norm_dim, max_block), 1, 1),
                    shared_mem=168, op_type="NORM",
                    op_attr={"batch": rows, "hidden": norm_dim,
                             "dtype_bytes": self.m.dtype_bytes})]
            return [sim._make_norm(self.m, a["batch"], a["seq_len"],
                                   g.tp_size, rec.position or "norm")]
        if rec.op_type == "rotary":
            if getattr(getattr(g, "hw", None), "aic_aiv_separated", False):
                # Rotary kernel choice depends on model architecture:
                # - Qwen3: vllm-ascend uses _triton_rope (custom triton kernel)
                # - Llama: vllm-ascend uses aclnnMul (Slice + Mul for cos/sin)
                # The distinction is by architecture name (from model config).
                _arch = getattr(self.m, "architecture", "")
                if "Llama" in _arch:
                    # Llama rotary: each layer has Q rotary + K rotary, each =
                    # (Slice Slice Mul)×2  Sub  (Slice Slice Mul)×2  Add  Cat  Cat_TM
                    # Cat_TM: only when the rotary's token count > aiv_core_count
                    # (same CANN fusion threshold as the FA-output TensorMove).
                    # B<=48 → CANN fuses Cat_TM too. Q's Cat_TM: when seq>48.
                    # K's Cat_TM: prefill/mixed only (decode K is fused).
                    # NOTE: use the rotary op's OWN seq_len (a["seq_len"] = the
                    # token count this rotary processes), NOT _step_b (request
                    # count from the sidecar). For chunked-prefill/mixed steps
                    # _step_b (requests) ≠ seq_len (prefill chunk tokens) → the
                    # Cat_TM gate would miss. Verified: seq_len=201/355/766 steps
                    # have Cat_TM in nsys; seq_len=1/11/48 don't.
                    _is_dec = bool(self._step_pure_decode.get(rec.step_id, False))
                    _rb = a.get("seq_len", a.get("batch", 1))
                    _has_cat_tm = _rb > 48
                    _nq = self.m.num_attention_heads // g.tp_size
                    _nkv = max(1, self.m.num_key_value_heads // g.tp_size)
                    _hd = self.m.head_dim
                    _hd2 = _hd // 2  # rotate_half: rotary applies to head_dim/2
                    _bl = self._block_lut
                    # Look up the real CANN Block Dim per (kernel, shape) from the
                    # case's op_profile (block_lut.json). Shapes constructed from
                    # the rotary's seq_len + nh (Q=num_attention_heads, K=num_kv
                    # heads) + head_dim/2. Fallback to the old hardcoded bd if the
                    # shape isn't in the LUT (unseen shape — best-effort).
                    def _bd(nm, shape, default):
                        v = _bl.get(nm, {}).get(shape)
                        return v if v is not None else default
                    _aiv = getattr(g.hw, "aiv_core_count", 48)
                    # Mul_Slice: for seq>1 the 2D shape "{seq,hd;2;2}" (covers
                    # seq 2-33), 3D fallback "{seq,aiv,hd;3;3}" (seq 2-8192). For
                    # seq=1 (B=1) the slice axis is heads, so the shape first dim
                    # is num_heads not seq: Q="{nh_q,hd;2;2}"->11, K="{nh_kv,hd;2;2}"->8.
                    def _sl_bd(nh):
                        if _rb <= 1:
                            return _bd("aclnnMul_SliceAiCore_Slice",
                                       f'"{nh},{_hd};2;2"', 11)
                        v = _bd("aclnnMul_SliceAiCore_Slice",
                                f'"{_rb},{_hd};2;2"', None)
                        if v is None:
                            v = _bd("aclnnMul_SliceAiCore_Slice",
                                    f'"{_rb},{_aiv},{_hd};3;3"', 11)
                        return v
                    # Prefill rotary unit = [Slice_3D, Slice_2D, Mul]. The two
                    # Slices have DIFFERENT shapes/bd (verified from real msprof
                    # inputShapes): Slice_3D "{seq},48,128;3;3" is q/k-head
                    # dependent (Q=num_attention_heads → HIGH bd, K=num_kv_heads
                    # → LOW bd; same "{seq},48,128;3;3" collision as the QK-norm
                    # Contiguous); Slice_2D "{seq},128;2;2" is always the low bd
                    # regardless of head. The old _sl_bd emitted BOTH as the 2D
                    # lookup → Slice_3D wrong for Q.
                    def _sl_bd_3d(nh):
                        _sh = f'"{_rb},{_aiv},{_hd};3;3"'
                        _band = "__hi__" if nh >= _nq else "__lo__"
                        v = _bl.get(_band, {}).get(
                            "aclnnMul_SliceAiCore_Slice", {}).get(_sh)
                        if v is None:
                            v = _bd("aclnnMul_SliceAiCore_Slice", _sh, 11)
                        return v
                    def _sl_bd_2d(nh):
                        # 2D slice "{seq},128;2;2" in the PREFILL rotary is the
                        # HIGH bd for the shape regardless of head (verified: q
                        # and k both = 12 for seq=48/721; the lower 8/11 values in
                        # op_profile for the same shape string are decode-batch
                        # contexts, not this prefill slice). Use __hi__.
                        _sh = f'"{_rb},{_hd};2;2"'
                        v = _bl.get("__hi__", {}).get(
                            "aclnnMul_SliceAiCore_Slice", {}).get(_sh)
                        if v is None:
                            v = _bd("aclnnMul_SliceAiCore_Slice", _sh, 12)
                        return v
                    def _mul_bd(nh):
                        return _bd("aclnnMul_MulAiCore_Mul",
                                   f'"{_rb},{nh},{_hd2};{_rb},1,{_hd2}"', 11)
                    def _sub_add_bd(nh, nm):
                        return _bd(nm, f'"{_rb},{nh},{_hd2};{_rb},{nh},{_hd2}"', 11)
                    def _cat_bd(nh):
                        return _bd("aclnnCat_ConcatD_ConcatD",
                                   f'"{_rb},{nh},{_hd2};{_rb},{nh},{_hd2}"', 48)
                    def _cat_tm_bd(nh):
                        return _bd("aclnnCat_TensorMoveAiCore_TensorMove",
                                   f'"{_rb},{nh},{_hd}"', 48)
                    # B=1 decode: CANN FUSES the 2nd Slice into Mul — the rotary
                    # unit is (Slice, Mul), not (Slice, Slice, Mul). Verified vs
                    # msprof: seq_len=1 half = [Slice,Mul]×2 Sub [Slice,Mul]×2 Add
                    # Cat (11 kern); seq_len>=2 keeps the full (Slice,Slice,Mul)
                    # unit (15).
                    def _unit(nh):
                        if _is_dec and _rb <= 1:
                            return [("aclnnMul_SliceAiCore_Slice", _sl_bd(nh)),
                                    ("aclnnMul_MulAiCore_Mul", _mul_bd(nh))]
                        # Prefill: Slice_3D (q/k-head dependent), Slice_2D
                        # (head-independent), Mul.
                        return [("aclnnMul_SliceAiCore_Slice", _sl_bd_3d(nh)),
                                ("aclnnMul_SliceAiCore_Slice", _sl_bd_2d(nh)),
                                ("aclnnMul_MulAiCore_Mul", _mul_bd(nh))]
                    def _half(nh, extra_cat_tm):
                        return (_unit(nh) * 2 +                 # (SSM)×2
                                [("aclnnSub_SubAiCore_Sub", _sub_add_bd(nh, "aclnnSub_SubAiCore_Sub"))] +
                                _unit(nh) * 2 +                 # (SSM)×2
                                [("aclnnAdd_AddAiCore_Add", _sub_add_bd(nh, "aclnnAdd_AddAiCore_Add")),
                                 ("aclnnCat_ConcatD_ConcatD", _cat_bd(nh))] +
                                ([("aclnnCat_TensorMoveAiCore_TensorMove", _cat_tm_bd(nh))]
                                 if (extra_cat_tm and _has_cat_tm) else []))
                    # Q gets Cat_TM when seq>aiv_core_count (48). K gets Cat_TM
                    # when seq exceeds the SAME byte threshold: K's per-token
                    # size (num_kv_heads*head_dim) is 4× smaller than Q's
                    # (num_q_heads*head_dim), so K's seq threshold is 4× higher.
                    # K_threshold = aiv * num_q_heads / num_kv_heads = 48*32/8 = 192.
                    # Verified vs msprof: seq<=192 -> Q only (32); seq>=201 -> Q+K
                    # (64). Pure-decode never reaches seq>192 (B<=128<n_req),
                    # so the seq gate alone subsumes the `not pure_decode` rule
                    # AND correctly fires for large-seq mixed-prefill steps
                    # (which `not _is_dec` over-fired for small-seq mixed, and
                    # `is_prefill` under-fired for mixed-prefill).
                    _k_thr = getattr(g.hw, "aiv_core_count", 48) * _nq // _nkv
                    _llama_rope = _half(_nq, True) + _half(_nkv, _rb > _k_thr)
                    return [
                        SyntheticKernelEntry(
                            kernel_name=nm, grid=(bd, 1, 1), block=(1, 1, 1),
                            shared_mem=0, op_type="ELEMENTWISE",
                            op_attr={"seq_len": a.get("seq_len", a.get("batch", 1)),
                                     "num_heads": self.m.num_attention_heads // g.tp_size,
                                     "head_dim": self.m.head_dim,
                                     "dtype_bytes": self.m.dtype_bytes,
                                     "reads": 1, "writes": 1})
                        for nm, bd in _llama_rope
                    ]
                # Qwen3: _triton_rope on AIV. CANN launches Block Dim =
                # min(seq_len, aiv_core_count): one AIV core per token up to the
                # core count, then capped. Verified vs msprof: seq=1..48 -> bd=seq
                # (exact), seq>48 -> bd=48. The neox FillFunctor workspace-init
                # below is A800-only (vllm-ascend's triton rope has no separate
                # zero-init kernel). Restored from 096a431 (regressed away).
                _rope_seq = a.get("seq_len", a.get("batch", 1))
                _rope_bd = min(_rope_seq, getattr(g.hw, "aiv_core_count", 48))
                return [SyntheticKernelEntry(
                    kernel_name="_triton_rope",
                    grid=(_rope_bd, 1, 1),
                    block=(1, 1, 1), shared_mem=0, op_type="ROTARY",
                    op_attr={"seq_len": _rope_seq,
                             "num_heads": self.m.num_attention_heads // g.tp_size,
                             "head_dim": self.m.head_dim,
                             "dtype_bytes": self.m.dtype_bytes})]
            out = [sim._make_rotary(self.m, a["batch"], a["seq_len"], g.tp_size)]
            # nsys: every rotary is followed by a FillFunctor<c10::Half |
            # BFloat16> that zero-inits the neox-rotary half-head workspace.
            # SOURCE (vLLM csrc/pos_encoding_kernels.cu::rotary_embedding):
            #   dim3 grid(num_tokens);
            #   dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
            # where rot_dim = cos_sin_cache.size(1) = head_size. The /2 is the
            # neox split (rotary rotates head_dim/2 per head; the other half is
            # the conjugate pair). The FillFunctor zero-inits the matching
            # buffer of num_tokens × (num_attention_heads/TP) × (head_dim/2)
            # elements → grid = ceil(n_tok × q_heads × head_dim / 2 / 512)
            # (block=128 × VEC=4 = 512 elems/block; nsys block=(128,1,1)).
            # n_tok = num_actual_tokens = the rotary op record's batch×seq_len
            # (the capture sets batch=1, seq_len=total varlen tokens) — this is
            # the rotary grid (== nsys rotary grid, type-aligned), so it is the
            # TRUE per-step token count for BOTH decode (=B, NOT 1) and prefill.
            # Verified: nsys fill grid == 4 × nsys rotary grid exactly (Llama:
            # rotary{1,2,3..} → fill{4,8,12..}, identical counts).
            from src.dse.model_sim import SyntheticKernelEntry
            from src.serve_trace.analytic_mirror import _ceil_div, _DENSE_EW_FILL_FLOOR_NS
            m = self.m
            tp = g.tp_size
            q_heads = m.num_attention_heads // tp
            n_tok = a["batch"] * a["seq_len"]
            half_typ = ("c10::BFloat16" if (eff_dt or self._dtype) == "bf16" else "c10::Half")

            # C500 (maca) + Llama: the fused vllm::rotary_embedding_kernel is NOT
            # used. Instead, C500 vLLM falls back to PyTorch's native elementwise
            # RoPE path: indexSelect (split Q/K for rotate_half) →
            # elementwise×2+VEW_add (cos/sin apply per half) → MACA_Cat (concat) →
            # DTOD → repeat for K → VEW_fill (KV page zero-init).
            # Total: 18 kernels per layer (vs 2 for Qwen3: rotary+fill).
            _is_llama = "Llama" in (m.architecture or "")
            if getattr(g, "_is_maca", False) and _is_llama:
                _ew3 = "void at::native::elementwise_kernel_3_2_dim0_contiguous"
                _vew_add = "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add"
                _cat = "void at::native::(anonymous namespace)::MACA_CatArrayBatchedCopyNoPartialWrite"
                _idx = "void at::native::indexSelectLargeIndexWithoutAssert"
                _ew_grid = max(1, _ceil_div(n_tok, 4))
                out = []
                # Q rotate_half: indexSelect + 2×(elementwise×2 + VEW_add) + MACA_Cat + DTOD
                out.append(SyntheticKernelEntry(kernel_name=_idx, grid=(1,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                for _ in range(2):  # 2 halves
                    out.append(SyntheticKernelEntry(kernel_name=_ew3, grid=(_ew_grid,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                    out.append(SyntheticKernelEntry(kernel_name=_ew3, grid=(_ew_grid,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                    out.append(SyntheticKernelEntry(kernel_name=_vew_add, grid=(_ew_grid,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                out.append(SyntheticKernelEntry(kernel_name=_cat, grid=(1,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={}))
                out.append(SyntheticKernelEntry(kernel_name="memcpy DTOD(device,device)", grid=(1,1,1), block=(1,1,1), shared_mem=0, op_type="MEMCPY_DTOD", op_attr={"bytes": m.hidden_size * m.dtype_bytes}))
                # K rotate_half: same pattern
                for _ in range(2):
                    out.append(SyntheticKernelEntry(kernel_name=_ew3, grid=(_ew_grid,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                    out.append(SyntheticKernelEntry(kernel_name=_ew3, grid=(_ew_grid,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                    out.append(SyntheticKernelEntry(kernel_name=_vew_add, grid=(_ew_grid,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={"element_count": n_tok, "dtype_bytes": m.dtype_bytes}))
                out.append(SyntheticKernelEntry(kernel_name=_cat, grid=(1,1,1), block=(128,1,1), shared_mem=0, op_type="ELEMENTWISE", op_attr={}))
                out.append(SyntheticKernelEntry(kernel_name="memcpy DTOD(device,device)", grid=(1,1,1), block=(1,1,1), shared_mem=0, op_type="MEMCPY_DTOD", op_attr={"bytes": m.hidden_size * m.dtype_bytes}))
                # FillFunctor (KV page zero-init) — same as Qwen3
                fill_elems = n_tok * q_heads * m.head_dim // 2
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at::native::vectorized_elementwise_kernel<(int)4, "
                                 f"at::native::FillFunctor<{half_typ}>, std::array<char *, "
                                 "(unsigned long)1>>(int, T2, T3)"),
                    grid=(_ceil_div(fill_elems, 512), 1, 1), block=(128, 1, 1),
                    shared_mem=0, op_type="ELEMENTWISE",
                    op_attr={"element_count": fill_elems, "dtype_bytes": m.dtype_bytes,
                             "reads": 0, "writes": 1, "floor_ns": _DENSE_EW_FILL_FLOOR_NS}))
                return out

            # Qwen3 path (default): fused rotary_embedding_kernel + FillFunctor
            out = [sim._make_rotary(self.m, a["batch"], a["seq_len"], g.tp_size)]
            fill_elems = n_tok * q_heads * m.head_dim // 2
            out.append(SyntheticKernelEntry(
                kernel_name=("void at::native::vectorized_elementwise_kernel<(int)4, "
                             f"at::native::FillFunctor<{half_typ}>, std::array<char *, "
                             "(unsigned long)1>>(int, T2, T3)"),
                grid=(_ceil_div(fill_elems, 512), 1, 1), block=(128, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": fill_elems, "dtype_bytes": m.dtype_bytes,
                         "reads": 0, "writes": 1, "floor_ns": _DENSE_EW_FILL_FLOOR_NS}))
            return out
        if rec.op_type == "kv_cache":
            if getattr(getattr(g, "hw", None), "aic_aiv_separated", False):
                # vllm-ascend: ReshapeAndCacheNd on AIV. CANN Block Dim =
                # min(seq_len, aiv_core_count), same per-token-core launch as
                # _triton_rope. Verified vs msprof: seq=1..48 -> bd=seq (exact),
                # seq>48 -> bd=48.
                _kv_seq = a.get("seq_len", a.get("batch", 1))
                _kv_bd = min(_kv_seq, getattr(g.hw, "aiv_core_count", 48))
                return [SyntheticKernelEntry(
                    kernel_name="ReshapeAndCacheNdKernel",
                    grid=(_kv_bd, 1, 1),
                    block=(1, 1, 1), shared_mem=0, op_type="KV_CACHE",
                    op_attr={"seq_len": _kv_seq,
                             "num_kv_heads": self.m.num_key_value_heads // g.tp_size,
                             "head_dim": self.m.head_dim,
                             "dtype_bytes": self.m.dtype_bytes})]
            return [sim._make_kv_cache(self.m, a["batch"], a["seq_len"], g.tp_size)]
        if rec.op_type == "activation":
            if getattr(getattr(g, "hw", None), "aic_aiv_separated", False):
                # vllm-ascend: SwiGlu (fused SiLU*gate) on AIV. Block Dim is CANN
                # per-shape (seq=1->2, 11->33, 65->48, 721->48); shape
                # "{seq,intermediate*2}". Look up from block_lut.
                # Same aclnn as the MoE w1/w2 activation — dense FFN is SwiGLU too.
                _sg_seq = a.get("seq_len", a.get("batch", 1))
                _sg_sh = f'"{_sg_seq},{self.m.intermediate_size * 2}"'
                _sg_bd = self._block_lut.get("SwiGlu", {}).get(_sg_sh)
                if _sg_bd is None:
                    _sg_bd = getattr(g.hw, "aiv_core_count", 48)
                return [SyntheticKernelEntry(
                    kernel_name="SwiGlu",
                    grid=(_sg_bd, 1, 1),
                    block=(1, 1, 1), shared_mem=0, op_type="ACTIVATION",
                    op_attr={"batch": _sg_seq,
                             "intermediate": self.m.intermediate_size,
                             "dtype_bytes": self.m.dtype_bytes})]
            return [sim._make_activation(self.m, a["batch"], a["seq_len"], g.tp_size)]
        if rec.op_type == "moe":
            # EP all-to-all dispatch/combine are NOT fabricated here. vLLM only
            # issues them (GroupCoordinator.dispatch/combine) when it truly routes
            # tokens across EP ranks (do_naive_dispatch_combine / pplx / deepep);
            # in the common single-DP TP/EP case it uses expert_map + a final TP
            # all-reduce instead, and nsys shows zero all-to-all kernels. When the
            # call really happens it is CAPTURED as a separate `all_to_all` record
            # by _operator_patch and handled by the all_to_all branch below.
            # Fabricating dispatch/combine around every MoE op (the old behavior)
            # invented kernels absent from the trace, so it was removed.
            return list(g._moe_ffn_kernels(a["batch"], a["seq_len"], a["is_decode"]))
        if rec.op_type == "all_reduce":
            return g._tp_allreduce(a["batch"], a["seq_len"], rec.position or "tp",
                                   msg_bytes=a.get("msg_bytes"),
                                   nranks=max(g.tp_size, g.ep_size))
        if rec.op_type == "all_to_all":
            return [g._ep_all2all(a["batch"], a["seq_len"], rec.position or "ep",
                                  msg_bytes=a.get("msg_bytes"))]
        if rec.op_type == "kv_transfer":
            # PD-disaggregation KV cache send/recv captured from the real
            # P2pNcclEngine transfer; msg_bytes is the real moved-tensor volume.
            return [g._kv_transfer(a["msg_bytes"], rec.position or "send")]
        raise ValueError(f"no last-hop mapping for op_type {rec.op_type!r}")

    # ----------------------------------------------------------------------
    # Phase 2 (T3) — H-layer DIRECT pricing.  See docs/serving_perf_modeling_
    # phase2_goal.md §T3 / §6.4 and redesign.md §2 + appendix A.
    # ----------------------------------------------------------------------

    # Compute-dominant SyntheticKernelEntry op_types: the kernels that carry
    # real GPU time.  Everything the synthesis path fabricates purely for
    # byte-exact nsys/msprof trace ALIGNMENT — sampler pipeline (argmax /
    # multinomial / scatter), framework memcpy (HTOD/DTOH/DTOD), KV-page Fill,
    # input cast / embedding gather / masked_fill, CUB DeviceScan — prices to a
    # ~launch-overhead floor and does NOT move G1.  The direct-pricing path
    # keeps only these categories; dropping the rest is the whole point of the
    # redesign (delete the ~317KB alignment middle layer).  REDUCE/GATHER are
    # dropped for dense; MoE routing REDUCE/GATHER are re-added (promoted to
    # FUSEDMOE) when the MoE direct path lands (T3 follow-up).
    _DIRECT_COMPUTE_OPTYPES = frozenset({
        "GEMM", "SPLITK", "GEMV", "FLASHATTN", "FUSEDMOE",
        "NORM", "ROTARY", "ACTIVATION", "KV_CACHE",
        "COLLECTIVE", "CUSTOM_COMM",
    })

    def records_to_trace_direct(self, records: List[OperatorRecord]):
        """H-layer DIRECT-pricing trace (the ONLY trace path since Phase 2
        T4-4b): one compute entry (or a few) per logical ``OperatorRecord``.

        Prices each record via its own recipe leaf (``_op_to_kernels`` →
        ``_attn_kernels`` / ``_make_*``) so FA (kv-len/num_splits/eff_mblocks),
        GEMM (real M/N/K), NORM (real input_shape) etc. carry their validated
        accuracy. The deleted legacy synthesis wrapper (``records_to_trace``)
        additionally layered launch-bound STEP-level alignment glue (sampler
        remap / htod-dtoh / fill_grid / front_ew / prefill prefix) on top; that
        glue was verified non-load-bearing for G1/G3 and is gone. G1 is carried
        by the compute categories, so this reproduces the per-iteration GPU time.

        Downstream is UNCHANGED: ``annotate_trace`` → ``predict`` → assembly
        consume these entries and ``sum(predicted_ns)`` is insensitive to kernel
        count. This is the single regen seam in build_eval_dataset
        (``_regenerate_trace_with_step_id``).
        """
        ordered = sorted(records, key=lambda r: (r.step_id, r.op_idx))
        has_real_ts = any(getattr(r, "ts_us", 0) > 0 for r in ordered)
        # MIDLAYER_DIRECT_FILTER selects which per-op kernels to keep.
        #   "none" (default) → keep ALL _op_to_kernels output, including each
        #                      op's directly-implied epilogue (e.g. the FA-output
        #                      direct_copy the attention op emits — a real 1:1
        #                      consequence of the op, ~5.6% of GPU, NOT alignment
        #                      glue). The STEP-level synthesis glue (sampler
        #                      remap / htod-dtoh / fill_grid / front_ew / prefill
        #                      prefix) that the deleted records_to_trace layered
        #                      on is gone entirely — verified non-load-bearing
        #                      for G1/G3 (Qwen3-8B-TP1-qps4 = 12.85%/3.76%).
        #   "compute"        → additionally drop per-op ELEMENTWISE/REDUCE/GATHER/
        #                      SAMPLER (the _DIRECT_COMPUTE_OPTYPES filter).
        #                      DIAGNOSTIC ONLY: quantifies the per-op framework-EW
        #                      dependence (+5pp G1 on the gate case), confirming
        #                      the FA-output epilogue must stay (it is kept here).
        import os as _os
        _filt = (_os.environ.get("MIDLAYER_DIRECT_FILTER", "none").strip()
                 or "none")
        # Seed with the first real ts so ts_us==0 records inherit the last valid
        # timestamp — keeps step ordering stable.
        last_valid_ts = next((r.ts_us for r in ordered
                              if getattr(r, "ts_us", 0) > 0), 0.0)
        entries = []
        ts = 0.0
        for rec in ordered:
            if has_real_ts and getattr(rec, "ts_us", 0) > 0:
                last_valid_ts = rec.ts_us
            if has_real_ts:
                effective_ts = rec.ts_us if rec.ts_us > 0 else last_valid_ts
            else:
                effective_ts = round(ts, 3)
            ts += 1.0
            for k in self._op_to_kernels(rec):
                if _filt != "none" and k.op_type not in self._DIRECT_COMPUTE_OPTYPES:
                    continue
                entry = {
                    "timestamp_us": effective_ts,
                    "kernel_name": k.kernel_name,
                    "grid": list(k.grid),
                    "block": list(k.block),
                    "shared_mem": k.shared_mem,
                    "stream": 0,
                    "step_id": rec.step_id,
                    "_sampler_pipeline": getattr(k, "sampler_pipeline", False),
                }
                # Carry real MoE routing (captured topk_ids histogram) onto the
                # fused-MoE entries — the MoE predictor REQUIRES the true
                # per-expert token counts (uniform fallback is forbidden).
                if rec.op_type == "moe":
                    tpe = rec.attrs.get("tok_per_expert")
                    if tpe is not None:
                        entry["tok_per_expert"] = tpe
                if getattr(k, "op_attr", None):
                    entry["op_attr"] = k.op_attr
                entries.append(entry)
        return entries
