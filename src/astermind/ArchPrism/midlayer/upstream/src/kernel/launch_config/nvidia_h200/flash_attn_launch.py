"""NVIDIA H200 FlashAttention-3 MLA launch selection.

vLLM 0.24.0 on Hopper runs MLA attention through ``vllm.vllm_flash_attn``
FlashAttention-3, built from vllm-project/flash-attention at the revision
``cmake/external_projects/vllm_flash_attn.cmake`` pins
(``803020a8fa15407871341d41eba4919ade2ee1ee``).  Every discrete launch fact
below is a static fact of that source (``hopper/flash_api.cpp``,
``tile_size.h``, ``heuristics.h``, ``flash_fwd_launch_template.h``,
``flash_fwd_combine_launch_template.h``, ``flash_prepare_scheduler.cu``) and
of the vLLM 0.24.0 callers (``v1/attention/backends/mla/flashattn_mla.py``
for decode, the FLASH_ATTN MLA prefill backend for prefill):

* prefill: ``flash_attn_varlen_func(q[., H, 192], k[., H, 192],
  v[., H, 128], causal=True)`` with no ``num_splits`` -> the
  ``get_num_splits`` heuristic; ``tile_size_fwd_sm90(192, 128, causal)`` =
  (kBlockM 128, kBlockN 128, MmaPV_is_RS, IntraWGOverlap); two MMA
  warpgroups + one producer warpgroup (384 threads); varlen dynamic
  persistent scheduler (grid = SM count) with LPT order for causal.
* decode: absorbed MLA, ``q = q_pe`` (64), ``qv = q_nope`` (512),
  ``k = k_pe`` (64), ``v = kv_c`` (512), one KV head, paged cache,
  ``num_splits = flash_attn_max_num_splits_for_cuda_graph`` (32) when the
  deployment captures full CUDA graphs and the decode fits the capture
  size (``full_cuda_graphs`` / ``max_cudagraph_capture_size`` attributes),
  else 0 (heuristic); ``tile_size_fwd_sm90(64, 512)`` = (64, 64, SS,
  no intra-WG overlap); PackGQA is forced by ``num_splits > 1``; the paged
  cache uses cp.async (PagedKVNonTMA) because
  ``seqlen_q * H <= kBlockM``; 2 MMA warpgroups (LargeHeadDimV) + a
  128-thread producer warpgroup; ``seqlen_q == 1`` executes non-causal.
* combine (``num_splits > 1``): ``FlashAttnFwdCombine<(8, 128), log2
  max splits, 256 threads>``, varlen tile scheduler (STANDARD or
  LINEARIZE_M_AND_BATCH by tile density).
* prepare (``prepare_varlen_num_blocks_kernel<NumWarps>``): one CTA of
  ``32 * NumWarps`` threads, launched by every varlen call that does not
  carry precomputed scheduler metadata (prefill), and once per step by the
  decode metadata builder.

Owned domains (``owns_*`` predicates, decided before selection): fresh
causal MLA prefill with qk 192 / v 128 that the heuristic runs unsplit; the
absorbed MLA decode (latent 512 + rope 64, one query token per request, at
most kBlockM query heads) and its split combine.  Everything else stays with
the projected package.

The selector records the source template in ``LaunchConfig.algorithm``
(``vllm_fa3_sm90_mla_prefill`` / ``vllm_fa3_sm90_mla_decode`` /
``vllm_fa3_sm90_combine``); the predictor re-derives the device-side tile
schedule from the same source semantics.
"""

from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.selection_primitives import (
    ceil_div, require_dtype, require_positive_integer, require_shape,
    require_variant,
)


FA3_SOURCE_REVISION = "803020a8fa15407871341d41eba4919ade2ee1ee"
FA3_PREFILL_ALGORITHM = "vllm_fa3_sm90_mla_prefill"
FA3_DECODE_ALGORITHM = "vllm_fa3_sm90_mla_decode"
FA3_COMBINE_ALGORITHM = "vllm_fa3_sm90_combine"
# SGLang's absorbed-MLA extend (flashinfer BatchMLAPagedAttentionWrapper,
# backend "auto" -> "fa3" on sm90): flashinfer 0.6.11
# include/flashinfer/attention/mla_hopper.cuh BatchMLAPageAttentionHopper.
FLASHINFER_MLA_PREFILL_ALGORITHM = "flashinfer_mla_hopper_sm90_paged"
_FLASHINFER_MLA_TILE = 64
_FLASHINFER_MLA_THREADS = 256
_FLASHINFER_MLA_STAGES = 2

# vLLM 0.24.0 vllm/config/attention.py
# (AttentionConfig.flash_attn_max_num_splits_for_cuda_graph).
_VLLM_CUDA_GRAPH_MAX_NUM_SPLITS = 32
# hopper/flash_api.cpp: splits above 1 require num_splits <= 256 and the
# heuristic's max_splits argument is 128.
_HEURISTIC_MAX_SPLITS = 128
# hopper/heuristics.h num_splits_heuristic: 50 MiB L2 assumption.
_HEURISTIC_L2_BYTES = 50 * 1024 * 1024
# hopper/flash_api.cpp PREPARE_VARLEN_MAX_BATCHES_1CTA.
_PREPARE_VARLEN_MAX_BATCHES_1CTA = 992
_ELEMENT_BYTES = 2
_MMA_WARPGROUPS = 2
_THREADS_PER_WARPGROUP = 128


def _source_parameters(splits: int, paged: bool) -> Mapping[str, float]:
    return {
        "unpadded_flops": 0.0,
        "mma_smem_bytes_per_flop": 0.0,
        "split_scratch_dram_bytes_per_element": 4.0 if splits > 1 else 0.0,
        "page_table_bytes_per_kv_panel": (
            4.0 if paged and splits > 1 else 0.0),
    }


def _request_lengths(shape: Mapping[str, int],
                     attributes: Mapping[str, object]) -> tuple:
    batch, q_len, kv_len = require_shape(shape, ("batch", "q_len", "kv_len"))
    raw_q = attributes["q_lens"]
    raw_kv = attributes["kv_lens"]
    if (not isinstance(raw_q, (list, tuple))
            or not isinstance(raw_kv, (list, tuple))
            or any(isinstance(value, bool) or not isinstance(value, int)
                   for value in tuple(raw_q) + tuple(raw_kv))):
        raise TypeError("H200 FA3 MLA request lengths must be integers")
    q_lens, kv_lens = tuple(raw_q), tuple(raw_kv)
    if any(value <= 0 for value in q_lens + kv_lens):
        raise ValueError("H200 FA3 MLA request lengths must be positive")
    if (len(q_lens) != batch or len(kv_lens) != batch
            or max(q_lens) != q_len or max(kv_lens) != kv_len):
        raise ValueError("H200 FA3 MLA request lengths disagree with shape")
    if any(query > key_value for query, key_value in zip(q_lens, kv_lens)):
        raise ValueError("H200 FA3 MLA requires q_len <= kv_len per request")
    return q_lens, kv_lens


def fa3_num_splits_heuristic(total_mblocks: int, num_sms: int,
                             num_n_blocks: int, num_m_blocks: int,
                             size_one_kv_head: int, is_causal_or_local: bool,
                             max_splits: int) -> int:
    """hopper/heuristics.h ``num_splits_heuristic`` verbatim."""

    if total_mblocks >= 0.8 * num_sms:
        if (size_one_kv_head > _HEURISTIC_L2_BYTES
                and num_m_blocks >= num_sms * 2
                and not is_causal_or_local):
            return min(ceil_div(size_one_kv_head, _HEURISTIC_L2_BYTES),
                       max_splits)
        return 1
    if num_n_blocks <= 4:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    efficiency = []
    for splits in range(1, max_splits + 1):
        waves = total_mblocks * splits / num_sms
        efficiency.append(waves / ceil_div(total_mblocks * splits, num_sms))
    best = max(efficiency)
    for splits in range(1, max_splits + 1):
        if efficiency[splits - 1] >= 0.85 * best:
            return splits
    return 1


def _varlen_num_splits(*, batch: int, max_seqlen_q: int, max_seqlen_k: int,
                       num_heads: int, num_heads_k: int, head_dim: int,
                       head_dim_v: int, block_m: int, block_n: int,
                       causal: bool, num_sms: int) -> int:
    """``get_num_splits`` for a varlen call (dynamic-split upper bound)."""

    qhead_per_khead = num_heads // num_heads_k
    num_n_blocks = ceil_div(max_seqlen_k, block_n)
    num_m_blocks = ceil_div(max_seqlen_q * qhead_per_khead, block_m)
    size_one_kv_head = max_seqlen_k * (head_dim + head_dim_v) * _ELEMENT_BYTES
    # Varlen sets num_splits_dynamic_ptr before the heuristic, which then
    # pretends batch = 1 (flash_api.cpp get_num_splits).
    total_mblocks = num_heads_k * num_m_blocks
    splits = fa3_num_splits_heuristic(
        total_mblocks, num_sms, num_n_blocks, num_m_blocks,
        size_one_kv_head, causal, _HEURISTIC_MAX_SPLITS)
    if batch > _PREPARE_VARLEN_MAX_BATCHES_1CTA:
        return 1
    return splits


def _decode_uses_cuda_graph_split_bound(attributes: Mapping[str, object],
                                        decode_tokens: int) -> bool:
    """vLLM FlashAttnMLAMetadataBuilder._build_decode split bound rule.

    ``max_num_splits`` is the CUDA-graph bound when the deployment enables
    full CUDA graphs (``cudagraph_mode.has_full_cudagraphs()``) and the
    decode token count fits ``max_cudagraph_capture_size``; otherwise 0
    (the FA3 heuristic).  Both are deployment facts the workload attaches
    to every MLA node; they are required.
    """

    full_graphs = attributes["full_cuda_graphs"]
    capture = attributes["max_cudagraph_capture_size"]
    if not isinstance(full_graphs, bool):
        raise TypeError("full_cuda_graphs must be boolean")
    if isinstance(capture, bool) or not isinstance(capture, int) or capture < 0:
        raise ValueError("max_cudagraph_capture_size must be a non-negative int")
    return full_graphs and decode_tokens <= capture


def _normalized_variant(variant) -> Optional[str]:
    if not isinstance(variant, str):
        return None
    return variant.strip().lower()


def _fa3_prefill_in_domain(shape, dtype, attributes, hardware) -> bool:
    """Fresh (no-context) causal MLA prefill run by one unsplit FA3 call."""

    if dtype != "bf16" or attributes["kv_layout"] != "mla":
        return False
    if (shape["qk_head_dim"], shape["v_head_dim"]) != (192, 128):
        return False
    q_lens, kv_lens = _request_lengths(shape, attributes)
    if q_lens != kv_lens:
        # vLLM runs chunked-context prefill as a causal call plus context
        # chunk calls and merge_attn_states; that sequence is not derived.
        return False
    heads = shape["num_query_heads"]
    return _varlen_num_splits(
        batch=len(q_lens), max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens), num_heads=heads, num_heads_k=heads,
        head_dim=192, head_dim_v=128, block_m=128, block_n=128,
        causal=True, num_sms=int(hardware.num_compute_units)) == 1


def _uniform_packed_decode_queries(q_lens, query_heads: int) -> bool:
    """Every request packs its query rows into one PackGQA M block.

    vLLM 0.24 FlashAttnMLA runs single-token decode and uniform speculative
    verify (every request with the same small query length) through the same
    FA3 decode call; PackGQA folds q_len x heads rows into kBlockM = 64.
    """

    return (len(set(q_lens)) == 1
            and q_lens[0] * query_heads <= 64)


def _fa3_decode_in_domain(shape, dtype, attributes) -> bool:
    """Absorbed MLA decode/verify, one PackGQA M block per request."""

    if dtype != "bf16" or attributes["kv_layout"] != "mla":
        return False
    if (shape["v_head_dim"], shape["kv_lora_rank"],
            shape["qk_head_dim"] - shape["v_head_dim"]) != (512, 512, 64):
        return False
    q_lens, _ = _request_lengths(shape, attributes)
    return _uniform_packed_decode_queries(q_lens, shape["num_query_heads"])


def _fa3_combine_in_domain(shape, dtype, attributes) -> bool:
    """The split combine of the absorbed MLA decode above."""

    if dtype != "bf16" or attributes["kv_layout"] != "mla":
        return False
    if (shape["head_dim"], shape["num_kv_heads"]) != (512, 1):
        return False
    q_lens, _ = _request_lengths(shape, attributes)
    return _uniform_packed_decode_queries(q_lens, shape["num_query_heads"])


def _flashinfer_mla_in_domain(shape, dtype, attributes) -> bool:
    if attributes["attention_backend"] != "flashinfer":
        raise ValueError(
            f"unknown MLA attention_backend {attributes['attention_backend']!r}")
    return (dtype == "bf16" and attributes["kv_layout"] == "mla"
            and shape["qk_head_dim"] - shape["v_head_dim"] == 64
            and shape["v_head_dim"] == 512)


def owns_select_target_mla_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether the target MLA request is in the derived FA3 domain."""

    normalized = _normalized_variant(variant)
    if normalized == "mla_prefill":
        return _fa3_prefill_in_domain(shape, dtype, attributes, hardware)
    if normalized == "mla_decode" and "attention_backend" in attributes:
        return _flashinfer_mla_in_domain(shape, dtype, attributes)
    if normalized == "mla_decode":
        return _fa3_decode_in_domain(shape, dtype, attributes)
    if normalized == "mla_absorbed_prefill":
        return (dtype == "bf16" and attributes["kv_layout"] == "mla"
                and shape["qk_head_dim"] - shape["v_head_dim"] == 64
                and shape["v_head_dim"] == 512)
    return False


def select_target_mla_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select the vLLM 0.24 FA3 MLA prefill or absorbed-decode launch."""

    if not owns_select_target_mla_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 target MLA selector owns only the derived FA3 domain; the "
            "dispatcher routes other requests to the projected package")
    execution_variant = require_variant(
        variant, ("mla_prefill", "mla_decode", "mla_absorbed_prefill"),
        "H200 target MLA")
    require_dtype(dtype, ("bf16",))
    if attributes["causal"] is not True:
        raise ValueError("vLLM MLA attention is causal")
    batch, query_heads, qk_head_dim, value_head_dim = require_shape(
        shape, ("batch", "num_query_heads", "qk_head_dim", "v_head_dim"))
    q_lens, kv_lens = _request_lengths(shape, attributes)
    num_sms = int(hardware.num_compute_units)
    if (execution_variant == "mla_absorbed_prefill"
            or "attention_backend" in attributes):
        # Cooperative persistent launch dim3(num_blks_x = cluster size,
        # num_blks_y = num_sm / cluster size) from scheduler.cuh MLAPlan;
        # CTA_TILE_Q = CTA_TILE_KV = 64, NUM_STAGES = 2, 256 threads; the
        # split-KV merge runs inside the same kernel after grid.sync().
        return LaunchConfig(
            algorithm=FLASHINFER_MLA_PREFILL_ALGORITHM,
            grid=(num_sms, 1, 1),
            block=(_FLASHINFER_MLA_THREADS, 1, 1),
            cluster=(1, 1, 1),
            tiling=TilingSpec(
                tile_shape={
                    "q": _FLASHINFER_MLA_TILE, "kv": _FLASHINFER_MLA_TILE,
                    "qk_head": qk_head_dim - value_head_dim,
                    "value_head": value_head_dim,
                },
                pipeline_depth=_FLASHINFER_MLA_STAGES,
                num_kv_splits=1,
            ),
            parameters=_source_parameters(1, True),
            execution_variant=execution_variant,
        )
    block = (_MMA_WARPGROUPS * _THREADS_PER_WARPGROUP
             + _THREADS_PER_WARPGROUP, 1, 1)
    if execution_variant == "mla_prefill":
        # run_mha_fwd_<90, bf16, 192, 128>: q/k (nope+rope) 192, v 128, H
        # KV heads; tile_size_fwd_sm90(192, 128, causal) = 128 x 128.
        return LaunchConfig(
            algorithm=FA3_PREFILL_ALGORITHM,
            grid=(num_sms, 1, 1),
            block=block,
            cluster=(1, 1, 1),
            tiling=TilingSpec(
                tile_shape={
                    "q": 128, "kv": 128,
                    "qk_head": qk_head_dim, "value_head": value_head_dim,
                },
                pipeline_depth=2,
                num_kv_splits=1,
            ),
            parameters=_source_parameters(1, False),
            execution_variant=execution_variant,
        )

    # run_mha_fwd_<90, bf16, 64, 512>: q_pe/k_pe carry the rope head,
    # qv/v the latent; tile_size_fwd_sm90(64, 512) = 64 x 64.
    rope_dim = qk_head_dim - value_head_dim
    block_m, block_n = 64, 64
    if _decode_uses_cuda_graph_split_bound(attributes, sum(q_lens)):
        splits = _VLLM_CUDA_GRAPH_MAX_NUM_SPLITS
    else:
        splits = _varlen_num_splits(
            batch=batch, max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
            num_heads=query_heads, num_heads_k=1, head_dim=rope_dim,
            head_dim_v=value_head_dim, block_m=block_m, block_n=block_n,
            causal=False, num_sms=num_sms)
    if batch > _PREPARE_VARLEN_MAX_BATCHES_1CTA:
        splits = 1
    return LaunchConfig(
        algorithm=FA3_DECODE_ALGORITHM,
        grid=(num_sms, 1, 1),
        block=block,
        cluster=(1, 1, 1),
        tiling=TilingSpec(
            tile_shape={
                "q": block_m, "kv": block_n,
                "qk_head": rope_dim, "value_head": value_head_dim,
            },
            pipeline_depth=2,
            num_kv_splits=splits,
        ),
        parameters=_source_parameters(splits, True),
        execution_variant=execution_variant,
    )


def _combine_launch(shape: Mapping[str, int], dtype: str,
                    attributes: Mapping[str, object]) -> LaunchConfig:
    """``run_mha_fwd_combine_<bf16, float, 512 -> kBlockK 128>`` launch."""

    require_dtype(dtype, ("bf16",))
    batch, q_len, head_dim, query_heads = require_shape(
        shape, ("batch", "q_len", "head_dim", "num_query_heads"))
    q_lens, _ = _request_lengths(shape, attributes)
    splits = require_positive_integer(attributes, "num_kv_splits")
    if splits <= 1:
        raise ValueError("FA3 combine runs only when num_splits > 1")
    block_k = 128
    block_m = 8 if block_k % 128 == 0 else (16 if block_k % 64 == 0 else 32)
    total_q = sum(q_lens)
    if "graph_model_input_tokens" in attributes:
        # FULL-graph padding (vLLM gpu_model_runner / cudagraph_dispatcher):
        # the decode runs num_reqs_padded = padded tokens requests of query
        # length 0 and seq_len 0 beyond the real ones, and total_q is the
        # padded token count; the combine grid is sized from them and the
        # padded CTAs return at their seqlen check.
        if attributes["graph_logical_tokens"] != total_q:
            raise ValueError("combine graph domain disagrees with q_lens")
        if attributes["graph_runtime_mode"] != "FULL":
            raise ValueError("padded MLA combine requires a FULL graph")
        total_q = attributes["graph_model_input_tokens"]
        batch = total_q
        q_len = 1
    # StaticVarlenTileScheduler::choose_scheduling_algo.
    lower_bound_non_empty = ceil_div(total_q, block_m)
    dense_grid = batch * ceil_div(q_len, block_m)
    if 2 * lower_bound_non_empty >= dense_grid:
        grid = (ceil_div(q_len * query_heads, block_m),
                ceil_div(head_dim, block_k), batch)
    else:
        grid = (ceil_div(total_q * query_heads, block_m) + batch,
                ceil_div(head_dim, block_k), 1)
    return LaunchConfig(
        algorithm=FA3_COMBINE_ALGORITHM,
        grid=grid,
        block=(256, 1, 1),
        tiling=TilingSpec(
            tile_shape={"rows": block_m, "value_head": block_k},
            pipeline_depth=4,
            num_kv_splits=splits,
        ),
        parameters=_source_parameters(splits, False),
        execution_variant="combine",
    )


def owns_select_flash_attn_launch(shape: Mapping[str, int],
                                  dtype: Optional[str],
                                  variant: Optional[str],
                                  attributes: Mapping[str, object],
                                  hardware) -> bool:
    """Whether the FLASH_ATTN request is the FA3 MLA decode split combine."""

    del hardware
    return (_normalized_variant(variant) in ("combine", "split_combine")
            and _fa3_combine_in_domain(shape, dtype, attributes))


def select_flash_attn_launch(shape: Mapping[str, int], dtype: Optional[str],
                             variant: Optional[str],
                             attributes: Mapping[str, object],
                             hardware) -> LaunchConfig:
    """The FA3 MLA split combine (the only FLASH_ATTN launch owned here)."""

    if not owns_select_flash_attn_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 FLASH_ATTN selector owns only the FA3 MLA split combine; "
            "the dispatcher routes other requests to the projected package")
    require_variant(variant, ("combine", "split_combine"), "FLASH_ATTN")
    return _combine_launch(shape, dtype, attributes)


def select_fa3_prepare_varlen_launch(batch: int) -> LaunchConfig:
    """``prepare_varlen_num_blocks`` launch (flash_prepare_scheduler.cu).

    ``num_warps = ceil(b / 31)`` rounded up by NUM_WARP_SWITCH to a power of
    two (at most 32), ``num_ctas = ceil(b / (31 * 32))``.
    """

    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("prepare_varlen batch must be a positive integer")
    warps = ceil_div(batch, 31)
    switched = 1
    while switched < warps:
        switched *= 2
    if switched > 32:
        switched = 32
    return LaunchConfig(
        algorithm="vllm_fa3_prepare_varlen_num_blocks",
        grid=(ceil_div(batch, 31 * 32), 1, 1),
        block=(32 * switched, 1, 1),
        execution_variant="fa3_prepare_varlen",
    )
