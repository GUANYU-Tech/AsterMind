"""Attention operands projected from actual forward allocations and views.

Logical cache selections preserve their real arena without inventing page IDs,
slot contents, or library workspace. Native ABIs lacking source ports stay partial.
"""
from dataclasses import replace
from math import prod

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import EventKind, EventRef, LogicalSelectionView, ProgramEvent
from src.kernel.memory_access_builder import (
    CacheUpdateBindings, ElementwiseBindings, has_public_attention_source,
)
from src.kernel.memory_cache_update import paged_kv_source_index_type
from src.serving.workload.workload_graph import WorkloadKind
from .memory_bindings import source_memory_events


def _reshape(view, name, shape):
    """Reinterpret only actual dense trailing dimensions, preserving row pitch."""
    if view.shape == shape:
        return replace(view, view_id=name)
    if len(view.shape) == 2 and len(shape) == 3 and view.shape == (shape[0], prod(shape[1:])):
        if view.strides_bytes[-1] != view.element_bytes:
            raise ValueError("attention source requires dense inner channels")
        return replace(view, view_id=name, shape=shape,
            strides_bytes=(view.strides_bytes[0], shape[2] * view.element_bytes, view.element_bytes))
    if len(view.shape) == 4 and len(shape) == 3 and prod(view.shape[:2]) == shape[0] and view.shape[2:] == shape[1:]:
        if view.strides_bytes[0] != view.shape[1] * view.strides_bytes[1]:
            raise ValueError("attention source batch rows must be contiguous")
        return replace(view, view_id=name, shape=shape, strides_bytes=view.strides_bytes[1:])
    raise ValueError(f"attention source shape {view.shape} does not match {name} {shape}")


def _logical_read_events(call, source, reads, producers):
    # Derive producer/barrier dependencies from the actual parent ports first.
    parents = tuple((parent, producer) for (_, parent), producer in zip(reads, producers, strict=True))
    allocations, events, ready, visibility = source_memory_events(call, source, parents)
    known_views = {view.view_id for view, _ in reads}
    events = [replace(event, view_id=None) if event.view_id is not None and event.view_id not in known_views
        else event for event in events]
    refs = []
    for (view, parent), ref in zip(reads, ready, strict=True):
        if isinstance(view, LogicalSelectionView):
            identity = call.call_id + ":selected:" + view.view_id
            events.append(ProgramEvent(identity, EventKind.LOGICAL_AVAILABLE, view.allocation_id,
                view.region(), visibility, (ref,)))
            refs.append(EventRef(identity, EventKind.LOGICAL_AVAILABLE))
        else:
            # The ready region is the actual parent and covers its role view.
            refs.append(ref)
    return allocations, tuple(events), tuple(refs), visibility


def _elementwise(call, source, reads, writes, producers):
    allocations, events, ready, visibility = _logical_read_events(call, source, reads, producers)
    views = tuple(view for view, _ in reads) + tuple(writes)
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        views, events, tuple(view.view_id for view, _ in reads),
        tuple(view.view_id for view in writes), ready, call.rank, "source-forward", visibility), None


def _bind_paged_cache(call, source):
    if len(source.inputs) == 2:
        return None, "paged_kv_source_slot_tensor_unavailable"
    if len(source.inputs) != 3 or len(source.outputs) != 1:
        raise ValueError("paged KV source requires K/V/slots and one actual cache arena")
    spec = call.kernel_spec
    shape = tuple(spec.shape[name] for name in ("seq_len", "num_kv_heads", "head_dim"))
    key, value = (_reshape(view, name, shape) for view, name in zip(source.inputs[:2], ("key", "value")))
    slots = replace(source.inputs[2], view_id="slots")
    if slots.shape != (shape[0],) or slots.dtype != "int64":
        raise ValueError("paged KV source slot tensor must use its declared INT64 ABI")
    slot_abi = paged_kv_source_index_type(spec)
    if slot_abi is None:
        return None, "paged_kv_source_tensor_only_compatibility_abi_unavailable"
    if (slots.dtype, slots.element_bytes) != slot_abi:
        return None, "paged_kv_source_slot_cast_unbound"
    cache, = source.outputs
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip((key, value, slots), source.input_producers, strict=True)))
    return CacheUpdateBindings(call.call_id + ":source-memory", call.call_id, allocations,
        (key, value, slots), events, "key", "value", "slots", cache.allocation_id,
        cache.allocation_id, *ready, call.rank, "source-forward", visibility), None


def _bind_public_attention(call, source):
    spec = call.kernel_spec
    if not has_public_attention_source(spec):
        return None, "attention_native_metadata_or_workspace_ports_unavailable"
    if "model_input_q_lens" in spec.attributes and (
            spec.attributes["model_input_q_lens"] != spec.attributes["q_lens"]
            or spec.attributes["model_input_kv_lens"] != spec.attributes["kv_lens"]):
        return None, "attention_graph_physical_operand_mapping_unavailable"
    if spec.launch.tiling.num_kv_splits not in (None, 1):
        return None, "attention_split_workspace_source_ports_unavailable"
    if len(source.inputs) != 4 or len(source.outputs) != 1:
        return None, "attention_source_public_operand_layout_unavailable"
    query, key, value, cache = source.inputs
    q, kv = sum(spec.attributes["q_lens"]), sum(spec.attributes["kv_lens"])
    heads, kv_heads, dimension = (spec.shape[name] for name in ("num_query_heads", "num_kv_heads", "head_dim"))
    query = _reshape(query, "query", (q, heads, dimension))
    output = _reshape(source.outputs[0], "output", (q, heads, dimension))
    reads = [(query, query)]
    producers = [source.input_producers[0]]
    # The actual cache call consumes prior-context K/V even when fresh K/V
    # tensors also appear on the public wrapper. Unknown native page selection
    # must not be substituted by those shorter fresh tensors.
    for role in ("key", "value"):
        selected = LogicalSelectionView(role, cache.allocation_id, (kv, kv_heads, dimension),
            spec.dtype, spec.shape["dtype_bytes"], call.call_id + ":cache:" + role, None)
        reads.append((selected, cache))
        producers.append(source.input_producers[3])
    return _elementwise(call, source, tuple(reads), (output,), tuple(producers))


def _bind_mla_decode(call, source):
    spec = call.kernel_spec
    if spec.launch is None:
        return None, "mla_mathematical_memory_source_unavailable"
    if spec.launch.tiling.num_kv_splits not in (None, 1) or spec.launch.algorithm == "vllm_fa3_sm90_mla_decode":
        return None, "mla_split_workspace_source_ports_unavailable"
    if len(source.inputs) != 2 or len(source.outputs) != 2:
        return None, "mla_source_public_operand_layout_unavailable"
    query, cache = source.inputs
    q_lens, kv_lens = spec.attributes["q_lens"], spec.attributes["kv_lens"]
    rows, heads, dimension = sum(q_lens), spec.shape["num_query_heads"], spec.shape["qk_head_dim"]
    if query.shape[0] != rows and len(query.shape) != 4:
        return None, "mla_padded_public_query_mapping_unavailable"
    query = _reshape(query, "query", (rows, heads, dimension))
    output = _reshape(source.outputs[0], "output", (rows, heads, spec.shape["v_head_dim"]))
    rank = spec.shape["kv_lora_rank"]
    reads, producers, offset = [], [], 0
    for index, (count, length) in enumerate(zip(q_lens, kv_lens, strict=True)):
        suffix = "" if len(q_lens) == 1 else f".request{index}"
        qview = replace(query, view_id="query" + suffix, shape=(count, heads, dimension),
            offset_bytes=query.offset_bytes + offset * query.strides_bytes[0])
        reads.append((qview, query)); producers.append(source.input_producers[0]); offset += count
        for role, width in (("compressed_kv", rank), ("key_rope", dimension - rank)):
            if not width:
                continue
            name = role + suffix
            view = LogicalSelectionView(name, cache.allocation_id, (length, width), spec.dtype,
                spec.shape["dtype_bytes"], call.call_id + ":" + name, None)
            reads.append((view, cache)); producers.append(source.input_producers[1])
    return _elementwise(call, source, tuple(reads), (output,), tuple(producers))


def _bind_mathematical_attention(call, source):
    spec = call.kernel_spec
    rows = sum(spec.attributes["q_lens"])
    cache_rows = sum(spec.attributes["kv_lens"])
    heads = spec.shape["num_query_heads"]
    if spec.variant in ("prefill", "decode"):
        dimension, kv_heads = spec.shape["head_dim"], spec.shape["num_kv_heads"]
        if len(source.inputs) not in (2, 3, 4) or not source.outputs:
            return None, "gqa_mathematical_source_operand_ports_unavailable"
        if (prod(source.inputs[0].shape) != rows * heads * dimension
                or prod(source.outputs[0].shape) != rows * heads * dimension):
            return None, "gqa_padded_public_operand_mapping_unavailable"
        query = _reshape(source.inputs[0], "query", (rows, heads, dimension))
        output = _reshape(source.outputs[0], "output", (rows, heads, dimension))
        reads = [(query, query)]
        producers = [source.input_producers[0]]
        if len(source.inputs) == 3:
            if spec.attributes["q_lens"] != spec.attributes["kv_lens"]:
                return None, "gqa_direct_history_source_mapping_unavailable"
            for index, role in enumerate(("key", "value"), 1):
                view = _reshape(source.inputs[index], role, (cache_rows, kv_heads, dimension))
                reads.append((view, view)); producers.append(source.input_producers[index])
        else:
            cache_index = 1 if len(source.inputs) == 2 else 3
            cache = source.inputs[cache_index]
            for role in ("key", "value"):
                view = LogicalSelectionView(role, cache.allocation_id, (cache_rows, kv_heads, dimension),
                    spec.dtype, spec.shape["dtype_bytes"], call.call_id + ":cache:" + role, None)
                reads.append((view, cache)); producers.append(source.input_producers[cache_index])
        return _elementwise(call, source, tuple(reads), (output,), tuple(producers))
    if spec.variant in ("mla_decode", "mla_absorbed_prefill"):
        if len(source.inputs) != 2 or len(source.outputs) not in (1, 2):
            return None, "mla_mathematical_source_operand_ports_unavailable"
        dimension, value_dim = spec.shape["qk_head_dim"], spec.shape["v_head_dim"]
        if (prod(source.inputs[0].shape) != rows * heads * dimension
                or prod(source.outputs[0].shape) != rows * heads * value_dim):
            return None, "mla_padded_public_operand_mapping_unavailable"
        query = _reshape(source.inputs[0], "query", (rows, heads, dimension))
        cache = source.inputs[1]
        if cache.dtype != spec.dtype or cache.shape[-1] != dimension:
            raise ValueError("MLA source cache must retain the declared latent/RoPE record")
        selected = LogicalSelectionView("compressed_kv", cache.allocation_id, (cache_rows, dimension),
            spec.dtype, spec.shape["dtype_bytes"], call.call_id + ":cache:compressed_kv", None)
        output = _reshape(source.outputs[0], "output", (rows, heads, value_dim))
        writes = (output,)
        if len(source.outputs) == 2:
            writes += (replace(source.outputs[1], view_id="lse"),)
        return _elementwise(call, source, ((query, query), (selected, cache)), writes,
            source.input_producers)
    return None, "attention_mathematical_variant_ports_unavailable"


def _bind_mla_cache(call, source):
    spec = call.kernel_spec
    if len(source.inputs) != 3 or len(source.outputs) != 1:
        return None, "mla_cache_source_latent_rope_slots_unavailable"
    latent, rope, slots = source.inputs
    cache, = source.outputs
    tokens, dimension = spec.shape["seq_len"], spec.shape["compressed_dim"]
    if (latent.shape[0] != tokens or len(latent.shape) != 2 or rope.shape[:2] != (tokens, 1)
            or latent.shape[1] + rope.shape[2] != dimension or slots.shape != (tokens,)
            or slots.dtype != "int64" or cache.dtype != spec.dtype
            or latent.dtype != spec.dtype or rope.dtype != spec.dtype):
        raise ValueError("MLA cache source latent/RoPE/slot geometry disagrees with the kernel")
    latent = _reshape(latent, "latent", (tokens, 1, latent.shape[1]))
    rope, slots = replace(rope, view_id="rope"), replace(slots, view_id="slots")
    selection = f"target_cache_tiles.mla_kv_cache.cache_payload.T{tokens}.D{dimension}.S0"
    output = LogicalSelectionView("cache_payload", cache.allocation_id, (tokens, 1, dimension),
        spec.dtype, spec.shape["dtype_bytes"], selection, None)
    return _elementwise(call, source, ((latent, latent), (rope, rope), (slots, slots)),
        (output,), source.input_producers)


def _bind_v4_insert(call, source):
    if len(source.inputs) == 4:
        return None, "v4_insert_source_slot_tensor_unavailable"
    if len(source.inputs) != 5 or len(source.outputs) != 2:
        raise ValueError("V4 insert requires actual Q/KV/position/RoPE/slot inputs and Q/cache outputs")
    spec = call.kernel_spec
    rows, heads, padded = (spec.shape[name] for name in ("seq_len", "num_query_heads", "padded_query_heads"))
    query, kv, positions, rotary, slots = source.inputs
    output, cache = source.outputs
    query = _reshape(query, "query", (rows, heads, 512))
    kv = _reshape(kv, "kv", (rows, 1, 512))
    positions, slots = replace(positions, view_id="positions"), replace(slots, view_id="slots")
    if (positions.shape != (rows,) or slots.shape != (rows,)
            or positions.dtype != "int64" or slots.dtype != "int64"
            or output.shape != (rows, padded, 512) or cache.dtype != "uint8" or cache.shape[-1] != 584
            or rotary.dtype != "fp32" or rotary.shape[-1] != 64):
        raise ValueError("V4 insert actual source ports disagree with native typed operands")
    def selection(role, parent, shape, dtype, width):
        identity = f"target_cache_tiles.dsv4_qnorm_rope_kv_insert.{role}.T{rows}.D512.S8.H{heads}.P{padded}"
        return LogicalSelectionView(role, parent.allocation_id, shape, dtype, width, identity, None)
    rope = selection("rope", rotary, (rows, 1, 64), "fp32", 4)
    live = replace(output, view_id="query_live", shape=(rows, heads, 512))
    writes = (live,)
    if heads < padded:
        writes += (replace(output, view_id="query_pad", shape=(rows, padded-heads, 512),
            offset_bytes=output.offset_bytes + heads * output.strides_bytes[1]),)
    writes += tuple(selection(role, cache, (rows, 1, dimension), dtype, width)
        for role, dimension, dtype, width in (("cache_payload", 448, "fp8", 1),
            ("cache_rope", 64, "bf16", 2), ("cache_scale", 8, "uint8", 1)))
    reads = ((query, query), (kv, kv), (positions, positions), (slots, slots), (rope, rotary))
    producers = tuple(source.input_producers[index] for index in (0, 1, 2, 4, 3))
    return _elementwise(call, source, reads, writes, producers)


def bind_attention_memory(call, source):
    """Return typed actual operands or a bounded reason for missing source ABI."""
    spec = call.kernel_spec
    if source.kind is WorkloadKind.KV_UPDATE and spec.kind is KernelKind.KV_CACHE:
        if source.variant != spec.variant:
            return None, "cache_source_variant_mapping_unavailable"
        if spec.variant == "paged_kv_cache":
            return _bind_paged_cache(call, source)
        if spec.variant == "mla_kv_cache":
            return _bind_mla_cache(call, source)
        if spec.variant == "dsv4_qnorm_rope_kv_insert":
            return _bind_v4_insert(call, source)
        reasons = {
            "dsv4_compressor_state_cache": "v4_compressor_source_slots_and_ape_unavailable",
            "dsv4_compressed_state_cache": "v4_finish_source_metadata_and_gamma_unavailable",
            "dsa_fp8_index_cache": "dsa_index_cache_source_key_and_slot_ports_unavailable",
        }
        if spec.variant in reasons:
            return None, reasons[spec.variant]
        return None, "cache_source_native_operand_mapping_unavailable"
    if source.kind is WorkloadKind.FLASH_ATTN and spec.kind is KernelKind.FLASH_ATTN:
        from src.kernel.memory_access_builder import has_mathematical_memory_source
        if has_mathematical_memory_source(spec):
            return _bind_mathematical_attention(call, source)
        if spec.variant in ("prefill", "decode") and source.variant in ("prefill", "decode"):
            return _bind_public_attention(call, source)
        if spec.variant == source.variant == "mla_decode":
            return _bind_mla_decode(call, source)
        return None, "attention_source_native_operand_mapping_unavailable"
    if spec.kind is KernelKind.SPARSE_ATTN_INDEXER:
        return None, "dsa_indexer_source_score_backing_unavailable"
    if spec.kind is KernelKind.SPARSE_ATTN:
        if spec.variant.startswith("dsv4_"):
            return None, "v4_sparse_source_sink_controls_and_native_workspace_unavailable"
        return None, "dsa_sparse_mathematical_memory_source_unavailable"
    if spec.kind is KernelKind.EAGER_ATTN:
        return None, "eager_attention_source_score_and_probability_backings_unavailable"
    return None, "attention_source_family_operand_mapping_unavailable"
