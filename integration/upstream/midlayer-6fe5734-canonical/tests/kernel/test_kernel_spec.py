import dataclasses
import math
from dataclasses import replace

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import (
    KernelKind,
    KernelPrediction,
    KernelSpec,
    PAGED_KV_CACHE_ATTRIBUTES,
    PhysicalTransferConfig,
)
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.model_contract import (
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
)


def _all_reduce_transfer(**overrides):
    fields = {
        "algorithm": "ring",
        "bytes": 4096,
        "direction": "collective",
        "link": "nvlink",
        "topology": "ring",
        "participants": (0, 1),
        "operation": "all_reduce",
        "bytes_semantics": "per_rank_input",
        "ring_order": (0, 1),
        "reduction": "sum",
        "backend": "analytical_roofline_v1",
        "resource_id": "node0-nvlink",
    }
    fields.update(overrides)
    return PhysicalTransferConfig(**fields)


def test_compute_kernel_requires_launch_config():
    with pytest.raises(ValueError, match="requires a complete LaunchConfig"):
        KernelSpec(KernelKind.GEMM, "nvidia_a800", {"m": 1, "n": 1, "k": 1})


def _unsupported_spec(kind, shape, dtype, variant, attributes):
    return KernelSpec(
        kind,
        "nvidia_a800",
        shape,
        dtype=dtype,
        variant=variant,
        attributes=attributes,
        execution_support="unsupported",
        unsupported_reason="launch_selector_unavailable",
    )


@pytest.mark.parametrize(
    ("kind", "shape", "dtype", "variant", "attributes"),
    (
        (
            KernelKind.GEMM,
            {"b": 2, "m": 3, "n": 4, "k": 5},
            "bf16",
            "batched_gemm",
            {
                "batch_layout": "strided",
                "a_batch_stride_elements": 15,
                "b_batch_stride_elements": 20,
                "c_batch_stride_elements": 12,
                "weight_dtype": "bf16",
                "weight_group_size": 0,
            },
        ),
        (
            KernelKind.GEMM,
            {"b": 1, "m": 3, "n": 8, "k": 16},
            "fp32",
            "moe_router",
            {
                "score_function": "sigmoid",
                "topk_method": "noaux_tc",
                "expert_group_count": 2,
                "topk_group_count": 1,
                "normalize_topk_probability": True,
                "weight_dtype": "bf16",
                "weight_group_size": 0,
            },
        ),
        (
            KernelKind.FLASH_ATTN,
            {
                "batch": 1, "q_len": 3, "kv_len": 5,
                "num_query_heads": 4, "qk_head_dim": 6,
                "v_head_dim": 8, "kv_lora_rank": 10,
            },
            "bf16",
            "mla_prefill",
            {
                "causal": True, "kv_layout": "mla", "kv_dist": "uniform",
                "q_lens": (3,), "kv_lens": (5,),
                "full_cuda_graphs": False, "max_cudagraph_capture_size": 0,
            },
        ),
        (
            KernelKind.SPARSE_ATTN_INDEXER,
            {
                "batch": 1, "q_len": 3, "kv_len": 5,
                "num_query_heads": 2, "head_dim": 8, "topk": 5,
            },
            "int32",
            "dsa_indexer_prefill",
            {
                "causal": True,
                "score_accumulator_dtype": "fp32",
                "score_function": "weighted_dot_product_topk",
                "cache_layout": "paged_mla_indexer",
                "cache_quantization": "fp8_e4m3_with_ue8m0_scale",
                "q_lens": (3,), "kv_lens": (5,),
                "request_phases": ("prefill",),
                "topk_capacity": 8,
                "topk_index_dtype": "int32",
                "topk_sentinel": -1,
            },
        ),
        (
            KernelKind.SPARSE_ATTN_INDEXER,
            {"element_count": 4},
            "int32",
            "dsa_index_reuse",
            {
                "reuse_semantics": "alias",
                "topk_capacity": 8,
                "topk_index_dtype": "int32",
            },
        ),
        (
            KernelKind.SPARSE_ATTN,
            {
                "batch": 1, "q_len": 1, "kv_len": 5,
                "num_query_heads": 4, "qk_head_dim": 6,
                "v_head_dim": 8, "topk": 5,
            },
            "bf16",
            "dsa_sparse_decode",
            {
                "causal": True, "indexer_type": "full",
                "kv_layout": "mla", "kv_dist": "uniform",
                "q_lens": (1,), "kv_lens": (5,),
                "selected_kv_lens": (5,), "topk_capacity": 8,
                "topk_index_dtype": "int32",
            },
        ),
        (
            KernelKind.KV_CACHE,
            {"element_count": 30, "seq_len": 3, "compressed_dim": 10},
            "bf16",
            "mla_kv_cache",
            {"cache_layout": "paged_mla", "cache_write_semantics": "append"},
        ),
        (
            KernelKind.KV_CACHE,
            {
                "element_count": 36, "seq_len": 3,
                "key_dim": 8, "scale_bytes_per_token": 4,
            },
            "uint8",
            "dsa_fp8_index_cache",
            {
                "cache_layout": "paged_mla_indexer",
                "cache_write_semantics": "append",
                "quantization": "fp8_e4m3_with_ue8m0_scale",
            },
        ),
        (
            KernelKind.FUSED_MOE,
            {
                "tokens": 2, "experts": 4, "topk": 2,
                "hidden": 8, "intermediate": 16,
            },
            "bf16",
            "routed_experts",
            {
                "tokens_per_expert": (1, 1, 1, 1),
                "local_tokens_per_expert": (1, 1, 1, 1),
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": (0, 1, 2, 3),
                "expert_weight_tile_rows": None,
                "routed_scaling_factor": 1.0,
                "routed_weight_dtype": "int4",
                "weight_group_size": 4,
            },
        ),
        (
            KernelKind.VECTOR,
            {"rows": 3, "hidden": 8},
            "bf16",
            "layernorm",
            {"affine": True, "epsilon": 1e-5},
        ),
        (
            KernelKind.VECTOR,
            {
                "element_count": 36, "seq_len": 3,
                "num_query_heads": 2, "num_key_heads": 1, "head_dim": 4,
            },
            "bf16",
            "mla_rope",
            {
                "rope_layout": "interleaved", "rope_type": "default",
                "rope_theta": 8_000_000.0, "rope_factor": None,
                "rope_original_max_position_embeddings": None,
                "rope_beta_fast": None, "rope_beta_slow": None,
                "rope_mscale": None, "rope_mscale_all_dim": None,
            },
        ),
    ),
)
def test_unsupported_target_compute_specs_are_typed_and_launchless(
        kind, shape, dtype, variant, attributes):
    spec = _unsupported_spec(kind, shape, dtype, variant, attributes)

    assert spec.kind is kind
    assert spec.execution_support == "unsupported"
    assert spec.unsupported_reason == "launch_selector_unavailable"
    assert spec.launch is None


def test_unsupported_compute_contract_fails_closed():
    fields = {
        "kind": KernelKind.KV_CACHE,
        "shape": {"element_count": 30, "seq_len": 3, "compressed_dim": 10},
        "dtype": "bf16",
        "variant": "mla_kv_cache",
        "attributes": {
            "cache_layout": "paged_mla",
            "cache_write_semantics": "append",
        },
        "execution_support": "unsupported",
    }
    with pytest.raises(ValueError, match="unsupported_reason"):
        KernelSpec(hardware_id="nvidia_a800", **fields)
    with pytest.raises(ValueError, match="unknown KV_CACHE"):
        KernelSpec(
            hardware_id="nvidia_a800",
            unsupported_reason="launch_selector_unavailable",
            **{**fields, "attributes": {**fields["attributes"], "extra": 1}},
        )


def test_dsa_indexer_allows_single_token_prefill_tail_chunk():
    # A 1-token chunked-prefill/resume tail chunk keeps its prefill phase
    # while sharing the indexer batch with q_len=1 decode requests, and a
    # decode-phase multi-token row (an MTP/EAGLE verify window) is equally
    # legal: the phase split places no q_len constraint on either group,
    # while genuinely wrong shapes (q_len above the request's KV length)
    # stay fail-closed.
    shape = {
        "batch": 2, "q_len": 1, "kv_len": 8,
        "num_query_heads": 2, "head_dim": 8, "topk": 8,
    }
    attributes = {
        "causal": True,
        "score_accumulator_dtype": "fp32",
        "score_function": "weighted_dot_product_topk",
        "cache_layout": "paged_mla_indexer",
        "cache_quantization": "fp8_e4m3_with_ue8m0_scale",
        "q_lens": (1, 1), "kv_lens": (8, 8),
        "request_phases": ("prefill", "decode"),
        "topk_capacity": 8,
        "topk_index_dtype": "int32",
        "topk_sentinel": -1,
    }

    spec = _unsupported_spec(
        KernelKind.SPARSE_ATTN_INDEXER,
        shape,
        "int32",
        "dsa_indexer_mixed",
        attributes,
    )

    assert spec.attributes["request_phases"] == ("prefill", "decode")
    verify_window = _unsupported_spec(
        KernelKind.SPARSE_ATTN_INDEXER,
        {**shape, "q_len": 2},
        "int32",
        "dsa_indexer_mixed",
        {
            **attributes,
            "q_lens": (2, 1), "kv_lens": (8, 8),
            "request_phases": ("decode", "prefill"),
        },
    )
    assert verify_window.attributes["request_phases"] == ("decode", "prefill")
    with pytest.raises(
            ValueError, match="requires q_lens <= kv_lens"):
        _unsupported_spec(
            KernelKind.SPARSE_ATTN_INDEXER,
            {**shape, "q_len": 9, "kv_len": 8},
            "int32",
            "dsa_indexer_mixed",
            {
                **attributes,
                "q_lens": (9, 1), "kv_lens": (8, 8),
                "request_phases": ("decode", "prefill"),
            },
        )


def _a800_vector_launch(shape, variant, attributes=None):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    normalized_attributes = {} if attributes is None else attributes
    return select_launch_config(
        hardware.hardware_id,
        KernelKind.VECTOR,
        shape,
        "bf16",
        variant,
        normalized_attributes,
        hardware=hardware,
    )


def test_memory_dependency_task_marker_rejects_fp16_rmsnorm_spoof():
    shape = {"rows": 1, "hidden": 4096}
    launch = _a800_vector_launch(shape, "rmsnorm")

    with pytest.raises(ValueError, match="requires NVIDIA A800 BF16"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="fp16",
            variant="rmsnorm",
            launch=launch,
        )


def test_memory_dependency_task_marker_rejects_block512_rmsnorm_spoof():
    shape = {"rows": 1, "hidden": 4096}
    launch = dataclasses.replace(
        _a800_vector_launch(shape, "rmsnorm"), block=(512, 1, 1))

    with pytest.raises(ValueError, match="source launch identity"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_graph_padding_validates_source_launch_against_model_input_shape():
    logical_shape = {"rows": 11, "hidden": 4096}
    model_input_shape = {"rows": 16, "hidden": 4096}
    launch = _a800_vector_launch(model_input_shape, "rmsnorm")

    spec = KernelSpec(
        KernelKind.VECTOR,
        "nvidia_a800",
        logical_shape,
        dtype="bf16",
        variant="rmsnorm",
        launch=launch,
        attributes={
            "graph_logical_tokens": 11,
            "graph_model_input_tokens": 16,
            "graph_runtime_mode": "FULL",
            "model_input_shape": model_input_shape,
        },
    )

    assert spec.shape["rows"] == 11
    assert spec.launch.grid == (16, 1, 1)


def test_vector_schema_rejects_warp_instruction_spoof():
    shape = {"element_count": 2048}
    attributes = {"reads": 2, "writes": 1, "operation": "add"}
    launch = _a800_vector_launch(
        shape,
        "elementwise_pytorch_vec4",
        attributes,
    )
    parameters = dict(launch.parameters)
    parameters["warp_instructions"] = 1.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(
            ValueError, match="disagrees with its exact source launch identity"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="elementwise_pytorch_vec4",
            launch=launch,
            attributes=attributes,
        )


def test_memory_dependency_task_marker_rejects_c500_silu_spoof():
    shape = {"tok": 1, "intermediate": 4096}
    launch = _a800_vector_launch(shape, "silu")

    with pytest.raises(ValueError, match="requires NVIDIA A800 BF16"):
        KernelSpec(
            KernelKind.VECTOR,
            "metax_c500",
            shape,
            dtype="bf16",
            variant="silu",
            launch=launch,
        )


def test_memory_dependency_task_marker_must_equal_one():
    shape = {"rows": 1, "hidden": 4096}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = 2.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="must equal one"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_memory_dependency_task_marker_requires_static_source_fields():
    shape = {"rows": 1, "hidden": 4096}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    del parameters["norm_reduction_tree_flops"]
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="launch parameter schema mismatch"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_source_bound_l2_read_transactions_reject_wrong_static_count():
    shape = {"rows": 1, "hidden": 4096}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    parameters[
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER] = 4.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(
            ValueError, match="tail-free source transaction geometry"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_rmsnorm_shared_gamma_rejects_wrong_residency_copy_count():
    shape = {"rows": 256, "hidden": 16384}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    parameters["norm_weight_l2_copies"] = 147.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="source launch identity"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_shared_gamma_copy_schema_does_not_expand_to_add_rmsnorm():
    shape = {"rows": 256, "hidden": 4096}
    launch = _a800_vector_launch(shape, "add_rmsnorm")
    parameters = dict(launch.parameters)
    parameters["norm_weight_l2_copies"] = 148.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="launch parameter schema mismatch"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="add_rmsnorm",
            launch=launch,
        )


def test_source_bound_l2_read_transactions_are_required_for_exact_identity():
    shape = {"rows": 1, "hidden": 4096}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    del parameters[SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER]
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="requires L2 read transaction"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_source_bound_l2_read_transactions_require_source_identity():
    shape = {"rows": 1, "hidden": 4096}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    del parameters[SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER]
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="launch parameter schema mismatch"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_source_bound_l2_read_transactions_reject_unsupported_variant():
    shape = {
        "seq_len": 1,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
    }
    attributes = {"is_neox": True, "key_present": True}
    launch = _a800_vector_launch(shape, "rope", attributes)
    parameters = dict(launch.parameters)
    parameters[
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER] = 1.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="source transaction geometry"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rope",
            launch=launch,
            attributes=attributes,
        )


def test_source_bound_l2_read_transactions_reject_producer_tail():
    shape = {"rows": 1, "hidden": 2080}
    launch = _a800_vector_launch(shape, "rmsnorm")
    parameters = dict(launch.parameters)
    parameters[
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER] = 2.0
    launch = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(ValueError, match="source transaction geometry"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def _tensor_iterator_copy_spec(
        *, dtype="bf16", rank=4.0, grid=2, operation="copy",
        hardware_id="nvidia_a800"):
    if operation in ("add", "mul"):
        reads = 2
    elif operation == "copy":
        reads = 1
    else:
        raise ValueError(f"unsupported tensor iterator operation: {operation!r}")
    launch = LaunchConfig(
        "observed_flash_attention_copy:elementwise",
        (grid, 1, 1),
        (128, 1, 1),
        parameters={
            SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: rank,
        },
        execution_variant="elementwise",
    )
    return KernelSpec(
        KernelKind.VECTOR,
        hardware_id,
        {"element_count": 1024},
        dtype=dtype,
        variant="elementwise",
        launch=launch,
        attributes={"reads": reads, "writes": 1, "operation": operation},
    )


def test_tensor_iterator_copy_task_accepts_source_rank_and_launch():
    spec = _tensor_iterator_copy_spec()
    assert spec.launch.parameters == {
        SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: 4.0,
    }


def _norm_input_materialization_copy_spec(*, algorithm=(
        "observed_norm_input_materialization_copy:elementwise"),
        include_materialization=True):
    attributes = {"reads": 1, "writes": 1, "operation": "copy"}
    if include_materialization:
        attributes.update({
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": 2,
            "input_rows_per_group": 4,
            "input_group_stride_elements": 6144,
            "input_row_stride_elements": 128,
        })
    launch = LaunchConfig(
        algorithm,
        (2, 1, 1),
        (128, 1, 1),
        parameters={
            SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: 4.0,
        },
        execution_variant="elementwise",
    )
    return KernelSpec(
        KernelKind.VECTOR,
        "nvidia_a800",
        {"element_count": 1024},
        dtype="bf16",
        variant="elementwise",
        launch=launch,
        attributes=attributes,
    )


def test_tensor_iterator_copy_task_accepts_norm_input_materialization():
    spec = _norm_input_materialization_copy_spec()
    assert spec.launch.algorithm == (
        "observed_norm_input_materialization_copy:elementwise")
    assert spec.launch.parameters == {
        SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: 4.0,
    }


def test_tensor_iterator_copy_task_requires_norm_materialization_contract():
    with pytest.raises(KeyError):
        _norm_input_materialization_copy_spec(include_materialization=False)



@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"rank": 3.0}, "source rank or launch geometry"),
        ({"grid": 1}, "source rank or launch geometry"),
        ({"operation": "add"},
         "FlashAttention or Norm input-materialization copy"),
        ({"hardware_id": "metax_c500"},
         "FlashAttention or Norm input-materialization copy"),
    ),
)
def test_tensor_iterator_copy_task_rejects_spoofed_identity(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _tensor_iterator_copy_spec(**kwargs)


def test_transfer_kernel_rejects_gpu_launch_geometry():
    transfer = _all_reduce_transfer()
    launch = LaunchConfig("invalid", (1, 1, 1), (1, 1, 1))
    with pytest.raises(ValueError, match="does not use GPU launch geometry"):
        KernelSpec(KernelKind.COMMUNICATION, "nvidia_a800", launch=launch, transfer=transfer)


def test_communication_requires_topology():
    transfer = _all_reduce_transfer(topology=None)
    with pytest.raises(ValueError, match="requires topology"):
        KernelSpec(KernelKind.COMMUNICATION, "nvidia_a800", transfer=transfer)


def test_communication_accepts_explicit_collective_and_p2p_semantics():
    all_reduce = _all_reduce_transfer()
    all_gather = replace(
        all_reduce,
        operation="all_gather",
        reduction=None,
    )
    all_to_all = replace(
        all_reduce,
        algorithm="pairwise",
        bytes=4096,
        operation="all_to_all",
        topology="fully_connected",
        ring_order=(),
        reduction=None,
    )
    send_recv = replace(
        all_reduce,
        algorithm="send_recv",
        direction="point_to_point",
        topology="point_to_point",
        operation="send_recv",
        bytes_semantics="payload",
        source_rank=0,
        destination_rank=1,
        ring_order=(),
        reduction=None,
    )

    specs = tuple(
        KernelSpec(KernelKind.COMMUNICATION, "nvidia_a800", transfer=transfer)
        for transfer in (all_reduce, all_gather, all_to_all, send_recv)
    )

    assert tuple(spec.transfer.operation for spec in specs) == (
        "all_reduce", "all_gather", "all_to_all", "send_recv")


@pytest.mark.parametrize(
    "transfer,message",
    (
        (
            replace(_all_reduce_transfer(), bytes_semantics="payload"),
            "per_rank_input bytes_semantics",
        ),
        (
            replace(_all_reduce_transfer(), ring_order=(1, 2)),
            "permutation of participants",
        ),
        (
            replace(_all_reduce_transfer(), reduction=None),
            "requires sum reduction",
        ),
        (
            replace(
                _all_reduce_transfer(),
                operation="send_recv",
                direction="point_to_point",
                algorithm="send_recv",
                topology="point_to_point",
                bytes_semantics="payload",
                source_rank=0,
                destination_rank=2,
                ring_order=(),
                reduction=None,
            ),
            "endpoints must exactly match participants",
        ),
    ),
)
def test_communication_rejects_inconsistent_operation_contract(
    transfer, message,
):
    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.COMMUNICATION,
            "nvidia_a800",
            transfer=transfer,
        )


def test_protobridge_backend_requires_versioned_config_hash():
    with pytest.raises(ValueError, match="config SHA256"):
        KernelSpec(
            KernelKind.COMMUNICATION,
            "nvidia_a800",
            transfer=replace(
                _all_reduce_transfer(), backend="protobridge"),
        )


def test_kv_offload_requires_tier():
    transfer = PhysicalTransferConfig("dma", 4096, "load", "pcie")
    with pytest.raises(ValueError, match="requires tier"):
        KernelSpec(KernelKind.KV_OFFLOAD, "nvidia_a800", transfer=transfer)


def test_native_flash_attention_d2d_copy_requires_c500():
    transfer = PhysicalTransferConfig(
        "native_d2d_flash_attention_copy", 262144, "d2d", "dram")

    with pytest.raises(ValueError, match="requires metax_c500"):
        KernelSpec(KernelKind.MEMCPY, "nvidia_a800", transfer=transfer)


@pytest.mark.parametrize(
    "kind, transfer, message",
    [
        (
            KernelKind.COMMUNICATION,
            _all_reduce_transfer(tier="host"),
            "does not accept tier",
        ),
        (
            KernelKind.MEMCPY,
            PhysicalTransferConfig(
                "dma", 4096, "d2d", "dram", participants=(0,)),
            "does not accept communication-specific fields",
        ),
        (
            KernelKind.KV_OFFLOAD,
            PhysicalTransferConfig(
                "dma", 4096, "load", "pcie", tier="host",
                topology="point_to_point"),
            "does not accept communication-specific fields",
        ),
    ],
)
def test_transfer_kinds_reject_irrelevant_fields(kind, transfer, message):
    with pytest.raises(ValueError, match=message):
        KernelSpec(kind, "nvidia_a800", transfer=transfer)


def test_transfer_participants_reject_one_shot_iterables():
    with pytest.raises(TypeError, match="list or tuple"):
        PhysicalTransferConfig(
            "ring", 4096, "collective", "nvlink",
            topology="ring", participants=(rank for rank in (0, 1)))


@pytest.mark.parametrize("missing", ["algorithm", "direction", "link"])
def test_all_transfers_require_common_physical_fields(missing):
    fields = {
        "algorithm": "dma",
        "bytes": 1024,
        "direction": "d2d",
        "link": "dram",
    }
    fields.pop(missing)
    with pytest.raises(TypeError, match=missing):
        PhysicalTransferConfig(**fields)


def test_prediction_exposes_nanoseconds_and_immutable_breakdowns():
    prediction = KernelPrediction(1e-6, {"dram": 1e-6}, 2e-7, "dram", {"dram": 1e-6})
    assert prediction.t_total_ns == 1000.0
    with pytest.raises(TypeError):
        prediction.t_per_group["dram"] = 0.0


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_prediction_rejects_non_finite_times(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        KernelPrediction(
            value,
            {"dram": 1e-6},
            2e-7,
            "dram",
            {"dram": 1e-6},
        )


@pytest.mark.parametrize(
    "kind,variant,shape,expected",
    [
        (KernelKind.VECTOR, "rotary", {
            "seq_len": 4, "num_query_heads": 8,
            "num_kv_heads": 2, "head_dim": 16,
        }, 640),
        (KernelKind.VECTOR, "activation", {
            "tok": 4, "intermediate": 32}, 128),
        (KernelKind.KV_CACHE, "paged_kv_cache", {
            "seq_len": 4, "num_kv_heads": 2, "head_dim": 16}, 128),
    ],
)
def test_vector_structured_shapes_normalize_one_element_count(
        kind, variant, shape, expected):
    if variant == "paged_kv_cache":
        attributes = PAGED_KV_CACHE_ATTRIBUTES
    elif variant in ("rotary", "activation"):
        attributes = {}
    else:
        raise AssertionError(f"unhandled test variant: {variant!r}")
    launch = LaunchConfig(
        variant, (1, 1, 1), (1, 1, 1), execution_variant=variant)
    spec = KernelSpec(
        kind,
        "nvidia_a800",
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes=attributes,
    )

    assert spec.shape["element_count"] == expected
    assert spec.shape["elements"] == expected


@pytest.mark.parametrize(
    "key_present,expected",
    ((True, 80), (False, 64)),
)
def test_typed_rope_derives_active_rotary_elements(key_present, expected):
    launch = LaunchConfig(
        "rope", (2, 1, 1), (64, 1, 1), execution_variant="rope")
    spec = KernelSpec(
        KernelKind.VECTOR,
        "nvidia_a800",
        {
            "seq_len": 2,
            "num_query_heads": 4,
            "num_kv_heads": 1,
            "head_dim": 16,
            "rotary_dim": 8,
        },
        dtype="bf16",
        variant="rope",
        launch=launch,
        attributes={"is_neox": True, "key_present": key_present},
    )

    assert spec.shape["element_count"] == expected
    assert spec.shape["elements"] == expected


@pytest.mark.parametrize(
    "rotary_dim,message",
    ((17, "cannot exceed head_dim"), (7, "must be even")),
)
def test_typed_rope_rejects_invalid_rotary_extent(rotary_dim, message):
    launch = LaunchConfig(
        "rope", (2, 1, 1), (64, 1, 1), execution_variant="rope")

    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {
                "seq_len": 2,
                "num_query_heads": 4,
                "num_kv_heads": 1,
                "head_dim": 16,
                "rotary_dim": rotary_dim,
            },
            dtype="bf16",
            variant="rope",
            launch=launch,
            attributes={"is_neox": True, "key_present": True},
        )


def test_typed_rope_requires_explicit_boolean_variant_facts():
    launch = LaunchConfig(
        "rope", (2, 1, 1), (64, 1, 1), execution_variant="rope")
    shape = {
        "seq_len": 2,
        "num_query_heads": 4,
        "num_kv_heads": 1,
        "head_dim": 16,
        "rotary_dim": 8,
    }

    with pytest.raises(ValueError, match="missing attributes.*key_present"):
        KernelSpec(
            KernelKind.VECTOR, "nvidia_a800", shape, dtype="bf16",
            variant="rope", launch=launch, attributes={"is_neox": True})
    with pytest.raises(ValueError, match="key_present must be boolean"):
        KernelSpec(
            KernelKind.VECTOR, "nvidia_a800", shape, dtype="bf16",
            variant="rope", launch=launch,
            attributes={"is_neox": True, "key_present": 1})


def test_vector_structured_shape_rejects_conflicting_element_count():
    launch = LaunchConfig(
        "activation", (1, 1, 1), (1, 1, 1),
        execution_variant="activation")

    with pytest.raises(ValueError, match="disagrees with structured"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"tok": 4, "intermediate": 32, "element_count": 127},
            dtype="bf16",
            variant="activation",
            launch=launch,
        )


def test_vector_shape_rejects_partial_structured_fields():
    launch = LaunchConfig(
        "rotary", (1, 1, 1), (1, 1, 1), execution_variant="rotary")

    with pytest.raises(ValueError, match="partial structured shape"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"seq_len": 4, "element_count": 512},
            dtype="bf16",
            variant="rotary",
            launch=launch,
        )


def _flash_mla_metadata_launch():
    return LaunchConfig(
        "observed_flash_mla_metadata:flash_mla_metadata",
        (1, 1, 1),
        (32, 1, 1),
        execution_variant="flash_mla_metadata",
    )


def test_flash_mla_metadata_accepts_complete_structured_shape():
    spec = KernelSpec(
        KernelKind.VECTOR,
        "metax_c500",
        {
            "batch": 8,
            "kv_block_count": 256,
            "scheduler_partitions": 104,
            "scheduler_metadata_ints": 832,
            "split_metadata_ints": 9,
            "scheduler_segment_count": 88,
            "scheduler_worker_run_count": 88,
        },
        dtype="int32",
        variant="flash_mla_metadata",
        launch=_flash_mla_metadata_launch(),
    )

    assert spec.shape["dtype_bytes"] == 4


@pytest.mark.parametrize(
    ("shape", "message"),
    (
        (
            {
                "batch": 8,
                "kv_block_count": 256,
                "scheduler_partitions": 104,
                "scheduler_metadata_ints": 832,
                "scheduler_segment_count": 88,
                "scheduler_worker_run_count": 88,
            },
            "missing shape fields.*split_metadata_ints",
        ),
        (
            {
                "batch": 8,
                "kv_block_count": 256,
                "scheduler_partitions": 104,
                "scheduler_metadata_ints": 831,
                "split_metadata_ints": 9,
                "scheduler_segment_count": 88,
                "scheduler_worker_run_count": 88,
            },
            "scheduler_metadata_ints must equal",
        ),
        (
            {
                "batch": 8,
                "kv_block_count": 256,
                "scheduler_partitions": 104,
                "scheduler_metadata_ints": 832,
                "split_metadata_ints": 8,
                "scheduler_segment_count": 88,
                "scheduler_worker_run_count": 88,
            },
            "split_metadata_ints must equal",
        ),
        (
            {
                "batch": 8,
                "kv_block_count": 7,
                "scheduler_partitions": 104,
                "scheduler_metadata_ints": 832,
                "split_metadata_ints": 9,
                "scheduler_segment_count": 88,
                "scheduler_worker_run_count": 88,
            },
            "kv_block_count cannot be smaller",
        ),
        (
            {
                "batch": 8,
                "kv_block_count": 256,
                "scheduler_partitions": 104,
                "scheduler_metadata_ints": 832,
                "split_metadata_ints": 9,
                "scheduler_segment_count": 88,
                "scheduler_worker_run_count": 88,
                "unexpected": 1,
            },
            "unknown shape fields.*unexpected",
        ),
    ),
)
def test_flash_mla_metadata_rejects_malformed_structured_shape(
        shape, message):
    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.VECTOR,
            "metax_c500",
            shape,
            dtype="int32",
            variant="flash_mla_metadata",
            launch=_flash_mla_metadata_launch(),
        )


@pytest.mark.parametrize("variant", ["moe_sum", "moe_reduce"])
@pytest.mark.parametrize("work_fraction", [0, -0.5, 1.1, math.nan, True])
def test_moe_reduction_rejects_invalid_work_fraction(
        variant, work_fraction):
    launch = LaunchConfig(
        variant, (1, 1, 1), (256, 1, 1),
        execution_variant=variant)

    error = TypeError if work_fraction is True else ValueError
    with pytest.raises(error, match="work_fraction"):
        KernelSpec(
            KernelKind.VECTOR,
            "metax_c500",
            {
                "tokens": 1,
                "experts": 1,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            dtype="bf16",
            variant=variant,
            launch=launch,
            attributes={"work_fraction": work_fraction},
        )


@pytest.mark.parametrize(
    ("attributes", "error", "message"),
    (
        ({}, ValueError, "missing attributes.*fill_role"),
        ({"fill_role": "unknown"}, ValueError, "fill_role must be one of"),
        ({"fill_role": 1}, TypeError, "fill_role must be a non-empty string"),
    ),
)
def test_moe_fill_requires_typed_output_role(attributes, error, message):
    launch = LaunchConfig(
        "moe_fill", (1, 1, 1), (256, 1, 1),
        execution_variant="moe_fill")

    with pytest.raises(error, match=message):
        KernelSpec(
            KernelKind.VECTOR,
            "metax_c500",
            {
                "tokens": 1,
                "experts": 8,
                "topk": 2,
                "hidden": 4096,
                "intermediate": 14336,
            },
            dtype="bf16",
            variant="moe_fill",
            launch=launch,
            attributes=attributes,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_fused_moe_spec_rejects_non_finite_expert_windows(value):
    launch = LaunchConfig(
        "grouped_gemm",
        (1, 1, 1),
        (1, 1, 1),
        tiling=TilingSpec(tile_shape={"m": 1, "n": 1, "k": 1}),
        parameters={
            "group_size_m": 1,
            "padded_rows": 1,
            "mte2_tile_bytes": 1,
            "weight_tile_rows": 1,
            "operand_replay_a": 1,
            "operand_replay_b": 1,
        },
        execution_variant="w1",
    )
    with pytest.raises(ValueError, match="expert_weight_tile_rows"):
        KernelSpec(
            KernelKind.FUSED_MOE,
            "nvidia_a800",
            {
                "tokens": 1,
                "experts": 1,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            dtype="bf16",
            variant="w1",
            launch=launch,
            attributes={
                "tokens_per_expert": (1,),
                "local_tokens_per_expert": (1,),
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": (0,),
                "expert_weight_tile_rows": (value,),
            },
        )


def test_fused_moe_spec_accepts_zero_window_only_for_inactive_expert():
    launch = LaunchConfig(
        "grouped_gemm",
        (1, 1, 1),
        (1, 1, 1),
        tiling=TilingSpec(tile_shape={"m": 1, "n": 1, "k": 1}),
        parameters={
            "group_size_m": 1,
            "padded_rows": 1,
            "mte2_tile_bytes": 1,
            "weight_tile_rows": 1,
            "operand_replay_a": 1,
            "operand_replay_b": 1,
        },
        execution_variant="w1",
    )
    spec = KernelSpec(
        KernelKind.FUSED_MOE,
        "nvidia_a800",
        {
            "tokens": 1,
            "experts": 2,
            "topk": 1,
            "hidden": 1,
            "intermediate": 1,
        },
        dtype="bf16",
        variant="w1",
        launch=launch,
        attributes={
            "tokens_per_expert": (1, 0),
            "local_tokens_per_expert": (1, 0),
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": (0, 1),
            "expert_weight_tile_rows": (1.0, 0.0),
        },
    )
    assert spec.attributes["expert_weight_tile_rows"] == (1.0, 0.0)


def _ep_fused_moe_spec_attributes(**overrides):
    attributes = {
        "tokens_per_expert": (1, 1),
        "local_tokens_per_expert": (1, 0),
        "expert_parallel_size": 2,
        "expert_rank": 0,
        "owned_experts": (0,),
        "expert_weight_tile_rows": None,
    }
    attributes.update(overrides)
    return attributes


def _ep_fused_moe_launch():
    return LaunchConfig(
        "grouped_gemm",
        (2, 1, 1),
        (256, 1, 1),
        tiling=TilingSpec(tile_shape={"m": 1, "n": 1, "k": 1}),
        parameters={
            "group_size_m": 1,
            "padded_rows": 2,
            "mte2_tile_bytes": 0,
            "weight_tile_rows": 0,
            "operand_replay_a": 1,
            "operand_replay_b": 1,
            "dynamic_shared_memory_bytes": 49152,
        },
        execution_variant="w1",
    )


def test_fused_moe_spec_preserves_global_and_local_ep_routing():
    spec = KernelSpec(
        KernelKind.FUSED_MOE,
        "nvidia_a800",
        {
            "tokens": 2,
            "experts": 2,
            "topk": 1,
            "hidden": 1,
            "intermediate": 1,
        },
        dtype="bf16",
        variant="w1",
        launch=_ep_fused_moe_launch(),
        attributes=_ep_fused_moe_spec_attributes(),
    )

    assert spec.attributes["tokens_per_expert"] == (1, 1)
    assert spec.attributes["local_tokens_per_expert"] == (1, 0)
    assert spec.attributes["owned_experts"] == (0,)


def test_fused_moe_spec_rejects_uneven_expert_parallel_placement():
    with pytest.raises(ValueError, match="must divide experts"):
        KernelSpec(
            KernelKind.FUSED_MOE,
            "nvidia_a800",
            {
                "tokens": 3,
                "experts": 3,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            dtype="bf16",
            variant="w1",
            launch=_ep_fused_moe_launch(),
            attributes={
                "tokens_per_expert": (1, 1, 1),
                "local_tokens_per_expert": (1, 1, 0),
                "expert_parallel_size": 2,
                "expert_rank": 0,
                "owned_experts": (0, 1),
                "expert_weight_tile_rows": None,
            },
        )


def test_fused_moe_spec_accepts_zero_local_rows_with_global_launch():
    spec = KernelSpec(
        KernelKind.FUSED_MOE,
        "nvidia_a800",
        {
            "tokens": 1,
            "experts": 2,
            "topk": 1,
            "hidden": 1,
            "intermediate": 1,
        },
        dtype="bf16",
        variant="w1",
        launch=_ep_fused_moe_launch(),
        attributes=_ep_fused_moe_spec_attributes(
            tokens_per_expert=(1, 0),
            local_tokens_per_expert=(0, 0),
            expert_rank=1,
            owned_experts=(1,),
        ),
    )

    assert sum(spec.attributes["tokens_per_expert"]) == 1
    assert sum(spec.attributes["local_tokens_per_expert"]) == 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"expert_rank": 2}, "expert_rank"),
        ({"owned_experts": (1,)}, "owned_experts"),
        ({"local_tokens_per_expert": (0, 1)},
         "local_tokens_per_expert"),
        ({"expert_parallel_size": 3}, "expert_parallel_size"),
    ),
)
def test_fused_moe_spec_rejects_forged_ep_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.FUSED_MOE,
            "nvidia_a800",
            {
                "tokens": 2,
                "experts": 2,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            dtype="bf16",
            variant="w1",
            launch=_ep_fused_moe_launch(),
            attributes=_ep_fused_moe_spec_attributes(**overrides),
        )


@pytest.mark.parametrize(
    "attributes, message",
    [
        (
            {"reads": True, "writes": 1, "operation": "copy"},
            "reads must be an integer",
        ),
    ],
)
def test_vector_spec_rejects_boolean_numeric_attributes(attributes, message):
    launch = LaunchConfig(
        "elementwise", (1, 1, 1), (1, 1, 1),
        execution_variant="elementwise",
    )
    with pytest.raises(TypeError, match=message):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"element_count": 1},
            dtype="bf16",
            variant="elementwise",
            launch=launch,
            attributes=attributes,
        )


def test_vector_spec_rejects_operation_io_mismatch():
    launch = LaunchConfig(
        "elementwise", (1, 1, 1), (1, 1, 1),
        execution_variant="elementwise",
    )
    with pytest.raises(ValueError, match="reads/writes disagree"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"element_count": 1},
            dtype="bf16",
            variant="elementwise",
            launch=launch,
            attributes={"reads": 1, "writes": 1, "operation": "add"},
        )


def test_flash_attention_spec_rejects_query_longer_than_request_kv():
    launch = LaunchConfig(
        "prefill", (1, 1, 1), (1, 1, 1),
        parameters={
            "unpadded_flops": 0.0,
            "mma_smem_bytes_per_flop": 0.0,
            "split_scratch_dram_bytes_per_element": 0.0,
            "page_table_bytes_per_kv_panel": 0.0,
        },
        execution_variant="prefill",
    )
    with pytest.raises(ValueError, match="q_lens <= kv_lens"):
        KernelSpec(
            KernelKind.FLASH_ATTN,
            "nvidia_a800",
            {
                "batch": 2,
                "q_len": 8,
                "kv_len": 8,
                "head_dim": 128,
                "num_query_heads": 32,
                "num_kv_heads": 8,
            },
            dtype="bf16",
            variant="prefill",
            launch=launch,
            attributes={
                "causal": True,
                "kv_layout": "paged_varlen",
                "kv_dist": "skew",
                "fa_ksteps": None,
                "fa_eff_mfrac": 1.0,
                "q_lens": (8, 8),
                "kv_lens": (4, 8),
            },
        )


@pytest.mark.parametrize(
    ("splits", "scratch_dram_width", "page_table_bytes", "message"),
    (
        (2, 2.0, 0.0, "split scratch DRAM width"),
        (1, 4.0, 0.0, "non-split FLASH_ATTN"),
        (2, 0.0, 2.0, "page-table access"),
        (1, 0.0, 4.0, "non-split FLASH_ATTN"),
    ),
)
def test_flash_attention_spec_rejects_invalid_memory_width(
        splits, scratch_dram_width, page_table_bytes, message):
    launch = LaunchConfig(
        "splitkv",
        (1, splits, 1),
        (128, 1, 1),
        tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
        parameters={
            "unpadded_flops": 1.0,
            "mma_smem_bytes_per_flop": 0.0,
            "split_scratch_dram_bytes_per_element": scratch_dram_width,
            "page_table_bytes_per_kv_panel": page_table_bytes,
        },
        execution_variant="decode",
    )
    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.FLASH_ATTN,
            "metax_c500",
            {
                "batch": 1,
                "q_len": 1,
                "kv_len": 1024,
                "head_dim": 128,
                "num_query_heads": 32,
                "num_kv_heads": 8,
            },
            dtype="bf16",
            variant="decode",
            launch=launch,
            attributes={
                "causal": False,
                "kv_layout": "contiguous",
                "kv_dist": "skew",
                "fa_ksteps": None,
                "fa_eff_mfrac": 1.0,
                "q_lens": (1,),
                "kv_lens": (1024,),
            },
        )


@pytest.mark.parametrize(
    "kind,variant,shape,attributes,missing",
    [
        (
            KernelKind.FLASH_ATTN,
            "decode",
            {
                "batch": 1,
                "q_len": 1,
                "kv_len": 1,
                "head_dim": 1,
                "num_query_heads": 1,
                "num_kv_heads": 1,
            },
            {
                "causal": True,
                "kv_dist": None,
                "fa_ksteps": None,
                "fa_eff_mfrac": 1.0,
            },
            "kv_layout",
        ),
        (
            KernelKind.FUSED_MOE,
            "w1",
            {
                "tokens": 1,
                "experts": 1,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            {
                "tokens_per_expert": (1,),
                "local_tokens_per_expert": (1,),
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": (0,),
            },
            "expert_weight_tile_rows",
        ),
    ],
)
def test_nullable_compute_attributes_must_be_explicit(
        kind, variant, shape, attributes, missing):
    if kind is KernelKind.FUSED_MOE:
        tiling = TilingSpec(tile_shape={"m": 1, "n": 1, "k": 1})
        parameters = {
            "group_size_m": 1,
            "padded_rows": 1,
            "mte2_tile_bytes": 1,
            "weight_tile_rows": 1,
            "operand_replay_a": 1,
            "operand_replay_b": 1,
        }
    else:
        tiling = None
        parameters = (
            {
                "unpadded_flops": 0.0,
                "mma_smem_bytes_per_flop": 0.0,
                "split_scratch_dram_bytes_per_element": 0.0,
                "page_table_bytes_per_kv_panel": 0.0,
            }
            if kind is KernelKind.FLASH_ATTN else {})
    launch = LaunchConfig(
        variant,
        (1, 1, 1),
        (1, 1, 1),
        tiling=tiling,
        parameters=parameters,
        execution_variant=variant,
    )

    with pytest.raises(ValueError, match=missing):
        KernelSpec(
            kind,
            "nvidia_a800",
            shape,
            dtype="bf16",
            variant=variant,
            launch=launch,
            attributes=attributes,
        )


@pytest.mark.parametrize("value", [True, "1"])
def test_launch_parameters_reject_non_numeric_types(value):
    with pytest.raises(TypeError, match="launch parameters must be numeric"):
        LaunchConfig(
            "invalid_parameter", (1, 1, 1), (1, 1, 1),
            parameters={"value": value},
        )


@pytest.mark.parametrize(
    "parameters,message",
    [
        ({"input_reread_count": 1.5},
         "input_reread_count must be a positive integer"),
        ({"input_reread_count": 1.0, "lowered_sfu_operations": 1.5},
         "lowered_sfu_operations must be a positive integer"),
        ({
            "lowered_vector_input_operations": 1.5,
            "lowered_vector_output_operations": 1.0,
            "norm_weight_l2_copies": 1.0,
        }, "lowered_vector_input_operations must be a positive integer"),
        ({"lowered_vector_operations": 1.5},
         "launch parameter schema mismatch"),
        ({"rstd_output_bytes": 1.5},
         "rstd_output_bytes must be a positive integer"),
        ({"cann_rmsnorm_rstd_transfer_bytes": 1.5},
         "cann_rmsnorm_rstd_transfer_bytes must be a positive integer"),
        ({"lowered_sfu_operations": 16.0},
         "launch parameter schema mismatch"),
    ],
)
def test_vector_launch_rejects_invalid_source_work_contract(
        parameters, message):
    launch = LaunchConfig(
        "rmsnorm", (1, 1, 1), (8, 1, 1), parameters=parameters,
        execution_variant="rmsnorm")

    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"rows": 1, "hidden": 8},
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_vector_launch_rejects_cann_rmsnorm_transfer_on_other_hardware():
    launch = LaunchConfig(
        "rmsnorm", (1, 1, 1), (8, 1, 1),
        parameters={"cann_rmsnorm_rstd_transfer_bytes": 32.0},
        execution_variant="rmsnorm")

    with pytest.raises(ValueError, match="requires ascend_910b BF16 RMSNorm"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"rows": 1, "hidden": 8},
            dtype="bf16",
            variant="rmsnorm",
            launch=launch,
        )


def test_elementwise_launch_rejects_fractional_lowered_vector_work():
    launch = LaunchConfig(
        "elementwise", (1, 1, 1), (1, 1, 1),
        parameters={"lowered_vector_operations": 1.5},
        execution_variant="elementwise")

    with pytest.raises(
            ValueError,
            match="lowered_vector_operations must be a positive integer"):
        KernelSpec(
            KernelKind.VECTOR,
            "ascend_910b",
            {"element_count": 8},
            dtype="bf16",
            variant="elementwise",
            launch=launch,
            attributes={"reads": 2, "writes": 1, "operation": "add"},
        )


def test_elementwise_lowered_vector_work_requires_source_closed_operation():
    launch = LaunchConfig(
        "elementwise", (1, 1, 1), (1, 1, 1),
        parameters={"lowered_vector_operations": 8.0},
        execution_variant="elementwise")

    with pytest.raises(
            ValueError,
            match="requires the source-closed add or mul operation"):
        KernelSpec(
            KernelKind.VECTOR,
            "ascend_910b",
            {"element_count": 8},
            dtype="bf16",
            variant="elementwise",
            launch=launch,
            attributes={"reads": 1, "writes": 1, "operation": "copy"},
        )


@pytest.mark.parametrize(
    ("tiling_key", "message"),
    (
        (200_000_000.5, "must be one of the supported CANN 8.5.1"),
        (201_000_000.0, "must be one of the supported CANN 8.5.1"),
    ),
)
def test_vector_kv_cache_rejects_unsupported_cann_tiling_key(
        tiling_key, message):
    launch = LaunchConfig(
        "kv_cache", (1, 1, 1), (1, 1, 1),
        parameters={"cann_tiling_key": tiling_key},
        execution_variant="paged_kv_cache")

    with pytest.raises(ValueError, match=message):
        KernelSpec(
            KernelKind.KV_CACHE,
            "ascend_910b",
            {
                "element_count": 512,
                "seq_len": 1,
                "num_kv_heads": 4,
                "head_dim": 128,
            },
            dtype="bf16",
            variant="paged_kv_cache",
            launch=launch,
            attributes=PAGED_KV_CACHE_ATTRIBUTES,
        )


@pytest.mark.parametrize("tiling_key", (1000.0, 1004.0, 1001.5))
def test_vector_moe_compute_expert_tokens_rejects_invalid_cann_tiling_key(
        tiling_key):
    launch = LaunchConfig(
        "aclnn_moe_compute_expert_tokens", (48, 1, 1), (1, 1, 1),
        parameters={
            "cann_tiling_key": tiling_key,
            "cann_moe_compute_expert_tokens_mix_aiv_1_0": 1.0,
        },
        execution_variant="moe_compute_expert_tokens")

    with pytest.raises(
            ValueError,
            match="supported CANN 8.5.1 MoeComputeExpertTokens keys"):
        KernelSpec(
            KernelKind.VECTOR,
            "ascend_910b",
            {
                "tokens": 1,
                "experts": 8,
                "topk": 2,
                "hidden": 4096,
                "intermediate": 14336,
            },
            dtype="bf16",
            variant="moe_compute_expert_tokens",
            launch=launch,
            attributes={"tokens_per_expert": (2,) + (0,) * 7},
        )


def test_vector_moe_compute_expert_tokens_rejects_key_shape_mismatch():
    launch = LaunchConfig(
        "aclnn_moe_compute_expert_tokens", (48, 1, 1), (1, 1, 1),
        parameters={
            "cann_tiling_key": 1001.0,
            "cann_moe_compute_expert_tokens_mix_aiv_1_0": 1.0,
        },
        execution_variant="moe_compute_expert_tokens")

    with pytest.raises(ValueError, match="disagrees with the source selector"):
        KernelSpec(
            KernelKind.VECTOR,
            "ascend_910b",
            {
                "tokens": 12_001,
                "experts": 256,
                "topk": 1,
                "hidden": 4096,
                "intermediate": 14336,
            },
            dtype="bf16",
            variant="moe_compute_expert_tokens",
            launch=launch,
            attributes={
                "tokens_per_expert": (12_001,) + (0,) * 255,
            },
        )


def test_vector_moe_compute_expert_tokens_requires_cann_tiling_key():
    launch = LaunchConfig(
        "aclnn_moe_compute_expert_tokens", (48, 1, 1), (1, 1, 1),
        parameters={}, execution_variant="moe_compute_expert_tokens")

    with pytest.raises(ValueError, match="launch parameter schema mismatch"):
        KernelSpec(
            KernelKind.VECTOR,
            "ascend_910b",
            {
                "tokens": 1,
                "experts": 8,
                "topk": 2,
                "hidden": 4096,
                "intermediate": 14336,
            },
            dtype="bf16",
            variant="moe_compute_expert_tokens",
            launch=launch,
            attributes={"tokens_per_expert": (2,) + (0,) * 7},
        )


def test_vector_rotary_rejects_fractional_cache_path_bytes():
    launch = LaunchConfig(
        "rotary", (1, 1, 1), (8, 1, 1),
        parameters={"rotary_cache_path_bytes": 1.5},
        execution_variant="rotary")

    with pytest.raises(
            ValueError,
            match="rotary_cache_path_bytes must be a positive integer"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"seq_len": 1, "num_query_heads": 1,
             "num_kv_heads": 1, "head_dim": 8},
            dtype="bf16",
            variant="rotary",
            launch=launch,
        )


def test_vector_rotary_rejects_fractional_position_path_bytes():
    launch = LaunchConfig(
        "rotary", (1, 1, 1), (8, 1, 1),
        parameters={
            "rotary_cache_path_bytes": 16.0,
            "rotary_position_path_bytes": 1.5,
        },
        execution_variant="rotary")

    with pytest.raises(
            ValueError,
            match="rotary_position_path_bytes must be a positive integer"):
        KernelSpec(
            KernelKind.VECTOR,
            "nvidia_a800",
            {"seq_len": 1, "num_query_heads": 1,
             "num_kv_heads": 1, "head_dim": 8},
            dtype="bf16",
            variant="rotary",
            launch=launch,
        )


def test_ascend_atb_bf16_rope_rejects_wrong_source_slice():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {
        "seq_len": 64,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "rotary",
        {"launch_policy": "injected_launch"}, hardware=hardware)
    parameters = dict(launch.parameters)
    parameters["atb_rope_slice_elements"] = 4096.0
    invalid = dataclasses.replace(launch, parameters=parameters)

    with pytest.raises(
            ValueError, match="slice disagrees with the selected source"):
        KernelSpec(
            KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
            variant="rotary", launch=invalid, attributes={})


def test_vector_kv_cache_rejects_fractional_slot_l1_bytes():
    launch = LaunchConfig(
        "kv_cache", (1, 1, 1), (8, 1, 1),
        parameters={"kv_cache_slot_l1_bytes": 1.5},
        execution_variant="paged_kv_cache")

    with pytest.raises(
            ValueError,
            match="kv_cache_slot_l1_bytes must be a positive integer"):
        KernelSpec(
            KernelKind.KV_CACHE,
            "nvidia_a800",
            {"seq_len": 1, "num_kv_heads": 1, "head_dim": 8},
            dtype="bf16",
            variant="paged_kv_cache",
            launch=launch,
            attributes=PAGED_KV_CACHE_ATTRIBUTES,
        )


def test_vector_kv_cache_rejects_fractional_source_residency():
    launch = LaunchConfig(
        "kv_cache", (1, 1, 1), (512, 1, 1),
        parameters={
            "source_bound_resident_ctas_per_unit": 1.5,
            "source_bound_l2_read_warps_per_cta": 2.0,
            "source_bound_memory_dependency_task": 1.0,
        },
        execution_variant="paged_kv_cache")

    with pytest.raises(
            ValueError,
            match="source_bound_resident_ctas_per_unit must be a positive"):
        KernelSpec(
            KernelKind.KV_CACHE,
            "nvidia_a800",
            {"seq_len": 1, "num_kv_heads": 4, "head_dim": 128},
            dtype="bf16",
            variant="paged_kv_cache",
            launch=launch,
            attributes=PAGED_KV_CACHE_ATTRIBUTES,
        )


@pytest.mark.parametrize(
    "name",
    ["group_size_m", "padded_rows", "operand_replay_a", "operand_replay_b"],
)
def test_fused_moe_launch_rejects_non_integral_counts(name):
    parameters = {
        "group_size_m": 1,
        "padded_rows": 1,
        "mte2_tile_bytes": 1,
        "weight_tile_rows": 1,
        "operand_replay_a": 1,
        "operand_replay_b": 1,
    }
    parameters[name] = 1.5
    launch = LaunchConfig(
        "grouped_gemm",
        (1, 1, 1),
        (1, 1, 1),
        tiling=TilingSpec(tile_shape={"m": 1, "n": 1, "k": 1}),
        parameters=parameters,
        execution_variant="w1",
    )
    with pytest.raises(ValueError, match=rf"{name} must be integral"):
        KernelSpec(
            KernelKind.FUSED_MOE,
            "nvidia_a800",
            {
                "tokens": 1,
                "experts": 1,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            dtype="bf16",
            variant="w1",
            launch=launch,
            attributes={
                "tokens_per_expert": (1,),
                "local_tokens_per_expert": (1,),
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": (0,),
                "expert_weight_tile_rows": None,
            },
        )


@pytest.mark.parametrize("value", (0, 49152.5))
def test_fused_moe_launch_rejects_invalid_dynamic_shared_memory(value):
    parameters = {
        "group_size_m": 1,
        "padded_rows": 1,
        "mte2_tile_bytes": 1,
        "weight_tile_rows": 1,
        "operand_replay_a": 1,
        "operand_replay_b": 1,
        "dynamic_shared_memory_bytes": value,
    }
    launch = LaunchConfig(
        "grouped_gemm",
        (1, 1, 1),
        (1, 1, 1),
        tiling=TilingSpec(tile_shape={"m": 1, "n": 1, "k": 1}),
        parameters=parameters,
        execution_variant="w1",
    )

    with pytest.raises(
            ValueError,
            match="dynamic_shared_memory_bytes must be a positive integer"):
        KernelSpec(
            KernelKind.FUSED_MOE,
            "nvidia_a800",
            {
                "tokens": 1,
                "experts": 1,
                "topk": 1,
                "hidden": 1,
                "intermediate": 1,
            },
            dtype="bf16",
            variant="w1",
            launch=launch,
            attributes={
                "tokens_per_expert": (1,),
                "local_tokens_per_expert": (1,),
                "expert_parallel_size": 1,
                "expert_rank": 0,
                "owned_experts": (0,),
                "expert_weight_tile_rows": None,
            },
        )


@pytest.mark.parametrize(
    "name", ["compulsory_traffic", "aiv_mixed", "n_major_traffic"])
def test_gemm_launch_rejects_non_binary_flags(name):
    parameters = {
        "compulsory_traffic": 0,
        "weight_tile_rows": 1,
        "mte2_tile_bytes": 1,
        "aiv_mixed": 0,
        "n_major_traffic": 0,
        "split_k_partial_bytes_per_element": 0,
        "split_k_partial_dram_bytes_per_element": 0,
    }
    parameters[name] = 2
    launch = LaunchConfig(
        "gemm", (1, 1, 1), (1, 1, 1),
        parameters=parameters, execution_variant="gemm")
    with pytest.raises(ValueError, match=rf"{name} must be 0 or 1"):
        KernelSpec(
            KernelKind.GEMM,
            "nvidia_a800",
            {"b": 1, "m": 1, "n": 1, "k": 1},
            dtype="bf16",
            variant="gemm",
            launch=launch,
        )


def test_gemm_launch_rejects_fractional_partial_dram_residency():
    launch = LaunchConfig(
        "gemm", (1, 1, 2), (128, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": 1, "n": 1, "k": 1}, split_k=2),
        parameters={
            "compulsory_traffic": 0,
            "weight_tile_rows": 0,
            "mte2_tile_bytes": 0,
            "aiv_mixed": 0,
            "n_major_traffic": 0,
            "split_k_partial_bytes_per_element": 4,
            "split_k_partial_dram_bytes_per_element": 2,
        },
        execution_variant="gemm")

    with pytest.raises(ValueError, match="zero or equal"):
        KernelSpec(
            KernelKind.GEMM,
            "nvidia_a800",
            {"b": 1, "m": 1, "n": 1, "k": 2},
            dtype="bf16",
            variant="gemm",
            launch=launch,
        )


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: KernelKind.parse(1), "kernel kind must be"),
        (lambda: KernelSpec(
            KernelKind.GEMM, 1, {"b": 1, "m": 1, "n": 1, "k": 1}),
         "hardware_id must be"),
        (lambda: TilingSpec(tile_shape={1: 16}),
         "tile dimension keys must be"),
        (lambda: LaunchConfig(
            "gemm", (1, 1, 1), (1, 1, 1), parameters={1: 1}),
         "launch parameter keys must be"),
    ],
)
def test_kernel_contract_does_not_coerce_invalid_identifiers(
        constructor, message):
    with pytest.raises(TypeError, match=message):
        constructor()


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: TilingSpec(tile_shape=[("m", 16)]),
         "tile_shape must be a mapping"),
        (lambda: LaunchConfig(
            "gemm", (1, 1, 1), (1, 1, 1), parameters=[("x", 1)]),
         "parameters must be a mapping"),
        (lambda: KernelSpec(KernelKind.GEMM, "nvidia_a800", []),
         "shape must be a mapping"),
        (lambda: KernelPrediction(
            1.0, [], 0.0, "dram", {"dram": 1.0}),
         "t_per_group must be a mapping"),
    ],
)
def test_kernel_contract_rejects_mapping_lookalikes(constructor, message):
    with pytest.raises(TypeError, match=message):
        constructor()

def test_legacy_kv_cache_update_kind_parses_to_canonical_kv_cache():
    assert KernelKind.parse("KV_CACHE_UPDATE") is KernelKind.KV_CACHE
    assert KernelKind.parse("KVCACHEUPDATE") is KernelKind.KV_CACHE
    assert KernelKind.parse("KV_CACHE") is KernelKind.KV_CACHE


def test_h200_tp8_small_all_reduce_uses_measured_custom_service():
    # Pinned vLLM routes small NVLink all-reduces through its IPC custom
    # kernel; the published graph-node service replaces the bandwidth-only
    # term below max_payload_bytes and leaves larger (NCCL) payloads alone.
    from src.kernel.hardware_spec import HWSpec
    from src.kernel.kernel_base import predict

    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    ranks = tuple(range(8))

    def all_reduce_ns(payload_bytes, participants=ranks):
        spec = KernelSpec(
            KernelKind.COMMUNICATION, "nvidia_h200",
            transfer=_all_reduce_transfer(
                bytes=payload_bytes, participants=participants,
                ring_order=participants))
        return predict(spec, hardware, "roofline").t_total_s * 1e9

    one_token = 7168 * 2
    assert all_reduce_ns(one_token) == pytest.approx(
        hardware.custom_all_reduce_service_ns(8, one_token))
    assert all_reduce_ns(64 * one_token) == pytest.approx(
        hardware.custom_all_reduce_service_ns(8, 64 * one_token))
    assert all_reduce_ns(64 * one_token) > all_reduce_ns(one_token) > 1000.0
    large = hardware.custom_all_reduce_service[
        "max_payload_bytes"] * 4
    bandwidth_ns = (
        2.0 * 7 / 8 * large
        / hardware.interconnect["nvlink_bw_bytes_s"] * 1e9)
    assert all_reduce_ns(large) == pytest.approx(bandwidth_ns)
    with pytest.raises(KeyError, match="no measurement for 6 participants"):
        all_reduce_ns(one_token, participants=tuple(range(6)))
