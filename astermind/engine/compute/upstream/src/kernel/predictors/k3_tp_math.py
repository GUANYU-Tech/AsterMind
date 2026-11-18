"""K3 full-expert channel shards; separate physical256 padded source program."""
PROGRAM='k3_tp_sharded_marlin_v1'
PADDED_PROGRAM='k3_tp_padded_marlin_v1'
def validate(shape,a):
    tp=a.get('tensor_parallel_size');lane=a.get('tensor_rank');padded=a.get('mathematical_program')==PADDED_PROGRAM
    if (type(tp) is not int or tp not in ((16,32) if padded else (2,4,8)) or type(lane) is not int or not 0<=lane<tp
        or (shape['experts'],shape['topk'],shape['hidden'],shape['intermediate'])!=(896,16,3584,256 if padded else 3072//tp)
        or a.get('logical_intermediate')!=3072//tp or a.get('mathematical_program') not in (PROGRAM,PADDED_PROGRAM)
        or a.get('expert_parallel_size')!=1 or a.get('expert_rank')!=0
        or a.get('expert_weight_tile_rows') is not None or a.get('routed_weight_dtype')!='mxfp4'
        or a.get('weight_group_size')!=32 or a.get('swiglu_limit') is not None
        or a.get('activation_beta')!=4.0 or a.get('activation_linear_beta')!=25.0 or a.get('routed_scaling_factor')!=1.0):
        raise ValueError('K3 TP math requires exact H200 source shard geometry and SiTU')
    counts=a['tokens_per_expert'];local=a['local_tokens_per_expert'];owned=a['owned_experts']
    if (len(counts)!=896 or any(type(x) is not int or x<0 or x>shape['tokens'] for x in counts)
        or sum(counts)!=shape['tokens']*16 or tuple(local)!=tuple(counts)
        or any(type(x) is not int for x in local) or tuple(owned)!=tuple(range(896))
        or any(type(x) is not int for x in owned)):
        raise ValueError('K3 TP keeps all experts and identical full routing on every lane')

def logical_work(kernel):
    """Standard-format W4A16 Marlin source roles with K3's exact SiTU.

    This is a conditional source program on supplied routing, not an auto
    backend selection or a GPU launch. Counts are live mathematical work;
    alignment, padded MMA, conversion, scratch and reduction implementation
    remain unbound. Repeated latent-role reads are not observed DRAM.
    """
    validate(kernel.shape,kernel.attributes)
    tokens, experts, topk, hidden, intermediate = (kernel.shape[key] for key in (
        "tokens", "experts", "topk", "hidden", "intermediate"))
    routed = tokens * topk
    active = sum(count > 0 for count in kernel.attributes["tokens_per_expert"])
    operands = []

    def operand(role, dtype, count, access):
        size = (count // 2 if dtype == "e2m1_packed" else
                {"bf16": 2, "fp32": 4, "int32": 4, "ue8m0": 1}[dtype] * count)
        operands.append({"role": role, "dtype": dtype, "elements": count,
                         "bytes": size, "access": access})

    operand("latent_per_route", "bf16", routed * hidden, "read")
    operand("topk_ids", "int32", routed, "read")
    operand("topk_weights", "fp32", routed, "read")
    operand("w13_active_weights", "e2m1_packed", active * 2 * hidden * intermediate, "read")
    operand("w13_active_scales", "ue8m0", active * 2 * intermediate * (hidden // 32), "read")
    operand("w13_output", "bf16", routed * 2 * intermediate, "write")
    operand("situ_input", "bf16", routed * 2 * intermediate, "read")
    operand("situ_output", "bf16", routed * intermediate, "write")
    operand("w2_input", "bf16", routed * intermediate, "read")
    operand("w2_active_weights", "e2m1_packed", active * hidden * intermediate, "read")
    operand("w2_active_scales", "ue8m0", active * hidden * (intermediate // 32), "read")
    operand("weighted_w2_output", "bf16", routed * hidden, "write")
    operand("combine_input", "bf16", routed * hidden, "read")
    operand("combined_latent", "bf16", tokens * hidden, "write")
    return {
        "source": "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:_fused_marlin_moe/fused_marlin_moe",
        "activation_source": "vllm/model_executor/layers/fused_moe/activation.py:apply_moe_activation:MoEActivation.SITU",
        "source_stages": ("route_alignment", "w13", "situ_and_mul", "weighted_w2", "topk_sum"),
        "experts": experts, "routed_rows": routed, "active_experts": active,
        "weight_format": {"values": "e2m1_packed", "scale": "ue8m0", "group_size": 32},
        "activation": {"name": "situ", "beta": kernel.attributes["activation_beta"],
                       "linear_beta": kernel.attributes["activation_linear_beta"],
                       "arithmetic_dtype": "fp32", "input_output_dtype": "bf16",
                       "tanhf_calls": 3 * routed * intermediate,
                       "multiplications_with_precomputed_coefficients": 7 * routed * intermediate,
                       "additions": routed * intermediate},
        "tensor_flops_live": 6 * routed * hidden * intermediate,
        "logical_intermediate":kernel.attributes['logical_intermediate'],
        "physical_intermediate":intermediate,
        "useful_tensor_flops":6*routed*hidden*kernel.attributes['logical_intermediate'],
        "padding_tensor_flops":6*routed*hidden*(intermediate-kernel.attributes['logical_intermediate']),
        "router_output_weight_multiplications": routed * hidden,
        "topk_sum_additions": tokens * hidden * (topk - 1),
        "apply_router_weight_on_input": False,
        "operands": tuple(operands),
        "workspace_aliases": (("w13_output", "weighted_w2_output"),),
        "alias_lifetimes": "w13 is dead after SiTU; its backing cache13 storage is reused for W2 output",
        "temporary_alignment_workspace_bound": False,
        "source_launch": None, "demand": None, "seconds": None,
        "limitations": (
            "declared_TP_Marlin_BF16_W4A16_source_physical_channels_including_padding_not_native_launch",
            "histogram_is_supplied_provenance_not_inferred_checkpoint_logits_or_observed_topk",
            "selected_IDs_FP32_weights_are_external_inputs_selection_not_in_this_aggregate",
            "physical_channel_math_excludes_M_row_MMA_padding_and_dequant_issue_services",
            "active_weight_storage_once_not_physical_reloads_cache_or_DRAM",
            "alignment_sorted_IDs_expert_IDs_post_padded_count_and_Marlin_scratch_unbound",
            "SiTU_three_tanhf_calls_not_one_instruction_per_tanh_or_SiLU",
            "FP32_accumulation_and_BF16_W2_weighted_store_rounding_retained_not_numerical_equivalence_proof",
            "several_physical_kernels_source_aggregate_not_one_fused_GPU_launch"),
    }


def source_work(kernel):
    source = logical_work(kernel)
    routed, active = source["routed_rows"], source["active_experts"]
    tokens, topk = kernel.shape["tokens"], kernel.shape["topk"]
    hidden, intermediate = kernel.shape["hidden"], kernel.shape["intermediate"]
    operands = {operand["role"]: operand for operand in source["operands"]}

    def demand(tensor=0, fma=0, sfu=0):
        return {"compute_tensor": float(tensor), "compute_vector": 0.0,
                "compute_reduction": 0.0, "compute_fma": float(fma),
                "compute_sfu": float(sfu), "compute_alu": 0.0}

    def stage(name, roles, stage_demand):
        return {"stage": name, "demand": stage_demand,
                "logical_bytes": sum(operands[role]["bytes"] for role in roles)}

    activation_values = routed * intermediate
    stages = (
        stage("route_alignment", ("topk_ids",), demand()),
        stage("w13", ("latent_per_route", "w13_active_weights", "w13_active_scales", "w13_output"),
              demand(tensor=4 * routed * hidden * intermediate,
                     fma=4 * active * hidden * intermediate)),
        stage("situ_and_mul", ("situ_input", "situ_output"),
              demand(fma=8 * activation_values, sfu=3 * activation_values)),
        stage("weighted_w2", ("w2_input", "w2_active_weights", "w2_active_scales",
                              "topk_weights", "weighted_w2_output"),
              demand(tensor=2 * routed * hidden * intermediate,
                     fma=2 * active * hidden * intermediate + routed * hidden)),
        stage("topk_sum", ("combine_input", "combined_latent"),
              demand(fma=routed * hidden)),
    )
    return {"stages": stages, "logical_bytes": sum(item["logical_bytes"] for item in stages),
            "source": source["source"], "source_stages": source["source_stages"],
            "tensor_flops_live": source["tensor_flops_live"],
            "logical_intermediate":source['logical_intermediate'],"physical_intermediate":source['physical_intermediate'],
            "useful_tensor_flops":source['useful_tensor_flops'],"padding_tensor_flops":source['padding_tensor_flops'],
            "dequantization_math": "two_BF16_multiplies_per_active_weight_value_once",
            "topk_source_additions": routed * hidden,
            "topk_minimal_additions": tokens * hidden * (topk - 1),
            "price_composition": "sum_serial_stage_rooflines",
            "limitations": source["limitations"] + (
                "active_weight_dequantization_math_excludes_M_padding_reloads_and_native_issue_services",
                "public_role_bytes_and_live_math_are_an_analytical_lower_bound_not_native_latency_acceptance")}

