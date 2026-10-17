"""Source-work analytical prediction for the DSA indexer pipeline."""

import math

from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import (
    build_aggregate_gpu_input,
    build_forward_demand,
    causal_attention_pairs,
    prediction_from_ns,
    roofline_ns,
)


def _model(kernel: KernelSpec, hardware):
    if kernel.variant == "dsa_index_reuse":
        return None, 0.0
    q_lens = tuple(kernel.attributes["q_lens"])
    kv_lens = tuple(kernel.attributes["kv_lens"])
    pairs = causal_attention_pairs(q_lens, kv_lens)
    query_tokens = sum(q_lens)
    heads = int(kernel.shape["num_query_heads"])
    head_dim = int(kernel.shape["head_dim"])
    topk = int(kernel.shape["topk"])
    quant_blocks = math.ceil(head_dim / 128)

    query_bytes = float(query_tokens * heads * head_dim)
    key_bytes = float(pairs * heads * (head_dim + 4 * quant_blocks))
    weight_bytes = float(query_tokens * heads * 4)
    score_bytes = float(pairs * 4)
    index_bytes = float(query_tokens * topk * 4)
    # sparse_attn_indexer materializes FP32 logits before its top-k selector;
    # account for the producer store and consumer reload explicitly.
    read_bytes = query_bytes + key_bytes + weight_bytes + score_bytes
    write_bytes = score_bytes + index_bytes
    candidate_rounds = max(1, math.ceil(math.log2(topk)))
    flops = {
        "compute_tensor": float(2 * pairs * heads * head_dim),
        "compute_fma": float(pairs * heads),
        "compute_alu": float(pairs * candidate_rounds + query_tokens * topk),
        "compute_sfu": 0.0,
    }
    total_bytes = read_bytes + write_bytes
    demand = build_forward_demand(total_bytes, total_bytes, True, flops)
    inp = build_aggregate_gpu_input(
        kernel,
        hardware,
        demand,
        write_l2_bytes=write_bytes,
        write_dram_bytes=write_bytes,
        limitations=(
            "dsa_indexer_multi_kernel_overlap_unavailable",
            "dsa_topk_selector_schedule_unavailable",
            "sm90_source_execution_projected_on_sm80",
            "production_duration_calibration_not_used",
        ),
    )
    return inp, total_bytes


class SparseAttentionIndexerPredictor(KernelPredictor):
    kind = KernelKind.SPARSE_ATTN_INDEXER

    def resource_input(self, kernel: KernelSpec, hardware):
        inp, compulsory = _model(kernel, hardware)
        if inp is None:
            raise NotImplementedError(
                "dsa_index_reuse is a zero-work alias without resource input")
        return inp, compulsory

    def midlayer(self, kernel: KernelSpec, hardware):
        if kernel.variant == "dsa_index_reuse":
            return prediction_from_ns(0.0, "alias")
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        if kernel.variant == "dsa_index_reuse":
            return prediction_from_ns(0.0, "alias")
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline",
        )
