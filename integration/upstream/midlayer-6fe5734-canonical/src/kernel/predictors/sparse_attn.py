"""Source-work analytical prediction for DSA sparse MLA attention."""

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
    q_lens = tuple(kernel.attributes["q_lens"])
    kv_lens = tuple(kernel.attributes["kv_lens"])
    capacity = int(kernel.attributes["topk_capacity"])
    pairs = causal_attention_pairs(q_lens, kv_lens, capacity=capacity)
    query_tokens = sum(q_lens)
    heads = int(kernel.shape["num_query_heads"])
    qk_dim = int(kernel.shape["qk_head_dim"])
    value_dim = int(kernel.shape["v_head_dim"])
    dtype_bytes = int(kernel.shape["dtype_bytes"])

    query_bytes = float(query_tokens * heads * qk_dim * dtype_bytes)
    kv_bytes = float(
        pairs * heads * (qk_dim + value_dim) * dtype_bytes)
    index_bytes = float(pairs * heads * 4)
    output_bytes = float(query_tokens * heads * value_dim * dtype_bytes)
    read_bytes = query_bytes + kv_bytes + index_bytes
    tensor_flops = float(
        2 * pairs * heads * (qk_dim + value_dim))
    score_elements = float(pairs * heads)
    flops = {
        "compute_tensor": tensor_flops,
        "compute_fma": 3.0 * score_elements,
        "compute_reduction": score_elements,
        "compute_sfu": score_elements,
        "compute_alu": float(pairs * heads),
    }
    total_bytes = read_bytes + output_bytes
    demand = build_forward_demand(total_bytes, total_bytes, True, flops)
    inp = build_aggregate_gpu_input(
        kernel,
        hardware,
        demand,
        write_l2_bytes=output_bytes,
        write_dram_bytes=output_bytes,
        limitations=(
            "flashmla_sparse_tile_pipeline_schedule_unavailable",
            "selected_kv_cache_reuse_unavailable",
            "sm90_source_execution_projected_on_sm80",
            "production_duration_calibration_not_used",
        ),
    )
    return inp, total_bytes


class SparseAttentionPredictor(KernelPredictor):
    kind = KernelKind.SPARSE_ATTN

    def resource_input(self, kernel: KernelSpec, hardware):
        return _model(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline",
        )
