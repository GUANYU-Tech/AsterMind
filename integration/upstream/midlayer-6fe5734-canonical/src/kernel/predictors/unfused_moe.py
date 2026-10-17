"""Prediction for aggregate, explicitly decomposed unfused MoE observations."""

from src.kernel.bank_stream_model import (
    bank_stream_ns, matmul_stream, pointwise_stream, softmax_stream)
from src.kernel.formula import PerformanceFormula
from src.kernel.kernel_base import KernelPredictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.predictors.common import (
    PredictionInput,
    attach_structured_contract,
    build_forward_demand,
    launch_wave_geometry,
    prediction_from_ns,
    roofline_ns,
)


def _routing(kernel: KernelSpec):
    experts = int(kernel.shape["experts"])
    values = kernel.attributes["tokens_per_expert"]
    counts = tuple(int(value) for value in values)
    if len(counts) != experts:
        raise ValueError(
            f"tokens_per_expert has {len(counts)} entries, expected {experts}")
    if not any(count > 0 for count in counts):
        raise ValueError("UNFUSED_MOE routing must contain an active expert")
    return counts


def _model(kernel: KernelSpec, hardware):
    if kernel.variant != "decomposed":
        raise ValueError(
            f"unknown UNFUSED_MOE variant: {kernel.variant!r}")
    memory_model = hardware.kernel_memory_model
    if memory_model not in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            "UNFUSED_MOE currently requires a PIM-bank or "
            f"systolic-scratchpad execution model, got {memory_model!r}")

    counts = _routing(kernel)
    routed_rows = float(sum(counts))
    active_experts = sum(1 for count in counts if count > 0)
    hidden = int(kernel.shape["hidden"])
    intermediate = int(kernel.shape["intermediate"])
    element_bytes = int(kernel.shape["dtype_bytes"])

    # The retained ground truth is the sum of router, routing-softmax,
    # per-expert W1/activation/W2, and output-reduction stages.  Preserve that
    # materialized decomposition instead of pricing it as one grouped GEMM.
    hierarchy_bytes = float(
        3.0 * active_experts * hidden * intermediate * element_bytes
        + routed_rows * (2.0 * hidden + 3.0 * intermediate) * element_bytes)
    flops = {
        "compute_tensor": float(
            6.0 * routed_rows * hidden * intermediate),
        "compute_fma": 0.0,
        "compute_alu": 0.0,
        "compute_sfu": 0.0,
    }
    demand = build_forward_demand(
        hierarchy_bytes, hierarchy_bytes, False, flops)
    for role in ("xbar", "smem", "l1_cache"):
        demand[role] = 0.0

    execution_floor_ns = None
    if memory_model == "pim_bank":
        # Router, routing softmax, every active expert's W1/activation/W2, and
        # the output reduction run back to back on the bank-attached units.
        experts = int(kernel.shape["experts"])
        tokens = int(kernel.shape["tokens"])
        stages = [
            matmul_stream(tokens, hidden, experts),
            softmax_stream(experts * tokens),
        ]
        for count in counts:
            if count <= 0:
                continue
            stages.append(matmul_stream(count, hidden, 2 * intermediate))
            stages.append(pointwise_stream(count * intermediate))
            stages.append(matmul_stream(count, intermediate, hidden))
        stages.append(pointwise_stream(routed_rows * hidden))
        execution_floor_ns = bank_stream_ns(tuple(stages), hardware)
        # No cache stands between a bank-attached unit and its array, and the
        # bank stream above already prices every bank access and the single
        # SIMD datapath that consumes it.
        demand["l2_cache"] = 0.0
        demand["dram"] = 0.0
        for role in ("compute_fma", "compute_sfu", "compute_alu"):
            demand[role] = 0.0

    eta, wave_efficiency, active_units = launch_wave_geometry(
        kernel, hardware)
    if memory_model == "pim_bank":
        active_units = hardware.num_compute_units
        # A bank-attached launch has no CTA waves to quantize against: the
        # partitioner spreads the operand over every unit, and the bank
        # stream carries the resulting per-unit column count.
        eta, wave_efficiency = 1.0, 1.0
    elif memory_model == "systolic_scratchpad":
        pass
    else:
        raise AssertionError(
            f"validated UNFUSED_MOE memory model became invalid: {memory_model!r}")

    inp = PredictionInput(
        demand_per_resource=demand,
        eta=eta,
        wave_efficiency=wave_efficiency,
        active_units=active_units,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        grid_size=kernel.launch.grid_size,
        block_size=kernel.launch.block_size,
        launch_grid=kernel.launch.grid,
        launch_block=kernel.launch.block,
        variant=kernel.variant,
        mem_model=memory_model,
        compute_demands=(),
        memory_paths=(),
        pipeline_plan=None,
        dram_capacity_multiplier=(
            hardware.dram_protocol_efficiency_l1cached),
        support_level="compatibility",
        limitations=("outside_a800_c500_910b_acceptance_scope",),
        execution_time_floor_ns=execution_floor_ns,
    )
    return attach_structured_contract(
        inp,
        hardware,
        transaction_bytes=None,
        path_name=None,
    ), hierarchy_bytes


class UnfusedMoEPredictor(KernelPredictor):
    kind = KernelKind.UNFUSED_MOE

    def resource_input(self, kernel: KernelSpec, hardware):
        return _model(kernel, hardware)

    def midlayer(self, kernel: KernelSpec, hardware):
        inp, _ = self.resource_input(kernel, hardware)
        return PerformanceFormula(hardware).predict_kernel(inp)

    def roofline(self, kernel: KernelSpec, hardware):
        inp, compulsory = self.resource_input(kernel, hardware)
        return prediction_from_ns(
            roofline_ns(inp.demand_per_resource, hardware, compulsory),
            "roofline")
