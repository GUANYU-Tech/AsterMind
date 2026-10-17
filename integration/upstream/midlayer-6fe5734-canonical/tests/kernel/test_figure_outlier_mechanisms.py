"""Contract tests for the wt-figure-outliers mechanism fixes.

Covers three generalizations landed against named outliers on the article's
kernel latency scatter and the A800 low-QPS serving regression:

1. A800 FLASH_ATTN: a single-query-tile source order (decode, or a short
   prefill continuation mixed into the same batch) is dtype/phase-agnostic,
   so ``_fa2_varlen_single_tile_source_order_classes`` and the
   work-conserving queue admission generalize beyond pure decode.
2. A800 GEMM: the copy-only finite async-copy completion floor
   (``finite_async_operand_stream_completion``) is a byte-width probe, not a
   dtype-identity one, so it also covers bf16 launches with the exact
   fp16-calibrated geometry.
3. A800 small-kernel span floor: its measured domain is grids 48-216: it must
   not extrapolate below 48 for the MoE per-expert-row routing bookkeeping
   family (moe_align, moe_align_small_batch), whose grid tracks live
   routed-token count and is often far below 48 at low serving QPS.
"""

import pytest

from evaluation.kernel.dataset import get_hardware_config_path, load_kernel_dataset
from evaluation.kernel.evaluation_runner import _process_op
from evaluation.kernel.fa_serving_coverage_report import build_fa_serving_specs
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict


def test_a800_fa_mixed_single_tile_outlier_within_gate():
    """FA_SERVING MIXED_B27 (one short prefill continuation mixed with many
    decode requests, all within one query tile) no longer serializes the
    round-robin lanes that host its longest-KV request onto a fixed subset.
    """

    hardware_id = "nvidia_a800"
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    dataset = load_kernel_dataset(hardware_id)
    task_key = next(
        key for key in dataset
        if key.startswith("FA_SERVING") and "MIXED_B27_QSUM40" in key)
    attributes = dataset[task_key]["op_info"]["Attr"]
    target = dataset[task_key]["op_info"]["evaluation"]["timing"][
        "total_duration_ns"]

    specs = build_fa_serving_specs(attributes, hardware_id, hardware)
    predicted = sum(
        predict(spec, hardware, method="midlayer").t_total_ns
        for spec in specs)

    ape = abs(predicted - target) / target * 100.0
    assert ape <= 25.0, (
        f"A800 FA MIXED single-tile outlier regressed: {ape:.2f}% APE "
        f"(pred={predicted:.1f}, act={target:.1f})")


def test_a800_finite_async_stream_floor_covers_bf16_geometry():
    """The copy-only finite async-copy probe was calibrated on fp16 but
    describes byte movement only; a bf16 GEMM at the identical measured
    geometry (grid 64, block 128, split_k 1, pipeline depth 6, 16 KiB
    operand pair) must resolve to the same non-null profile as fp16.
    """

    import dataclasses

    import evaluation.kernel.evaluation_runner as evaluation_runner
    from src.kernel.predictors.gemm import _finite_async_operand_stream_profile

    hardware_id = "nvidia_a800"
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    dataset = load_kernel_dataset(hardware_id)
    # The canonical fp16 case whose launch selects the exact measured
    # (grid 64, block 128, operand_pair_bytes 16384) topology.
    task_key = "GEMM_B1_M1_K14336_N4096"

    captured = []
    original = evaluation_runner.predict_production

    def _capture(kernel, hw, method="midlayer", **kwargs):
        if method == "midlayer":
            captured.append(kernel)
        return original(kernel, hw, method=method, **kwargs)

    evaluation_runner.predict_production = _capture
    try:
        _process_op(
            task_key, dataset[task_key], get_hardware_config_path(hardware_id),
            False, True)
    finally:
        evaluation_runner.predict_production = original

    fp16_kernel = captured[0]
    assert fp16_kernel.dtype == "fp16"
    bf16_kernel = dataclasses.replace(fp16_kernel, dtype="bf16")

    fp16_profile = _finite_async_operand_stream_profile(fp16_kernel, hardware)
    bf16_profile = _finite_async_operand_stream_profile(bf16_kernel, hardware)

    assert fp16_profile is not None, (
        "fp16 must still resolve the finite async-copy profile")
    assert bf16_profile is not None, (
        "bf16 at the identical measured geometry must resolve the same "
        "copy-only, byte-width-based profile as fp16")
    assert fp16_profile == bf16_profile


def test_a800_span_floor_does_not_extrapolate_for_moe_routing_bookkeeping():
    """A grid below the span floor's measured minimum (48) must not pin the
    MoE per-expert-row routing bookkeeping kernels to the grid-48 value; any
    other memory-touching launch keeps the existing floor unchanged.
    """

    from src.kernel.formula import _resolve_small_kernel_span_floor
    from src.kernel.model_contract import (
        MemoryNodeDemand,
        MemoryPathDemand,
        MemorySourceDemand,
    )
    from src.kernel.predictors.common import PredictionInput

    hardware_id = "nvidia_a800"
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    memory_paths = (MemoryPathDemand(
        name="read",
        access_kind="read",
        initial_transactions=1.0,
        nodes=(MemoryNodeDemand("l2_cache", 1.0, 128.0, 128.0),),
        source=MemorySourceDemand(1, 1.0, None, None),
    ),)
    required_resources = (
        "l1_cache", "smem", "xbar", "l2_cache", "dram", "mte3",
        "compute_tensor", "compute_vector", "compute_reduction",
        "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
        "scalar_aiv",
    )
    base_kwargs = dict(
        demand_per_resource={role: 0.0 for role in required_resources},
        eta=1.0,
        wave_efficiency=1.0,
        active_units=1,
        sm_freq_hz=1.0,
        dram_freq_hz=1.0,
        grid_size=4,
        block_size=128,
        mem_model="gpu_mshr",
        compute_demands=(),
        memory_paths=memory_paths,
        pipeline_plan=None,
        include_dispatch_overhead=True,
        support_level="partial",
        limitations=("workload_producer_independence_unavailable",),
    )

    moe_inp = PredictionInput(variant="moe_align", **base_kwargs)
    other_inp = PredictionInput(variant="moe_sum", **base_kwargs)

    moe_floor = _resolve_small_kernel_span_floor(hardware, moe_inp)
    other_floor = _resolve_small_kernel_span_floor(hardware, other_inp)

    assert moe_floor == 0.0, (
        "moe_align below the measured grid-48 minimum must not extrapolate "
        "the span floor")
    assert other_floor > 0.0, (
        "non-routing-bookkeeping variants keep the existing floor at small "
        "grids, unchanged from before this fix")


def test_a800_silu_cold_operand_traffic_matches_counter_evidence():
    """act_and_mul's measured DRAM traffic is "reads miss, writes absorbed".

    The launch has no producer in its own dispatch (its L2ExecutionContext
    declares no same-program operand replay) and the canonical capture is
    cold, so the read stream cannot be L2-resident.  The launch's own
    non-duration NCU counters settle it: a 2:1 read:write stream measuring
    ~33-40% L2 hit rate is "every read misses, every write is absorbed", not
    the ~67% the opposite credit would require.  Pin the resulting predicted
    HBM bytes against the measured ones inside the segmented-service domain,
    where the round drain is modelled by the measured burst capacity.
    """

    from evaluation.kernel.memory_traffic_report import _kernel_traffic_points

    hardware_id = "nvidia_a800"
    dataset = load_kernel_dataset(hardware_id)
    # tok >= 108 with a 1024-thread block selects the published segmented
    # first/resident/queued service domain this correction is scoped to.
    task_key = "VECTOR_SILU_T256_I6144_B1"
    # Same argument shape the traffic report itself uses: oracle + forward on,
    # mechanism capture enabled so the per-kernel observation is attached.
    result = _process_op(
        task_key, dataset[task_key], get_hardware_config_path(hardware_id),
        True, True, False, 0.0, False, None, True)
    point, = _kernel_traffic_points(hardware_id, result)

    measured = point["measured_hbm_bytes"]
    predicted = point["predicted_hbm_bytes"]
    assert measured is not None and predicted is not None
    error = abs(predicted - measured) / measured * 100.0
    assert error <= 5.0, (
        "cold-operand act_and_mul HBM traffic drifted from its measured "
        f"counter: predicted={predicted}, measured={measured} ({error:.1f}%)")


def test_c500_fa_mixed_aliasing_launches_are_not_lane_skewed():
    """A per-request CTA block that divides the AP count must not pin the
    deepest query tiles onto the same lanes for every request.

    The MetaX Q64 prefill path declares a fixed CTA-to-AP mapping as an
    assumption (`metax_full_ap_round_robin_cta_assignment_assumed`).  Each
    request contributes ``q_tiles * query_heads`` consecutive CTAs whose KV
    depth grows with the tile, so when that block is a whole multiple of the
    104 APs, lane L draws the same tile depths from every request.  The three
    launches in this family whose block divides 104 -- HQ16/13-tile (208) and
    HQ32/13-tile (416) -- were exactly the three most over-predicted.
    """

    import statistics

    from evaluation.kernel.fa_serving_coverage_report import (
        build_fa_serving_specs,
    )

    hardware_id = "metax_c500"
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    dataset = load_kernel_dataset(hardware_id)
    aliasing = "MIXED_B39_QSUM810"
    rotating = "MIXED_B25_QSUM715"

    skew = {}
    for pattern in (aliasing, rotating):
        task_key = next(key for key in dataset if pattern in key)
        specs = build_fa_serving_specs(
            dataset[task_key]["op_info"]["Attr"], hardware_id, hardware)
        makespans = predict(
            specs[0], hardware,
            method="midlayer").diagnostics.pipeline.worker_makespan_s
        lanes = []
        for name, value in makespans.items():
            multiplicity = int(name.split("_x")[1]) if "_x" in name else 1
            lanes.extend([value] * multiplicity)
        skew[pattern] = max(lanes) / statistics.median(lanes)

    # The rotating launch was never skewed and must stay unskewed; the
    # aliasing one was 10x before the queued remainder stopped carrying a
    # fixed lane identity.
    assert skew[rotating] < 2.0, (
        f"non-aliasing launch became lane-skewed: {skew[rotating]:.2f}x")
    assert skew[aliasing] < 2.0, (
        "aliasing launch is still lane-skewed: "
        f"{skew[aliasing]:.2f}x (was 10.01x)")


def test_c500_padded_query_tiles_are_priced_from_measured_retire():
    """A padded query tile is a CTA that starts, rejects its tile and
    returns; it must not be priced at zero.

    The term is the published measured marginal cost of one such CTA
    (`early_exit_cta_retire`, 256 threads / 32 KiB, the source's own
    footprint) times the number of them the launch schedules.  A launch with
    no padded tiles must not acquire the term at all.
    """

    import evaluation.kernel.evaluation_runner as evaluation_runner

    hardware_id = "metax_c500"
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    rate = hardware.early_exit_cta_retire[
        "launch_wide_ns_per_early_exit_cta"]
    dataset = load_kernel_dataset(hardware_id)

    def _main_input(pattern):
        task_key = next(key for key in dataset if pattern in key)
        captured = []
        original = evaluation_runner.predict_production

        def _capture(kernel, hw, method="midlayer", **kwargs):
            result = original(kernel, hw, method=method, **kwargs)
            if method == "midlayer":
                captured.append(kernel)
            return result

        evaluation_runner.predict_production = _capture
        try:
            _process_op(
                task_key, dataset[task_key],
                get_hardware_config_path(hardware_id), False, True)
        finally:
            evaluation_runner.predict_production = original
        from src.kernel.kernel_base import create_predictor
        return create_predictor(
            captured[0].kind).resource_input(captured[0], hardware)[0]

    # B39: 39 requests x 13 scheduled tiles x 16 heads = 8112 CTAs, of which
    # 816 are active, so 7296 are padded.
    padded = _main_input("MIXED_B39_QSUM810")
    assert padded.early_exit_cta_retire_ns == pytest.approx(7296 * rate)

    # B27's longest request needs one query tile, so the launch schedules no
    # padded tile and must carry no retire term.
    unpadded = _main_input("MIXED_B27_QSUM40")
    assert unpadded.early_exit_cta_retire_ns is None


def test_a800_graph_replayed_launches_use_the_graph_span_floor():
    """A replayed CUDA-graph node and a standalone launch do not span the
    same time, and the published span floor was measured on standalone
    launches.

    The canonical kernel dataset captures standalone launches and carries no
    graph runtime mode, so it must keep that floor; a spec that declares a
    captured graph mode must take the measured graph-node span instead,
    which is lower at every measured grid.
    """

    from src.kernel.formula import _resolve_small_kernel_span_floor
    from src.kernel.model_contract import (
        MemoryNodeDemand,
        MemoryPathDemand,
        MemorySourceDemand,
    )
    from src.kernel.predictors.common import (
        PredictionInput,
        graph_replayed_launch,
    )

    hardware = HWSpec(get_hardware_config_path("nvidia_a800"))
    required = (
        "l1_cache", "smem", "xbar", "l2_cache", "dram", "mte3",
        "compute_tensor", "compute_vector", "compute_reduction",
        "compute_fma", "compute_sfu", "compute_alu", "warp_issue",
        "scalar_aiv",
    )
    memory_paths = (MemoryPathDemand(
        name="read", access_kind="read", initial_transactions=1.0,
        nodes=(MemoryNodeDemand("l2_cache", 1.0, 128.0, 128.0),),
        source=MemorySourceDemand(1, 1.0, None, None),
    ),)

    def _floor(grid, graph_replayed):
        return _resolve_small_kernel_span_floor(hardware, PredictionInput(
            demand_per_resource={role: 0.0 for role in required},
            eta=1.0, wave_efficiency=1.0, active_units=1, sm_freq_hz=1.0,
            dram_freq_hz=1.0, grid_size=grid, block_size=128,
            variant="elementwise", mem_model="gpu_mshr", compute_demands=(),
            memory_paths=memory_paths, pipeline_plan=None,
            include_dispatch_overhead=True, support_level="partial",
            limitations=("workload_producer_independence_unavailable",),
            graph_replayed=graph_replayed,
        ))

    # At every grid the graph node spans strictly less than the standalone
    # floor, so charging a graph-replayed launch the standalone floor
    # over-constrains it.
    for grid in (1, 48, 108, 216):
        standalone = _floor(grid, False)
        replayed = _floor(grid, True)
        assert replayed < standalone, (
            f"grid {grid}: graph span {replayed} is not below the "
            f"standalone floor {standalone}")

    # The regime comes from the spec: no declared graph mode means a
    # standalone launch, which is how the kernel dataset is captured.
    class _Spec:
        def __init__(self, attributes):
            self.attributes = attributes

    assert graph_replayed_launch(_Spec({})) is False
    assert graph_replayed_launch(_Spec({"graph_runtime_mode": "NONE"})) is False
    assert graph_replayed_launch(_Spec({"graph_runtime_mode": "FULL"})) is True
    # PIECEWISE leaves the configured split-op kinds eager and captures only
    # the runs between them.  Which side a call falls on is a property of the
    # step's call sequence, not of the KernelSpec, so a PIECEWISE spec cannot
    # claim a replayed node's span.
    assert graph_replayed_launch(
        _Spec({"graph_runtime_mode": "PIECEWISE"})) is False


def test_c500_kv_copy_reads_two_tensors_concurrently():
    """The pinned KV copy keeps its key and value loads outstanding together.

    ``reshape_and_cache_flash_kernel`` in the canonical vLLM-MetaX 0.8.5 build
    (identified by its exact 15-parameter signature) is one block-strided
    *scalar* loop whose body loads the key and then the value:

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
          scalar_t tgt_key   = key[src_key_idx];
          scalar_t tgt_value = value[src_value_idx];
          ...
        }

    Every physical thread of the block is therefore a read producer, and
    neither load's address depends on the other's result, so one iteration
    keeps both tensors' requests in flight.  Publishing only one tensor's
    requests halved the memory-level parallelism the path reaches.
    """

    hardware = HWSpec(get_hardware_config_path("metax_c500"))
    raw = load_kernel_dataset("metax_c500")
    hardware_path = get_hardware_config_path("metax_c500")
    for kv_heads in (4, 8):
        case = f"VECTOR_KVCACHE_T1024_KVH{kv_heads}D128_B1"
        _k, _m, _o, _f, _r, _p, kernels = _process_op(
            case, raw[case], hardware_path, True, True,
            capture_mechanisms=True)
        launch = kernels[0]["mechanisms"]["observed_launch"]
        parameters = launch["parameters"]
        elements_per_token = kv_heads * 128
        active_threads = min(elements_per_token, launch["block_size"])

        # Read producers are the block's physical threads, not a vec8 subset.
        assert parameters["source_bound_l2_read_warps_per_cta"] == float(
            -(-active_threads // hardware.parallel_width))
        # One warp's 64 contiguous BF16 reads fill exactly one L2
        # transaction, and the body issues that once per tensor.
        assert parameters["source_bound_l2_read_transactions_per_warp"] == 2.0

    # Both loop tensors in flight is what keeps the widest launch of the
    # family on its measured duration instead of over-predicting it by a
    # third; the whole C500 KVCACHE bucket stays inside the same gate.
    for case, bound in (
            ("VECTOR_KVCACHE_T1024_KVH4D128_B1", 25.0),
            ("VECTOR_KVCACHE_T1024_KVH8D128_B1", 25.0)):
        _key, _meta, _oracle, forward, _r, _p, _kernels = _process_op(
            case, raw[case], hardware_path, False, True)
        predicted, actual = forward
        error = 100.0 * (predicted - actual) / actual
        assert abs(error) <= bound, f"{case}: {error:+.2f}%"
