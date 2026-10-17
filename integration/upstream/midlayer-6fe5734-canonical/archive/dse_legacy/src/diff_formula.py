"""Differentiable performance formula.

Implements forward + backward pass for t_total computation.
After P50 refactor: DiffPerformanceFormula inherits PerformanceFormula's
forward pass and adds backward pass only. This eliminates ~100 lines of
forward-pass duplication.

Gradient rules:
  t_r = demand_r / eff_cap_r  (per-resource time)
  t_g = sum(t_r for r in group g)  (within-group serial)
  t_bn = max(t_g)  (bottleneck = max across groups)
  t_total = max(t_bn + C_launch, pipeline_floor) + max(0, waves*serial - t_dram)

Subgradients of max:
  ∂max/∂t_g* = 1 where g* is the argmax group, 0 otherwise.
  For ties, pick the first group (deterministic subgradient).

Phase 2 L2 BW eff:
  If L2 is the bottleneck and l2_bw_eff != 1.0:
    t_l2 = demand_l2 / (cap_l2 * l2_bw_eff)
  The gradient of t_total w.r.t. l2_bw_eff is:
    ∂t_total/∂l2_bw_eff = -demand_l2 / (cap_l2 * l2_bw_eff^2)
    (only if L2 group is the bottleneck after Phase 2)
"""

import numpy as np
from src.kernel.hardware_spec import HWSpec
from src.kernel.validation.legacy_kernel_base import PredictionInput
from src.kernel.formula import PerformanceFormula


class DiffKernelPrediction:
    """Prediction result with gradients."""

    def __init__(self, t_total_s, t_per_group, t_per_resource,
                 bottleneck_group,
                 grad_demand, grad_capacity, grad_l2_bw_eff,
                 grad_eta, grad_active_units,
                 eff_cap_per_resource=None,
                 dram_concurrency=None,
                 grad_l2_size=0.0,
                 grad_n_part=0.0):
        self.t_total_s = t_total_s
        self.t_per_group = t_per_group
        self.t_per_resource = t_per_resource
        self.bottleneck_group = bottleneck_group
        self.grad_demand = grad_demand
        self.grad_capacity = grad_capacity
        self.grad_l2_bw_eff = grad_l2_bw_eff
        self.grad_eta = grad_eta
        self.grad_active_units = grad_active_units
        # ∂t/∂(L2_size_bytes): only nonzero when DRAM-bound and the predictor
        # supplied d_dram_d_l2size (GEMM stack-distance hit curve). See P48.
        self.grad_l2_size = grad_l2_size
        # ∂t/∂N_part: gradient through M/D/1 queueing model for DSE
        self.grad_n_part = grad_n_part
        # Effective capacity per resource name at the actual active_units/eta for
        # this kernel. Used for correct chain-rule when converting grad_capacity to
        # ∂t/∂hw_scale: ∂t/∂scale = sum_k grad_capacity_k × eff_cap_k (not × global_cap).
        self.eff_cap_per_resource = eff_cap_per_resource or {}
        self.dram_concurrency = dram_concurrency

    @property
    def t_total_ns(self):
        return self.t_total_s * 1e9


class DiffPerformanceFormula(PerformanceFormula):
    """Differentiable version of PerformanceFormula.

    Inherits the forward pass from PerformanceFormula to eliminate
    code duplication. Adds backward pass that computes analytical gradients
    of t_total w.r.t. demand, capacity, l2_bw_eff, eta, and active_units.
    """

    def predict_kernel(self, inp):
        """Forward + backward pass. Returns DiffKernelPrediction.

        Forward result is numerically identical to PerformanceFormula.predict_kernel().
        """
        if not isinstance(inp, PredictionInput):
            raise TypeError(
                f"predict_kernel requires PredictionInput, got {type(inp).__name__}")

        demand_per_resource = inp.demand_per_resource
        eta = inp.eta
        active_units = inp.active_units
        sm_freq_hz = inp.sm_freq_hz
        dram_freq_hz = inp.dram_freq_hz
        l1_bypass = inp.l1_bypass

        # ── DRAM concurrency (mirrors PerformanceFormula) ──
        if l1_bypass and self.hw.sm_saturate_latency > 0:
            sm_sat_bw = self.hw.sm_saturate_bw
            sm_sat_lat = self.hw.sm_saturate_latency
            # Scale saturation thresholds when frequencies differ from nominal.
            # sm_saturate_bw = N_SM * DRAM_cap / L2_cap, so:
            #   scaled = sm_saturate_bw * (dram_freq_hz / nominal_dram) / (sm_freq_hz / nominal_sm)
            if self.hw.nominal_dram_freq_hz > 0 and self.hw.nominal_sm_freq_hz > 0:
                freq_ratio = (dram_freq_hz / self.hw.nominal_dram_freq_hz) / (sm_freq_hz / self.hw.nominal_sm_freq_hz)
                sm_sat_bw *= freq_ratio
                sm_sat_lat *= freq_ratio
            sm_sat = max(sm_sat_bw, sm_sat_lat)
        else:
            sm_sat = self.hw.sm_saturate_bw
            if self.hw.nominal_dram_freq_hz > 0 and self.hw.nominal_sm_freq_hz > 0:
                freq_ratio = (dram_freq_hz / self.hw.nominal_dram_freq_hz) / (sm_freq_hz / self.hw.nominal_sm_freq_hz)
                sm_sat *= freq_ratio
        dram_concurrency = min(1.0, active_units / sm_sat)

        # ── Per-resource time + effective capacities (for backward) ──
        t_per_resource = {}
        eff_cap_per_resource = {}
        eff_cap_per_role = {}  # role -> effective capacity for chain-rule
        l2_raw_demand = 0.0
        l2_raw_cap = 0.0
        l2_resource_name = None
        unified_dram_bound = False

        for res in self.hw.resources:
            if res.role not in demand_per_resource:
                t_per_resource[res.name] = 0.0
                eff_cap_per_resource[res.name] = 0.0
                continue
            demand = demand_per_resource[res.role]
            if demand <= 0:
                t_per_resource[res.name] = 0.0
                eff_cap_per_resource[res.name] = 0.0
                continue
            cap = res.effective_capacity(
                active_units, sm_freq_hz=sm_freq_hz, dram_freq_hz=dram_freq_hz)
            if cap <= 0:
                t_per_resource[res.name] = 0.0
                eff_cap_per_resource[res.name] = 0.0
                continue
            if res.type == "compute":
                effective_eta = max(eta, 1e-9)
                eff_cap = cap * effective_eta
                t_per_resource[res.name] = demand / eff_cap
                eff_cap_per_resource[res.name] = eff_cap
                eff_cap_per_role[res.role] = eff_cap
            elif res.type == "memory" and res.role == "dram":
                if self.UNIFIED_MEMORY_PHASE >= 2:
                    t_per_resource[res.name] = 0.0
                    eff_cap_per_resource[res.name] = 0.0
                    eff_cap_per_role[res.role] = 0.0
                else:
                    eff_cap = cap * dram_concurrency * inp.dram_bw_eff
                    t_per_resource[res.name] = demand / eff_cap
                    eff_cap_per_resource[res.name] = eff_cap
                    eff_cap_per_role[res.role] = eff_cap
            elif res.type == "memory" and res.role == "l2_cache":
                if self._unified_mem is not None:
                    dram_demand_bytes = demand_per_resource.get("dram", 0.0)
                    if self.UNIFIED_MEMORY_PHASE >= 2 and demand > 0 and dram_demand_bytes > 0:
                        m = dram_demand_bytes / demand
                    else:
                        m = 0.0
                    unified_cap, unified_dram_bound = self._unified_mem.effective_bw_bytes_per_sec(
                        m, self.hw.num_compute_units, sm_freq_hz, dram_freq_hz,
                        dram_bw_eff=inp.dram_bw_eff,
                        dram_concurrency=dram_concurrency,
                        lsu_pct=inp.lsu_pct,
                        l2_issue_rate=inp.l2_issue_rate)
                    t_per_resource[res.name] = demand / unified_cap
                    eff_cap_per_resource[res.name] = unified_cap
                    eff_cap_per_role[res.role] = unified_cap
                    l2_raw_demand = demand
                    l2_raw_cap = unified_cap
                    l2_resource_name = res.name
                else:
                    unified_dram_bound = False
                    t_per_resource[res.name] = demand / cap
                    eff_cap_per_resource[res.name] = cap
                    eff_cap_per_role[res.role] = cap
                    l2_raw_demand = demand
                    l2_raw_cap = cap
                    l2_resource_name = res.name
            elif res.type == "memory":
                t_per_resource[res.name] = demand / cap
                eff_cap_per_resource[res.name] = cap
                eff_cap_per_role[res.role] = cap
            else:
                raise ValueError(
                    f"Unknown resource type={res.type} for {res.name}")

        # ── Per-group time ──
        t_per_group = {}
        for group in self.hw.resource_groups:
            t_per_group[group.name] = sum(
                t_per_resource[member] for member in group.members)

        # ── Bottleneck ──
        if t_per_group:
            t_bottleneck = max(t_per_group.values())
            bottleneck_group = max(t_per_group, key=lambda g: t_per_group[g])
        else:
            t_bottleneck = 0.0
            bottleneck_group = "none"

        # ── Phase 2: L2 BW eff ──
        l2_applied = False
        if (l2_resource_name and bottleneck_group == "l2"
                and inp.l2_bw_eff != 1.0
                and not unified_dram_bound):
            effective_cap = l2_raw_cap * inp.l2_bw_eff
            if effective_cap > 0:
                t_per_resource[l2_resource_name] = l2_raw_demand / effective_cap
                eff_cap_per_resource[l2_resource_name] = effective_cap
                eff_cap_per_role["l2_cache"] = effective_cap
                for group in self.hw.resource_groups:
                    if group.name == "l2":
                        t_per_group[group.name] = sum(
                            t_per_resource[member] for member in group.members)
                t_bottleneck = max(t_per_group.values())
                bottleneck_group = max(
                    t_per_group, key=lambda g: t_per_group[g])
                l2_applied = True

        # ── Launch overhead ──
        t_launch = self.hw.dispatch_overhead_ns * 1e-9
        t_total = t_bottleneck + t_launch

        if t_launch > t_bottleneck:
            bottleneck_group = "launch"

        # ── Serial L2 latency for launch-bound non-tensor low-occupancy CTAs ──
        t_l2_additive_active = False
        t_l2_serial_d_l2_demand = 0.0
        if (bottleneck_group == "launch" and inp.is_non_tensor
                and inp.grid_size <= self.hw.num_compute_units
                and inp.enable_l2_serial_latency):
            l2_bytes_dem = demand_per_resource["l2_cache"]
            if l2_bytes_dem > 0 and inp.grid_size > 0:
                n_per_cta = l2_bytes_dem / inp.grid_size / self.hw.cacheline_bytes
                t_l2_serial_ns = n_per_cta * self.hw.l2_latency_ns / self.hw.mshr_concurrency
                t_per_resource_max_ns = t_bottleneck * 1e9
                t_l2_additive_ns = max(0.0, t_l2_serial_ns - t_per_resource_max_ns)
                t_total += t_l2_additive_ns * 1e-9
                if t_l2_additive_ns > 0:
                    t_l2_additive_active = True
                    t_l2_serial_d_l2_demand = (
                        self.hw.l2_latency_ns * 1e-9
                        / (inp.grid_size * self.hw.cacheline_bytes * self.hw.mshr_concurrency)
                    )

        # ── Pipeline floor (max semantics) ──
        pipeline_floor_dominant = False
        if inp.pipeline_floor_ns > 0:
            floor_s = inp.pipeline_floor_ns * 1e-9
            if floor_s > t_total:
                t_total = floor_s
                bottleneck_group = "pipeline_floor"
                pipeline_floor_dominant = True

        # ── Pipeline serial (additive with per-wave DRAM hiding) ──
        if inp.pipeline_serial_ns > 0:
            import math as _math
            serial_s = inp.pipeline_serial_ns * 1e-9
            waves = _math.ceil(inp.grid_size / self.hw.num_compute_units)
            total_serial_s = waves * serial_s
            t_dram_s = max(t_per_group.get("dram", 0), t_per_group.get("l2", 0))
            additive = max(0.0, total_serial_s - t_dram_s)
            t_total += additive

        # ── Exposed pipeline latency (always additive, never DRAM-discounted) ──
        # Mirrors formula.py: FA k-loop serial dependency latency hidden only by
        # per-step compute (folded into pipeline_exposed_ns by the predictor).
        if getattr(inp, "pipeline_exposed_ns", 0.0) > 0:
            import math as _math
            waves = _math.ceil(inp.grid_size / self.hw.num_compute_units)
            t_total += waves * inp.pipeline_exposed_ns * 1e-9

        # ══════════════════════════════════════════
        # BACKWARD PASS: analytical gradients
        # ══════════════════════════════════════════

        grad_demand = {}
        grad_capacity = {}
        grad_l2_bw_eff = 0.0
        grad_eta = 0.0
        grad_active_units = 0.0
        grad_l2_size = 0.0
        grad_n_part = 0.0  # ∂t/∂N_part for DSE on partition count

        if pipeline_floor_dominant or bottleneck_group == "launch":
            if t_l2_additive_active and not pipeline_floor_dominant:
                for res in self.hw.resources:
                    if res.role not in demand_per_resource:
                        grad_demand[res.role] = 0.0
                        grad_capacity[res.role] = 0.0
                        continue
                    if res.role == "l2_cache":
                        grad_demand[res.role] = t_l2_serial_d_l2_demand
                        grad_capacity[res.role] = 0.0
                    else:
                        grad_demand[res.role] = 0.0
                        grad_capacity[res.role] = 0.0
            # Gradient of active_units through dram_concurrency
            if dram_concurrency < 1.0:
                bn_group = None
                for group in self.hw.resource_groups:
                    if group.name == bottleneck_group:
                        bn_group = group
                        break
                if bn_group is not None:
                    for res in self.hw.resources:
                        if res.role == "dram" and res.name in bn_group.members:
                            cap_d = res.effective_capacity(
                                active_units, sm_freq_hz=sm_freq_hz,
                                dram_freq_hz=dram_freq_hz)
                            demand_d = demand_per_resource["dram"]
                            if cap_d > 0 and dram_concurrency > 0 and sm_sat > 0:
                                grad_active_units = (
                                    -demand_d
                                    / (cap_d * inp.dram_bw_eff * dram_concurrency * dram_concurrency * sm_sat)
                                )
                            break
        else:
            bn_group = None
            for group in self.hw.resource_groups:
                if group.name == bottleneck_group:
                    bn_group = group
                    break

            if bn_group is not None:
                for res in self.hw.resources:
                    if res.role not in demand_per_resource:
                        grad_demand[res.role] = 0.0
                        grad_capacity[res.role] = 0.0
                        continue
                    is_bn_member = res.name in bn_group.members
                    if is_bn_member:
                        eff_cap = eff_cap_per_resource[res.name]
                        if eff_cap > 0:
                            grad_demand[res.role] = 1.0 / eff_cap
                            demand_val = demand_per_resource[res.role]
                            grad_capacity[res.role] = -demand_val / (eff_cap * eff_cap)
                        else:
                            grad_demand[res.role] = 0.0
                            grad_capacity[res.role] = 0.0
                    else:
                        grad_demand[res.role] = 0.0
                        grad_capacity[res.role] = 0.0

                # ∂t/∂(L2_size): when DRAM-bound, increasing L2 size changes the
                # L2/DRAM traffic split via the stack-distance hit curve. The
                # predictor supplies d(dram_bytes)/d(L2_size) in inp.d_dram_d_l2size
                # (analytic, parameter-free, P48). Chain rule:
                #   t_dram = dram_demand / eff_cap_dram
                #   dt/dC  = (1/eff_cap_dram) · d(dram_demand)/dC
                #          = grad_demand["dram"] · d_dram_d_l2size
                # Zero on plateaus (d_dram_d_l2size = 0), which is physically
                # correct: enlarging L2 there does not reduce DRAM traffic.
                dram_is_bn = any(
                    res2.role == "dram" and res2.name in bn_group.members
                    for res2 in self.hw.resources)
                if dram_is_bn and inp.d_dram_d_l2size != 0.0 and "dram" in grad_demand:
                    grad_l2_size = grad_demand["dram"] * inp.d_dram_d_l2size
                    eff_cap_l2 = l2_raw_cap * inp.l2_bw_eff
                    if eff_cap_l2 > 0:
                        dt_l2_d_eff = -l2_raw_demand / (eff_cap_l2 * inp.l2_bw_eff)
                    else:
                        dt_l2_d_eff = 0.0
                    if bottleneck_group == "l2":
                        grad_l2_bw_eff = dt_l2_d_eff
                    else:
                        grad_l2_bw_eff = 0.0

                # ∂t/∂N_part: chain rule through M/D/1 queueing model
                # Only when L2 is the bottleneck group (not DRAM, compute, etc.)
                # and when the binding BW term depends on τ_hit (i.e., BW_mshr or BW_part).
                if (bottleneck_group == "l2" and self._unified_mem is not None
                        and inp.l2_issue_rate > 0):
                    # Compute BW components to find binding term
                    bw_comps = self._unified_mem.effective_bw_components(
                        m, active_units, sm_freq_hz, dram_freq_hz,
                        dram_bw_eff=inp.dram_bw_eff,
                        dram_concurrency=dram_concurrency,
                        lsu_pct=inp.lsu_pct,
                        l2_issue_rate=inp.l2_issue_rate)
                    binding_term = bw_comps['binding_term']

                    d_tau_hit_dN = self._unified_mem.grad_tau_hit_wrt_npart(
                        inp.l2_issue_rate, active_units)
                    if d_tau_hit_dN != 0:
                        # ∂t/∂N_part depends on which BW term is binding
                        if binding_term == 'mshr':
                            # BW_mshr = MSHR × cacheline × N_SM × freq / τ_avg
                            # ∂BW_mshr/∂τ_hit = -BW_mshr × (1-m)/τ_avg
                            tau_hit = bw_comps['tau_hit']
                            tau_avg = bw_comps['tau_avg']
                            m_phys = min(m, 1.0)
                            if tau_avg > 0:
                                dbw_dtau = -bw_comps['bw_mshr'] * (1.0 - m_phys) / tau_avg
                                dt_dbw = -l2_raw_demand / (l2_raw_cap ** 2)
                                grad_n_part = dt_dbw * dbw_dtau * d_tau_hit_dN
                        elif binding_term == 'part':
                            # BW_part = N_part × part_peak_bpc × freq
                            # ∂BW_part/∂N_part = part_peak_bpc × freq
                            dbw_dNpart = self._unified_mem.partition_peak_bpc * sm_freq_hz
                            dt_dbw = -l2_raw_demand / (l2_raw_cap ** 2)
                            grad_n_part = dt_dbw * dbw_dNpart
                        else:
                            # BW_issue and BW_dram don't depend on N_part
                            grad_n_part = 0.0

                effective_eta = max(eta, 1e-9)
                has_compute_bn = any(
                    res2.type == "compute" and res2.name in bn_group.members
                    for res2 in self.hw.resources)
                if has_compute_bn:
                    grad_eta = 0.0
                    for res2 in self.hw.resources:
                        if res2.name in bn_group.members and res2.type == "compute":
                            cap2 = res2.effective_capacity(
                                active_units, sm_freq_hz=sm_freq_hz,
                                dram_freq_hz=dram_freq_hz)
                            demand2 = demand_per_resource[res2.role]
                            if cap2 > 0 and effective_eta > 0:
                                grad_eta += -demand2 / (cap2 * effective_eta * effective_eta)

                if dram_concurrency < 1.0:
                    for res2 in self.hw.resources:
                        if res2.role == "dram" and res2.name in bn_group.members:
                            cap_d = res2.effective_capacity(
                                active_units, sm_freq_hz=sm_freq_hz,
                                dram_freq_hz=dram_freq_hz)
                            demand_d = demand_per_resource["dram"]
                            if cap_d > 0 and dram_concurrency > 0 and sm_sat > 0:
                                grad_active_units = (
                                    -demand_d
                                    / (cap_d * inp.dram_bw_eff * dram_concurrency * dram_concurrency * sm_sat)
                                )
                            break

        # ── Unified model: recover DRAM BW gradients ──
        # When DRAM ceiling is binding in the unified model, the L2 time
        # is actually determined by DRAM bandwidth:
        #   t_l2 = l2_demand / (dram_peak * conc * eff / m)
        #        = dram_demand / (dram_peak * conc * eff)
        # Map the gradient from L2 to DRAM so DSE can explore DRAM BW.
        if (self.UNIFIED_MEMORY_PHASE >= 2 and unified_dram_bound
                and not pipeline_floor_dominant
                and bottleneck_group != "launch"):
            dram_demand = demand_per_resource.get("dram", 0.0)
            if dram_demand > 0:
                cap_d = 0.0
                for res2 in self.hw.resources:
                    if res2.role == "dram":
                        cap_d = res2.effective_capacity(
                            active_units, sm_freq_hz=sm_freq_hz,
                            dram_freq_hz=dram_freq_hz)
                        break
                if cap_d > 0:
                    dram_eff_cap = cap_d * dram_concurrency * inp.dram_bw_eff
                    grad_demand["dram"] = 1.0 / dram_eff_cap
                    grad_capacity["dram"] = -dram_demand / (dram_eff_cap ** 2)
                    eff_cap_per_role["dram"] = dram_eff_cap
                    # L2 hardware capacity is irrelevant when DRAM ceiling binds
                    grad_capacity["l2_cache"] = 0.0
                    eff_cap_per_role["l2_cache"] = 0.0
                    if dram_concurrency < 1.0 and sm_sat > 0:
                        grad_active_units = (
                            -dram_demand
                            / (cap_d * inp.dram_bw_eff
                               * dram_concurrency * dram_concurrency * sm_sat)
                        )
                    if inp.d_dram_d_l2size != 0.0:
                        grad_l2_size = grad_demand["dram"] * inp.d_dram_d_l2size

        return DiffKernelPrediction(
            t_total_s=t_total,
            t_per_group=t_per_group,
            t_per_resource=t_per_resource,
            bottleneck_group=bottleneck_group,
            grad_demand=grad_demand,
            grad_capacity=grad_capacity,
            grad_l2_bw_eff=grad_l2_bw_eff,
            grad_eta=grad_eta,
            grad_active_units=grad_active_units,
            eff_cap_per_resource=eff_cap_per_role,
            dram_concurrency=dram_concurrency,
            grad_l2_size=grad_l2_size,
            grad_n_part=grad_n_part,
        )


def aggregate_gradients(predictions):
    """Aggregate gradients across multiple serial kernel predictions.

    Physical basis: serial kernels' latencies add (t_op = sum t_kernel_i),
    so the gradient of t_op w.r.t. any resource demand or capacity is the sum
    of per-kernel gradients. This enables per-op DSE when an op decomposes into
    multiple kernels (e.g., GEMM prefill + splitKreduce, FA fwd + combine).

    Args:
        predictions: list of DiffKernelPrediction from predict_kernel()

    Returns:
        dict with keys:
            grad_demand[role]     = sum of per-kernel grad_demand[role]
            grad_capacity[role]   = sum of per-kernel grad_capacity[role]
            grad_l2_bw_eff        = sum of per-kernel grad_l2_bw_eff
            grad_eta              = sum of per-kernel grad_eta
            grad_active_units     = sum of per-kernel grad_active_units
            t_total_s             = sum of per-kernel t_total_s
    """
    if not predictions:
        return {
            "grad_demand": {},
            "grad_capacity": {},
            "grad_capacity_scaled": {},
            "dram_scale_grad_capacity_scaled": 0.0,
            "compute_scale_extra_grad_capacity_scaled": 0.0,
            "grad_l2_bw_eff": 0.0,
            "grad_eta": 0.0,
            "grad_active_units": 0.0,
            "grad_l2_size": 0.0,
            "grad_n_part": 0.0,
            "t_total_s": 0.0,
        }

    all_roles = set()
    for p in predictions:
        all_roles.update(p.grad_demand.keys())
        all_roles.update(p.grad_capacity.keys())

    grad_demand = {}
    for role in all_roles:
        total = 0.0
        for p in predictions:
            if role in p.grad_demand:
                total += p.grad_demand[role]
        grad_demand[role] = total
    grad_capacity = {}
    for role in all_roles:
        total = 0.0
        for p in predictions:
            if role in p.grad_capacity:
                total += p.grad_capacity[role]
        grad_capacity[role] = total
    grad_l2_bw_eff = sum(p.grad_l2_bw_eff for p in predictions)
    grad_eta = sum(p.grad_eta for p in predictions)
    grad_active_units = sum(p.grad_active_units for p in predictions)
    grad_l2_size = sum(p.grad_l2_size for p in predictions)
    grad_n_part = sum(p.grad_n_part for p in predictions)
    t_total_s = sum(p.t_total_s for p in predictions)

    # grad_capacity_scaled: sum_k (grad_capacity_k[role] × eff_cap_k[role])
    # This is the correct multiplier for ∂t/∂hw_scale chain-rule:
    #   ∂t/∂scale = sum_k grad_capacity_k × eff_cap_k   (not × global cap_base)
    # For bottleneck resources: grad_capacity_k × eff_cap_k = -t_resource_k (the roofline time)
    grad_capacity_scaled = {}
    for role in all_roles:
        total = 0.0
        for p in predictions:
            gc = p.grad_capacity.get(role, 0.0)
            ec = p.eff_cap_per_resource.get(role, 0.0)
            total += gc * ec
        grad_capacity_scaled[role] = total

    dram_scale_grad_capacity_scaled = 0.0
    compute_scale_extra_grad_capacity_scaled = 0.0
    for p in predictions:
        gc = p.grad_capacity.get("dram", 0.0)
        ec = p.eff_cap_per_resource.get("dram", 0.0)
        scaled = gc * ec
        if scaled == 0.0:
            continue
        if p.dram_concurrency is not None and p.dram_concurrency < 1.0:
            compute_scale_extra_grad_capacity_scaled += scaled
        else:
            dram_scale_grad_capacity_scaled += scaled

    return {
        "grad_demand": grad_demand,
        "grad_capacity": grad_capacity,
        "grad_capacity_scaled": grad_capacity_scaled,
        "dram_scale_grad_capacity_scaled": dram_scale_grad_capacity_scaled,
        "compute_scale_extra_grad_capacity_scaled": compute_scale_extra_grad_capacity_scaled,
        "grad_l2_bw_eff": grad_l2_bw_eff,
        "grad_eta": grad_eta,
        "grad_active_units": grad_active_units,
        "grad_l2_size": grad_l2_size,
        "grad_n_part": grad_n_part,
        "t_total_s": t_total_s,
    }
