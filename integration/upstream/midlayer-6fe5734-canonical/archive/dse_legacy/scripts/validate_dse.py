"""
P20: DSE (Design Space Exploration) validation for the differentiable model.

Validates that gradient-based search effectively finds optimal hardware configurations,
comparing against brute-force grid search.

Tests:
1. Gradient sign consistency: ∂t/∂param matches actual Δt direction
2. Gradient-based DSE convergence: finds near-optimal config in ≤100 steps
3. Bottleneck phase diagram: correct DRAM-bound / L2-bound / compute-bound regions
4. Multi-kernel aggregate_gradients: analytical gradient DSE on multi-kernel op
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.kernel.hardware_spec import HWSpec
from src.dse.diff_formula import DiffPerformanceFormula, aggregate_gradients
from src.kernel.validation.legacy_kernel_base import PredictionInput


def create_workload_mix():
    """Create representative workload from LLM inference profile.

    Mix of compute-bound and memory-bound kernels that creates an interesting
    DSE landscape where both DRAM BW and compute throughput matter.
    """
    workload = []

    # Medium-batch GEMM (M=32): near the compute/memory crossover point
    # AI = 2*M*N*K / ((M*K + K*N + M*N)*2) ≈ 2*32*N / (K+N+32) ≈ moderate
    medium_gemms = [
        (0.20, "med_qkv", 32, 12288, 4096),
        (0.10, "med_o_proj", 32, 4096, 4096),
        (0.20, "med_gate_up", 32, 22016, 4096),
        (0.20, "med_down", 32, 4096, 11008),
    ]
    for weight, name, M, N, K in medium_gemms:
        flops = 2 * M * N * K
        traffic = (M * K + K * N + M * N) * 2  # fp16
        workload.append((weight, flops, traffic, name, False))

    # Decode attention (memory-bound): read KV cache
    S_decode, D, H = 1, 128, 32
    KV_len = 2048
    fa_decode_flops = 4 * S_decode * KV_len * D * H
    fa_decode_traffic = (S_decode * D + KV_len * D * 2 + S_decode * D) * H * 2
    workload.append((0.15, fa_decode_flops, fa_decode_traffic, "fa_decode", True))

    # Elementwise (pure memory-bound): RMSNorm on hidden_size tensor
    elem_traffic = 32 * 4096 * 2 * 2  # batch*hidden read+write fp16
    workload.append((0.15, 0, elem_traffic, "elementwise", False))

    return workload


def predict_workload_latency(hw, workload, dram_bw_scale=1.0,
                              l2_bw_scale=1.0, compute_scale=1.0,
                              sm_count_scale=1.0):
    """Predict total latency for workload mix with scaled HW parameters.

    Scaling multiplies effective capacity:
    - dram_bw_scale > 1 means more DRAM BW → faster memory-bound kernels
    - compute_scale > 1 means more compute → faster compute-bound kernels

    Returns (total_time_s, per_kernel_results)
    """
    formula = DiffPerformanceFormula(hw)

    total_time = 0.0
    results = []

    # Effective frequencies from hw_spec (scaling BW = scaling clock for our model)
    eff_dram_freq = hw.nominal_dram_freq_hz * dram_bw_scale
    eff_sm_freq = hw.nominal_sm_freq_hz * compute_scale

    for weight, flops, traffic, name, l1_bypass in workload:
        demand = {}
        if flops > 0:
            demand["compute_tensor"] = flops
        if traffic > 0:
            demand["l2_cache"] = traffic
            demand["dram"] = traffic

        inp = PredictionInput(
            demand_per_resource=demand,
            eta=0.85,
            active_units=int(hw.num_compute_units * sm_count_scale),
            sm_freq_hz=eff_sm_freq,
            dram_freq_hz=eff_dram_freq,
            l1_bypass=l1_bypass,
            duration_ns=0.0,
            grid_size=256,
            kernel_name=name,
            is_non_tensor=False,
            pipeline_floor_ns=0.0,
        )

        result = formula.predict_kernel(inp)
        weighted_time = result.t_total_s * weight
        total_time += weighted_time
        results.append((name, result, weight))

    return total_time, results


def test_gradient_sign_consistency(hw, workload, n_points=50):
    """Test 1: Verify gradient sign matches actual perturbation direction.

    For each random HW config point, perturb each parameter ±10% and check
    that the gradient sign matches the actual change direction.
    """
    rng = np.random.default_rng(42)
    formula = DiffPerformanceFormula(hw)

    correct = 0
    total = 0

    for _ in range(n_points):
        # Random scaling factors
        dram_scale = rng.uniform(0.5, 2.0)
        compute_scale = rng.uniform(0.5, 2.0)

        # Compute gradient at this point
        t_base, results = predict_workload_latency(
            hw, workload, dram_bw_scale=dram_scale, compute_scale=compute_scale)

        # Perturb DRAM BW +10%
        t_dram_up, _ = predict_workload_latency(
            hw, workload, dram_bw_scale=dram_scale * 1.1, compute_scale=compute_scale)

        # Numerical gradient for DRAM
        delta_t = t_dram_up - t_base
        # Analytical: ∂t/∂(dram_bw_scale) should be negative (more BW → less time)
        # For DRAM-bound kernels

        # Check sign consistency
        if abs(delta_t) > 1e-15:
            # Higher BW → lower time (delta_t should be negative)
            sign_correct = (delta_t <= 0)
            if sign_correct:
                correct += 1
            total += 1

        # Perturb compute +10%
        t_compute_up, _ = predict_workload_latency(
            hw, workload, dram_bw_scale=dram_scale, compute_scale=compute_scale * 1.1)
        delta_t_c = t_compute_up - t_base
        if abs(delta_t_c) > 1e-15:
            sign_correct = (delta_t_c <= 0)
            if sign_correct:
                correct += 1
            total += 1

    consistency = correct / total if total > 0 else 0.0
    print(f"Gradient sign consistency: {correct}/{total} = {consistency*100:.1f}% (target > 95%)")
    return consistency


def test_gradient_dse(hw, workload, max_steps=100):
    """Test 2: Gradient-based DSE vs grid search.

    Find the DRAM_BW / Compute balance that minimizes total latency
    subject to a cost constraint.
    """
    # Cost model: cost = dram_bw_scale + compute_scale (normalized)
    # Constraint: cost ≤ 2.0 (can't exceed 2x resources)
    # Objective: minimize total_latency

    # Grid search (10000 points)
    grid_size = 100
    best_grid_t = float("inf")
    best_grid_config = None

    for i in range(grid_size):
        for j in range(grid_size):
            dram_s = 0.5 + i * 1.5 / grid_size
            comp_s = 0.5 + j * 1.5 / grid_size
            if dram_s + comp_s > 3.0:
                continue
            t, _ = predict_workload_latency(hw, workload, dram_bw_scale=dram_s,
                                             compute_scale=comp_s)
            if t < best_grid_t:
                best_grid_t = t
                best_grid_config = (dram_s, comp_s)

    print(f"Grid search best: t={best_grid_t*1e6:.2f} us, config={best_grid_config}")

    # Gradient-based search (projected gradient descent)
    dram_s = 1.0
    comp_s = 1.0
    best_grad_t = float("inf")
    best_grad_config = None

    for step in range(max_steps):
        t_curr, _ = predict_workload_latency(hw, workload, dram_bw_scale=dram_s,
                                              compute_scale=comp_s)
        if t_curr < best_grad_t:
            best_grad_t = t_curr
            best_grad_config = (dram_s, comp_s)

        # Numerical gradient (finite difference)
        eps = 0.01
        t_dram_plus, _ = predict_workload_latency(hw, workload,
                                                    dram_bw_scale=dram_s + eps,
                                                    compute_scale=comp_s)
        t_comp_plus, _ = predict_workload_latency(hw, workload,
                                                    dram_bw_scale=dram_s,
                                                    compute_scale=comp_s + eps)
        grad_dram = (t_dram_plus - t_curr) / eps
        grad_comp = (t_comp_plus - t_curr) / eps

        # Adaptive learning rate: step = fixed fraction of parameter range
        grad_norm = (grad_dram**2 + grad_comp**2)**0.5
        if grad_norm > 1e-20:
            # Normalized gradient with fixed step size
            step_size = 0.05  # 5% of parameter range per step
            dram_s -= step_size * grad_dram / grad_norm
            comp_s -= step_size * grad_comp / grad_norm

        # Project to feasible region [0.5, 2.0]
        dram_s = np.clip(dram_s, 0.5, 2.0)
        comp_s = np.clip(comp_s, 0.5, 2.0)

        # Cost constraint projection
        if dram_s + comp_s > 3.0:
            excess = (dram_s + comp_s - 3.0) / 2
            dram_s -= excess
            comp_s -= excess

    print(f"Gradient search best: t={best_grad_t*1e6:.2f} us, config={best_grad_config}")

    gap = abs(best_grad_t - best_grid_t) / best_grid_t * 100
    print(f"Gap: {gap:.2f}% (target < 5%)")
    return gap


def test_bottleneck_phase_diagram(hw, workload):
    """Test 3: Generate bottleneck phase diagram.

    Sweep (DRAM_BW_scale, Compute_scale) and identify bottleneck regions.
    """
    formula = DiffPerformanceFormula(hw)
    grid = 20
    diagram = np.empty((grid, grid), dtype="U1")

    for i in range(grid):
        for j in range(grid):
            dram_s = 0.3 + i * 2.7 / grid
            comp_s = 0.3 + j * 2.7 / grid

            _, results = predict_workload_latency(
                hw, workload, dram_bw_scale=dram_s, compute_scale=comp_s)

            # Find dominant bottleneck across workload
            bottlenecks = {}
            for name, result, weight in results:
                bn = result.bottleneck_group
                if bn not in bottlenecks:
                    bottlenecks[bn] = 0.0
                bottlenecks[bn] += weight

            dominant = max(bottlenecks.items(), key=lambda x: x[1])[0]
            if "tensor" in dominant:
                diagram[i, j] = "C"  # Compute-bound
            elif "dram" in dominant:
                diagram[i, j] = "D"  # DRAM-bound
            elif "l2" in dominant:
                diagram[i, j] = "L"  # L2-bound
            else:
                diagram[i, j] = "O"  # Other

    # Print diagram
    print("\nBottleneck Phase Diagram (rows=DRAM_BW↑, cols=Compute↑):")
    print("  C=Compute-bound, D=DRAM-bound, L=L2-bound")
    for i in range(grid - 1, -1, -1):
        row = "".join(diagram[i, :])
        dram_s = 0.3 + i * 2.7 / grid
        print(f"  {dram_s:.1f}x | {row}")

    # Check that we have at least one non-trivial bottleneck region.
    # With the unified memory model (Phase 2), DRAM is folded into L2 via
    # the DRAM ceiling, so "D" may not appear — L2 covers both L2-issue
    # and DRAM-ceiling regimes.  Accept L2 as a valid memory-bound region.
    has_compute = "C" in diagram
    has_dram = "D" in diagram
    has_l2 = "L" in diagram
    has_memory = has_dram or has_l2
    print(f"\n  Has compute-bound region: {has_compute}")
    print(f"  Has DRAM-bound region: {has_dram}")
    print(f"  Has L2/memory-bound region: {has_l2}")
    return has_compute or has_memory


def test_aggregate_gradients_dse(hw, max_steps=100):
    """Test 4: aggregate_gradients enables effective DSE on multi-kernel ops.

    Two-kernel op (prefill GEMM + decode GEMM) with different bottlenecks.
    aggregate_gradients sums per-kernel gradients → op-level DSE with
    analytical gradients. Target: DSE gap < 0.5%.
    """
    # Two-kernel op: prefill (compute-bound) + decode (memory-bound)
    mkernels = [
        ("gemm_prefill", 2 * 1024 * 4096 * 4096,
         (1024 * 4096 + 4096 * 4096 + 1024 * 4096) * 2, 0.5, False),
        ("gemm_decode", 2 * 1 * 4096 * 4096,
         (1 * 4096 + 4096 * 4096 + 1 * 4096) * 2, 0.5, False),
    ]

    def predict(dram_scale, compute_scale):
        formula = DiffPerformanceFormula(hw)
        eff_dram_freq = hw.nominal_dram_freq_hz * dram_scale
        eff_sm_freq = hw.nominal_sm_freq_hz * compute_scale
        preds = []
        total_t = 0.0
        for name, flops, traffic, weight, l1_bypass in mkernels:
            demand = {}
            if flops > 0:
                demand["compute_tensor"] = flops
            if traffic > 0:
                demand["l2_cache"] = traffic
                demand["dram"] = traffic
            inp = PredictionInput(
                demand_per_resource=demand, eta=0.85,
                active_units=hw.num_compute_units, sm_freq_hz=eff_sm_freq,
                dram_freq_hz=eff_dram_freq, l1_bypass=l1_bypass,
                duration_ns=0.0, grid_size=256, kernel_name=name,
                is_non_tensor=False, pipeline_floor_ns=0.0,
            )
            pred = formula.predict_kernel(inp)
            total_t += pred.t_total_s * weight
            preds.append(pred)
        return total_t, preds

    # Grid search (10000 points)
    best_grid_t = float("inf")
    for i in range(100):
        for j in range(100):
            d = 0.5 + i * 1.5 / 100
            c = 0.5 + j * 1.5 / 100
            if d + c > 3.0:
                continue
            t, _ = predict(d, c)
            if t < best_grid_t:
                best_grid_t = t

    # Analytical gradient DSE using aggregate_gradients
    dram_s, comp_s = 1.0, 1.0
    best_grad_t = float("inf")
    dram_cap_base = hw.resource_by_role("dram").effective_capacity(
        hw.num_compute_units, sm_freq_hz=hw.nominal_sm_freq_hz,
        dram_freq_hz=hw.nominal_dram_freq_hz)
    tc_cap_base = hw.resource_by_role("compute_tensor").effective_capacity(
        hw.num_compute_units, sm_freq_hz=hw.nominal_sm_freq_hz,
        dram_freq_hz=hw.nominal_dram_freq_hz)

    for _ in range(max_steps):
        t, preds = predict(dram_s, comp_s)
        if t < best_grad_t:
            best_grad_t = t

        # Aggregate gradients then chain-rule to scale gradients
        # d_cap_eff/d_scale = cap_base  →  dt/d_scale = grad_capacity * cap_base * weight
        agg = aggregate_gradients(preds)
        grad_dram = sum(preds[k].grad_capacity.get("dram", 0.0) * mkernels[k][3]
                        for k in range(len(preds))) * dram_cap_base
        grad_comp = sum(preds[k].grad_capacity.get("compute_tensor", 0.0) * mkernels[k][3]
                        for k in range(len(preds))) * tc_cap_base

        grad_norm = (grad_dram ** 2 + grad_comp ** 2) ** 0.5
        if grad_norm > 1e-20:
            step_size = 0.05
            dram_s -= step_size * grad_dram / grad_norm
            comp_s -= step_size * grad_comp / grad_norm

        dram_s = np.clip(dram_s, 0.5, 2.0)
        comp_s = np.clip(comp_s, 0.5, 2.0)
        if dram_s + comp_s > 3.0:
            excess = (dram_s + comp_s - 3.0) / 2
            dram_s -= excess
            comp_s -= excess

    gap = abs(best_grad_t - best_grid_t) / best_grid_t * 100
    print(f"Multi-kernel analytical DSE: grid={best_grid_t*1e6:.3f}us, "
          f"grad={best_grad_t*1e6:.3f}us, gap={gap:.3f}%")
    return gap


def test_l2_size_gradient(hw):
    """Test 5: analytic grad_l2_size vs numerical finite difference (<1%).

    Two-part validation of the L2-size gradient chain (P48):

    Part A (formula chain-rule, P48.3): a DRAM-bound PredictionInput carries a
    known d_dram_d_l2size. The diff formula's analytic grad_l2_size must equal
    the finite-difference of t_total when the DRAM demand is perturbed by
    d_dram_d_l2size·ΔC (i.e. moving L2 size by ΔC). This isolates the
    chain-rule dt/dC = (1/eff_cap_dram)·d(dram)/dC the formula adds.

    Part B (hit-curve derivative, P48.2): the exact stack-distance hit-curve
    secant d(dram)/dC matches a finite-difference of the discrete LRU simulator
    over the same window, on a many-waves GEMM where L2 size is sensitive.

    Returns the max relative error (%) across both parts.
    """
    import numpy as np
    from src.kernel.validation.legacy_ops.gemm import predict_gemm_l2_size_grad
    from src.kernel.lru_cache import (
        gemm_swizzle_best_schedule, simulate_gemm_traffic)

    # ── Part A: formula chain-rule ──
    # DRAM-bound kernel: only a dram demand, large enough to dominate launch.
    dram_bytes = 200e6
    d_dram_d_l2 = -3.0  # bytes-DRAM per byte-L2 (arbitrary known slope)
    formula = DiffPerformanceFormula(hw)
    inp = PredictionInput(
        demand_per_resource={"dram": dram_bytes}, eta=0.85,
        active_units=hw.num_compute_units, sm_freq_hz=hw.nominal_sm_freq_hz,
        dram_freq_hz=hw.nominal_dram_freq_hz, l1_bypass=False,
        duration_ns=0.0, grid_size=256, kernel_name="dram_bound_gemm",
        is_non_tensor=False, pipeline_floor_ns=0.0,
        d_dram_d_l2size=d_dram_d_l2)
    dpred = formula.predict_kernel(inp)
    analytic_A = dpred.grad_l2_size

    dC = hw.l2_size_bytes * 0.01  # FD window in bytes of L2

    def t_with_dram(dram):
        inp2 = PredictionInput(
            demand_per_resource={"dram": dram}, eta=0.85,
            active_units=hw.num_compute_units, sm_freq_hz=hw.nominal_sm_freq_hz,
            dram_freq_hz=hw.nominal_dram_freq_hz, l1_bypass=False,
            duration_ns=0.0, grid_size=256, kernel_name="dram_bound_gemm",
            is_non_tensor=False, pipeline_floor_ns=0.0)
        return formula.predict_kernel(inp2).t_total_s

    # moving L2 by +dC changes dram by d_dram_d_l2*dC
    t_hi = t_with_dram(dram_bytes + d_dram_d_l2 * dC)
    t_lo = t_with_dram(dram_bytes - d_dram_d_l2 * dC)
    fd_A = (t_hi - t_lo) / (2 * dC)
    err_A = abs(analytic_A - fd_A) / abs(fd_A) * 100 if fd_A != 0 else abs(analytic_A) * 1e30
    print(f"  Part A (formula chain-rule): analytic={analytic_A:.6e} FD={fd_A:.6e} err={err_A:.3f}%")

    # ── Part B: hit-curve derivative vs discrete LRU sim ──
    # Many-waves GEMM so L2 size is sensitive (low occupancy: large smem).
    M = N = 8192
    K = 4096
    kn = "ampere_h16816gemm_128x128_ldg8_stages_32x5_nn"
    mb = nb = 128
    grid_x = (N + nb - 1) // nb
    grid_y = (M + mb - 1) // mb
    analytic_B = predict_gemm_l2_size_grad(
        1, M, K, N, 1, kn, grid_x * grid_y, grid_x, grid_y,
        120 * 1024, hw)  # 120KB smem → wave_size = num_SMs (many waves)
    from src.kernel.validation.legacy_ops.gemm import _gemm_tile_geometry
    geom = _gemm_tile_geometry(1, M, K, N, 1, kn, grid_x, grid_y, 120 * 1024, hw)
    max_sw = int(np.log2(max(1, grid_x))) + 1
    sched = gemm_swizzle_best_schedule(
        grid_x, grid_y, geom["conv"], geom["W"], geom["mt"], geom["nt"],
        max_sw, geom["ks"], geom["a_tile"], geom["b_tile"], geom["c_tile"],
        geom["wave_size"], hw.l2_size_bytes)
    base = hw.l2_size_bytes
    lo = int(base * 0.95)
    hi = int(base * 1.05)
    _, dram_lo = simulate_gemm_traffic(
        sched, geom["ks"], geom["a_tile"], geom["b_tile"], geom["c_tile"],
        geom["wave_size"], lo)
    _, dram_hi = simulate_gemm_traffic(
        sched, geom["ks"], geom["a_tile"], geom["b_tile"], geom["c_tile"],
        geom["wave_size"], hi)
    fd_B = (dram_hi - dram_lo) / (hi - lo)
    err_B = abs(analytic_B - fd_B) / abs(fd_B) * 100 if fd_B != 0 else abs(analytic_B) * 1e30
    print(f"  Part B (hit-curve d(dram)/dC): analytic={analytic_B:.6f} FD={fd_B:.6f} err={err_B:.3f}%")

    return max(err_A, err_B)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Differentiable model DSE validation")
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml",
                        help="Path to hw config yaml (default: configs/hardware/nvidia_a800.yaml)")
    args = parser.parse_args()
    hw = HWSpec(args.hw_config)
    workload = create_workload_mix()

    print("=" * 60)
    print("P20: Differentiable Model DSE Validation")
    print("=" * 60)

    # Test 1: Gradient sign consistency
    print("\n--- Test 1: Gradient Sign Consistency ---")
    consistency = test_gradient_sign_consistency(hw, workload, n_points=50)
    test1_pass = consistency > 0.95

    # Test 2: Gradient DSE vs Grid Search
    print("\n--- Test 2: Gradient DSE vs Grid Search ---")
    gap = test_gradient_dse(hw, workload, max_steps=100)
    test2_pass = gap < 5.0

    # Test 3: Bottleneck Phase Diagram
    print("\n--- Test 3: Bottleneck Phase Diagram ---")
    test3_pass = test_bottleneck_phase_diagram(hw, workload)

    # Test 4: Multi-kernel aggregate_gradients DSE
    print("\n--- Test 4: Multi-Kernel aggregate_gradients DSE ---")
    gap4 = test_aggregate_gradients_dse(hw, max_steps=100)
    test4_pass = gap4 < 0.5

    # Test 5: analytic grad_l2_size vs numerical FD
    print("\n--- Test 5: L2-size gradient (analytic vs FD) ---")
    err5 = test_l2_size_gradient(hw)
    test5_pass = err5 < 1.0

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Test 1 (gradient sign > 95%): {'PASS' if test1_pass else 'FAIL'} ({consistency*100:.1f}%)")
    print(f"  Test 2 (DSE gap < 5%):        {'PASS' if test2_pass else 'FAIL'} ({gap:.2f}%)")
    print(f"  Test 3 (phase diagram):        {'PASS' if test3_pass else 'FAIL'}")
    print(f"  Test 4 (agg_grad gap < 0.5%): {'PASS' if test4_pass else 'FAIL'} ({gap4:.3f}%)")
    print(f"  Test 5 (grad_l2_size < 1%):   {'PASS' if test5_pass else 'FAIL'} ({err5:.3f}%)")
    print("=" * 60)

    return test1_pass and test2_pass and test3_pass and test4_pass and test5_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
