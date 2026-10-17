"""Hardware Design Space Exploration via gradient-based search.

Runs projected gradient descent on {dram_bw_scale, compute_scale} ∈ [0.5, 2.0]²
to find the hardware configuration that minimizes end-to-end GPU kernel latency
for an LLM inference workload.

Analytical gradients are derived from DiffKernelPrediction objects collected by
ModelSimulator.predict_diff(). The chain rule from grad_capacity to ∂t/∂scale:

  t_kernel = demand / (cap_base × scale)
  ∂t/∂cap  = -demand / (cap × cap)            (from DiffKernelPrediction.grad_capacity)
  ∂t/∂scale = ∂t/∂cap × ∂cap/∂scale
  ∂cap/∂scale = cap_base   (capacity = cap_base × scale, linear)
  ∴ ∂t/∂scale = grad_capacity[role] × cap_base

The gradient for t_e2e is the sum over all kernels (via aggregate_gradients).

Usage:
  python scripts/hw_dse.py \\
      --model /share_data/public_models/Qwen3-0.6B \\
      --hw-config configs/hw/A800-SXM4-80GB.yaml \\
      --batch 1 --seq-len 128 --n-decode 8

Notes:
  - Does NOT modify configs/hw/*.yaml files; only prints recommendations.
  - l2_size_scale and NVLink are not differentiable parameters in this version;
    both are excluded from the DSE search (see deferred items).
"""

import sys
import argparse
import math

sys.path.insert(0, ".")

import numpy as np
from src.serve_predict.model_config import load_model_config
from src.kernel.hw_spec import HWSpec
from src.dse.model_sim import ModelSimulator
from src.dse.diff_formula import aggregate_gradients
from src.kernel.kernel_base import create_predictor
from dataclasses import replace as _dc_replace


def _predict_with_scales(sim, model_config, hw_config_path,
                          dram_bw_scale, compute_scale,
                          batch, seq_len, n_decode, tp_size):
    """Predict (t_e2e_gpu_ns, agg_grad) with scaled HW parameters.

    Scaling is applied by overriding the effective SM and DRAM clock
    frequencies in each PredictionInput before calling DiffPerformanceFormula.
    This is linear in capacities → ∂cap/∂scale = cap_base (constant).

    Raises:
        ValueError: if dram_bw_scale <= 0 or compute_scale <= 0
    """
    return sim.predict_diff_with_scaled_hw(
        model_config, hw_config_path,
        dram_bw_scale=dram_bw_scale,
        compute_scale=compute_scale,
        batch=batch, seq_len=seq_len,
        n_decode=n_decode, tp_size=tp_size,
    )


def _compute_scale_gradients(agg_grad, hw, dram_bw_scale, compute_scale):
    """Convert grad_capacity to ∂t/∂scale via correct per-kernel chain rule.

    Uses grad_capacity_scaled from aggregate_gradients(), divided by the
    current scale value.

    Derivation at current scale s:
      eff_cap_k(s) = cap_nominal_k × s × f(active_units_k, eta_k)
      grad_capacity_k(s) = -demand_k / eff_cap_k(s)^2
      grad_capacity_scaled(s) = sum_k(grad_capacity_k × eff_cap_k) = -sum_k(t_resource_k(s))
      dt_k/ds = -t_resource_k(s) / s  (from d/ds [demand/(cap_nom × s × f)] = -t_k/s)
      dt/ds = grad_capacity_scaled(s) / s

    Returns (grad_dram_scale, grad_compute_scale).
    Both expected negative for physical workloads.
    """
    if "grad_capacity_scaled" in agg_grad:
        grad_dram_scale = (agg_grad["grad_capacity_scaled"].get("dram", 0.0)
                          / dram_bw_scale)
        grad_compute_scale = (agg_grad["grad_capacity_scaled"].get("compute_tensor", 0.0)
                             / compute_scale)
    else:
        dram_res = hw.resource_by_role("dram")
        tc_res = hw.resource_by_role("compute_tensor")
        dram_cap_base = dram_res.effective_capacity(
            hw.num_compute_units,
            sm_freq_hz=hw.nominal_sm_freq_hz,
            dram_freq_hz=hw.nominal_dram_freq_hz,
        )
        tc_cap_base = tc_res.effective_capacity(
            hw.num_compute_units,
            sm_freq_hz=hw.nominal_sm_freq_hz,
            dram_freq_hz=hw.nominal_dram_freq_hz,
        )
        grad_dram_scale = agg_grad["grad_capacity"].get("dram", 0.0) * dram_cap_base
        grad_compute_scale = agg_grad["grad_capacity"].get("compute_tensor", 0.0) * tc_cap_base

    return grad_dram_scale, grad_compute_scale


def grid_search(sim, model_config, hw_config_path, hw,
                batch, seq_len, n_decode, tp_size,
                grid_size=10, lo=0.5, hi=2.0):
    """Brute-force grid search over {dram_bw_scale, compute_scale}.

    Returns (best_t_ns, best_dram_scale, best_compute_scale).
    """
    best_t = float("inf")
    best_cfg = (1.0, 1.0)
    step = (hi - lo) / (grid_size - 1) if grid_size > 1 else 0
    for i in range(grid_size):
        dram_s = lo + i * step
        for j in range(grid_size):
            comp_s = lo + j * step
            t_ns, _ = _predict_with_scales(
                sim, model_config, hw_config_path,
                dram_s, comp_s, batch, seq_len, n_decode, tp_size)
            if t_ns < best_t:
                best_t = t_ns
                best_cfg = (dram_s, comp_s)
    return best_t, best_cfg[0], best_cfg[1]


def gradient_descent(sim, model_config, hw_config_path, hw,
                     batch, seq_len, n_decode, tp_size,
                     max_steps=100, step_size=0.05, lo=0.5, hi=2.0):
    """Projected gradient descent on {dram_bw_scale, compute_scale}.

    Uses analytical gradients from predict_diff_with_scaled_hw() for each step.
    Projection to [lo, hi]² applied after each update.

    Returns:
        (best_t_ns, trajectory)  where trajectory is list of
        (step, dram_s, comp_s, t_ns) tuples.
    """
    dram_s = 1.0
    comp_s = 1.0
    best_t = float("inf")
    best_cfg = (dram_s, comp_s)
    trajectory = []

    for step in range(max_steps):
        t_ns, agg_grad = _predict_with_scales(
            sim, model_config, hw_config_path,
            dram_s, comp_s, batch, seq_len, n_decode, tp_size)

        if t_ns < best_t:
            best_t = t_ns
            best_cfg = (dram_s, comp_s)

        trajectory.append((step, dram_s, comp_s, t_ns))

        grad_dram, grad_comp = _compute_scale_gradients(agg_grad, hw, dram_s, comp_s)
        grad_norm = math.sqrt(grad_dram ** 2 + grad_comp ** 2)

        if grad_norm < 1e-20:
            import logging
            logging.warning(
                "no differentiable resource bottleneck found — "
                "all kernels appear launch-overhead-bound; gradients are zero")
            break

        dram_s -= step_size * grad_dram / grad_norm
        comp_s -= step_size * grad_comp / grad_norm

        dram_s = float(np.clip(dram_s, lo, hi))
        comp_s = float(np.clip(comp_s, lo, hi))

    return best_t, best_cfg[0], best_cfg[1], trajectory


def print_gradient_table(agg_grad, hw, dram_bw_scale=1.0, compute_scale=1.0):
    """Print per-resource gradient magnitudes sorted by bottleneck contribution.

    Uses grad_capacity_scaled / current_scale for correct dt/d(scale) values.
    """
    # Prefer corrected per-kernel scale gradient; fall back to global cap approx
    grad_cap_scaled = agg_grad.get("grad_capacity_scaled", {})
    grad_cap = agg_grad.get("grad_capacity", {})

    scale_map = {"dram": dram_bw_scale, "compute_tensor": compute_scale}

    all_roles = set(grad_cap_scaled.keys()) | set(grad_cap.keys())
    rows = []
    for role in all_roles:
        if role in grad_cap_scaled:
            current_scale = scale_map.get(role, 1.0)
            sg = grad_cap_scaled[role] / current_scale  # dt/d(scale) in seconds
            g = grad_cap.get(role, 0.0)
        elif role in grad_cap and grad_cap[role] != 0.0:
            g = grad_cap[role]
            try:
                res = hw.resource_by_role(role)
                cap = res.effective_capacity(
                    hw.num_compute_units,
                    sm_freq_hz=hw.nominal_sm_freq_hz,
                    dram_freq_hz=hw.nominal_dram_freq_hz,
                )
                sg = g * cap
            except Exception:
                sg = g
        else:
            continue
        if sg != 0.0:
            rows.append((role, g, sg))

    rows.sort(key=lambda x: -abs(x[2]))

    print(f"  {'Resource':<25} {'grad_capacity':>14}  {'dt/d_scale (s)':>16}  note")
    print(f"  {'-'*25} {'-'*14}  {'-'*16}  {'-'*20}")
    for role, g, sg in rows:
        note = "bottleneck" if sg < 0 else "non-bottleneck"
        print(f"  {role:<25} {g:>14.4e}  {sg:>16.4e}  {note}")


def main():
    parser = argparse.ArgumentParser(
        description="Gradient-based hardware DSE on dram_bw_scale + compute_scale")
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B",
                        help="Path to HuggingFace model directory")
    parser.add_argument("--hw-config", default="configs/hw/A800-SXM4-80GB.yaml",
                        help="Path to hardware config YAML")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-decode", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100,
                        help="Gradient descent steps (default: 100)")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Grid search dimension (default: 10×10)")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--lo", type=float, default=0.5)
    parser.add_argument("--hi", type=float, default=2.0)
    args = parser.parse_args()

    model_config = load_model_config(args.model)
    hw = HWSpec(args.hw_config)
    sim = ModelSimulator(args.hw_config)

    print("=" * 60)
    print("Hardware DSE: gradient-based dram_bw_scale + compute_scale")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"HW config:   {args.hw_config}")
    print(f"Workload:    batch={args.batch} seq_len={args.seq_len} "
          f"n_decode={args.n_decode} tp={args.tp_size}")
    print(f"Search:      [{args.lo}, {args.hi}]² x {args.steps} steps")
    print()

    # Baseline prediction
    t_base, agg_base = _predict_with_scales(
        sim, model_config, args.hw_config,
        1.0, 1.0, args.batch, args.seq_len, args.n_decode, args.tp_size)
    print(f"Baseline (scale=1.0):  t_e2e = {t_base/1e6:.3f} ms")
    print()

    # Check for launch-bound-only workload
    grad_dram_base, grad_comp_base = _compute_scale_gradients(agg_base, hw, 1.0, 1.0)
    if abs(grad_dram_base) < 1e-25 and abs(grad_comp_base) < 1e-25:
        import logging
        print("WARNING: no differentiable resource bottleneck found — "
              "all kernels appear launch-overhead-bound; gradients are zero")
        print("Hardware DSE cannot optimize a launch-bound workload.")
        return

    print("Gradient table at baseline (scale=1.0):")
    print_gradient_table(agg_base, hw, dram_bw_scale=1.0, compute_scale=1.0)
    print()

    # Grid search
    print(f"Running {args.grid_size}×{args.grid_size} grid search...")
    t_grid, dram_grid, comp_grid = grid_search(
        sim, model_config, args.hw_config, hw,
        args.batch, args.seq_len, args.n_decode, args.tp_size,
        grid_size=args.grid_size, lo=args.lo, hi=args.hi)
    print(f"Grid search:   t_e2e = {t_grid/1e6:.3f} ms "
          f"at dram_bw_scale={dram_grid:.2f}, compute_scale={comp_grid:.2f}")
    print()

    # Gradient descent
    print(f"Running {args.steps}-step projected gradient descent...")
    t_grad, dram_grad, comp_grad, trajectory = gradient_descent(
        sim, model_config, args.hw_config, hw,
        args.batch, args.seq_len, args.n_decode, args.tp_size,
        max_steps=args.steps, step_size=args.step_size,
        lo=args.lo, hi=args.hi)
    print(f"Gradient DSE:  t_e2e = {t_grad/1e6:.3f} ms "
          f"at dram_bw_scale={dram_grad:.2f}, compute_scale={comp_grad:.2f}")
    print()

    # Monotonicity check
    first_20 = [t for _, _, _, t in trajectory[:20]]
    is_monotone = all(first_20[i] >= first_20[i + 1] for i in range(len(first_20) - 1))
    print(f"Trajectory monotone (first 20 steps): {is_monotone}")

    # Convergence gap
    if t_grid > 0:
        gap_pct = abs(t_grad - t_grid) / t_grid * 100
        print(f"Convergence gap vs grid:  {gap_pct:.2f}%  (target < 2%)")

    print()
    print("Recommended parameter changes (vs A800 baseline):")
    print(f"  dram_bw_scale = {dram_grad:.3f}  "
          f"({'increase DRAM BW' if dram_grad > 1.0 else 'decrease DRAM BW'})")
    print(f"  compute_scale = {comp_grad:.3f}  "
          f"({'increase compute' if comp_grad > 1.0 else 'decrease compute'})")
    print()
    print("Note: These are relative scale factors on current HW capacities.")
    print("      configs/hw/*.yaml files are NOT modified.")

    return t_grad, t_grid, trajectory


if __name__ == "__main__":
    main()
