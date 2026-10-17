"""Hardware DSE from real vLLM emulation trace via gradient-based search.

Runs projected gradient descent on {dram_bw_scale, compute_scale} in [0.5, 2.0]^2
to find the hardware configuration that minimizes the GPU kernel latency sum for
a captured LLM inference trace.

Uses TracePredictor.annotate_trace_diff() + aggregate_gradients() to compute
analytical gradients of the sequential per-kernel GPU latency sum w.r.t.
hardware scale factors.  Uses TracePredictor + DiffPerformanceFormula directly.

Chain rule from grad_capacity_scaled to dt/d_scale:
  grad_capacity_scaled[role] = sum_k(grad_capacity_k * eff_cap_k) = -sum_k(t_resource_k)
  dt/d_scale = grad_capacity_scaled[role] / current_scale

Usage:
  python scripts/dse/run_dse.py \\
      --trace /path/to/emulation_trace.json \\
      --model /share_data/public_models/Qwen3-0.6B \\
      --hw-config configs/hardware/nvidia_a800.yaml \\
      --seq-len 512 --mode gradient

Notes:
  - Does NOT modify configs/hw/*.yaml files; only prints recommendations.
  - l2_size_scale and NVLink are not differentiable parameters in this version.
"""

import sys
import os
import argparse
import json
import math
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from src.serve_predict.predict_kernel_time_from_trace import TracePredictor
from src.serve_predict.model_config import load_model_config
from src.kernel.hardware_spec import HWSpec
from src.dse.diff_formula import aggregate_gradients


# ---------------------------------------------------------------------------
# Gradient descent + grid search (ported from archive hw_dse.py, adapted for
# real-trace TracePredictor + DiffPerformanceFormula)
# ---------------------------------------------------------------------------

def _predict_with_scales(predictor, trace, seq_len, hw,
                          dram_bw_scale, compute_scale,
                          batch_size, tp_size):
    """Predict (t_e2e_gpu_ns, agg_grad) with scaled HW parameters.

    Passes sm_freq_override and dram_freq_override to annotate_trace_diff(),
    which forwards them to _predict_with_midlayer_diff().  Each
    PredictionInput.sm_freq_hz / dram_freq_hz is overridden before
    DiffPerformanceFormula.predict_kernel() runs, so the formulas and
    gradients reflect the scaled hardware.

    Raises:
        ValueError: if dram_bw_scale <= 0 or compute_scale <= 0
    """
    if dram_bw_scale <= 0:
        raise ValueError(f"dram_bw_scale must be > 0, got {dram_bw_scale}")
    if compute_scale <= 0:
        raise ValueError(f"compute_scale must be > 0, got {compute_scale}")

    preds = predictor.annotate_trace_diff(
        trace, seq_len, batch_size=batch_size, tp_size=tp_size,
        sm_freq_override=predictor.clock_hz * compute_scale,
        dram_freq_override=predictor.dram_clock_hz * dram_bw_scale,
    )
    agg = aggregate_gradients(preds)
    t_ns = agg["t_total_s"] * 1e9
    return t_ns, agg


def _compute_scale_gradients(agg_grad, hw, dram_bw_scale, compute_scale):
    """Convert grad_capacity to dt/d_scale via correct per-kernel chain rule.

    At current scale s:
      eff_cap_k(s) = cap_nominal_k * s * f(active_units_k, eta_k)
      grad_capacity_scaled(s) = sum_k(grad_capacity_k * eff_cap_k) = -sum_k(t_resource_k)
      dt/ds = grad_capacity_scaled(s) / s

    Returns (grad_dram_scale, grad_compute_scale). Both expected negative for
    physical workloads.
    """
    grad_cap_scaled = agg_grad.get("grad_capacity_scaled", {})

    # Saturated DRAM capacity scales with DRAM clock.  Unsaturated DRAM
    # capacity scales with SM clock because the DRAM clock term cancels
    # through the concurrency factor.
    if ("dram_scale_grad_capacity_scaled" in agg_grad
            or "compute_scale_extra_grad_capacity_scaled" in agg_grad):
        grad_dram_scale = (
            agg_grad.get("dram_scale_grad_capacity_scaled", 0.0)
            / dram_bw_scale
        )
        grad_compute_extra = (
            agg_grad.get("compute_scale_extra_grad_capacity_scaled", 0.0)
            / compute_scale
        )
    else:
        grad_dram_scale = grad_cap_scaled.get("dram", 0.0) / dram_bw_scale
        grad_compute_extra = 0.0

    # compute_scale affects all SM-clocked resources (not just compute_tensor).
    # Sum grad_capacity_scaled across all roles whose clock_domain == "sm".
    grad_compute_scale = grad_compute_extra
    for res in hw.resources:
        if res.clock_domain == "sm" and res.role in grad_cap_scaled:
            grad_compute_scale += grad_cap_scaled[res.role] / compute_scale

    return grad_dram_scale, grad_compute_scale


def grid_search(predictor, trace, seq_len, hw,
                batch_size, tp_size,
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
                predictor, trace, seq_len, hw,
                dram_s, comp_s, batch_size, tp_size)
            if t_ns < best_t:
                best_t = t_ns
                best_cfg = (dram_s, comp_s)
    return best_t, best_cfg[0], best_cfg[1]


def gradient_descent(predictor, trace, seq_len, hw,
                     batch_size, tp_size,
                     max_steps=100, step_size=0.05, lo=0.5, hi=2.0,
                     init_dram_scale=1.0, init_compute_scale=1.0):
    """Projected gradient descent on {dram_bw_scale, compute_scale}.

    Uses analytical gradients from aggregate_gradients() for each step.
    Projection to [lo, hi]^2 applied after each update.

    Returns:
        (best_t_ns, best_dram_scale, best_compute_scale, trajectory)
        where trajectory is list of (step, dram_s, comp_s, t_ns) tuples.
    """
    dram_s = float(np.clip(init_dram_scale, lo, hi))
    comp_s = float(np.clip(init_compute_scale, lo, hi))
    best_t = float("inf")
    best_cfg = (dram_s, comp_s)
    trajectory = []

    for step in range(max_steps):
        t_ns, agg_grad = _predict_with_scales(
            predictor, trace, seq_len, hw,
            dram_s, comp_s, batch_size, tp_size)

        if t_ns < best_t:
            best_t = t_ns
            best_cfg = (dram_s, comp_s)

        grad_dram, grad_comp = _compute_scale_gradients(agg_grad, hw, dram_s, comp_s)
        trajectory.append((step, dram_s, comp_s, t_ns, grad_dram, grad_comp))

        grad_norm = math.sqrt(grad_dram ** 2 + grad_comp ** 2)

        if grad_norm < 1e-20:
            # No differentiable bottleneck: try a few grid points then stop
            for _ in range(5):
                dram_s = float(np.random.uniform(lo, hi))
                comp_s = float(np.random.uniform(lo, hi))
                t_ns, _ = _predict_with_scales(
                    predictor, trace, seq_len, hw,
                    dram_s, comp_s, batch_size, tp_size)
                if t_ns < best_t:
                    best_t = t_ns
                    best_cfg = (dram_s, comp_s)
                trajectory.append((step + 1, dram_s, comp_s, t_ns, 0.0, 0.0))
            break

        dram_s -= step_size * grad_dram / grad_norm
        comp_s -= step_size * grad_comp / grad_norm

        dram_s = float(np.clip(dram_s, lo, hi))
        comp_s = float(np.clip(comp_s, lo, hi))

    return best_t, best_cfg[0], best_cfg[1], trajectory


def print_gradient_table(agg_grad, hw, dram_bw_scale=1.0, compute_scale=1.0):
    """Print per-resource gradient magnitudes sorted by bottleneck contribution."""
    grad_cap_scaled = agg_grad.get("grad_capacity_scaled", {})
    grad_cap = agg_grad.get("grad_capacity", {})

    scale_map = {"dram": dram_bw_scale, "compute_tensor": compute_scale}

    all_roles = set(grad_cap_scaled.keys()) | set(grad_cap.keys())
    rows = []
    for role in all_roles:
        if role in grad_cap_scaled:
            current_scale = scale_map.get(role, 1.0)
            sg = grad_cap_scaled[role] / current_scale
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
            except (KeyError, AttributeError):
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


# ---------------------------------------------------------------------------
# Provenance recording
# ---------------------------------------------------------------------------

def get_vllm_version():
    """Return the vLLM version if importable, otherwise from env or 'unknown'."""
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        return os.environ.get("VLLM_VERSION", "unknown")


def get_git_commit():
    """Return the current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-based hardware DSE on dram_bw_scale + compute_scale "
                    "from a captured vLLM emulation trace")
    parser.add_argument("--trace", required=True,
                        help="Path to captured emulation trace JSON")
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B",
                        help="Path to HuggingFace model directory")
    parser.add_argument("--hw-config", required=True,
                        help="Path to hardware config YAML")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mode", choices=["gradient", "grid", "both"],
                        default="gradient",
                        help="DSE mode: gradient descent, grid search, or both")
    parser.add_argument("--steps", type=int, default=100,
                        help="Gradient descent steps (default: 100)")
    parser.add_argument("--grid-size", type=int, default=10,
                        help="Grid search dimension (default: 10x10)")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--lo", type=float, default=0.5)
    parser.add_argument("--hi", type=float, default=2.0)
    parser.add_argument("--dram-bw-scale", type=float, default=1.0,
                        help="Initial DRAM BW scale factor (default: 1.0)")
    parser.add_argument("--compute-scale", type=float, default=1.0,
                        help="Initial compute scale factor (default: 1.0)")
    parser.add_argument("--output", default=None,
                        help="Save DSE results as JSON (default: stdout only)")
    args = parser.parse_args()
    if args.dram_bw_scale <= 0:
        parser.error("--dram-bw-scale must be > 0")
    if args.compute_scale <= 0:
        parser.error("--compute-scale must be > 0")

    # Load trace
    with open(args.trace) as f:
        trace = json.load(f)

    # Setup
    model_config = load_model_config(args.model)
    hw = HWSpec(args.hw_config)
    predictor = TracePredictor(args.hw_config, model_config)

    print("=" * 60)
    print("Hardware DSE: real-trace gradient-based search")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"HW config:   {args.hw_config}")
    print(f"Trace:       {args.trace} ({len(trace)} entries)")
    print(f"Workload:    batch={args.batch} seq_len={args.seq_len} tp={args.tp_size}")
    print(f"Search:      [{args.lo}, {args.hi}]^2 x {args.steps} steps")
    print()

    # Baseline at user-specified initial scales
    t_base, agg_base = _predict_with_scales(
        predictor, trace, args.seq_len, hw,
        args.dram_bw_scale, args.compute_scale, args.batch, args.tp_size)
    print(f"Baseline (dram={args.dram_bw_scale}, compute={args.compute_scale}): "
          f"t_gpu = {t_base/1e6:.3f} ms")
    print()

    # Gradient table
    print("Gradient table at baseline (scale=1.0):")
    print_gradient_table(agg_base, hw, dram_bw_scale=1.0, compute_scale=1.0)
    print()

    # Coverage summary (get_coverage from prior annotate_trace_diff in baseline)
    coverage = predictor.get_diff_coverage()
    total_kernels = len(trace)
    excluded = sum(coverage.values())
    print(f"Kernel coverage: {total_kernels - excluded}/{total_kernels} "
          f"({(total_kernels - excluded)/total_kernels*100:.1f}%) modelled, "
          f"{excluded} excluded:")
    for cat, count in sorted(coverage.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    print()

    # Read provenance sidecar if present (written by run_vllm_emulation.py)
    provenance_path = args.trace + "_provenance.json"
    capture_meta = {}
    if os.path.exists(provenance_path):
        with open(provenance_path) as f:
            capture_meta = json.load(f)

    # Check for no differentiable predictions
    if not agg_base.get("grad_capacity_scaled"):
        print("WARNING: no differentiable resource bottleneck found — "
              "all kernels appear launch-overhead-bound; gradients are zero")
        # Still write output if requested (provenance + baseline data)
        if args.output:
            results_partial = {
                "dev_commit": get_git_commit(),
                "archive_commit": "d0003e0",
                "trace_path": os.path.abspath(args.trace),
                "hw_config_path": args.hw_config,
                "model_path": args.model,
                "vllm_version": get_vllm_version(),
                "capture_command": capture_meta.get("capture_command",
                                  "not available"),
                "baseline_t_gpu_ns": t_base,
                "warning": "no differentiable bottleneck",
            }
            with open(args.output, "w") as f:
                json.dump(results_partial, f, indent=2)
        return

    results = {
        "dev_commit": get_git_commit(),
        "archive_commit": "d0003e0",  # archive/system-e2e-sim-hw-dse tip
        "trace_path": os.path.abspath(args.trace),
        "hw_config_path": args.hw_config,
        "model_path": args.model,
        "vllm_version": get_vllm_version(),
        "capture_command": capture_meta.get("capture_command",
                           "not available (no .provenance sidecar)"),
        "baseline_t_gpu_ns": t_base,
        "baseline_coverage_pct": (total_kernels - excluded) / total_kernels * 100,
        "coverage": coverage,
    }

    if args.mode in ("grid", "both"):
        print(f"Running {args.grid_size}x{args.grid_size} grid search...")
        t_grid, dram_grid, comp_grid = grid_search(
            predictor, trace, args.seq_len, hw,
            args.batch, args.tp_size,
            grid_size=args.grid_size, lo=args.lo, hi=args.hi)
        print(f"Grid search:   t_gpu = {t_grid/1e6:.3f} ms "
              f"at dram={dram_grid:.2f}, compute={comp_grid:.2f}")
        print()
        results["grid_t_ns"] = t_grid
        results["grid_dram_scale"] = dram_grid
        results["grid_compute_scale"] = comp_grid

    if args.mode in ("gradient", "both"):
        print(f"Running {args.steps}-step projected gradient descent...")
        t_grad, dram_grad, comp_grad, trajectory = gradient_descent(
            predictor, trace, args.seq_len, hw,
            args.batch, args.tp_size,
            max_steps=args.steps, step_size=args.step_size,
            lo=args.lo, hi=args.hi,
            init_dram_scale=args.dram_bw_scale,
            init_compute_scale=args.compute_scale)
        print(f"Gradient DSE:  t_gpu = {t_grad/1e6:.3f} ms "
              f"at dram={dram_grad:.2f}, compute={comp_grad:.2f}")
        print()

        # Gradient magnitudes per iteration
        print("Gradient magnitudes per iteration (first 5):")
        for _, _, _, _, gd, gc in trajectory[:5]:
            print(f"  dram_grad={gd:.4e}  compute_grad={gc:.4e}")
        print()

        # Monotonicity
        first_20 = [t for _, _, _, t, _, _ in trajectory[:20]]
        is_monotone = all(first_20[i] >= first_20[i + 1] for i in range(len(first_20) - 1))
        print(f"Trajectory monotone (first 20 steps): {is_monotone}")

        results["gradient_t_ns"] = t_grad
        results["gradient_dram_scale"] = dram_grad
        results["gradient_compute_scale"] = comp_grad
        results["trajectory_len"] = len(trajectory)

        if args.mode == "both" and t_grid > 0:
            gap_pct = abs(t_grad - t_grid) / t_grid * 100
            print(f"Convergence gap vs grid:  {gap_pct:.2f}%")

    # Recommendations
    final_dram = results.get("gradient_dram_scale", results.get("grid_dram_scale", 1.0))
    final_comp = results.get("gradient_compute_scale", results.get("grid_compute_scale", 1.0))
    print()
    print("Recommended parameter changes (scale factors on current HW capacities):")
    print(f"  dram_bw_scale = {final_dram:.3f}  "
          f"({'increase DRAM BW' if final_dram > 1.0 else 'decrease DRAM BW'})")
    print(f"  compute_scale = {final_comp:.3f}  "
          f"({'increase compute' if final_comp > 1.0 else 'decrease compute'})")
    print()
    print("Note: These are relative scale factors. configs/hw/*.yaml files are "
          "NOT modified.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"DSE results saved to: {args.output}")


if __name__ == "__main__":
    main()
