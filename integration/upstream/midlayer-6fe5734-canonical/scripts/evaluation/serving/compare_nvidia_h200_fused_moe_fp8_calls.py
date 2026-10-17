#!/usr/bin/env python3
"""Per-call H200 FP8 Triton fused-MoE prediction versus real trace durations.

Evaluation only.  For every in-scope real launch geometry of the DSV3-L32 TP4
vLLM 0.24 traces (extracted by ``extract_nvidia_h200_fused_moe_trace_calls.py``), this
builds the FUSED_MOE KernelSpec the Serving lowering emits (BF16 compute
dtype, FP8 routed weights, EP1) with a seeded uniform top-8 routing of the
window's token count, selects the H200 launch through the public dispatcher,
checks the selected grid/block/shared memory against the trace, and prices it
with ``predict(..., "midlayer")``.  ``--before`` additionally prices the same
spec with the projected vLLM 0.11 launch the baseline used.  Real durations
are only compared, never consumed by a model.

Usage:
    PYTHONPATH=. python scripts/evaluation/serving/compare_nvidia_h200_fused_moe_fp8_calls.py \
        --trace-calls <fused_moe_trace_calls.json> --output <json>
"""
import argparse
import json
from pathlib import Path
import random

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.selection_primitives import (
    select_gpu_fused_moe_launch)

EXPERTS, TOPK, HIDDEN, INTERMEDIATE = 256, 8, 7168, 512
# Real windows: tokens per MoE invocation (decode batch C; prefill steps of
# one, three and four 7373-token prompts, identified from the W1 grid).
WINDOW_TOKENS = {
    "C=1": (1,), "C=4": (4,), "C=16": (16,), "C=64": (64,),
    "prefill": (7373, 22119, 29492),
}


def routing(tokens, seed):
    rng = random.Random(seed)
    counts = [0] * EXPERTS
    for _ in range(tokens):
        for expert in rng.sample(range(EXPERTS), TOPK):
            counts[expert] += 1
    return tuple(counts)


def kernel_spec(hardware, tokens, variant, counts, launch=None):
    shape = {"tokens": tokens, "experts": EXPERTS, "topk": TOPK,
             "hidden": HIDDEN, "intermediate": INTERMEDIATE, "dtype_bytes": 2}
    attributes = {
        "tokens_per_expert": counts,
        "local_tokens_per_expert": counts,
        "expert_parallel_size": 1,
        "expert_rank": 0,
        "owned_experts": tuple(range(EXPERTS)),
        "expert_weight_tile_rows": None,
        "routed_scaling_factor": 2.5,
        "weight_group_size": 0,
        "routed_weight_dtype": "fp8",
    }
    if launch is None:
        launch = select_launch_config(
            "nvidia_h200", KernelKind.FUSED_MOE, shape, "bf16", variant,
            attributes, hardware)
    return KernelSpec(
        kind=KernelKind.FUSED_MOE, hardware_id="nvidia_h200", shape=shape,
        dtype="bf16", variant=variant, launch=launch, attributes=attributes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-calls", type=Path, required=True)
    parser.add_argument("--hardware-config", type=Path,
                        default=Path("configs/hardware/nvidia_h200.yaml"))
    parser.add_argument("--before", action="store_true")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hardware = HWSpec(str(args.hardware_config))
    real = json.loads(args.trace_calls.read_text())
    rows = []
    for window, tokens_list in WINDOW_TOKENS.items():
        geometries = real["windows"][window]["geometries"]
        for tokens in tokens_list:
            counts = routing(tokens, args.seed + tokens)
            for variant in ("w1", "w2"):
                spec = kernel_spec(hardware, tokens, variant, counts)
                launch = spec.launch
                matched = [
                    geometry for geometry in geometries
                    if geometry["variant"] == variant
                    and geometry["grid"] == list(launch.grid)
                    and geometry["block"] == list(launch.block)]
                if len(matched) != 1:
                    raise ValueError(
                        f"{window} tokens={tokens} {variant}: selected grid "
                        f"{launch.grid} matches {len(matched)} real geometries")
                geometry, = matched
                shared_mb = (
                    f"{launch.parameters['dynamic_shared_memory_bytes'] / 1e6:.3f}")
                prediction = predict(spec, hardware, "midlayer")
                real_us = geometry["duration_ns_median"] / 1e3
                row = {
                    "window": window, "tokens": tokens, "variant": variant,
                    "algorithm": launch.algorithm,
                    "grid": launch.grid[0], "block": launch.block[0],
                    "tile": dict(launch.tiling.tile_shape),
                    "group_size_m": launch.parameters["group_size_m"],
                    "num_stages": launch.parameters["num_stages"],
                    "dynamic_shared_memory_mb": shared_mb,
                    "real_dynamic_shared_memory_mb": geometry[
                        "dynamic_shared_memory_mb_nsys"],
                    "real_registers_per_thread": geometry[
                        "registers_per_thread"],
                    "real_launch_count": geometry["launch_count"],
                    "real_us_median": real_us,
                    "real_us_p10": geometry["duration_ns_p10"] / 1e3,
                    "real_us_p90": geometry["duration_ns_p90"] / 1e3,
                    "predicted_us": prediction.t_total_s * 1e6,
                    "signed_error": prediction.t_total_s * 1e6 / real_us - 1.0,
                    "bottleneck_group": prediction.bottleneck_group,
                    "t_per_group_us": {
                        name: value * 1e6
                        for name, value in prediction.t_per_group.items()},
                }
                if args.before:
                    before_launch = select_gpu_fused_moe_launch(
                        spec.shape, variant, spec.attributes,
                        block_threads=128)
                    before = predict(
                        kernel_spec(hardware, tokens, variant, counts,
                                    before_launch),
                        hardware, "midlayer")
                    row["before_algorithm"] = before_launch.algorithm
                    row["before_predicted_us"] = before.t_total_s * 1e6
                    row["before_signed_error"] = (
                        before.t_total_s * 1e6 / real_us - 1.0)
                rows.append(row)
                print(f"{window:8s} T={tokens:5d} {variant} grid={launch.grid[0]:6d} "
                      f"real {real_us:9.1f} us  pred {row['predicted_us']:9.1f} us "
                      f"({row['signed_error']:+.1%}) {prediction.bottleneck_group}"
                      + (f"  before {row['before_predicted_us']:9.1f} "
                         f"({row['before_signed_error']:+.1%})"
                         if args.before else ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "artifact": "h200_fused_moe_fp8_call_comparison_v1",
        "trace_calls": str(args.trace_calls),
        "routing": f"seeded uniform distinct top-{TOPK} of {EXPERTS} "
                   f"(seed {args.seed} + tokens)",
        "evaluation_only": True,
        "production_or_evaluation_duration_consumed_by_model": False,
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
