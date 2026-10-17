#!/usr/bin/env python3
"""Per-call comparison of H200 block-FP8 dense GEMMs: Midlayer vs real nsys.

Evaluation only.  Simulates one DSV3-L32 TP4 step exactly like
``compare_nvidia_h200_decode_kernel_families.py`` (``--phase decode --batch C`` or
``--phase prefill --batch 4``), keeps the rank-0 GEMM calls whose launch is a
block-FP8 algorithm, and reports for each (M, N, K) the launch facts, the
predicted per-call duration with its template groups, and the real per-call
median from ``extract_nvidia_h200_fp8_gemm_real_calls.py`` output (GEMM kernel plus,
for swap-AB calls, the preceding ``scale_1x128_kernel``).  No real duration
feeds any model parameter.

Usage:
    PYTHONPATH=. python scripts/evaluation/serving/compare_nvidia_h200_fp8_gemm_calls.py \\
        --real-calls real_fp8_gemm_calls.json --window C=1 --phase decode \\
        --batch 1 --output out.json
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def simulate_calls(batch, phase):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts/evaluation/serving'))
    from compare_nvidia_h200_decode_kernel_families import simulate_step
    from src.kernel.hardware_spec import HWSpec
    from src.kernel.kernel_base import predict
    from src.kernel.predictors.common import graph_replayed_launch
    from src.kernel.predictors.gemm_block_fp8 import is_block_fp8_gemm

    step = simulate_step(batch, 7373, 'midlayer', phase)
    hardware = HWSpec(str(ROOT / 'configs/hardware/nvidia_h200.yaml'))
    calls = {call.call_id: call for call in step.kernel_plan.calls}
    rows = {}
    for interval in step.execution_timeline.intervals:
        if interval.rank != 0:
            continue
        spec = calls[interval.call_id].kernel_spec
        if spec.kind.name != 'GEMM' or not is_block_fp8_gemm(spec):
            continue
        key = (spec.shape['m'], spec.shape['n'], spec.shape['k'])
        duration_us = (interval.end_time_s - interval.start_time_s) * 1e6
        if key not in rows:
            prediction = predict(spec, hardware, 'midlayer')
            launch = spec.launch
            rows[key] = {
                'm': key[0], 'n': key[1], 'k': key[2],
                'algorithm': launch.algorithm,
                'grid': list(launch.grid), 'block': list(launch.block),
                'cluster': list(launch.cluster),
                'tile_m': launch.tiling.tile_shape['m'],
                'tile_n': launch.tiling.tile_shape['n'],
                'stages': launch.tiling.pipeline_depth,
                'graph_replayed': graph_replayed_launch(spec),
                'predicted_groups_us': {
                    name: value * 1e6
                    for name, value in prediction.t_per_group.items()},
                'bottleneck_group': prediction.bottleneck_group,
                'sim_interval_us': [],
            }
        rows[key]['sim_interval_us'].append(duration_us)
    return list(rows.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-calls', type=Path, required=True)
    parser.add_argument('--window', required=True)
    parser.add_argument('--phase', choices=('decode', 'prefill'),
                        required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--real-m', type=int, required=True,
                        help='GEMM M of the real window rows to compare')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    real_window = json.loads(args.real_calls.read_text())['windows'][
        args.window]
    real = {}
    for call in real_window['calls']:
        launch = call['launch']
        # Swap-AB rows come from pure-decode windows whose M is the window's
        # batch; 1D2D rows carry the M recovered from their quant launch.
        if launch['kernel'] == 'deepgemm_sm90_fp8_gemm_1d2d_impl':
            if launch['gemm_m'] != args.real_m:
                continue
        elif real_window['gemm_m'] != args.real_m:
            raise ValueError('swap-AB window M differs from --real-m')
        key = (launch['n'], launch['k'])
        if key in real:
            raise ValueError(f'duplicate real rows for {key}')
        real[key] = call
    rows = simulate_calls(args.batch, args.phase)
    for row in rows:
        sim_us = sum(row['sim_interval_us']) / len(row['sim_interval_us'])
        row['sim_per_call_us'] = sim_us
        del row['sim_interval_us']
        match = real[(row['n'], row['k'])]
        launch = match['launch']
        real_us = match['duration']['median_ns'] / 1e3
        if launch['kernel'] == 'trtllm_deepgemm_fp8_gemm_kernel_swapAB':
            real_us += match['scale_1x128_duration']['median_ns'] / 1e3
            row['real_scale_1x128_us'] = (
                match['scale_1x128_duration']['median_ns'] / 1e3)
        row['real_gemm_us'] = match['duration']['median_ns'] / 1e3
        row['real_per_call_us'] = real_us
        row['real_stream'] = launch['stream']
        row['real_calls'] = match['duration']['count']
        row['signed_error'] = sim_us / real_us - 1.0
        row['real_launch'] = {
            name: launch[name] for name in (
                'grid', 'block', 'block_m', 'block_n', 'num_stages',
                'num_tma_multicast', 'registers_per_thread',
                'dynamic_shared_memory_mb_nsys')}
    rows.sort(key=lambda row: (row['n'], row['k']))
    report = {
        'window': args.window, 'phase': args.phase, 'batch': args.batch,
        'real_m': args.real_m, 'rows': rows,
        'production_or_evaluation_duration_consumed_by_model': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')
    for row in rows:
        print(f"M={row['m']:>6} {row['n']:>5}x{row['k']:<5} "
              f"{row['algorithm']:<34} sim {row['sim_per_call_us']:9.2f} us"
              f"  real {row['real_per_call_us']:9.2f} us "
              f"({row['signed_error']:+.1%})")


if __name__ == '__main__':
    main()
