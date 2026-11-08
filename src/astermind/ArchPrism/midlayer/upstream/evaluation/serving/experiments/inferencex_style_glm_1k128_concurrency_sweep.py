#!/usr/bin/env python3
"""Sweep first-wave client concurrency for the 1K/128 GLM-5.2 H200 smoke setup.

Each point submits C requests at t=0.  It deliberately does not emulate later
closed-loop replenishment; use it to validate and visualize scheduler behavior.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

from functools import partial
from evaluation.serving.experiments.inferencex_comparison import sha256, write_requests, qvalue

write_requests = partial(write_requests, input_tokens=1024, output_tokens=128)

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
OUTDIR = ROOT / 'experiment_logs/inferencex_style_1k128_concurrency_sweep'
DATA_ROOT = ROOT / 'serving_dataset/synthetic/inferencex_style/glm_5_2_h200_1k128_concurrency_sweep'
BASE_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_aggregate_tp8_runtime_memory.json'
MODEL = ROOT / 'configs/model/glm-5.2-fp8.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'




def make_case(concurrency: int) -> tuple[Path, Path, Path]:
    """Generate a deterministic InferenceX-style first wave and matching policy."""
    case_dir = DATA_ROOT / f'c{concurrency}'
    case_dir.mkdir(parents=True, exist_ok=True)
    requests = case_dir / 'requests.csv'
    write_requests(requests, concurrency)
    manifest = {
        'kind': 'inferencex_style_first_wave_concurrency_sweep',
        'client_concurrency': concurrency,
        'arrival_semantics': 'C requests submit at t=0; no post-completion replenishment',
        'target_input_tokens': 1024,
        'target_output_tokens': 128,
        'random_range_ratio': .8,
        'sampling': 'numpy RandomState(0), independent uniform integer [floor(0.8*target), target]',
        'seed': 0,
        'measured_requests': concurrency,
        'warmup_requests': 0,
        'mtp_enabled': False,
        'request_csv_sha256': sha256(requests),
    }
    (case_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    policy = json.loads(BASE_POLICY.read_text())
    policy['scheduler_pools'][0]['binding']['max_requests'] = concurrency
    policy_path = case_dir / 'runtime_memory_policy.json'
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    return requests, policy_path, case_dir / 'manifest.json'


def run_case(concurrency: int) -> Path:
    requests, policy, _ = make_case(concurrency)
    output = OUTDIR / f'c{concurrency}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), '-c', 'from evaluation.serving.simulation.simulation_runner import main; raise SystemExit(main())',
        '--canonical-requests', str(requests), '--canonical-requests-sha256', sha256(requests),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE), '--hardware-config-sha256', sha256(HARDWARE),
        '--deployment', 'colocated', '--backend', 'vllm', '--prediction-method', 'roofline',
        '--collective-backend', 'analytical_roofline_v1', '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '8', '--prefill-pp-size', '1', '--pipeline-ranks-per-node', '8',
        '--scheduler-pipeline-depth', '1', '--max-batched-tokens', '8192',
        '--runtime-memory-policy', str(policy), '--kv-block-size-tokens', '16',
        '--max-model-len', '9216', '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'inferencex-style-concurrency-sweep-routing-v1',
        '--expert-routing-seed', '0', '--max-requests', str(concurrency),
        '--no-enable-prefix-caching', '--output', str(output),
    ]
    env = dict(os.environ, PYTHONHASHSEED='0')
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return output


def metric_quantile(result: dict, name: str, percentile: int) -> float:
    q = result['metrics']['latency_distributions'][name]['quantiles']
    return q['values_s'][q['percentiles'].index(percentile)] * 1e3


def plot(outputs: list[Path]) -> Path:
    import matplotlib.pyplot as plt
    rows = []
    for path in sorted(outputs, key=lambda p: int(p.stem[1:])):
        r = json.loads(path.read_text())
        rows.append({
            'C': int(path.stem[1:]),
            'TTFT p50 (ms)': metric_quantile(r, 'ttft_s', 50),
            'TPOT p50 (ms)': metric_quantile(r, 'tpot_s', 50),
            # InferenceX conventional interactivity: reciprocal of the
            # matching slow-tail TPOT percentile, in tok/s/user.
            'Interactivity p90 (tok/s/user)': 1e3 / metric_quantile(r, 'tpot_s', 90),
            'Output throughput (tok/s)': r['metrics']['output_token_throughput_per_s'],
            # InferenceX's primary fixed-sequence ``tput_per_gpu`` field is
            # total (input + output) token throughput, divided by all GPUs.
            'Total throughput/GPU (tok/s/GPU)': r['metrics']['token_throughput_per_s'] / 8,
            'Sim wall time (s)': r['simulation_wall_time_s'],
        })
    (OUTDIR / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    x = [r['C'] for r in rows]
    # The InferenceX fixed-sequence trade-off: conventional p90 decode
    # interactivity (1 / p90 TPOT) against total input+output TPUT/GPU.
    fig, ax = plt.subplots(figsize=(4.0, 3.0), constrained_layout=True)
    xv = [r['Interactivity p90 (tok/s/user)'] for r in rows]
    yv = [r['Total throughput/GPU (tok/s/GPU)'] for r in rows]
    ax.plot(xv, yv, 'o-', color='#28527a', lw=1.8, ms=5)
    for row, px, py in zip(rows, xv, yv):
        ax.annotate(f"C={row['C']}", (px, py), xytext=(4, 4),
                    textcoords='offset points', fontsize=8)
    ax.set_xlabel('Interactivity p90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.grid(True, alpha=.25)
    ax.set_title('GLM-5.2, 8$\\times$H200, 1K/128, MTP Off', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(OUTDIR / f'concurrency_sweep.{ext}', dpi=220, bbox_inches='tight')
    return OUTDIR / 'concurrency_sweep.png'


def main() -> None:
    # Run every independent point concurrently; each subprocess has isolated outputs.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CONCURRENCIES)) as pool:
        outputs = list(pool.map(run_case, CONCURRENCIES))
    print(plot(outputs))


if __name__ == '__main__':
    main()
