#!/usr/bin/env python3
"""Compare GLM-5.2 1K/128 concurrency with MTP disabled/enabled.

For each C, simulate 2C warmup requests separately (discarded) and 10C
measured requests. The serving model is stateless across simulation invocations;
all requests in each canonical CSV arrive at t=0, with scheduler capacity C.
This preserves InferenceX request counts and concurrency while documenting the
static-arrival approximation used by this simulator.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from functools import partial
from evaluation.serving.experiments.inferencex_comparison import sha256, write_requests, qvalue

write_requests = partial(write_requests, input_tokens=1024, output_tokens=128)

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
OUTDIR = ROOT / 'experiment_logs/inferencex_style_1k128_mtp_sweep'
DATA_ROOT = ROOT / 'serving_dataset/synthetic/inferencex_style/glm_5_2_h200_1k128_mtp_sweep'
BASE_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_aggregate_tp8_runtime_memory.json'
MODEL = ROOT / 'configs/model/glm-5.2-fp8.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'






def make_case(concurrency: int, mtp: bool) -> tuple[Path, Path, Path, Path]:
    variant = 'mtp_on' if mtp else 'mtp_off'
    case_dir = DATA_ROOT / variant / f'c{concurrency}'
    case_dir.mkdir(parents=True, exist_ok=True)
    measured_csv = case_dir / 'measured_requests.csv'
    warmup_csv = case_dir / 'warmup_requests.csv'
    write_requests(measured_csv, concurrency * 10)
    write_requests(warmup_csv, concurrency * 2)

    policy = json.loads(BASE_POLICY.read_text())
    policy['scheduler_pools'][0]['binding']['max_requests'] = concurrency
    policy_path = case_dir / 'runtime_memory_policy.json'
    policy_path.write_text(json.dumps(policy, indent=2) + '\n')
    manifest = {
        'schema_version': 1,
        'workload_id': f'glm_5_2_h200_1k128_c{concurrency}_{variant}',
        'model': 'GLM-5.2-FP8',
        'hardware': '8x NVIDIA H200',
        'parallelism': 'TP8/PP1/EP1 colocated',
        'target_input_tokens': 1024,
        'target_output_tokens': 128,
        'random_range_ratio': 0.8,
        'sampling': 'numpy RandomState(0), independent inclusive uniform integer lengths',
        'seed': 0,
        'client_concurrency': concurrency,
        'measured_requests': concurrency * 10,
        'warmup_requests': concurrency * 2,
        'warmup_simulated_separately_and_discarded': True,
        'mtp_enabled': mtp,
        'mtp_contract': ({'num_speculative_tokens': 1, 'acceptance_policy': 'fixed_count',
                          'fixed_accepted_tokens': 1, 'acceptance_source_id': 'glm5-mtp-fixed-one'}
                         if mtp else {'num_speculative_tokens': 0, 'acceptance_policy': 'disabled'}),
        'arrival_semantics': 'all measured requests in canonical stream arrive at t=0; scheduler max_requests=C',
        'arrival_semantics_limit': 'simulator currently accepts static request streams and does not model InferenceX client-side closed-loop replenishment',
        'measured_requests_sha256': sha256(measured_csv),
        'warmup_requests_sha256': sha256(warmup_csv),
    }
    manifest_path = case_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return measured_csv, warmup_csv, policy_path, manifest_path


def build_static_mtp_command(csv_path: Path, policy_path: Path, concurrency: int, output: Path, mtp: bool) -> list[str]:
    cmd = [
        str(PYTHON), str(ROOT / 'scripts/evaluation/serving/simulation_runner_entrypoint.py'),
        '--canonical-requests', str(csv_path), '--canonical-requests-sha256', sha256(csv_path),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE), '--hardware-config-sha256', sha256(HARDWARE),
        '--deployment', 'colocated', '--backend', 'vllm', '--prediction-method', 'roofline',
        '--collective-backend', 'analytical_roofline_v1', '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '8', '--prefill-pp-size', '1', '--pipeline-ranks-per-node', '8',
        '--scheduler-pipeline-depth', '1', '--max-batched-tokens', '8192',
        '--runtime-memory-policy', str(policy_path), '--kv-block-size-tokens', '16',
        '--max-model-len', '9216', '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'inferencex-style-1k128-mtp-sweep-routing-v1',
        '--expert-routing-seed', '0', '--max-requests', str(concurrency),
        '--no-enable-prefix-caching', '--output', str(output),
    ]
    if mtp:
        cmd.extend(['--mtp-num-speculative-tokens', '1', '--mtp-acceptance-policy', 'fixed_count',
                    '--mtp-fixed-accepted-tokens', '1', '--mtp-acceptance-source-id', 'glm5-mtp-fixed-one'])
    return cmd


def run_static_mtp_point(concurrency: int, mtp: bool) -> tuple[Path, float]:
    measured, warmup, policy, _ = make_case(concurrency, mtp)
    variant = 'mtp_on' if mtp else 'mtp_off'
    result_dir = OUTDIR / variant
    result_dir.mkdir(parents=True, exist_ok=True)
    warmup_out = result_dir / f'c{concurrency}_warmup.json'
    output = result_dir / f'c{concurrency}.json'
    env = dict(os.environ, PYTHONHASHSEED='0')
    start = time.monotonic()
    # Warmup is a separate run, matching its request count while keeping its
    # metrics out of the measured point. No cache state crosses runner calls.
    subprocess.run(build_static_mtp_command(warmup, policy, concurrency, warmup_out, mtp),
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(build_static_mtp_command(measured, policy, concurrency, output, mtp),
                   cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return output, time.monotonic() - start




def summarize_and_plot(results: dict[str, list[Path]]) -> Path:
    import matplotlib.pyplot as plt
    summary = {}
    fig, ax = plt.subplots(figsize=(5.0, 3.5), constrained_layout=True)
    for variant, paths in results.items():
        rows = []
        for path in sorted(paths, key=lambda p: int(p.stem[1:])):
            r = json.loads(path.read_text())
            c = int(path.stem[1:])
            rows.append({
                'C': c,
                'measured_requests': 10 * c,
                'warmup_requests': 2 * c,
                'interactivity_p90_tok_s_user': 1.0 / qvalue(r, 'tpot_s', 90),
                'total_token_throughput_per_gpu_tok_s': r['metrics']['token_throughput_per_s'] / 8,
                'simulated_device_makespan_s': r['schedule_end_time_s'],
                'simulator_wall_time_s': r['simulation_wall_time_s'],
            })
        summary[variant] = rows
        color = '#28527a' if variant == 'mtp_off' else '#d26a7a'
        marker = 'o' if variant == 'mtp_off' else 's'
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, marker + '-', color=color, lw=1.8, ms=4, label=variant.replace('_', ' ').upper())
        if variant == 'mtp_off':
            # Both lines use the same concurrency points in the same order;
            # labeling one line avoids duplicate text where the curves meet.
            for row, px, py in zip(rows, x, y):
                ax.annotate(f"C={row['C']}", (px, py), xytext=(5, -9),
                            textcoords='offset points', fontsize=7, va='top')
    ax.set_xlabel('Interactivity P90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.set_title('GLM-5.2 FP8, 8×H200, 1K/128')
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False)
    for ext in ('png', 'pdf'):
        fig.savefig(OUTDIR / f'mtp_comparison.{ext}', dpi=220, bbox_inches='tight')
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return OUTDIR / 'mtp_comparison.png'


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(c, mtp) for mtp in (False, True) for c in CONCURRENCIES]
    completed: dict[str, list[Path]] = {'mtp_off': [], 'mtp_on': []}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        future_map = {pool.submit(run_static_mtp_point, c, mtp): (c, mtp) for c, mtp in jobs}
        for future in concurrent.futures.as_completed(future_map):
            c, mtp = future_map[future]
            path, elapsed = future.result()
            variant = 'mtp_on' if mtp else 'mtp_off'
            completed[variant].append(path)
            print(f'completed {variant} C={c} elapsed={elapsed:.1f}s ({len(completed[variant])}/{len(CONCURRENCIES)})', flush=True)
    print(summarize_and_plot(completed))


if __name__ == '__main__':
    main()
