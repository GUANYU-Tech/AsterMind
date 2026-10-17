#!/usr/bin/env python3
"""Compare colocated and PD GLM-5.2 1K/128 sweeps with MTP off/on."""
from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path('/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python')
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
GROUPS = ('colocated_mtp_off', 'colocated_mtp_on', 'pd_mtp_off', 'pd_mtp_on')
OUTDIR = ROOT / 'experiment_logs/inferencex_style_glm_1k128_mtp5_pd_tp8_ep1'
DATA_ROOT = ROOT / 'serving_dataset/synthetic/inferencex_style/glm_5_2_h200_1k128_mtp5_pd_tp8_ep1'
COLOCATED_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_aggregate_tp8_runtime_memory.json'
PD_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_pd_tp8_ep1_runtime_memory.json'
MODEL = ROOT / 'configs/model/glm-5.2-fp8.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'
PROPOSAL_DEPTH = 5
MEAN_ACCEPTANCE_RATE = '0.82'  # 5 speculative tokens * 0.82 = 4.1 expected accepts.
ACCEPTANCE_SOURCE = 'glm5-top1-depth5-mean4.1-v1'
MAX_WORKERS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_requests(path: Path, count: int) -> None:
    rng = np.random.RandomState(0)
    inputs = rng.randint(819, 1025, size=count)
    outputs = rng.randint(102, 129, size=count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Query Time Stamp [ms]', 'In Token [tokens]',
                         'Out Token [tokens]', 'Cached hit [tokens]'])
        writer.writerows((0, int(i), int(o), 0) for i, o in zip(inputs, outputs))


def prepare_case(group: str, concurrency: int):
    deployment, mtp = group.split('_mtp_')
    mtp = mtp == 'on'
    case_dir = DATA_ROOT / group / f'c{concurrency}'
    case_dir.mkdir(parents=True, exist_ok=True)
    measured = case_dir / 'measured_requests.csv'
    warmup = case_dir / 'warmup_requests.csv'
    write_requests(measured, concurrency * 10)
    write_requests(warmup, concurrency * 2)

    if deployment == 'colocated':
        raw_policy = json.loads(COLOCATED_POLICY.read_text())
        raw_policy['scheduler_pools'][0]['binding']['max_requests'] = concurrency
    else:
        raw_policy = json.loads(PD_POLICY.read_text())
        for pool in raw_policy['scheduler_pools']:
            binding = pool['binding']
            binding['hardware_config_sha256'] = sha256(HARDWARE)
            binding['model_config_sha256'] = sha256(MODEL)
            binding['max_requests'] = concurrency
            binding['expert_parallel_size'] = 1
            binding['max_model_len'] = 9216
            binding['max_batched_tokens'] = 8192
    policy = case_dir / 'runtime_memory_policy.json'
    policy.write_text(json.dumps(raw_policy, indent=2) + '\n')

    manifest = {
        'schema_version': 1,
        'workload_id': f'glm_5_2_h200_1k128_{group}_c{concurrency}',
        'model': 'GLM-5.2-FP8',
        'hardware': ('8x NVIDIA H200' if deployment == 'colocated'
                     else '8x NVIDIA H200 prefill + 8x NVIDIA H200 decode'),
        'deployment': 'colocated' if deployment == 'colocated' else 'pd_disagg',
        'parallelism': ('TP8/PP1/EP1 colocated' if deployment == 'colocated'
                        else 'P/D: TP8/PP1/EP1, one instance per role'),
        'pd_transfer': None if deployment == 'colocated' else 'send_recv over NVLink',
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
        'mtp_contract': ({
            'top1': True,
            'num_speculative_tokens': PROPOSAL_DEPTH,
            'maximum_total_tokens_per_verify': PROPOSAL_DEPTH + 1,
            'acceptance_policy': 'maximum_entropy',
            'mean_accepted_speculative_tokens': 4.1,
            'mean_acceptance_rate': MEAN_ACCEPTANCE_RATE,
            'acceptance_seed': 0,
            'acceptance_source_id': ACCEPTANCE_SOURCE,
        } if mtp else {'num_speculative_tokens': 0,
                       'acceptance_policy': 'disabled'}),
        'arrival_semantics': 'all measured requests arrive at t=0; scheduler max_requests=C',
        'arrival_semantics_limit': 'static-arrival approximation; no InferenceX client-side closed-loop replenishment',
        'measured_requests_sha256': sha256(measured),
        'warmup_requests_sha256': sha256(warmup),
    }
    manifest_path = case_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return measured, warmup, policy


def command_for(group: str, csv_path: Path, policy: Path,
                concurrency: int, output: Path) -> list[str]:
    deployment, mtp = group.split('_mtp_')
    mtp = mtp == 'on'
    cmd = [
        str(PYTHON), str(ROOT / 'scripts/evaluation/serving/simulation_runner_entrypoint.py'),
        '--canonical-requests', str(csv_path), '--canonical-requests-sha256', sha256(csv_path),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE), '--hardware-config-sha256', sha256(HARDWARE),
        '--backend', 'vllm', '--prediction-method', 'roofline',
        '--collective-backend', 'analytical_roofline_v1', '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '8', '--prefill-pp-size', '1', '--max-batched-tokens', '8192',
        '--scheduler-pipeline-depth', '1',
        '--runtime-memory-policy', str(policy), '--kv-block-size-tokens', '16',
        '--max-model-len', '9216', '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'inferencex-style-1k128-mtp5-routing-v1',
        '--expert-routing-seed', '0', '--max-requests', str(concurrency),
        '--no-enable-prefix-caching', '--enable-chunked-prefill', '--output', str(output),
    ]
    if deployment == 'colocated':
        cmd.extend(['--deployment', 'colocated', '--pipeline-ranks-per-node', '8'])
    else:
        cmd.extend([
            '--deployment', 'pd_disagg', '--decode-hardware-config', str(HARDWARE),
            '--decode-tp-size', '8', '--num-prefill-instances', '1',
            '--num-decode-instances', '1', '--transfer-algorithm', 'send_recv',
            '--transfer-link', 'nvlink',
        ])
    if mtp:
        cmd.extend([
            '--mtp-num-speculative-tokens', str(PROPOSAL_DEPTH),
            '--mtp-acceptance-policy', 'maximum_entropy',
            '--mtp-mean-acceptance-rate', MEAN_ACCEPTANCE_RATE,
            '--mtp-acceptance-seed', '0',
            '--mtp-acceptance-source-id', ACCEPTANCE_SOURCE,
        ])
    return cmd


def run_point(group: str, concurrency: int):
    measured, warmup, policy = prepare_case(group, concurrency)
    result_dir = OUTDIR / group
    log_dir = result_dir / 'logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    warmup_output = result_dir / f'c{concurrency}_warmup.json'
    output = result_dir / f'c{concurrency}.json'
    if output.exists() and warmup_output.exists():
        return group, concurrency, output, 0.0, 'reused'
    env = dict(os.environ, PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    for label, source, destination in (
            ('warmup', warmup, warmup_output),
            ('measured', measured, output)):
        stderr_path = log_dir / f'c{concurrency}_{label}.stderr'
        with stderr_path.open('w') as stderr:
            subprocess.run(command_for(group, source, policy, concurrency, destination),
                           cwd=ROOT, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=stderr)
    return group, concurrency, output, time.monotonic() - start, 'completed'


def qvalue(result: dict, metric: str, percentile: int) -> float:
    quantiles = result['metrics']['latency_distributions'][metric]['quantiles']
    return quantiles['values_s'][quantiles['percentiles'].index(percentile)]


def summarize_and_plot() -> Path:
    import matplotlib.pyplot as plt
    summary = {}
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    styles = {
        'colocated_mtp_off': ('#28527a', 'o', '-'),
        'colocated_mtp_on': ('#d26a7a', 's', '-'),
        'pd_mtp_off': ('#28527a', '^', '--'),
        'pd_mtp_on': ('#d26a7a', 'D', '--'),
    }
    for group in GROUPS:
        deployment, mtp = group.split('_mtp_')
        # InferenceX convention for disaggregated configs: normalize PD by
        # decode-chip count, with the apples-to-apples disclosure below.
        # Colocated is normalized by its total GPU count.
        gpu_count = 8
        normalization = ('total_gpus' if deployment == 'colocated'
                         else 'decode_gpus')
        rows = []
        for concurrency in CONCURRENCIES:
            path = OUTDIR / group / f'c{concurrency}.json'
            result = json.loads(path.read_text())
            total_throughput = result['metrics']['token_throughput_per_s']
            rows.append({
                'C': concurrency,
                'measured_requests': 10 * concurrency,
                'warmup_requests': 2 * concurrency,
                'interactivity_p90_tok_s_user': 1.0 / qvalue(result, 'tpot_s', 90),
                'total_token_throughput_tok_s': total_throughput,
                'total_token_throughput_per_gpu_tok_s': (
                    total_throughput / gpu_count),
                'throughput_normalization': normalization,
                'simulated_device_makespan_s': result['schedule_end_time_s'],
                'simulator_wall_time_s': result['simulation_wall_time_s'],
            })
        summary[group] = rows
        color, marker, linestyle = styles[group]
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, marker + linestyle, color=color, lw=1.7, ms=4.2,
                label=group.replace('_', ' ').upper())
    ax.set_xlabel('Interactivity P90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.set_title('GLM-5.2 FP8 on H200 (analytical projection), 1K/128, '
                 'TP8/EP1 all pools')
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.text(
        0.01, -0.02,
        'MTP on: propose 5 tokens, top-1, max 6 committed per verify; assumed '
        'mean 4.1 accepted speculative tokens (acceptance rate 0.82) -\n'
        'an explicit assumption (source glm5-top1-depth5-mean4.1-v1), not a '
        'measured trace. Arrivals: 10xC requests at t=0, scheduler cap C '
        '(static backlog, not closed-loop).\n'
        'Y axis: total generated-token throughput per GPU. Colocated '
        'normalized by total GPUs (8); PD normalized by decode GPUs (8 of '
        '16), following the InferenceX convention for disaggregated\n'
        'configs - per-decode-GPU is not apples-to-apples with per-total-GPU. '
        'X axis: 1 / P90 TPOT. Warmup 2xC requests discarded.',
        fontsize=6.5, va='top')
    for ext in ('png', 'pdf'):
        fig.savefig(OUTDIR / f'colocated_vs_pd_mtp5.{ext}', dpi=220, bbox_inches='tight')
    summary['assumptions'] = {
        'projection': 'analytical H200 roofline projection, not a measured run',
        'mtp_on_contract': {
            'num_speculative_tokens': PROPOSAL_DEPTH,
            'top1': True,
            'maximum_total_tokens_per_verify': PROPOSAL_DEPTH + 1,
            'mean_accepted_speculative_tokens': 4.1,
            'mean_acceptance_rate': MEAN_ACCEPTANCE_RATE,
            'status': 'explicit assumption, not a measured acceptance trace',
            'acceptance_source_id': ACCEPTANCE_SOURCE,
        },
        'arrival_semantics': ('all 10xC measured requests arrive at t=0; '
                              'scheduler admission cap C; static backlog, '
                              'not a client-side closed-loop benchmark'),
        'throughput_per_gpu_normalization': (
            'colocated: total token throughput / total GPUs (8); PD: total '
            'token throughput / decode GPUs (8 of 16), InferenceX '
            'disaggregated convention - per-decode-GPU is not '
            'apples-to-apples with per-total-GPU'),
        'parallelism': 'TP8/PP1/EP1 for colocated and for both PD roles',
        'pd_transfer_contract': (
            'compressed MLA kv_c||k_pe (78 layers x 576 x 2 B) plus FP8 '
            'index cache (21 full-indexer layers x 132 B) = 92,628 B/token '
            'per TP rank, replicated single-head caches transferred over 8 '
            'identity rank pairs; EP-independent per pinned vLLM semantics'),
    }
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return OUTDIR / 'colocated_vs_pd_mtp5.png'


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(group, c) for group in GROUPS for c in CONCURRENCIES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(run_point, group, c): (group, c)
                      for group, c in jobs}
        for future in concurrent.futures.as_completed(future_map):
            group, c = future_map[future]
            result = future.result()
            print(f'{group} C={c} {result[4]} elapsed={result[3]:.1f}s', flush=True)
    print(summarize_and_plot())


if __name__ == '__main__':
    main()
