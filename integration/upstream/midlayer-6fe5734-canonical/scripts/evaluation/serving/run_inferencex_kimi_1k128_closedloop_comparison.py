#!/usr/bin/env python3
"""Closed-loop colocated vs PD Kimi-K2.6 1K/128 sweeps, MTP off only.

Every group runs under the InferenceX-style zero-think-time closed-loop
client contract: C clients, each with at most one outstanding request, the
successor injected at the client's own completion instant
(--closed-loop-client-count C).  Both deployments use the H200 Kimi
TP8/PP1/EP8 scope: one colocated pool, or one prefill plus one decode
pool with the same per-role topology (the standard vLLM
DeepSeek-V3-family single-node layout).  Kimi-K2.6 declares zero MTP
layers, so only the MTP-off configuration exists.

PD KV handoff uses the registered H200 scale-out ProtoBridge send_recv
profile (400GbE RoCEv2 theory preset, development-only).  Client-visible
TTFT/TPOT are attributed to the terminal pool: the decode slot wait after
a prefill handoff is TTFT, not an inter-token gap.
"""
from __future__ import annotations

import argparse
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
GROUPS = ('colocated_mtp_off', 'pd_mtp_off')
OUTDIR = ROOT / 'experiment_logs/inferencex_style_kimi_1k128_closedloop_tp8_ep8'
DATA_ROOT = ROOT / 'serving_dataset/synthetic/inferencex_style/kimi_k2_6_h200_1k128_closedloop_tp8_ep8'
COLOCATED_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/kimi_k2_6_nvidia_h200_colocated_tp8_ep8_runtime_memory.json'
PD_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/kimi_k2_6_nvidia_h200_pd_tp8_ep8_runtime_memory.json'
MODEL = ROOT / 'configs/model/kimi-k2.6.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'
TRANSFER_LINK = 'infiniband'
ROUTING_SOURCE_ID = 'inferencex-style-kimi-1k128-routing-v1'
ROUTING_SEED = 0
MAX_MODEL_LEN = 9216
MAX_BATCHED_TOKENS = 8192


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


def prepare_case(group: str, concurrency: int, measured_per_client: int,
                 warmup_per_client: int):
    deployment = group.split('_mtp_')[0]
    case_dir = DATA_ROOT / group / f'c{concurrency}'
    case_dir.mkdir(parents=True, exist_ok=True)
    measured = case_dir / 'measured_requests.csv'
    warmup = case_dir / 'warmup_requests.csv'
    write_requests(measured, concurrency * measured_per_client)
    write_requests(warmup, concurrency * warmup_per_client)

    raw_policy = json.loads(
        (COLOCATED_POLICY if deployment == 'colocated' else PD_POLICY)
        .read_text())
    for pool in raw_policy['scheduler_pools']:
        binding = pool['binding']
        binding['hardware_config_sha256'] = sha256(HARDWARE)
        binding['model_config_sha256'] = sha256(MODEL)
        binding['max_requests'] = concurrency
        binding['max_model_len'] = MAX_MODEL_LEN
        binding['max_batched_tokens'] = MAX_BATCHED_TOKENS
    policy = case_dir / 'runtime_memory_policy.json'
    policy.write_text(json.dumps(raw_policy, indent=2) + '\n')

    manifest = {
        'schema_version': 1,
        'workload_id': f'kimi_k2_6_h200_1k128_closedloop_{group}_c{concurrency}',
        'model': 'Kimi-K2.6 (W4A16: INT4 g32 routed experts, BF16 elsewhere)',
        'hardware': ('8x NVIDIA H200' if deployment == 'colocated'
                     else '8x NVIDIA H200 prefill + 8x NVIDIA H200 decode'),
        'deployment': 'colocated' if deployment == 'colocated' else 'pd_disagg',
        'parallelism': ('TP8/PP1/EP8 colocated' if deployment == 'colocated'
                        else 'P/D: TP8/PP1/EP8, one instance per role'),
        'pd_transfer': (None if deployment == 'colocated' else
                        'send_recv over scale-out 400GbE RoCEv2, ProtoBridge '
                        'theory preset (development-only, uncalibrated)'),
        'target_input_tokens': 1024,
        'target_output_tokens': 128,
        'random_range_ratio': 0.8,
        'sampling': 'numpy RandomState(0), independent inclusive uniform integer lengths',
        'seed': 0,
        'client_concurrency': concurrency,
        'measured_requests': concurrency * measured_per_client,
        'warmup_requests': concurrency * warmup_per_client,
        'warmup_simulated_separately_and_discarded': True,
        'mtp_enabled': False,
        'mtp_contract': {'num_speculative_tokens': 0,
                         'acceptance_policy': 'disabled',
                         'note': ('Kimi-K2.6 declares zero MTP layers; MTP is '
                                  'not supported for Kimi anywhere in this '
                                  'simulator')},
        'arrival_semantics': (
            'zero-think-time closed-loop: C clients, each with at most one '
            'outstanding request; the successor is injected at the client\'s '
            'own completion instant (engine clock); chains assigned '
            'interleaved (client c owns requests c::C); each client\'s first '
            'request arrives at t=0'),
        'measured_requests_sha256': sha256(measured),
        'warmup_requests_sha256': sha256(warmup),
    }
    manifest_path = case_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return measured, warmup, policy


def command_for(group: str, csv_path: Path, policy: Path,
                concurrency: int, output: Path) -> list[str]:
    deployment = group.split('_mtp_')[0]
    cmd = [
        str(PYTHON), str(ROOT / 'scripts/evaluation/serving/simulation_runner_entrypoint.py'),
        '--canonical-requests', str(csv_path), '--canonical-requests-sha256', sha256(csv_path),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE), '--hardware-config-sha256', sha256(HARDWARE),
        '--backend', 'vllm', '--prediction-method', 'roofline',
        '--collective-backend', 'analytical_roofline_v1', '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '8', '--prefill-pp-size', '1',
        '--max-batched-tokens', str(MAX_BATCHED_TOKENS),
        '--scheduler-pipeline-depth', '1',
        '--runtime-memory-policy', str(policy), '--kv-block-size-tokens', '16',
        '--max-model-len', str(MAX_MODEL_LEN),
        '--enable-expert-parallel',
        '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', ROUTING_SOURCE_ID,
        '--expert-routing-seed', str(ROUTING_SEED),
        '--max-requests', str(concurrency),
        '--no-enable-prefix-caching', '--enable-chunked-prefill', '--output', str(output),
        '--closed-loop-client-count', str(concurrency),
    ]
    if deployment == 'colocated':
        cmd.extend(['--deployment', 'colocated', '--pipeline-ranks-per-node', '8'])
    else:
        cmd.extend([
            '--deployment', 'pd_disagg', '--decode-hardware-config', str(HARDWARE),
            '--decode-tp-size', '8', '--num-prefill-instances', '1',
            '--num-decode-instances', '1', '--transfer-algorithm', 'send_recv',
            '--transfer-link', TRANSFER_LINK,
        ])
    return cmd


def run_point(group: str, concurrency: int, run_dir: Path,
              measured_per_client: int, warmup_per_client: int):
    measured, warmup, policy = prepare_case(
        group, concurrency, measured_per_client, warmup_per_client)
    result_dir = run_dir / group
    log_dir = result_dir / 'logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    warmup_output = result_dir / f'c{concurrency}_warmup.json'
    output = result_dir / f'c{concurrency}.json'
    if output.exists() and warmup_output.exists():
        return group, concurrency, output, 0.0, 'reused'
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
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


def group_rows(run_dir: Path, group: str, concurrencies: tuple[int, ...],
               measured_per_client: int, warmup_per_client: int,
               normalization: str) -> list[dict]:
    rows = []
    for concurrency in concurrencies:
        path = run_dir / group / f'c{concurrency}.json'
        result = json.loads(path.read_text())
        total_throughput = result['metrics']['token_throughput_per_s']
        rows.append({
            'C': concurrency,
            'measured_requests': measured_per_client * concurrency,
            'warmup_requests': warmup_per_client * concurrency,
            'interactivity_p90_tok_s_user': 1.0 / qvalue(result, 'tpot_s', 90),
            'ttft_p50_s': qvalue(result, 'ttft_s', 50),
            'ttft_p90_s': qvalue(result, 'ttft_s', 90),
            'tpot_p50_ms': 1000.0 * qvalue(result, 'tpot_s', 50),
            'tpot_p90_ms': 1000.0 * qvalue(result, 'tpot_s', 90),
            'total_token_throughput_tok_s': total_throughput,
            'total_token_throughput_per_gpu_tok_s': total_throughput / 8,
            'throughput_normalization': normalization,
            'simulated_device_makespan_s': result['schedule_end_time_s'],
            'simulator_wall_time_s': result['simulation_wall_time_s'],
        })
    return rows


def summarize_and_plot(run_dir: Path, concurrencies: tuple[int, ...],
                       measured_per_client: int,
                       warmup_per_client: int) -> Path:
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(ROOT))
    from src.kernel.predictors.protobridge.adapter import (
        development_profile_config_sha256, resolve_registered_profile)
    transfer_profile_sha256 = development_profile_config_sha256(
        'nvidia_h200', 'send_recv', link=TRANSFER_LINK)
    transfer_profile = resolve_registered_profile(transfer_profile_sha256)
    summary = {}
    fig, ax = plt.subplots(figsize=(6.9, 5.4), constrained_layout=True)
    styles = {
        'colocated_mtp_off': ('#28527a', 'o', '-', 1.7),
        'pd_mtp_off': ('#d26a7a', '^', '--', 1.7),
    }
    # InferenceX convention for disaggregated configs: normalize PD by
    # decode-chip count, with the apples-to-apples disclosure below.
    # Colocated is normalized by its total GPU count.
    sources = (
        ('colocated_mtp_off', 'total_gpus', 'COLOCATED MTP OFF'),
        ('pd_mtp_off', 'decode_gpus', 'PD SCALE-OUT MTP OFF'),
    )
    for group, normalization, label in sources:
        rows = group_rows(run_dir, group, concurrencies, measured_per_client,
                          warmup_per_client, normalization)
        summary[group] = rows
        color, marker, linestyle, lw = styles[group]
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, marker + linestyle, color=color, lw=lw, ms=4.2, label=label)
    ax.set_xlabel('Interactivity P90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.set_title('Kimi-K2.6 W4A16 on H200 (analytical projection), 1K/128, '
                 'TP8/PP1/EP8 all pools, closed-loop clients')
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=7.5)
    fig.text(
        0.01, -0.02,
        'Analytical H200 roofline projection, not a measured run. Arrivals: '
        'zero-think-time closed loop - C clients, one outstanding request '
        'each, successor injected at the client\'s own completion instant; '
        f'{measured_per_client} measured + {warmup_per_client} warmup '
        'requests per client\n'
        '(warmup discarded). Client-visible attribution: PD decode-slot '
        'wait after the KV handoff is TTFT, not an inter-token gap. MTP '
        'OFF: Kimi-K2.6 declares zero MTP layers (no MTP support anywhere '
        'for Kimi). Serving dtype W4A16:\n'
        'INT4 group-32 routed experts, BF16 for all other modules. PD KV '
        'handoff: scale-out 400GbE RoCEv2 x8 NIC via the ProtoBridge H200 '
        'theory preset (ring 160 GB/s, 40us RDMA startup, 0.95 wire '
        'efficiency) - UNCALIBRATED,\n'
        'development-only, not a measured fabric profile. Y: total '
        'generated-token throughput per GPU; colocated / 8 total GPUs, PD '
        '/ 8 decode GPUs (of 16) per InferenceX disaggregated convention - '
        'per-decode-GPU is not\n'
        'apples-to-apples with per-total-GPU. X: 1 / P90 TPOT.',
        fontsize=6.3, va='top')
    for ext in ('png', 'pdf'):
        fig.savefig(run_dir / f'colocated_vs_pd_closedloop_mtp_off.{ext}',
                    dpi=220, bbox_inches='tight')
    summary['assumptions'] = {
        'projection': 'analytical H200 roofline projection, not a measured run',
        'arrival_semantics': (
            'zero-think-time closed-loop: C clients, one outstanding request '
            'each, successor injected at the client completion instant; '
            'interleaved chain assignment; first request per client at t=0'),
        'client_visible_attribution': (
            'TTFT/TPOT attributed to the terminal pool: PD decode-slot wait '
            'after the KV handoff is TTFT; TPOT divides the decode pool\'s '
            'own production window by the gaps it produced'),
        'serving_dtype': (
            'W4A16: INT4 group-32 routed expert weights, BF16 for every '
            'other module; BF16 compute'),
        'mtp': ('off; Kimi-K2.6 declares zero MTP layers and Kimi MTP is '
                'unsupported everywhere in this simulator'),
        'expert_routing': {
            'policy': 'seeded_assumption (uniform seeded routing; Kimi '
                      'production datasets carry no MoE dynamic inputs)',
            'source_id': ROUTING_SOURCE_ID,
            'seed': ROUTING_SEED,
        },
        'pd_transfer_scale_out': {
            'link': 'infiniband (scale-out 400GbE RoCEv2, 8 NICs per node)',
            'backend': 'protobridge',
            'profile_sha256': transfer_profile_sha256,
            'ring_bw_bytes_per_us': transfer_profile.model_parameters[
                'ring_bw_bytes_per_us'],
            'rdma_startup_us': transfer_profile.protocol_policy.simple_startup_us,
            'wire_efficiency': transfer_profile.model_parameters[
                'wire_efficiency'],
            'source': ('vendored upstream ProtoBridge make_h200_roce_theory '
                       'preset; spec-derived, UNCALIBRATED (ns-3-anchored, '
                       'not measured hardware)'),
            'status': 'development_only; results are marked '
                      'projected_with_protobridge_development_only',
        },
        'throughput_per_gpu_normalization': (
            'colocated: total token throughput / total GPUs (8); PD: total '
            'token throughput / decode GPUs (8 of 16), InferenceX '
            'disaggregated convention - per-decode-GPU is not '
            'apples-to-apples with per-total-GPU'),
        'parallelism': 'TP8/PP1/EP8 for the colocated pool and for both PD roles',
        'pd_transfer_contract': (
            'compressed MLA kv_c||k_pe (61 layers x 576 x 2 B) = 70,272 '
            'B/token per TP rank, replicated single-head caches '
            'transferred over 8 identity rank pairs; EP-independent per '
            'pinned vLLM semantics'),
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return run_dir / 'colocated_vs_pd_closedloop_mtp_off.png'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrencies', type=int, nargs='+',
                        default=(1, 2, 4, 8))
    parser.add_argument('--measured-per-client', type=int, default=10)
    parser.add_argument('--warmup-per-client', type=int, default=2)
    parser.add_argument('--max-workers', type=int, default=4)
    parser.add_argument('--tag', default=None,
                        help='run directory name under OUTDIR; defaults to '
                             'm<measured>w<warmup>')
    args = parser.parse_args()
    if (min(args.concurrencies) < 1 or args.measured_per_client < 1
            or args.warmup_per_client < 1 or args.max_workers < 1):
        parser.error('concurrencies, per-client counts, and workers must be positive')
    tag = args.tag or f'm{args.measured_per_client}w{args.warmup_per_client}'
    run_dir = OUTDIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = tuple(sorted(set(args.concurrencies)))
    jobs = [(group, c) for group in GROUPS for c in concurrencies]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(run_point, group, c, run_dir,
                                  args.measured_per_client,
                                  args.warmup_per_client): (group, c)
                      for group, c in jobs}
        for future in concurrent.futures.as_completed(future_map):
            group, c = future_map[future]
            result = future.result()
            print(f'{group} C={c} {result[4]} elapsed={result[3]:.1f}s', flush=True)
    print(summarize_and_plot(run_dir, concurrencies,
                             args.measured_per_client,
                             args.warmup_per_client))


if __name__ == '__main__':
    main()
