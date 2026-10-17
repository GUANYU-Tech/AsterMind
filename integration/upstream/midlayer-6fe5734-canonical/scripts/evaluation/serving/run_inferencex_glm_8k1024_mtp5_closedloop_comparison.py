#!/usr/bin/env python3
"""Closed-loop colocated vs PD GLM-5.2 8K/1K sweeps with MTP off/on.

Same zero-think-time closed-loop client contract and ProtoBridge scale-out
PD handoff as the 1K/128 sweep; only the request-length targets change
(8192/1024 with random_range_ratio 0.8, i.e. uniform [6553, 8192] input and
[819, 1024] output tokens).

Usage:
  python .../run_inferencex_glm_8k1024_mtp5_closedloop_comparison.py \
      --prediction-method {roofline,midlayer} --tag <results subdir> [group:cN ...]

With no arguments all 32 points run.  SWEEP_WORKERS sets the subprocess
parallelism (default 4); points whose outputs already exist are skipped, so
a memory probe phase (big points first) can be followed by a wide rerun.
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
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path('/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python')
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
GROUPS = ('colocated_mtp_off', 'colocated_mtp_on', 'pd_mtp_off', 'pd_mtp_on')
OUTDIR = ROOT / 'experiment_logs/inferencex_style_glm_8k1024_mtp5_closedloop_tp8_ep1'
DATA_ROOT = ROOT / 'serving_dataset/synthetic/inferencex_style/glm_5_2_h200_8k1024_mtp5_closedloop_tp8_ep1'
COLOCATED_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_aggregate_tp8_runtime_memory.json'
PD_POLICY = ROOT / 'configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_pd_tp8_ep1_runtime_memory.json'
MODEL = ROOT / 'configs/model/glm-5.2-fp8.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'
TRANSFER_LINK = 'infiniband'
PROPOSAL_DEPTH = 5
MEAN_ACCEPTANCE_RATE = '0.82'  # 5 speculative tokens * 0.82 = 4.1 expected accepts.
ACCEPTANCE_SOURCE = 'glm5-top1-depth5-mean4.1-v1'
TARGET_INPUT_TOKENS = 8192
TARGET_OUTPUT_TOKENS = 1024
MAX_WORKERS = int(os.environ.get('SWEEP_WORKERS', '4'))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_requests(path: Path, count: int) -> None:
    rng = np.random.RandomState(0)
    inputs = rng.randint(int(0.8 * TARGET_INPUT_TOKENS),
                         TARGET_INPUT_TOKENS + 1, size=count)
    outputs = rng.randint(int(0.8 * TARGET_OUTPUT_TOKENS),
                          TARGET_OUTPUT_TOKENS + 1, size=count)
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
        'workload_id': f'glm_5_2_h200_8k1024_closedloop_{group}_c{concurrency}',
        'model': 'GLM-5.2-FP8',
        'hardware': ('8x NVIDIA H200' if deployment == 'colocated'
                     else '8x NVIDIA H200 prefill + 8x NVIDIA H200 decode'),
        'deployment': 'colocated' if deployment == 'colocated' else 'pd_disagg',
        'parallelism': ('TP8/PP1/EP1 colocated' if deployment == 'colocated'
                        else 'P/D: TP8/PP1/EP1, one instance per role'),
        'pd_transfer': (None if deployment == 'colocated' else
                        'send_recv over scale-out 400GbE RoCEv2, ProtoBridge '
                        'theory preset (development-only, uncalibrated)'),
        'target_input_tokens': TARGET_INPUT_TOKENS,
        'target_output_tokens': TARGET_OUTPUT_TOKENS,
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
                concurrency: int, output: Path,
                prediction_method: str) -> list[str]:
    deployment, mtp = group.split('_mtp_')
    mtp = mtp == 'on'
    cmd = [
        str(PYTHON), str(ROOT / 'scripts/evaluation/serving/simulation_runner_entrypoint.py'),
        '--canonical-requests', str(csv_path), '--canonical-requests-sha256', sha256(csv_path),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE), '--hardware-config-sha256', sha256(HARDWARE),
        '--backend', 'vllm', '--prediction-method', prediction_method,
        '--collective-backend', 'analytical_roofline_v1', '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '8', '--prefill-pp-size', '1', '--max-batched-tokens', '8192',
        '--scheduler-pipeline-depth', '1',
        '--runtime-memory-policy', str(policy), '--kv-block-size-tokens', '16',
        '--max-model-len', '9216', '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'inferencex-style-8k1024-mtp5-routing-v1',
        '--expert-routing-seed', '0', '--max-requests', str(concurrency),
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
    if mtp:
        cmd.extend([
            '--mtp-num-speculative-tokens', str(PROPOSAL_DEPTH),
            '--mtp-acceptance-policy', 'maximum_entropy',
            '--mtp-mean-acceptance-rate', MEAN_ACCEPTANCE_RATE,
            '--mtp-acceptance-seed', '0',
            '--mtp-acceptance-source-id', ACCEPTANCE_SOURCE,
        ])
    return cmd


def run_point(group: str, concurrency: int, run_dir: Path,
              prediction_method: str):
    measured, warmup, policy = prepare_case(group, concurrency)
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
            subprocess.run(command_for(group, source, policy, concurrency, destination,
                                       prediction_method),
                           cwd=ROOT, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=stderr)
    return group, concurrency, output, time.monotonic() - start, 'completed'


def qvalue(result: dict, metric: str, percentile: int) -> float:
    quantiles = result['metrics']['latency_distributions'][metric]['quantiles']
    return quantiles['values_s'][quantiles['percentiles'].index(percentile)]


def group_rows(outdir: Path, group: str, normalization: str) -> list[dict]:
    rows = []
    for concurrency in CONCURRENCIES:
        path = outdir / group / f'c{concurrency}.json'
        if not path.exists():
            print(f'warning: {group} C={concurrency} missing, skipped')
            continue
        result = json.loads(path.read_text())
        total_throughput = result['metrics']['token_throughput_per_s']
        rows.append({
            'C': concurrency,
            'measured_requests': 10 * concurrency,
            'warmup_requests': 2 * concurrency,
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


def summarize_and_plot(run_dir: Path) -> Path:
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
        'colocated_mtp_on': ('#d26a7a', 's', '-', 1.7),
        'pd_mtp_off': ('#28527a', '^', '--', 1.7),
        'pd_mtp_on': ('#d26a7a', 'D', '--', 1.7),
    }
    # InferenceX convention for disaggregated configs: normalize PD by
    # decode-chip count, with the apples-to-apples disclosure below.
    # Colocated is normalized by its total GPU count.
    sources = (
        ('colocated_mtp_off', 'total_gpus', 'COLOCATED MTP OFF'),
        ('colocated_mtp_on', 'total_gpus', 'COLOCATED MTP ON'),
        ('pd_mtp_off', 'decode_gpus', 'PD SCALE-OUT MTP OFF'),
        ('pd_mtp_on', 'decode_gpus', 'PD SCALE-OUT MTP ON'),
    )
    for group, normalization, label in sources:
        rows = group_rows(run_dir, group, normalization)
        summary[group] = rows
        color, marker, linestyle, lw = styles[group]
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, marker + linestyle, color=color, lw=lw, ms=4.2, label=label)
    ax.set_xlabel('Interactivity P90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.set_title('GLM-5.2 FP8 on H200 (analytical projection), 8K/1K, '
                 'TP8/EP1 all pools, closed-loop clients')
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=7.5)
    fig.text(
        0.01, -0.02,
        'Arrivals: zero-think-time closed loop - C clients, one outstanding '
        'request each, successor injected at the client\'s own completion '
        'instant; 10 measured + 2 warmup requests per client (warmup\n'
        'discarded). Client-visible attribution: PD decode-slot wait after '
        'the KV handoff is TTFT, not an inter-token gap. PD KV handoff: '
        'scale-out 400GbE RoCEv2 x8 NIC via the ProtoBridge H200 theory\n'
        'preset (ring 160 GB/s, 40us RDMA startup, 0.95 wire efficiency) - '
        'UNCALIBRATED, development-only, not a measured fabric profile. '
        'MTP on: propose 5 tokens, top-1, max 6 committed per verify;\n'
        'assumed mean 4.1 accepted speculative tokens (acceptance rate '
        '0.82) - an explicit assumption (glm5-top1-depth5-mean4.1-v1), not '
        'a measured trace. Y: total generated-token throughput per GPU; '
        'colocated / 8 total\n'
        'GPUs, PD / 8 decode GPUs (of 16) per InferenceX disaggregated '
        'convention - per-decode-GPU is not apples-to-apples with '
        'per-total-GPU. X: 1 / P90 TPOT.',
        fontsize=6.3, va='top')
    for ext in ('png', 'pdf'):
        fig.savefig(run_dir / f'colocated_vs_pd_closedloop_mtp5.{ext}', dpi=220,
                    bbox_inches='tight')
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
        'mtp_on_contract': {
            'num_speculative_tokens': PROPOSAL_DEPTH,
            'top1': True,
            'maximum_total_tokens_per_verify': PROPOSAL_DEPTH + 1,
            'mean_accepted_speculative_tokens': 4.1,
            'mean_acceptance_rate': MEAN_ACCEPTANCE_RATE,
            'status': 'explicit assumption, not a measured acceptance trace',
            'acceptance_source_id': ACCEPTANCE_SOURCE,
        },
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
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return run_dir / 'colocated_vs_pd_closedloop_mtp5.png'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-method', required=True,
                        choices=('roofline', 'midlayer'))
    parser.add_argument('--tag', required=True,
                        help='results subdirectory under OUTDIR')
    parser.add_argument('points', nargs='*', help='group:cN selections')
    args = parser.parse_args()
    run_dir = OUTDIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.points)
    jobs = [(group, c) for group in GROUPS for c in CONCURRENCIES
            if not selected or f'{group}:c{c}' in selected]
    # Long poles first: descending C keeps the pool busy on the big points.
    jobs.sort(key=lambda job: -job[1])
    print(f'running {len(jobs)} points with {MAX_WORKERS} workers', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {pool.submit(run_point, group, c, run_dir,
                                  args.prediction_method): (group, c)
                      for group, c in jobs}
        for future in concurrent.futures.as_completed(future_map):
            group, c = future_map[future]
            result = future.result()
            print(f'{group} C={c} {result[4]} elapsed={result[3]:.1f}s', flush=True)
    if selected:
        print('partial point selection; skipping summarize_and_plot')
    else:
        print(summarize_and_plot(run_dir))


if __name__ == '__main__':
    main()
