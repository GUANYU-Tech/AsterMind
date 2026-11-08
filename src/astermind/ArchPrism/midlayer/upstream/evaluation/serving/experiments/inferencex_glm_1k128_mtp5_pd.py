#!/usr/bin/env python3
"""Compare colocated and PD GLM-5.2 1K/128 sweeps with MTP off/on."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
from functools import partial
from evaluation.serving.experiments.inferencex_profiles import resolve_profile
from evaluation.serving.experiments.inferencex_comparison import run_point, qvalue

PROFILE = resolve_profile('inferencex_glm_1k128_mtp5_pd')
PYTHON = PROFILE['constants']['PYTHON']
CONCURRENCIES = PROFILE['constants']['CONCURRENCIES']
GROUPS = PROFILE['constants']['GROUPS']
OUTDIR = PROFILE['constants']['OUTDIR']
DATA_ROOT = PROFILE['constants']['DATA_ROOT']
COLOCATED_POLICY = PROFILE['constants']['COLOCATED_POLICY']
PD_POLICY = PROFILE['constants']['PD_POLICY']
MODEL = PROFILE['constants']['MODEL']
HARDWARE = PROFILE['constants']['HARDWARE']
PROPOSAL_DEPTH = PROFILE['constants']['PROPOSAL_DEPTH']
MEAN_ACCEPTANCE_RATE = PROFILE['constants']['MEAN_ACCEPTANCE_RATE']
ACCEPTANCE_SOURCE = PROFILE['constants']['ACCEPTANCE_SOURCE']
MAX_WORKERS = PROFILE['constants']['MAX_WORKERS']
run_point = partial(run_point, PROFILE)







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
