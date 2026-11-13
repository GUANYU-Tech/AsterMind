#!/usr/bin/env python3
"""Compare colocated and scale-out PD GLM-5.2 1K/128 sweeps with MTP off/on.

PD KV handoff uses the registered H200 scale-out ProtoBridge send_recv
profile (400GbE RoCEv2 theory preset, development-only) instead of the
scale-up NVLink profile.  Colocated groups are reused verbatim from the
TP8/EP1 sweep because they involve no inter-pool transfer.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
from functools import partial
from evaluation.serving.experiments.inferencex_profiles import resolve_profile
from evaluation.serving.experiments.inferencex_comparison import run_point, qvalue, group_rows

PROFILE = resolve_profile('inferencex_glm_1k128_mtp5_pd_scaleout')
PYTHON = PROFILE['constants']['PYTHON']
CONCURRENCIES = PROFILE['constants']['CONCURRENCIES']
PD_GROUPS = PROFILE['constants']['PD_GROUPS']
OUTDIR = PROFILE['constants']['OUTDIR']
DATA_ROOT = PROFILE['constants']['DATA_ROOT']
NVLINK_OUTDIR = PROFILE['constants']['NVLINK_OUTDIR']
PD_POLICY = PROFILE['constants']['PD_POLICY']
MODEL = PROFILE['constants']['MODEL']
HARDWARE = PROFILE['constants']['HARDWARE']
TRANSFER_LINK = PROFILE['constants']['TRANSFER_LINK']
PROPOSAL_DEPTH = PROFILE['constants']['PROPOSAL_DEPTH']
MEAN_ACCEPTANCE_RATE = PROFILE['constants']['MEAN_ACCEPTANCE_RATE']
ACCEPTANCE_SOURCE = PROFILE['constants']['ACCEPTANCE_SOURCE']
MAX_WORKERS = PROFILE['constants']['MAX_WORKERS']
run_point = partial(run_point, PROFILE)
group_rows = partial(group_rows, PROFILE)

# TP8/EP1 sweep with the scale-up (NVLink) PD transfer: source of the reused
# colocated groups and of the reference PD-NVLink curves.


def summarize_and_plot() -> Path:
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
        (NVLINK_OUTDIR, 'colocated_mtp_off', 'total_gpus', 'COLOCATED MTP OFF'),
        (NVLINK_OUTDIR, 'colocated_mtp_on', 'total_gpus', 'COLOCATED MTP ON'),
        (OUTDIR, 'pd_mtp_off', 'decode_gpus', 'PD SCALE-OUT MTP OFF'),
        (OUTDIR, 'pd_mtp_on', 'decode_gpus', 'PD SCALE-OUT MTP ON'),
    )
    for outdir, group, normalization, label in sources:
        rows = group_rows(outdir, group, normalization)
        summary[group] = rows
        color, marker, linestyle, lw = styles[group]
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, marker + linestyle, color=color, lw=lw, ms=4.2, label=label)
    # Reference: the same PD sweeps with the scale-up (NVLink) transfer
    # profile, isolating the link assumption change.
    for group, label in (('pd_mtp_off', 'PD NVLINK MTP OFF (ref)'),
                         ('pd_mtp_on', 'PD NVLINK MTP ON (ref)')):
        rows = group_rows(NVLINK_OUTDIR, group, 'decode_gpus')
        summary[f'{group}_nvlink_reference'] = rows
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, ':', color='#7f7f7f', lw=1.1, ms=3,
                marker='x' if group.endswith('off') else '+', label=label)
    ax.set_xlabel('Interactivity P90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.set_title('GLM-5.2 FP8 on H200 (analytical projection), 1K/128, '
                 'TP8/EP1 all pools')
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=7.5)
    fig.text(
        0.01, -0.02,
        'PD KV handoff: scale-out 400GbE RoCEv2 x8 NIC via the ProtoBridge '
        'H200 theory preset (ring 160 GB/s, 40us RDMA startup, 0.95 wire\n'
        'efficiency) - UNCALIBRATED, development-only, not a measured fabric '
        'profile; dotted grey: same PD sweeps with the previous scale-up '
        'NVLink profile. MTP on: propose 5 tokens, top-1,\n'
        'max 6 committed per verify; assumed mean 4.1 accepted speculative '
        'tokens (acceptance rate 0.82) - an explicit assumption '
        '(glm5-top1-depth5-mean4.1-v1), not a measured trace.\n'
        'Arrivals: 10xC requests at t=0, scheduler cap C (static backlog, '
        'not closed-loop). Y: total generated-token throughput per GPU; '
        'colocated / 8 total GPUs, PD / 8 decode GPUs (of 16) per\n'
        'InferenceX disaggregated convention - per-decode-GPU is not '
        'apples-to-apples with per-total-GPU. X: 1 / P90 TPOT. Warmup 2xC '
        'requests discarded.',
        fontsize=6.3, va='top')
    for ext in ('png', 'pdf'):
        fig.savefig(OUTDIR / f'colocated_vs_pd_scaleout_mtp5.{ext}', dpi=220,
                    bbox_inches='tight')
    summary['assumptions'] = {
        'projection': 'analytical H200 roofline projection, not a measured run',
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
        'colocated_groups_source': str(NVLINK_OUTDIR),
    }
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return OUTDIR / 'colocated_vs_pd_scaleout_mtp5.png'


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    jobs = [(group, c) for group in PD_GROUPS for c in CONCURRENCIES]
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
