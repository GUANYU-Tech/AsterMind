#!/usr/bin/env python3
"""Closed-loop colocated vs PD DeepSeek-V3 1K/128 sweeps with MTP off/on.

Every group runs under the InferenceX-style zero-think-time closed-loop
client contract: C clients, each with at most one outstanding request, the
successor injected at the client's own completion instant
(--closed-loop-client-count C).  This mirrors the GLM-5.2/Kimi-K2.6
InferenceX comparisons so the curves are directly comparable.

DeepSeek-V3 is the official FP8 release (block-wise 128x128 e4m3 weights,
dynamic activations).  The simulator's FP8 weight contract models one
storage byte per element and has no scale-granularity dimension, which the
config loader states explicitly; no per-tensor requantization variant is
invented.

MTP on means the official single NextN (deepseek_mtp) draft layer
proposing 1 token per step, top-1, with an explicit declared acceptance
assumption (dsv3-top1-depth1-mean0.87-v1): the DeepSeek-V3 technical
report (arXiv:2412.19437) reports 85-90% MTP acceptance across generation
topics; 0.87 is the approximate midpoint and is an ASSUMPTION, not a
measured trace on this workload.

PD KV handoff uses the registered H200 scale-out ProtoBridge send_recv
profile (400GbE RoCEv2 theory preset, development-only).  Client-visible
TTFT/TPOT are attributed to the terminal pool: the decode slot wait after
a prefill handoff is TTFT, not an inter-token gap.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
from functools import partial
from evaluation.serving.experiments.inferencex_profiles import resolve_profile
from evaluation.serving.experiments.inferencex_comparison import run_point, qvalue, group_rows

PROFILE = resolve_profile('inferencex_deepseek_v3_1k128_closedloop')
PYTHON = PROFILE['constants']['PYTHON']
CONCURRENCIES = PROFILE['constants']['CONCURRENCIES']
GROUPS = PROFILE['constants']['GROUPS']
OUTDIR = PROFILE['constants']['OUTDIR']
DATA_ROOT = PROFILE['constants']['DATA_ROOT']
COLOCATED_POLICY = PROFILE['constants']['COLOCATED_POLICY']
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


def summarize_and_plot(run_dir: Path, concurrencies, measured_per_client: int,
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
        rows = group_rows(run_dir, group, concurrencies, measured_per_client,
                          warmup_per_client, normalization)
        summary[group] = rows
        color, marker, linestyle, lw = styles[group]
        x = [row['interactivity_p90_tok_s_user'] for row in rows]
        y = [row['total_token_throughput_per_gpu_tok_s'] for row in rows]
        ax.plot(x, y, marker + linestyle, color=color, lw=lw, ms=4.2, label=label)
    ax.set_xlabel('Interactivity P90 (tok/s/user)')
    ax.set_ylabel('Total Token Throughput / GPU (tok/s/GPU)')
    ax.set_title('DeepSeek-V3 FP8 on H200 (analytical projection), 1K/128, '
                 'TP8/EP1 all pools, closed-loop clients')
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=7.5)
    fig.text(
        0.01, -0.02,
        'Arrivals: zero-think-time closed loop - C clients, one outstanding '
        'request each, successor injected at the client\'s own completion '
        f'instant; {measured_per_client} measured + {warmup_per_client} '
        'warmup requests per client (warmup\n'
        'discarded). Client-visible attribution: PD decode-slot wait after '
        'the KV handoff is TTFT, not an inter-token gap. PD KV handoff: '
        'scale-out 400GbE RoCEv2 x8 NIC via the ProtoBridge H200 theory\n'
        'preset (ring 160 GB/s, 40us RDMA startup, 0.95 wire efficiency) - '
        'UNCALIBRATED, development-only, not a measured fabric profile. '
        'DeepSeek-V3: official block-wise 128x128 e4m3 FP8 weights,\n'
        'dynamic activations; the simulator FP8 contract models 1 '
        'byte/element weight storage and has no scale-granularity '
        'dimension. MTP on: official NextN layer, propose 1 token,\n'
        'top-1, max 2 committed per verify; assumed mean acceptance rate '
        '0.87 - an explicit ASSUMPTION (dsv3-top1-depth1-mean0.87-v1, '
        'DeepSeek-V3 tech report 85-90% range), not a\n'
        'measured trace. Y: total generated-token throughput per GPU; '
        'colocated / 8 total GPUs, PD / 8 decode GPUs (of 16) per '
        'InferenceX disaggregated\n'
        'convention - per-decode-GPU is not apples-to-apples with '
        'per-total-GPU. X: 1 / P90 TPOT.',
        fontsize=6.3, va='top')
    for ext in ('png', 'pdf'):
        fig.savefig(run_dir / f'colocated_vs_pd_closedloop_mtp1.{ext}', dpi=220,
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
        'fp8_contract': (
            'official DeepSeek-V3 FP8 release (block-wise 128x128 e4m3 '
            'weights, dynamic activations, pinned revision e815299b); the '
            'simulator FP8 weight contract models one storage byte per '
            'element and has no scale-granularity dimension, so block-wise '
            'vs per-tensor scaling does not change the model; no '
            'requantized deployment variant is invented'),
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
            'mean_accepted_speculative_tokens': 0.87,
            'mean_acceptance_rate': MEAN_ACCEPTANCE_RATE,
            'status': ('explicit ASSUMPTION (DeepSeek-V3 technical report '
                       'arXiv:2412.19437 reports 85-90% MTP acceptance; '
                       '0.87 midpoint), not a measured acceptance trace'),
            'acceptance_source_id': ACCEPTANCE_SOURCE,
        },
        'throughput_per_gpu_normalization': (
            'colocated: total token throughput / total GPUs (8); PD: total '
            'token throughput / decode GPUs (8 of 16), InferenceX '
            'disaggregated convention - per-decode-GPU is not '
            'apples-to-apples with per-total-GPU'),
        'parallelism': 'TP8/PP1/EP1 for colocated and for both PD roles',
        'pd_transfer_contract': (
            'compressed MLA kv_c||k_pe only (61 layers x 576 x 2 B) = '
            '70,272 B/token per TP rank; no DSA index cache (GLM-5.2 '
            'specific); replicated single-head caches transferred over 8 '
            'identity rank pairs; EP-independent per pinned vLLM semantics'),
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return run_dir / 'colocated_vs_pd_closedloop_mtp1.png'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concurrencies', type=int, nargs='+',
                        default=list(CONCURRENCIES))
    parser.add_argument('--measured-per-client', type=int, default=3)
    parser.add_argument('--warmup-per-client', type=int, default=1)
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--groups', type=str, nargs='+', default=list(GROUPS),
                        choices=GROUPS)
    parser.add_argument('--tag', default=None,
                        help='results subdirectory tag (default '
                             'm<measured>w<warmup>)')
    args = parser.parse_args()
    if (min(args.concurrencies) < 1 or args.measured_per_client < 1
            or args.warmup_per_client < 1 or args.max_workers < 1):
        parser.error('concurrencies, per-client counts, and workers must be '
                     'positive')
    tag = args.tag or f'm{args.measured_per_client}w{args.warmup_per_client}'
    run_dir = OUTDIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    concurrencies = tuple(sorted(set(args.concurrencies)))
    jobs = [(group, c) for group in args.groups for c in concurrencies]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        future_map = {pool.submit(run_point, group, c,
                                  args.measured_per_client,
                                  args.warmup_per_client,
                                  run_dir): (group, c)
                      for group, c in jobs}
        for future in concurrent.futures.as_completed(future_map):
            group, c = future_map[future]
            result = future.result()
            print(f'{group} C={c} {result[4]} elapsed={result[3]:.1f}s', flush=True)
    if set(args.groups) == set(GROUPS):
        print(summarize_and_plot(run_dir, concurrencies,
                                 args.measured_per_client,
                                 args.warmup_per_client))
    else:
        print(f'partial group set {sorted(args.groups)}; skipping '
              'summarize_and_plot (requires all four groups)')


if __name__ == '__main__':
    main()
