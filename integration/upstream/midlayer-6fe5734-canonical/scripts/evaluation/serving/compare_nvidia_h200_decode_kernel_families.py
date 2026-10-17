#!/usr/bin/env python3
"""Compare one simulated H200 DSV3-L32 TP4 engine step with real nsys families.

Real: ``decode_summary.json`` written by ``summarize_nvidia_h200_nsys_decode_windows.py``
for the vLLM 0.24 server (rank 0, per-step busy time by kernel family).
Simulated: one steady decode step at batch C (``--phase decode``) or the
first all-prefill step of C fresh prompts (``--phase prefill``) of the same
TP4 deployment, run in process (SimulationResult with full step retention), with
every rank-0 kernel interval grouped by KernelSpec kind/variant.  Families are
merged where the two stacks split work differently (the real allreduce kernel
fuses RMSNorm; the real stack quantises activations to FP8).  Only C=1 and C=4
real windows are pure decode; larger windows mix chunked-prefill work.
Evaluation only: no timing feeds any model parameter.

Usage:
    PYTHONPATH=. python scripts/evaluation/serving/compare_nvidia_h200_decode_kernel_families.py \
        --real-summary <decode_summary.json> --batch 1 --prompt-tokens 7373 \
        --method midlayer --phase decode --real-key C=1 --output <json>
"""
import argparse
import collections
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / 'configs/model/deepseek-v3-l32.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'
POLICY = ROOT / ('configs/runtime_memory_policy/inferencex/'
                 'deepseek_v3_l32_nvidia_h200_aggregate_tp4_ep1_runtime_memory.json')
# Per process: concurrent comparisons (and worktrees) must not share the
# policy file bound to their own hardware-config SHA.
SCRATCH = Path(f'/tmp/h200_step_family_compare_{os.getpid()}')
OUT_TOKENS = 24

# The audited vLLM 0.24.0 server's CUDA-graph configuration (server log
# /mnt/public/yangxinhao/h200_microbench/nsys/dsv3l32_tp4_pure_decode_uniform_r2/
# server.log and .../logs/serve_dsv3_l32_uniform.log, "Initializing a V1 LLM
# engine ... compilation_config"): cudagraph_mode FULL_AND_PIECEWISE and
# cudagraph_capture_sizes below (max_cudagraph_capture_size 256).  Piecewise
# graphs split out compilation_config.splitting_ops, which for this model are
# vllm::unified_mla_attention_with_output (the MLA attention op) and
# vllm::unified_mla_kv_cache_update (the MLA KV-cache write): workload kinds
# flash_attn and kv_update run eagerly between the piecewise graph pieces.
GRAPH_ARGS = (
    '--graph-mode', 'FULL_AND_PIECEWISE',
    '--graph-capture-sizes', *(str(size) for size in (
        1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
        120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224,
        232, 240, 248, 256)),
    '--graph-piecewise-eager-workload-kinds', 'flash_attn', 'kv_update',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(batch, prompt_tokens, method):
    """Write B equal-arrival requests and a TP4 runtime policy bound to them."""
    case_dir = SCRATCH / f'c{batch}_isl{prompt_tokens}_{method}'
    case_dir.mkdir(parents=True, exist_ok=True)
    csv_path = case_dir / 'requests.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Query Time Stamp [ms]', 'In Token [tokens]',
                         'Out Token [tokens]', 'Cached hit [tokens]'])
        for index in range(batch):
            writer.writerow((0, prompt_tokens + index % 128, OUT_TOKENS, 0))
    raw_policy = json.loads(POLICY.read_text())
    binding = raw_policy['scheduler_pools'][0]['binding']
    binding['hardware_config_sha256'] = sha256(HARDWARE)
    binding['model_config_sha256'] = sha256(MODEL)
    binding['max_requests'] = batch
    binding['expert_parallel_size'] = 1
    binding['max_model_len'] = 9216
    binding['max_batched_tokens'] = 32768
    policy_path = case_dir / 'policy.json'
    policy_path.write_text(json.dumps(raw_policy, indent=2) + '\n')
    return csv_path, policy_path, case_dir / 'result.json'


# (group, real families, predicate over (kind, variant))
GROUPS = (
    ('gemm', ('dense_fp8_gemm', 'gemm_other'), lambda k, v: k == 'GEMM'),
    ('moe_grouped_gemm', ('moe_grouped_gemm',), lambda k, v: k == 'FUSED_MOE'),
    ('moe_routing', ('moe_routing',),
     lambda k, v: k == 'VECTOR' and v.startswith('moe_')),
    ('attention', ('attention',), lambda k, v: k in ('FLASH_ATTN', 'KV_CACHE')),
    ('allreduce_norm', ('allreduce', 'norm_rope_act'),
     lambda k, v: k == 'COMMUNICATION' or (k == 'VECTOR' and v in (
         'rmsnorm', 'add_rmsnorm', 'mla_rope', 'silu'))),
    ('other', ('quant', 'copy_misc', 'other'), None),
)


def group_of(kind, variant):
    for name, _, predicate in GROUPS:
        if predicate is not None and predicate(kind, variant or ''):
            return name
    return 'other'


def simulate_step(batch, prompt_tokens, method, phase):
    """Return the retained SimulationResult step compared for ``phase``."""
    sys.path.insert(0, str(ROOT))
    import evaluation.serving.simulation.simulation_runner as runner

    csv_path, policy_path, output_path = prepare(batch, prompt_tokens, method)
    captured = {}
    original = runner.system_resource_manifest

    def capture(deployment, requests, result):
        captured['result'] = result
        return original(deployment, requests, result)

    runner.system_resource_manifest = capture
    sys.argv = [
        'compare_nvidia_h200_decode_kernel_families',
        '--canonical-requests', str(csv_path),
        '--canonical-requests-sha256', sha256(csv_path),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE),
        '--hardware-config-sha256', sha256(HARDWARE),
        '--backend', 'vllm', '--prediction-method', method,
        '--collective-backend', 'analytical_roofline_v1',
        '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '4', '--prefill-pp-size', '1',
        '--max-batched-tokens', '32768', '--scheduler-pipeline-depth', '1',
        '--runtime-memory-policy', str(policy_path),
        '--kv-block-size-tokens', '16', '--max-model-len', '9216',
        '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'inferencex-style-1k128-mtp1-routing-v1',
        '--expert-routing-seed', '0', '--max-requests', str(batch),
        '--no-enable-prefix-caching', '--enable-chunked-prefill',
        '--closed-loop-client-count', str(batch),
        '--deployment', 'colocated', '--pipeline-ranks-per-node', '8',
        '--step-retention', 'full', '--output', str(output_path),
        *GRAPH_ARGS,
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        runner.main()
    result = captured['result']
    if phase == 'decode':
        steps = [
            step for step in result.steps
            if {s.phase for s in step.backend_step.schedule.requests} == {'decode'}
            and step.backend_step.schedule.token_count == batch]
    elif phase == 'prefill':
        # The first step prefills every prompt from offset zero in one chunk.
        steps = [
            step for step in result.steps
            if {s.phase for s in step.backend_step.schedule.requests} == {'prefill'}
            and len(step.backend_step.schedule.requests) == batch
            and step.backend_step.schedule.token_count
            == sum(prompt_tokens + index % 128 for index in range(batch))]
    else:
        raise ValueError(f'unknown phase {phase!r}')
    if not steps:
        raise RuntimeError(f'no pure {phase} step at batch {batch}')
    return steps[-1]


def simulate(batch, prompt_tokens, method, phase):
    step = simulate_step(batch, prompt_tokens, method, phase)
    timeline = step.execution_timeline
    occurrences = dict(timeline.call_occurrence_counts)
    calls = {call.call_id: call for call in step.kernel_plan.calls}
    by_group = collections.Counter()
    by_kind = collections.Counter()
    by_algorithm = collections.Counter()
    attention_calls = collections.OrderedDict()
    for interval in timeline.intervals:
        if interval.rank != 0:
            continue
        spec = calls[interval.call_id].kernel_spec
        repeat = occurrences[interval.call_id] if interval.call_id in occurrences else 1
        us = (interval.end_time_s - interval.start_time_s) * 1e6 * repeat
        by_group[group_of(spec.kind.name, spec.variant)] += us
        by_kind[f'{spec.kind.name}/{spec.variant}'] += us
        # Transfers carry no compute launch (KernelSpec.launch is None).
        algorithm = ('no_compute_launch' if spec.launch is None
                     else spec.launch.algorithm)
        by_algorithm[f'{spec.kind.name}/{spec.variant}/{algorithm}'] += us
        if spec.kind.name == 'FLASH_ATTN':
            kv_lens = tuple(spec.attributes['kv_lens'])
            key = (spec.variant, algorithm, spec.launch.grid,
                   spec.launch.tiling.num_kv_splits, len(kv_lens),
                   min(kv_lens), max(kv_lens))
            if key not in attention_calls:
                attention_calls[key] = {
                    'variant': spec.variant, 'algorithm': algorithm,
                    'grid': list(spec.launch.grid),
                    'block': list(spec.launch.block),
                    'num_kv_splits': spec.launch.tiling.num_kv_splits,
                    'batch': len(kv_lens), 'kv_len_min': min(kv_lens),
                    'kv_len_max': max(kv_lens),
                    'q_tokens': sum(spec.attributes['q_lens']),
                    'calls': 0, 'us': 0.0}
            attention_calls[key]['calls'] += repeat
            attention_calls[key]['us'] += us
    for entry in attention_calls.values():
        entry['us_per_call'] = entry['us'] / entry['calls']
    return (timeline.duration_s * 1e6, by_group, by_kind, by_algorithm,
            list(attention_calls.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-summary', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--prompt-tokens', type=int, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--phase', choices=('decode', 'prefill'), required=True)
    parser.add_argument('--real-key', required=True,
                        help='decode_summary.json key, e.g. C=4')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    real = json.loads(args.real_summary.read_text())[args.real_key]
    step_us, sim_groups, sim_kinds, sim_algorithms, sim_attention = simulate(
        args.batch, args.prompt_tokens, args.method, args.phase)
    rows = []
    for name, families, _ in GROUPS:
        real_us = sum(real['family_us_per_step'][f]
                      for f in families if f in real['family_us_per_step'])
        rows.append({'group': name, 'real_families': list(families),
                     'real_us': real_us, 'sim_us': sim_groups[name],
                     'sim_minus_real_us': sim_groups[name] - real_us})
    report = {
        'phase': args.phase, 'batch': args.batch,
        'prompt_tokens': args.prompt_tokens,
        'method': args.method,
        'real_step_wall_us': real['step_wall_us'],
        'real_step_busy_us': real['step_busy_union_us'],
        'real_step_idle_us': real['step_idle_us'],
        'sim_step_us': step_us, 'groups': rows,
        'sim_by_kind_us': dict(sim_kinds.most_common()),
        'sim_by_kind_algorithm_us': dict(sim_algorithms.most_common()),
        'sim_flash_attn_calls': sim_attention,
        'production_or_evaluation_duration_consumed_by_model': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'C={args.batch}: real wall {real["step_wall_us"]:.0f} us '
          f'(busy {real["step_busy_union_us"]:.0f}), sim {step_us:.0f} us')
    for row in rows:
        print(f'  {row["group"]:<18} real {row["real_us"]:8.0f}  sim {row["sim_us"]:8.0f}  '
              f'diff {row["sim_minus_real_us"]:+8.0f}')


if __name__ == '__main__':
    main()
