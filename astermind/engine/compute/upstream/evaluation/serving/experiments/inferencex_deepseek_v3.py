#!/usr/bin/env python3
"""Run the published DSV3/H200 8K/1K closed-loop framework recipes.

Six settings share five immutable, dataset-owner validated input cases.
vLLM uses its real Scheduler with eager execution; SGLang uses a source
mirror and decode CUDA graphs. MTP acceptance is a transferred external
workload statistic, not a measurement on these generated requests. PD
communication remains a development-only analytical projection.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from src.serving.kv_cache.kv_capacity import load_runtime_memory_policy
import subprocess
import sys

from evaluation.serving.experiments.inferencex_comparison import sha256


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
GROUPS = ('colocated_mtp_off', 'colocated_mtp_on', 'pd_mtp_off', 'pd_mtp_on')
DATA_ROOT = ROOT / 'serving_dataset/synthetic/inferencex_style/dsv3_h200_8k1024_closedloop_tp8_ep1'
MODEL = ROOT / 'configs/model/deepseek-v3.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'
TRANSFER_LINK = 'infiniband'
PROPOSAL_DEPTH = 3
MEAN_ACCEPTANCE_RATE = '0.87'  # per-token, i.i.d.: 3 speculative tokens * 0.87 = 2.61 expected accepts.
ACCEPTANCE_SOURCE = 'dsv3-top1-depth3-mean0.87-v1'
# MTP settings the sweep can model: (speculative tokens per verify, per-token
# acceptance by client concurrency, source id, status text).
#  - assumed_depth3_087: the original declared assumption above (any C).
#  - inferencex_sglang_h200_measured: the InferenceX SGLang H200 DSR1 8K/1K
#    TP8 MTP run itself (GitHub Actions run 34450728410, server_logs_*
#    artifacts): speculative_num_steps 2, speculative_num_draft_tokens 3,
#    eagle_topk 1, i.e. 2 speculative tokens per verify; per-token acceptance
#    = (mean "accept len" - 1) / 2 over steady-state decode batches
#    (#running-req >= 0.75 C, #token > 1000).  A measured workload statistic
#    of the compared run, not a duration.
MTP_PROFILES = {
    'assumed_depth3_087': (
        PROPOSAL_DEPTH, None, ACCEPTANCE_SOURCE,
        'ASSUMPTION: DeepSeek-V3 technical report (arXiv:2412.19437) reports '
        '85-90% per-token MTP acceptance; 0.87 midpoint applied i.i.d. per '
        'proposed token, not a measured trace'),
    'inferencex_sglang_h200_measured': (
        2, {4: '0.755', 8: '0.633', 16: '0.6265', 32: '0.6705', 64: '0.681'},
        'inferencex-sglang-h200-dsr1-8k1k-tp8-mtp-run34450728410',
        'MEASURED: per-concurrency mean accept length in the compared '
        'InferenceX SGLang server logs'),
    # Same run, one rate for every concurrency: the unweighted mean of the
    # five per-concurrency acceptances above ((0.755 + 0.633 + 0.6265 +
    # 0.6705 + 0.681) / 5 = 0.673), so C without an InferenceX point
    # (C=1, 2) can be simulated with the measured MTP setting.
    'inferencex_sglang_h200_mean': (
        2, '0.673',
        'inferencex-sglang-h200-dsr1-8k1k-tp8-mtp-run34450728410-mean',
        'MEASURED: mean over C=4..64 of the per-concurrency mean accept '
        'length in the compared InferenceX SGLang server logs'),
}


def mtp_settings(profile: str, concurrency: int):
    depth, rates, source, status = MTP_PROFILES[profile]
    if rates is None:
        return depth, MEAN_ACCEPTANCE_RATE, source, status
    if isinstance(rates, str):
        return depth, rates, source, status
    if concurrency not in rates:
        raise ValueError(
            f'MTP profile {profile!r} has no measured acceptance for C={concurrency}')
    return depth, rates[concurrency], source, status
MAX_WORKERS = os.cpu_count() or 1
# Expert-routing inputs.  "uniform" is the seeded uniform assumption.
# "dsr1_profile" samples each MoE layer's routed experts from the DeepSeek-R1
# per-layer expert-selection frequencies in
# serving_dataset/moe_expert_freq (canonical id deepseek-r1-671b, mapped to
# the executing deepseek-ai/DeepSeek-V3 config; source workload undocumented,
# filename says MT-bench).
ROUTING_CONFIGS = {
    'uniform': (
        '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'inferencex-style-8k1024-mtp1-routing-v1',
        '--expert-routing-seed', '0',
    ),
    'dsr1_profile': (
        '--expert-routing-policy', 'profile_observation',
        '--expert-routing-source-id', 'deepseek-r1-671b',
        '--expert-routing-profile-manifest',
        str(ROOT / 'serving_dataset/moe_expert_freq/manifest.json'),
    ),
    # Same profile, one histogram per MoE layer from that layer's own row
    # (every layer has its own gate); "dsr1_profile" draws every layer of a
    # step from the model-level layer-mean row.
    'dsr1_layer_profile': (
        '--expert-routing-policy', 'profile_observation',
        '--expert-routing-source-id', 'deepseek-r1-671b',
        '--expert-routing-profile-manifest',
        str(ROOT / 'serving_dataset/moe_expert_freq/manifest.json'),
        '--expert-routing-profile-resolution', 'per_moe_layer',
    ),
}
# CUDA-graph configurations the simulator can be told to model.  "none" runs
# every step eagerly.  "vllm_0_24_h200_server" is the audited vLLM 0.24.0
# H200 server's compilation_config (server log
# /mnt/public/yangxinhao/h200_microbench/logs/serve_dsv3_l32_uniform.log):
# cudagraph_mode FULL_AND_PIECEWISE, the capture sizes below, and the MLA
# attention / MLA KV-cache write ops as piecewise splitting ops.  Same values
# as scripts/evaluation/serving/run_nvidia_h200_deepseek_v3_l32_tp4_closedloop_sim.py.
GRAPH_CONFIGS = {
    'none': ('--graph-mode', 'NONE'),
    'vllm_0_24_h200_server': (
        '--graph-mode', 'FULL_AND_PIECEWISE',
        '--graph-capture-sizes', *(str(size) for size in (
            1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112,
            120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216,
            224, 232, 240, 248, 256)),
        '--graph-piecewise-eager-workload-kinds', 'flash_attn', 'kv_update',
    ),
}


def sglang_graph_config(mtp_depth: int) -> tuple[str, ...]:
    """SGLang decode CUDA graphs, prefill eager (FULL_DECODE_ONLY).

    Batch sizes: sglang v0.5.19 arg_groups/cuda_graph_hook.py
    generate_decode_cuda_graph_batch_sizes (cuda_graph_max_bs_decode 256 in the
    InferenceX server args; v0.5.12 uses the same lists).  The simulator's
    capture sizes are model-input tokens, i.e. batch size x tokens per request
    (EAGLE verify: speculative_num_draft_tokens = 3).
    """
    if mtp_depth:
        batch_sizes = (list(range(1, 9)) + list(range(10, 33, 2))
                       + list(range(40, 65, 4)) + list(range(72, 257, 8)))
        width = mtp_depth + 1
    else:
        batch_sizes = [1, 2, 4, 8, 12] + list(range(16, 257, 8))
        width = 1
    # EAGLE verify replays graphs captured at `width` tokens per request
    # (decode_cuda_graph_runner.py:323 captured_req_width).
    return ('--graph-mode', 'FULL_DECODE_ONLY', '--graph-capture-sizes',
            *(str(bs * width) for bs in batch_sizes),
            '--graph-uniform-decode-query-len', str(width))


# SGLang recipe facts for the InferenceX DSR1 H200 runs (server args, not
# latency): max_total_num_tokens as logged by the SGLang server ("KV Cache is
# allocated ... #tokens").  MTP on: the InferenceX v0.5.19 server log.  MTP
# off: no InferenceX log is available; logged by SGLang 0.5.12 with the
# InferenceX h200-fp8 8k1k recipe arguments on h200-tmp (2026-09-27,
# profiler_output/h200_sglang_dsr1/runs/mtpoff_20260927_1054/server.log;
# --load-format dummy allocates the same weight tensors).
SGLANG_STACK = {
    False: ('v0.5.12', 498057,
            'observed configuration fact: SGLang 0.5.12 server log '
            'max_total_num_tokens=498057 (InferenceX h200-fp8 8k1k recipe '
            'args, h200-tmp, dummy weights)'),
    True: ('v0.5.19', 471756,
           'observed configuration fact: SGLang v0.5.19 server log '
           'max_total_num_tokens=471756 (InferenceX DSR1 H200 MTP recipe)'),
}


def build_suite_command(group: str, csv_path: Path, policy: Path,
                concurrency: int, output: Path,
                prediction_method: str, graph_config: str,
                mtp_profile: str, routing: str,
                stack: str = 'vllm') -> list[str]:
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
        '--prefill-tp-size', '8', '--prefill-pp-size', '1', '--max-batched-tokens', '32768',
        '--scheduler-pipeline-depth', '1',
        '--runtime-memory-policy', str(policy), '--kv-block-size-tokens', '16',
        '--max-model-len', '9216', *ROUTING_CONFIGS[routing],
        '--max-requests', str(concurrency),
        '--no-enable-prefix-caching', '--enable-chunked-prefill', '--output', str(output),
        '--closed-loop-client-count', str(concurrency),
        *(sglang_graph_config(mtp_settings(mtp_profile, concurrency)[0]
                              if mtp else 0)
          if graph_config == 'sglang_h200_server'
          else GRAPH_CONFIGS[graph_config]),
    ]
    if stack == 'sglang':
        # The sglang stack setting was served by the now-deleted
        # sglang_scheduler_mirror backend; reject before any job launch.
        # The real-SGLang replay path lives in
        # scripts/evaluation/serving/agentx_session_simulate.py --backend sglang.
        raise ValueError(
            'sglang stack settings require the deleted sglang_scheduler_mirror '
            'backend; use scripts/evaluation/serving/agentx_session_simulate.py '
            '--backend sglang for the real-SGLang scheduler replay')
    if stack != 'vllm':
        raise ValueError(f'unknown execution stack: {stack!r}')
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
        depth, rate, source, _ = mtp_settings(mtp_profile, concurrency)
        cmd.extend([
            '--mtp-num-speculative-tokens', str(depth),
            '--mtp-acceptance-policy', 'maximum_entropy',
            '--mtp-mean-acceptance-rate', rate,
            '--mtp-acceptance-seed', '0',
            '--mtp-acceptance-source-id', source,
        ])
    return cmd



def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{label} fields must be exactly {sorted(expected)}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def load_suite(path, *, case_loader):
    """Resolve framework settings against dataset-owner validated inputs."""

    suite = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)
    _keys(suite, ('schema_version', 'suite_id', 'dataset_root', 'settings'), 'suite')
    if type(suite['schema_version']) is not int or suite['schema_version'] != 1:
        raise ValueError('unsupported closed-loop suite schema')
    if suite['suite_id'] != 'deepseek_v3_nvidia_h200_8k1024_closed_loop':
        raise ValueError('unexpected DSV3 suite identity')
    if suite['dataset_root'] != str(DATA_ROOT.relative_to(ROOT)):
        raise ValueError('DSV3 suite must bind the canonical synthetic root')
    if not isinstance(suite['settings'], list) or not suite['settings']:
        raise ValueError('suite settings must be a non-empty list')
    seen = set()
    inputs = {}
    jobs = []
    for setting in suite['settings']:
        _keys(setting, ('id', 'stack', 'group', 'concurrencies', 'graph_config',
                        'mtp_profile', 'expert_routing', 'prediction_method'), 'setting')
        stack, group = setting['stack'], setting['group']
        if stack not in ('vllm', 'sglang') or group not in GROUPS:
            raise ValueError('unknown framework or deployment group')
        if stack == 'sglang' and group.startswith('pd_'):
            raise ValueError('SGLang PD is not supported by this suite')
        identity = f'{stack}_{group}'
        if setting['id'] != identity or identity in seen:
            raise ValueError('setting identity must be unique and match framework/group')
        seen.add(identity)
        if setting['concurrencies'] != [4, 8, 16, 32, 64]:
            raise ValueError('published suite requires C=4,8,16,32,64')
        if setting['prediction_method'] != 'midlayer':
            raise ValueError('published suite uses the Midlayer predictor')
        if setting['graph_config'] != ('none' if stack == 'vllm' else 'sglang_h200_server'):
            raise ValueError('graph configuration disagrees with the published recipe')
        if setting['mtp_profile'] != 'inferencex_sglang_h200_mean':
            raise ValueError('published suite uses the declared depth-2 acceptance profile')
        if setting['expert_routing'] != 'dsr1_profile':
            raise ValueError('published suite requires the declared DSR1 routing profile')
        for c in setting['concurrencies']:
            case_path = DATA_ROOT / f'c{c}'
            if case_path not in inputs:
                case = case_loader(case_path)
                manifest = case.manifest
                if manifest['identity'] != {
                        'source_kind': 'synthetic', 'case_id': f'c{c}',
                        'hardware_id': 'nvidia_h200', 'model_id': 'deepseek-v3'}:
                    raise ValueError('suite case identity mismatch')
                runtime = manifest['runtime']
                expected_runtime = {
                    'tensor_parallel_size': 8, 'pipeline_parallel_size': 1,
                    'expert_parallel_size': 1, 'client_concurrency': c,
                    'requests_per_client': 3, 'warmup_requests_per_client': 1,
                    'arrival_semantics': 'zero_think_time_closed_loop',
                    'prefix_caching': False,
                }
                if runtime != expected_runtime:
                    raise ValueError('case protocol disagrees with the suite')
                if manifest['generation'] != {
                        'algorithm': 'numpy_randomstate_randint_inclusive', 'seed': 0,
                        'input_tokens': [6553, 8192], 'output_tokens': [819, 1024],
                        'cached_hit_provenance': 'assumed_zero_prefix_cache_disabled'}:
                    raise ValueError('case token generation disagrees with 8K/1K recipe')
                for deployment in ('colocated', 'pd_disagg'):
                    policy_path = case_path / manifest['runtime_memory_policies'][deployment]['path']
                    source_policies = load_runtime_memory_policy(policy_path)
                    # The production policy loader validates all fields; bind each
                    # pool to the selected model/hardware and concurrency here too.
                    raw_policy = json.loads(policy_path.read_text(), object_pairs_hook=_unique_object)
                    expected_pools = {'main'} if deployment == 'colocated' else {'prefill-0', 'decode-0'}
                    if {entry['pool_name'] for entry in raw_policy['scheduler_pools']} != expected_pools:
                        raise ValueError('runtime memory pool set disagrees with deployment')
                    for entry in raw_policy['scheduler_pools']:
                        binding = entry['binding']
                        expected_binding = {
                            'deployment': deployment,
                            'expert_parallel_size': 1,
                            # Capture provenance is pinned by the dataset manifest.
                            # Production capacity validates live HWSpec fields, not
                            # the bytes of an older hardware YAML snapshot.
                            'hardware_config_sha256': source_policies.policy_for_pool(
                                entry['pool_name']).binding.hardware_config_sha256,
                            'hardware_id': 'nvidia_h200', 'kv_block_size_tokens': 16,
                            'max_batched_tokens': 32768, 'max_model_len': 9216, 'max_requests': c,
                            'model_config_sha256': sha256(MODEL), 'pipeline_parallel_size': 1,
                            'tensor_parallel_size': 8,
                            'worker_role': 'colocated' if deployment == 'colocated' else {'prefill-0': 'prefill', 'decode-0': 'decode'}[entry['pool_name']],
                        }
                        if binding != expected_binding:
                            raise ValueError('runtime memory binding disagrees with suite')
                inputs[case_path] = case
            jobs.append((setting, c, inputs[case_path]))
    expected_ids = {f'vllm_{group}' for group in GROUPS} | {
        'sglang_colocated_mtp_off', 'sglang_colocated_mtp_on'}
    if seen != expected_ids:
        raise ValueError('suite must contain the six published settings')
    return suite, tuple(jobs)


def run_suite_point(setting, concurrency, case, output_root):
    """Run warmup and measured requests separately; inputs remain immutable."""

    result_dir = output_root / setting['id']
    result_dir.mkdir(parents=True, exist_ok=True)
    deployment = 'pd_disagg' if setting['group'].startswith('pd_') else 'colocated'
    policy = case.case_dir / case.manifest['runtime_memory_policies'][deployment]['path']
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    result = None
    for field, suffix in (('warmup_requests', '_warmup'), ('requests', '')):
        output = result_dir / f'c{concurrency}{suffix}.json'
        source = case.case_dir / case.manifest[field]['path']
        command = build_suite_command(
            setting['group'], source, policy, concurrency, output,
            setting['prediction_method'], setting['graph_config'],
            setting['mtp_profile'], setting['expert_routing'], setting['stack'])
        (result_dir / f'c{concurrency}{suffix}.command.json').write_text(
            json.dumps(command, indent=2) + '\n')
        with (result_dir / f'c{concurrency}{suffix}.stderr').open('w') as stderr:
            subprocess.run(command, cwd=ROOT, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=stderr)
        if field == 'requests':
            result = json.loads(output.read_text())
    quantiles = result['metrics']['latency_distributions']['tpot_s']['quantiles']
    tpot = quantiles['values_s'][quantiles['percentiles'].index(90)]
    return {
        'setting': setting['id'], 'concurrency': concurrency,
        'interactivity_p90_tok_s_user': 1.0 / tpot,
        'total_token_throughput_per_decode_gpu_tok_s': result['metrics']['token_throughput_per_s'] / 8,
        'output_token_throughput_per_decode_gpu_tok_s': result['metrics']['output_token_throughput_per_s'] / 8,
        'gpu_count': 16 if setting['group'].startswith('pd_') else 8,
        'normalization_gpu_count': 8,
    }


def main(*, case_loader):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, default=ROOT / 'configs/evaluation/experiments/deepseek_v3_h200_closedloop.json')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--settings', nargs='+')
    parser.add_argument('--concurrencies', type=int, nargs='+')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-workers', type=int, default=8)
    args = parser.parse_args()
    suite, jobs = load_suite(args.suite, case_loader=case_loader)
    if args.max_workers < 1:
        parser.error('--max-workers must be positive')
    if args.settings is not None:
        if not set(args.settings) <= {s['id'] for s in suite['settings']}:
            parser.error('unknown --settings')
        jobs = tuple(job for job in jobs if job[0]['id'] in args.settings)
    if args.concurrencies is not None:
        if not set(args.concurrencies) <= {4, 8, 16, 32, 64}:
            parser.error('unsupported --concurrencies')
        jobs = tuple(job for job in jobs if job[1] in args.concurrencies)
    print(f"Validated {len(suite['settings'])} settings, 5 shared inputs, {len(jobs)} selected runs")
    if args.validate_only:
        return
    if args.output_dir is None:
        parser.error('--output-dir is required to execute')
    # Preflight: reject selected unsupported SGLang points before launching
    # any job.  The 30-job plan and input binding checks above are
    # unaffected; explicit vLLM selections still run.
    for setting, concurrency, _case in jobs:
        if setting['stack'] == 'sglang':
            parser.error(
                f"setting {setting['id']!r} (concurrency {concurrency}) "
                "requires the deleted sglang_scheduler_mirror backend; "
                "use scripts/evaluation/serving/agentx_session_simulate.py "
                "--backend sglang for the real-SGLang scheduler replay")
    root = args.output_dir.resolve()
    if not root.is_relative_to(ROOT / 'experiment_logs'):
        parser.error('--output-dir must be under experiment_logs')
    if root.exists() and any(root.iterdir()):
        parser.error('--output-dir must be new or empty; existing results are not reused')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n')
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_point, *job, root) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"Completed {row['setting']} C={row['concurrency']}", flush=True)
    rows.sort(key=lambda row: (row['setting'], row['concurrency']))
    (root / 'summary.json').write_text(json.dumps({
        'schema_version': 1, 'suite_id': suite['suite_id'],
        'performance_timing_status': 'analytical_projection_not_measured',
        'metric_note': 'Both total (input+output) and output-only throughput are reported. PD divides by 8 decode GPUs out of 16; colocated divides by 8 total GPUs.',
        'rows': rows,
    }, indent=2) + '\n')
    write_plots(root, rows)


def write_plots(root, rows):
    """Label both throughput definitions and the asymmetric PD denominator."""

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fields = (
        ('output_token_throughput_per_decode_gpu_tok_s', 'output_tgs', 'Output tokens/s/GPU'),
        ('total_token_throughput_per_decode_gpu_tok_s', 'total_tgs', 'Input + output tokens/s/GPU'),
    )
    for field, suffix, ylabel in fields:
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        for setting in sorted({row['setting'] for row in rows}):
            points = sorted((row for row in rows if row['setting'] == setting),
                            key=lambda row: row['concurrency'])
            ax.plot([row['interactivity_p90_tok_s_user'] for row in points],
                    [row[field] for row in points], '-o', label=setting)
            for row in points:
                ax.annotate(f"C={row['concurrency']}",
                            (row['interactivity_p90_tok_s_user'], row[field]),
                            fontsize=7)
        ax.set_xlabel('Interactivity = 1 / P90 TPOT (tokens/s/user)')
        ax.set_ylabel(ylabel)
        ax.set_title('DSV3 FP8 / H200 8K/1K — analytical projection\n'
                     'vLLM: eager; SGLang: decode graph\n'
                     'PD: per 8 decode GPUs (16 total); colocated: per 8 total GPUs')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        for ext in ('png', 'pdf'):
            fig.savefig(root / f'interactivity_{suffix}.{ext}', dpi=180)
        plt.close(fig)


command_for = build_suite_command
run_point = run_suite_point
