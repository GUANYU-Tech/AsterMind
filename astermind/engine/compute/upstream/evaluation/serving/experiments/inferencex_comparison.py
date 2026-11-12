"""Shared input generation and execution for the established InferenceX recipes."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from evaluation.serving.experiments.inferencex_profiles import COMMAND_START

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_requests(path: Path, count: int, input_tokens: int, output_tokens: int) -> None:
    rng = np.random.RandomState(0)
    inputs = rng.randint(int(0.8 * input_tokens), input_tokens + 1, size=count)
    outputs = rng.randint(int(0.8 * output_tokens), output_tokens + 1, size=count)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Query Time Stamp [ms]', 'In Token [tokens]',
                         'Out Token [tokens]', 'Cached hit [tokens]'])
        writer.writerows((0, int(i), int(o), 0) for i, o in zip(inputs, outputs))


def prepare_case(profile, group, concurrency, measured_per_client, warmup_per_client):
    constants = profile['constants']
    deployment = group.split('_mtp_')[0]
    case_dir = constants['DATA_ROOT'] / group / f'c{concurrency}'
    case_dir.mkdir(parents=True, exist_ok=True)
    measured = case_dir / 'measured_requests.csv'
    warmup = case_dir / 'warmup_requests.csv'
    write_requests(measured, concurrency * measured_per_client,
                   profile['input_tokens'], profile['output_tokens'])
    write_requests(warmup, concurrency * warmup_per_client,
                   profile['input_tokens'], profile['output_tokens'])
    policy_source = constants['COLOCATED_POLICY' if deployment == 'colocated' else 'PD_POLICY']
    raw_policy = json.loads(policy_source.read_text())
    binding_values = {
        'hardware_config_sha256': lambda: sha256(constants['HARDWARE']),
        'model_config_sha256': lambda: sha256(constants['MODEL']),
        'max_requests': lambda: concurrency,
        'expert_parallel_size': lambda: 1,
        'max_model_len': lambda: 9216,
        'max_batched_tokens': lambda: 8192,
    }
    scope = profile['binding_scope'][deployment]
    if scope == 'first':
        pools = (raw_policy['scheduler_pools'][0],)
    elif scope == 'all':
        pools = raw_policy['scheduler_pools']
    else:
        raise ValueError(f'unknown runtime policy binding scope: {scope!r}')
    for pool in pools:
        for field in profile['binding_fields'][deployment]:
            pool['binding'][field] = binding_values[field]()
    policy = case_dir / 'runtime_memory_policy.json'
    policy.write_text(json.dumps(raw_policy, indent=2) + '\n')
    variables = {
        'workload_id': profile['workload_template'].format(group=group, concurrency=concurrency),
        'client_concurrency': concurrency,
        'measured_requests': concurrency * measured_per_client,
        'warmup_requests': concurrency * warmup_per_client,
        'measured_requests_sha256': sha256(measured),
        'warmup_requests_sha256': sha256(warmup),
    }
    template = profile['manifest_common'] | profile['manifest_templates'][group]
    manifest = {key: variables[key] if key in variables else template[key]
                for key in profile['manifest_order']}
    (case_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return measured, warmup, policy


def command_for(profile, group, csv_path, policy, concurrency, output,
                prediction_method='roofline'):
    constants = profile['constants']
    fields = {
        'root': str(ROOT), 'csv_path': str(csv_path), 'policy': str(policy),
        'concurrency': str(concurrency), 'output': str(output),
        'prediction_method': prediction_method,
        'requests_sha256': sha256(csv_path),
        'model_sha256': sha256(constants['MODEL']),
        'hardware_sha256': sha256(constants['HARDWARE']),
    }
    return [token.format_map(fields) for token in COMMAND_START + profile['command_prefix'] + profile['commands'][group]]


def run_point(profile, group, concurrency, *point_args):
    settings = {
        'measured_per_client': 10, 'warmup_per_client': 2,
        'run_dir': profile['constants']['OUTDIR'], 'prediction_method': 'roofline',
    }
    settings.update(zip(profile['point_arguments'], point_args, strict=True))
    measured, warmup, policy = prepare_case(
        profile, group, concurrency, settings['measured_per_client'],
        settings['warmup_per_client'])
    result_dir = settings['run_dir'] / group
    log_dir = result_dir / 'logs'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    warmup_output = result_dir / f'c{concurrency}_warmup.json'
    output = result_dir / f'c{concurrency}.json'
    if output.exists() and warmup_output.exists():
        return group, concurrency, output, 0.0, 'reused'
    env = dict(os.environ, PYTHONHASHSEED='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if profile['pythonpath']:
        env['PYTHONPATH'] = str(ROOT)
    start = time.monotonic()
    for label, source, destination in (
            ('warmup', warmup, warmup_output), ('measured', measured, output)):
        with (log_dir / f'c{concurrency}_{label}.stderr').open('w') as stderr:
            subprocess.run(
                command_for(profile, group, source, policy, concurrency, destination,
                            settings['prediction_method']),
                cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=stderr)
    return group, concurrency, output, time.monotonic() - start, 'completed'


def qvalue(result: dict, metric: str, percentile: int) -> float:
    quantiles = result['metrics']['latency_distributions'][metric]['quantiles']
    return quantiles['values_s'][quantiles['percentiles'].index(percentile)]


def group_rows(profile, run_dir, group, *row_args):
    settings = {
        'concurrencies': profile['row_default_concurrencies'],
        'measured_per_client': 10, 'warmup_per_client': 2,
    }
    settings.update(zip(profile['row_arguments'], row_args, strict=True))
    if settings['concurrencies'] is None:
        raise TypeError('this recipe requires explicit concurrencies')
    rows = []
    for concurrency in settings['concurrencies']:
        path = run_dir / group / f'c{concurrency}.json'
        if profile['row_skip_missing'] and not path.exists():
            print(f'warning: {group} C={concurrency} missing, skipped')
            continue
        result = json.loads(path.read_text())
        total_throughput = result['metrics']['token_throughput_per_s']
        values = {
            'C': concurrency,
            'measured_requests': settings['measured_per_client'] * concurrency,
            'warmup_requests': settings['warmup_per_client'] * concurrency,
            'interactivity_p90_tok_s_user': 1.0 / qvalue(result, 'tpot_s', 90),
            'total_token_throughput_tok_s': total_throughput,
            'total_token_throughput_per_gpu_tok_s': total_throughput / 8,
            'throughput_normalization': settings['normalization'],
            'simulated_device_makespan_s': result['schedule_end_time_s'],
            'simulator_wall_time_s': result['simulation_wall_time_s'],
        }
        latency_fields = {
            'ttft_p50_s': ('ttft_s', 50, 1.0),
            'ttft_p90_s': ('ttft_s', 90, 1.0),
            'tpot_p50_ms': ('tpot_s', 50, 1000.0),
            'tpot_p90_ms': ('tpot_s', 90, 1000.0),
        }
        for field in profile['row_fields']:
            if field in latency_fields:
                metric, percentile, scale = latency_fields[field]
                values[field] = (qvalue(result, metric, percentile) if scale == 1.0
                                 else scale * qvalue(result, metric, percentile))
        rows.append({field: values[field] for field in profile['row_fields']})
    return rows
