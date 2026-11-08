"""Bounded H200 proxy concurrency sweeps using the public metric contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from evaluation.serving.metrics.interactivity_tgs import (
    load_interactivity_tgs_contract, sweep_metric_point_from_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
CAPACITIES = (1, 2, 4, 8, 16, 32, 64, 128)
MODELS = {'glm': ('glm-5.2-fp8', 565248), 'kimi': ('kimi-k2.6', 229376)}
FIELDS = ('Query Time Stamp [ms]', 'In Token [tokens]', 'Out Token [tokens]', 'Cached hit [tokens]')
CAPTION = ('H200 A800-proxy projection: 108 scheduling units (public H200 SM count: 132); '
           'A800 launch package; seeded_assumption expert routing, seed=1. '
           'PD TP=EP=8, PP=1, one prefill and one decode instance. '
           'GLM: MTP fixed accepted=1; Kimi: MTP disabled. '
           'Exploratory simulation; not validated H200 hardware performance.')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def prepare(args, model):
    output = args.output.resolve() / model
    if (output / 'requests.csv').exists():
        raise FileExistsError(f'refusing to overwrite prepared trace: {output}')
    output.mkdir(parents=True, exist_ok=True)
    source = args.source_dir / f'{model}_medium_qps4_300s.csv'
    parent_path = source.with_suffix('.provenance.json')
    parent = json.loads(parent_path.read_text())
    if digest(source) != parent['candidate_sha256']:
        raise ValueError('source trace digest does not match provenance')
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.start_request_index:
        if args.window_seconds is not None:
            raise ValueError('start-request-index requires request-count selection')
        rows = rows[args.start_request_index:]
    if args.window_seconds is not None:
        end_ms = float(rows[0][FIELDS[0]]) + args.window_seconds * parent['arrival_rate_scale'] * args.arrival_rate_multiplier * 1000
        rows = [row for row in rows if float(row[FIELDS[0]]) <= end_ms]
    if args.request_count is not None:
        rows = rows[:args.request_count]
    if len(rows) < 2:
        raise ValueError('at least two original requests are needed for the arrival window')
    trace = output / 'requests.csv'
    with trace.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    scale = parent['arrival_rate_scale'] * args.arrival_rate_multiplier
    span = (float(rows[-1][FIELDS[0]]) - float(rows[0][FIELDS[0]])) / 1000 / scale
    provenance = {
        'source_csv': str(source.resolve()), 'source_sha256': digest(source),
        'parent_provenance': str(parent_path.resolve()), 'parent_provenance_sha256': digest(parent_path),
        'candidate_csv': str(trace), 'candidate_sha256': digest(trace),
        'selection': 'contiguous chronological window; original timestamps and token fields preserved',
        'start_request_index': args.start_request_index,
        'request_count': len(rows), 'arrival_rate_scale': scale, 'anchor_qps': parent['anchor_qps'],
        'arrival_rate_multiplier_from_parent': args.arrival_rate_multiplier,
        'arrival_span_seconds': span, 'achieved_requests_per_arrival_span_s': len(rows) / span,
        'token_totals': {field: sum(int(row[field]) for row in rows) for field in FIELDS[1:]},
    }
    write_json(trace.with_suffix('.provenance.json'), provenance)
    return output, trace, provenance


def build_pd_command(args, model, capacity, output, trace, provenance):
    model_name, default_max_len = MODELS[model]
    max_len = getattr(args, f'max_model_len_{model}') or default_max_len
    model_path = ROOT / 'configs/model' / f'{model_name}.json'
    hardware = ROOT / 'configs/hardware/nvidia_h200.yaml'
    template = args.memory_policy_dir / f'{model}_runtime_memory_max_requests_1.json'
    memory = json.loads(template.read_text())
    for pool in memory['scheduler_pools']:
        binding = pool['binding']
        if binding['hardware_id'] != 'nvidia_h200' or binding['deployment'] != 'pd_disagg':
            raise ValueError('memory policy template is outside H200 PD scope')
        binding.update(hardware_config_sha256=digest(hardware), model_config_sha256=digest(model_path),
                       max_requests=capacity, max_model_len=max_len)
    policy = output / f'memory_{capacity}.json'
    write_json(policy, memory)
    stats = output / f'point_{capacity}.time'
    wrapper = (
        'import sys, atexit, resource; from pathlib import Path; '
        f'atexit.register(lambda: Path({str(stats)!r}).write_text(str(max('
        'resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, '
        'resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)))); '
        'from evaluation.serving.simulation.simulation_runner import main; '
        'raise SystemExit(main(sys.argv[1:]))')
    command = [str(PYTHON), '-c', wrapper]
    options = {
        'production-trace': trace, 'trace-source-sha256': digest(trace),
        'trace-format': 'csv', 'trace-time-unit': 'ms',
        'trace-arrival-field': FIELDS[0], 'trace-input-tokens-field': FIELDS[1],
        'trace-output-tokens-field': FIELDS[2], 'trace-prefix-hit-tokens-field': FIELDS[3],
        'trace-success-filter-mode': 'upstream_attested',
        'trace-success-provenance': trace.with_suffix('.provenance.json'),
        'trace-replay-mode': 'arrival_rate_scale', 'trace-arrival-rate-scale': provenance['arrival_rate_scale'],
        'model': model_path, 'model-config-sha256': digest(model_path),
        'hardware-config': hardware, 'hardware-config-sha256': digest(hardware),
        'decode-hardware-config': hardware, 'deployment': 'pd_disagg', 'backend': 'vllm',
        'prediction-method': 'roofline', 'collective-backend': 'analytical_roofline_v1',
        'launch-policy': 'analytical_launch', 'attention-kv-layout': 'mla',
        'attention-kv-distribution': 'uniform', 'prefill-tp-size': 8, 'prefill-pp-size': 1,
        'decode-tp-size': 8, 'num-prefill-instances': 1, 'num-decode-instances': 1,
        'transfer-algorithm': 'send_recv', 'transfer-link': 'nvlink', 'scheduler-pipeline-depth': 1,
        'max-batched-tokens': 32768, 'kv-block-size-tokens': 16, 'step-retention': args.step_retention,
        'runtime-memory-policy': policy, 'max-model-len': max_len,
        'expert-routing-policy': 'seeded_assumption', 'expert-routing-source-id': 'h200-proxy-routing-v1',
        'expert-routing-seed': 1, 'max-requests': capacity, 'output': output / f'point_{capacity}.json',
    }
    if model == 'glm':
        options.update({'mtp-num-speculative-tokens': 1, 'mtp-acceptance-policy': 'fixed_count',
                        'mtp-fixed-accepted-tokens': 1, 'mtp-acceptance-source-id': 'glm5-mtp-fixed-one'})
    for key, value in options.items():
        command.extend((f'--{key}', str(value)))
    command.extend(('--enable-expert-parallel', '--enable-chunked-prefill', '--enable-prefix-caching'))
    return command, template


def point_timeout_seconds(args, capacity):
    if args.unbounded_all_capacities:
        return None
    if capacity <= 2 and args.unbounded_low_capacities:
        return None
    if capacity <= 2 and args.low_capacity_timeout_seconds is not None:
        return args.low_capacity_timeout_seconds
    return args.timeout_seconds


def average_scheduler_concurrency(payload):
    summary = payload['scheduler_progress']['summary']
    step_count = summary['step_count']
    if step_count <= 0:
        raise ValueError('scheduler progress must contain at least one step')
    return summary['scheduled_request_rows'] / step_count


def cluster_observed_concurrency(points):
    """Group only numerically indistinguishable observed-concurrency points."""
    clusters = []
    for point in sorted(points, key=lambda value: value['average_scheduler_concurrency']):
        observed = point['average_scheduler_concurrency']
        if (not clusters
                or not math.isclose(
                    observed,
                    clusters[-1][0]['average_scheduler_concurrency'],
                    rel_tol=1e-3,
                    abs_tol=1e-9)):
            clusters.append([point])
        else:
            clusters[-1].append(point)
    return tuple(tuple(cluster) for cluster in clusters)


def run_pd_point(args, model, capacity, output, trace, provenance):
    destination = output / f'point_{capacity}.json'
    record_path = output / f'point_{capacity}.run.json'
    if destination.exists() or record_path.exists():
        raise FileExistsError(f'refusing to overwrite point: {destination}')
    command, template = command_for(args, model, capacity, output, trace, provenance)
    record = {'model': model, 'max_requests': capacity, 'command': command,
              'timeout_seconds': point_timeout_seconds(args, capacity),
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'dirty_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
              'trace': provenance, 'memory_policy_template': str(template.resolve()),
              'memory_policy_template_sha256': digest(template), 'started_at_unix': time.time()}
    write_json(output / f'point_{capacity}.command.json', record)
    stats = output / f'point_{capacity}.time'
    started = time.monotonic()
    reason = None
    environment = dict(os.environ)
    environment.update(PYTHONHASHSEED='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (output / f'point_{capacity}.stdout').open('w') as stdout, (output / f'point_{capacity}.stderr').open('w') as stderr:
        process = subprocess.Popen(command,
                                   cwd=ROOT, stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
        record['pid'] = process.pid
        write_json(output / f'point_{capacity}.command.json', record)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            memory_current = int(Path('/sys/fs/cgroup/memory.current').read_text())
            if record['timeout_seconds'] is not None and elapsed > record['timeout_seconds']:
                reason = 'wall_timeout'
            elif memory_current > 850 * 1024 ** 3:
                reason = 'shared_cgroup_memory_guard_850_GiB'
            if reason is not None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(1)
    record.update(wall_seconds=time.monotonic() - started, exit_code=process.returncode,
                  stopped_reason=reason, finished_at_unix=time.time(), peak_rss_bytes=None, metric=None)
    if stats.exists():
        record['peak_rss_bytes'] = int(stats.read_text()) * 1024
    else:
        heartbeat_peaks = re.findall(r'peak_rss_bytes=(\d+)', (output / f'point_{capacity}.stderr').read_text())
        if heartbeat_peaks:
            record['peak_rss_bytes'] = max(int(value) for value in heartbeat_peaks)
    if process.returncode == 0:
        payload = json.loads(destination.read_text())
        contract = load_interactivity_tgs_contract(ROOT / 'configs/evaluation/interactivity_tgs_h200.json')
        record['metric'] = asdict(sweep_metric_point_from_manifest(provenance['arrival_rate_scale'], payload['metrics'], contract))
        record['average_scheduler_concurrency'] = average_scheduler_concurrency(payload)
    write_json(record_path, record)
    print(f'{model} capacity={capacity} exit={process.returncode} wall={record["wall_seconds"]:.1f}s peak={record["peak_rss_bytes"]}', flush=True)
    return record


def render(output, records, *, report_title='# Goal 未完成 — checkpoint'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = [report_title, '', CAPTION, '',
             'Metrics use sweep_metric_point_from_manifest and configs/evaluation/interactivity_tgs_h200.json: P99 TPOT (seconds) and output tokens/s. Smaller TPOT means better interactivity. The plots use observed scheduler concurrency; numerically indistinguishable observed values are one physical platform with a vertical metric range. The raw table retains every nominal-cap run.', '',
             '| Model | Nominal cap | Average scheduler concurrency | Requests | Scale | Arrival span s | Actual req/s | Wall s | Peak GiB | P99 TPOT s | Output tokens/s | Exit |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        points = [r for r in records if r['model'] == model and r['metric'] is not None]
        if points:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            clusters = cluster_observed_concurrency(points)
            concurrency = [sum(r['average_scheduler_concurrency'] for r in cluster) / len(cluster)
                           for cluster in clusters]
            tpot = [sum(r['metric']['interactivity_value_s'] for r in cluster) / len(cluster)
                    for cluster in clusters]
            tgs = [sum(r['metric']['output_token_throughput_per_s'] for r in cluster) / len(cluster)
                   for cluster in clusters]
            tpot_range = [[value - min(r['metric']['interactivity_value_s'] for r in cluster)
                           for value, cluster in zip(tpot, clusters)],
                          [max(r['metric']['interactivity_value_s'] for r in cluster) - value
                           for value, cluster in zip(tpot, clusters)]]
            tgs_range = [[value - min(r['metric']['output_token_throughput_per_s'] for r in cluster)
                          for value, cluster in zip(tgs, clusters)],
                         [max(r['metric']['output_token_throughput_per_s'] for r in cluster) - value
                          for value, cluster in zip(tgs, clusters)]]
            labels = [','.join(str(r['max_requests']) for r in cluster) for cluster in clusters]
            axes[0].errorbar(concurrency, tpot, yerr=tpot_range, fmt='o-')
            axes[0].set(xlabel='Average scheduler concurrency', ylabel='P99 TPOT (s)', title=model.upper())
            axes[1].errorbar(concurrency, tgs, yerr=tgs_range, fmt='o-')
            axes[1].set(xlabel='Average scheduler concurrency', ylabel='Output tokens/s', title='Labels: nominal max_requests')
            for axis, values in zip(axes, (tpot, tgs)):
                for label, x, y in zip(labels, concurrency, values):
                    axis.annotate(label, (x, y))
            for axis in axes:
                axis.grid(alpha=.3)
            fig.text(.02, .02, 'H200 A800 proxy: 108 scheduling units / 132 public SMs; A800 launches; seeded routing.\nGLM MTP=1; Kimi MTP off. Exploratory only; unvalidated H200 performance.', fontsize=8)
            fig.tight_layout(rect=(0, .12, 1, 1))
            for extension in ('png', 'pdf'):
                fig.savefig(output / f'{model}_tgs_interactivity.{extension}', dpi=160)
            plt.close(fig)
    for r in records:
        p, m = r['trace'], r['metric']
        rss = None if r['peak_rss_bytes'] is None else r['peak_rss_bytes'] / 1024 ** 3
        values = [r['model'], r['max_requests'],
                  r['average_scheduler_concurrency'] if m is not None else None,
                  p['request_count'], p['arrival_rate_scale'],
                  p['arrival_span_seconds'], p['achieved_requests_per_arrival_span_s'], r['wall_seconds'], rss,
                  None if m is None else m['interactivity_value_s'],
                  None if m is None else m['output_token_throughput_per_s'], r['exit_code']]
        lines.append('| ' + ' | '.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in values) + ' |')
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'results.json', records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--memory-policy-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-count', type=int)
    parser.add_argument('--start-request-index', type=int, default=0)
    parser.add_argument('--window-seconds', type=float)
    parser.add_argument('--arrival-rate-multiplier', type=float, default=1.0)
    parser.add_argument('--max-model-len-glm', type=int)
    parser.add_argument('--max-model-len-kimi', type=int)
    parser.add_argument('--timeout-seconds', type=float, default=590)
    parser.add_argument('--low-capacity-timeout-seconds', type=float,
                        help='optional timeout for max_requests <= 2')
    parser.add_argument('--unbounded-low-capacities', action='store_true',
                        help='do not apply a wall-clock timeout to max_requests <= 2')
    parser.add_argument('--unbounded-all-capacities', action='store_true',
                        help='do not apply a wall-clock timeout to any capacity')
    parser.add_argument('--step-retention', choices=('auto', 'full'), default='full',
                        help='PD transfer artifact serialization currently requires full retention')
    parser.add_argument('--models', nargs='+', choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument('--capacities', nargs='+', type=int, choices=CAPACITIES, default=list(CAPACITIES))
    args = parser.parse_args(argv)
    if args.request_count is None and args.window_seconds is None:
        parser.error('select request-count or window-seconds explicitly')
    if args.request_count is not None and args.request_count < 2:
        parser.error('request-count must be at least two')
    if args.start_request_index < 0:
        parser.error('start-request-index must be non-negative')
    if args.window_seconds is not None and args.window_seconds <= 0:
        parser.error('window-seconds must be positive')
    if args.arrival_rate_multiplier <= 0:
        parser.error('arrival-rate-multiplier must be positive')
    if ((args.max_model_len_glm is not None and args.max_model_len_glm <= 0) or
            (args.max_model_len_kimi is not None and args.max_model_len_kimi <= 0)):
        parser.error('max model lengths must be positive')
    if (args.timeout_seconds <= 0 or
            (args.low_capacity_timeout_seconds is not None and args.low_capacity_timeout_seconds <= 0) or
            len(set(args.models)) != len(args.models)):
        parser.error('timeout must be positive and models must be unique')
    if len(set(args.capacities)) != len(args.capacities):
        parser.error('capacities must be unique')
    records = []
    def run_model(model):
        output, trace, provenance = prepare(args, model)
        return [run_point(args, model, n, output, trace, provenance) for n in args.capacities]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(run_model, args.models):
            records.extend(result)
    render(args.output, records)
    return int(any(r['exit_code'] != 0 for r in records))


command_for = build_pd_command
run_point = run_pd_point
