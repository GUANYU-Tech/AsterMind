#!/usr/bin/env python3
"""Simulate the real 4x H200 DeepSeek-V3-L32 TP4 closed-loop runs.

The real side is vLLM 0.24.0 serving the DeepSeek-V3 architecture with
num_hidden_layers reduced to 32 (dummy FP8 block weights, TP4/EP1,
--max-num-batched-tokens 32768, prefix caching off) on four H200s, driven by
``vllm bench serve`` with the InferenceX closed-loop protocol.  This script
replays exactly the per-request input/output lengths the real benchmark
recorded (``--save-detailed``) through the simulator, colocated TP4/EP1 on
``nvidia_h200``, and writes one result per concurrency.

Usage:
    python scripts/evaluation/serving/run_nvidia_h200_deepseek_v3_l32_tp4_closedloop_sim.py \
        --real-dir <closedloop/TAG> --tag <sim tag> --concurrencies 1 2 4 ...
"""
# Retention note (C4 scripts audit): this is a one-off DSV3-L32 TP4 alignment
# driver with no tests/docs references.  It is retained because the maintained
# owner ``evaluation/serving/experiments/inferencex_deepseek_v3.py`` cites its
# graph-capture values in the GRAPH_CONFIGS provenance comment, and that file
# is outside the C4 edit scope.
import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from evaluation.json_shape import file_sha256 as sha256
PYTHON = Path(sys.executable)
MODEL = ROOT / 'configs/model/deepseek-v3-l32.json'
HARDWARE = ROOT / 'configs/hardware/nvidia_h200.yaml'
POLICY = ROOT / ('configs/runtime_memory_policy/inferencex/'
                 'deepseek_v3_l32_nvidia_h200_aggregate_tp4_ep1_runtime_memory.json')
OUTDIR = ROOT / 'experiment_logs/h200_dsv3_l32_tp4_closedloop_sim'
DATA_ROOT = ROOT / 'serving_dataset/synthetic/h200_dsv3_l32_tp4_closedloop'


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



def write_request_pairs(path: Path, pairs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Query Time Stamp [ms]', 'In Token [tokens]',
                         'Out Token [tokens]', 'Cached hit [tokens]'])
        writer.writerows((0, int(i), int(o), 0) for i, o in pairs)


def prepare(real_dir: Path, concurrency: int):
    real = json.loads((real_dir / f'c{concurrency}.json').read_text())
    pairs = list(zip(real['input_lens'], real['output_lens']))
    if len(pairs) != 10 * concurrency:
        raise ValueError(
            f'C={concurrency}: real run has {len(pairs)} requests, expected '
            f'{10 * concurrency}')
    case_dir = DATA_ROOT / f'c{concurrency}'
    measured = case_dir / 'measured_requests.csv'
    warmup = case_dir / 'warmup_requests.csv'
    write_requests(measured, pairs)
    # The real benchmark warms the server with 2C requests before timing;
    # the simulator's separate warmup run is discarded, so any 2C lengths do.
    write_requests(warmup, pairs[:2 * concurrency])
    raw_policy = json.loads(POLICY.read_text())
    binding = raw_policy['scheduler_pools'][0]['binding']
    binding['hardware_config_sha256'] = sha256(HARDWARE)
    binding['model_config_sha256'] = sha256(MODEL)
    binding['max_requests'] = concurrency
    policy = case_dir / 'runtime_memory_policy.json'
    policy.write_text(json.dumps(raw_policy, indent=2) + '\n')
    return measured, warmup, policy


def build_l32_command(csv_path: Path, policy: Path, concurrency: int, output: Path,
                prediction_method: str) -> list[str]:
    return [
        str(PYTHON), str(ROOT / 'scripts/evaluation/serving/simulation_runner_entrypoint.py'),
        '--canonical-requests', str(csv_path), '--canonical-requests-sha256', sha256(csv_path),
        '--model', str(MODEL), '--model-config-sha256', sha256(MODEL),
        '--hardware-config', str(HARDWARE), '--hardware-config-sha256', sha256(HARDWARE),
        '--backend', 'vllm', '--prediction-method', prediction_method,
        '--collective-backend', 'analytical_roofline_v1', '--launch-policy', 'analytical_launch',
        '--attention-kv-layout', 'mla', '--attention-kv-distribution', 'uniform',
        '--prefill-tp-size', '4', '--prefill-pp-size', '1', '--max-batched-tokens', '32768',
        '--scheduler-pipeline-depth', '1',
        '--runtime-memory-policy', str(policy), '--kv-block-size-tokens', '16',
        '--max-model-len', '9216', '--expert-routing-policy', 'seeded_assumption',
        '--expert-routing-source-id', 'h200-dsv3-l32-tp4-routing-v1',
        '--expert-routing-seed', '0', '--max-requests', str(concurrency),
        '--no-enable-prefix-caching', '--enable-chunked-prefill', '--output', str(output),
        '--closed-loop-client-count', str(concurrency),
        '--deployment', 'colocated', '--pipeline-ranks-per-node', '8',
        *GRAPH_ARGS,
    ]


def run_l32_point(real_dir: Path, concurrency: int, run_dir: Path,
              prediction_method: str):
    measured, warmup, policy = prepare(real_dir, concurrency)
    log_dir = run_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / f'c{concurrency}.json'
    warmup_output = run_dir / f'c{concurrency}_warmup.json'
    if output.exists() and warmup_output.exists():
        return concurrency, 0.0, 'reused'
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    for label, source, destination in (
            ('warmup', warmup, warmup_output), ('measured', measured, output)):
        with (log_dir / f'c{concurrency}_{label}.stderr').open('w') as stderr:
            subprocess.run(command_for(source, policy, concurrency, destination,
                                       prediction_method),
                           cwd=ROOT, env=env, check=True,
                           stdout=subprocess.DEVNULL, stderr=stderr)
    return concurrency, time.monotonic() - start, 'completed'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-dir', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--concurrencies', type=int, nargs='+', required=True)
    parser.add_argument('--prediction-method', default='roofline_span_floor',
                        choices=('roofline', 'roofline_span_floor', 'midlayer'))
    parser.add_argument('--max-workers', type=int, default=8)
    args = parser.parse_args()
    run_dir = OUTDIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [pool.submit(run_point, args.real_dir, c, run_dir,
                               args.prediction_method)
                   for c in sorted(set(args.concurrencies))]
        for future in concurrent.futures.as_completed(futures):
            concurrency, elapsed, status = future.result()
            print(f'C={concurrency} {status} elapsed={elapsed:.1f}s', flush=True)


command_for = build_l32_command
run_point = run_l32_point
write_requests = write_request_pairs

if __name__ == '__main__':
    main()
