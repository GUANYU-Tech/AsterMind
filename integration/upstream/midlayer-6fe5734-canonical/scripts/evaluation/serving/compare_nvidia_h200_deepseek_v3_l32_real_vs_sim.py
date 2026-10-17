#!/usr/bin/env python3
"""Compare real 4x H200 DeepSeek-V3-L32 TP4 closed-loop runs with the simulator.

Real: ``vllm bench serve`` results (vLLM 0.24.0) copied under
``experiment_logs/h200_real/<real tag>``.  Simulated: the same per-request
lengths replayed by ``run_nvidia_h200_deepseek_v3_l32_tp4_closedloop_sim.py`` under
``experiment_logs/h200_dsv3_l32_tp4_closedloop_sim/<sim tag>``.  Axes follow
InferenceX: x = 1 / P90 TPOT, y = total token throughput per GPU (4 GPUs).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[3]
REAL_ROOT = ROOT / 'experiment_logs/h200_real'
SIM_ROOT = ROOT / 'experiment_logs/h200_dsv3_l32_tp4_closedloop_sim'
OUT_ROOT = ROOT / 'experiment_logs/h200_dsv3_l32_alignment'
GPUS = 4
COLORS = ('#eb6834', '#1baf7a', '#eda100', '#e87ba4')


def real_point(path: Path):
    r = json.loads(path.read_text())
    if r['failed'] != 0:
        raise ValueError(f'{path}: {r["failed"]} failed requests')
    return {'x': 1000.0 / r['p90_tpot_ms'],
            'y': r['total_token_throughput'] / GPUS,
            'tpot50_ms': r['median_tpot_ms'],
            'ttft50_ms': r['median_ttft_ms']}


def sim_point(path: Path):
    r = json.loads(path.read_text())
    dist = r['metrics']['latency_distributions']

    def q(metric, pct):
        quantiles = dist[metric]['quantiles']
        return quantiles['values_s'][quantiles['percentiles'].index(pct)]

    return {'x': 1.0 / q('tpot_s', 90),
            'y': r['metrics']['token_throughput_per_s'] / GPUS,
            'tpot50_ms': 1000.0 * q('tpot_s', 50),
            'ttft50_ms': 1000.0 * q('ttft_s', 50)}


def concurrencies(directory: Path):
    return sorted(int(p.stem[1:]) for p in directory.glob('c*.json')
                  if not p.stem.endswith('_warmup'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-tag', required=True)
    parser.add_argument('--sim-tags', nargs='+', required=True)
    args = parser.parse_args()
    real_dir = REAL_ROOT / args.real_tag
    real = {c: real_point(real_dir / f'c{c}.json') for c in concurrencies(real_dir)}
    report = {'real_tag': args.real_tag, 'real': real, 'sim': {}}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot([real[c]['x'] for c in real], [real[c]['y'] for c in real], '-o',
            color='#2a78d6', linewidth=2, markersize=8, markeredgecolor='white',
            label='real vLLM 0.24.0, 4x H200')
    for c in real:
        ax.annotate(f'C{c}', (real[c]['x'], real[c]['y']), xytext=(6, 6),
                    textcoords='offset points', fontsize=8, color='#6b6a63')
    for color, tag in zip(COLORS, args.sim_tags):
        sim_dir = SIM_ROOT / tag
        sim = {c: sim_point(sim_dir / f'c{c}.json')
               for c in concurrencies(sim_dir) if c in real}
        report['sim'][tag] = {
            c: {**point,
                'x_err_pct': 100.0 * (point['x'] / real[c]['x'] - 1.0),
                'y_err_pct': 100.0 * (point['y'] / real[c]['y'] - 1.0),
                'tpot50_err_pct': 100.0 * (point['tpot50_ms'] / real[c]['tpot50_ms'] - 1.0),
                'ttft50_err_pct': 100.0 * (point['ttft50_ms'] / real[c]['ttft50_ms'] - 1.0)}
            for c, point in sim.items()}
        ax.plot([sim[c]['x'] for c in sim], [sim[c]['y'] for c in sim], '--s',
                color=color, linewidth=2, markersize=7, markerfacecolor='white',
                markeredgewidth=1.8, label=f'sim: {tag}')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('Interactivity (tok/s/user, 1 / P90 TPOT)')
    ax.set_ylabel('Total token throughput per GPU (tok/s)')
    ax.set_title('DeepSeek-V3 arch (32 layers, FP8), TP4 on 4x H200, 8K/1K closed-loop')
    ax.grid(True, which='both', color='#e4e3dc', linewidth=0.6)
    ax.set_ylim(bottom=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = OUT_ROOT / f'{args.real_tag}__{"__".join(args.sim_tags)}'
    fig.savefig(stem.with_suffix('.png'), dpi=180)
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    for tag, points in report['sim'].items():
        print(tag)
        for c, p in points.items():
            print(f"  C{c:<3} real tpot50 {real[c]['tpot50_ms']:6.2f} sim {p['tpot50_ms']:6.2f} "
                  f"({p['tpot50_err_pct']:+5.1f}%) | ttft50 real {real[c]['ttft50_ms']:6.0f} "
                  f"sim {p['ttft50_ms']:6.0f} ({p['ttft50_err_pct']:+5.1f}%) | "
                  f"x {p['x_err_pct']:+5.1f}% y {p['y_err_pct']:+5.1f}%")
    print(stem.with_suffix('.png'))


if __name__ == '__main__':
    main()
