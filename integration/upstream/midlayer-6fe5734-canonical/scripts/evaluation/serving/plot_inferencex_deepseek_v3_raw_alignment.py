"""Plot DSV3 H200 8K/1K simulation against InferenceX per-config raw points.

InferenceX source: ``/api/v1/benchmarks?model=DeepSeek-R1-0528`` snapshot
saved under ``experiment_logs/inferencex_alignment/raw``.  Only the
single-node SGLang TP8/EP1 (no DP-attention, no disaggregation) rows are
used, which is the same deployment as our colocated TP8/EP1 sweep.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'experiment_logs/inferencex_alignment/raw/dsr1_all_20260924.json'
SIM_ROOT = ROOT / 'experiment_logs/inferencex_style_dsv3_8k1024_closedloop_tp8_ep1'
OUT_ROOT = ROOT / 'experiment_logs/inferencex_alignment'

MEASURED_COLOR = '#2a78d6'
SIM_COLOR = '#eb6834'
INK = '#1f1f1e'
MUTED = '#6b6a63'
GRID = '#e4e3dc'


def inferencex_rows(spec_method):
    rows = [r for r in json.loads(RAW.read_text())
            if r['hardware'] == 'h200' and r['framework'] == 'sglang'
            and r['precision'] == 'fp8' and not r['disagg']
            and r['isl'] == 8192 and r['osl'] == 1024
            and r['spec_method'] == spec_method
            and r['decode_tp'] == 8 and r['decode_ep'] == 1
            and not r['decode_dp_attention']]
    rows.sort(key=lambda r: r['conc'])
    concs = [r['conc'] for r in rows]
    if len(set(concs)) != len(concs):
        raise ValueError(f'duplicate InferenceX concurrency rows: {concs}')
    return [(r['conc'], r['metrics']['p90_intvty'],
             r['metrics']['tput_per_gpu'], r['image'], r['date']) for r in rows]


def sim_rows(tag, group):
    # Colocated points are read per concurrency so a colocated-only sweep
    # (which writes no four-group summary.json) can still be compared.
    rows = []
    for path in sorted((SIM_ROOT / tag / group).glob('c*.json')):
        if path.stem.endswith('_warmup'):
            continue
        result = json.loads(path.read_text())
        quantiles = result['metrics']['latency_distributions']['tpot_s']['quantiles']
        p90_tpot = quantiles['values_s'][quantiles['percentiles'].index(90)]
        rows.append((int(path.stem[1:]), 1.0 / p90_tpot,
                     result['metrics']['token_throughput_per_s'] / 8))
    rows.sort()
    return rows


def style_axis(ax):
    ax.set_xscale('log')
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.grid(True, which='minor', color=GRID, linewidth=0.4, alpha=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_xlabel('Interactivity (tok/s/user, 1 / P90 TPOT)', color=INK)
    ax.set_xlim(10, 700)
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 2200)


def label_points(ax, pts, dx, dy):
    for c, x, y in pts:
        ax.annotate(f'C{c}', (x, y), xytext=(dx, dy), textcoords='offset points',
                    fontsize=7.5, color=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True,
                        help='sweep results subdirectory of the MTP-off run, '
                             'e.g. full3_fp8fix; also names the output')
    parser.add_argument('--tag-mtp-on', required=True,
                        help='sweep results subdirectory of the MTP-on run')
    parser.add_argument('--sim-label-mtp-off', required=True,
                        help='legend text for the MTP-off simulation')
    parser.add_argument('--sim-label-mtp-on', required=True,
                        help='legend text for the MTP-on simulation')
    parser.add_argument('--mtp-on-title', required=True,
                        help='title of the MTP-on panel (state the MTP settings)')
    args = parser.parse_args()
    sim_labels = {'colocated_mtp_off': args.sim_label_mtp_off,
                  'colocated_mtp_on': args.sim_label_mtp_on}
    out = OUT_ROOT / (f'dsv3_vs_inferencex_raw_sglang_tp8_{args.tag}'
                      f'__mtpon_{args.tag_mtp_on}')
    panels = [('none', 'colocated_mtp_off', 'MTP off', args.tag),
              ('mtp', 'colocated_mtp_on', args.mtp_on_title, args.tag_mtp_on)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    report = {}
    for ax, (spec, group, title, tag) in zip(axes, panels):
        meas = inferencex_rows(spec)
        sim = sim_rows(tag, group)
        ax.plot([m[1] for m in meas], [m[2] for m in meas], '-o', color=MEASURED_COLOR,
                linewidth=2, markersize=8, markeredgecolor='white', markeredgewidth=1.5,
                label=f'InferenceX measured (SGLang TP8/EP1, {meas[0][3].split(":")[-1]})')
        ax.plot([s[1] for s in sim], [s[2] for s in sim], '--s', color=SIM_COLOR,
                linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=1.8,
                label=sim_labels[group])
        label_points(ax, [(m[0], m[1], m[2]) for m in meas], 6, 6)
        label_points(ax, sim, 6, -12)
        style_axis(ax)
        ax.set_title(title, color=INK, fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5, loc='upper right', labelcolor=INK)
        sim_by_c = {s[0]: s for s in sim}
        report[group] = [
            {'C': c, 'infx_x': x, 'infx_y': y,
             'sim_x': sim_by_c[c][1], 'sim_y': sim_by_c[c][2],
             'x_err_pct': 100 * (sim_by_c[c][1] / x - 1),
             'y_err_pct': 100 * (sim_by_c[c][2] / y - 1)}
            for c, x, y, _, _ in meas]
    fig.suptitle('DeepSeek-V3/R1 FP8, 8×H200, ISL/OSL 8K/1K, closed-loop — same-config InferenceX raw points', color=INK, fontsize=11.5)
    axes[0].set_ylabel('Total token throughput per GPU (tok/s)', color=INK)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'), dpi=200)
    fig.savefig(out.with_suffix('.pdf'))
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    for group, rows in report.items():
        print(group)
        for r in rows:
            print(f"  C{r['C']:<3} infx {r['infx_x']:6.1f}/{r['infx_y']:7.1f}  "
                  f"sim {r['sim_x']:6.1f}/{r['sim_y']:7.1f}  "
                  f"err x {r['x_err_pct']:+5.1f}% y {r['y_err_pct']:+5.1f}%")
    print(out.with_suffix('.png'))


if __name__ == '__main__':
    main()
