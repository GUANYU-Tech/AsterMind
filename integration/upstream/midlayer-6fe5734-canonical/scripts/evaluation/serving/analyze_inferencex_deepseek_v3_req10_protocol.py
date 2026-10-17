"""Old (3 req/client) vs InferenceX-protocol (10 req/client) DSV3 8K/1K errors.

Compares the colocated TP8/EP1 closed-loop sweeps against the same-config
InferenceX SGLang H200 raw points (see plot_inferencex_deepseek_v3_raw_alignment.py
for the row filter).  Per C and per MTP setting it reports the signed error
of

  x90 = 1 / P90 TPOT        (InferenceX "p90_intvty", the plotted x axis)
  x50 = 1 / median TPOT     (step time without tail effects)
  y   = total token throughput per GPU (InferenceX "tput_per_gpu")

plus the P90/median TPOT ratio of sim and InferenceX.  Every sweep tag must
contain every InferenceX concurrency; a missing point is an error.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'experiment_logs/inferencex_alignment/raw/dsr1_all_20260924.json'
SIM_ROOT = ROOT / 'experiment_logs/inferencex_style_dsv3_8k1024_closedloop_tp8_ep1'
GROUPS = {'colocated_mtp_off': 'none', 'colocated_mtp_on': 'mtp'}


def inferencex_rows(spec_method):
    rows = [r for r in json.loads(RAW.read_text())
            if r['hardware'] == 'h200' and r['framework'] == 'sglang'
            and r['precision'] == 'fp8' and not r['disagg']
            and r['isl'] == 8192 and r['osl'] == 1024
            and r['spec_method'] == spec_method
            and r['decode_tp'] == 8 and r['decode_ep'] == 1
            and not r['decode_dp_attention']]
    concs = [r['conc'] for r in rows]
    if len(set(concs)) != len(concs):
        raise ValueError(f'duplicate InferenceX concurrency rows: {concs}')
    return {r['conc']: {
        'tpot_p90_ms': 1000 * r['metrics']['p90_tpot'],
        'tpot_p50_ms': 1000 * r['metrics']['median_tpot'],
        'x90': 1 / r['metrics']['p90_tpot'],
        'x50': 1 / r['metrics']['median_tpot'],
        'p90_intvty_reported': r['metrics']['p90_intvty'],
        'y': r['metrics']['tput_per_gpu'],
        'image': r['image'], 'date': r['date']} for r in rows}


def quantile(result, metric, percentile):
    q = result['metrics']['latency_distributions'][metric]['quantiles']
    return q['values_s'][q['percentiles'].index(percentile)]


def sim_point(tag, group, c):
    result = json.loads((SIM_ROOT / tag / group / f'c{c}.json').read_text())
    p90 = quantile(result, 'tpot_s', 90)
    p50 = quantile(result, 'tpot_s', 50)
    return {'requests_per_client': result['arrival_process']['requests_per_client'],
            'measured_requests': result['request_source']['row_count'],
            'tpot_p90_ms': 1000 * p90, 'tpot_p50_ms': 1000 * p50,
            'x90': 1 / p90, 'x50': 1 / p50,
            'y': result['metrics']['token_throughput_per_s'] / 8,
            'git_commit': result['execution_provenance']['repository']['git_commit']}


def err(sim, real):
    return 100 * (sim / real - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tags', nargs='+', required=True,
                        help='label=sweep_tag pairs, e.g. old=paper_tradeoff_midlayer_r1routing')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    tags = dict(item.split('=', 1) for item in args.tags)
    report = {'inferencex_source': str(RAW.relative_to(ROOT)),
              'definitions': {
                  'x90': '1 / P90 TPOT (tok/s/user)',
                  'x50': '1 / median TPOT (tok/s/user)',
                  'y': 'total (input+output) token throughput per GPU, 8 GPUs',
                  'err_pct': '100 * (sim / InferenceX - 1)'},
              'tags': tags, 'groups': {}}
    for group, spec in GROUPS.items():
        infx = inferencex_rows(spec)
        g = {'inferencex': {str(c): v for c, v in sorted(infx.items())}}
        for label, tag in tags.items():
            rows = []
            for c in sorted(infx):
                s = sim_point(tag, group, c)
                r = infx[c]
                rows.append({
                    'C': c, **{f'sim_{k}': v for k, v in s.items()},
                    'x90_err_pct': err(s['x90'], r['x90']),
                    'x50_err_pct': err(s['x50'], r['x50']),
                    'y_err_pct': err(s['y'], r['y']),
                    'sim_p90_over_p50': s['tpot_p90_ms'] / s['tpot_p50_ms'],
                    'infx_p90_over_p50': r['tpot_p90_ms'] / r['tpot_p50_ms']})
            mean_abs = {k: sum(abs(row[k]) for row in rows) / len(rows)
                        for k in ('x90_err_pct', 'x50_err_pct', 'y_err_pct')}
            mean_signed = {k: sum(row[k] for row in rows) / len(rows)
                           for k in ('x90_err_pct', 'x50_err_pct', 'y_err_pct')}
            g[label] = {'tag': tag, 'rows': rows,
                        'mean_abs_err_pct': mean_abs,
                        'mean_signed_err_pct': mean_signed}
        report['groups'][group] = g
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for group, g in report['groups'].items():
        print(group)
        for label in tags:
            print(f'  [{label}] mean|err| ' + ' '.join(
                f'{k}={v:.2f}' for k, v in g[label]['mean_abs_err_pct'].items()))
            for row in g[label]['rows']:
                print(f"    C{row['C']:<3} rpc={row['sim_requests_per_client']:<2} "
                      f"x90 {row['x90_err_pct']:+6.2f}% x50 {row['x50_err_pct']:+6.2f}% "
                      f"y {row['y_err_pct']:+6.2f}%  p90/p50 sim {row['sim_p90_over_p50']:.3f} "
                      f"infx {row['infx_p90_over_p50']:.3f}")
    print(args.output)


if __name__ == '__main__':
    main()
