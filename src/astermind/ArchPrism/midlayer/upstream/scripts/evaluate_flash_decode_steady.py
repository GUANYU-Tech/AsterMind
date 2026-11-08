#!/usr/bin/env python3
"""CPU-only declared FP4 decode steady-state experiment."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.serving.metrics.flash_decode_steady import evaluate_case, evaluate_matrix, HARDWARE, SCHEMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hardware', choices=HARDWARE)
    parser.add_argument('--scheme', choices=SCHEMES)
    parser.add_argument('--microbatches', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--model', choices=('flash', 'pro'), default='flash')
    parser.add_argument('--bw216', action='store_true')
    parser.add_argument('--fpga-cards', type=int, choices=(16, 32), default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-html', action='store_true')
    args = parser.parse_args()
    if (args.hardware is None) != (args.scheme is None):
        parser.error('--hardware and --scheme must be provided together')
    if args.hardware is None:
        if args.model != 'flash' or args.bw216 or args.fpga_cards != 16:
            parser.error('matrix mode supports only the original Flash hardware configurations; use --hardware and --scheme for Pro or BW216')
        result = evaluate_matrix(output_dir=args.output_dir, workers=args.workers, steps=args.steps, html=not args.no_html)
        print(json.dumps({'status': result['status'], 'cases': len(result['cases'])}))
    else:
        result = evaluate_case(hardware_id=args.hardware, scheme=args.scheme, microbatches=args.microbatches,
            steps=args.steps, layer_count=args.layers, output_dir=args.output_dir, html=not args.no_html, model_name=args.model, bw216=args.bw216, fpga_cards=args.fpga_cards)
        print(json.dumps({k: result[k] for k in ('status', 'cpu_wall_s')}))


if __name__ == '__main__':
    main()
