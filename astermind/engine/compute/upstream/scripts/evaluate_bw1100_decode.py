#!/usr/bin/env python3
"""CPU-only Flash D-only analytical scenario; no hardware execution."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.serving.metrics.flash_gpu_decode import evaluate_case, write_checkpoint_report, inspect_unpriced_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", action="store_true", help="Regenerate the four-row checkpoint from saved actual windows")
    parser.add_argument("--inspect-unpriced", action="store_true", help="Execute full EP source, retain all unpriced experts and null complete timing")
    parser.add_argument("--scheme", choices=("tp8", "dp8_ep8"), default="tp8")
    parser.add_argument("--concurrency", type=int, choices=(1, 128, 192), default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        result = write_checkpoint_report(out)
        print(json.dumps({"status": result["status"], "report": str(out / "serving_report.md")}))
        return
    if args.inspect_unpriced:
        if args.scheme != "dp8_ep8":
            parser.error("--inspect-unpriced requires --scheme dp8_ep8")
        result = inspect_unpriced_case(concurrency=args.concurrency, steps=args.steps,
            seed=args.seed, output_dir=out)
        print(json.dumps({k: result[k] for k in ("status", "concurrency", "actual_source_steps", "actual_fully_priced_steps", "cpu_wall_s")}))
        return
    result = evaluate_case(scheme=args.scheme, concurrency=args.concurrency, steps=args.steps,
        seed=args.seed, output_dir=out)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "scheme", "concurrency", "actual_steps",
        "output_tokens_per_s", "subsequent_mean_tpot_s", "cpu_wall_s")}), flush=True)


if __name__ == "__main__":
    main()
