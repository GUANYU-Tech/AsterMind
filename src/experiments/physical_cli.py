"""Compute reusable physical cores or export optional views without Runtime."""

import argparse
import json
from pathlib import Path

from .network import run_network_experiment
from .network_exports import export_network_artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compute = commands.add_parser(
        "compute", help="Compute a reusable physical dataset."
    )
    compute.add_argument("manifest", type=Path)
    compute.add_argument("--out", type=Path, required=True)
    compute.add_argument("--exports", choices=("core", "full"), default="core")
    export = commands.add_parser("export", help="Export views from a completed core.")
    export.add_argument("dataset", type=Path)
    for command in (compute, export):
        command.add_argument("--workers", type=int)
        command.add_argument("--compression-level", type=int, default=1)
        command.add_argument("--chunk-samples", type=int, default=32)
    args = parser.parse_args(argv)
    if args.command == "compute":
        value = json.loads(args.manifest.read_text(encoding="utf-8"))
        physical = value.get("physical_network", value)
        base = args.manifest.parent
        if isinstance(physical, str):
            path = base / physical
            physical = json.loads(path.read_text(encoding="utf-8"))
            base = path.parent
        result = run_network_experiment(
            physical,
            args.out,
            base_dir=base,
            export_mode=args.exports,
            _workers=args.workers,
            compression_level=args.compression_level,
            chunk_samples=args.chunk_samples,
        )
        summary = {
            k: result[k]
            for k in ("trace_id", "checks", "timing_s", "physical_counts", "artifacts")
        }
    else:
        summary = export_network_artifacts(
            args.dataset,
            workers=args.workers if args.workers is not None else 8,
            compression_level=args.compression_level,
            chunk_samples=args.chunk_samples,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
