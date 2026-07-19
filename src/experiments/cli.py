"""Command-line entrypoint for OrbitServe experiment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import load_experiment_spec
from .output import default_artifact_dir_for_output, experiment_output_payload
from .runner import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an OrbitServe experiment manifest.")
    parser.add_argument("manifest", type=Path, help="Path to an experiment JSON manifest.")
    parser.add_argument("--out", type=Path, help="Optional output JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    parser.add_argument(
        "--output-mode",
        choices=("summary", "full"),
        default="summary",
        help="Write lightweight summary JSON by default, or the historical monolithic full JSON.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory for bulky audit artifacts in summary mode. Defaults next to --out.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Shortcut for --output-mode full.",
    )
    args = parser.parse_args(argv)

    spec = load_experiment_spec(args.manifest)
    result = run_experiment(spec)
    output_mode = "full" if args.full else args.output_mode
    artifact_dir = (
        None
        if output_mode == "full"
        else args.artifact_dir
        if args.artifact_dir is not None
        else default_artifact_dir_for_output(args.out)
    )
    payload = experiment_output_payload(
        result,
        output_mode=output_mode,
        artifact_dir=artifact_dir,
        output_path=args.out,
        indent=args.indent,
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=args.indent)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
