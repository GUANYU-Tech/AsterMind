"""Command-line entrypoint for Native SimulationConfig scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from astermind.experiments.native_cells import run_native_cell_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a Native SimulationConfig scenario."
    )
    parser.add_argument(
        "manifest", type=Path, help="Path to a SimulationConfig JSON scenario."
    )
    parser.add_argument("--out", type=Path, help="Optional output JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    parser.add_argument('--midlayer-root', type=Path, help='Explicit external M-L root for native Cell scenarios.')
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

    return run_native_cell_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
