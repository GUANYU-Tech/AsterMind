"""Command-line interface for the single-satellite cost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calculator import estimate_single_satellite_cost
from .models import load_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate OrbitServe single-satellite cost.")
    parser.add_argument("case", type=Path, help="Path to a JSON case file.")
    parser.add_argument("--out", type=Path, help="Optional output JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    args = parser.parse_args(argv)

    data = json.loads(args.case.read_text(encoding="utf-8"))
    design_limits, config = load_case(data)
    result = estimate_single_satellite_cost(design_limits, config)
    payload = result.to_dict()

    encoded = json.dumps(payload, ensure_ascii=False, indent=args.indent)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if result.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())

