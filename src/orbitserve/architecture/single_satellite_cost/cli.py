"""CLI for resolving one canonical HardwareSpec JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..hardware import HardwareSpec, resolve_architecture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve OrbitServe single-satellite architecture.")
    parser.add_argument("case", type=Path, help="Path to a HardwareSpec JSON file.")
    parser.add_argument("--out", type=Path, help="Optional output JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation.")
    args = parser.parse_args(argv)

    spec = HardwareSpec.from_dict(json.loads(args.case.read_text(encoding="utf-8")))
    result = resolve_architecture(spec)
    encoded = json.dumps(result.to_dict(), ensure_ascii=False, indent=args.indent)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0 if result.cost_profile.feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
