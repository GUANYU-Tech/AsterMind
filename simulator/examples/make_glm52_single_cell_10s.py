"""Generate 24 evenly spaced requests for the 32-rank GLM-5.2 Cell."""

import argparse
import json
from pathlib import Path


def make_scenario():
    base = Path(__file__).with_name("glm52_single_cell_32ranks.json")
    scenario = json.loads(base.read_text(encoding="utf-8"))
    template = scenario["requests"]
    scenario["requests"] = [
        dict(request, id=f"R{index + 1}", arrival_ns=index * 430_000_000)
        for index, request in enumerate(template * 4)
    ]
    scenario["horizon_ns"] = 12_000_000_000
    return scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(make_scenario(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
