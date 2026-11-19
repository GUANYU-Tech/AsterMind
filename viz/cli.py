"""Run a physical network experiment and export its diagnostic replay."""

import argparse
import json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from runtime_environment import activate
    from paths import owned

    activate()
    from astermind.builder.physical.network import run_network_experiment

    args.out=owned(args.out)
    result = run_network_experiment(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        args.out,
        base_dir=args.manifest.parent,
        progress=lambda stage: print(stage, flush=True),
    )
    checks = result["checks"]
    print(
        json.dumps(
            {
                key: checks[key]
                for key in (
                    "passed",
                    "snapshot_count",
                    "physical_satellites",
                    "ground_nodes",
                    "event_frame_count",
                )
            },
            ensure_ascii=False,
        )
    )
    print(args.out / "network_result.json")


if __name__ == "__main__":
    main()
