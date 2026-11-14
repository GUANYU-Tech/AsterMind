"""Run a physical network experiment and export its diagnostic replay."""

import argparse
import json
import os
import platform
from pathlib import Path

from runtime_environment import activate
activate()
from astermind.builder.physical.network import run_network_experiment


def main() -> None:
    if (
        platform.system() != "Linux"
        or "microsoft" not in platform.release().lower()
        or os.environ.get("CONDA_DEFAULT_ENV") != "OS"
    ):
        raise SystemExit("Network experiments require WSL and conda activate OS")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from paths import owned
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
