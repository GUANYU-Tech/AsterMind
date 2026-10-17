import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import os
from pathlib import Path

from .config import SimulationConfig


def json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description="OrbitServe / Cell / M-L / physical communication")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--midlayer-root", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    error = None
    config = SimulationConfig.load(args.scenario)
    if args.validate:
        topology = config.topology()
        result = dict(valid=True, cells=[c.id for c in config.cells],
                      devices=list(topology.devices), links=len(topology.links),
                      execution_checked=False)
    else:
        from .engine import Simulation
        try:
            simulation = Simulation(config, midlayer_root=args.midlayer_root)
        except Exception as caught:
            error = caught
            result = dict(schema_version=1, status="failed", phase="initialization",
                error_type=type(caught).__name__, error=str(caught), configuration=config.model_dump())
        else:
            with simulation:
                try:
                    result = simulation.run()
                except Exception as caught:
                    error = caught
                    result = dict(schema_version=1, status="failed", error_type=type(caught).__name__,
                        error=str(caught), configuration=config.model_dump(), requests=simulation.state,
                        transfers=simulation.executor.transfer_events, batches=simulation.executor.records,
                        migrations=simulation.migrations.records, network=simulation.network.snapshot())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output}")
    if error is not None:
        parser.exit(1, f"Simulation failed: {type(error).__name__}: {error}\n")
    if result.get("status") == "completed_with_failures":
        parser.exit(1, "Simulation completed with failed requests; inspect output.\n")


if __name__ == "__main__":
    main()
