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
    parser.add_argument("--gc-gen2-threshold", type=int,
                        help="set the full GC threshold for this CLI run")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.gc_gen2_threshold is None:
        return _run(args, parser)
    if args.gc_gen2_threshold < 1:
        parser.error("--gc-gen2-threshold must be positive")
    import gc

    thresholds = gc.get_threshold()
    gc.set_threshold(*thresholds[:2], args.gc_gen2_threshold)
    try:
        return _run(args, parser)
    finally:
        gc.set_threshold(*thresholds)


def _run(args, parser):
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
            network = None
            try:
                with simulation:
                    try:
                        result = simulation.run()
                    except Exception as caught:
                        try:
                            network = simulation.network.snapshot()
                        except Exception as snapshot_error:
                            caught.add_note(f"Network snapshot failed: {snapshot_error}")
                        raise
            except Exception as caught:
                error = caught
                result = dict(schema_version=1, status="failed", error_type=type(caught).__name__,
                    error=str(caught), configuration=config.model_dump(), requests=simulation.state,
                    transfers=simulation.transport.transfer_events, batches=simulation.batches,
                    migrations=simulation.migrations.records, network=network, event_log=simulation.loop.events,
                    request_state_events=simulation.request_state_events)
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
