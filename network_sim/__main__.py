"""Run synthetic transfers independently of all compute-side code."""

import argparse
import json
from pathlib import Path

from .model import Scenario, TransferRequest
from .runtime import CommunicationRuntime
from .topology import PhysicalTopology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--transfers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--until-ns", type=int, help="finite simulation horizon; required for dynamic policy")
    args = parser.parse_args()
    topology = PhysicalTopology(Scenario.load(args.scenario))
    requests = [TransferRequest.model_validate(r) for r in
                json.loads(args.transfers.read_text(encoding="utf-8"))]
    with CommunicationRuntime(topology) as runtime:
        isolated = {}
        for r in requests:
            try:
                isolated[r.id] = runtime.route(r.source, r.target, r.size_bytes)
            except ValueError:
                if runtime.controller is None:
                    raise
                isolated[r.id] = dict(candidate_path=runtime.candidate_path(r.source, r.target),
                                      active_route=None)
        for request in requests:
            runtime.submit(request)
        events = runtime.run(args.until_ns)
        report = runtime.snapshot()
        report["isolated_estimates"] = isolated
        report["completion_events"] = [event.model_dump() for event in events if event.kind == "completed"]
        report["failure_events"] = [event.model_dump() for event in events if event.kind == "failed"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in {
        "report.json": report,
        "canonical_topology.json": topology.canonical.model_dump(mode="json"),
        "view_cells.json": topology.view("cells"),
        "view_cell0.json": topology.view("satellites", topology.scenario.cells[0].id),
        "view_devices.json": topology.view("devices"),
    }.items():
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"completed": len(report["completion_events"]),
                      "failed": len(report["failure_events"]), "end_time_ns": report["end_time_ns"],
                      "report": str(args.output_dir / "report.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
