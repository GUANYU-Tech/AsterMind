"""Audit causal execution records. Integration consistency, not accuracy fit."""

import argparse
import json
from decimal import ROUND_CEILING, Decimal
from pathlib import Path


def audit(result):
    transfers = {event["id"]: event for event in result["transfers"]}
    if len(transfers) != len(result["transfers"]):
        raise AssertionError("duplicate transfer terminal event")
    counts = {
        "requests": len(result["requests"]),
        "kernels": 0,
        "communications": 0,
        "dependencies": 0,
        "transfers": len(transfers),
    }
    for request in result["requests"].values():
        assert request["status"] == "completed", request
        assert request["completed_ns"] >= request["compute_complete_ns"]
        assert request["cell_ready_ns"] >= request["arrival_ns"]
        if request["first_token_received_ns"] is not None:
            assert request["first_token_received_ns"] >= request["cell_ready_ns"]
    for batch in result["batches"]:
        assert batch["native_communication_duration_charged_s"] == 0
        calls = {(call["call_id"], call["occurrence"]): call for call in batch["calls"]}
        assert batch["end_time_ns"] == max(
            call["end_time_ns"] for call in calls.values()
        )
        for call in calls.values():
            start, end = call["start_time_ns"], call["end_time_ns"]
            assert (
                start >= batch["start_time_ns"] and start >= call["host_ready_time_ns"]
            )
            for dependency in call["dependencies"]:
                assert (
                    start
                    >= calls[tuple(dependency)]["end_time_ns"] + call["launch_gap_ns"]
                )
                counts["dependencies"] += 1
            if not call["communication"]:
                expected = int(
                    (
                        Decimal(str(call["native_prediction_s"])) * 1_000_000_000
                    ).to_integral_value(rounding=ROUND_CEILING)
                )
                if result.get('configuration',{}).get('timeline_mode') == 'simgrid':
                    from astermind.engine.clock_audit import computation_duration_valid
                    assert computation_duration_valid(call,result['network']['clock']), call
                else:
                    assert end - start == expected, call
                counts["kernels"] += 1
                continue
            if call.get("communication_backend") == "protobridge":
                row = call["surrogate"]
                assert not call["phases"], (
                    "surrogate must not also submit collective flows"
                )
                assert row["end_s"] >= row["start_s"] >= row["ready_s"]
                assert end == int(
                    (Decimal(str(row["end_s"])) * 1_000_000_000).to_integral_value(
                        rounding=ROUND_CEILING
                    )
                )
                assert row["attempts"] and row["source_sha256"]
                last = row["attempts"][-1]
                assert abs(row["end_s"] - last["start_s"] - last["duration_s"]) < 1e-12
                counts["communications"] += 1
                continue
            phase_ready = start
            for phase in call["phases"]:
                for flow in phase:
                    assert flow["ready_time_ns"] == phase_ready
                    event = transfers[flow["id"]]
                    assert event["kind"] == "completed"
                    assert event["completion_time_ns"] >= phase_ready
                phase_ready = max(
                    transfers[f["id"]]["completion_time_ns"] for f in phase
                )
            assert phase_ready == end
            counts["communications"] += 1
    for migration in result.get("migrations", {}).values():
        assert migration["state"] == "completed", migration
        partitions = migration.get(
            "partitions", [{"transfer_id": f"migration-{migration['id']}"}]
        )
        events = [transfers[p["transfer_id"]] for p in partitions]
        assert all(event["kind"] == "completed" for event in events)
        from astermind.engine.clock_audit import migration_barrier_valid
        assert migration_barrier_valid(migration, events), migration
        if "bytes_per_rank" in migration:
            assert sum(event["size_bytes"] for event in events) == migration["bytes"]
            assert all(
                event["size_bytes"] == migration["bytes_per_rank"] for event in events
            )
    batch_starts = {
        (b["cell_id"], b["step_id"]): b["start_time_ns"] for b in result["batches"]
    }
    for distribution in result.get("input_distributions", []):
        start = batch_starts[distribution["cell_id"], distribution["step_id"]]
        assert distribution["completed_ns"] <= start, (
            f"input distribution {distribution['cell_id']}/{distribution['step_id']} "
            f"completed at {distribution['completed_ns']} ns after batch start {start} ns"
        )
        for phase in distribution["phases"]:
            for flow in phase:
                assert (
                    transfers[flow["id"]]["completion_time_ns"]
                    <= distribution["completed_ns"]
                )
    from astermind.engine.service_precision import bandwidth_check
    services = warnings = 0
    for index, row in enumerate(result.get('network', {}).get('event_log', [])):
        if row['kind'] != 'bandwidth_update':
            continue
        outcome = bandwidth_check(row, native_ns_clock=result.get('configuration', {}).get('backend') == 'flowsim')
        assert outcome['state'] != 'fail', dict(evidence=f'network.event_log[{index}]', **outcome)
        services += 1
        warnings += outcome['state'] == 'precision'
    counts.update(bandwidth_services=services, bandwidth_precision_warnings=warnings)
    from astermind.engine.clock_audit import clock_checks
    exact = clock_checks(result)
    assert all(row['state'] == 'pass' for row in exact), [row for row in exact if row['state'] == 'fail']
    if exact:
        counts['clock_checks'] = {row['id']: row['checked'] for row in exact}
        counts['clock'] = result['network']['clock']
    return dict(passed=True, **counts, measurement=result.get("measurement", {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        path.stem: audit(json.loads(path.read_text(encoding="utf-8")))
        for path in args.reports
    }
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
