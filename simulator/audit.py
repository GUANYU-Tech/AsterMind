"""Audit causal execution records. Integration consistency, not accuracy fit."""
from decimal import Decimal, ROUND_CEILING
import argparse
import json
from pathlib import Path


def audit_estimate(estimate):
    assert all(type(value) is int and value >= 0 for value in estimate.values())
    assert estimate["cross_satellite_flows"] <= estimate["flow_count"]
    assert estimate["rounds"] <= estimate["flow_count"] or estimate["duration_ns"] == 0


def audit(result):
    transfers = {event["id"]: event for event in result["transfers"]}
    if len(transfers) != len(result["transfers"]):
        raise AssertionError("duplicate transfer terminal event")
    counts = dict(requests=len(result["requests"]), kernels=0, communications=0,
                  dependencies=0, transfers=len(transfers))
    for request in result["requests"].values():
        assert request["status"] == "completed", request
        assert request["completed_ns"] >= request["compute_complete_ns"]
        assert request["cell_ready_ns"] >= request["arrival_ns"]
        if request["first_token_received_ns"] is not None:
            assert request["first_token_received_ns"] >= request["cell_ready_ns"]
    if "request_state_events" in result:
        events = {item["sequence"]: item for item in result["event_log"]}
        expected_tokens = {item["id"]: item["output_tokens"]
                           for item in result["configuration"]["requests"]}
        status = {request_id: "pending" for request_id in result["requests"]}
        for change in result["request_state_events"]:
            request_id = change["request_id"]
            event = events[change["event_sequence"]]
            assert change["time_ns"] == round(event["time_s"] * 1e9)
            assert change["previous"] == status[request_id]
            status[request_id] = change["status"]
        for request_id, state in result["requests"].items():
            assert status[request_id] == state["status"] == "completed"
            assert state["received_tokens"] == expected_tokens[request_id]
    for batch in result["batches"]:
        assert batch["native_communication_duration_charged_s"] == 0
        calls = {(call["call_id"], call["occurrence"]): call for call in batch["calls"]}
        assert batch["end_time_ns"] == max(call["end_time_ns"] for call in calls.values())
        for call in calls.values():
            start, end = call["start_time_ns"], call["end_time_ns"]
            assert start >= batch["start_time_ns"] and start >= call["host_ready_time_ns"]
            for dependency in call["dependencies"]:
                assert start >= calls[tuple(dependency)]["end_time_ns"] + call["launch_gap_ns"]
                counts["dependencies"] += 1
            if not call["communication"]:
                expected = int((Decimal(str(call["native_prediction_s"])) * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
                assert end - start == expected, call
                counts["kernels"] += 1
                continue
            if "communication_estimate" in call:
                estimate = call["communication_estimate"]
                audit_estimate(estimate)
                assert end - start == estimate["duration_ns"]
                counts["communications"] += 1
                continue
            # Original event-driven traces remain auditable.
            phase_ready = start
            for phase in call["phases"]:
                for flow in phase:
                    assert flow["ready_time_ns"] == phase_ready
                    event = transfers[flow["id"]]
                    assert event["kind"] == "completed"
                    assert event["completion_time_ns"] >= phase_ready
                phase_ready = max(transfers[f["id"]]["completion_time_ns"] for f in phase)
            assert phase_ready == end
            counts["communications"] += 1
    for migration in result.get("migrations", {}).values():
        assert migration["state"] == "completed", migration
        partitions = migration.get("partitions", [{"transfer_id": f"migration-{migration['id']}"}])
        events = [transfers[p["transfer_id"]] for p in partitions]
        assert all(event["kind"] == "completed" for event in events)
        assert max(event["completion_time_ns"] for event in events) <= round(migration["completed_time_s"] * 1e9)
        if "bytes_per_rank" in migration:
            assert sum(event["size_bytes"] for event in events) == migration["bytes"]
            assert all(event["size_bytes"] == migration["bytes_per_rank"] for event in events)
    for cell in result.get("cells", {}).values():
        for request_id, handoff in cell.get("pd_handoffs", {}).items():
            assert handoff["partitions"] and not handoff["failures"]
            terminal = [transfers[part["id"]] for part in handoff["partitions"]]
            assert all(event["kind"] == "completed" for event in terminal)
            assert max(event["completion_time_ns"] for event in terminal) == handoff["end_time_ns"]
            assert all(event["size_bytes"] == part["bytes"]
                       for event, part in zip(terminal, handoff["partitions"], strict=True))
            decode = [step for step in cell["steps"] if step["step"]["schedule"]["pool_name"] == "decode"
                      and any(row["request_id"] == request_id for row in step["step"]["schedule"]["requests"])]
            assert decode and round(decode[0]["start_time_s"] * 1e9) >= handoff["end_time_ns"]
    batches = {(b["cell_id"], b["step_id"]): b for b in result["batches"]}
    for distribution in result.get("input_distributions", []):
        batch = batches[distribution["cell_id"], distribution["step_id"]]
        assert distribution["completed_ns"] <= batch["start_time_ns"]
        if "communication_estimate" in distribution:
            estimate = distribution["communication_estimate"]
            audit_estimate(estimate)
            assert distribution["completed_ns"] - distribution["ready_ns"] == estimate["duration_ns"]
            continue
        for phase in distribution["phases"]:
            for flow in phase:
                assert transfers[flow["id"]]["completion_time_ns"] <= distribution["completed_ns"]
    return dict(passed=True, **counts, measurement=result.get("measurement", {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {path.stem: audit(json.loads(path.read_text(encoding="utf-8"))) for path in args.reports}
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
