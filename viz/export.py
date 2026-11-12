"""Export inspectable replay data from actual native Cell reports, without rerunning a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

SCHEMA = "orbitserve.native_replay.v1"
WEB = Path(__file__).parent


def native_ns(seconds):
    from astermind.engine.clock import Time
    from astermind.engine.timeline import nanoseconds
    if isinstance(seconds, Time):
        return nanoseconds(seconds)
    return int(
        (Decimal(str(seconds)) * 1_000_000_000).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


def physical_ns(seconds):
    value = float(seconds) * 1e9
    if not math.isfinite(value) or abs(value) > 2**53 - 1:
        raise ValueError("replay time must fit exact browser nanosecond integers")
    return value


def audit_checks(result):
    """Expose individual invariants and concrete failures, including faulted runs.

    These are consistency checks over recorded evidence, not hardware validation.
    Never turn a missing trace into a successful check.
    """
    groups = {}

    def check(group, condition, evidence, detail):
        row = groups.setdefault(group, {"id": group, "checked": 0, "violations": []})
        row["checked"] += 1
        if not condition:
            row["violations"].append({"evidence": evidence, "detail": detail})

    transfers = {e["id"]: e for e in result["transfers"]}
    physical_terminals = {e["id"]: e for e in result["network"].get("flows", [])}
    check(
        "transfer_identity",
        len(transfers) == len(result["transfers"]),
        "transfers",
        "传输终态 ID 必须唯一",
    )
    for b, batch in enumerate(result["batches"]):
        prefix = f"batches[{b}]"
        calls = {(c["call_id"], c["occurrence"]): c for c in batch["calls"]}
        check(
            "dependency_order",
            len(calls) == len(batch["calls"]),
            prefix,
            "算子 occurrence 必须唯一",
        )
        check(
            "communication_accounting",
            batch["native_communication_duration_charged_s"] == 0,
            prefix,
            "原生 COMM 参考值不能再次计入实际通信耗时",
        )
        for i, call in enumerate(batch["calls"]):
            at = f"{prefix}.calls[{i}]"
            start, end = call["start_time_ns"], call["end_time_ns"]
            check(
                "dependency_order",
                end >= start >= max(batch["start_time_ns"], call["host_ready_time_ns"]),
                at,
                "算子不能早于批次或主机就绪时间执行",
            )
            for identity in call["dependencies"]:
                dependency = calls.get(tuple(identity))
                check(
                    "dependency_order",
                    dependency is not None
                    and start >= dependency["end_time_ns"] + call["launch_gap_ns"],
                    at,
                    f"依赖尚未完成：{identity}",
                )
            if not call["communication"]:
                if result.get('configuration',{}).get('timeline_mode') == 'simgrid':
                    from astermind.engine.clock_audit import computation_duration_valid
                    duration_valid=computation_duration_valid(call,result['network']['clock'])
                else:
                    duration_valid=end-start==native_ns(call['native_prediction_s'])
                check(
                    "kernel_duration",
                    duration_valid,
                    at,
                    "算子耗时必须符合原生预测及所选引擎的时钟契约",
                )
                continue
            if call.get("communication_backend") == "protobridge":
                row = call.get("surrogate", {})
                attempts = row.get("attempts", [])
                check(
                    "communication_barrier",
                    bool(attempts)
                    and not call["phases"]
                    and row.get("end_s") is not None
                    and native_ns(row["end_s"]) == end
                    and row["end_s"] >= row["start_s"] >= row["ready_s"],
                    at,
                    "ProtoBridge 完成事件必须满足通信屏障，不能同时重复提交逐流通信",
                )
                continue
            ready = start
            for phase in call["phases"]:
                delivered = []
                for flow in phase:
                    event = transfers.get(flow["id"])
                    valid = (
                        event is not None
                        and event["kind"] == "completed"
                        and event["completion_time_ns"] >= ready
                        and flow["ready_time_ns"] == ready
                        and event["size_bytes"] == flow["size_bytes"]
                    )
                    check(
                        "communication_barrier",
                        valid,
                        at,
                        f"通信阶段未按就绪、字节量或完成屏障执行：{flow['id']}",
                    )
                    if event is not None and event["kind"] == "completed":
                        delivered.append(event["completion_time_ns"])
                if delivered:
                    ready = max(delivered)
            check(
                "communication_barrier",
                ready == end,
                at,
                "COMM 结束时刻必须等于末阶段的最后一次实际交付",
            )

    batches = {(b["cell_id"], b["step_id"]): b["start_time_ns"] for b in result["batches"]}
    for i, distribution in enumerate(result.get("input_distributions", [])):
        batch = batches.get((distribution["cell_id"], distribution["step_id"]))
        valid = (
            batch is not None and distribution["completed_ns"] <= batch
        )
        for phase in distribution["phases"]:
            for flow in phase:
                event = transfers.get(flow["id"])
                valid = (
                    valid
                    and event is not None
                    and event["kind"] == "completed"
                    and event["completion_time_ns"] <= distribution["completed_ns"]
                )
        check(
            "input_distribution",
            valid,
            f"input_distributions[{i}]",
            "输入分发必须先于批次启动完成",
        )

    for identity, request in result["requests"].items():
        check(
            "request_delivery",
            request["status"] in {"completed", "failed"},
            f"requests.{identity}",
            "请求应有明确终态",
        )
        if request["status"] == "completed":
            check(
                "request_delivery",
                request["completed_ns"] >= request["compute_complete_ns"]
                and request["cell_ready_ns"] >= request["arrival_ns"],
                f"requests.{identity}",
                "请求交付与计算的因果次序",
            )
            produced = sum(
                o["token_count"]
                for o in result["produced_outputs"]
                if o["request_id"] == identity
            )
            check(
                "request_delivery",
                produced == request["received_tokens"],
                f"requests.{identity}",
                "产出与用户接收 token 数量相同",
            )

    for index, row in enumerate(result["network"]["event_log"]):
        if row["kind"] == "bandwidth_update":
            from astermind.engine.service_precision import bandwidth_check
            outcome=bandwidth_check(row,native_ns_clock=result.get('configuration',{}).get('backend')=='flowsim')
            check(
                "service_capacity",
                outcome['state']!='fail',
                f"network.event_log[{index}]",
                outcome.get('detail',''),
            )
            if outcome['state']=='precision':
                groups['service_capacity'].setdefault('warnings',[]).append(dict(evidence=f'network.event_log[{index}]',**outcome))

    for event in transfers.values():
        physical = physical_terminals.get(event["id"])
        if physical is not None and event["kind"] == "completed":
            delta = event["completion_time_ns"] - physical["completion_time_ns"]
            check(
                "delivery_clock",
                -1e-3 <= delta < 1.001,
                f"transfers.{event['id']}",
                "算子交付通知只能晚于物理完成 0–1 ns",
            )
        if event["source"].startswith("gpu") and event["target"].startswith("gpu"):
            devices = {
                d["id"]: d for d in result["network"]["device_topology"]["devices"]
            }
            ground = set(result["configuration"].get("ground_owners", []))
            check(
                "ground_forwarding",
                all(
                    devices[n]["satellite"] not in ground
                    for n in event["path"]
                    if n in devices
                ),
                f"transfers.{event['id']}",
                "GPU 间通信不能绕经地面用户中继",
            )

    for identity, migration in result.get("migrations", {}).items():
        at = f"migrations.{identity}"
        partitions = migration.get("partitions", [])
        terminals = [transfers.get(p["transfer_id"]) for p in partitions]
        target = result["cells"][migration["target"]]["final_load"]["storage"]
        check(
            "migration_transaction",
            target["migration_reserved_blocks"] == 0,
            at,
            "终态不能遗留目标迁移保留块",
        )
        if migration["state"] == "completed":
            check(
                "migration_transaction",
                bool(terminals)
                and all(e is not None and e["kind"] == "completed" for e in terminals),
                at,
                "全部 rank 分片完成后才可提交",
            )
            check(
                "migration_transaction",
                sum(e["size_bytes"] for e in terminals if e) == migration["bytes"],
                at,
                "分片总字节量守恒",
            )
            check(
                "migration_transaction",
                migration["completed_time_s"] >= migration["network_complete_time_s"],
                at,
                "目标写入完成不得早于网络屏障",
            )
            from astermind.engine.clock_audit import migration_barrier_valid
            check(
                "migration_transaction",
                migration_barrier_valid(migration, terminals),
                at,
                "网络屏障等待最慢分片",
            )
        elif migration["state"] == "failed":
            check(
                "migration_transaction",
                not migration.get("source_release", {}).get("removed_blocks", 0),
                at,
                "失败不能删除源缓存",
            )
            check(
                "migration_transaction",
                bool(migration.get("failures")),
                at,
                "回滚必须保留实际故障证据",
            )
        else:
            check(
                "migration_transaction",
                migration["state"] in {"capacity_blocked", "cancelled"},
                at,
                "迁移应处于明确终态",
            )

    for row in groups.values():
        row["state"] = "fail" if row['violations'] else 'warn' if row.get('warnings') else 'pass'
    from astermind.engine.clock_audit import clock_checks
    for row in clock_checks(result):
        if row['id'] in groups:
            existing = groups[row['id']]
            row['checked'] = max(row['checked'], existing['checked'])
            row['violations'].extend(existing['violations'])
            row['state'] = 'fail' if row['violations'] else 'pass'
        groups[row['id']] = row
    return list(groups.values())


def replay_data(result, *, source_name="report.json", source_sha256=None):
    from astermind.builder.config import SimulationConfig
    from astermind.engine.record_store import CompressedRecords

    config = SimulationConfig.model_validate(result["configuration"])
    graph = config.topology()
    nodes = graph.node_device
    placements = {(c.id, p.rank): p.device for c in config.cells for p in c.placements}
    owners = {d.id: d.satellite for d in graph.devices.values()}
    calls, steps, ready_times = CompressedRecords(), [], {}
    ready_ticks = {}
    for cell_id, cell in result["cells"].items():
        for index, row in enumerate(cell["steps"]):
            schedule = row["step"]["schedule"]
            steps.append(
                {
                    "cell": cell_id,
                    "id": schedule["step_id"],
                    "start": physical_ns(row["start_time_s"]),
                    "end": physical_ns(row["end_time_s"]),
                    "requests": schedule["requests"],
                    "kv": row["step"]["kv_cache_events"],
                    "generated": row["step"]["generated_tokens"],
                    "evidence": f"cells.{cell_id}.steps[{index}]",
                }
            )
    for batch_index, batch in enumerate(result["batches"]):
        offset = len(calls)
        lookup = {
            (c["call_id"], c["occurrence"]): offset + i
            for i, c in enumerate(batch["calls"])
        }
        for i, call in enumerate(batch["calls"]):
            phases = [[flow["id"] for flow in phase] for phase in call["phases"]]
            for phase in call["phases"]:
                for flow in phase:
                    ready_times[flow["id"]] = flow["ready_time_ns"]
                    if flow.get('ready_time_ticks') is not None:
                        ready_ticks[flow['id']] = flow['ready_time_ticks']
            calls.append(
                {
                    "id": call["call_id"],
                    "occurrence": call["occurrence"],
                    "cell": batch["cell_id"],
                    "batch": batch["step_id"],
                    "rank": call["rank"],
                    "device": placements.get((batch["cell_id"], call["rank"])),
                    "owner": owners.get(placements.get((batch["cell_id"], call["rank"]))),
                    "kind": call["kind"],
                    "comm": call["communication"],
                    "start": call["start_time_ns"],
                    "end": call["end_time_ns"],
                    "start_ticks": call.get('start_time_ticks'),
                    "end_ticks": call.get('end_time_ticks'),
                    "deps": [lookup.get(tuple(d), -1) for d in call["dependencies"]],
                    "gap": call["launch_gap_ns"],
                    "host_ready": call["host_ready_time_ns"],
                    "native_ns": native_ns(call["native_prediction_s"]),
                    "phases": phases,
                    "communication_backend": call.get(
                        "communication_backend", config.backend
                    )
                    if call["communication"]
                    else None,
                    "surrogate": call.get("surrogate"),
                    "evidence": f"batches[{batch_index}].calls[{i}]",
                }
            )
    for distribution in result.get("input_distributions", []):
        for phase in distribution["phases"]:
            for flow in phase:
                ready_times[flow["id"]] = flow["ready_time_ns"]
    resource_map, hops, events, terminal_clock = {}, CompressedRecords(), [], {}
    observed_capacity = {}
    physical_events = result["network"].get("flows", [])
    for event in physical_events:
        terminal_clock[event["id"]] = event.get(
            "completion_time_ns", event.get("failure_time_ns")
        )
    meaningful = {
        "link_state",
        "fault_injected",
        "fault_restored",
        "route_wait",
        "route_recovery",
        "transfer_cancelled",
    }
    for index, row in enumerate(result["network"]["event_log"]):
        if (
            row.get("source_node_id") in nodes
            and row.get("target_node_id") in nodes
            and row.get("link_id")
        ):
            resource_map[row["link_id"]] = {
                "id": row["link_id"],
                "source": nodes[row["source_node_id"]],
                "target": nodes[row["target_node_id"]],
            }
        flow_id = row.get("cell_transfer_id")
        if flow_id and row["kind"] == "transfer_queued":
            ready_times[flow_id] = min(
                ready_times.get(flow_id, math.inf), physical_ns(row["ready_s"])
            )
        if row["kind"] == "bandwidth_update":
            key = row["link_id"], row["source_node_id"], row["target_node_id"]
            previous = observed_capacity.get(key)
            if previous is not None and not math.isclose(
                previous, row["link_capacity_bps"], rel_tol=1e-9
            ):
                events.append(
                    {
                        "kind": "capacity_observed",
                        "time": physical_ns(row["interval_start_s"]),
                        "link_id": row["link_id"],
                        "capacity": row["link_capacity_bps"],
                        "previous_capacity": previous,
                        "evidence": f"network.event_log[{index}]",
                    }
                )
            observed_capacity[key] = row["link_capacity_bps"]
            hops.append(
                {
                    "flow": flow_id,
                    "resource": row["link_id"],
                    "source": nodes.get(row["source_node_id"], row["source_node_id"]),
                    "target": nodes.get(row["target_node_id"], row["target_node_id"]),
                    "start": physical_ns(row["interval_start_s"]),
                    "end": physical_ns(row["interval_end_s"]),
                    "start_ticks": row.get('interval_start_ticks'),
                    "end_ticks": row.get('interval_end_ticks'),
                    "rate": row["allocated_bandwidth_bps"],
                    "capacity": row["link_capacity_bps"],
                    "bytes": row["bytes_sent"],
                    "evidence": f"network.event_log[{index}]",
                }
            )
        if row["kind"] in meaningful:
            events.append(
                {
                    **row,
                    "time": physical_ns(row["time_s"]),
                    "evidence": f"network.event_log[{index}]",
                }
            )
    for output in result["produced_outputs"]:
        events.append(
            {
                "kind": "token_produced",
                "time": physical_ns(output["produced_time_s"]),
                **output,
            }
        )
    transfers = []
    for index, event in enumerate(result["transfers"]):
        end = event.get("completion_time_ns", event.get("failure_time_ns"))
        transfers.append(
            {
                **event,
                "start": ready_times.get(event["id"]),
                "end": end,
                "start_ticks": ready_ticks.get(event['id']),
                "end_ticks": event.get('time_ticks'),
                "physical_end": terminal_clock.get(event["id"]),
                "evidence": f"transfers[{index}]",
            }
        )
    migrations = {}
    for identity, migration in result.get("migrations", {}).items():
        ready = [
            ready_times[p["transfer_id"]]
            for p in migration.get("partitions", [])
            if p["transfer_id"] in ready_times
        ]
        definition = next((m for m in config.migrations if m.id == identity), None)
        migrations[identity] = {
            **migration,
            "source_read_complete_ns": min(ready) if ready else None,
            "mode": definition.mode if definition is not None else None,
        }
    end = max(
        max((t["end"] for t in transfers), default=1),
        max((c["end"] for c in calls), default=1),
        max((
            physical_ns(m["completed_time_s"])
            for m in result.get("migrations", {}).values()
            if "completed_time_s" in m
        ), default=1),
        1,
    )
    cells = {
        c.id: {
            "config": c.model_dump(mode="json"),
            "scheduler": result["cells"][c.id]["scheduler_runtime"],
            "storage": result["cells"][c.id]["final_load"]["storage"],
        }
        for c in config.cells
    }
    title = (
        "跨星 TP2 × PP2"
        if any(c.parallel.pp > 1 for c in config.cells)
        else (
            "KV 失败回滚"
            if any(
                m["state"] == "failed" for m in result.get("migrations", {}).values()
            )
            else f"KV {config.migrations[0].mode.title() if config.migrations else '迁移'} · {config.backend}"
            if result.get("migrations")
            else "MoE TP2 / EP2"
            if any(c.native_expert_parallel for c in config.cells)
            else "异构 GPU / 显存账目"
            if len({c.hardware_id for c in config.cells}) > 1
            or any(c.memory_policy_file for c in config.cells)
            else "原生 Cell 执行"
        )
    )
    return {
        "schema": SCHEMA,
        "title": title,
        "source": {"name": source_name, "sha256": source_sha256},
        "status": result["status"],
        "backend": config.backend,
        "horizon": config.horizon_ns,
        "end": end,
        "devices": [d.model_dump(mode="json") for d in graph.devices.values()],
        "links": [link.model_dump(mode="json") for link in graph.links],
        "resources": list(resource_map.values()),
        "ground": list(config.ground_owners),
        "cells": cells,
        "requests": result["requests"],
        "request_inputs": [r.model_dump(mode="json") for r in config.requests],
        "outputs": result["produced_outputs"],
        "steps": sorted(steps, key=lambda s: s["start"]),
        "calls": calls,
        "transfers": transfers,
        "hops": hops,
        "events": sorted(events, key=lambda e: e["time"]),
        "migrations": migrations,
        "audit": audit_checks(result),
        "original_audit": result.get("audit"),
        "provenance": result["provenance"],
        "physical_identity": result["network"].get("physical_dataset_identity"),
        "transport": result["network"].get("transport"),
        "clock": result['network'].get('clock'),
        "clock_note": "SimGrid 全局事件时钟" if config.timeline_mode=='simgrid' else "统一时钟 · 1 as" if result['network'].get('clock') else "旧时钟记录 · 重新运行后使用统一时钟",
    }


def export_reports(reports, destination, *, loaded_results=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    catalog = []
    used = set()
    for report in reports:
        report = Path(report)
        digest = hashlib.sha256()
        with report.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        result = (loaded_results or {}).get(str(report))
        if result is None:
            with report.open(encoding='utf-8') as stream:
                result = json.load(stream)
        data = replay_data(
            result,
            source_name=report.name,
            source_sha256=digest.hexdigest(),
        )
        identity = re.sub(r"[^a-zA-Z0-9_-]", "-", report.stem)
        if identity in used:
            raise ValueError(f"duplicate replay filename: {identity}")
        used.add(identity)
        name = identity + ".json"
        path = destination / name
        temporary = path.with_suffix(".json.tmp")
        with temporary.open('w',encoding='utf8',buffering=1024*1024) as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(path)
        catalog.append(
            {
                "id": identity,
                "title": data["title"],
                "backend": data["backend"],
                "status": data["status"],
                "file": name,
                "calls": len(data["calls"]),
                "transfers": len(data["transfers"]),
            }
        )
    for name in ("index.html", "replay.js", "replay.css"):
        shutil.copyfile(WEB / name, destination / name)
    temporary = destination / "catalog.json.tmp"
    temporary.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf8"
    )
    temporary.replace(destination / "catalog.json")
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    catalog = export_reports(args.reports, args.out)
    print(f"Exported {len(catalog)} recorded runs to {args.out}")


if __name__ == "__main__":
    main()
