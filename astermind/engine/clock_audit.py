"""Independent invariants under the selected native clock contract."""

import math

from astermind.engine.clock import CLOCK
from astermind.engine.service_precision import bandwidth_check
from astermind.engine.timeline import nanoseconds


def computation_duration_valid(call, clock):
    """Validate model service under the selected native clock contract."""
    if clock.get('authority') != 'simgrid_engine':
        return int(call['end_time_ticks'])-int(call['start_time_ticks']) == nanoseconds(call['native_prediction_s'])*10**9
    start,end = int(call['start_time_ticks']),int(call['end_time_ticks'])
    expected = nanoseconds(call['native_prediction_s'])*10**9
    # A native deadline is rounded upward once to a representable double.
    budget = math.ceil((math.ulp(start/1e18)+math.ulp(end/1e18))*1e18)+2
    return expected <= end-start <= expected+budget


def migration_barrier_valid(migration, terminals):
    """Compare one clock representation; legacy ns displays round upward."""
    try:
        if not terminals or any(event is None or event['kind'] != 'completed' for event in terminals):
            return False
        if 'network_complete_time_ticks' in migration:
            barrier = int(migration['network_complete_time_ticks'])
            return (barrier >= 0
                    and int(migration['completed_time_ticks']) >= barrier
                    and all(0 <= int(event['time_ticks']) <= barrier for event in terminals))
        # Old files have no exact migration stamp. Their integer ns completion
        # fields must be compared to the same ceil-ns projection of the barrier.
        barrier = nanoseconds(migration['network_complete_time_s'])
        return (migration['completed_time_s'] >= migration['network_complete_time_s']
                and all(event['completion_time_ns'] <= barrier for event in terminals))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def clock_checks(result):
    clock = result.get("network", {}).get("clock", {}) or {}
    if clock != CLOCK and clock.get('authority') != 'simgrid_engine':
        return []
    exact = clock == CLOCK
    groups = {}

    def check(group, valid, evidence, detail):
        row = groups.setdefault(group, dict(id=group, checked=0, violations=[]))
        row["checked"] += 1
        if not valid:
            row["violations"].append(dict(evidence=evidence, detail=detail))

    previous = -1
    service = {}
    terminal = {}
    for index, row in enumerate(result["network"]["event_log"]):
        at = f"network.event_log[{index}]"
        tick = int(row.get("time_ticks", "-1"))
        check(
            "unified_clock",
            tick >= previous and tick >= 0,
            at,
                "全局事件时钟必须单调，记录不得缺失时钟",
        )
        previous = tick
        if row["kind"] == "resource_service":
            for resource in row["resources"]:
                rate, capacity = (
                    resource["allocated_bandwidth_bps"],
                    resource["capacity_bps"],
                )
                check(
                    "shared_capacity",
                    math.isfinite(rate)
                    and math.isfinite(capacity)
                    and 0 <= rate <= capacity * (1 + 1e-12)+(resource.get('allocation_error_bps',0) if not exact else 0),
                    at + ":" + resource["resource_id"],
                    "共享链路或容量池的所有并发分配之和不得超过容量",
                )
        elif row["kind"] == "bandwidth_update":
            outcome = bandwidth_check(row, native_ns_clock=exact)
            check(
                "service_capacity",
                outcome["state"] == "pass",
                at,
                outcome.get("detail", ""),
            )
            identity = row["transfer_id"]
            total, correction = service.get(identity, (0.0, 0.0))
            value = row["bytes_sent"] - correction
            updated = total + value
            service[identity] = updated, (updated - total) - value
        elif row["kind"] == "transmit_complete":
            payload = row.get("payload_bytes", 0)
            if payload > 0:
                amount = service.pop(row["transfer_id"], (0.0, 0.0))[0]
                expected = max(1, round(payload * 1e6)) / 1e6
                check(
                    "byte_conservation",
                abs(amount - expected) <= max(1e-12 if exact else 1e-6, math.ulp(expected) * 32),
                    at,
                    "逐跳累计服务量必须等于该跳量化负载，不能提前完成或重复计费",
                )
        elif row["kind"] == "cell_transfer_terminal":
            terminal[row["id"]] = tick
    transfers = {row["id"]: row for row in result.get("transfers", [])}
    for identity, tick in terminal.items():
        event = transfers.get(identity)
        if event is not None:
            check(
                "delivery_clock",
                int(event.get("time_ticks", "-1")) == tick,
                f"transfers.{identity}",
                "物理完成与算子通知必须使用完全相同的整数时刻",
            )
    for b, batch in enumerate(result.get("batches", [])):
        calls = {(c["call_id"], c["occurrence"]): c for c in batch["calls"]}
        for i, call in enumerate(batch["calls"]):
            at = f"batches[{b}].calls[{i}]"
            start, end = int(call["start_time_ticks"]), int(call["end_time_ticks"])
            check(
                "unified_clock",
                end >= start >= int(call["host_ready_time_ticks"]),
                at,
                "计算与主机就绪使用同一整数时钟",
            )
            for dependency in call["dependencies"]:
                check(
                    "dependency_order",
                    start
                    >= int(calls[tuple(dependency)]["end_time_ticks"])
                    + call["launch_gap_ns"] * 10**9,
                    at,
                    "前驱完成之前不能释放资源或启动后继算子",
                )
            if not call["communication"]:
                check(
                    "kernel_duration",
                    computation_duration_valid(call, clock),
                    at,
                    "GPU 模型耗时必须按所选原生时钟的明确数值契约执行",
                )
            elif call.get("communication_backend") == "protobridge":
                check(
                    "communication_barrier",
                    end == int(call["surrogate"]["end_ticks"]),
                    at,
                    "ProtoBridge 完成与依赖释放必须是同一个时刻",
                )
            else:
                ready = start
                for phase in call["phases"]:
                    for flow in phase:
                        event = transfers[flow["id"]]
                        check(
                            "communication_barrier",
                            int(flow["ready_time_ticks"]) == ready
                            and int(event["time_ticks"]) >= ready,
                            at,
                            "逐流阶段使用精确就绪与完成时刻",
                        )
                    ready = max(int(transfers[f["id"]]["time_ticks"]) for f in phase)
                check(
                    "communication_barrier",
                    ready == end,
                    at,
                    "最后一条流完成时释放通信算子",
                )
    for row in groups.values():
        row["state"] = "fail" if row["violations"] else "pass"
    return list(groups.values())
