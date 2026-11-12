"""Private RPC worker owning one real SimGrid engine per Runtime simulation.

Only this worker imports SimGrid. A controller actor waits for native activity
completion or the next external event, so neither engine runs past a causal
boundary. Native link latency is zero: OrbitServe accounts for propagation after
serialization. Capacities are converted from bits/s to SimGrid bytes/s here.
"""

from __future__ import annotations

import json
import math
import sys
from importlib.metadata import version

BYTE_SCALE = 1_000_000


def main():
    import simgrid as sg

    initial = json.loads(sys.stdin.readline())
    engine = sg.Engine(
        [
            "orbitserve-simgrid",
            "--cfg=network/model:CM02",
            "--cfg=network/crosstraffic:0",
            "--cfg=network/weight-S:0",
            "--cfg=precision/timing:1e-15",
            "--cfg=precision/work-amount:1e-9",
            "--log=root.thres:critical",
        ]
    )
    zone = engine.netzone_root
    hosts = {
        name: zone.add_host(f"node-{i}", 1e12).seal()
        for i, name in enumerate(initial["nodes"])
    }
    links = {}
    pools = {}
    for key, rate in initial.get('pools', {}).items():
        pools[key] = zone.add_link(key, rate / 8 * BYTE_SCALE).set_latency(0).seal()
    for row in initial["links"]:
        link = zone.add_link(
            row["key"], max(row["bandwidth_bps"] / 8 * BYTE_SCALE, 1e-30)
        )
        link.set_latency(0).seal()
        links[row["key"]] = link
        zone.add_route(
            hosts[row["source"]], hosts[row["target"]],
            [sg.LinkInRoute(link), *(sg.LinkInRoute(pools[k]) for k in row.get('pools', ()))], False
        )
    zone.seal()
    active = {}
    link_metadata = {row['key']:row for row in initial['links']}
    from astermind.engine.simgrid_events import NativeGlobalEvents
    scheduler = NativeGlobalEvents(sg, next(iter(hosts.values())), active) if initial.get('global_scheduler') else None

    def emit(payload):
        print(
            json.dumps({"time_s": sg.Engine.clock, **payload}, allow_nan=False),
            flush=True,
        )

    def controller():
        if initial.get('start_s', 0) > 0:
            sg.this_actor.sleep_until(initial['start_s'])
        emit({"ready": True, "version": version("simgrid"), "model": "CM02"})
        while line := sys.stdin.readline():
            command = json.loads(line)
            try:
                op = command["op"]
                if op == "close":
                    if scheduler is not None:
                        scheduler.close()
                    for row in active.values():
                        row["comm"].cancel()
                    active.clear()
                    emit({"closed": True})
                    return
                if op == 'remaining':
                    emit({'remaining': {identity: row['comm'].remaining / BYTE_SCALE for identity, row in active.items()}})
                elif op == "start":
                    comm = sg.Comm.sendto_async(
                        hosts[command["source"]],
                        hosts[command["target"]],
                        command["bytes"],
                    )
                    active[command["id"]] = {"comm": comm, "key": command["key"]}
                    emit({"started": command["id"]})
                elif op == 'schedule_event':
                    scheduler.schedule(command['id'], command['at_s'], command['priority'],restore=command.get('restore',False))
                    emit({'scheduled':command['id']})
                elif op == 'scheduler_state':
                    emit({'scheduler':scheduler.report()})
                elif op == 'restore_scheduler':
                    stats=command['stats']
                    if scheduler.stats['dispatched'] or stats['pending']!=len(scheduler.pending):
                        raise ValueError('scheduler restoration must match all pending deadlines')
                    scheduler.stats={key:stats[key] for key in scheduler.stats}
                    emit({'scheduler':scheduler.report()})
                elif op == 'cancel_events':
                    scheduler.cancel(command['ids'])
                    emit({'cancelled_events':command['ids']})
                elif op == 'pop_event':
                    emit({'event':scheduler.pop(), 'scheduler':scheduler.report()})
                elif op == "configure":
                    for key, bits_per_s in command["capacities"].items():
                        rows = [r for r in active.values() if r["key"] == key]
                        if bits_per_s == 0:
                            for row in rows:
                                if not row["comm"].is_suspended:
                                    row["comm"].suspend()
                        else:
                            links[key].bandwidth = bits_per_s / 8 * BYTE_SCALE
                            for row in rows:
                                if row["comm"].is_suspended:
                                    row["comm"].resume()
                    emit({"configured": True})
                elif op == "cancel":
                    remaining = {}
                    for identity in command["ids"]:
                        row = active.pop(identity, None)
                        if row is not None:
                            remaining[identity] = row["comm"].remaining / BYTE_SCALE
                            row["comm"].cancel()
                    emit({"cancelled": remaining})
                elif op in {"advance", "advance_next"}:
                    target = float(command["until_s"])
                    before_time = sg.Engine.clock
                    before = {
                        identity: row["comm"].remaining
                        for identity, row in active.items()
                    }
                    if target < before_time - 1e-12:
                        raise ValueError("SimGrid cannot move backwards")
                    if op == 'advance_next':
                        scheduler.advance(target)
                    elif target > before_time:
                        if active:
                            activities = sg.ActivitySet(
                                [r["comm"] for r in active.values()]
                            )
                            try:
                                activities.wait_any_for(target - before_time)
                            except sg.TimeoutException:
                                pass
                        else:
                            sg.this_actor.sleep_until(target)
                    done = sorted(
                        identity
                        for identity, row in active.items()
                        if row["comm"].test()
                    )
                    progress = {
                        identity: max(0.0, amount - active[identity]["comm"].remaining)
                        / BYTE_SCALE
                        for identity, amount in before.items()
                    }
                    service_errors = {identity:(math.ulp(amount)+math.ulp(active[identity]['comm'].remaining))*2/BYTE_SCALE
                                      + links[active[identity]['key']].bandwidth/BYTE_SCALE*1e-15
                                      for identity,amount in before.items()}
                    elapsed = sg.Engine.clock - before_time
                    rates = {identity:amount*8/elapsed for identity,amount in progress.items()} if elapsed>0 else {}
                    resources = {}
                    for identity,rate in rates.items():
                        key = active[identity]['key']
                        metadata = link_metadata[key]
                        for resource,capacity in [(key,links[key].bandwidth*8/BYTE_SCALE),
                            *[(pool,pools[pool].bandwidth*8/BYTE_SCALE) for pool in metadata.get('pools',())]]:
                            entry = resources.setdefault(resource,dict(resource_id=resource,capacity_bps=capacity,allocated_bandwidth_bps=0,allocation_error_bps=0))
                            entry['allocated_bandwidth_bps'] += rate
                            entry['allocation_error_bps'] += service_errors[identity]*8/elapsed
                    for identity in done:
                        active.pop(identity)
                    emit(
                        {
                            "completed": done,
                            "bytes_sent": progress,
                            "interval_start_s": before_time,
                            "horizon_reached": sg.Engine.clock >= target,
                            "scheduler": scheduler.report() if scheduler else None,
                            "allocated_rates_bps": rates,
                            "resources": list(resources.values()),
                            "service_clock": "simgrid_float64_seconds",
                            "service_errors_bytes": service_errors,
                            "native_interval_start_s": before_time,
                            "native_interval_end_s": sg.Engine.clock,
                        }
                    )
                else:
                    raise ValueError("unknown SimGrid worker operation")
            except Exception as error:
                emit({"error": f"{type(error).__name__}: {error}"})

    next(iter(hosts.values())).add_actor("orbitserve-controller", controller)
    engine.run()


if __name__ == "__main__":
    main()
