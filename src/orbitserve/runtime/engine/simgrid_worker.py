"""Private RPC worker owning one real SimGrid engine per Runtime simulation.

Only this worker imports SimGrid. A controller actor waits for native activity
completion or the next external event, so neither engine runs past a causal
boundary. Native link latency is zero: OrbitServe accounts for propagation after
serialization. Capacities are converted from bits/s to SimGrid bytes/s here.
"""

from __future__ import annotations

import json
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
            "--cfg=precision/timing:1e-12",
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
                elif op == "advance":
                    target = float(command["until_s"])
                    before_time = sg.Engine.clock
                    before = {
                        identity: row["comm"].remaining
                        for identity, row in active.items()
                    }
                    if target < before_time - 1e-12:
                        raise ValueError("SimGrid cannot move backwards")
                    if target > before_time:
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
                    for identity in done:
                        active.pop(identity)
                    emit(
                        {
                            "completed": done,
                            "bytes_sent": progress,
                            "interval_start_s": before_time,
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
