"""Deterministic synthetic acceptance fixtures, not research parameters."""
import json
import copy
from pathlib import Path


def example():
    satellites = [dict(id=s) for s in ("satA", "satB", "ground")]
    devices = [dict(id="user", satellite="ground", kind="terminal", external_access=True)]
    links = []
    for index, satellite in enumerate(("satA", "satB")):
        devices.extend([dict(id=f"gpu{index}", satellite=satellite, kind="gpu"),
                        dict(id=f"switch{index}", satellite=satellite, kind="switch", capacity_gbps=8),
                        dict(id=f"isl{index}", satellite=satellite, kind="terminal"),
                        dict(id=f"exit{index}", satellite=satellite, kind="terminal", external_access=True)])
        links.extend([dict(id=f"local{index}", source=f"gpu{index}", target=f"switch{index}", bandwidth_gbps=64),
                      dict(id=f"to-isl{index}", source=f"switch{index}", target=f"isl{index}", bandwidth_gbps=8),
                      dict(id=f"to-exit{index}", source=f"switch{index}", target=f"exit{index}", bandwidth_gbps=8)])
    devices[0]["max_external_links"] = 2
    links.append(dict(id="isl", source="isl0", target="isl1", bandwidth_gbps=8, latency_ns=2000))
    links.extend(dict(id=f"ground{index}", source="user", target=f"exit{index}", bandwidth_gbps=1,
                      latency_ns=10000) for index in (0, 1))
    physical = dict(cells=[dict(id="cellA", satellites=["satA"]), dict(id="cellB", satellites=["satB"]),
                           dict(id="ground-resources", satellites=["ground"])],
                    satellites=satellites, devices=devices, links=links)
    # ISL is inter-Cell, so both terminals explicitly permit external access.
    for d in devices:
        if d["id"].startswith("isl"):
            d["external_access"] = True
    cells = [dict(id=f"cell{label}", placements=[dict(rank=0, device=f"gpu{index}")],
                  ingress_device=f"gpu{index}", output_device=f"gpu{index}",
                  external_terminals=[f"exit{index}", f"isl{index}"], max_batched_tokens=16)
             for index, label in enumerate(("A", "B"))]
    requests = [dict(id=f"r{index}", arrival_ns=at, user_device="user", input_tokens=prompt,
                     output_tokens=2, input_bytes=200000, output_bytes_per_token=10000)
                for index, (at, prompt) in enumerate(((0, 16), (0, 32), (2000000, 24)))]
    return dict(physical=physical, ground_owners=["ground"], cells=cells, requests=requests,
                satellite_roles=dict(satA="mixed", satB="mixed"), horizon_ns=1_000_000_000)


def sequential_cells(count=4):
    """Staggered requests activate independent Cells sharing a ground terminal."""
    physical = dict(cells=[dict(id="ground-resources", satellites=["ground"])],
        satellites=[dict(id="ground")], devices=[dict(id="user", satellite="ground", kind="terminal",
        external_access=True, capacity_gbps=1, max_external_links=count)], links=[])
    cells, requests, roles = [], [], {}
    for index in range(count):
        cell, satellite = f"cell{index}", f"sat{index}"
        physical["cells"].append(dict(id=cell, satellites=[satellite]))
        physical["satellites"].append(dict(id=satellite))
        physical["devices"].extend([dict(id=f"gpu{index}", satellite=satellite, kind="gpu"),
            dict(id=f"exit{index}", satellite=satellite, kind="terminal", external_access=True)])
        physical["links"].extend([dict(id=f"local{index}", source=f"gpu{index}", target=f"exit{index}", bandwidth_gbps=8),
            dict(id=f"ground{index}", source="user", target=f"exit{index}", bandwidth_gbps=1, latency_ns=10000)])
        cells.append(dict(id=cell, placements=[dict(rank=0, device=f"gpu{index}")],
            ingress_device=f"gpu{index}", output_device=f"gpu{index}", external_terminals=[f"exit{index}"],
            max_batched_tokens=32))
        roles[satellite] = "mixed"
        for wave in range(3):
            requests.append(dict(id=f"r{index}_{wave}", cell=cell, arrival_ns=index*3_000_000+wave*12_000_000,
                user_device="user", input_tokens=32 if wave == 1 else 16, output_tokens=4,
                input_bytes=250000, output_bytes_per_token=10000))
    return dict(physical=physical, ground_owners=["ground"], satellite_roles=roles,
                cells=cells, requests=sorted(requests, key=lambda r: r["arrival_ns"]), horizon_ns=1_000_000_000)


if __name__ == "__main__":
    here = Path(__file__).parent
    config = example()
    (here / "two_cells.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    packet = copy.deepcopy(config)
    packet["backend"] = "ns3"
    packet["requests"] = packet["requests"][:2]
    (here / "two_cells_ns3.json").write_text(json.dumps(packet, indent=2), encoding="utf-8")
    migration = copy.deepcopy(config)
    for c in migration["cells"]:
        c["prefix_caching"] = True
    migration["requests"] = [dict(id="context0", cell="cellA", arrival_ns=0, user_device="user",
        input_tokens=32, output_tokens=2, input_bytes=128, session_id="session", context_id="c0"),
        dict(id="context1", cell="cellB", arrival_ns=200000000, user_device="user", input_tokens=48,
        output_tokens=2, input_bytes=64, session_id="session", context_id="c1", parent_context_id="c0", prefix_hit_tokens=32)]
    migration["migrations"] = [dict(id="copy0", after_request="context0", source_cell="cellA", target_cell="cellB",
                                    session_id="session", context_id="c0")]
    (here / "session_migration.json").write_text(json.dumps(migration, indent=2), encoding="utf-8")
    multi_migration = copy.deepcopy(migration)
    for cell_index, (satellite, gpu) in enumerate((("satA", 2), ("satB", 3))):
        multi_migration["physical"]["devices"].append(dict(id=f"gpu{gpu}", satellite=satellite, kind="gpu"))
        multi_migration["physical"]["links"].append(dict(id=f"local{gpu}", source=f"gpu{gpu}",
            target=f"switch{cell_index}", bandwidth_gbps=64))
        multi_migration["cells"][cell_index].update(parallel=dict(tp=2),
            placements=[dict(rank=0, device=f"gpu{cell_index}"), dict(rank=1, device=f"gpu{gpu}")])
    (here / "session_migration_tp2.json").write_text(json.dumps(multi_migration, indent=2), encoding="utf-8")
    (here / "sequential_cells.json").write_text(json.dumps(sequential_cells(), indent=2), encoding="utf-8")
    heterogeneous = example()
    heterogeneous["cells"][1].update(hardware="configs/hardware/metax_c500.yaml", hardware_id="metax_c500")
    (here / "heterogeneous_cells.json").write_text(json.dumps(heterogeneous, indent=2), encoding="utf-8")
    config["physical"]["cells"] = [dict(id="cellA", satellites=["satA", "satB"]),
                                      dict(id="ground-resources", satellites=["ground"])]
    config["cells"] = [{**config["cells"][0], "parallel": dict(tp=2),
        "placements": [dict(rank=i, device=f"gpu{i}") for i in (0, 1)],
        "egress_mode": "multi_gateway", "external_terminals": ["exit0", "exit1"]}]
    for r in config["requests"]:
        r["cell"] = "cellA"
    (here / "tp2_two_satellites.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    moe = copy.deepcopy(config)
    moe["cells"][0].update(model="configs/model/qwen3-30b-a3b.json", native_expert_parallel=True,
                            expert_routing_seed=42)
    moe["requests"] = [dict(id="moe", arrival_ns=0, user_device="user", cell="cellA",
                            input_tokens=8, output_tokens=2, input_bytes=128)]
    (here / "native_moe_tp2_ep2.json").write_text(json.dumps(moe, indent=2), encoding="utf-8")
