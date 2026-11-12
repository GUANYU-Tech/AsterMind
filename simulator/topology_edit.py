"""Validated topology edits for native scene configuration (never a live run)."""

from copy import deepcopy
from .config import SimulationConfig


def pairs_for(nodes, kind):
    if kind == "fully_connected":
        return [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1 :]]
    if kind not in {"ring", "chain", "switch"}:
        raise ValueError("不支持的拓扑类型")
    if kind == "switch":
        return []
    pairs = list(zip(nodes, nodes[1:]))
    if kind == "ring" and len(nodes) > 2:
        pairs.append((nodes[-1], nodes[0]))
    return pairs


def edit_topology(value, edit):
    allowed = {
        "action",
        "cell",
        "satellite",
        "kind",
        "source",
        "target",
        "link",
        "enabled",
        "bandwidth_gbps",
        "latency_ns",
    }
    if not isinstance(edit, dict) or set(edit) - allowed:
        raise ValueError("拓扑修改包含未知字段")
    config = SimulationConfig.model_validate(value)
    data = deepcopy(config.model_dump(mode="json"))
    graph = config.topology()
    devices = graph.devices
    links = data["physical"].setdefault("links", [])
    bindings = data["physical_link_bindings"]
    action = edit.get("action")
    cell = next(
        (
            c
            for c in data["physical"]["cells"]
            if c["id"] == edit.get("cell")
            and any(s.id == c["id"] for s in config.cells)
        ),
        None,
    )
    if cell is None:
        raise ValueError("请选择有效的计算 Cell")
    members = set(cell["satellites"])

    def local_device(identifier):
        d = devices.get(identifier)
        if d is None or d.satellite not in members:
            raise ValueError("设备不属于所选 Cell")
        return d

    def add_cell_pair(a, b):
        if a not in members or b not in members or a == b:
            raise ValueError("Cell 链路需要同一 Cell 的两颗不同卫星")
        matches = [
            link
            for link in links
            if {devices[link["source"]].satellite, devices[link["target"]].satellite}
            == {a, b}
        ]
        if matches:
            return matches[0]
        terminals = []
        for owner in (a, b):
            found = [
                d.id
                for d in devices.values()
                if d.satellite == owner and d.kind == "terminal"
            ]
            if len(found) != 1:
                raise ValueError("多终端部署请在高级 JSON 中明确指定链路端口")
            terminals.append(found[0])
        data["additional_cell_links"].append(dict(source=a, target=b))
        physical_id = "cell_isl:" + ":".join(sorted((a, b)))
        identifier = "edited/" + physical_id
        internal = [
            link
            for link in links
            if devices[link["source"]].satellite in members
            and devices[link["target"]].satellite in members
            and devices[link["source"]].satellite != devices[link["target"]].satellite
        ]
        rate = internal[0]["bandwidth_gbps"] if internal else 10
        row = dict(
            id=identifier,
            source=terminals[0],
            target=terminals[1],
            bandwidth_gbps=rate,
            bidirectional=True,
            enabled=True,
        )
        links.append(row)
        bindings[identifier] = physical_id
        return row

    disabled = set(data["disabled_physical_links"])
    if action == "cell_topology":
        desired = {frozenset(p) for p in pairs_for(sorted(members), edit.get("kind"))}
        for pair in desired:
            add_cell_pair(*sorted(pair))
        for link in links:
            a, b = devices[link["source"]].satellite, devices[link["target"]].satellite
            if a != b and {a, b} <= members:
                physical_id = bindings[link["id"]]
                if frozenset((a, b)) in desired:
                    disabled.discard(physical_id)
                else:
                    disabled.add(physical_id)
    elif action == "cell_link":
        if type(edit.get("enabled")) is not bool:
            raise ValueError("链路启用状态必须为布尔值")
        row = add_cell_pair(edit.get("source"), edit.get("target"))
        if edit["enabled"]:
            disabled.discard(bindings[row["id"]])
        else:
            disabled.add(bindings[row["id"]])
    elif action == "gpu_topology":
        owner = edit.get("satellite")
        if owner not in members:
            raise ValueError("卫星不属于所选 Cell")
        gpus = sorted(
            d.id for d in devices.values() if d.satellite == owner and d.kind == "gpu"
        )

        def direct(link):
            return (
                devices[link["source"]].satellite == owner
                and devices[link["target"]].satellite == owner
                and devices[link["source"]].kind
                == devices[link["target"]].kind
                == "gpu"
            )

        links[:] = [link for link in links if not direct(link)]
        reference = next(
            (
                link
                for link in links
                if link["source"] in gpus or link["target"] in gpus
            ),
            None,
        )
        for index, (a, b) in enumerate(pairs_for(gpus, edit.get("kind"))):
            links.append(
                dict(
                    id=f"{owner}/gpu-direct-{index}",
                    source=a,
                    target=b,
                    bandwidth_gbps=reference["bandwidth_gbps"] if reference else 100,
                    latency_ns=reference.get("latency_ns", 0) if reference else 0,
                )
            )
    elif action in {"gpu_link", "gpu_add"}:
        if action == "gpu_link":
            row = next((link for link in links if link["id"] == edit.get("link")), None)
            if row is None:
                raise ValueError("GPU 链路不存在")
            a, b = local_device(row["source"]), local_device(row["target"])
        else:
            a, b = local_device(edit.get("source")), local_device(edit.get("target"))
            if a.id == b.id or any(
                {link["source"], link["target"]} == {a.id, b.id} for link in links
            ):
                raise ValueError("两个端点必须不同，且不能重复已有链路")
            row = dict(
                id=f"edited/{a.id}:{b.id}",
                source=a.id,
                target=b.id,
                bandwidth_gbps=edit.get("bandwidth_gbps", 100),
            )
            links.append(row)
        if a.satellite != b.satellite:
            raise ValueError("GPU 互联修改限同一卫星内部")
        for key in ("enabled", "bandwidth_gbps", "latency_ns"):
            if key in edit:
                row[key] = edit[key]
    else:
        raise ValueError("不支持的拓扑修改")
    data["disabled_physical_links"] = sorted(disabled)
    # Candidate counts are declarations; the physical allocator enforces actual
    # orbital terminal concurrency and geometry at every simulated time.
    for device in data["physical"]["devices"]:
        if device["kind"] == "terminal" and device["satellite"] in members:
            count = sum(
                1
                for link in links
                if device["id"] in (link["source"], link["target"])
                and devices[link["source"]].satellite
                != devices[link["target"]].satellite
            )
            device["max_external_links"] = max(device["max_external_links"], count)
    checked = SimulationConfig.model_validate(data)
    checked.topology()
    return checked.model_dump(mode="json")
