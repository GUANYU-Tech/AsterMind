"""Deterministic physical neighbor selection, independent of orbital geometry."""

from __future__ import annotations

from dataclasses import replace

from ..setup.network_spec import NetworkSpec


def candidate_links(spec: NetworkSpec) -> tuple[dict, ...]:
    nodes = {n.node_id: n for n in spec.nodes}
    edges: dict[tuple[str, str], str] = {}
    patches: set[tuple[str, str]] = set()

    def add(a: str, b: str, kind: str) -> None:
        if a not in nodes or b not in nodes or a == b:
            raise ValueError("candidate edge requires two known distinct nodes")
        first, second = nodes[a], nodes[b]
        if first.role == second.role == 'ground':
            raise ValueError('ground-to-ground candidate links are unsupported')
        if first.cell_id != second.cell_id and any(
            not spec.external_member(n) for n in (first, second)
        ):
            raise ValueError('external link bypasses cell egress policy')
        if (
            first.role == second.role == "communication"
            and first.shell_id != second.shell_id
        ):
            raise ValueError(
                "communication shells are independent; cross-shell links are unsupported"
            )
        if kind not in spec.link_profiles:
            raise ValueError(f"missing network link profile: {kind}")
        edges[tuple(sorted((a, b)))] = kind

    comm = sorted(
        (n for n in spec.nodes if n.role == "communication"), key=lambda n: n.node_id
    )
    if spec.communication_shells:
        for shell in spec.communication_shells:
            subset = tuple(n for n in comm if n.shell_id == shell["shell_id"])
            local = replace(
                spec,
                nodes=subset,
                communication_shells=(),
                communication_topology=shell["topology"],
                wrap_planes=shell.get("wrap_planes", True),
                wrap_slots=shell.get("wrap_slots", True),
                cross_plane_offset=shell.get("cross_plane_offset", 0),
                explicit_pairs=tuple(tuple(p) for p in shell.get('explicit_pairs', ())),
                cell_topologies={},
                cell_egress_policies={},
                faults=(),
                communication_grid_shape=shell.get('grid_shape'),
                topology_repair=shell.get('topology_repair', spec.topology_repair),
            )
            for edge in candidate_links(local):
                add(edge["endpoint_a"], edge["endpoint_b"], edge["kind"])
                if edge.get('topology_patch'):
                    patches.add(tuple(sorted((edge['endpoint_a'], edge['endpoint_b']))))
    if (
        not spec.communication_shells
        and spec.communication_topology in {"grid", "honeycomb"}
        and comm
    ):
        placed = [n for n in comm if n.plane is not None and n.slot is not None]
        if any((n.plane is None) != (n.slot is None) for n in comm):
            raise ValueError('plane and slot must both be set or both be null')
        if spec.communication_grid_shape is None:
            if len(placed) != len(comm):
                raise ValueError('unplaced nodes require communication_grid_shape')
            p_count = max(n.plane for n in placed) + 1
            s_count = max(n.slot for n in placed) + 1
        else:
            p_count, s_count = spec.communication_grid_shape
        index = {(n.plane, n.slot): n.node_id for n in placed}
        if len(index) != len(placed):
            raise ValueError('duplicate communication grid position')
        if any(p >= p_count or s >= s_count for p, s in index):
            raise ValueError('communication grid position outside grid_shape')
        if spec.communication_topology == "honeycomb":
            if spec.cross_plane_offset:
                raise ValueError("honeycomb currently requires cross_plane_offset=0")
            if (spec.wrap_planes and p_count > 1 and p_count % 2) or (spec.wrap_slots and s_count > 1 and s_count % 2):
                raise ValueError("periodic honeycomb requires even wrapped dimensions")
        def grid_edge(a, b, patch=False):
            if a in index and b in index and a != b:
                add(index[a], index[b], 'communication_isl')
                if patch:
                    patches.add(tuple(sorted((index[a], index[b]))))
        for p, s in sorted(index):
            if s + 1 < s_count or spec.wrap_slots:
                grid_edge((p, s), (p, (s + 1) % s_count))
            if spec.communication_topology == "honeycomb" and (p + s) % 2:
                continue
            q, r = p + 1, s + spec.cross_plane_offset
            if q >= p_count:
                if not spec.wrap_planes:
                    continue
                q %= p_count
            if not 0 <= r < s_count:
                if not spec.wrap_slots:
                    continue
                r %= s_count
            grid_edge((p, s), (q, r))
        if spec.topology_repair == 'ronchetti_intra_orbit':
            # §4 patched hexGrid: bridge only consecutive intra-orbit holes.
            # Zero-laser satellites count as holes. Grid uses the same extension.
            for p in range(p_count):
                occupied = sorted(n.slot for n in placed if n.plane == p and n.isl_terminals > 0)
                pairs = list(zip(occupied, occupied[1:]))
                if spec.wrap_slots and len(occupied) > 1:
                    pairs.append((occupied[-1], occupied[0] + s_count))
                for a, b in pairs:
                    if b - a > 1:
                        grid_edge((p, a), (p, b % s_count), patch=True)

    compute = sorted(
        (n for n in spec.nodes if n.role == "compute"), key=lambda n: n.node_id
    )
    cells = sorted({n.cell_id for n in compute if n.cell_id})
    for cell in cells:
        members = [n.node_id for n in compute if n.cell_id == cell]
        for i, a in enumerate(members):
            for j in range(i + 1, len(members)):
                if (
                    spec.cell_topologies.get(cell, spec.cell_topology)
                    == "fully_connected"
                    or j == i + 1
                    or (i == 0 and j == len(members) - 1)
                ):
                    add(a, members[j], "cell_isl")
    ingress = [n for n in compute if spec.external_member(n)]
    for n in ingress:
        for relay in comm:
            add(n.node_id, relay.node_id, "cross_layer")
    if spec.inter_cell_links:
        for i, a in enumerate(ingress):
            for b in ingress[i + 1 :]:
                if a.cell_id != b.cell_id:
                    add(a.node_id, b.node_id, "inter_cell")
    for ground in (n for n in spec.nodes if n.role == "ground"):
        for node in spec.nodes:
            if node.role in spec.ground_roles and spec.external_member(node):
                add(ground.node_id, node.node_id, "sgl")
    for pair in spec.explicit_pairs:
        if len(pair) != 2:
            raise ValueError("explicit_pairs requires endpoint pairs")
        a, b = pair
        add(
            a,
            b,
            "sgl"
            if any(nodes[x].role == "ground" for x in pair)
            else "communication_isl",
        )
    shell_configs = {row["shell_id"]: row for row in spec.communication_shells}

    def profile_id(a, b, kind):
        shell = nodes[a].shell_id
        profile = (
            shell_configs.get(shell, {}).get("isl_profile", kind)
            if kind == "communication_isl" and shell == nodes[b].shell_id
            else kind
        )
        if profile not in spec.link_profiles:
            raise ValueError(f"missing link profile: {profile}")
        return profile

    return tuple(
        {
            "profile_id": profile_id(a, b, kind),
            "link_id": f"{kind}:{a}:{b}",
            "endpoint_a": a,
            "endpoint_b": b,
            "kind": kind,
            **({'topology_patch': 'ronchetti_intra_orbit'} if (a, b) in patches else {}),
        }
        for (a, b), kind in sorted(edges.items())
    )
