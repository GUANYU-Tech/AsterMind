"""Physical network exchange for AsterMind/FlexET/ASTRA-SIM consumers.

The adjacent AsterMind currently accepts one homogeneous static network. The
exact snapshot below is deliberately not presented as that narrower schema.
"""

from __future__ import annotations


def export_cell_network(snapshot):
    nodes = snapshot["nodes"]
    cells = sorted({n["cell_id"] for n in nodes if n.get("cell_id")})
    result = []
    for cell_id in cells:
        members = sorted(
            (n for n in nodes if n.get("cell_id") == cell_id),
            key=lambda n: n["node_id"],
        )
        ids = {n["node_id"] for n in members}
        links = [
            dict(
                link,
                bandwidth_gbps_per_direction=(
                    link["directions"][0]["bandwidth_bps"] / 1e9
                    if link.get("directions")
                    and len({d["bandwidth_bps"] for d in link["directions"]}) == 1
                    else None
                    if link.get("directions")
                    else link["bandwidth_bps"] / 1e9
                ),
                bandwidth_gbps_by_direction=[
                    dict(
                        source_node_id=d["source_node_id"],
                        target_node_id=d["target_node_id"],
                        bandwidth_gbps=d["bandwidth_bps"] / 1e9,
                        shared_pools=d.get("shared_pools", []),
                    )
                    for d in link.get("directions", ())
                ],
                latency_ns=link["propagation_delay_s"] * 1e9,
            )
            for link in snapshot["links"]
            if link["endpoint_a"] in ids or link["endpoint_b"] in ids
        ]
        internal = [link for link in links if link["kind"] == "cell_isl"]
        count = len(members)
        degree = {n: 0 for n in ids}
        for link in internal:
            degree[link["endpoint_a"]] += 1
            degree[link["endpoint_b"]] += 1
        topology = (
            "FullyConnected"
            if len(internal) == count * (count - 1) // 2
            else "Ring"
            if count > 2 and all(v == 2 for v in degree.values())
            else None
        )
        static = None
        if (
            internal
            and topology
            and all(
                link["state"] == "UP" and link["bandwidth_bps"] > 0 for link in internal
            )
        ):
            static = dict(
                network=dict(
                    topology=topology,
                    bandwidth_gbps=min(link["bandwidth_bps"] for link in internal)
                    / 1e9,
                    latency_ns=max(link["propagation_delay_s"] for link in internal)
                    * 1e9,
                ),
                approximation="static_uniform_lower_bandwidth_upper_propagation_envelope",
                valid_until_s=snapshot["valid_until_s"],
                excludes="external_links_faults_shared_runtime_and_orbital_dynamics",
            )
        result.append(
            dict(
                cell_id=cell_id,
                deployment=dict(rank_mapping={str(i): i for i in range(len(members))}),
                physical_rank_mapping=[
                    dict(
                        rank=i,
                        node=i,
                        node_id=n["node_id"],
                        position_itrs_km=n["position_itrs_km"],
                    )
                    for i, n in enumerate(members)
                ],
                links=links,
                astermind_static_fragment=static,
                serving_formation_fragment=(
                    dict(
                        member_count=count,
                        topology=topology.lower().replace(
                            "fullyconnected", "fully_connected"
                        ),
                        link_bandwidth_gbps=static["network"]["bandwidth_gbps"],
                        link_propagation_delay_s=static["network"]["latency_ns"] / 1e9,
                    )
                    if static
                    else None
                ),
                astermind_compatibility=dict(
                    exact_static_import=False,
                    reason="current_AsterMind_NetworkConfig_cannot_represent_dynamic_per_edge_graphs",
                    adapter_required=True,
                    bandwidth_conversion="bits_per_second / 1e9 = Gbps; AsterMind converts Gbps / 8 for ASTRA",
                    latency_scope="propagation_only_queue_and_serialization_simulated_separately",
                    internal_link_count=len(internal),
                ),
            )
        )
    return dict(
        schema_version="orbitserve.astermind_network_exchange.v1",
        time_s=snapshot["time_s"],
        valid_until_s=snapshot["valid_until_s"],
        shared_capacity_pools_bps=snapshot.get("shared_capacity_pools_bps", {}),
        radio_parameters={
            n["node_id"]: n["radio_parameters"]
            for n in nodes
            if n.get("radio_parameters")
        },
        units=dict(
            bandwidth_bps="bit/s per direction", latency_ns="ns", position_itrs_km="km"
        ),
        duplex="independent_full_duplex_directions",
        cells=result,
    )
