"""Export M-L's actual communication DAG without predicting a physical route.

This is a demand description, not a second timing charge or a live transfer.
Repeated layer templates retain recurrence edges: consumers must expand them
before scheduling dynamic links. Native timing remains the working backend.
"""

from dataclasses import asdict, dataclass

from src.kernel.kernel_spec import KernelKind
from src.serving.workload.workload_graph import WorkloadKind

DEVICE_COMMUNICATION = frozenset(
    (
        WorkloadKind.COLLECTIVE,
        WorkloadKind.P2P,
        WorkloadKind.PD_KV_TRANSFER,
        WorkloadKind.AF_ACTIVATION_TRANSFER,
    )
)


@dataclass(frozen=True)
class RankPlacement:
    cell_id: str
    pool_name: str
    rank: int
    satellite_id: str
    gpu_device_id: str

    def __post_init__(self):
        for name in ("cell_id", "pool_name", "satellite_id", "gpu_device_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a nonnegative integer")


def bind_communication_batch(batch, placements):
    """Attach endpoints; logical groups and communication quantities stay intact.

    First slice supports a single pool and exclusive GPUs per Cell. Cross-pool
    PD/AF must carry explicit source/destination pool identities before binding.
    Satellite topology/route availability is validated by the network backend.
    """
    placements = tuple(placements)
    if any(not isinstance(p, RankPlacement) for p in placements):
        raise TypeError("placements require RankPlacement objects")
    identities = {(p.cell_id, p.pool_name, p.rank): p for p in placements}
    if len(identities) != len(placements):
        raise ValueError("duplicate logical rank placement")
    if len({(p.satellite_id, p.gpu_device_id) for p in placements}) != len(placements):
        raise NotImplementedError(
            "multiple logical ranks sharing a GPU are not simulated"
        )
    bound = []
    for demand in batch["demands"]:
        if demand["kind"] not in ("collective", "p2p"):
            raise NotImplementedError(
                "cross-pool PD/AF endpoint binding is not connected"
            )
        endpoints = []
        for rank in demand["participants"]:
            key = (demand["cell_id"], demand["pool_name"], rank)
            if key not in identities:
                raise ValueError(f"unmapped communication participant: {key}")
            endpoints.append(asdict(identities[key]))
        bound.append({**demand, "physical_endpoints": endpoints})
    return {**batch, "demands": bound}


def communication_batch(cell_id, step_id, components):
    graph, plan = components.workload, components.kernel_plan
    calls_by_node = {}
    for call in plan.calls:
        for node_id in call.source_node_ids:
            calls_by_node.setdefault(node_id, []).append(call)
    repeats_by_node = {
        node_id: repeat for repeat in graph.block_repeats for node_id in repeat.node_ids
    }
    predecessors = {}
    successors = {}
    for edge in graph.edges:
        predecessors.setdefault(edge.destination, []).append(edge.source)
        successors.setdefault(edge.source, []).append(edge.destination)
    demands = []
    for node in graph.nodes:
        if node.kind not in DEVICE_COMMUNICATION:
            continue
        attributes = node.attributes
        if attributes.get("bytes_semantics") not in ("per_rank_input", "payload"):
            raise ValueError(
                f"communication byte semantics unspecified: {node.node_id}"
            )
        participants = tuple(attributes["participants"])
        if len(participants) < 2:
            raise ValueError("device communication requires at least two participants")
        operation = attributes.get("operation", node.variant)
        if not isinstance(operation, str) or not operation:
            raise ValueError(f"communication operation unspecified: {node.node_id}")
        calls = calls_by_node[node.node_id]
        repeat = repeats_by_node.get(node.node_id)
        demands.append(
            {
                "demand_id": f"{cell_id}/{step_id}/{node.node_id}",
                "cell_id": cell_id,
                "step_id": step_id,
                "node_id": node.node_id,
                "kind": node.kind.value,
                "operation": operation,
                "pool_name": node.pool_name,
                "participants": list(participants),
                "dtype": node.dtype,
                "bytes": attributes["bytes"],
                "bytes_semantics": attributes["bytes_semantics"],
                "predecessor_node_ids": predecessors.get(node.node_id, []),
                "successor_node_ids": successors.get(node.node_id, []),
                "kernel_call_ids": [c.call_id for c in calls],
                "kernel_dependencies": {c.call_id: list(c.dependencies) for c in calls},
                "native_call_duration_s": {
                    c.call_id: components.predictions[c.call_id].t_total_s
                    for c in calls
                },
                "repeat_block_id": None if repeat is None else repeat.block_id,
                "repeat_count": 1 if repeat is None else repeat.repeat_count,
                # Keep algorithm/resource attributes for the backend to interpret;
                # rank-to-device placement is deliberately a separate input.
                "attributes": dict(attributes),
            }
        )
    return {
        "schema_version": 1,
        "cell_id": cell_id,
        "step_id": step_id,
        "status": "native_timing_with_exported_demand_templates",
        "demands": demands,
        "repeated_blocks": [
            {
                "block_id": r.block_id,
                "node_ids": list(r.node_ids),
                "repeat_count": r.repeat_count,
                "feedback_edges": [
                    {"source": e.source, "destination": e.destination, "kind": e.kind}
                    for e in r.feedback_edges
                ],
                "trace_node_ids": [(n, list(ids)) for n, ids in r.trace_node_ids],
            }
            for r in graph.block_repeats
        ],
    }


def kernel_communication_batch(cell_id, step_id, components):
    """Export the executed communication of a composed, uncompressed plan.

    A composed plan can replace native graph nodes and add DCP collectives;
    the original workload remains lineage, not the executed dependency DAG.
    """
    plan = components.kernel_plan
    if plan.block_repeats:
        raise NotImplementedError(
            "composed communication export requires an uncompressed plan"
        )
    users = {call.call_id: [] for call in plan.calls}
    for call in plan.calls:
        for dependency in call.dependencies:
            users[dependency].append(call.call_id)
    demands = []
    for call in plan.calls:
        if call.kernel_spec.kind is not KernelKind.COMMUNICATION:
            continue
        transfer = call.kernel_spec.transfer
        demands.append(
            {
                "demand_id": f"{cell_id}/{step_id}/{call.call_id}",
                "cell_id": cell_id,
                "step_id": step_id,
                "node_id": call.call_id,
                "kind": "p2p" if transfer.operation == "send_recv" else "collective",
                "operation": transfer.operation,
                "pool_name": call.pool_name,
                "participants": list(transfer.participants),
                "dtype": call.kernel_spec.dtype,
                "bytes": transfer.bytes,
                "bytes_semantics": transfer.bytes_semantics,
                "predecessor_node_ids": list(call.dependencies),
                "successor_node_ids": users[call.call_id],
                "kernel_call_ids": [call.call_id],
                "kernel_dependencies": {call.call_id: list(call.dependencies)},
                "native_call_duration_s": {
                    call.call_id: components.predictions[call.call_id].t_total_s
                },
                "repeat_block_id": None,
                "repeat_count": 1,
                "attributes": asdict(transfer),
            }
        )
    return {
        "schema_version": 1,
        "cell_id": cell_id,
        "step_id": step_id,
        "status": "native_timing_with_exported_kernel_demands",
        "demands": demands,
        "repeated_blocks": [],
        "dependency_scope": "executed KernelPlan; original WorkloadGraph is native lineage",
    }
