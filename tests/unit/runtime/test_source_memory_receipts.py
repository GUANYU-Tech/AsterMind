"""Deferred receipts reuse the metadata detached by the source snapshot."""

from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest


def test_deferred_trace_receipt_reuses_the_frozen_snapshot():
    from astermind.common.bootstrap import activate

    activate()
    from src.serving.kernel_lowering.memory_bindings import snapshot_trace_memory
    from src.serving.workload.traced_forward import ForwardTrace
    from src.serving.workload.workload_graph import WorkloadKind, WorkloadNode

    trace = ForwardTrace(rank=0, namespace="receipt")
    source = trace.tensor("input", (1,), "bf16")
    mutable = {"rows": [1, 2]}
    node = WorkloadNode("op", WorkloadKind.VECTOR, "main", {"elements": 1},
                        dtype="bf16", rank=0, attributes={"metadata": mutable})
    trace.call(node, (source,), shape=(1,), dtype="bf16")
    graph = trace.graph
    context = snapshot_trace_memory(trace, graph)
    node_id = graph.nodes[0].node_id
    frozen = context._by_node.rows[node_id][0].attributes

    mutable["rows"].append(3)
    receipt = context._by_node[node_id]
    assert receipt.attributes is frozen
    assert receipt.attributes["metadata"]["rows"] == (1, 2)
    with pytest.raises(TypeError):
        receipt.attributes["metadata"]["rows"] = (0,)

    # A read-only outer mapping can still wrap mutable source values.
    ordinary = replace(receipt, attributes=MappingProxyType({"metadata": mutable}))
    mutable["rows"].append(4)
    assert ordinary.attributes["metadata"]["rows"] == (1, 2, 3)



def test_binding_reuses_live_backing_and_preserves_original_free_event():
    from astermind.common.bootstrap import activate

    activate()
    from src.kernel.memory_access import AddressLevel, EventKind, MemoryAllocation, ProgramEvent
    from src.serving.kernel_lowering.memory_bindings import source_memory_events

    live = MemoryAllocation("live", "rank:0", "dram", 16, AddressLevel.OBJECT,
                            None, None, "allocate-live", None)
    temporary = replace(live, allocation_id="temporary", allocated_event="allocate-temporary",
                        freed_event="free-temporary")
    call = SimpleNamespace(call_id="call", pool_name="main", rank=0)
    source = SimpleNamespace(allocations=(live, temporary), predecessors=())

    allocations, events, ready, visibility = source_memory_events(call, source, ())

    assert allocations[0] is live
    assert allocations[1] == replace(temporary, freed_event=None)
    assert temporary.freed_event == "free-temporary"
    assert events == (
        ProgramEvent("allocate-live", EventKind.ALLOCATED, "live", None, None, ()),
        ProgramEvent("allocate-temporary", EventKind.ALLOCATED, "temporary", None, None, ()),
    )
    assert ready == () and visibility == "main:rank:0"
