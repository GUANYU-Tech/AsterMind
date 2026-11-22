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


def test_packed_batch_rows_are_shared_by_source_receipts():
    from astermind.common.bootstrap import activate

    activate()
    from src.serving.kernel_lowering.memory_bindings import snapshot_trace_memory
    from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
    from src.serving.workload.runtime_inputs import vllm_0_30_source_model_inputs
    from src.serving.workload.traced_forward import ForwardTrace
    from src.serving.workload.workload_graph import WorkloadKind, WorkloadNode

    step = ScheduleStep("mixed", "main", (
        ScheduledRequest("prefill", "prefill", 3, 0),
        ScheduledRequest("decode", "decode", 0, 1, computed_tokens=5),
    ))
    packed, ids, order = vllm_0_30_source_model_inputs(
        step, embedding_token_ids=(10, 11, 12, 13), prompt_lengths=(3, 5),
    )
    assert tuple(request.request_id for request in packed.requests) == ("decode", "prefill")
    assert ids == (13, 10, 11, 12)
    assert order["token_row_permutation"] == (3, 0, 1, 2)
    assert order["model_positions"] == (5, 0, 1, 2)

    trace = ForwardTrace(rank=0, namespace="shared_batch")
    source = trace.tensor("input", (4,), "bf16")
    for name in ("first", "second"):
        node = WorkloadNode(name, WorkloadKind.VECTOR, "main", {"elements": 4},
                            dtype="bf16", rank=0, attributes={"model_input_order": order})
        trace.call(node, (source,), shape=(4,), dtype="bf16")
    graph = trace.graph
    receipts = snapshot_trace_memory(trace, graph).nodes

    for receipt in receipts:
        saved = receipt.attributes["model_input_order"]
        assert saved["token_row_permutation"] is order["token_row_permutation"]
        assert saved["model_positions"] is order["model_positions"]



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
