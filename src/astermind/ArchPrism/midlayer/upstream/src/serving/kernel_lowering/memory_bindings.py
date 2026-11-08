"""Project the current source trace into typed per-call memory bindings.

This module owns no allocations. Every backing and view comes from the actual
ForwardTrace; unsupported operand mappings remain explicit on the KernelCall.
"""
from dataclasses import dataclass, fields, replace
from math import prod
from types import MappingProxyType
from collections.abc import Mapping

from src.kernel.kernel_spec import KernelKind
from src.kernel.memory_access import (
    EventKind, EventRef, MemoryAllocation, ProgramEvent, TensorView, Unmapped,
)
from src.kernel.memory_access_builder import ElementwiseBindings, GEMMWork, MemoryBindings
from src.serving.workload.workload_graph import WorkloadKind


@dataclass(frozen=True)
class SourceCallMemory:
    node_id: str
    rank: int
    pool_name: str
    inputs: tuple[TensorView, ...]
    outputs: tuple[TensorView, ...]
    input_producers: tuple[str | None, ...]
    allocations: tuple[MemoryAllocation, ...]
    predecessors: tuple[str, ...]
    kind: WorkloadKind
    variant: str | None
    attributes: Mapping[str, object]

    def __post_init__(self):
        if not isinstance(self.attributes, Mapping):
            raise TypeError("memory source attributes require the actual source metadata mapping")
        object.__setattr__(self, "attributes", _source_metadata(self.attributes, frozen=True))
        if not isinstance(self.kind, WorkloadKind):
            raise TypeError("memory source requires its actual operation kind")
        if self.variant is not None and (type(self.variant) is not str or not self.variant):
            raise ValueError("memory source variant must be explicit or absent")
        if (type(self.node_id) is not str or not self.node_id or type(self.rank) is not int or self.rank < 0
                or type(self.pool_name) is not str or not self.pool_name):
            raise ValueError("memory source requires a named physical-rank call")
        if any(type(values) is not tuple for values in
               (self.inputs, self.outputs, self.input_producers, self.allocations, self.predecessors)):
            raise TypeError("memory source facts must be immutable tuples")
        if len(self.inputs) != len(self.input_producers):
            raise ValueError("source readiness must cover every input port")
        if any(producer is not None and (type(producer) is not str or not producer)
               for producer in self.input_producers):
            raise ValueError("source producer identities must be nonempty or external")
        allocations = {item.allocation_id: item for item in self.allocations}
        if len(allocations) != len(self.allocations):
            raise ValueError("duplicate source allocations")
        for allocation in self.allocations:
            if type(allocation) is not MemoryAllocation:
                raise TypeError("source allocation must use the canonical memory contract")
            allocation.__post_init__()
        for view in (*self.inputs, *self.outputs):
            if type(view) is not TensorView or view.allocation_id not in allocations:
                raise ValueError("source port must reference its actual backing")
            view.__post_init__()

    def __reduce__(self):
        return type(self), tuple(_source_metadata(getattr(self, item.name), frozen=False)
            if item.name == "attributes" else getattr(self, item.name) for item in fields(self))


def _source_metadata(value, *, frozen):
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("source attribute keys must be strings")
        result = {key: _source_metadata(item, frozen=frozen) for key, item in value.items()}
        return MappingProxyType(result) if frozen else result
    if isinstance(value, (tuple, list)):
        return tuple(_source_metadata(item, frozen=frozen) for item in value)
    return value


@dataclass(frozen=True)
class TraceMemoryContext:
    """A detached receipt of current source-owned ports, never a storage arena."""
    nodes: tuple[SourceCallMemory, ...]

    def __post_init__(self):
        if type(self.nodes) is not tuple or any(type(node) is not SourceCallMemory for node in self.nodes):
            raise TypeError("trace memory context requires immutable source-call facts")
        names = tuple(node.node_id for node in self.nodes)
        if len(names) != len(set(names)):
            raise ValueError("trace memory source identities must be unique")
        object.__setattr__(self, "_by_node", MappingProxyType({node.node_id: node for node in self.nodes}))

    def __reduce__(self):
        # Rebuild the immutable derived index at deserialization; only the
        # actual source facts cross an execution/export process boundary.
        return type(self), (self.nodes,)

    def bind_call(self, call):
        if len(call.source_node_ids) != 1:
            return replace(call, memory_bindings=None, memory_source=None,
                memory_unbound_reason="fused_source_operand_mapping_unavailable")
        node_id, = call.source_node_ids
        if node_id not in self._by_node:
            return replace(call, memory_bindings=None, memory_source=None,
                memory_unbound_reason="source_tensor_ports_unavailable")
        source = self._by_node[node_id]
        if source.rank != call.rank or source.pool_name != call.pool_name:
            raise ValueError("kernel call and source memory rank disagree")
        spec = call.kernel_spec
        if spec.kind in (KernelKind.KV_CACHE, KernelKind.FLASH_ATTN, KernelKind.SPARSE_ATTN, KernelKind.SPARSE_ATTN_INDEXER):
            from .attention_memory_bindings import bind_attention_memory
            binding, reason = bind_attention_memory(call, source)
        elif (spec.kind is KernelKind.FUSED_MOE or spec.kind is KernelKind.VECTOR
                and isinstance(spec.variant, str) and (spec.variant.startswith("moe_") or spec.variant.endswith("_moe_route"))):
            from .moe_memory_bindings import bind_moe_memory
            binding, reason = bind_moe_memory(call, source)
        elif spec.kind is KernelKind.GEMM:
            binding, reason = _bind_gemm(call, source)
        elif call.kernel_spec.kind is KernelKind.COMMUNICATION:
            binding, reason = _bind_communication(call, source)
        elif call.kernel_spec.kind is KernelKind.VECTOR:
            binding, reason = _bind_elementwise(call, source)
        else:
            binding, reason = None, "source_family_operand_mapping_unavailable"
        return replace(call, memory_bindings=binding, memory_source=source,
            memory_unbound_reason=reason)


def snapshot_trace_memory(source, graph):
    """Read actual ports in source call order, including alias writer changes.

    Final trace._writers is intentionally not used: an earlier input must see
    the writer that preceded that invocation, not the final writer of storage.
    """
    from src.serving.workload.traced_forward import (
        ForwardTrace, ForwardCollectiveGroup, ForwardPipelineGroup, ForwardActivationGroup, ForwardKVCacheGroup,
    )
    from src.serving.workload.workload_graph import WorkloadKind
    traces = (source,) if isinstance(source, ForwardTrace) else source.traces
    records = []
    graph_ids = {item.node_id for item in graph.nodes}
    owner = source._source if isinstance(source, ForwardKVCacheGroup) else source
    if isinstance(owner, ForwardTrace):
        groups = ()
    elif isinstance(owner, ForwardCollectiveGroup):
        groups = (owner,)
    elif isinstance(owner, (ForwardPipelineGroup, ForwardActivationGroup)):
        groups = owner.collective_groups
    else:
        raise TypeError("memory source must be an actual supported forward owner")
    joined_ids = {local_id: canonical for group in groups
        for canonical, submissions in group._submissions.items() for local_id in submissions.values()}
    transport_kinds = (WorkloadKind.P2P, WorkloadKind.AF_ACTIVATION_TRANSFER)
    collective_parts = {}
    all_views, all_allocations, allocation_events = {}, {}, {}
    for trace in traces:
        events = {event: index for index, event in enumerate(trace.event_order)}
        for view in trace.views:
            if view.view_id in all_views:
                raise ValueError("source ranks must have distinct view identities")
            all_views[view.view_id] = view
        for allocation in trace.allocations:
            if allocation.allocation_id in all_allocations:
                raise ValueError("source ranks must have distinct backing identities")
            all_allocations[allocation.allocation_id] = allocation
            allocation_events[allocation.allocation_id] = events
    for trace in traces:
        allocations = {item.allocation_id: item for item in trace.allocations}
        views = {item.view_id: item for item in trace.views}
        writers = {}
        # Receives are declared by the sending rank, before the destination
        # consumes that backing. Preserve their canonical producer identity.
        for node in graph.nodes:
            if node.kind in transport_kinds:
                for name in node.attributes["symbolic_tensor_outputs"]:
                    if name in views:
                        writers[views[name].allocation_id] = node.node_id
        event_positions = {event: index for index, event in enumerate(trace.event_order)}
        return_positions = {event.split(":", 2)[2]: index for event, index in event_positions.items()
            if event.split(":", 2)[1] == "return"}
        for node in trace._nodes.values():
            canonical = joined_ids[node.node_id] if node.node_id in joined_ids else node.node_id
            if canonical not in graph_ids:
                raise ValueError("memory source trace is not part of the executed graph")
            if "symbolic_tensor_inputs" not in node.attributes or "symbolic_tensor_outputs" not in node.attributes:
                continue
            inputs = tuple(views[name] for name in node.attributes["symbolic_tensor_inputs"])
            outputs = tuple(all_views[name] for name in node.attributes["symbolic_tensor_outputs"])
            ready = tuple(writers[view.allocation_id] if view.allocation_id in writers else None for view in inputs)
            ids = tuple(dict.fromkeys(view.allocation_id for view in (*inputs, *outputs)))
            transport = node.kind in transport_kinds
            if not transport and node.node_id not in return_positions:
                raise ValueError("source tensor call lacks its lifecycle boundary")
            input_ids = {view.allocation_id for view in inputs}
            for name in ids:
                allocation = all_allocations[name]
                positions = allocation_events[name]
                if transport:
                    label = ("send:" if name in input_ids else "receive:") + node.node_id
                    boundaries = tuple(index for event, index in positions.items()
                        if event.split(":", 1)[1] == label)
                    if len(boundaries) != 1:
                        raise ValueError("transport source needs its exact send/receive lifecycle boundary")
                    position, = boundaries
                else:
                    position = return_positions[node.node_id]
                if (positions[allocation.allocated_event] >= position
                        or allocation.freed_event is not None and positions[allocation.freed_event] <= position):
                    raise ValueError("source tensor backing is not live at its call boundary")
            record = SourceCallMemory(node.node_id, trace.rank, node.pool_name, inputs, outputs,
                ready, tuple(all_allocations[name] for name in ids),
                tuple(graph.predecessors(canonical)), node.kind, node.variant, node.attributes)
            if node.node_id in joined_ids:
                if canonical not in collective_parts:
                    collective_parts[canonical] = []
                collective_parts[canonical].append(record)
            else:
                records.append(record)
            for view in outputs:
                writers[view.allocation_id] = canonical
    for canonical, parts in collective_parts.items():
        node = graph.node(canonical)
        ports = {view.view_id: view for part in parts for view in (*part.inputs, *part.outputs)}
        ready_by_port = {view.view_id: producer for part in parts
            for view, producer in zip(part.inputs, part.input_producers, strict=True)}
        inputs = tuple(ports[name] for name in node.attributes["symbolic_tensor_inputs"])
        outputs = tuple(ports[name] for name in node.attributes["symbolic_tensor_outputs"])
        allocations = tuple({item.allocation_id: item for part in parts for item in part.allocations}.values())
        records.append(SourceCallMemory(canonical, node.rank, node.pool_name, inputs, outputs,
            tuple(ready_by_port[view.view_id] for view in inputs), allocations,
            tuple(graph.predecessors(canonical)), node.kind, node.variant, node.attributes))
    return TraceMemoryContext(tuple(records))


def _bind_gemm(call, source):
    spec = call.kernel_spec
    if (source.kind not in (WorkloadKind.LINEAR, WorkloadKind.LM_HEAD, WorkloadKind.ROUTER)
            or spec.kind is not KernelKind.GEMM or spec.variant not in ("gemm", "gemv")):
        return None, "source_family_operand_mapping_unavailable"
    if "epilogue" in spec.attributes and spec.attributes["epilogue"] not in (None, "none"):
        return None, "gemm_epilogue_source_operand_mapping_unavailable"
    if spec.shape["b"] != 1 or len(source.inputs) != 2 or len(source.outputs) != 1:
        return None, "gemm_source_operand_arity_or_batch_unavailable"
    a, b = source.inputs
    c, = source.outputs
    m, n, k = (spec.shape[name] for name in ("m", "n", "k"))
    if (a.shape != (m, k) or b.shape != (n, k) or c.shape != (m, n)
            or a.dtype != spec.dtype or b.dtype != spec.dtype or c.dtype != spec.dtype):
        return None, "gemm_source_operand_geometry_or_dtype_unavailable"
    if a.allocation_id == c.allocation_id or b.allocation_id == c.allocation_id:
        return None, "inplace_gemm_source_unavailable"
    if any(view.strides_bytes[1] != view.element_bytes
            or view.strides_bytes[0] < view.shape[1] * view.element_bytes for view in (a, b, c)):
        return None, "gemm_source_operand_layout_unavailable"
    from src.kernel.memory_cann_gemm import (
        has_cann_gemm_source, has_cann_single_core_gemm_source,
        has_cann_deterministic_gemm_source, cann_split_execution_kernel,
    )
    from src.kernel.memory_gpu_split_gemm import has_gpu_split_gemm_abi
    from src.kernel.memory_access_builder import has_public_tensor_gemm_source
    execution = cann_split_execution_kernel(spec)
    if (has_gpu_split_gemm_abi(spec) or has_cann_single_core_gemm_source(execution)
            or has_cann_deterministic_gemm_source(execution)):
        return None, "gemm_native_split_workspace_source_ports_unavailable"
    if has_public_tensor_gemm_source(spec):
        return None, "gemm_opaque_backend_operand_mapping_unavailable"
    if has_cann_gemm_source(execution) and execution.shape != spec.shape:
        return None, "gemm_captured_physical_operand_mapping_unavailable"
    if has_cann_gemm_source(execution):
        # CANN exposes the same logical matrices, but does not identify the
        # backend ND/NZ format. Keep actual parents and advertise only logical
        # operand availability, never a native address/format mapping.
        roles = (
            replace(a, view_id="A", shape=(1, m, k), strides_bytes=(m * a.strides_bytes[0], *a.strides_bytes)),
            replace(b, view_id="B"),
            replace(c, view_id="C", shape=(1, m, n), strides_bytes=(m * c.strides_bytes[0], *c.strides_bytes)),
        )
        allocations, events, ready, visibility = source_memory_events(call, source,
            tuple(zip((a, b), source.input_producers, strict=True)))
        events = [replace(event, view_id=None) if event.view_id is not None else event for event in events]
        logical_ready = []
        for view, ref in zip(roles[:2], ready, strict=True):
            event_id = call.call_id + ":logical-native:" + view.view_id
            events.append(ProgramEvent(event_id, EventKind.LOGICAL_AVAILABLE, view.allocation_id,
                Unmapped(view.shape, view.dtype, view.element_bytes, event_id), visibility, (ref,), view.view_id))
            logical_ready.append(EventRef(event_id, EventKind.LOGICAL_AVAILABLE))
        return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations, roles,
            tuple(events), ("A", "B"), ("C",), tuple(logical_ready), call.rank, "source-forward", visibility), None
    work = GEMMWork(m, n, k, spec.dtype, a.element_bytes, trans_b=True)
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(source.inputs, source.input_producers, strict=True)))
    binding = MemoryBindings(work, call.call_id + ":source-memory", call.call_id, allocations,
        tuple(dict((view.view_id, view) for view in (a, b, c)).values()), events,
        a.view_id, b.view_id, c.view_id, *ready, call.rank, "source-forward", visibility)
    return binding, None


def source_memory_events(call, source, operands):
    """Project caller-ready facts without extending the source storage lifetime."""
    binding_id = call.call_id + ":source-memory"
    visibility = f"{call.pool_name}:rank:{call.rank}"
    # The receipt keeps original lifetimes. A single kernel cannot free its
    # caller's storage; its subprogram sees the same actual live backing.
    allocations = tuple(item.with_lifetime(item.allocation_id, item.allocated_event, None)
        for item in source.allocations)
    events = [ProgramEvent(item.allocated_event, EventKind.ALLOCATED,
        item.allocation_id, None, None, ()) for item in allocations]
    barriers = []
    for index, predecessor in enumerate(source.predecessors):
        event_id = f"{binding_id}:predecessor:{index}:{predecessor}"
        events.append(ProgramEvent(event_id, EventKind.BARRIER, None, None, None, ()))
        barriers.append(EventRef(event_id, EventKind.BARRIER))
    ready = []
    for index, (view, producer) in enumerate(operands):
        allocated = next(item.allocated_event for item in allocations if item.allocation_id == view.allocation_id)
        dependencies = (EventRef(allocated, EventKind.ALLOCATED), *barriers)
        if producer is not None:
            done = f"{binding_id}:input:{index}:source-done:{producer}"
            events.append(ProgramEvent(done, EventKind.COMPUTE_DONE, None, None, None, ()))
            dependencies += (EventRef(done, EventKind.COMPUTE_DONE),)
        initialized = f"{binding_id}:input:{index}:available"
        events.append(ProgramEvent(initialized, EventKind.INITIALIZED,
            view.allocation_id, view.region(), visibility, dependencies, view.view_id))
        ready.append(EventRef(initialized, EventKind.INITIALIZED))
    return allocations, tuple(events), tuple(ready), visibility


def _bind_elementwise(call, source):
    spec = call.kernel_spec
    if spec.variant in ("rmsnorm", "add_rmsnorm"):
        return _bind_normalization(call, source)
    if spec.variant in ("activation", "silu"):
        if source.kind is not WorkloadKind.ACTIVATION or source.variant != "silu":
            return None, "activation_source_operation_mapping_unavailable"
        if len(source.inputs) != 1 or len(source.outputs) != 1 or "tok" not in spec.shape:
            return None, "activation_source_packed_operand_unavailable"
        packed, = source.inputs
        output, = source.outputs
        rows, width = spec.shape["tok"], spec.shape["intermediate"]
        if (packed.shape != (rows, 2 * width) or output.shape != (rows, width)
                or packed.dtype != spec.dtype or output.dtype != spec.dtype):
            return None, "activation_source_operand_geometry_unavailable"
        # The source SiluAndMul reads two columns of the same packed tensor.
        # These role-local views alias that exact backing; no storage is made.
        left = replace(packed, view_id=call.call_id + ":gate", shape=(rows, width))
        right = replace(packed, view_id=call.call_id + ":up", shape=(rows, width),
            offset_bytes=packed.offset_bytes + width * packed.strides_bytes[-1])
        reads, writes = (left, right), (output,)
        producers = source.input_producers * 2
    elif spec.variant in ("elementwise", "elementwise_pytorch_vec4"):
        if source.kind is not WorkloadKind.ELEMENTWISE:
            return None, "elementwise_source_operation_mapping_unavailable"
        arity = {"add": (2, 1), "mul": (2, 1), "unary": (1, 1), "copy": (1, 1),
            "fill": (0, 1), "read": (1, 0)}
        operation = spec.attributes["operation"]
        if operation not in arity or (len(source.inputs), len(source.outputs)) != arity[operation]:
            return None, "elementwise_source_operand_arity_unavailable"
        reads, writes, producers = source.inputs, source.outputs, source.input_producers
        ports = (*reads, *writes)
        if (any(prod(view.shape) != spec.shape["element_count"] or view.dtype != spec.dtype for view in ports)
                or len({view.shape for view in ports}) != 1
                or "input_materialization" in spec.attributes):
            return None, "elementwise_source_operand_geometry_unavailable"
    else:
        return None, "source_family_operand_mapping_unavailable"
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, producers, strict=True)))
    views = tuple({view.view_id: view for view in (*reads, *writes)}.values())
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id,
        allocations, views, events, tuple(view.view_id for view in reads),
        tuple(view.view_id for view in writes), ready, call.rank, "source-forward", visibility), None


def _bind_normalization(call, source):
    spec = call.kernel_spec
    fused = spec.variant == "add_rmsnorm"
    if source.kind is not WorkloadKind.NORM or source.variant != spec.variant:
        return None, "normalization_source_operation_mapping_unavailable"
    if (len(source.inputs), len(source.outputs)) != ((3, 2) if fused else (2, 1)):
        return None, "normalization_source_operand_arity_unavailable"
    names = ("input", "residual", "weight") if fused else ("input", "weight")
    reads = tuple(replace(view, view_id=name) for name, view in zip(names, source.inputs, strict=True))
    if fused:
        if source.outputs != source.inputs[:2]:
            return None, "normalization_source_inplace_mapping_unavailable"
        writes = reads[:2]
    else:
        writes = (replace(source.outputs[0], view_id="output"),)
    rows, hidden, width = (spec.shape[name] for name in ("rows", "hidden", "dtype_bytes"))
    for view in (*reads, *writes):
        shape = (hidden,) if view.view_id == "weight" else (rows, hidden)
        strides = (width,) if view.view_id == "weight" else (hidden * width, width)
        if (view.shape, view.strides_bytes, view.dtype, view.element_bytes) != (shape, strides, spec.dtype, width):
            return None, "normalization_source_operand_layout_unavailable"
    if len({view.allocation_id for view in reads}) != len(reads):
        return None, "normalization_source_input_alias_unavailable"
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, source.input_producers, strict=True)))
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        tuple({view.view_id: view for view in (*reads, *writes)}.values()), events,
        names, tuple(view.view_id for view in writes), ready, call.rank, "source-forward", visibility), None


def _dense_role_view(view, *, numeric):
    stride = view.element_bytes
    for dimension, actual in zip(reversed(view.shape), reversed(view.strides_bytes), strict=True):
        if dimension != 1 and actual != stride:
            return None
        stride *= dimension
    if numeric:
        return replace(view, view_id=view.view_id + ":scalar-role", shape=(prod(view.shape),),
            strides_bytes=(view.element_bytes,))
    return replace(view, view_id=view.view_id + ":byte-role", shape=(prod(view.shape) * view.element_bytes,),
        strides_bytes=(1,), dtype="uint8", element_bytes=1)


def _bind_communication(call, source):
    from src.kernel.memory_communication import communication_operand_layouts, has_communication_source
    spec = call.kernel_spec
    if (source.kind not in (WorkloadKind.COLLECTIVE, WorkloadKind.P2P, WorkloadKind.AF_ACTIVATION_TRANSFER)
            or not has_communication_source(spec)):
        return None, "communication_source_operation_mapping_unavailable"
    input_layouts, output_layouts = communication_operand_layouts(spec)
    if (len(source.inputs), len(source.outputs)) != (len(input_layouts), len(output_layouts)):
        return None, "communication_source_participant_ports_unavailable"
    numeric = spec.transfer.operation == "all_reduce"
    reads = tuple(_dense_role_view(view, numeric=numeric) for view in source.inputs)
    writes = tuple(_dense_role_view(view, numeric=numeric) for view in source.outputs)
    if any(view is None for view in (*reads, *writes)):
        return None, "communication_source_contiguous_payload_unavailable"
    if any(prod(view.shape) * view.element_bytes != size for view, (_, size)
            in zip((*reads, *writes), (*input_layouts, *output_layouts), strict=True)):
        return None, "communication_source_payload_extent_unavailable"
    allocations, events, ready, visibility = source_memory_events(call, source,
        tuple(zip(reads, source.input_producers, strict=True)))
    return ElementwiseBindings(call.call_id + ":source-memory", call.call_id, allocations,
        tuple({view.view_id: view for view in (*reads, *writes)}.values()), events,
        tuple(view.view_id for view in reads), tuple(view.view_id for view in writes), ready,
        call.rank, "source-forward", visibility), None


def bind_trace_memory(plan, context):
    from .kernel_plan import KernelPlan
    if type(context) is not TraceMemoryContext:
        raise TypeError("source memory binding requires a current trace receipt")
    return KernelPlan(tuple(context.bind_call(call) for call in plan.calls),
        plan.block_repeats, memory_context=context)
