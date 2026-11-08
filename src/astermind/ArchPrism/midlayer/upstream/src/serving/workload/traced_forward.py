"""CPU symbolic eager forward tracing into the existing workload/storage contracts.

No tensor payloads or weights are allocated. Python references retain symbolic
storage exactly as the described forward does; graph records retain IDs only.
This models synchronous CPython eager scopes, not compiler liveness, asynchronous
GPU retirement, allocator reservation, or undocumented library workspace.
"""

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from collections import OrderedDict
from copy import copy
from functools import lru_cache
import sys
import weakref
import operator
from math import prod, isfinite

from src.kernel.kernel_spec import DTYPE_BYTES
from src.kernel.memory_access import AddressLevel, MemoryAllocation, TensorView, U64_MAX, region_end
from src.serving.deployment.deployment_config import (
    CollectiveConfig, PipelineP2PConfig, worker_role_operator_group, worker_role_phase,
)
from .workload_graph import WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode, _workload_node_field_updates, _workload_node_field_updates_read, _node_update_source_unchanged


def _frame_rebinding(value, prefix, view_counter, bound_names):
    """Compile binding leaves without a recursive closure retaining its frame."""
    if isinstance(value, str):
        if (value in bound_names or value == prefix or value.startswith(prefix + ":")
                or value == view_counter or value.startswith(view_counter + ":")):
            return lambda rename: rename(value)
        return None
    if isinstance(value, (tuple, list, Mapping)):
        items = value.items() if isinstance(value, Mapping) else enumerate(value)
        changes = tuple((key, program) for key, item in items
            if (program := _frame_rebinding(item, prefix, view_counter, bound_names)) is not None)
        if not changes:
            return None

        def apply(rename):
            result = dict(value) if isinstance(value, Mapping) else list(value)
            for key, program in changes:
                result[key] = program(rename)
            return tuple(result) if isinstance(value, tuple) else result

        return apply
    return None


class _ForwardModuleFrame:
    """Detached metadata program recorded from one actual source invocation."""

    def __init__(self, trace, prefix, arguments, result, before):
        self.supported = False
        if (not isinstance(result, tuple) or not result
                or any(not isinstance(value, ForwardTensor) for value in result)
                or not before["persistent"] <= trace._persistent_ids
                or before["checkpoints"] != trace._checkpoints
                or before["workspaces"] != trace._workspace_capacities
                or before["captures"] != len(trace._captures)
                or before["collectives"] != trace._collective_ids):
            return
        self.persistent = frozenset(trace._persistent_ids - before["persistent"])
        self.prefix = prefix
        self.arguments = tuple(None if value is None else value.view for value in arguments)
        self.argument_writers = tuple(None if value is None else before["writers"][value.view.allocation_id]
            for value in arguments)
        self.outputs = tuple((value.view.view_id,
            next((index for index, argument in enumerate(arguments) if value is argument), None))
            for value in result)
        self.nodes = tuple(trace._nodes.values())[before["nodes"]:]
        self.edges = tuple(trace._edges[before["edges"]:])
        self.views = tuple(value for name, value in trace._views.items() if name not in before["views"])
        self.allocations = tuple(value for name, value in trace._allocations.items() if name not in before["allocations"])
        self.events = tuple(trace._events[before["events"]:])
        if any(event.split(":", 2)[1] not in ("allocate", "return", "free") for event in self.events):
            return
        self.view_counter = trace.qualify("view")
        bound_names = {name for view in self.arguments if view is not None
            for name in (view.view_id, view.allocation_id)}
        bound_names.update(writer for writer in self.argument_writers if writer is not None)

        self.node_rebindings = tuple((node, tuple((name, program) for name, value in node.attributes.items()
            if (program := _frame_rebinding(value, prefix, self.view_counter, bound_names)) is not None))
            for node in self.nodes)
        self.names = tuple((name, before["names"][name] if name in before["names"] else None, value)
            for name, value in trace._names.items() if name not in before["names"] or value != before["names"][name])
        ports = set()
        written = set()
        for node in self.nodes:
            for field in ("symbolic_tensor_inputs", "symbolic_tensor_outputs", "symbolic_tensor_controls", "symbolic_parameter_inputs"):
                if field in node.attributes:
                    ports.update(node.attributes[field])
            written.update(trace._views[name].allocation_id for name in node.attributes["symbolic_tensor_outputs"])
        touched = {trace._views[name].allocation_id for name in ports}
        touched.update(view.allocation_id for view in self.views)
        touched.update(allocation.allocation_id for allocation in self.allocations)
        self.external_views = tuple(before["views"][name] for name in ports if name in before["views"])
        self.external = {name: (before["allocations"][name], before["references"][name],
            before["writers"][name], before["readers"][name])
            for name in touched if name in before["allocations"]}
        self.writers = {name: trace._writers[name] for name in touched}
        self.readers = {name: set(trace._readers[name]) for name in touched}
        self.written = written
        output_views = {name for name, argument in self.outputs if argument is None}
        counts = {}
        for name in output_views:
            storage = trace._views[name].allocation_id
            if storage not in counts:
                counts[storage] = 0
            counts[storage] += 1
        for name in touched:
            expected = counts[name] if name in counts else 0
            if name in self.external:
                expected += before["references"][name]
                if name not in written and not before["readers"][name] <= trace._readers[name]:
                    return
            if trace._references[name] != expected:
                return
        self.supported = True

    def replay(self, trace, prefix, arguments):
        substitutions = {}
        for old, old_writer, current in zip(self.arguments, self.argument_writers, arguments, strict=True):
            if old is None:
                continue
            pairs = [(old.view_id, current.view.view_id), (old.allocation_id, current.view.allocation_id)]
            if old_writer is not None:
                writer = trace._writers[current.view.allocation_id]
                if writer is None:
                    return None
                pairs.append((old_writer, writer))
            for source, target in pairs:
                if source in substitutions and substitutions[source] != target:
                    return None
                substitutions[source] = target
        old_view_start, new_view_start = None, None
        for name, initial, _ in self.names:
            if name == self.view_counter:
                old_view_start = -1 if initial is None else initial
                new_view_start = trace._names[name] if name in trace._names else -1

        def rename(value):
            if value in substitutions:
                return substitutions[value]
            if value == self.prefix or value.startswith(self.prefix + ":"):
                result = prefix + value[len(self.prefix):]
            elif old_view_start is not None and (value == self.view_counter or value.startswith(self.view_counter + ":")):
                index = 0 if value == self.view_counter else int(value[len(self.view_counter) + 1:])
                if index > old_view_start:
                    index += new_view_start - old_view_start
                    result = self.view_counter if index == 0 else self.view_counter + ":" + str(index)
                else:
                    result = value
            else:
                result = value
            substitutions[value] = result
            return result

        def geometry(view):
            return view.offset_bytes, view.shape, view.strides_bytes, view.dtype, view.element_bytes

        for old in self.external_views:
            name = rename(old.view_id)
            if name not in trace._views or geometry(trace._views[name]) != geometry(old):
                return None
            if trace._views[name].allocation_id != rename(old.allocation_id):
                return None
        for name, (allocation, references, writer, readers) in self.external.items():
            target = rename(name)
            if target not in trace._allocations:
                return None
            actual = trace._allocations[target]
            if (actual.size_bytes != allocation.size_bytes or actual.memory_domain != allocation.memory_domain
                    or actual.device_id != allocation.device_id or trace._references[target] != references
                    or trace._writers[target] != (None if writer is None else rename(writer))):
                return None
            if name in self.written and trace._readers[target] != {rename(reader) for reader in readers}:
                return None
        names = {}
        for name, initial, final in self.names:
            target = name if name == self.view_counter else rename(name)
            actual = trace._names[target] if target in trace._names else None
            if name == self.view_counter:
                names[target] = (-1 if actual is None else actual) + final - (-1 if initial is None else initial)
            else:
                if actual != initial:
                    return None
                names[target] = final
        if (any(rename(node.node_id) in trace._nodes for node in self.nodes)
                or any(rename(view.view_id) in trace._views for view in self.views)
                or any(rename(a.allocation_id) in trace._allocations for a in self.allocations)):
            return None
        events = {}
        for index, event in enumerate(self.events, start=len(trace._events)):
            _, kind, payload = event.split(":", 2)
            events[event] = f"{index}:{kind}:{rename(payload)}"

        trace._events.extend(events[event] for event in self.events)
        trace._names.update(names)
        for allocation in self.allocations:
            target = rename(allocation.allocation_id)
            trace._allocations[target] = allocation.with_lifetime(target, events[allocation.allocated_event],
                None if allocation.freed_event is None else events[allocation.freed_event])
            trace._references[target] = 0
        trace._persistent_ids.update(rename(name) for name in self.persistent)
        for view in self.views:
            target = rename(view.view_id)
            trace._views[target] = view.with_identity(target, rename(view.allocation_id))
        for node, programs in self.node_rebindings:
            target = rename(node.node_id)
            attributes = dict(node.attributes)
            for name, program in programs:
                attributes[name] = program(rename)
            trace._nodes[target] = node.with_attributes(attributes, node_id=target)
        trace._edges.extend(WorkloadEdge(rename(edge.source), rename(edge.destination), edge.kind) for edge in self.edges)
        for name, writer in self.writers.items():
            trace._writers[rename(name)] = None if writer is None else rename(writer)
        for name, readers in self.readers.items():
            target = rename(name)
            if name in self.external and name not in self.written:
                trace._readers[target].update(rename(reader) for reader in readers - self.external[name][3])
            else:
                trace._readers[target] = {rename(reader) for reader in readers}
        handles = {}
        result = []
        for name, argument in self.outputs:
            if argument is not None:
                result.append(arguments[argument])
            else:
                target = rename(name)
                if target not in handles:
                    handles[target] = trace._handle(trace._views[target])
                result.append(handles[target])
        return tuple(result)


@lru_cache(maxsize=4096)
def _source_allocation_template(device_id, memory_domain, size_bytes):
    """Reuse validated source storage geometry; each call binds fresh IDs/events."""
    return MemoryAllocation("source-allocation-layout", device_id, memory_domain,
        size_bytes, AddressLevel.OBJECT, None, None, "source-allocation-layout", None)


@lru_cache(maxsize=2048)
def _dense_layout(shape, dtype):
    """Reuse a fully validated immutable layout, without retaining a trace.

    The caller checks concrete dimension types and dtype before lookup.
    New views validate their identities and reuse only validated geometry.
    """
    stride = DTYPE_BYTES[dtype]
    strides = []
    for dimension in reversed(shape):
        strides.append(stride)
        stride *= dimension
    return TensorView("layout", "layout", 0, shape, tuple(reversed(strides)), dtype, DTYPE_BYTES[dtype])


@lru_cache(maxsize=4096)
def _strided_layout(shape, strides_bytes, dtype, offset_bytes):
    return TensorView("layout", "layout", offset_bytes, shape, strides_bytes, dtype, DTYPE_BYTES[dtype])


def _validate_layout_input(name, shape, dtype):
    if (not isinstance(name, str) or not name or not isinstance(shape, tuple)
            or not shape or any(type(d) is not int or d <= 0 for d in shape)):
        raise ValueError("symbolic tensors require a name and concrete positive dimensions")
    if dtype not in DTYPE_BYTES:
        raise NotImplementedError("symbolic forward requires an unpacked canonical dtype")
    if type(shape) is not tuple:
        raise TypeError("contract sequences must be immutable tuples")
    if type(dtype) is not str:
        raise ValueError("expected nonempty string")


def _release_forward_tensor(trace_reference, storage):
    # finalize keeps its callback alive globally. A bound trace method would
    # pin captured tensor handles and their entire trace after every step.
    trace = trace_reference()
    if trace is not None:
        trace._release(storage)


@dataclass(frozen=True)
class ForwardTensor:
    """A reference to a canonical view; Python aliases keep this handle alive."""

    trace: "ForwardTrace"
    view: TensorView

    @property
    def shape(self):
        return self.view.shape

    @property
    def dtype(self):
        return self.view.dtype

    @property
    def is_contiguous(self):
        stride = self.view.element_bytes
        for size, actual in zip(reversed(self.shape), reversed(self.view.strides_bytes), strict=True):
            if size != 1 and actual != stride:
                return False
            stride *= size
        return True

    def split(self, sizes, dim=-1):
        if dim not in (-1, len(self.shape) - 1) or sum(sizes) != self.shape[-1]:
            raise NotImplementedError("symbolic split requires a complete last-axis partition")
        offset = self.view.offset_bytes
        outputs = []
        for size in sizes:
            if type(size) is not int or size <= 0:
                raise ValueError("symbolic split sizes must be positive integers")
            outputs.append(self.trace.alias(self, self.trace.unique_name("view"),
                shape=(*self.shape[:-1], size), strides_bytes=self.view.strides_bytes,
                offset_bytes=offset))
            offset += size * self.view.strides_bytes[-1]
        return tuple(outputs)

    def narrow(self, dim, start, length):
        """A source slice retaining its full backing, including head padding."""
        if type(dim) is not int or not -len(self.shape) <= dim < len(self.shape):
            raise ValueError("symbolic narrow dimension must identify an existing axis")
        dim %= len(self.shape)
        if (type(start) is not int or type(length) is not int or start < 0 or length <= 0
                or start + length > self.shape[dim]):
            raise ValueError("symbolic narrow requires a nonempty in-bounds slice")
        shape = tuple(length if axis == dim else size for axis, size in enumerate(self.shape))
        return self.trace.alias(self, self.trace.unique_name("view"), shape=shape,
            strides_bytes=self.view.strides_bytes,
            offset_bytes=self.view.offset_bytes + start * self.view.strides_bytes[dim])

    def transpose(self, dim0, dim1):
        """Swap two metadata axes without copying or releasing the backing."""
        ndim = len(self.shape)
        if any(type(dim) is not int or not -ndim <= dim < ndim for dim in (dim0, dim1)):
            raise ValueError("symbolic transpose dimensions must identify existing axes")
        dim0, dim1 = dim0 % ndim, dim1 % ndim
        axes = list(range(ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.trace.alias(self, self.trace.unique_name("view"),
            shape=tuple(self.shape[axis] for axis in axes),
            strides_bytes=tuple(self.view.strides_bytes[axis] for axis in axes),
            offset_bytes=self.view.offset_bytes)

    def unsqueeze(self, dim):
        """Insert a singleton view axis, preserving all original strides."""
        axes = len(self.shape) + 1
        if type(dim) is not int or not -axes <= dim < axes:
            raise ValueError("symbolic unsqueeze dimension must identify an insertion axis")
        dim %= axes
        stride = self.view.element_bytes if dim == len(self.shape) else self.shape[dim] * self.view.strides_bytes[dim]
        shape = (*self.shape[:dim], 1, *self.shape[dim:])
        strides = (*self.view.strides_bytes[:dim], stride, *self.view.strides_bytes[dim:])
        return self.trace.alias(self, self.trace.unique_name("view"), shape=shape,
            strides_bytes=strides, offset_bytes=self.view.offset_bytes)

    def expand(self, *shape):
        """Broadcast singleton axes as zero-stride views, without allocating."""
        if len(shape) != len(self.shape):
            raise NotImplementedError("symbolic expand requires explicit existing axes")
        dimensions, strides = [], []
        for original, requested, stride in zip(self.shape, shape, self.view.strides_bytes, strict=True):
            if requested == -1:
                requested = original
            if type(requested) is not int or requested <= 0 or (original != requested and original != 1):
                raise ValueError("symbolic expand only broadcasts singleton dimensions")
            dimensions.append(requested)
            strides.append(0 if original != requested else stride)
        return self.trace.alias(self, self.trace.unique_name("view"), shape=tuple(dimensions),
            strides_bytes=tuple(strides), offset_bytes=self.view.offset_bytes)

    def view_as_shape(self, *shape):
        """Dense reshape, or split/merge contiguous trailing axes of row views."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if -1 in shape:
            if shape.count(-1) != 1 or any(type(d) is not int or d == 0 or d < -1 for d in shape):
                raise ValueError("symbolic view accepts one inferred dimension and positive explicit dimensions")
            known = prod(d for d in shape if d != -1)
            if prod(self.shape) % known:
                raise ValueError("symbolic inferred view must preserve the number of elements")
            shape = tuple(prod(self.shape) // known if d == -1 else d for d in shape)
        if not shape or any(type(d) is not int or d <= 0 for d in shape) or prod(shape) != prod(self.shape):
            raise ValueError("symbolic view must preserve the number of elements")
        dense = self.trace._dense_view("layout", self.shape, self.dtype)
        result = self.trace._dense_view("layout", tuple(shape), self.dtype)
        strides = result.strides_bytes
        if not self.is_contiguous:
            if (len(self.shape) < 2 or len(shape) < 2 or shape[0] != self.shape[0]
                    or self.view.strides_bytes[1:] != dense.strides_bytes[1:]):
                raise NotImplementedError("symbolic view cannot merge noncontiguous storage chunks")
            strides = (self.view.strides_bytes[0], *strides[1:])
        return self.trace.alias(self, self.trace.unique_name("view"), shape=tuple(shape),
            strides_bytes=strides, offset_bytes=self.view.offset_bytes)


    def dim(self):
        return len(self.shape)


class ForwardTrace:
    """Record one rank's declared synchronous forward without real tensors.

    Operators supply existing WorkloadNodes and their source output layouts.
    Data and mutation dependencies, allocation/free events and alias ownership
    are inferred. Caller-local references remain live until reassigned, deleted
    or their function returns. Views retain their backing allocation. Internal
    workspace must be explicitly described by the operator's source program.
    """

    def __init__(self, *, rank: int = 0, namespace: str | None = None):
        if sys.implementation.name != "cpython":
            raise NotImplementedError("eager reference lifetimes require CPython")
        if type(rank) is not int or rank < 0:
            raise ValueError("forward trace requires a nonnegative physical rank")
        if namespace is not None and (not isinstance(namespace, str) or not namespace):
            raise ValueError("forward namespace must be nonempty when declared")
        self.rank = rank
        self.namespace = namespace
        self._allocations = {}
        self._references = {}
        self._views = {}
        self._writers = {}
        self._readers = {}
        self._events = []
        self._nodes = {}
        self._edges = []
        self._names = {}
        self._persistent_ids = set()
        self._checkpoints = {}
        self._collective_ids = set()
        self._workspace_capacities = {}
        self._capture = None
        self._captures = []
        self._module_frames = {}
        self._module_program_count = 0
        self._module_program_types = {}
        self._recorded_graph = None

    def _module_program(self, module_type):
        """Name one homogeneous constructor scope without retaining its model."""
        self._module_program_count += 1
        self._module_program_types[self._module_program_count] = module_type
        return self._module_program_count

    def _call_module_frame(self, program, prefix, module, arguments):
        """Replay a recorded metadata frame within one homogeneous source loop."""
        if type(program) is not int or program not in self._module_program_types:
            raise ValueError("module frame requires its registered constructor scope")
        if type(module) is not self._module_program_types[program]:
            raise ValueError("module frame type differs from its homogeneous constructor scope")
        if not isinstance(arguments, tuple):
            raise TypeError("module frame arguments must be a tuple")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("module frame needs its nonempty source prefix")
        tensors = tuple(value for value in arguments if value is not None)
        for tensor in tensors:
            self._check_tensor(tensor)
        layouts = tuple(None if value is None else (
            value.view.offset_bytes, value.shape, value.view.strides_bytes, value.dtype,
            self._references[value.view.allocation_id]) for value in arguments)
        aliases = tuple(left.view.allocation_id == right.view.allocation_id
            for left in tensors for right in tensors)
        writers = tuple(self._writers[value.view.allocation_id] for value in tensors)
        writer_aliases = tuple(left == right for left in writers for right in writers)
        key = (program, type(module).__call__, layouts, aliases, tuple(value is None for value in writers), writer_aliases)
        if key in self._module_frames:
            frame = self._module_frames[key]
            if frame is not None:
                replay = frame.replay(self, self.qualify(prefix), arguments)
                if replay is not None:
                    return replay
            self._module_frames[key] = None
            return module(*arguments)
        before = {
            "views": dict(self._views), "allocations": dict(self._allocations),
            "references": dict(self._references), "writers": dict(self._writers),
            "readers": {name: set(readers) for name, readers in self._readers.items()},
            "names": dict(self._names), "events": len(self._events),
            "nodes": len(self._nodes), "edges": len(self._edges),
            "persistent": set(self._persistent_ids), "checkpoints": dict(self._checkpoints),
            "workspaces": dict(self._workspace_capacities), "captures": len(self._captures),
            "collectives": set(self._collective_ids),
        }
        result = module(*arguments)
        frame = _ForwardModuleFrame(self, self.qualify(prefix), arguments, result, before)
        self._module_frames[key] = frame if frame.supported else None
        return result

    def bind_launch_policy(self, policy):
        """Bind kernel selection while preserving transport-owned contracts."""
        if policy not in ("analytical_launch", "injected_launch"):
            raise ValueError("virtual source requires an explicit supported launch policy")
        if (type(self._nodes) is _SourceFrameNodes and self._nodes.launch_policy == policy
                and all("launch_policy" not in attrs for _, attrs in self._nodes.updates.values())):
            return
        kernel_kinds = {WorkloadKind.LINEAR, WorkloadKind.LM_HEAD, WorkloadKind.ROUTER,
            WorkloadKind.FLASH_ATTN, WorkloadKind.EAGER_ATTN, WorkloadKind.FUSED_MOE,
            WorkloadKind.NORM, WorkloadKind.ACTIVATION, WorkloadKind.ROTARY,
            WorkloadKind.ELEMENTWISE, WorkloadKind.REDUCE, WorkloadKind.GATHER,
            WorkloadKind.VECTOR, WorkloadKind.KV_UPDATE}
        transport_kinds = {WorkloadKind.COLLECTIVE, WorkloadKind.P2P, WorkloadKind.PD_KV_TRANSFER,
            WorkloadKind.AF_ACTIVATION_TRANSFER, WorkloadKind.MEMCPY, WorkloadKind.KV_OFFLOAD}
        bound = {}
        for identifier, node in self._nodes.items():
            if node.kind in kernel_kinds:
                bound[identifier] = (node if "launch_policy" in node.attributes
                    and node.attributes["launch_policy"] == policy else
                    node.with_attributes({**node.attributes, "launch_policy": policy}))
            elif node.kind in transport_kinds:
                bound[identifier] = node
            else:
                raise ValueError(f"launch policy has no contract for workload kind {node.kind!r}")
        if any(bound[name] is not node for name, node in self._nodes.items()):
            self._nodes = bound

    def begin_capture(self, *, key, logical_tokens, model_input_tokens, inputs, runtime_mode="FULL", eager_kinds=(),
            padding_program="vllm_masked_minus1", source_stage=None, extra_inputs=None,
            source_graph_batch_size=None):
        """Bind static source ports while tracing the actual padded forward.

        This records a source graph's object identities, not CUDA pool
        reservation or a compiled allocation/replay observation.
        """
        pipeline = source_stage == "vllm_pipeline"
        af = source_stage == "midlayer_af"
        local_stage = pipeline or af
        pipeline_ports = ({"input_ids", "positions", "slot_mapping"},
            {"hidden_states", "residual", "positions", "slot_mapping"})
        stage_ports = (*pipeline_ports, {"hidden_states", "residual"}) if af else pipeline_ports
        if (self._capture is not None or not isinstance(key, str) or not key
                or type(logical_tokens) is not int or type(model_input_tokens) is not int
                or not 0 < logical_tokens <= model_input_tokens
                or (set(inputs) not in stage_ports if local_stage else set(inputs) != pipeline_ports[0])):
            raise ValueError("capture requires one explicit bucket and complete static source ports")
        if (runtime_mode not in ("FULL", "PIECEWISE") or not isinstance(eager_kinds, tuple)
                or len(eager_kinds) != len(set(eager_kinds))
                or (runtime_mode == "FULL" and eager_kinds)
                or (runtime_mode == "PIECEWISE" and set(eager_kinds) != {"flash_attn", "kv_update"})):
            raise ValueError("piecewise source capture requires explicit eager boundaries")
        if padding_program not in ("vllm_masked_minus1", "sglang_zero_slot0_seq1"):
            raise ValueError("source capture requires a declared padding program")
        if padding_program == "sglang_zero_slot0_seq1" and (runtime_mode != "FULL" or logical_tokens >= model_input_tokens):
            raise ValueError("SGLang zero-slot padding requires an underfilled FULL capture")
        if source_stage not in (None, "vllm_pipeline", "midlayer_af", "sglang_eagle_draft_decode", "sglang_eagle_target_verify", "sglang_eagle_draft_extend"):
            raise ValueError("source capture stage must name a supported source boundary")
        if source_stage is not None and not local_stage and (runtime_mode != "FULL"
                or (logical_tokens != model_input_tokens and padding_program != "sglang_zero_slot0_seq1")):
            raise NotImplementedError("EAGLE source stage padding requires its explicit SGLang FULL program")
        if source_graph_batch_size is not None and (source_stage is None
                or type(source_graph_batch_size) is not int or source_graph_batch_size <= 0):
            raise ValueError("source graph request-batch identity requires a positive staged declaration")
        if extra_inputs is not None and (source_stage is None or not isinstance(extra_inputs, dict)
                or any(not isinstance(role, str) or not role or role in inputs for role in extra_inputs)):
            raise ValueError("additional capture inputs require distinct named staged source ports")
        if local_stage and (extra_inputs is not None or source_graph_batch_size is not None):
            raise ValueError("pipeline capture takes only its original model stage ports")
        if source_stage is not None and not local_stage:
            batch = source_graph_batch_size
            if type(batch) is not int or batch <= 0:
                raise ValueError("EAGLE staged capture requires its explicit positive request batch")
            draft = source_stage == "sglang_eagle_draft_decode"
            if logical_tokens % (1 if draft else 3):
                raise ValueError("EAGLE stage logical rows must contain complete request windows")
            if model_input_tokens != batch * (1 if draft else 3):
                raise ValueError("EAGLE stage physical rows must be draft B or verify/extend3B")
            extras = {} if extra_inputs is None else extra_inputs
            expected = ({"hidden_states", "topk_p", "topk_index"} if draft else
                {"hidden_states", "num_correct_drafts", "num_accept_tokens"}
                if source_stage == "sglang_eagle_draft_extend" else set())
            if set(extras) != expected:
                raise ValueError("EAGLE stage requires every declared hidden/topk/acceptance input port")
            for role, tensor in extras.items():
                self._check_tensor(tensor)
                if role == "hidden_states":
                    valid = tensor.dtype == "bf16" and len(tensor.shape) == 2 and tensor.shape[0] == model_input_tokens
                elif role in ("topk_p", "topk_index"):
                    valid = tensor.shape == (batch, 1) and tensor.dtype == ("fp32" if role == "topk_p" else "int64")
                else:
                    valid = tensor.shape == (batch,) and tensor.dtype == "int32"
                if not valid:
                    raise ValueError("EAGLE stage input dtype/shape must match its source role")
        if any(capture["key"] == key for capture in self._captures):
            raise ValueError("source capture keys must be distinct within a forward")
        ports = {}
        all_inputs = {**inputs, **({} if extra_inputs is None else extra_inputs)}
        for role, tensor in all_inputs.items():
            self._check_tensor(tensor)
            if local_stage and role in ("hidden_states", "residual"):
                if tensor.dtype != "bf16" or len(tensor.shape) != 2 or tensor.shape[0] != model_input_tokens:
                    raise ValueError("pipeline hidden/residual ports require physical BF16 rows")
            elif role in inputs and tensor.shape != (model_input_tokens,):
                raise ValueError("capture source port must contain its physical token extent")
            if local_stage and role not in ("hidden_states", "residual") and tensor.dtype != (
                    "int32" if role == "input_ids" else "int64"):
                raise ValueError("pipeline capture token/position/slot dtype must match its source port")
            self._persistent_ids.add(tensor.view.allocation_id)
            self._retain_capture_tensor(tensor)
            ports[role] = {"storage_id": f"rank:{self.rank}:{key}:{role}",
                "allocation_id": tensor.view.allocation_id, "shape": tensor.shape, "dtype": tensor.dtype,
                "bytes": self._allocations[tensor.view.allocation_id].size_bytes}
        self._capture = {"key": key, "logical_tokens": logical_tokens,
            "source_stage": source_stage,
            "source_graph_batch_size": source_graph_batch_size,
            "runtime_mode": runtime_mode, "eager_kinds": tuple(eager_kinds),
            "model_input_tokens": model_input_tokens, "inputs": ports, "output": None,
            "begin_event": self._event("capture.begin"), "end_event": None,
            "node_start": len(self._nodes), "node_ids": (), "allocator_reserved_bytes": None,
            "padding_ids": "unknown_retained_in_vocabulary_source_buffer_values",
            "padding_positions": "unknown_retained_in_range_source_buffer_values",
            "padding_slot_mapping": -1, "actual_provider_bound": False,
            "input_backing_capacity_bound": False,
            "port_identity_scope": "logical_rank_bucket_role_not_physical_backing_identity",
            "cross_bucket_backing_relationship": None,
            "cross_bucket_allocator_reserved_bytes": None,
            "static_input_scope": "captured_active_port_slices_not_complete_worker_buffer_capacity_graph_local_allocation_ids_not_cross_step_physical_backing"}
        if padding_program == "sglang_zero_slot0_seq1":
            self._capture.update(padding_ids="caller_declared_retained_source_values",
                padding_positions=0, padding_slot_mapping=0, padding_sequence_length=1,
                anonymous_dummy_kv_write_rows=model_input_tokens - logical_tokens,
                dummy_cache_disjointness="caller_declared_not_observed")
            if source_stage is not None:
                self._capture["padding_attention_kv_length"] = 4 if source_stage == "sglang_eagle_target_verify" else 1
                if source_stage == "sglang_eagle_draft_decode":
                    self._capture["padding_ids"] = 0

    def end_capture(self, output=None, *, outputs=None):
        if self._capture is None or self._capture["end_event"] is not None:
            raise ValueError("capture end requires one active source capture")
        if self._capture["source_stage"] is None:
            if outputs is not None:
                raise ValueError("ordinary capture retains its single-output source contract")
            self._check_tensor(output)
            named_outputs = {"hidden_states": output}
        else:
            if (output is not None or not isinstance(outputs, dict) or not outputs
                    or any(not isinstance(role, str) or not role for role in outputs)):
                raise ValueError("staged capture requires its complete named source output ports")
            pipeline = self._capture["source_stage"] in ("vllm_pipeline", "midlayer_af")
            draft = self._capture["source_stage"] == "sglang_eagle_draft_decode"
            expected = {"parent_list", "top_scores_index", "draft_tokens"} if draft else {"next_token_logits", "hidden_states"}
            if ((set(outputs) not in ({"hidden_states"}, {"hidden_states", "residual"})) if pipeline else set(outputs) != expected):
                raise ValueError("staged capture requires its complete named source output ports")
            for role, tensor in outputs.items():
                self._check_tensor(tensor)
                if pipeline:
                    valid = tensor.dtype == "bf16" and len(tensor.shape) == 2 and tensor.shape[0] == self._capture["model_input_tokens"]
                elif draft:
                    valid = tensor.dtype == "int64" and tensor.shape == (self._capture["source_graph_batch_size"], 2)
                else:
                    valid = (tensor.dtype == ("fp32" if role == "next_token_logits" else "bf16")
                        and len(tensor.shape) == 2 and tensor.shape[0] == self._capture["model_input_tokens"])
                if not valid:
                    raise ValueError("EAGLE stage output dtype/shape must match its complete source port")
            named_outputs = outputs
        ports = {}
        for role, tensor in named_outputs.items():
            self._check_tensor(tensor)
            self._persistent_ids.add(tensor.view.allocation_id)
            self._retain_capture_tensor(tensor)
            ports[role] = {"storage_id": f"rank:{self.rank}:{self._capture['key']}:{role}",
                "allocation_id": tensor.view.allocation_id, "shape": tensor.shape, "dtype": tensor.dtype,
                "bytes": self._allocations[tensor.view.allocation_id].size_bytes}
        if self._capture["source_stage"] is None:
            self._capture["output"] = ports["hidden_states"]
        else:
            self._capture["outputs"] = ports
        model_nodes = tuple(self._nodes)[self._capture["node_start"]:]
        eager = {name for name in model_nodes if self._nodes[name].kind.value in self._capture["eager_kinds"]
            or (self._capture["runtime_mode"] == "PIECEWISE" and self._nodes[name].kind is WorkloadKind.ELEMENTWISE
                and "source_attention_program" in self._nodes[name].attributes
                and self._nodes[name].attributes["source_attention_program"] == "ascend_0_18_piecewise_fia"
                and "source_component" in self._nodes[name].attributes
                and self._nodes[name].attributes["source_component"] == "symbolic_ascend_fia_output_copy")}
        segments, current = [], []
        for node in model_nodes:
            if node in eager:
                if current:
                    segments.append(tuple(current))
                    current = []
            else:
                current.append(node)
        if current:
            segments.append(tuple(current))
        self._capture["model_node_ids"] = model_nodes
        self._capture["eager_node_ids"] = tuple(node for node in model_nodes if node in eager)
        self._capture["node_ids"] = tuple(node for node in model_nodes if node not in eager)
        self._capture["source_segments"] = tuple(segments)
        self._capture["segment_scope"] = "source_op_split_regions_not_compiler_partitions_or_actual_replay_launches"
        self._capture["end_event"] = self._event("capture.end")
        self._captures.append(self._capture)
        self._capture = None

    @property
    def source_capture(self):
        if len(self._captures) > 1:
            raise ValueError("multi-stage source graphs require source_captures")
        return self._capture if self._capture is not None else (self._captures[0] if self._captures else None)

    @property
    def source_captures(self):
        if self._capture is not None:
            raise ValueError("source capture sequence cannot omit an unfinished active capture")
        return tuple(self._captures)

    @property
    def workspace_capacity_bytes(self):
        """Final declared arena sizes, without retaining managers or tensors."""
        return dict(self._workspace_capacities)

    def checkpoint(self, label):
        """Mark a source-program measurement boundary without tensor work."""
        if not isinstance(label, str) or not label or label in self._checkpoints:
            raise ValueError("forward checkpoint labels must be nonempty and unique")
        event = self._event(label)
        self._checkpoints[label] = event
        return event

    @property
    def checkpoints(self):
        return dict(self._checkpoints)

    def unique_name(self, prefix):
        """Stable module invocation/view IDs without per-model counters."""
        if self._recorded_graph is not None:
            raise ValueError("a bound source program is a closed forward trace")
        prefix = self.qualify(prefix)
        if prefix not in self._names:
            self._names[prefix] = 0
            return prefix
        self._names[prefix] += 1
        return f"{prefix}:{self._names[prefix]}"

    def qualify(self, name):
        if self.namespace is None or name.startswith(self.namespace + ":"):
            return name
        return self.namespace + ":" + name

    def _event(self, label):
        if self._recorded_graph is not None:
            raise ValueError("a bound source program is a closed forward trace")
        event = f"{len(self._events)}:{label}"
        self._events.append(event)
        return event

    def _handle(self, view):
        self._views[view.view_id] = view
        storage = view.allocation_id
        self._references[storage] += 1
        tensor = ForwardTensor(self, view)
        object.__setattr__(tensor, "_release_finalizer",
            weakref.finalize(tensor, _release_forward_tensor, weakref.ref(self), storage))
        return tensor

    def _retain_capture_tensor(self, tensor):
        # Capture owns this logical handle for the remaining trace lifetime.
        # Keep its reference count/lifetime unchanged when Python callers drop
        # it, without retaining tensor -> trace through a trace-owned list.
        # Detaching twice for an aliased static port is deliberately idempotent.
        tensor._release_finalizer.detach()

    def _release(self, storage):
        self._references[storage] -= 1
        if self._references[storage] == 0:
            allocation = self._allocations[storage]
            self._allocations[storage] = allocation.with_lifetime(storage,
                allocation.allocated_event, self._event(f"free:{storage}"))

    def _check_tensor(self, tensor):
        if not isinstance(tensor, ForwardTensor) or tensor.trace is not self:
            raise ValueError("forward tensor belongs to a different trace")

    @staticmethod
    def _dense_view(name, shape, dtype):
        # Tuple/str subclasses compare equal to canonical cache keys, but the
        # memory contract rejects them. Preserve that boundary before lookup.
        _validate_layout_input(name, shape, dtype)
        return _dense_layout(shape, dtype).with_identity(name, name)

    def tensor(self, name, shape, dtype, *, persistent=False, memory_domain="dram"):
        """Create independent activation storage, e.g. a source input or scratch."""
        if type(persistent) is not bool:
            raise TypeError("symbolic persistent flag must be boolean")
        name = self.qualify(name)
        view = self._dense_view(name, shape, dtype)
        return self._tensor_from_view(view, persistent=persistent, memory_domain=memory_domain)

    def parameter_view(self, name, shape, dtype):
        """Declare a readonly module parameter in this trace's storage owner.

        This is metadata only. The module source supplies its shape and dtype;
        no weight payload is loaded. Unlike activation handles, parameters stay
        live throughout the forward and never enter activation accounting.
        """
        name = self.qualify(name)
        layout = self._dense_view(name, shape, dtype)
        if name in self._views:
            existing = self._views[name]
            if (name not in self._persistent_ids or existing != layout
                    or self._writers[name] is not None):
                raise ValueError("source parameter identity or geometry changed")
            return existing
        if self._recorded_graph is not None:
            raise ValueError("cannot add parameters to a bound source trace")
        if name in self._allocations:
            raise ValueError("source parameter conflicts with an existing allocation")
        event = self._event(f"allocate:{name}")
        self._allocations[name] = _source_allocation_template(
            f"rank:{self.rank}", "dram", region_end(layout.region())).with_lifetime(name, event, None)
        self._views[name] = layout
        self._references[name] = 0
        self._writers[name] = None
        self._readers[name] = set()
        self._persistent_ids.add(name)
        return layout

    def _tensor_from_view(self, view, *, persistent=False, memory_domain="dram"):
        """Allocate from the exact view already validated by this call."""
        if memory_domain not in ("dram", "ddr", "buffer"):
            raise ValueError("forward allocation requires a declared dram, ddr or unbound on-chip buffer memory domain")
        name = view.view_id
        if name in self._allocations or name in self._views:
            raise ValueError("forward storage/view names must be unique")
        size_bytes = region_end(view.region())
        allocated_event = self._event(f"allocate:{name}")
        device_id = f"rank:{self.rank}"
        if type(memory_domain) is str and type(size_bytes) is int:
            self._allocations[name] = _source_allocation_template(device_id, memory_domain, size_bytes).with_lifetime(
                name, allocated_event, None)
        else:
            # Noncanonical values retain the full public constructor boundary.
            self._allocations[name] = MemoryAllocation(name, device_id, memory_domain,
                size_bytes, AddressLevel.OBJECT, None, None, allocated_event, None)
        self._references[name] = 0
        self._writers[name] = None
        self._readers[name] = set()
        if persistent:
            self._persistent_ids.add(name)
        return self._handle(view)

    def alias(self, tensor, name, *, shape, strides_bytes, offset_bytes=0, dtype=None):
        """Describe a typed view with an absolute offset into the same backing."""
        self._check_tensor(tensor)
        name = self.qualify(name)
        if not isinstance(name, str) or not name or name in self._views or name in self._allocations:
            raise ValueError("forward view names must be nonempty and unique")
        original = tensor.view
        dtype = original.dtype if dtype is None else dtype
        _validate_layout_input(name, shape, dtype)
        if type(strides_bytes) is not tuple:
            raise TypeError("contract sequences must be immutable tuples")
        if (type(offset_bytes) is not int or not 0 <= offset_bytes <= U64_MAX
                or any(type(stride) is not int or not 0 <= stride <= U64_MAX for stride in strides_bytes)):
            raise ValueError("expected unsigned 64-bit integer")
        width = DTYPE_BYTES[dtype]
        if offset_bytes % width or any(stride % width for stride in strides_bytes):
            raise ValueError("typed forward view offsets/strides must align to its element size")
        view = _strided_layout(shape, strides_bytes, dtype, offset_bytes).with_identity(name, original.allocation_id)
        if region_end(view.region()) > self._allocations[view.allocation_id].size_bytes:
            raise ValueError("forward view exceeds its backing storage")
        return self._handle(view)

    def _complete_producer_view(self, tensor):
        """Resolve a same-layout alias to its one complete producer port."""
        self._check_tensor(tensor)
        writer = self._writers[tensor.view.allocation_id]
        if writer is None:
            raise ValueError("transfer requires an explicit source tensor producer")
        ports = tuple(self._views[view_id] for view_id in self._nodes[writer].attributes["symbolic_tensor_outputs"]
            if self._views[view_id].allocation_id == tensor.view.allocation_id
            and self._views[view_id].shape == tensor.view.shape
            and self._views[view_id].strides_bytes == tensor.view.strides_bytes
            and self._views[view_id].offset_bytes == tensor.view.offset_bytes
            and self._views[view_id].dtype == tensor.view.dtype)
        if len(ports) != 1:
            raise ValueError("transfer must match exactly one complete producer region")
        return ports[0]

    def call(self, node: WorkloadNode, inputs: tuple[ForwardTensor, ...], *, shape, dtype, out=None, writes=(), controls=(), parameters=()):
        """Record an operator, allocating its output unless source declares out.

        Reading aliases sees the current storage writer. Reusing out also waits
        for earlier readers/writers of that storage (WAR/WAW). Source operator
        contracts remain responsible for math, layout and supported branches.
        """
        if not isinstance(node, WorkloadNode) or node.rank != self.rank:
            raise ValueError("forward operator must declare this physical rank")
        node_id = self.qualify(node.node_id)
        if node_id in self._nodes:
            raise ValueError("forward operator IDs must be unique")
        if not isinstance(inputs, tuple) or (not inputs and out is None):
            raise ValueError("forward operators require tensor inputs or an explicit write-only output")
        for tensor in inputs:
            self._check_tensor(tensor)
        if not isinstance(writes, tuple):
            raise TypeError("symbolic mutation targets must be a tuple")
        for tensor in writes:
            self._check_tensor(tensor)
        if not isinstance(controls, tuple):
            raise TypeError("symbolic control dependencies must be a tuple")
        for tensor in controls:
            self._check_tensor(tensor)
        if out is not None:
            self._check_tensor(out)
            if out.view.shape != shape or out.view.dtype != dtype:
                raise ValueError("forward out layout disagrees with the declared result")
        if type(parameters) is not tuple:
            raise TypeError("source parameter ports must be a tuple")
        for view in parameters:
            if (type(view) is not TensorView or view.view_id not in self._views
                    or self._views[view.view_id] is not view
                    or view.allocation_id not in self._persistent_ids
                    or self._writers[view.allocation_id] is not None):
                raise ValueError("parameter port requires this trace's readonly parameter")
        # Validate before recording any events or allocating output storage.
        layout = self._dense_view(node_id, shape, dtype)
        dependencies = set()
        for tensor in inputs:
            writer = self._writers[tensor.view.allocation_id]
            if writer is not None:
                dependencies.add(writer)
        for target in ((*writes, out) if out is not None else writes):
            storage = target.view.allocation_id
            writer = self._writers[storage]
            if writer is not None:
                dependencies.add(writer)
            dependencies.update(self._readers[storage])
        control_dependencies = {self._writers[tensor.view.allocation_id] for tensor in controls}
        control_dependencies.discard(None)
        result = self._tensor_from_view(layout) if out is None else out
        self._nodes[node_id] = node.with_attributes({**node.attributes,
            "symbolic_tensor_inputs": (*tuple(t.view.view_id for t in inputs), *tuple(view.view_id for view in parameters)),
            "symbolic_tensor_outputs": tuple(t.view.view_id for t in (result, *writes)),
            **({"symbolic_parameter_inputs": tuple(view.view_id for view in parameters)} if parameters else {}),
            **({"symbolic_tensor_controls": tuple(t.view.view_id for t in controls)} if controls else {})}, node_id=node_id)
        self._edges.extend(WorkloadEdge(source, node_id) for source in sorted(dependencies))
        self._edges.extend(WorkloadEdge(source, node_id, "barrier")
            for source in sorted(control_dependencies - dependencies))
        for tensor in inputs:
            self._readers[tensor.view.allocation_id].add(node_id)
        for view in parameters:
            self._readers[view.allocation_id].add(node_id)
        for target in (*writes, result):
            storage = target.view.allocation_id
            self._writers[storage] = node_id
            self._readers[storage].clear()
        self._event(f"return:{node_id}")
        return result

    def call_outputs(self, node, inputs, *, outputs, workspace=(), controls=(), parameters=()):
        """One source operation with multiple independent fresh tensor outputs.

        Each output is (role, shape, dtype). Roles name storage; the call and
        input dependencies remain single. Optional source workspace uses the
        same layouts and is retained only within the operator scope. No tuple
        or graph record pins a handle after the caller releases it.
        """
        if not isinstance(node, WorkloadNode) or node.rank != self.rank:
            raise ValueError("forward operator must declare this physical rank")
        name = self.qualify(node.node_id)
        if name in self._nodes or not isinstance(inputs, tuple) or not inputs:
            raise ValueError("multi-output call needs a fresh node and explicit inputs")
        for tensor in inputs:
            self._check_tensor(tensor)
        if not isinstance(controls, tuple):
            raise TypeError("symbolic control dependencies must be a tuple")
        for tensor in controls:
            self._check_tensor(tensor)
        if not isinstance(outputs, tuple) or not outputs:
            raise ValueError("multi-output call requires output role/shape/dtype tuples")
        if not isinstance(workspace, tuple):
            raise TypeError("operator workspace requires role/shape/dtype tuples")
        layouts = []
        for output in (*outputs, *workspace):
            if not isinstance(output, tuple) or len(output) != 3:
                raise TypeError("output layout must contain role, shape and dtype")
            role, shape, dtype = output
            if not isinstance(role, str) or not role:
                raise ValueError("output role must be a nonempty string")
            layout = self._dense_view(name + ":" + role, shape, dtype)
            if layout.view_id in self._views or layout.allocation_id in self._allocations:
                raise ValueError("multi-output storage IDs must be fresh")
            layouts.append(layout)
        if len({view.view_id for view in layouts}) != len(layouts):
            raise ValueError("multi-output roles must be unique")
        results = tuple(self._tensor_from_view(view) for view in layouts[:len(outputs)])
        scratch = tuple(self._tensor_from_view(view) for view in layouts[len(outputs):])
        self.call(node, inputs, shape=results[0].shape, dtype=results[0].dtype,
            out=results[0], writes=results[1:], controls=controls, parameters=parameters)
        # Keep scratch across the source operation, then let its local scope
        # release it. It is neither a returned tensor nor a separately priced op.
        del scratch
        return results

    @property
    def graph(self):
        if self._collective_ids:
            raise ValueError("rank-local collective traces require the joined group graph")
        if self._recorded_graph is not None:
            nodes, graph = self._recorded_graph
            if nodes is self._nodes and len(nodes) == len(graph.nodes):
                return graph
        return WorkloadGraph(tuple(self._nodes.values()), self._capture_program_edges())

    def _capture_program_edges(self):
        """Preserve synchronous source-stream boundaries around capture regions.

        Only capture entry/exit and PIECEWISE split boundaries are ordered.
        Adjacent region sinks join at the next region's first source operation,
        which gates its other independent roots. This linear-size boundary hub
        preserves the remaining intra-region DAG concurrency; it does not
        serialize every operation or infer compiler/internal stream parallelism.
        ForwardTrace describes one synchronous source program, not multistream
        execution. No new operator, tensor access or timing parameter is added.
        Ordinary traces without capture retain their exact original edges.
        """
        if not self._captures and self._capture is None:
            return tuple(self._edges)
        if self._capture is not None:
            raise ValueError("capture program boundaries require a finished source capture")
        names = tuple(self._nodes)
        regions = []

        def append_region(members, captured):
            if not members:
                return
            if not captured and regions and not regions[-1][1]:
                regions[-1][0].extend(members)
            else:
                regions.append((list(members), captured))

        cursor = 0
        for capture in self._captures:
            start = capture["node_start"]
            stop = start + len(capture["model_node_ids"])
            append_region(names[cursor:start], False)
            members, captured = [], None
            for name in names[start:stop]:
                node = self._nodes[name]
                inside = not (node.kind.value in capture["eager_kinds"]
                    or (capture["runtime_mode"] == "PIECEWISE" and node.kind is WorkloadKind.ELEMENTWISE
                        and "source_attention_program" in node.attributes
                        and node.attributes["source_attention_program"] == "ascend_0_18_piecewise_fia"
                        and "source_component" in node.attributes
                        and node.attributes["source_component"] == "symbolic_ascend_fia_output_copy"))
                if members and inside != captured:
                    append_region(members, captured)
                    members = []
                members.append(name)
                captured = inside
            append_region(members, captured)
            cursor = stop
        append_region(names[cursor:], False)
        ownership = {name: index for index, (members, _) in enumerate(regions) for name in members}
        incoming, outgoing = set(), set()
        existing = {(edge.source, edge.destination) for edge in self._edges}
        for source, destination in existing:
            if source in ownership and destination in ownership and ownership[source] == ownership[destination]:
                incoming.add(destination)
                outgoing.add(source)
        barriers = []

        def barrier(source, destination):
            if (source, destination) not in existing:
                existing.add((source, destination))
                barriers.append(WorkloadEdge(source, destination, "barrier"))

        for (before, before_capture), (after, after_capture) in zip(regions, regions[1:]):
            if not (before_capture or after_capture):
                continue
            first = after[0]
            if first in incoming:
                raise ValueError("capture boundary hub must be the next region's first source root")
            for source in before:
                if source not in outgoing:
                    barrier(source, first)
            for destination in after[1:]:
                if destination not in incoming:
                    barrier(first, destination)
        return (*self._edges, *barriers)

    @property
    def operator_order_by_rank(self):
        """Host forward call order, independent of legal DAG schedules."""
        if self._collective_ids:
            raise ValueError("collective traces require their joined group's call order")
        return {self.rank: tuple(self._nodes)}

    @property
    def allocations(self):
        return tuple(self._allocations.values())

    @property
    def activation_allocations(self):
        if (type(self) is ForwardTrace and self._recorded_graph is not None
                and type(self._allocations) is MappingProxyType
                and type(self._persistent_ids) is frozenset and type(self._events) is tuple
                and "_closed_storage_inputs" in self.__dict__):
            allocations, persistent, events, activations = self.__dict__["_closed_storage_inputs"]
            if self._allocations is allocations and self._persistent_ids is persistent and self._events is events:
                return activations
        return tuple(a for name, a in self._allocations.items() if name not in self._persistent_ids)

    @property
    def parameter_allocation_ids(self):
        """Readonly module parameters, charged by the model-weight ledger.

        These descriptors may be declared lazily while tracing a forward;
        their declaration is not a runtime parameter allocation or profile
        workspace growth. Only explicit validated parameter ports qualify.
        """
        return frozenset(self._views[name].allocation_id for node in self._nodes.values()
            if "symbolic_parameter_inputs" in node.attributes
            for name in node.attributes["symbolic_parameter_inputs"])

    @property
    def views(self):
        return tuple(self._views.values())

    @property
    def event_order(self):
        return tuple(self._events)


class ForwardSourceProgram:
    """Bind a recorded complete source to values from its actual source prefix.

    Bindings describe source-produced metadata fields, not a second model DAG.
    Every static operator, typed view, alias, lifetime and capture event remains
    in the complete trace. IDs are local to each graph; reusing their spelling
    never declares cross-step physical allocation or KV backing identity.
    """

    def __init__(self, source, prefix, *, node_bindings, allocation_bindings, collective_groups=None,
                 transient_allocation_bindings=(), view_bindings=(), capture_bindings=()):
        if not isinstance(source, ForwardTrace) or not isinstance(prefix, ForwardTrace):
            raise TypeError("source program requires actual complete and prefix traces")
        if (source.namespace != prefix.namespace or source.rank != prefix.rank
                or source._capture is not None or prefix._capture is not None
                or source._checkpoints or source._workspace_capacities
                or (source._collective_ids and collective_groups is None)):
            raise ValueError("source program requires a closed ordinary rank-local source")
        self.source, self.prefix = source, prefix
        self.node_bindings = tuple(node_bindings)
        persistent_bindings = tuple(allocation_bindings)
        transient_bindings = tuple(transient_allocation_bindings)
        self.allocation_bindings = (*persistent_bindings, *transient_bindings)
        self.view_bindings, self.capture_bindings = tuple(view_bindings), tuple(capture_bindings)
        if len({target for target, _ in self.allocation_bindings}) != len(self.allocation_bindings):
            raise ValueError("source geometry allocation targets must be unique")
        if len({target for target, _ in self.view_bindings}) != len(self.view_bindings):
            raise ValueError("source geometry view targets must be unique")
        self.transient_origins = {origin for _, origin in transient_bindings}
        self.bound_prefix_views = {origin for _, origin in self.view_bindings}
        self.capture_fields = {}
        capture_targets = set()
        for target, origin, fields in self.capture_bindings:
            if (type(target) is not int or type(origin) is not int
                    or fields != ("logical_tokens",) or target in capture_targets or origin in self.capture_fields
                    or not 0 <= target < len(source.source_captures)
                    or not 0 <= origin < len(prefix.source_captures)
                    or any(type(capture["logical_tokens"]) is not int
                        or type(capture["model_input_tokens"]) is not int
                        or not 0 < capture["logical_tokens"] <= capture["model_input_tokens"]
                        for capture in (source.source_captures[target], prefix.source_captures[origin]))
                    or source.source_captures[target]["logical_tokens"] != prefix.source_captures[origin]["logical_tokens"]):
                raise ValueError("source capture binding requires explicit matching logical token extents")
            capture_targets.add(target)
            self.capture_fields[origin] = fields
        for target, origin in self.view_bindings:
            left, right = source._views[target], prefix._views[origin]
            if (left.shape, left.strides_bytes, left.offset_bytes, left.dtype) != (
                    right.shape, right.strides_bytes, right.offset_bytes, right.dtype):
                raise ValueError("initial source view binding has incompatible typed geometry")
        if collective_groups is not None:
            if (not isinstance(collective_groups, tuple) or len(collective_groups) != 2
                    or any(not isinstance(group, ForwardCollectiveGroup) for group in collective_groups)
                    or not any(source is trace for trace in collective_groups[0].traces)
                    or not any(prefix is trace for trace in collective_groups[1].traces)):
                raise ValueError("rank program requires its complete source and prefix collective groups")
            self.graph = WorkloadGraph(tuple(source._nodes.values()), source._capture_program_edges())
        else:
            self.graph = source.graph
        ignored_shapes, ignored_attributes = {}, {}
        for target, origin, shapes, attributes in self.node_bindings:
            node, current = source._nodes[target], prefix._nodes[origin]
            if any(node.shape[name] != current.shape[name] for name in shapes):
                raise ValueError("source program initial shape binding disagrees with complete source")
            if any(node.attributes[name] != current.attributes[name] for name in attributes):
                raise ValueError("source program initial attribute binding disagrees with complete source")
            if origin not in ignored_shapes:
                ignored_shapes[origin], ignored_attributes[origin] = set(), set()
            ignored_shapes[origin].update(shapes)
            ignored_attributes[origin].update(attributes)
        self.prefix_contract = tuple((name, node.kind, node.pool_name, node.dtype, node.variant, node.rank,
            {key: value for key, value in node.shape.items()
                if name not in ignored_shapes or key not in ignored_shapes[name]},
            {key: value for key, value in node.attributes.items()
                if name not in ignored_attributes or key not in ignored_attributes[name]})
            for name, node in prefix._nodes.items())
        self.ignored_shapes, self.ignored_attributes = ignored_shapes, ignored_attributes
        self.bound_prefix_allocations = {origin for _, origin in self.allocation_bindings}
        for target, origin in self.allocation_bindings:
            transient = origin in self.transient_origins
            if ((target in source._persistent_ids) != (not transient)
                    or (origin in prefix._persistent_ids) != (not transient)):
                raise ValueError("source geometry binding must preserve its declared persistent/transient role")
            if self._allocation_geometry(source._allocations[target]) != self._allocation_geometry(prefix._allocations[origin]):
                raise ValueError("source program initial persistent allocation geometry disagrees")
            left, right = source._views[target], prefix._views[origin]
            if left.allocation_id != target or right.allocation_id != origin:
                raise ValueError("persistent program geometry requires the complete backing view")
            if (left.shape, left.strides_bytes, left.offset_bytes, left.dtype) != (
                    right.shape, right.strides_bytes, right.offset_bytes, right.dtype):
                raise ValueError("source program initial persistent view geometry disagrees")

    @staticmethod
    def _allocation_geometry(allocation):
        return (allocation.size_bytes, allocation.device_id, allocation.memory_domain,
            allocation.address_level, allocation.base_address_bytes, allocation.mapping_ref)

    def _capture_contract(self, trace):
        if not self.capture_fields:
            return trace.source_captures
        return tuple({name: value for name, value in capture.items()
            if index not in self.capture_fields or name not in self.capture_fields[index]}
            for index, capture in enumerate(trace.source_captures))

    def bind(self, prefix, *, native_frame=False):
        """Return a complete bound trace, or a cache miss for changed source structure."""
        if not isinstance(prefix, ForwardTrace):
            raise TypeError("source binding requires an actual prefix trace")
        if (prefix.namespace != self.source.namespace or prefix.rank != self.source.rank
                or tuple(prefix._nodes) != tuple(item[0] for item in self.prefix_contract)):
            return None
        if len(prefix.source_captures) != len(self.prefix.source_captures):
            return None
        for _, origin, _ in self.capture_bindings:
            capture = prefix.source_captures[origin]
            if (type(capture["logical_tokens"]) is not int or type(capture["model_input_tokens"]) is not int
                    or not 0 < capture["logical_tokens"] <= capture["model_input_tokens"]):
                return None
        if (prefix.event_order != self.prefix.event_order or self._capture_contract(prefix) != self._capture_contract(self.prefix)
                or prefix._persistent_ids != self.prefix._persistent_ids
                or prefix._references != self.prefix._references or prefix._writers != self.prefix._writers
                or prefix._readers != self.prefix._readers
                or tuple(prefix._allocations) != tuple(self.prefix._allocations)
                or tuple(prefix._views) != tuple(self.prefix._views)):
            return None
        for name, allocation in prefix._allocations.items():
            if name not in self.bound_prefix_allocations:
                if allocation != self.prefix._allocations[name]:
                    return None
            elif tuple(getattr(allocation, field) for field in ("device_id", "memory_domain", "address_level", "base_address_bytes", "mapping_ref", "allocated_event", "freed_event")) != tuple(
                    getattr(self.prefix._allocations[name], field) for field in ("device_id", "memory_domain", "address_level", "base_address_bytes", "mapping_ref", "allocated_event", "freed_event")):
                return None
        for name, view in prefix._views.items():
            if name not in self.bound_prefix_allocations and name not in self.bound_prefix_views:
                if view != self.prefix._views[name]:
                    return None
            else:
                old = self.prefix._views[name]
                if (view.allocation_id, view.offset_bytes, view.dtype, view.element_bytes, len(view.shape)) != (
                        old.allocation_id, old.offset_bytes, old.dtype, old.element_bytes, len(old.shape)):
                    return None
        for name, kind, pool, dtype, variant, rank, shape, attributes in self.prefix_contract:
            node = prefix._nodes[name]
            if (node.kind, node.pool_name, node.dtype, node.variant, node.rank) != (kind, pool, dtype, variant, rank):
                return None
            actual_shape = {key: value for key, value in node.shape.items()
                if name not in self.ignored_shapes or key not in self.ignored_shapes[name]}
            actual_attributes = {key: value for key, value in node.attributes.items()
                if name not in self.ignored_attributes or key not in self.ignored_attributes[name]}
            if actual_shape != shape or actual_attributes != attributes:
                return None
        result = copy(self.source)
        result._nodes, result._allocations, result._views = (
            dict(self.source._nodes), dict(self.source._allocations), dict(self.source._views))
        if native_frame:
            updates = {}
            for target, origin, shapes, attributes in self.node_bindings:
                current = prefix._nodes[origin]
                shape = {name: current.shape[name] for name in shapes}
                if any(type(value) is not int or value < 0 for value in shape.values()):
                    raise ValueError("source frame requires canonical bound shape dimensions")
                updates[target] = (shape, {name: current.attributes[name] for name in attributes})
            result._nodes = _source_rank_frame(self, updates)
            if result._nodes is None: return None
        elif self.source._collective_ids:
            for target, origin, shapes, attributes in self.node_bindings:
                node, current = result._nodes[target], prefix._nodes[origin]
                result._nodes[target] = node.with_updates(
                    {name: current.shape[name] for name in shapes},
                    {name: current.attributes[name] for name in attributes})
        else:
            validated_shapes = {}
            for target, origin, shapes, attributes in self.node_bindings:
                node, current = result._nodes[target], prefix._nodes[origin]
                bound_attributes = dict(node.attributes)
                for name in attributes:
                    bound_attributes[name] = current.attributes[name]
                if shapes:
                    shape = dict(node.shape)
                    for name in shapes:
                        shape[name] = current.shape[name]
                    # Bindings often update the same homogeneous attention shape
                    # in every layer. Validate once per exact shape in this bind;
                    # node IDs, rank and source annotations remain independently
                    # bound. Noncanonical values retain the ordinary constructor.
                    canonical = (type(node) is WorkloadNode and all(
                        type(name) is str and type(value) is int
                        for name, value in shape.items()))
                    key = tuple(shape.items()) if canonical else None
                    if canonical and key in validated_shapes:
                        bound = node.with_attributes(bound_attributes)
                        object.__setattr__(bound, "shape", validated_shapes[key])
                    else:
                        bound = replace(node, shape=shape, attributes=bound_attributes)
                        if canonical:
                            validated_shapes[key] = bound.shape
                    result._nodes[target] = bound
                else:
                    result._nodes[target] = node.with_attributes(bound_attributes)
        self._bind_storage_parameters(result, prefix._allocations, prefix._views, prefix.source_captures)
        if native_frame:
            result._recorded_graph = (result._nodes, None)
            _seal_native_bound_storage(self, result)
        else:
            graph = self.graph.with_nodes(result._nodes.values())
            result._recorded_graph = (result._nodes, graph)
        return result


    def _bind_storage_parameters(self, result, allocations, views, captures):
        for target, origin in self.allocation_bindings:
            old = result._allocations[target]
            result._allocations[target] = allocations[origin].with_lifetime(
                target, old.allocated_event, old.freed_event)
            result._views[target] = views[origin].with_identity(target, target)
        for target, origin in self.view_bindings:
            old = result._views[target]
            result._views[target] = views[origin].with_identity(target, old.allocation_id)
        if self.view_bindings or self.transient_origins:
            for view in result._views.values():
                if region_end(view.region()) > result._allocations[view.allocation_id].size_bytes:
                    raise ValueError("bound source view exceeds its actual backing allocation")
        if self.capture_bindings:
            result._captures = [dict(capture) for capture in self.source.source_captures]
            for target, origin, fields in self.capture_bindings:
                for name in fields:
                    result._captures[target][name] = captures[origin][name]


def _seal_native_bound_storage(program, trace):
    """Own readonly closed storage; ordinary mutable source traces stay live.

    A source may legitimately edit its template after an earlier bind. Compare
    actual referents rather than IDs or hashes; changed input gets a fresh
    tuple and the storage owner's complete validation on first use.
    """
    if (type(program) is not ForwardSourceProgram or type(trace) is not ForwardTrace
            or trace._recorded_graph is None or type(trace._allocations) is not dict
            or any(type(value) is not MemoryAllocation for value in trace._allocations.values())
            or any(type(value) is not str for value in trace._persistent_ids)):
        return
    activations = tuple(value for name, value in trace._allocations.items() if name not in trace._persistent_ids)
    events = tuple(trace._events)
    if "_native_storage_tuples" in program.__dict__:
        old_activations, old_events = program.__dict__["_native_storage_tuples"]
        if (len(activations) == len(old_activations)
                and all(map(operator.is_, activations, old_activations))):
            activations = old_activations
        if len(events) == len(old_events) and all(map(operator.is_, events, old_events)):
            events = old_events
    program.__dict__["_native_storage_tuples"] = (activations, events)
    trace._allocations = MappingProxyType(trace._allocations)
    trace._persistent_ids = frozenset(trace._persistent_ids)
    trace._events = events
    trace.__dict__["_closed_storage_inputs"] = (trace._allocations, trace._persistent_ids, events, activations)


class _NativeStorageOverlay(Mapping):
    """Closed source mapping with independently bound current backing values."""

    __slots__ = ("base", "bindings", "current_values", "allocation", "cache")

    def __init__(self, base, bindings, values, *, allocation):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "current_values", MappingProxyType(dict(values)))
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "cache", {})

    def __setattr__(self, name, value):
        raise TypeError("bound source storage is immutable")

    def __delattr__(self, name):
        raise TypeError("bound source storage is immutable")

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        return iter(self.base)

    def __getitem__(self, name):
        if name not in self.bindings:
            return self.base[name]
        if name not in self.cache:
            value = self.current_values[self.bindings[name]]
            if self.allocation:
                old = self.base[name]
                value = value.with_lifetime(name, old.allocated_event, old.freed_event)
            else:
                value = value.with_identity(name, name)
            self.cache[name] = value
        return self.cache[name]


def _bind_native_storage(program, trace, allocations, views):
    """Reuse detached static storage after comparing the actual template.

    Ordered keys and every current frozen referent are checked in C;
    mutable source identity is never a validity proof. Canonical frozen storage
    values can then be shared while every declared backing is derived afresh.
    More general view/transient programs keep the complete ordinary binder.
    """
    source = program.source
    if (program.view_bindings or program.capture_bindings or program.transient_origins
            or type(source._allocations) is not dict or type(source._views) is not dict
            or type(source._persistent_ids) is not set or type(source._events) is not list):
        return False
    # Boundary validation remains eager even when no consumer reads persistent
    # backing values. Only canonical frozen inputs enter the lazy derivation.
    for _, origin in program.allocation_bindings:
        if type(allocations[origin]) is not MemoryAllocation or type(views[origin]) is not TensorView:
            return False
    snapshot = program.__dict__["_native_storage_template"] if "_native_storage_template" in program.__dict__ else None
    if (snapshot is None or tuple(source._allocations) != snapshot[8] or tuple(source._views) != snapshot[9]
            or not all(map(operator.is_, source._allocations.values(), snapshot[0].values()))
            or not all(map(operator.is_, source._views.values(), snapshot[1].values()))
            or source._persistent_ids != snapshot[2] or source._events != snapshot[3]
            or program.allocation_bindings is not snapshot[11]):
        if (any(type(value) is not MemoryAllocation for value in source._allocations.values())
                or any(type(value) is not TensorView for value in source._views.values())
                or any(type(name) is not str for name in source._persistent_ids)
                or any(target not in source._persistent_ids for target, _ in program.allocation_bindings)):
            return False
        allocation_base, view_base = dict(source._allocations), dict(source._views)
        persistent, events = frozenset(source._persistent_ids), list(source._events)
        activations = tuple(value for name, value in allocation_base.items() if name not in persistent)
        snapshot = (allocation_base, view_base, persistent, events,
            MappingProxyType(allocation_base), MappingProxyType(view_base), tuple(events), activations,
            tuple(allocation_base), tuple(view_base), MappingProxyType(dict(program.allocation_bindings)),
            program.allocation_bindings)
        program.__dict__["_native_storage_template"] = snapshot
    trace._allocations = MappingProxyType(_NativeStorageOverlay(snapshot[4], snapshot[10], allocations, allocation=True))
    trace._views = MappingProxyType(_NativeStorageOverlay(snapshot[5], snapshot[10], views, allocation=False))
    trace._persistent_ids, trace._events = snapshot[2], snapshot[6]
    trace.__dict__["_closed_storage_inputs"] = (trace._allocations, trace._persistent_ids, trace._events, snapshot[7])
    return True


class SourceProgramCache:
    """Bounded run-local source programs; eviction resumes actual source execution."""

    def __init__(self, max_entries=128):
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("source program cache capacity must be a positive integer")
        self.max_entries = max_entries
        self.programs = OrderedDict()
        self.next_program_id = 0

    def namespace(self, model_type, pool, rank, source_version):
        identity = self.next_program_id
        self.next_program_id += 1
        return f"source-program:{model_type}:{pool}:rank-{rank}:{source_version}:{identity}"

    def lookup(self, key):
        if key not in self.programs:
            return None
        self.programs.move_to_end(key)
        return self.programs[key]

    def store(self, key, program):
        if not isinstance(program, ForwardSourceProgram):
            raise TypeError("source cache entries must be actual recorded source programs")
        self.programs[key] = program
        self.programs.move_to_end(key)
        while len(self.programs) > self.max_entries:
            self.programs.popitem(last=False)


class ForwardWorkspace:
    """One explicitly initialized, aligned source workspace lane.

    Mirrors vLLM0.30 WorkspaceManager.get_simultaneous/_ensure_workspace_size.
    Views share a UINT8 arena. Growth drops the manager reference before new
    allocation; caller-retained views alone determine whether old storage
    overlaps. This models source live allocations, not allocator reserved
    bytes, asynchronous lanes, a warm runtime history or an installed GPU.
    """

    def __init__(self, trace, name, *, alignment_bytes, initial_capacity_bytes, locked):
        if not isinstance(trace, ForwardTrace) or not isinstance(name, str) or not name:
            raise TypeError("workspace requires a forward trace and nonempty source name")
        if (type(alignment_bytes) is not int or alignment_bytes <= 0
                or type(initial_capacity_bytes) is not int or initial_capacity_bytes < 0
                or type(locked) is not bool):
            raise ValueError("workspace requires explicit positive alignment, nonnegative initial size and lock state")
        self.trace, self.name = trace, name
        if name in trace._workspace_capacities:
            raise ValueError("source workspace names must be unique within a rank forward")
        self.alignment_bytes, self.locked = alignment_bytes, locked
        self._current = (None if initial_capacity_bytes == 0 else
            trace.tensor(trace.unique_name(name), (initial_capacity_bytes,), "uint8"))
        trace._workspace_capacities[name] = self.capacity_bytes

    @property
    def capacity_bytes(self):
        return 0 if self._current is None else self._current.shape[0]

    def get_simultaneous(self, *shapes_and_dtypes):
        if not shapes_and_dtypes:
            raise ValueError("workspace needs at least one nonempty typed layout")
        layouts, offsets, total = [], [], 0
        for item in shapes_and_dtypes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("workspace layout must be a shape/dtype pair")
            shape, dtype = item
            layout = self.trace._dense_view("layout", shape, dtype)
            if self.alignment_bytes % layout.element_bytes:
                raise ValueError("workspace alignment must align every requested dtype")
            layouts.append(layout)
            offsets.append(total)
            size = region_end(layout.region())
            total += -(-size // self.alignment_bytes) * self.alignment_bytes
        if total > self.capacity_bytes:
            if self.locked:
                raise ValueError("locked source workspace cannot grow")
            self._current = None
            self._current = self.trace.tensor(self.trace.unique_name(self.name), (total,), "uint8")
            self.trace._workspace_capacities[self.name] = self.capacity_bytes
        return tuple(self.trace.alias(self._current, self.trace.unique_name(self.name + ":view"),
            shape=layout.shape, strides_bytes=layout.strides_bytes, offset_bytes=offset, dtype=layout.dtype)
            for layout, offset in zip(layouts, offsets, strict=True))


class ForwardPipelineGroup:
    """Synchronous PP transport between independent physical rank traces.

    Source scopes retain send buffers; receives allocate on the destination.
    This records data/control edges, not transport metadata, async retirement
    or a distributed runtime. Equal TP2 or TP8 stages use flattened lane views,
    full receiver backings and separately traced destination all-gathers.
    """

    def __init__(self, traces, *, namespace, collective_groups=()):
        if (not isinstance(traces, tuple) or len(traces) < 2
                or any(not isinstance(trace, ForwardTrace) or trace.namespace is None for trace in traces)
                or len({trace.rank for trace in traces}) != len(traces)
                or len({trace.namespace for trace in traces}) != len(traces)
                or not isinstance(namespace, str) or not namespace):
            raise ValueError("pipeline tracing requires distinct, namespaced stage ranks")
        self.traces, self.namespace = traces, namespace
        if collective_groups:
            if (not isinstance(collective_groups, tuple) or len(collective_groups) < 2
                    or any(not isinstance(group, ForwardCollectiveGroup) or len(group.traces) not in (2, 8)
                        for group in collective_groups)
                    or len({len(group.traces) for group in collective_groups}) != 1
                    or tuple(trace for group in collective_groups for trace in group.traces) != traces):
                raise ValueError("sliced pipeline requires ordered equal TP2 or TP8 stage groups")
        self.collective_groups = collective_groups
        self._previous = {trace.rank: None for trace in traces}

    def send(self, tensor, *, config, pool_name, tensor_name, frontier, out=None):
        if not isinstance(config, PipelineP2PConfig) or tensor_name not in ("hidden_states", "residual"):
            raise ValueError("pipeline transport requires canonical config and intermediate tensor role")
        tp = len(self.collective_groups[0].traces) if self.collective_groups else 1
        if not 0 <= config.source_stage < config.destination_stage < len(self.traces) // tp:
            raise ValueError("pipeline stages are outside the declared rank placement")
        if tp == 1:
            source, destination = self.traces[config.source_stage], self.traces[config.destination_stage]
            lane = 0
        else:
            source_group = self.collective_groups[config.source_stage]
            if not any(tensor.trace is trace for trace in source_group.traces):
                raise ValueError("pipeline slice belongs to another source stage")
            lane = source_group.traces.index(tensor.trace)
            source = source_group.traces[lane]
            destination = self.collective_groups[config.destination_stage].traces[lane]
        source._check_tensor(tensor)
        source._check_tensor(frontier)
        if tensor.dtype != "bf16" or len(tensor.shape) != (2 if tp == 1 else 1) or not tensor.is_contiguous:
            raise NotImplementedError("pipeline requires a contiguous BF16 full tensor or TP lane")
        if (tp == 1 and out is not None) or (tp > 1 and out is None):
            raise ValueError("sliced PP requires an explicit full-backing destination view")
        suffix = "" if tp == 1 else f":lane-{lane}"
        boundary = f"{self.namespace}:pp-{config.source_stage}-to-{config.destination_stage}"
        name = f"{boundary}:{tensor_name}{suffix}"
        if name in source._nodes:
            raise ValueError("pipeline tensor has already been sent across this boundary")
        expected_previous = (None if tensor_name == "hidden_states" else
            f"{boundary}:hidden_states{suffix}")
        if self._previous[source.rank] != expected_previous:
            raise ValueError("pipeline sends must follow hidden_states then residual order")
        writer = source._writers[tensor.view.allocation_id]
        ready = source._writers[frontier.view.allocation_id]
        if writer is None or ready is None:
            raise ValueError("pipeline transport requires produced data and a completed stage frontier")
        if out is None:
            received = destination.tensor("receive:" + name, tensor.shape, tensor.dtype)
        else:
            destination._check_tensor(out)
            if out.shape != tensor.shape or out.dtype != tensor.dtype or not out.is_contiguous:
                raise ValueError("pipeline receive view must match the transmitted lane")
            received = out
        size = prod(tensor.shape) * tensor.view.element_bytes
        source._nodes[name] = WorkloadNode(name, WorkloadKind.P2P, pool_name, {}, rank=source.rank,
            attributes={"algorithm": config.algorithm, "bytes": size, "bytes_semantics": "payload",
                "backend": config.backend, "backend_config_sha256": config.backend_config_sha256,
                "direction": "point_to_point", "link": config.link, "topology": config.topology,
                "participants": (source.rank, destination.rank), "operation": "send_recv",
                "source_rank": source.rank, "destination_rank": destination.rank,
                "ring_order": (), "reduction": None, "resource_id": config.resource_id,
                "source_stage": config.source_stage, "destination_stage": config.destination_stage,
                "payload_tensor": {"name": tensor_name, "shape": tensor.shape, "dtype": tensor.dtype, "bytes": size},
                "symbolic_tensor_inputs": (tensor.view.view_id,),
                "symbolic_tensor_outputs": (received.view.view_id,),
                **({"symbolic_pipeline_views": {
                    "source": asdict(tensor.view), "destination": asdict(received.view),
                    "source_backing_bytes": source._allocations[tensor.view.allocation_id].size_bytes,
                    "destination_backing_bytes": destination._allocations[received.view.allocation_id].size_bytes,
                    "source_producer_views": tuple(asdict(source._views[view_id])
                        for view_id in source._nodes[writer].attributes["symbolic_tensor_outputs"])}}
                    if tp > 1 else {})})
        source._edges.append(WorkloadEdge(writer, name))
        previous = self._previous[source.rank]
        for dependency in (ready, previous):
            if dependency is not None and dependency != writer:
                edge = WorkloadEdge(dependency, name, "barrier")
                if edge not in source._edges:
                    source._edges.append(edge)
        source._readers[tensor.view.allocation_id].add(name)
        destination._writers[received.view.allocation_id] = name
        self._previous[source.rank] = name
        source._event("send:" + name)
        destination._event("receive:" + name)
        return received

    @property
    def operator_order_by_rank(self):
        return {rank: order for source in (self.collective_groups or self.traces)
            for rank, order in source.operator_order_by_rank.items()}

    @property
    def graph(self):
        if self.collective_groups:
            parts = tuple(group._joined_parts() for group in self.collective_groups)
            graph = WorkloadGraph(tuple(node for nodes, _ in parts for node in nodes),
                tuple(edge for _, edges in parts for edge in edges))
            graph.topological_order()
            return graph
        if any(trace._collective_ids for trace in self.traces):
            raise NotImplementedError("full-tensor PP group does not join TP collectives")
        graph = WorkloadGraph(tuple(node for trace in self.traces for node in trace._nodes.values()),
            tuple(edge for trace in self.traces for edge in trace._capture_program_edges()))
        graph.topological_order()
        return graph


class ForwardActivationGroup:
    """Blocking full hidden/residual exchanges for the existing AF contract.

    This is the source projection of the configured AF service, not a claim
    that upstream vanilla vLLM implements its transport/provider. Each receive
    has independent storage; Python forward scopes determine its lifetime.
    """

    def __init__(self, deployment, pool_traces, *, collective_groups=(), graph_token_shape=None, graph_eager_kinds=()):
        if (deployment.mode not in ("af_disagg", "pd_af_disagg") or len(pool_traces) != 2
                or not set(pool_traces) <= set(deployment.worker_pools)
                or (deployment.mode == "af_disagg" and set(pool_traces) != set(deployment.worker_pools))):
            raise ValueError("activation tracing requires the two configured AF pools")
        selected = tuple(deployment.worker_pools[name] for name in pool_traces)
        phases = {worker_role_phase(pool.role) for pool in selected}
        if ({worker_role_operator_group(pool.role) for pool in selected} != {"attention", "ffn"}
                or len(phases) != 1
                or (deployment.mode == "af_disagg" and phases != {"colocated"})
                or (deployment.mode == "pd_af_disagg" and not phases <= {"prefill", "decode"})):
            raise ValueError("activation tracing requires an Attention/FFN pair in the same phase")
        for name, traces in pool_traces.items():
            if (not isinstance(traces, tuple) or tuple(trace.rank for trace in traces) != deployment.worker_pools[name].ranks
                    or any(not isinstance(trace, ForwardTrace) or trace.namespace is None for trace in traces)):
                raise ValueError("activation tracing requires each pool's ordered physical rank traces")
        self.traces = tuple(trace for traces in pool_traces.values() for trace in traces)
        if len({trace.rank for trace in self.traces}) != len(self.traces):
            raise ValueError("activation tracing requires independent physical workers")
        if collective_groups and (not isinstance(collective_groups, tuple)
                or {trace.rank for group in collective_groups for trace in group.traces} != {trace.rank for trace in self.traces}):
            raise ValueError("activation collective groups must cover the declared workers")
        self.deployment, self.pool_traces, self.collective_groups = deployment, pool_traces, collective_groups
        self.graph_token_shape, self.graph_eager_kinds = graph_token_shape, graph_eager_kinds
        self.attention_capture_inputs = {}

    def begin_local_capture(self, trace, inputs, *, boundary):
        """A single worker's compute segment; AF sends remain outside it."""
        if self.graph_token_shape is None:
            return
        shape = self.graph_token_shape
        if trace.rank in self.attention_capture_inputs:
            inputs = {**inputs, **self.attention_capture_inputs[trace.rank]}
        trace.begin_capture(key=f"midlayer_af:{boundary}:{shape.runtime_mode}:{shape.model_input_tokens}",
            logical_tokens=shape.logical_tokens, model_input_tokens=shape.model_input_tokens,
            inputs=inputs, runtime_mode=shape.runtime_mode, eager_kinds=self.graph_eager_kinds,
            source_stage="midlayer_af")
        trace._capture["execution_scope"] = "midlayer_af_service_compute_segment_not_upstream_native_runtime"

    def end_local_capture(self, trace, outputs):
        if self.graph_token_shape is not None:
            trace.end_capture(outputs=outputs)

    def exchange(self, hidden_states, residual, *, source_pool, destination_pool, boundary):
        """Carry both tensors on matching TP lanes without a TP reshard."""
        if (source_pool == destination_pool or source_pool not in self.pool_traces
                or destination_pool not in self.pool_traces or not isinstance(boundary, str) or not boundary):
            raise ValueError("activation exchange requires a named cross-pool boundary")
        traces = self.pool_traces[source_pool]
        if not any(hidden_states.trace is trace for trace in traces):
            raise ValueError("activation tensor must belong to the sending pool")
        lane = traces.index(hidden_states.trace)
        source = traces[lane]
        destination = self.pool_traces[destination_pool][lane]
        source._check_tensor(hidden_states)
        source._check_tensor(residual)
        if (hidden_states.shape != residual.shape or hidden_states.dtype != residual.dtype
                or hidden_states.dtype != "bf16" or len(hidden_states.shape) != 2
                or not hidden_states.is_contiguous or not residual.is_contiguous):
            raise NotImplementedError("activation exchange requires equal contiguous full BF16 tensors")
        routes = tuple(route for route in self.deployment.routes if route.payload == "activation"
            and route.source_pool == source_pool and route.destination_pool == destination_pool)
        if len(routes) != 1:
            raise ValueError("activation exchange requires exactly one configured route")
        route = routes[0]
        received, previous = {}, None
        for role, tensor in (("hidden_states", hidden_states), ("residual", residual)):
            name = source.qualify(f"{boundary}:{role}:transfer")
            if name in source._nodes:
                raise ValueError("activation boundary cannot be emitted twice on one lane")
            # A model may return a same-layout view of its last output. Bind
            # the send to that exact producer region without inventing a copy
            # or relaxing the lowering boundary to accept unrelated aliases.
            port = source._complete_producer_view(tensor)
            writer = source._writers[tensor.view.allocation_id]
            output = destination.tensor("receive:" + name, tensor.shape, tensor.dtype)
            size = prod(tensor.shape) * tensor.view.element_bytes
            source._nodes[name] = WorkloadNode(name, WorkloadKind.AF_ACTIVATION_TRANSFER, source_pool, {}, rank=source.rank,
                attributes={"algorithm": route.algorithm, "bytes": size, "bytes_semantics": "payload",
                    "backend": route.backend, "backend_config_sha256": route.backend_config_sha256,
                    "direction": "point_to_point", "link": route.link, "topology": route.topology,
                    "participants": (source.rank, destination.rank), "operation": "send_recv",
                    "source_rank": source.rank, "destination_rank": destination.rank,
                    "ring_order": (), "reduction": None, "resource_id": route.resource_id,
                    "destination_pool": destination_pool,
                    "payload_tensor": {"name": role, "shape": tensor.shape, "dtype": tensor.dtype, "bytes": size},
                    "symbolic_tensor_inputs": (port.view_id,), "symbolic_tensor_outputs": (output.view.view_id,),
                    "symbolic_af_boundary": boundary})
            source._edges.append(WorkloadEdge(writer, name, "ready"))
            if previous is not None:
                source._edges.append(WorkloadEdge(previous, name, "barrier"))
            source._readers[tensor.view.allocation_id].add(name)
            destination._writers[output.view.allocation_id] = name
            source._event("send:" + name)
            destination._event("receive:" + name)
            received[role] = output
            previous = name
        return received["hidden_states"], received["residual"]

    @property
    def operator_order_by_rank(self):
        return {rank: order for source in (self.collective_groups or self.traces)
            for rank, order in source.operator_order_by_rank.items()}

    @property
    def graph(self):
        parts = (tuple(group._joined_parts() for group in self.collective_groups) if self.collective_groups else
            tuple((tuple(trace._nodes.values()), trace._capture_program_edges()) for trace in self.traces))
        graph = WorkloadGraph(tuple(node for nodes, _ in parts for node in nodes),
            tuple(edge for _, edges in parts for edge in edges))
        graph.topological_order()
        return graph


class ForwardCollectiveGroup:
    """Join synchronous virtual TP calls without retaining tensor handles.

    Each rank runs its source forward independently. Local collective calls
    update that rank's storage writers; joining maps them to one communication
    node with all producer/consumer dependencies. Rank execution order in
    Python is never a cross-device storage lifetime order. Collective scratch,
    asynchronous retirement and actual provider selection are not inferred.
    """

    def __init__(self, traces, *, namespace, participant_nic_ids=(), ring_order=()):
        if (not isinstance(traces, tuple) or len(traces) < 2
                or any(not isinstance(trace, ForwardTrace) for trace in traces)):
            raise ValueError("collective tracing requires at least two rank traces")
        if len({trace.rank for trace in traces}) != len(traces):
            raise ValueError("collective traces must have unique physical ranks")
        if (any(trace.namespace is None for trace in traces)
                or len({trace.namespace for trace in traces}) != len(traces)):
            raise ValueError("collective traces require distinct rank namespaces")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("collective group requires a nonempty namespace")
        self.traces = traces
        self.namespace = namespace
        self.participants = tuple(trace.rank for trace in traces)
        if (not isinstance(participant_nic_ids, tuple)
                or (participant_nic_ids and (len(participant_nic_ids) != len(traces)
                    or any(not isinstance(nic, str) or not nic or nic != nic.strip() for nic in participant_nic_ids)))):
            raise ValueError("collective NIC mapping requires an explicit immutable physical ID per rank")
        self.participant_nic_ids = participant_nic_ids
        if (not isinstance(ring_order, tuple) or ring_order and (
                len(ring_order) != len(self.participants)
                or any(type(rank) is not int for rank in ring_order)
                or set(ring_order) != set(self.participants))):
            raise ValueError("collective ring order must be an immutable permutation of physical participants")
        self.ring_order = self.participants if not ring_order else ring_order
        self._explicit_ring_order = bool(ring_order)
        self._sequences = {rank: [] for rank in self.participants}
        self._contracts = {}
        self._layouts = {}
        self._submissions = {}
        self._recorded_graph = None

    def call(self, name, tensor, *, pool_name, config, inplace, dim=None, moe_invocation_id=None, output_shape=None,
             implementation=None, residual=None, rmsnorm_weight=None, rmsnorm_epsilon=None):
        """All-reduce (fresh or in-place) or fresh, axis-expanding all-gather.

        Input bytes are per-rank logical bytes, as in PhysicalTransferConfig.
        Layout, dtype, storage mode and call order must agree on every rank.
        Invocation IDs are derived from module calls, not a model DAG table.
        Optional output_shape records the source's dense post-gather view;
        it allocates no additional backing and remains the public output port.
        """
        if self._recorded_graph is not None:
            raise ValueError("a bound collective source program is a closed forward trace")
        if not isinstance(name, str) or not name:
            raise ValueError("symbolic collective requires a nonempty module name")
        if not isinstance(tensor, ForwardTensor) or not any(tensor.trace is trace for trace in self.traces):
            raise ValueError("symbolic collective input belongs to another rank group")
        if not isinstance(config, CollectiveConfig) or type(inplace) is not bool:
            raise TypeError("symbolic collective requires canonical config and explicit storage mode")
        if config.link == "roce" and not self.participant_nic_ids:
            raise ValueError("RoCE collective requires explicit shared physical NIC placement")
        if self._explicit_ring_order and (config.algorithm != "ring" or config.backend != "analytical_roofline_v1"):
            raise ValueError("explicit virtual ring order requires the analytical ring backend")
        if moe_invocation_id is not None and (not isinstance(moe_invocation_id, str) or not moe_invocation_id
                or config.operation != "all_reduce"):
            raise ValueError("MoE collective requires its explicit invocation and all-reduce")
        if not tensor.is_contiguous:
            raise NotImplementedError("symbolic collective requires a contiguous source input")
        if implementation not in (None, "sglang_custom_all_reduce_v2", "sglang_flashinfer_allreduce_residual_rmsnorm"):
            raise ValueError("collective implementation requires its existing source contract")
        fused_norm = implementation == "sglang_flashinfer_allreduce_residual_rmsnorm"
        if implementation is not None and (config.operation != "all_reduce" or config.link != "nvlink"):
            raise ValueError("SGLang collective implementation requires an NVLink all-reduce")
        if fused_norm:
            if (inplace or dim is not None or output_shape is not None or tensor.dtype != "bf16" or len(tensor.shape) != 2
                    or not isinstance(residual, ForwardTensor) or not isinstance(rmsnorm_weight, ForwardTensor)):
                raise ValueError("fused all-reduce/RMSNorm requires BF16 rank2 input, residual and weight with fresh outputs")
            tensor.trace._check_tensor(residual)
            tensor.trace._check_tensor(rmsnorm_weight)
            if (residual.shape != tensor.shape or residual.dtype != tensor.dtype or not residual.is_contiguous
                    or rmsnorm_weight.shape != (tensor.shape[-1],) or rmsnorm_weight.dtype != tensor.dtype
                    or not rmsnorm_weight.is_contiguous or isinstance(rmsnorm_epsilon, bool)
                    or not isinstance(rmsnorm_epsilon, (int, float)) or not isfinite(rmsnorm_epsilon) or rmsnorm_epsilon <= 0):
                raise ValueError("fused all-reduce/RMSNorm requires matching contiguous source ports and positive epsilon")
        elif residual is not None or rmsnorm_weight is not None or rmsnorm_epsilon is not None:
            raise ValueError("residual/RMSNorm ports require the explicit fused collective implementation")
        shape = tensor.shape
        if config.operation == "all_reduce":
            if dim is not None:
                raise ValueError("all-reduce has no gather dimension")
        elif config.operation == "all_gather":
            if inplace or type(dim) is not int or not -len(shape) <= dim < len(shape):
                raise ValueError("all-gather requires fresh output and an explicit valid dimension")
            dim %= len(shape)
            shape = tuple(size * len(self.traces) if axis == dim else size
                for axis, size in enumerate(shape))
        else:
            raise NotImplementedError("symbolic TP group supports only all-reduce and all-gather")
        if output_shape is not None and (config.operation != "all_gather"
                or not isinstance(output_shape, tuple) or not output_shape
                or any(type(size) is not int or size <= 0 for size in output_shape)
                or prod(output_shape) != prod(shape)):
            raise ValueError("all-gather output reshape must preserve its fresh dense allocation")
        rank = tensor.trace.rank
        sequence = self._sequences[rank]
        invocation = sequence.count(name)
        identifier = self.namespace + ":" + name + (f":{invocation}" if invocation else "")
        contract = WorkloadNode(identifier, WorkloadKind.COLLECTIVE, pool_name, {},
            dtype=tensor.dtype, rank=self.participants[0], attributes={
                "algorithm": config.algorithm, "backend": config.backend,
                "backend_config_sha256": config.backend_config_sha256,
                "bytes": prod(tensor.shape) * tensor.view.element_bytes,
                "bytes_semantics": "per_rank_input", "direction": "collective",
                "link": config.link, "topology": config.topology,
                "participants": self.participants, "operation": config.operation,
                **({"participant_nic_ids": self.participant_nic_ids} if config.link == "roce" else {}),
                "source_rank": None, "destination_rank": None,
                "ring_order": self.ring_order if config.algorithm == "ring" else (),
                "reduction": "sum" if config.operation == "all_reduce" else None,
                "resource_id": config.resource_id,
                "symbolic_collective_storage": "inplace" if inplace else "fresh",
                "symbolic_collective_dim": dim,
                **({"implementation": implementation} if implementation is not None else {}),
                **({"rmsnorm_epsilon": rmsnorm_epsilon, "symbolic_workspace_complete": False,
                    "symbolic_fused_outputs": ("normalized", "residual"), "symbolic_inputs_mutated": False}
                    if fused_norm else {}),
                **({"moe_invocation_id": moe_invocation_id} if moe_invocation_id is not None else {}),
            })
        layout = (tensor.shape, tensor.dtype, shape, output_shape,
            (residual.shape, residual.dtype, rmsnorm_weight.shape, rmsnorm_weight.dtype) if fused_norm else None)
        for other in self._sequences.values():
            if len(other) > len(sequence) and other[len(sequence)] != name:
                raise ValueError("symbolic collective order disagrees across ranks")
        if identifier in self._contracts:
            if contract != self._contracts[identifier] or layout != self._layouts[identifier]:
                raise ValueError("symbolic collective contract/layout disagrees across ranks")
            if rank in self._submissions[identifier]:
                raise ValueError("symbolic collective invocation collides with another module name")
        local = replace(contract, rank=rank)
        trace = tensor.trace
        producer_port = trace._complete_producer_view(tensor) if moe_invocation_id is not None else None
        if fused_norm:
            # FlashInfer's wrapper allocates both empty_like outputs; it does
            # not mutate/alias x or residual. Workspace remains unbound.
            result = trace.call_outputs(local, (tensor, residual, rmsnorm_weight), outputs=(
                ("normalized", tensor.shape, tensor.dtype), ("residual", residual.shape, residual.dtype)))
        else:
            result = trace.call(local, (tensor,), shape=shape, dtype=tensor.dtype,
                out=tensor if inplace else None)
        local_id = trace.qualify(identifier)
        if producer_port is not None:
            attrs = trace._nodes[local_id].attributes
            trace._nodes[local_id] = trace._nodes[local_id].with_attributes({**attrs,
                "symbolic_tensor_inputs": (producer_port.view_id, *attrs["symbolic_tensor_inputs"][1:])})
        if output_shape is not None:
            result = result.view_as_shape(output_shape)
            trace._nodes[local_id] = trace._nodes[local_id].with_attributes({
                **trace._nodes[local_id].attributes, "symbolic_tensor_outputs": (result.view.view_id,)})
        if sequence:
            previous_name = sequence[-1]
            previous_count = sequence.count(previous_name) - 1
            previous_id = self.namespace + ":" + previous_name + (f":{previous_count}" if previous_count else "")
            trace._edges.append(WorkloadEdge(trace.qualify(previous_id), local_id, "barrier"))
        trace._collective_ids.add(local_id)
        sequence.append(name)
        if identifier not in self._contracts:
            self._contracts[identifier] = contract
            self._layouts[identifier] = layout
            self._submissions[identifier] = {}
        self._submissions[identifier][rank] = local_id
        return result

    @property
    def operator_order_by_rank(self):
        """Preserve each source program's order with canonical collective IDs."""
        self._joined_parts()  # Validate the complete rank submissions first.
        mapping = {local_id: identifier for identifier, submissions in self._submissions.items()
            for local_id in submissions.values()}
        return {trace.rank: tuple(mapping[name] if name in mapping else name for name in trace._nodes)
            for trace in self.traces}

    def _joined_parts(self):
        reference = self._sequences[self.participants[0]]
        if any(sequence != reference for sequence in self._sequences.values()):
            raise ValueError("symbolic collective group has missing rank submissions")
        mapping = {}
        joined = {}
        for identifier, submissions in self._submissions.items():
            if set(submissions) != set(self.participants):
                raise ValueError("symbolic collective group has missing rank submissions")
            inputs, outputs = [], []
            for trace in self.traces:
                local_id = submissions[trace.rank]
                mapping[local_id] = identifier
                local = trace._nodes[local_id]
                inputs.extend(local.attributes["symbolic_tensor_inputs"])
                outputs.extend(local.attributes["symbolic_tensor_outputs"])
            joined[identifier] = replace(self._contracts[identifier], attributes={
                **self._contracts[identifier].attributes,
                "symbolic_tensor_inputs": tuple(inputs),
                "symbolic_tensor_outputs": tuple(outputs),
            })
        nodes, edges = [], set()
        emitted = set()
        for trace in self.traces:
            if trace._collective_ids - set(mapping):
                raise ValueError("rank trace contains collectives from another group")
            for local_id, node in trace._nodes.items():
                if local_id in mapping:
                    identifier = mapping[local_id]
                    if identifier not in emitted:
                        nodes.append(joined[identifier])
                        emitted.add(identifier)
                else:
                    nodes.append(node)
            for edge in trace._capture_program_edges():
                source = mapping[edge.source] if edge.source in mapping else edge.source
                destination = mapping[edge.destination] if edge.destination in mapping else edge.destination
                edges.add(WorkloadEdge(source, destination, edge.kind))
        # A data edge already enforces the process-group order. Avoid duplicate
        # KernelCall dependencies when ranks contributed both edge kinds.
        edges = {edge for edge in edges if edge.kind != "barrier"
            or WorkloadEdge(edge.source, edge.destination) not in edges}
        return tuple(nodes), tuple(sorted(edges, key=lambda edge:
            (edge.source, edge.destination, edge.kind)))

    @property
    def graph(self):
        if self._recorded_graph is not None:
            return self._recorded_graph
        graph = WorkloadGraph(*self._joined_parts())
        graph.topological_order()
        return graph



class ForwardCollectiveSourceProgram(ForwardSourceProgram):
    """Replay a complete source group with original submissions and local storage.

    Full collective contracts/layouts/submissions come from the complete source,
    never from extrapolating prefix payloads. The actual dual-rank prefix checks
    current static contracts and supplies only current attention/KV parameters.
    """

    def __init__(self, source, prefix, *, node_bindings, allocation_bindings, collective_attribute_bindings=(),
                 transient_allocation_bindings=(), view_bindings=(), capture_bindings=(), collective_geometry_bindings=()):
        if not isinstance(source, ForwardCollectiveGroup) or not isinstance(prefix, ForwardCollectiveGroup):
            raise TypeError("collective source program requires actual source/prefix groups")
        source_parts = source._joined_parts()
        prefix._joined_parts()
        if (source.namespace != prefix.namespace or source.participants != prefix.participants
                or len(source.traces) != 2 or source.participant_nic_ids != prefix.participant_nic_ids
                or source.ring_order != prefix.ring_order or source._explicit_ring_order != prefix._explicit_ring_order):
            raise ValueError("collective source programs require matching explicit dual-rank groups")
        self.collective_geometry_bindings = tuple(collective_geometry_bindings)
        for target, origin in self.collective_geometry_bindings:
            if (not self._collective_geometry_matches(source, target)
                    or not self._collective_geometry_matches(prefix, origin)
                    or replace(source._contracts[target], node_id=prefix._contracts[origin].node_id) != prefix._contracts[origin]
                    or source._layouts[target] != prefix._layouts[origin]):
                raise ValueError("initial collective geometry must be actual matching source facts")
        self.collective_attribute_bindings = tuple(collective_attribute_bindings)
        targets = tuple(binding[0] for binding in self.collective_attribute_bindings)
        if len(targets) != len(set(targets)):
            raise ValueError("collective attribute binding targets must be unique")
        local_bindings = []
        ignored = {}
        for target, origin, attributes in self.collective_attribute_bindings:
            if (target not in source._contracts or origin not in prefix._contracts
                    or attributes != ("moe_invocation_id",)):
                raise ValueError("collective binding requires declared MoE invocation identity only")
            for name in attributes:
                if (name not in source._contracts[target].attributes or name not in prefix._contracts[origin].attributes
                        or source._contracts[target].attributes[name] != prefix._contracts[origin].attributes[name]
                        or type(prefix._contracts[origin].attributes[name]) is not str
                        or not prefix._contracts[origin].attributes[name]):
                    raise ValueError("initial collective attribute binding disagrees with complete source")
            ignored[origin] = attributes
            for trace, current in zip(source.traces, prefix.traces, strict=True):
                source_id, prefix_id = source._submissions[target][trace.rank], prefix._submissions[origin][current.rank]
                if any(trace._nodes[source_id].attributes[name] != source._contracts[target].attributes[name]
                        or current._nodes[prefix_id].attributes[name] != prefix._contracts[origin].attributes[name]
                        for name in attributes):
                    raise ValueError("collective local and joined invocation identities disagree")
                local_bindings.append((source_id, prefix_id, (), attributes))
        for target, origin in self.collective_geometry_bindings:
            if origin in ignored and ignored[origin] != ("bytes",):
                raise ValueError("collective geometry cannot overlap annotation bindings")
            ignored[origin] = ("bytes",)
            for trace, current in zip(source.traces, prefix.traces, strict=True):
                local_bindings.append((source._submissions[target][trace.rank], prefix._submissions[origin][current.rank], (), ("bytes",)))
        self.ignored_collective_attributes = ignored
        self.prefix_collective_contracts = {identifier: self._collective_signature(node,
            ignored[identifier] if identifier in ignored else ()) for identifier, node in prefix._contracts.items()}
        node_bindings, allocation_bindings = (*tuple(node_bindings), *local_bindings), tuple(allocation_bindings)
        for bindings, field in ((node_bindings, "_nodes"), ((*allocation_bindings, *tuple(transient_allocation_bindings)), "_allocations"), (tuple(view_bindings), "_views")):
            targets = tuple(binding[0] for binding in bindings)
            if len(targets) != len(set(targets)):
                raise ValueError("collective source program binding targets must be unique")
            for binding in bindings:
                target, origin = binding[:2]
                owners = tuple(index for index, trace in enumerate(source.traces)
                    if target in getattr(trace, field))
                if len(owners) != 1 or origin not in getattr(prefix.traces[owners[0]], field):
                    raise ValueError("collective source program binding must belong to exactly one matching source/prefix rank")
        # The same constructor already joined and validated this source.
        # Reuse only that local result; do not cache/close either public group.
        if source._recorded_graph is None:
            graph = WorkloadGraph(*source_parts)
            graph.topological_order()
        else:
            graph = source.graph
        self.source, self.prefix, self.graph = source, prefix, graph
        self.rank_programs = tuple(ForwardSourceProgram(trace, current,
            node_bindings=tuple(binding for binding in node_bindings if binding[0] in trace._nodes),
            allocation_bindings=tuple(binding for binding in allocation_bindings if binding[0] in trace._allocations),
            collective_groups=(source, prefix),
            transient_allocation_bindings=tuple(binding for binding in transient_allocation_bindings if binding[0] in trace._allocations),
            view_bindings=tuple(binding for binding in view_bindings if binding[0] in trace._views),
            capture_bindings=capture_bindings)
            for trace, current in zip(source.traces, prefix.traces, strict=True))

    @staticmethod
    def _collective_signature(node, ignored):
        return (node.node_id, node.kind, node.pool_name, node.shape, node.dtype, node.variant, node.rank,
            {name: value for name, value in node.attributes.items() if name not in ignored})

    @staticmethod
    def _collective_geometry_matches(group, identifier):
        """Validate actual typed ports, including the all-reduce alias contract."""
        contract, layout = group._contracts[identifier], group._layouts[identifier]
        operation, storage = contract.attributes["operation"], contract.attributes["symbolic_collective_storage"]
        if (operation not in ("all_gather", "all_reduce")
                or storage not in (("fresh", "inplace") if operation == "all_reduce" else ("fresh",))
                or contract.attributes["symbolic_collective_dim"] != (0 if operation == "all_gather" else None)
                or layout[4] is not None):
            return False
        value = contract.attributes["bytes"]
        if (type(value) is not int or value <= 0
                or any(type(size) is not int or size <= 0 for size in layout[0])
                or value != prod(layout[0]) * DTYPE_BYTES[layout[1]]
                or layout[2] != ((len(group.participants) * layout[0][0], *layout[0][1:])
                    if operation == "all_gather" else layout[0])):
            return False
        output_shape = layout[2] if layout[3] is None else layout[3]
        if (any(type(size) is not int or size <= 0 for size in output_shape)
                or prod(output_shape) != prod(layout[2])
                or (storage == "inplace" and layout[3] is not None)):
            return False
        for trace in group.traces:
            local = trace._nodes[group._submissions[identifier][trace.rank]]
            if any(local.attributes[field] != contract.attributes[field] for field in (
                    "operation", "symbolic_collective_storage", "symbolic_collective_dim", "bytes")):
                return False
            inputs, outputs = local.attributes["symbolic_tensor_inputs"], local.attributes["symbolic_tensor_outputs"]
            if len(inputs) != 1 or len(outputs) != 1:
                return False
            actual_input, actual_output = trace._views[inputs[0]], trace._views[outputs[0]]
            if (actual_input.shape, actual_input.dtype, actual_output.shape, actual_output.dtype, local.dtype) != (
                    layout[0], layout[1], output_shape, layout[1], layout[1]):
                return False
            if storage == "inplace":
                if inputs != outputs:
                    return False
            elif actual_input.allocation_id == actual_output.allocation_id:
                return False
            for view in (actual_input, actual_output):
                stride = view.element_bytes
                for size, actual in zip(reversed(view.shape), reversed(view.strides_bytes), strict=True):
                    if size != 1 and actual != stride:
                        return False
                    stride *= size
        return True

    def bind(self, prefix):
        if not isinstance(prefix, ForwardCollectiveGroup):
            raise TypeError("collective source binding requires its actual dual-rank prefix")
        native_frame = _native_source_scope(self, prefix)
        # Validate the same submissions without constructing a temporary
        # joined DAG that binding discards. Full signatures and rank-prefix
        # contracts are still checked below before rebinding the actual graph.
        reference = prefix._sequences[prefix.participants[0]]
        if any(sequence != reference for sequence in prefix._sequences.values()):
            raise ValueError("symbolic collective group has missing rank submissions")
        local_ids = set()
        for identifier, submissions in prefix._submissions.items():
            if set(submissions) != set(prefix.participants):
                raise ValueError("symbolic collective group has missing rank submissions")
            for trace in prefix.traces:
                local_id = submissions[trace.rank]
                local = trace._nodes[local_id]
                # Preserve the original required-port and iterable boundaries.
                tuple(local.attributes["symbolic_tensor_inputs"])
                tuple(local.attributes["symbolic_tensor_outputs"])
                local_ids.add(local_id)
        for trace in prefix.traces:
            if trace._collective_ids - local_ids:
                raise ValueError("rank trace contains collectives from another group")
        if any(getattr(prefix, name) != getattr(self.prefix, name) for name in (
                "namespace", "participants", "participant_nic_ids", "ring_order", "_explicit_ring_order",
                "_sequences", "_submissions")):
            return None
        geometry_origins = {origin for _, origin in self.collective_geometry_bindings}
        if set(prefix._layouts) != set(self.prefix._layouts):
            return None
        for name, layout in prefix._layouts.items():
            old = self.prefix._layouts[name]
            if name not in geometry_origins:
                if layout != old:
                    return None
            elif (len(layout) != len(old) or tuple(layout[index] for index in (1, 3, 4)) != tuple(old[index] for index in (1, 3, 4))):
                return None
        contracts = {identifier: self._collective_signature(node,
            self.ignored_collective_attributes[identifier] if identifier in self.ignored_collective_attributes else ())
            for identifier, node in prefix._contracts.items()}
        if contracts != self.prefix_collective_contracts:
            return None
        for target, origin, attributes in self.collective_attribute_bindings:
            for name in attributes:
                if name not in prefix._contracts[origin].attributes:
                    return None
                value = prefix._contracts[origin].attributes[name]
                if type(value) is not str or not value or any(
                        name not in trace._nodes[prefix._submissions[origin][trace.rank]].attributes
                        or trace._nodes[prefix._submissions[origin][trace.rank]].attributes[name] != value
                        for trace in prefix.traces):
                    return None
        for _, origin in self.collective_geometry_bindings:
            if not self._collective_geometry_matches(prefix, origin):
                return None
        traces = tuple(program.bind(trace, native_frame=native_frame) for program, trace in zip(self.rank_programs, prefix.traces, strict=True))
        if any(trace is None for trace in traces):
            return None
        result = copy(self.source)
        result.traces = traces
        if native_frame:
            result._layouts = dict(self.source._layouts)
            updates = {}
            for target, origin, attributes in self.collective_attribute_bindings:
                updates[target] = ({}, {name: prefix._contracts[origin].attributes[name] for name in attributes})
            for target, origin in self.collective_geometry_bindings:
                value = prefix._contracts[origin].attributes["bytes"]
                updates[target] = ({}, {"bytes": value})
                result._layouts[target] = prefix._layouts[origin]
            result._contracts = _SourceFrameNodes(MappingProxyType(dict(self.source._contracts)),updates,None)
            joined = _SourceFrameNodes(MappingProxyType({target:self.graph.node(target) for target in updates}),updates,None)
        else:
            result._contracts = dict(self.source._contracts)
            result._layouts = dict(self.source._layouts)
            joined = {}
            for target, origin, attributes in self.collective_attribute_bindings:
                changes = {name: prefix._contracts[origin].attributes[name] for name in attributes}
                contract = result._contracts[target]
                result._contracts[target] = contract.with_updates({}, changes)
                original = self.graph.node(target)
                joined[target] = original.with_updates({}, changes)
            for target, origin in self.collective_geometry_bindings:
                contract = result._contracts[target]
                value = prefix._contracts[origin].attributes["bytes"]
                result._contracts[target] = contract.with_updates({}, {"bytes": value})
                result._layouts[target] = prefix._layouts[origin]
                original = self.graph.node(target)
                joined[target] = original.with_updates({}, {"bytes": value})
        # Every collective node, full submission and edge is retained from the
        # original complete group. Only current per-rank compute nodes change.
        if native_frame:
            result._recorded_graph = _issue_source_frame(self, result, joined)
        else:
            local = {name: node for trace in traces for name, node in trace._nodes.items()
                if name not in trace._collective_ids}
            result._recorded_graph = self.graph.with_nodes(
                local[node.node_id] if node.node_id in local else joined[node.node_id] if node.node_id in joined else node
                for node in self.graph.nodes)
            if type(self) is ForwardCollectiveSourceProgram:
                graph = result._recorded_graph
                if type(graph) is WorkloadGraph:
                    object.__setattr__(graph, "_ep_source_binding", _EPSourceBinding(self, graph, result.traces))
        return result


class _EPSourceBinding:
    """Immutable receipt from one actual complete source-program bind."""

    __slots__ = ("program_reference", "graph_reference", "nodes", "edges", "token_domains", "node_layout")

    def __init__(self, program, graph, traces):
        if type(program) is not ForwardCollectiveSourceProgram or type(graph) is not WorkloadGraph:
            raise TypeError("source receipt requires canonical source program and graph")
        object.__setattr__(self, "program_reference", weakref.ref(program))
        object.__setattr__(self, "graph_reference", weakref.ref(graph))
        object.__setattr__(self, "nodes", graph.nodes)
        object.__setattr__(self, "edges", graph.edges)
        object.__setattr__(self, "node_layout", _lineage_source_node_layout(program, graph))
        domains = []
        for trace in traces:
            if type(trace) is not ForwardTrace or type(trace.rank) is not int:
                domains = None
                break
            captures = trace.source_captures
            if len(captures) != 1:
                domains = None
                break
            capture = captures[0]
            if ("logical_tokens" not in capture or "model_input_tokens" not in capture
                    or type(capture["logical_tokens"]) is not int
                    or type(capture["model_input_tokens"]) is not int):
                domains = None
                break
            domains.append((trace.rank, capture["logical_tokens"], capture["model_input_tokens"]))
        object.__setattr__(self, "token_domains", None if domains is None else tuple(domains))

    def __setattr__(self, name, value):
        raise TypeError("source EP receipt is immutable")


def _ep_source_owner(graph, *, check_source=True):
    """Return the actual binding owner while its exact source receipt holds."""
    if type(graph) is not WorkloadGraph or "_ep_source_binding" not in graph.__dict__:
        return None
    receipt = graph.__dict__["_ep_source_binding"]
    if type(receipt) is _NativeSourceFrame:
        return receipt.owner(check_source=check_source)
    if (type(receipt) is not _EPSourceBinding or receipt.graph_reference() is not graph
            or receipt.nodes is not graph.nodes or receipt.edges is not graph.edges):
        return None
    owner = receipt.program_reference()
    if type(owner) is not ForwardCollectiveSourceProgram:
        return None
    if check_source:
        from . import traced_models as source
        if not source._DENSE_METADATA_SOURCE_CONTRACT.unchanged():
            return None
    return owner


def _ep_source_token_domains(graph):
    """Return native capture pairs copied by the actual source issuer."""
    if _ep_source_owner(graph, check_source=False) is None:
        return None
    return graph.__dict__["_ep_source_binding"].token_domains


_SOURCE_LINEAGE_RECIPES = weakref.WeakKeyDictionary()


def _lineage_binding_recipe(program):
    if (type(program) is not ForwardCollectiveSourceProgram or type(program.graph) is not WorkloadGraph
            or type(program.rank_programs) is not tuple):
        return None
    if any(type(rank) is not ForwardSourceProgram for rank in program.rank_programs):return None
    if any(type(node) is not WorkloadNode or type(node.node_id) is not str for node in program.graph.nodes):return None
    declarations = (tuple(rank.node_bindings for rank in program.rank_programs),
        program.collective_attribute_bindings, program.collective_geometry_bindings)
    if program in _SOURCE_LINEAGE_RECIPES:
        graph_reference, previous, bindings = _SOURCE_LINEAGE_RECIPES[program]
        if graph_reference() is program.graph and len(declarations[0]) == len(previous[0]) and all(current is old
                for current, old in zip(declarations[0], previous[0], strict=True)) and all(
                    current is old for current, old in zip(declarations[1:], previous[1:], strict=True)):
            return bindings
    targets = {node.node_id for node in program.graph.nodes}
    writes = {}
    def write(target, origin, shapes, attrs):
        if (type(target) is not str or type(shapes) is not tuple or type(attrs) is not tuple
                or any(type(name) is not str for name in (*shapes, *attrs)) or target not in targets):
            return False
        if target in writes:
            return False
        writes[target] = (origin, shapes, attrs)
        return True
    for index, rank in enumerate(program.rank_programs):
        if type(rank) is not ForwardSourceProgram or type(rank.node_bindings) is not tuple:
            return None
        for binding in rank.node_bindings:
            if type(binding) is not tuple or len(binding) != 4:
                return None
            target, origin, shapes, attrs = binding
            if type(origin) is not str:
                return None
            if target not in targets:
                if target not in rank.source._collective_ids:
                    return None
                continue
            if not write(target, (index, origin), shapes, attrs):return None
    for binding in program.collective_attribute_bindings:
        if type(binding) is not tuple or len(binding) != 3:return None
        target, origin, attrs = binding
        if type(origin) is not str or not write(target, ("collective", origin), (), attrs):return None
    for binding in program.collective_geometry_bindings:
        if type(binding) is not tuple or len(binding) != 2:return None
        target, origin = binding
        if type(origin) is not str or not write(target, ("collective", origin), (), ("bytes",)):return None
    bindings = tuple((node.node_id, *writes[node.node_id]) for node in program.graph.nodes if node.node_id in writes)
    _SOURCE_LINEAGE_RECIPES[program] = (weakref.ref(program.graph), declarations, bindings)
    return bindings


def _lineage_source_node_layout(program, graph):
    # Receipt payload is an untrusted recipe descriptor. Only the accessor
    # below establishes the actual-node facts from owner-issued lineage.
    bindings = _lineage_binding_recipe(program)
    template = program.graph
    if (bindings is None or len(template.nodes) != len(graph.nodes)
            or template.edges is not graph.edges):return None
    return weakref.ref(template), template.nodes, bindings


def _source_bound_node_layout(graph):
    if "_native_source_frame" in graph.__dict__:
        frame = graph.__dict__["_native_source_frame"]
        if type(frame) is _NativeSourceFrame and frame.owner() is not None:
            return frame.layout
        return None
    if _ep_source_owner(graph) is None or not _node_update_source_unchanged():
        return None
    receipt = graph.__dict__["_ep_source_binding"]
    layout = receipt.node_layout
    if layout is None:
        return None
    template = layout[0]()
    if (type(template) is not WorkloadGraph or template.nodes is not layout[1]
            or template.edges is not graph.edges or template.block_repeats is not graph.block_repeats):
        return None
    # Recheck intrinsic seals, including parent/actual shape+attribute refs;
    # no mutable prefix validator or program-history audit is involved.
    bound = {target: (origin, shapes, attrs) for target, origin, shapes, attrs in layout[2]}
    referents = {}
    for base, actual in zip(template.nodes, graph.nodes, strict=True):
        if base.node_id not in bound:
            if actual is not base:return None
        else:
            updates = _workload_node_field_updates_read(actual)
            origin, shapes, attrs = bound[base.node_id]
            if updates is None or updates[0] is not base or updates[1] != shapes or updates[2] != attrs:return None
            for mapping, fields, kind in ((actual.shape, shapes, "shape"), (actual.attributes, attrs, "attributes")):
                for name in fields:
                    key = (origin, kind, name)
                    value = mapping[name]
                    if key in referents:
                        if referents[key] is not value:return None
                    else:referents[key] = value
    return layout


class ForwardKVCacheGroup:
    """Join an executed virtual forward with externally owned KV DMA effects.

    KVCacheTierManager owns the persistent Host/HBM pages and their retirement.
    These events carry exact per-rank bytes; they do not manufacture activation
    tensors, physical page addresses or an unobserved DMA workspace lifetime.
    """

    def __init__(self, source, *, pool, step_id, transfer_attributes):
        if isinstance(source, ForwardTrace):
            self.traces = (source,)
        elif isinstance(source, (ForwardCollectiveGroup, ForwardPipelineGroup, ForwardActivationGroup)):
            self.traces = source.traces
        else:
            raise TypeError("KV effects require the actual virtual model forward")
        if not isinstance(transfer_attributes, tuple) or not transfer_attributes:
            raise ValueError("KV effects require explicit validated transfer intents")
        if not set(pool.ranks) <= {trace.rank for trace in self.traces}:
            raise ValueError("KV transfer owner ranks are absent from the virtual forward")
        identifiers = tuple(attributes["ready_event"] for attributes in transfer_attributes)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("KV offload ready_event values must be unique")
        graph = source.graph
        incoming = {edge.destination for edge in graph.edges}
        outgoing = {edge.source for edge in graph.edges}
        roots = tuple(node.node_id for node in graph.nodes if node.node_id not in incoming)
        leaves = tuple(node.node_id for node in graph.nodes if node.node_id not in outgoing)
        if not roots or not leaves:
            raise ValueError("KV effects require a nonempty model dependency graph")
        before, after, edges = [], [], list(graph.edges)
        for rank in pool.ranks:
            loads, evicts = [], []
            for attributes in transfer_attributes:
                name = f"{step_id}:kv-offload:{attributes['ready_event']}"
                if len(pool.ranks) > 1:
                    name += f":tp-rank-{rank}"
                node = WorkloadNode(name, WorkloadKind.KV_OFFLOAD, pool.name, {}, rank=rank,
                    attributes=attributes)
                if attributes["direction"] == "load":
                    loads.append(node)
                elif attributes["direction"] == "evict":
                    evicts.append(node)
                else:
                    raise ValueError("KV effects only accept load and evict directions")
            # Each intent must independently be a pre-compute root or a
            # post-compute leaf, as required by the existing completion
            # contract. Copy engines/link resources serialize overlapping
            # transfers; unrelated page intents are not payload dependencies.
            edges.extend(WorkloadEdge(load.node_id, root, "barrier") for load in loads for root in roots)
            edges.extend(WorkloadEdge(leaf, evict.node_id, "barrier") for evict in evicts for leaf in leaves)
            before.extend(loads)
            after.extend(evicts)
        self._graph = WorkloadGraph((*before, *graph.nodes, *after), tuple(edges))
        self._graph.topological_order()
        self._source = source

    @property
    def operator_order_by_rank(self):
        # Capture offsets index the original trace's calls. External DMA
        # nodes have no trace positions and remain outside every capture.
        return self._source.operator_order_by_rank

    @property
    def graph(self):
        return self._graph


class ForwardLocalGroup:
    """One TP rank: the source's collective is an identity, not a transfer."""
    def __init__(self, trace):
        if not isinstance(trace, ForwardTrace):
            raise TypeError("local source group requires a ForwardTrace")
        self.traces = (trace,)
        self.participants = (trace.rank,)

    def call(self, name, tensor, *, pool_name, config, inplace):
        if config is not None or type(inplace) is not bool or not name or not pool_name:
            raise ValueError("TP1 source collective requires explicit absent communication config")
        self.traces[0]._check_tensor(tensor)
        return tensor


class _SourceFrameNodes(Mapping):
    """One immutable actual binding; canonical node objects are inspection views."""
    __slots__ = ("base", "updates", "cache", "launch_policy")
    def __init__(self, base, updates, launch_policy):
        self.base = base
        self.updates = MappingProxyType({name: (MappingProxyType(shape), MappingProxyType(attrs))
            for name, (shape, attrs) in updates.items()})
        self.cache = {}
        self.launch_policy = launch_policy
    def __setattr__(self, name, value):
        if hasattr(self, name): raise TypeError("source frame node binding is immutable")
        object.__setattr__(self, name, value)
    def __len__(self): return len(self.base)
    def __iter__(self): return iter(self.base)
    def __getitem__(self, name):
        if name not in self.updates: return self.base[name]
        if name not in self.cache:
            shape, attrs = self.updates[name]
            self.cache[name] = self.base[name].with_updates(dict(shape), dict(attrs))
        return self.cache[name]


_NATIVE_RANK_LAYOUTS = weakref.WeakKeyDictionary()


def _source_rank_frame(program, updates):
    base_nodes = program.graph.nodes
    current_nodes = tuple(program.source._nodes.values())
    if (len(base_nodes) != len(current_nodes)
            or any(type(actual) is not WorkloadNode or node.node_id != actual.node_id
                for node, actual in zip(base_nodes, current_nodes, strict=True))):
        return None
    if program in _NATIVE_RANK_LAYOUTS:
        graph_reference, previous_nodes, base, policy = _NATIVE_RANK_LAYOUTS[program]
        if (graph_reference() is program.graph and all(current is previous
                for current, previous in zip(current_nodes, previous_nodes, strict=True))):
            return _SourceFrameNodes(base, updates, policy)
    policies = frozenset(node.attributes["launch_policy"] for node in current_nodes if "launch_policy" in node.attributes)
    policy = next(iter(policies)) if len(policies) == 1 else None
    base = MappingProxyType(dict(program.source._nodes))
    _NATIVE_RANK_LAYOUTS[program] = (weakref.ref(program.graph), current_nodes, base, policy)
    return _SourceFrameNodes(base, updates, policy)


class _SourceFrameSequence(Sequence):
    def __init__(self, frame): self.frame = frame
    def __len__(self): return len(self.frame.ids)
    def __getitem__(self, index):
        if isinstance(index, slice): return tuple(self.frame.node(name) for name in self.frame.ids[index])
        return self.frame.node(self.frame.ids[index])


class _SourceFrameLookup(Mapping):
    def __init__(self, frame): self.frame = frame
    def __len__(self): return len(self.frame.ids)
    def __iter__(self): return iter(self.frame.ids)
    def __getitem__(self, name): return self.frame.node(name)


_NATIVE_SOURCE_RECIPES = weakref.WeakKeyDictionary()


class _NativeSourceFrame:
    def __init__(self, program, result, graph, joined, compiled):
        self.program_reference, self.graph_reference = weakref.ref(program), weakref.ref(graph)
        self.template_reference = weakref.ref(program.graph)
        self.recipe_signature = _native_recipe_signature(program)
        self.trace_references = tuple(weakref.ref(trace) for trace in result.traces)
        self.trace_types = tuple(type(trace) for trace in result.traces)
        self.node_maps = tuple(trace._nodes for trace in result.traces)
        self.joined = joined
        self.ids, self.origins, self.layout, self.workload_contract, self.routing_ids, self.storage_rows, self.base_nodes, self.expected_rank_bases = compiled
        self.nodes = _SourceFrameSequence(self)
        self.materialized = None
        self.token_domains = tuple((trace.rank, trace.source_captures[0]["logical_tokens"],
            trace.source_captures[0]["model_input_tokens"]) for trace in result.traces)
    def owner(self, *, check_source=True):
        owner, graph = self.program_reference(), self.graph_reference()
        if (type(owner) is not ForwardCollectiveSourceProgram or owner.graph is not self.template
                or graph is None or graph.__dict__["_ep_source_binding"] is not self
                or not _native_recipe_matches(owner,self.recipe_signature)
                or self.expected_rank_bases is None
                or any(nodes.base is not expected for nodes, expected in zip(
                    self.node_maps, self.expected_rank_bases, strict=True))
                or any(reference() is not None and reference()._nodes is not nodes
                    for reference, nodes in zip(self.trace_references, self.node_maps, strict=True))):
            return None
        if check_source:
            if (type(owner.source) is ForwardCollectiveGroup
                    and all(type(rank) is ForwardSourceProgram for rank in owner.rank_programs)
                    and type(self.trace_types) is tuple
                    and len(self.trace_types) == len(self.trace_references)
                    and all(kind is ForwardTrace for kind in self.trace_types)
                    and all(reference() is None or type(reference()) is ForwardTrace for reference in self.trace_references)
                    and all(type(nodes) is _SourceFrameNodes for nodes in self.node_maps)):
                if not _native_consumer_source_unchanged(): return None
            else:
                from . import traced_models as source
                if not source._DENSE_METADATA_SOURCE_CONTRACT.unchanged(): return None
        return owner
    def node(self, name):
        origin = self.origins[name]
        if origin is None: return self.base_nodes[name]
        if origin == "collective": return self.joined[name]
        return self.node_maps[origin][name]
    def fields(self, name):
        base = self.base_nodes[name]
        origin = self.origins[name]
        if origin is None: return base.shape, base.attributes
        if origin == "collective":
            base = self.joined.base[name]
            shape, attrs = self.joined.updates[name]
            return _OverlayFields(base.shape, shape), _OverlayFields(base.attributes, attrs)
        nodes = self.node_maps[origin]
        if name not in nodes.updates: return base.shape, base.attributes
        shape, attrs = nodes.updates[name]
        return _OverlayFields(base.shape, shape), _OverlayFields(base.attributes, attrs)
    def public_nodes(self):
        if self.materialized is None:
            object.__setattr__(self,"materialized",tuple(self.node(name) for name in self.ids))
        return self.materialized

    @property
    def template(self):
        return self.template_reference()
    def __setattr__(self, name, value):
        if name in self.__dict__: raise TypeError("source frame binding is immutable")
        object.__setattr__(self,name,value)


class _OverlayFields(Mapping):
    def __init__(self, base, updates): self.base, self.updates = base, updates
    def __len__(self): return len(self.base)
    def __iter__(self): return iter(self.base)
    def __getitem__(self, name):
        return self.updates[name] if name in self.updates else self.base[name]


def _issue_source_frame(program, result, joined):
    if program in _NATIVE_SOURCE_RECIPES and not _native_recipe_matches(program,_NATIVE_SOURCE_RECIPES[program][0]):
        del _NATIVE_SOURCE_RECIPES[program]
    if program not in _NATIVE_SOURCE_RECIPES:
        template = program.graph
        bindings = _lineage_binding_recipe(program)
        origins = {}
        for node in template.nodes:
            if node.node_id in joined: origins[node.node_id] = "collective"
            else:
                indices = tuple(index for index, trace in enumerate(result.traces)
                    if node.node_id in trace._nodes and node.node_id not in trace._collective_ids)
                if len(indices) > 1: raise ValueError("source frame has ambiguous rank node")
                origins[node.node_id] = indices[0] if indices else None
        # Only unchanged activation storage may use the compiled projection.
        # Dynamic transient/view bindings retain their ordinary reporter path.
        static_storage = not any(rank.transient_origins for rank in program.rank_programs)
        rows = None if not static_storage else tuple((trace.rank, trace.activation_allocations,
            len(trace._views), MappingProxyType(trace.workspace_capacity_bytes)) for trace in program.source.traces)
        rank_bases = tuple(trace._nodes.base for trace in result.traces)
        static_matches = all(node is template.node(name)
            for trace, base in zip(result.traces, rank_bases, strict=True)
            for name, node in base.items() if name not in trace._collective_ids)
        compiled = (tuple(node.node_id for node in template.nodes), MappingProxyType(origins),
            (weakref.ref(template), template.nodes, bindings),
            (tuple((node.node_id,node.kind.value,node.pool_name,node.rank) for node in template.nodes),
                template.edges, template.block_repeats, template.logical_node_count,template.logical_edge_count),
            tuple(node.node_id for node in template.nodes if node.kind is WorkloadKind.FUSED_MOE), rows,
            MappingProxyType({node.node_id: node for node in template.nodes}),
            rank_bases if static_matches else None)
        _NATIVE_SOURCE_RECIPES[program] = (_native_recipe_signature(program),compiled)
    graph = object.__new__(WorkloadGraph)
    graph.__dict__.update(program.graph.__dict__)
    frame = _NativeSourceFrame(program, result, graph, joined, _NATIVE_SOURCE_RECIPES[program][1])
    object.__setattr__(graph, "_native_source_frame", frame)
    object.__setattr__(graph, "_ep_source_binding", frame)
    object.__setattr__(graph, "_is_node_binding", True)
    object.__setattr__(graph, "_nodes_by_id", _SourceFrameLookup(frame))
    return graph


def _native_recipe_signature(program):
    ranks = (id(program.rank_programs), tuple(weakref.ref(rank) for rank in program.rank_programs))
    return (weakref.ref(program.graph), ranks, tuple((rank.node_bindings,rank.allocation_bindings,
            rank.view_bindings,rank.capture_bindings) for rank in program.rank_programs),
        program.collective_attribute_bindings,program.collective_geometry_bindings)


def _native_recipe_matches(program, signature):
    graph_reference,ranks,bindings,attrs,geometry = signature
    return (program.graph is graph_reference() and id(program.rank_programs) == ranks[0]
        and len(program.rank_programs) == len(ranks[1])
        and all(rank is reference() for rank, reference in zip(program.rank_programs, ranks[1], strict=True))
        and all(all(current is old for current,old in zip((rank.node_bindings,rank.allocation_bindings,
            rank.view_bindings,rank.capture_bindings),binding,strict=True)) for rank,binding in zip(program.rank_programs,bindings,strict=True))
        and program.collective_attribute_bindings is attrs and program.collective_geometry_bindings is geometry)


def _native_source_scope(program, prefix):
    if (type(program) is not ForwardCollectiveSourceProgram or len(prefix.traces) != 2
            or program.collective_geometry_bindings
            or any(rank.transient_origins or any(target not in rank.source._persistent_ids
                for target,origin in rank.allocation_bindings) for rank in program.rank_programs)):
        return False
    for index,trace in enumerate(prefix.traces):
        captures = trace.source_captures
        if (len(captures) != 1 or type(captures[0]["logical_tokens"]) is not int
                or captures[0]["logical_tokens"] <= 0
                or type(captures[0]["model_input_tokens"]) is not int
                or captures[0]["logical_tokens"] > captures[0]["model_input_tokens"]): return False
        if captures[0]["logical_tokens"] != captures[0]["model_input_tokens"]:
            capture = captures[0]
            exact_fields = ("key","logical_tokens","model_input_tokens","runtime_mode","source_stage","source_graph_batch_size","inputs","output")
            if (not set(exact_fields) <= set(capture) or capture["runtime_mode"] != "FULL"
                    or capture["source_stage"] is not None): return False
            for recorded in (program.source.traces[index],program.prefix.traces[index]):
                actual = recorded.source_captures
                if (len(actual) != 1 or not set(exact_fields) <= set(actual[0])
                        or any(capture[name] != actual[0][name] for name in exact_fields)): return False
    if "_native_frame_scope" in program.__dict__:
        signature, eligible = program.__dict__["_native_frame_scope"]
        if _native_recipe_matches(program,signature): return eligible
    attention = tuple(node for node in program.graph.nodes if node.kind is WorkloadKind.FLASH_ATTN)
    eligible = (bool(attention) and all(node.shape["batch"] > 0 and (node.shape["q_len"] == 1
            or ("mathematical_program" in node.attributes
                and node.attributes["mathematical_program"] == "ascend_0_18_full_fia_softmax_v1"
                and "q_lens" in node.attributes and type(node.attributes["q_lens"]) is tuple
                and bool(node.attributes["q_lens"])
                and all(type(length) is int and length == 1 for length in node.attributes["q_lens"]))) for node in attention)
        and all(len(trace.source_captures) == 1 for trace in program.source.traces))
    program._native_frame_scope = (_native_recipe_signature(program),eligible)
    return eligible


def _source_bound_execution_projection(graph):
    if "_native_source_frame" not in graph.__dict__: return None
    frame = graph.__dict__["_native_source_frame"]
    if type(frame) is not _NativeSourceFrame or frame.owner() is None: return None
    routing = []
    for name in frame.routing_ids:
        _, attrs = frame.fields(name)
        routing.append((name,attrs["moe_invocation_id"],attrs["tokens_per_expert"],attrs["routing_provenance"]))
    return frame.workload_contract, tuple(routing)


def _source_bound_storage_projection(source, graph):
    if "_native_source_frame" not in graph.__dict__: return None
    frame = graph.__dict__["_native_source_frame"]
    if (type(frame) is not _NativeSourceFrame or frame.owner() is None or source._recorded_graph is not graph
            or len(source.traces) != len(frame.trace_references) or frame.storage_rows is None
            or any(trace is not reference() for trace,reference in zip(source.traces,frame.trace_references,strict=True))): return None
    rows = tuple((row[0], trace.activation_allocations, len(trace._views),
        MappingProxyType(trace.workspace_capacity_bytes))
        for row, trace in zip(frame.storage_rows, source.traces, strict=True))
    return len(frame.ids), len(graph.edges), rows


def _source_bound_node_sequence(graph):
    if "_native_source_frame" in graph.__dict__:
        frame = graph.__dict__["_native_source_frame"]
        if type(frame) is _NativeSourceFrame and frame.owner() is not None: return frame.nodes
    return graph.nodes


def _bind_native_source_parameters(program, rank_values, collective_values):
    """Execute the already validated source recipe on actual current fields."""
    traces = []
    for rank_program, (updates, allocations, views) in zip(program.rank_programs, rank_values, strict=True):
        result = copy(rank_program.source)
        result._nodes = _source_rank_frame(rank_program, updates)
        if result._nodes is None:
            return None
        result._recorded_graph = (result._nodes, None)
        if not _bind_native_storage(rank_program, result, allocations, views):
            result._allocations, result._views = dict(rank_program.source._allocations), dict(rank_program.source._views)
            rank_program._bind_storage_parameters(result, allocations, views, rank_program.prefix.source_captures)
            _seal_native_bound_storage(rank_program, result)
        traces.append(result)
    result = copy(program.source)
    result.traces = tuple(traces)
    result._layouts = dict(program.source._layouts)
    result._contracts = _SourceFrameNodes(MappingProxyType(dict(program.source._contracts)), collective_values, None)
    joined = _SourceFrameNodes(MappingProxyType({name:program.graph.node(name) for name in collective_values}), collective_values, None)
    result._recorded_graph = _issue_source_frame(program, result, joined)
    return result


# Issued frames read an already recorded graph. Builders retain the complete
# traced_models source guard; consumers validate only their execution surface.
def _native_consumer_source_unchanged():
    if not _node_update_source_unchanged(): return False
    for name, cls, bases, keys, members in _NATIVE_CONSUMER_CLASSES:
        if globals()[name] is not cls or cls.__bases__ != bases: return False
        namespace = vars(cls)
        if cls in (ForwardTrace, ForwardCollectiveGroup, MemoryAllocation) and "__slotnames__" in namespace:
            if type(namespace["__slotnames__"]) is not list or namespace["__slotnames__"]: return False
            current_keys = namespace.keys() - {"__slotnames__"}
        else:
            current_keys = namespace.keys()
        if current_keys != keys: return False
        if any((key in namespace) != present or (present and namespace[key] is not value)
                for key, present, value in members): return False
    valid = (all(globals()[name] is function for name, function in _NATIVE_CONSUMER_GLOBALS)
        and all(function.__code__ is code and function.__defaults__ == defaults
            and function.__kwdefaults__ == kwdefaults
            for function, code, defaults, kwdefaults in _NATIVE_CONSUMER_FUNCTIONS))
    return valid


def _native_consumer_snapshot():
    from types import FunctionType
    declarations = (
        ('_NativeSourceFrame', tuple(vars(_NativeSourceFrame))),
        ('_SourceFrameNodes', tuple(vars(_SourceFrameNodes))),
        ('_SourceFrameSequence', tuple(vars(_SourceFrameSequence))),
        ('_SourceFrameLookup', tuple(vars(_SourceFrameLookup))),
        ('_OverlayFields', tuple(vars(_OverlayFields))),
        ('_NativeStorageOverlay', tuple(vars(_NativeStorageOverlay))),
        ('WorkloadGraph', tuple(vars(WorkloadGraph))),
        ('WorkloadNode', tuple(vars(WorkloadNode))),
        ('MemoryAllocation', tuple(vars(MemoryAllocation))),
        ('Mapping', tuple(vars(Mapping))), ('Sequence', tuple(vars(Sequence))),
        ('ForwardTrace', ('source_capture', 'source_captures', 'workspace_capacity_bytes',
            'activation_allocations', 'event_order', 'allocations', 'views', 'graph',
            '_nodes', '_allocations', '_persistent_ids', '_events', '_views', '_captures', '_capture', '_recorded_graph')),
        ('ForwardCollectiveGroup', ('graph', '_recorded_graph', 'traces')),
        ('ForwardSourceProgram', ('graph', 'source', 'rank_programs', 'node_bindings', 'allocation_bindings', 'view_bindings', 'capture_bindings', 'collective_attribute_bindings', 'collective_geometry_bindings')),
        ('ForwardCollectiveSourceProgram', ('graph', 'source', 'rank_programs', 'node_bindings', 'allocation_bindings', 'view_bindings', 'capture_bindings', 'collective_attribute_bindings', 'collective_geometry_bindings')))
    classes, functions = [], []
    for name, names in declarations:
        cls = globals()[name]; namespace = vars(cls)
        names = tuple(dict.fromkeys((*names, '__getattribute__', '__getattr__', '__setattr__')))
        if cls in (ForwardTrace, ForwardCollectiveGroup, MemoryAllocation):
            # Pickle may have populated this empty CPython cache before import.
            names = tuple(key for key in names if key != "__slotnames__")
        members = tuple((key, key in namespace, namespace[key] if key in namespace else None) for key in names)
        keys = frozenset(key for key in namespace if key != "__slotnames__"
            or cls not in (ForwardTrace, ForwardCollectiveGroup, MemoryAllocation))
        classes.append((name, cls, cls.__bases__, keys, members))
        for _, present, member in members:
            if present and type(member) is FunctionType: functions.append(member)
            elif present and type(member) is property:
                functions.extend(function for function in (member.fget, member.fset, member.fdel) if function is not None)
    names = ('_native_consumer_source_unchanged', '_native_recipe_matches', '_source_bound_node_layout',
        '_source_bound_execution_projection', '_source_bound_storage_projection', '_source_bound_node_sequence',
        '_ep_source_owner', '_ep_source_token_domains', '_node_update_source_unchanged')
    global_functions = tuple((name, globals()[name]) for name in names)
    functions.extend(function for _, function in global_functions)
    return (tuple(classes), global_functions, tuple((function, function.__code__, function.__defaults__,
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)) for function in functions))


_NATIVE_CONSUMER_CLASSES, _NATIVE_CONSUMER_GLOBALS, _NATIVE_CONSUMER_FUNCTIONS = _native_consumer_snapshot()
