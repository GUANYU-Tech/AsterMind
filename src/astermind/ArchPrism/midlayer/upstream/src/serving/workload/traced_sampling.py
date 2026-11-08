"""Source storage program for native sampling, using virtual metadata only.

vLLM0.11/0.30 Sampler.forward/sample and TopKTopPSampler.forward_native,
apply_top_k_top_p_pytorch, random_sample and sample_with_exponential_noise.
Explicit profiles select PyTorch-sort, CUDA Triton filtering or the0.11 worker
with FlashInfer0.4.1 top-k-first sampling, no-logprobs, no-penalties and mixed
greedy/random dummy sampler metadata. Native noise is explicit FP32.
It does not establish that a captured runtime selected this source version.
Unrepresented library workspace, compiled fusion/lifetimes and other
FlashInfer source versions remain open.
No random numbers, sorts, arithmetic values or real tensors are computed.
"""

from functools import partial
from math import prod

from .traced_forward import ForwardTrace
from .traced_operators import SymbolicOperator
from .workload_graph import WorkloadKind


class MTPSourceOps(SymbolicOperator):
    """Versioned metadata-only eager Tensor operations for target/proposer.

    Layouts come from real virtual ports, never computed token values.
    The kernel owner prices the explicit useful typed-payload program;
    backend temporaries, instruction issue and actual launches stay unbound.
    """

    def __init__(self, trace, name, pool_name, *, model_execution_version="vllm_0_11_0"):
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_27_1", "vllm_0_30_0"):
            raise NotImplementedError("MTP eager primitives require an explicit supported source version")
        super().__init__(trace, name, pool_name)
        self.model_execution_version = model_execution_version

    def _node(self, operation, *, rows, width, source_rows, dtype, attributes=None):
        return self.node(WorkloadKind.VECTOR, {"elements": rows * width, "rows": rows,
            "width": width, "source_rows": source_rows}, dtype, "vllm_mtp_eager", {
                "source_component": ("vllm_0_30_symbolic_mtp_eager"
                    if self.model_execution_version == "vllm_0_30_0" else "vllm_0_27_1_symbolic_mtp_eager"
                    if self.model_execution_version == "vllm_0_27_1" else "symbolic_mtp_eager"),
                "model_execution_version": self.model_execution_version,
                "operation": operation, "symbolic_workspace_complete": False,
                **({} if attributes is None else attributes)})

    def _call(self, operation, inputs, *, rows, width, source_rows, dtype, shape, output_dtype,
              out=None, attributes=None):
        node = self._node(operation, rows=rows, width=width, source_rows=source_rows,
            dtype=dtype, attributes=attributes)
        return self.trace.call(node, inputs, shape=shape, dtype=output_dtype, out=out)

    def _require_027(self):
        if self.model_execution_version not in ("vllm_0_27_1", "vllm_0_30_0"):
            raise NotImplementedError("padded MTP source primitives require explicit vLLM0.27.1 or0.30.0")

    def cast_fp32(self, x):
        self._require_027()
        if x.dtype != "bf16" or len(x.shape) != 2:
            raise ValueError("MTP logits conversion requires a BF16 matrix")
        return self._call("cast_fp32", (x,), rows=x.shape[0], width=x.shape[1],
            source_rows=x.shape[0], dtype=x.dtype, shape=x.shape, output_dtype="fp32")

    def prepare_next_padded(self, sampled, discard, backup, *, vocab_size,
                            valid_sampled_counts, discarded_rows):
        """utils.py::eagle_prepare_next_token_padded_kernel, no token values."""
        self._require_027()
        if (sampled.dtype != "int32" or len(sampled.shape) != 2
                or discard.dtype != "bool" or discard.shape != (sampled.shape[0],)
                or backup.dtype != "int32" or backup.shape != (sampled.shape[0],)
                or type(vocab_size) is not int or not 0 < vocab_size <= 2**31 - 1):
            raise ValueError("padded next-token preparation requires typed sampled rows, mask, backup and vocabulary")
        rows, width = sampled.shape
        if (not isinstance(valid_sampled_counts, tuple) or len(valid_sampled_counts) != rows
                or any(type(count) is not int or not 0 <= count <= width for count in valid_sampled_counts)
                or not isinstance(discarded_rows, tuple) or len(discarded_rows) != rows
                or any(type(discarded) is not bool for discarded in discarded_rows)
                or any(discarded and count != 0 for discarded, count in zip(discarded_rows, valid_sampled_counts))):
            raise ValueError("padded MTP next-token preparation requires explicit valid counts and discard scenario")
        node = self._node("prepare_next_padded", rows=rows, width=width, source_rows=rows,
            dtype="int32", attributes={"vocab_size": vocab_size,
                "valid_sampled_counts": valid_sampled_counts, "discarded_rows": discarded_rows})
        return self.trace.call_outputs(node, (sampled, discard, backup), outputs=(
            ("next_token_ids", (rows,), "int32"), ("valid_sampled_tokens_count", (rows,), "int32")))

    def prepare_inputs_padded(self, prefix, count, query_start):
        """utils.py::eagle_prepare_inputs_padded_kernel's two INT32 outputs."""
        self._require_027()
        if (prefix.dtype != "int32" or len(prefix.shape) != 1 or count.dtype != "int32"
                or count.shape != prefix.shape or query_start.dtype != "int32"
                or query_start.shape != (prefix.shape[0] + 1,)):
            raise ValueError("padded MTP preparation requires INT32 prefix/count and B+1 query starts")
        rows = prefix.shape[0]
        node = self._node("prepare_inputs_padded", rows=rows, width=1, source_rows=rows, dtype="int32")
        return self.trace.call_outputs(node, (prefix, count, query_start), outputs=(
            ("token_indices_to_sample", (rows,), "int32"), ("num_rejected_tokens", (rows,), "int32")))

    def step_metadata(self, positions, block_table, seq_lens, *, out_positions, out_slots,
                      block_size, max_model_len):
        """utils.py:26–94: write positions/slots and mutate sequence lengths."""
        self._require_027()
        if (positions.dtype != "int64" or len(positions.shape) != 1
                or block_table.dtype != "int32" or len(block_table.shape) != 2
                or block_table.shape[0] != positions.shape[0] or block_table.shape[1] < 1
                or seq_lens.dtype != "int32" or seq_lens.shape != positions.shape
                or any(out.dtype != "int64" or out.shape != positions.shape for out in (out_positions, out_slots))
                or out_positions.view.allocation_id == out_slots.view.allocation_id
                or any(type(value) is not int or value <= 0 for value in (block_size, max_model_len))):
            raise ValueError("MTP step metadata requires typed positions/block table/lengths and distinct INT64 outputs")
        rows = positions.shape[0]
        node = self._node("step_metadata", rows=rows, width=1, source_rows=rows, dtype="int64",
            attributes={"block_table_columns": block_table.shape[1], "block_size": block_size,
                "max_model_len": max_model_len})
        return self.trace.call(node, (positions, block_table, seq_lens), shape=positions.shape, dtype="int64",
            out=out_positions, writes=(out_slots, seq_lens))

    def copy(self, x, out=None):
        if out is not None and (out.shape != x.shape or out.dtype != x.dtype):
            raise ValueError("MTP copy requires matching typed extents")
        if out is not None and out.view.allocation_id == x.view.allocation_id:
            raise NotImplementedError("MTP copy requires independent source/destination backing")
        return self._call("copy", (x,), rows=x.shape[0], width=prod(x.shape[1:]),
            source_rows=x.shape[0], dtype=x.dtype, shape=x.shape, output_dtype=x.dtype, out=out)

    def gather(self, x, indices, *, selected_indices):
        rows = indices.shape[0]
        if (indices.shape != (rows,) or indices.dtype not in ("int32", "int64")
                or not isinstance(selected_indices, tuple) or len(selected_indices) != rows
                or any(type(i) is not int or not 0 <= i < x.shape[0] for i in selected_indices)):
            raise ValueError("MTP gather requires explicit typed in-bounds indices")
        return self._call("gather", (x, indices), rows=rows, width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=(rows, *x.shape[1:]), output_dtype=x.dtype,
            attributes={"index_bytes": indices.view.element_bytes, "selected_indices": selected_indices})

    def scatter(self, x, indices, out, *, selected_indices):
        rows = x.shape[0]
        if (x.shape != (rows,) or x.dtype != "int32" or indices.shape != (rows,)
                or indices.dtype not in ("int32", "int64") or out.dtype != "int32" or len(out.shape) != 1):
            raise ValueError("MTP ID scatter requires scalar INT32 IDs and typed row indices")
        if (not isinstance(selected_indices, tuple) or len(selected_indices) != rows
                or any(type(i) is not int or not 0 <= i < out.shape[0] for i in selected_indices)
                or len(set(selected_indices)) != rows):
            raise ValueError("MTP ID scatter requires distinct valid destination indices")
        return self._call("scatter", (x, indices), rows=rows, width=1, source_rows=out.shape[0],
            dtype="int32", shape=out.shape, output_dtype="int32", out=out,
            attributes={"index_bytes": indices.view.element_bytes, "selected_indices": selected_indices})

    def fill(self, out, value=0):
        return self._call("fill", (), rows=out.shape[0], width=prod(out.shape[1:]), source_rows=out.shape[0],
            dtype=out.dtype, shape=out.shape, output_dtype=out.dtype, out=out, attributes={"source_constant": value})

    def unary(self, x, operation="unary", *, inplace=False, expression=None, scalar=None):
        if operation in ("compare", "unary"):
            expressions = (("not_equal", "less", "greater_equal", "greater") if operation == "compare" else
                ("add", "subtract", "clamp_min", "floor_divide", "multiply", "remainder"))
            if expression not in expressions or type(scalar) is not int:
                raise ValueError("MTP integer source expression requires its explicit operation and scalar")
        if operation == "compare":
            output_dtype, shape = "bool", x.shape
        elif operation == "sum":
            if x.dtype != "bool" or len(x.shape) != 2:
                raise ValueError("MTP mask sum requires boolean rows")
            output_dtype, shape = "int64", (x.shape[0],)
        elif operation == "argmax":
            if x.dtype not in ("bf16", "fp32") or len(x.shape) != 2:
                raise ValueError("MTP argmax requires floating-point logit rows")
            output_dtype, shape = "int64", (x.shape[0],)
        elif operation == "cast_ids":
            if x.dtype != "int64" or len(x.shape) != 1:
                raise ValueError("MTP draft-ID cast requires INT64 scalar IDs")
            output_dtype, shape = "int32", x.shape
        elif operation == "unary":
            if x.dtype not in ("int32", "int64"):
                raise ValueError("MTP integer unary operation requires INT32/INT64")
            output_dtype, shape = x.dtype, x.shape
        else:
            raise ValueError("unsupported MTP unary source operation")
        if inplace and (shape != x.shape or output_dtype != x.dtype):
            raise ValueError("MTP dtype/shape change cannot alias its input")
        return self._call(operation, (x,), rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=shape, output_dtype=output_dtype, out=x if inplace else None,
            attributes={} if expression is None else {"source_expression": expression, "source_scalar": scalar})

    def binary(self, x, y, operation="subtract", *, inplace=False):
        if type(inplace) is not bool:
            raise TypeError("MTP binary inplace requires an explicit boolean")
        if x.shape != y.shape:
            raise ValueError("MTP binary inputs require equal shapes")
        if operation in ("promote_subtract", "promote_add"):
            if x.dtype != "int32" or y.dtype != "int64":
                raise ValueError("MTP promoted arithmetic requires INT32 and INT64 operands")
            dtype = "int64"
        elif operation == "logical_and":
            if x.dtype != "bool" or y.dtype != "bool":
                raise ValueError("MTP logical conjunction requires boolean masks")
            dtype = "bool"
        elif operation == "subtract":
            if x.dtype != y.dtype or x.dtype not in ("int32", "int64"):
                raise ValueError("MTP integer binary operation requires matching integer types")
            dtype = x.dtype
        else:
            raise ValueError("unsupported MTP binary source operation")
        if inplace and dtype != x.dtype:
            raise ValueError("MTP promoted binary output cannot alias its input storage")
        return self._call(operation, (x, y), rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=x.shape, output_dtype=dtype, out=x if inplace else None)

    def where(self, mask, x, y=None):
        if mask.shape != x.shape or mask.dtype != "bool" or x.dtype not in ("int32", "int64"):
            raise ValueError("MTP where requires matching boolean condition and integer values")
        promote = y is not None and x.dtype == "int64" and y.dtype == "int32"
        if y is not None and (y.shape != x.shape or (y.dtype != x.dtype and not promote)):
            raise ValueError("MTP where branches must share integer dtype and shape")
        # The two-input form is source torch.where(mask, 0, x).
        return self._call("where_zero" if y is None else "where_promote" if promote else "where",
            (mask, x) if y is None else (mask, x, y),
            rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0], dtype=x.dtype,
            shape=x.shape, output_dtype=x.dtype)

    def concat(self, values):
        if not values or any(len(x.shape) != 1 or x.dtype != values[0].dtype for x in values):
            raise ValueError("MTP prefix concat requires matching scalar vectors")
        rows = sum(x.shape[0] for x in values)
        return self._call("concat", values, rows=rows, width=1, source_rows=rows,
            dtype=values[0].dtype, shape=(rows,), output_dtype=values[0].dtype)

    def masked_fill(self, x, mask, *, masked_rows):
        if x.dtype not in ("int32", "int64") or mask.shape != (x.shape[0],) or mask.dtype != "bool":
            raise ValueError("MTP masked fill requires integer rows and boolean row mask")
        return self._call("masked_fill", (mask,), rows=x.shape[0], width=prod(x.shape[1:]), source_rows=x.shape[0],
            dtype=x.dtype, shape=x.shape, output_dtype=x.dtype, out=x, attributes={"masked_rows": masked_rows})

    def row_gather(self, x, indices):
        if len(x.shape) != 2 or x.dtype != "int32" or indices.shape != (x.shape[0],) or indices.dtype != "int64":
            raise ValueError("MTP row gather requires INT32 rows and INT64 per-row indices")
        return self._call("row_gather", (x, indices), rows=x.shape[0], width=x.shape[1], source_rows=x.shape[0],
            dtype=x.dtype, shape=(x.shape[0],), output_dtype=x.dtype)

    def stack(self, values):
        if (not isinstance(values, tuple) or len(values) < 2 or len(values[0].shape) != 1
                or any(v.shape != values[0].shape or v.dtype != "int64" for v in values)):
            raise ValueError("MTP stack requires multiple matching INT64 draft-ID vectors")
        rows = values[0].shape[0]
        return self._call("stack", values, rows=rows, width=len(values), source_rows=rows,
            dtype="int64", shape=(rows, len(values)), output_dtype="int64")

    def rejection(self, out, prefix, draft, target, bonus, *, verified, accepted):
        rows = out.shape[0]
        if (out.dtype != "int32" or len(out.shape) != 2 or prefix.shape != (rows,) or prefix.dtype != "int32"
                or draft.dtype != "int32" or target.dtype != "int64" or draft.shape != target.shape
                or bonus.shape != (rows,) or bonus.dtype != "int32" or draft.shape != (sum(verified),)):
            raise ValueError("MTP rejection requires typed prefix/draft/argmax/bonus ports")
        return self._call("rejection", (prefix, draft, target, bonus), rows=rows, width=out.shape[1],
            source_rows=draft.shape[0], dtype="int32", shape=out.shape, output_dtype="int32", out=out,
            attributes={"verified_tokens": verified, "accepted_tokens": accepted})

    def h2d(self, out):
        node = self.node(WorkloadKind.MEMCPY, {}, None, None, {"algorithm": "dma", "direction": "h2d",
            "link": "pcie", "bytes": prod(out.shape) * out.view.element_bytes})
        return self.trace.call(node, (), shape=out.shape, dtype=out.dtype, out=out)


def sample_padded_raw_logits(ops, logits, *, prefix, previous_draft, target_indices,
                             bonus_indices, verified, accepted, target_values, bonus_values):
    """Shared0.27/0.30 all-greedy rejection with raw-logit preservation.

    This is the source sampler chain used by both adapted MLA families;
    token values/acceptance remain caller-declared inputs, never computed.
    """
    from .traced_operators import SymbolicGreedy
    bonus_logits = ops.gather(logits, bonus_indices, selected_indices=bonus_values)
    bonus_raw_logits = ops.cast_fp32(bonus_logits)
    bonus_ids = SymbolicGreedy(ops.trace, "target-bonus-sampler", ops.pool_name)(bonus_logits,
        model_execution_version=ops.model_execution_version).view_as_shape(len(verified))
    raw_target_logits = ops.cast_fp32(ops.gather(logits, target_indices, selected_indices=target_values))
    target_logits = ops.copy(raw_target_logits)
    target_argmax = ops.unary(target_logits, "argmax")
    sampled = ops.fill(ops.trace.tensor("target.rejection-output", (len(verified), max(verified) + 1), "int32"), value=-1)
    return ops.rejection(sampled, prefix, previous_draft, target_argmax, bonus_ids,
        verified=verified, accepted=accepted)


def prepare_padded_drafter(ops, sampled, discard, backup, query_start, *, vocab_size,
                           valid_sampled_counts, discarded_rows, prefix):
    """Shared legacy padded next-token and retained-query preparation."""
    next_ids, valid_count = ops.prepare_next_padded(sampled, discard, backup,
        vocab_size=vocab_size, valid_sampled_counts=valid_sampled_counts, discarded_rows=discarded_rows)
    if prefix is None:
        batch = len(valid_sampled_counts)
        sample_indices = ops.unary(query_start.narrow(0, 1, batch), expression="subtract", scalar=1)
        rejected = None
    else:
        sample_indices, rejected = ops.prepare_inputs_padded(prefix, valid_count, query_start)
    return next_ids, sample_indices, rejected


class _VLLMWorkerStorage:
    """Version-bound known device buffers of pure-text CP1/MTP-off workers.

    Adapt GPUModelRunner.__init__ and InputBatch's unconditional sampling
    arrays. CpuGpuBuffer owns one device allocation; its CPU/pinned/NumPy
    storage is not another HBM allocation. Block tables, backend metadata,
    optional multimodal/speculative buffers and library state stay separate.
    No source module import, real tensor, initialization or numerical values.
    """

    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype, model_execution_version):
        if not isinstance(trace, ForwardTrace):
            raise TypeError("worker storage requires a ForwardTrace")
        if any(type(value) is not int or value <= 0 for value in (tokens, max_requests, inputs_embeds_size)):
            raise ValueError("worker storage requires explicit positive token/request/embedding capacities")
        if dtype not in ("bf16", "fp16"):
            raise NotImplementedError("worker storage requires a supported 16-bit pure-text source")
        if model_execution_version not in ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_30_0"):
            raise NotImplementedError("worker storage requires its version-bound constructor")
        t, r = tokens, max_requests
        layouts = {
            "input_ids": ((t,), "int32"), "positions": ((t,), "int64"),
            "query_start_loc": ((r + 1,), "int32"), "seq_lens": ((r,), "int32"),
            "inputs_embeds": ((t, inputs_embeds_size), dtype), "is_token_ids": ((t,), "bool")}
        if model_execution_version == "vllm_0_11_0":
            # v0.11 GPUModelRunner.__init__: no0.30 req_indices/query_pos
            # buffers; discard indices and accepted counts are both INT64.
            layouts.update({"discard_request_indices": ((r,), "int64"),
                "num_decode_draft_tokens": ((r,), "int32"),
                "num_accepted_tokens": ((r,), "int64")})
        else:
            #0.24 and0.30 share these16 unconditional device constructors.
            layouts = {
                "input_ids": layouts["input_ids"], "positions": layouts["positions"],
                "query_start_loc": layouts["query_start_loc"], "seq_lens": layouts["seq_lens"],
                "num_computed_tokens": ((r,), "int32"), "prev_num_draft_tokens": ((r,), "int32"),
                "req_indices": ((t,), "int64"), "prev_positions": ((r,), "int64"),
                "num_scheduled_tokens": ((r,), "int32"), "encoder_seq_lens": ((r,), "int32"),
                "inputs_embeds": ((t, inputs_embeds_size), dtype), "is_token_ids": ((t,), "bool"),
                "discard_request_mask": ((r,), "bool"), "num_decode_draft_tokens": ((r,), "int32"),
                "num_accepted_tokens": ((r,), "int32"), "query_pos": ((max(r + 1, t),), "int64")}
        self.buffers = {role: trace.tensor("profile." + role, shape, item_dtype, persistent=True)
            for role, (shape, item_dtype) in layouts.items()}
        self.input_batch_buffers = {role: trace.tensor("profile.input_batch." + role, (r,), item_dtype, persistent=True)
            for role, item_dtype in (("temperature", "fp32"), ("top_p", "fp32"), ("top_k", "int32"),
                ("frequency_penalties", "fp32"), ("presence_penalties", "fp32"), ("repetition_penalties", "fp32"))}
        self.input_ids, self.positions = self.buffers["input_ids"], self.buffers["positions"]


class VLLM030WorkerStorage(_VLLMWorkerStorage):
    """Known v0.30 device constructors; unknown runtime storage stays outside."""
    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype):
        super().__init__(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size,
            dtype=dtype, model_execution_version="vllm_0_30_0")


class VLLM011WorkerStorage(_VLLMWorkerStorage):
    """v0.11 GPUModelRunner/InputBatch unconditional device constructors.

    Bound to local vLLM0.11 gpu_model_runner.py:345–368 and
    gpu_input_batch.py:150–212. CPU/pinned arrays are not HBM copies.
    Block tables, compiler/library workspace and allocator state are uncovered.
    """
    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype):
        super().__init__(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size,
            dtype=dtype, model_execution_version="vllm_0_11_0")


class VLLM024WorkerStorage(_VLLMWorkerStorage):
    """v0.24 GPUModelRunner:721–801 device buffers; CPU sides excluded."""
    def __init__(self, trace, *, tokens, max_requests, inputs_embeds_size, dtype):
        super().__init__(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size,
            dtype=dtype, model_execution_version="vllm_0_24_0")


def vllm030_worker_storage_bytes(*, tokens, max_requests, inputs_embeds_size, dtype):
    """Known source residency using the same descriptors, not a peak formula.

    Intended for an explicitly before-profile base policy. These descriptors
    alone do not establish complete worker memory or an actual provider.
    """
    trace = ForwardTrace()
    owner = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
        inputs_embeds_size=inputs_embeds_size, dtype=dtype)
    return {a.allocation_id: a.size_bytes for a in trace.allocations}


def vllm011_worker_storage_bytes(*, tokens, max_requests, inputs_embeds_size, dtype):
    """Known0.11 constructor storage only; no complete-runtime assertion."""
    trace = ForwardTrace()
    owner = VLLM011WorkerStorage(trace, tokens=tokens, max_requests=max_requests,
        inputs_embeds_size=inputs_embeds_size, dtype=dtype)
    return {a.allocation_id: a.size_bytes for a in trace.allocations}


class StorageOnlyOps(SymbolicOperator):
    """Allocation/alias contracts for source operators without timing claims."""
    def operation(self, operation, inputs, *, shape, dtype, out=None, writes=()):
        node = self.node(WorkloadKind.VECTOR, {"elements": prod(shape)}, dtype,
            "symbolic_storage_only", {"source_operation": operation})
        return self.trace.call(node, inputs, shape=shape, dtype=dtype, out=out, writes=writes)

    def cast(self, x, dtype):
        if x.dtype == dtype:
            return x
        return self.operation("to", (x,), shape=x.shape, dtype=dtype)

    def sort(self, x):
        node = self.node(WorkloadKind.VECTOR, {"elements": prod(x.shape)}, x.dtype,
            "symbolic_storage_only", {"source_operation": "sort"})
        return self.trace.call_outputs(node, (x,), outputs=(
            ("values", x.shape, x.dtype), ("indices", x.shape, "int64")))

    def inplace(self, operation, x, *inputs):
        return self.operation(operation, (x, *inputs), shape=x.shape, dtype=x.dtype, out=x)

    def unary(self, operation, x, *, dtype=None, shape=None):
        return self.operation(operation, (x,), shape=x.shape if shape is None else shape,
            dtype=x.dtype if dtype is None else dtype)

    def compare(self, operation, x, y):
        # The source profile compares a [B,V] tensor with [B,1].
        if len(x.shape) != 2 or y.shape not in (x.shape, (x.shape[0], 1)):
            raise NotImplementedError("native source comparison requires its explicit row broadcast")
        return self.operation(operation, (x, y), shape=x.shape, dtype="bool")


class TritonSamplingStorageCache:
    """Version-bound0.24/0.30 cache allocations; no kernel execution.

    One cache belongs to one physical rank/trace. Source constants are the
    two 200-entry FP32 lookup tables; only0.30 adds split [64*32,4] plus
    [64,5,32,8,3]. Per-program buffers are keyed by vocabulary and grow to
    min(next_power_of_2(min(SM,B)),SM), retaining their full backing on a
    smaller view. Cache creation and replacement follow Python ownership.
    """
    def __init__(self, trace, *, num_compute_units, model_execution_version="vllm_0_30_0"):
        if not isinstance(trace, ForwardTrace):
            raise TypeError("sampler cache requires ForwardTrace")
        if type(num_compute_units) is not int or num_compute_units <= 0:
            raise ValueError("Triton sampler needs an explicit positive SM count")
        if model_execution_version not in ("vllm_0_24_0", "vllm_0_30_0"):
            raise NotImplementedError("Triton sampler cache requires its0.24 or0.30 source")
        self.model_execution_version = model_execution_version
        self.trace = trace
        self.num_compute_units = num_compute_units
        self._buffers = {}
        self._tables = None
        self._split = None

    def _allocate(self, role, shape):
        return self.trace.tensor(self.trace.unique_name("sampler.triton." + role),
            shape, "fp32", persistent=True)

    def apply(self, ops, logits, k, p):
        for tensor in (logits, k, p):
            self.trace._check_tensor(tensor)
        if ops.trace is not self.trace or logits.dtype != "fp32" or len(logits.shape) != 2:
            raise ValueError("Triton storage cache requires its own FP32 row logits")
        rows, vocab = logits.shape
        if k.shape != (rows,) or p.shape != (rows,):
            raise ValueError("Triton sampling metadata must match its row count")
        if self.model_execution_version == "vllm_0_24_0" and rows < 8:
            #0.24 apply_top_k_top_p uses the in-place PyTorch sort path
            # below eight rows even when HAS_TRITON is true.
            return _apply_top_k_top_p(ops, logits, k, p)
        if logits.view.strides_bytes[-1] != logits.view.element_bytes:
            logits = ops.unary("contiguous_last_axis", logits)
        k_ptr, p_ptr = ops.cast(k, "int32"), ops.cast(p, "fp32")
        programs = min(self.num_compute_units, rows)
        buffer = self._buffers[vocab] if vocab in self._buffers else None
        if buffer is None or buffer.shape[0] < programs:
            size = min(1 << (programs - 1).bit_length(), self.num_compute_units)
            buffer = self._allocate("buffer", (size, vocab))
            self._buffers[vocab] = buffer
        if buffer.shape[0] > programs:
            buffer = self.trace.alias(buffer, self.trace.unique_name("sampler.triton.buffer-view"),
                shape=(programs, vocab), strides_bytes=buffer.view.strides_bytes)
        if self._tables is None:
            self._tables = (self._allocate("normal-cdf-table", (200,)),
                self._allocate("percentile-table", (200,)))
        ops.operation("triton_topk_topp", (logits, buffer, *self._tables, k_ptr, p_ptr),
            shape=logits.shape, dtype=logits.dtype, out=logits, writes=(buffer,))
        # CUDA+p enables the split pipeline for B<=64 even with active k;
        # value-dependent work inside the kernels does not skip allocation.
        if self.model_execution_version == "vllm_0_30_0" and rows <= 64:
            if self._split is None:
                self._split = (self._allocate("split-stats", (64 * 32, 4)),
                    self._allocate("split-parts", (64, 5, 32, 8, 3)))
            stats, parts = self._split
            ops.operation("triton_topp_split_stats", (logits, k_ptr, p_ptr),
                shape=stats.shape, dtype=stats.dtype, out=stats)
            for round_index in range(5):
                ops.operation("triton_topp_split_step_" + str(round_index),
                    (logits, stats, parts, k_ptr, p_ptr), shape=parts.shape, dtype=parts.dtype, out=parts)
            ops.operation("triton_topp_split_mask", (logits, stats, parts, k_ptr, p_ptr),
                shape=logits.shape, dtype=logits.dtype, out=logits)
        return logits

    def clear(self):
        """Drop source cache ownership; external aliases retain their backing."""
        self._buffers.clear()
        self._tables = None
        self._split = None


def _apply_top_k_top_p(ops, logits, k, p, *, inplace_scatter=True):
    """Allocation-faithful adaptation of apply_top_k_top_p_pytorch's path."""
    logits_sort, logits_idx = ops.sort(logits)
    top_k_mask = ops.unary("vocab_minus_k", ops.cast(k, "int64"))
    top_k_mask = ops.operation("gather", (logits_sort, top_k_mask.view_as_shape(k.shape[0], 1)),
        shape=(k.shape[0], 1), dtype=logits.dtype)
    top_k_mask = ops.compare("less_than", logits_sort, top_k_mask)
    ops.inplace("masked_fill", logits_sort, top_k_mask)

    probs_sort = ops.unary("softmax", logits_sort)
    probs_sum = ops.inplace("cumsum_out", probs_sort)
    top_p_mask = ops.compare("less_equal", probs_sum,
        ops.unary("one_minus_p", p.view_as_shape(p.shape[0], 1)))
    ops.inplace("mask_last_column_false", top_p_mask)
    ops.inplace("masked_fill", logits_sort, top_p_mask)
    if inplace_scatter:
        return ops.inplace("scatter", logits, logits_idx, logits_sort)
    # v0.11 apply_top_k_top_p returns logits_sort.scatter(...), retaining
    # the incoming logits in its caller while materializing a fresh result.
    return ops.operation("scatter", (logits_sort, logits_idx, logits_sort), shape=logits.shape, dtype=logits.dtype)


def _random_sample(ops, probs):
    q = ops.unary("empty_exponential_noise", probs)
    ops.inplace("exponential_random", q)
    # Per-request generator overwrite changes values, not the allocation.
    scores = ops.inplace("divide", probs, q)
    return ops.unary("argmax", scores, dtype="int64", shape=(probs.shape[0],))


def _forward_native(ops, logits, k, p, filter_logits):
    logits = filter_logits(ops, logits, k, p)
    probs = ops.unary("softmax", logits)
    return _random_sample(ops, probs)


def _flashinfer_041_top_k_first(ops, logits, k, p):
    """FlashInfer 0.4.1 sampling.py's default logits path, storage only.

    top_k_mask_logits converts the row thresholds to INT32 inside its
    helper, returns a fresh FP32 masked matrix, then torch.softmax creates
    another matrix while masked_logits remains live. The top-p custom op
    returns INT32 rows. This path does not call flashinfer.softmax or its
    cached workspace. Generator state tensors are CPU objects, not HBM.
    This explicitly selected source version does not identify historical
    logs which merely name FlashInfer without a library version.
    """
    def top_k_mask_logits():
        thresholds = ops.cast(k, "int32")
        return ops.operation("flashinfer_0_4_1_top_k_mask_logits", (logits, thresholds),
            shape=logits.shape, dtype=logits.dtype)

    masked_logits = top_k_mask_logits()
    probs = ops.unary("softmax", masked_logits)
    return ops.operation("flashinfer_0_4_1_top_p_sampling", (probs, p),
        shape=(logits.shape[0],), dtype="int32")


def _sample(ops, logits, temperature, k, p, random_sampler):
    rows = logits.shape[0]
    greedy_sampled = ops.unary("argmax", logits, dtype="int64", shape=(rows,))

    def apply_temperature():
        condition = ops.unary("temperature_less_than_eps", temperature, dtype="bool")
        adjusted = ops.operation("where_temperature", (condition, temperature), shape=temperature.shape, dtype=temperature.dtype)
        return ops.inplace("divide_by_temperature", logits, adjusted.view_as_shape(rows, 1))

    logits = apply_temperature()
    random_sampled = random_sampler(ops, logits, k, p)
    condition = ops.unary("temperature_less_than_eps", temperature, dtype="bool")
    return ops.inplace("where_reuse_greedy", greedy_sampled, condition, random_sampled)


def _sampling_filter(trace, source_profile, triton_cache):
    if source_profile in ("vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise",
            "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise"):
        if triton_cache is not None:
            raise ValueError("PyTorch sort does not consume Triton cache state")
        filter_logits = (_apply_top_k_top_p if source_profile == "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise"
            else partial(_apply_top_k_top_p, inplace_scatter=False))
    elif source_profile == "vllm_0_30_triton_mixed_no_logprobs_fp32_noise":
        if (not isinstance(triton_cache, TritonSamplingStorageCache) or triton_cache.trace is not trace
                or triton_cache.model_execution_version != "vllm_0_30_0"):
            raise ValueError("Triton sampler requires explicitly owned cache state")
        filter_logits = triton_cache.apply
    elif source_profile == "vllm_0_24_native_triton_or_sort_mixed_no_logprobs_fp32_noise":
        if (not isinstance(triton_cache, TritonSamplingStorageCache) or triton_cache.trace is not trace
                or triton_cache.model_execution_version != "vllm_0_24_0"):
            raise ValueError("0.24 native sampler requires its explicitly versioned cold cache")
        filter_logits = triton_cache.apply
    else:
        raise NotImplementedError("native sampler storage requires a fully declared source profile")
    return filter_logits


def _sampling_program(trace, source_profile, triton_cache):
    if source_profile == "vllm_0_11_flashinfer_0_4_1_top_k_first_mixed_no_logprobs":
        if triton_cache is not None:
            raise ValueError("FlashInfer top-k-first does not consume Triton cache state")
        return _flashinfer_041_top_k_first
    return partial(_forward_native, filter_logits=_sampling_filter(trace, source_profile, triton_cache))


def native_sampler_storage_forward(trace, logits, *, source_profile, temperature, top_k, top_p,
                                   pool_name="main", triton_cache=None):
    """Return a symbolic INT32 sampler output while tracing real source locals.

    All sampler metadata is explicit. The profile covers nonempty top-k/top-p,
    FP32 noise and the raw-logprobs/no-requested-logprobs branch. No output
    token value is computed. Storage-only nodes must not be timed by lowering.
    """
    random_sampler = _sampling_program(trace, source_profile, triton_cache)
    if (logits.dtype not in ("bf16", "fp16") or len(logits.shape) != 2
            or logits.dtype == "fp16" and source_profile not in (
                "vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise",
                "vllm_0_30_pytorch_sort_mixed_no_logprobs_fp32_noise")):
        raise NotImplementedError("native sampler storage expects supported 16-bit row logits; FP16 requires PyTorch-sort")
    rows = logits.shape[0]
    for value in (temperature, top_k, top_p):
        trace._check_tensor(value)
        if value.shape != (rows,):
            raise ValueError("sampling metadata must have one value per row")
    if temperature.dtype != "fp32" or top_p.dtype != "fp32" or top_k.dtype != "int64":
        raise ValueError("dummy sampler source requires FP32 temperature/top-p and INT64 top-k")
    ops = StorageOnlyOps(trace, "native-sampler", pool_name)
    # Both bound Sampler.forward versions use logits.to(torch.float32).
    # The model head's 16-bit dtype is retained by its caller; this cast
    # allocates fresh FP32 logits, with all later sort/noise storage FP32.
    logits = ops.cast(logits, "fp32")
    sampled = _sample(ops, logits, temperature, top_k, top_p, random_sampler)
    sampled = ops.cast(sampled, "int64")
    sampled = ops.cast(sampled, "int32")
    return sampled.view_as_shape(rows, 1)


def worker_profile_storage_forward(trace, model_forward, compute_logits, *,
                                   input_ids, positions, query_lengths, sampler_profile, pool_name="main",
                                   triton_cache=None, drafter_dummy_run=None, model_execution_version="vllm_0_30_0"):
    """Adapt version-bound profile_run/_dummy_run/_dummy_sampler_run scopes.

    Model and head are supplied symbolic callables; there is no model-specific
    peak formula. The caller retains both full and gathered hidden states
    while the sampler replaces its own hidden input with rand_like. First
    sampler output remains live across0.24/0.30's second native warmup;0.11 only
    calls its sampler once. No random
    values, logits, probabilities or tokens are generated.
    """
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_24_0", "vllm_0_30_0"):
        raise NotImplementedError("worker profile requires its version-bound source")
    if ((model_execution_version == "vllm_0_11_0")
            != (sampler_profile in ("vllm_0_11_pytorch_sort_mixed_no_logprobs_fp32_noise",
                "vllm_0_11_flashinfer_0_4_1_top_k_first_mixed_no_logprobs"))):
        raise ValueError("worker and sampler source versions disagree")
    if ((model_execution_version == "vllm_0_24_0")
            != (sampler_profile == "vllm_0_24_native_triton_or_sort_mixed_no_logprobs_fp32_noise")):
        raise ValueError("0.24 worker and sampler source versions disagree")
    if model_execution_version in ("vllm_0_11_0", "vllm_0_24_0") and drafter_dummy_run is not None:
        raise NotImplementedError("vLLM0.11/0.24 startup storage currently requires MTP-off")
    _sampling_program(trace, sampler_profile, triton_cache)
    if drafter_dummy_run is not None and not callable(drafter_dummy_run):
        raise TypeError("worker profile drafter must be an explicit metadata-only callable")
    trace._check_tensor(input_ids)
    trace._check_tensor(positions)
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(type(n) is not int or n <= 0 for n in query_lengths)
            or input_ids.shape != positions.shape or input_ids.shape != (sum(query_lengths),)):
        raise ValueError("worker profile requires explicit positive query lengths covering its input rows")
    ops = StorageOnlyOps(trace, "profile.worker", pool_name)

    def dummy_run():
        hidden_states = model_forward(input_ids, positions)
        if drafter_dummy_run is not None:
            # GPUModelRunner._dummy_run calls the drafter before selecting
            # last_hidden_states. The target output stays live during every
            # sequential draft forward; no acceptance or draft sampling runs.
            drafter_dummy_run()
        # gpu_model_runner uses cumsum - 1 as an INT64 device gather index.
        indices = trace.tensor("profile.logit_indices", (len(query_lengths),), "int64")
        selected = ops.operation("profile_last_hidden_gather", (hidden_states, indices),
            shape=(len(query_lengths), *hidden_states.shape[1:]), dtype=hidden_states.dtype)
        return hidden_states, selected

    def dummy_sampler_run(hidden_states):
        hidden_states = ops.unary("rand_like", hidden_states)
        logits = compute_logits(hidden_states)
        rows = logits.shape[0]
        temperature = trace.tensor("profile.temperature", (rows,), "fp32")
        top_p = trace.tensor("profile.top_p", (rows,), "fp32")
        top_k = trace.tensor("profile.top_k", (rows,), "int64")
        # no_penalties=True skips the operations; dummy metadata still owns
        # these three device arrays throughout its version's sampling calls.
        penalty_metadata = tuple(trace.tensor("profile." + name, (rows,), "fp32")
            for name in ("frequency", "presence", "repetition"))
        sampler_output = native_sampler_storage_forward(trace, logits,
            source_profile=sampler_profile, temperature=temperature, top_k=top_k, top_p=top_p,
            pool_name=pool_name, triton_cache=triton_cache)
        if model_execution_version in ("vllm_0_24_0", "vllm_0_30_0"):
            native_sampler_storage_forward(trace, logits,
                source_profile=sampler_profile, temperature=temperature, top_k=top_k, top_p=top_p,
                pool_name=pool_name, triton_cache=triton_cache)
        return sampler_output

    hidden_states, last_hidden_states = dummy_run()
    output = dummy_sampler_run(last_hidden_states)
    # profile_run synchronizes before releasing these source-owned objects.
    # The metadata-only program describes that synchronous retirement order.
    del hidden_states, output


def ascend018_profile_storage_forward(trace, model_forward, compute_logits, *,
        input_ids, positions, query_lengths, pool_name="main"):
    """NPUModelRunner0.18 profile_run's head-only dummy sampler.

    The Ascend override returns (hidden_states, hidden_states) from
    _dummy_run. _dummy_sampler_run gathers the last rows and only calls
    compute_logits: no rand_like, logits cast, filtering or random sampler.
    Source indices originate in CPU NumPy metadata; the logical device
    index port is represented, without claiming its runtime copy lowering.
    Other preallocated worker objects, compilation and library workspace
    are separate from this caller-storage scope.
    """
    trace._check_tensor(input_ids)
    trace._check_tensor(positions)
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(type(n) is not int or n <= 0 for n in query_lengths)
            or input_ids.shape != positions.shape or input_ids.shape != (sum(query_lengths),)
            or input_ids.dtype != "int32" or positions.dtype != "int64"):
        raise ValueError("Ascend profile needs typed source ports and complete positive query lengths")
    ops = StorageOnlyOps(trace, "profile.worker", pool_name)
    hidden_states = model_forward(input_ids, positions)
    last_hidden_states = hidden_states

    def dummy_head(hidden_states):
        indices = trace.tensor("profile.logit_indices", (len(query_lengths),), "int64")
        hidden_states = ops.operation("ascend018_profile_last_hidden_gather", (hidden_states, indices),
            shape=(len(query_lengths), *hidden_states.shape[1:]), dtype=hidden_states.dtype)
        return compute_logits(hidden_states)

    output = dummy_head(last_hidden_states)
    # profile_run synchronizes before deleting its returned full hidden and
    # output. last_hidden_states still aliases the original model result.
    del hidden_states, output


def _mtp_retained_last_rows(requests):
    """vLLM 0.30 selects retained rows in padded queries.

    Source: vLLM 0.30 autoregressive/speculator.py query_len -= num_rejected,
    last_token_index = query_start + query_len - 1. Query starts retain the
    full scheduled extents; accepted output does not resize the draft input.
    Legacy eagle_prepare_inputs_padded_kernel uses the same query-end minus
    rejected-row rule and produces INT32 token_indices_to_sample.
    """
    rows = []
    query_start = 0
    for request in requests:
        retained = request.scheduled_tokens - request.mtp_rejected_tokens
        if retained <= 0:
            raise ValueError("MTP row selection requires a retained anchor row")
        rows.append(query_start + retained - 1)
        query_start += request.scheduled_tokens
    return tuple(rows)



def glm_gpu_v2_draft_update_work(*, num_requests, hidden_size, step, proposal_depth,
                               sampled_token_dtype, advance_draft_positions):
    """Logical roles of vLLM 0.30 _update_draft_inputs_kernel, without timing.

    InputBuffers stores INT32 ids/seq_lens and INT64 positions; speculator
    stores INT64 draft tokens, sampling positions and current step. Sampled
    token dtype is an explicit source-scenario operand, not inferred from a
    missing sampler binding. The source supplies grid=N and BLOCK_SIZE=1024
    but no num_warps, so this is not a complete LaunchConfig.
    """
    if any(type(value) is not int for value in (num_requests, hidden_size, step, proposal_depth)):
        raise TypeError("draft update dimensions/step/depth must be integers")
    if num_requests <= 0 or hidden_size <= 0 or not 1 <= step < proposal_depth:
        raise ValueError("draft update requires live requests and a uniform proposal step")
    if sampled_token_dtype not in ("int32", "int64") or type(advance_draft_positions) is not bool:
        raise ValueError("draft update requires explicit sampled-token dtype and advance flag")
    token_bytes = {"int32": 4, "int64": 8}[sampled_token_dtype]

    def role(name, dtype, elements, element_bytes, access):
        return {"role": name, "dtype": dtype, "elements": elements,
                "bytes": elements * element_bytes, "access": access}

    roles = [role("sampled_token", sampled_token_dtype, num_requests, token_bytes, "read"),
             role("current_step", "int64", 1, 8, "read"),
             role("draft_token_column", "int64", num_requests, 8, "write")]
    final = step == proposal_depth - 1
    additions = minimums = 0
    if not final:
        roles.extend((
            role("sample_src_positions", "int64", num_requests, 8, "read_write"),
            role("next_input_ids", "int32", num_requests, 4, "write"),
            role("post_norm_hidden", "bf16", num_requests * hidden_size, 2, "read"),
            role("persistent_feedback_hidden", "bf16", num_requests * hidden_size, 2, "write")))
        additions = num_requests
        if advance_draft_positions:
            roles.extend((role("input_positions", "int64", num_requests, 8, "read_write"),
                          role("seq_lens", "int32", num_requests, 4, "read_write")))
            additions += 2 * num_requests
            minimums = 2 * num_requests
    return {
        "step": step, "proposal_depth": proposal_depth, "final_step": final,
        "source_grid": (num_requests, 1, 1), "source_block_elements": 1024,
        "source_threads": None, "launch": None, "seconds": None, "demand": None,
        "operands": tuple(roles), "integer_additions": additions, "integer_minimums": minimums,
        "sampled_to_output_dtype": "int64",
        "sampled_to_next_input_dtype": None if final else "int32",
        "feedback_operation_id": None if final else f"uniform_{step}_feedback_update_role",
        "limitations": (
            "logical_object_role_bytes_not_repeated_transactions_or_observed_dram",
            "current_step_scalar_is_one_object_not_per_cta_physical_reads",
            "feedback_role_is_same_event_as_storage_subset_not_an_extra_copy",
            "runtime_max_model_len_position_values_and_saturating_results_unbound",
            "sampler_dtype_is_declared_source_scenario_not_runtime_binding",
            "triton_default_warps_compiler_local_services_and_launch_binding_unavailable",
            "address_predicate_conversion_and_instruction_work_not_priced",
        ),
    }



def glm_gpu_v2_prepare_decode_work(*, target_seq_lens, num_rejected, positions,
                                 sample_src_positions, max_num_requests, max_model_len,
                                 proposal_depth, num_rejected_dtype,
                                 advance_draft_positions):
    """Unpriced source state/roles for 0.30 prepare_decode_inputs.

    InputBuffers owns INT32 ids/query starts/seq_lens and INT64 positions.
    DraftModelSpeculator owns INT64 draft_tokens; this producer reads that
    buffer after sampling, regardless of sampler return dtype. Rejection
    count width is declared separately. The extra CTA writes full request capacity, even in eager.
    Depth1 returns before this producer. No Triton warps are specified by
    the source, so grid and BLOCK_SIZE do not form an actual launch.
    """
    if any(type(value) is not int for value in (max_num_requests, max_model_len, proposal_depth)):
        raise TypeError("prepare decode capacity/length/depth must be integers")
    if min(max_num_requests, max_model_len, proposal_depth) <= 0:
        raise ValueError("prepare decode capacity/length/depth must be positive")
    vectors = (target_seq_lens, num_rejected, positions, sample_src_positions)
    if any(not isinstance(values, (tuple, list)) or not values for values in vectors):
        raise ValueError("prepare decode requires nonempty state vectors")
    batch = len(target_seq_lens)
    if any(len(values) != batch for values in vectors) or batch > max_num_requests:
        raise ValueError("prepare decode state vectors must match live request capacity")
    if any(type(value) is not int for values in vectors for value in values):
        raise TypeError("prepare decode states must be integers")
    if (any(not 1 <= length <= max_model_len for length in target_seq_lens)
            or any(not 0 <= rejected < length for rejected, length in zip(num_rejected, target_seq_lens))
            or any(not 0 <= position < max_model_len for position in positions)
            or any(position < 0 for position in sample_src_positions)):
        raise ValueError("prepare decode requires valid lengths, retained anchors and positions")
    if (num_rejected_dtype not in ("int32", "int64")
            or type(advance_draft_positions) is not bool):
        raise ValueError("prepare decode requires explicit count dtype and advance flag")
    if proposal_depth == 1:
        return None

    def role(name, dtype, elements, access):
        return {"role": name, "dtype": dtype, "elements": elements,
                "bytes": elements * {"int32": 4, "int64": 8}[dtype], "access": access}

    roles = [role("draft_token_column_zero", "int64", batch, "read"),
             role("target_seq_lens", "int32", batch, "read"),
             role("num_rejected", num_rejected_dtype, batch, "read"),
             role("sample_src_positions", "int64", batch, "read_write"),
             role("next_input_ids", "int32", batch, "write"),
             role("query_start_loc_capacity", "int32", max_num_requests + 1, "write"),
             role("seq_lens_capacity", "int32", max_num_requests, "write")]
    next_positions = tuple(positions)
    next_lens = tuple(length - rejected for length, rejected in zip(target_seq_lens, num_rejected))
    if advance_draft_positions:
        roles.append(role("input_positions", "int64", batch, "read_write"))
        next_positions = tuple(min(position + 1, max_model_len - 1) for position in positions)
        next_lens = tuple(min(length + 1, max_model_len) for length in next_lens)
    return {
        "source": "vllm_0_30_autoregressive_speculator.prepare_decode_inputs",
        "source_grid": (batch + 1, 1, 1), "source_block_elements": 1024,
        "source_threads": None, "launch": None, "seconds": None, "demand": None,
        "operands": tuple(roles), "positions_after": next_positions,
        "sample_src_positions_after": tuple(position + 1 for position in sample_src_positions),
        "seq_lens_after_with_padding": next_lens + (0,) * (max_num_requests - batch),
        "query_start_loc_after_with_padding": tuple(range(batch)) + (batch,) * (max_num_requests + 1 - batch),
        "integer_subtractions": batch, "integer_additions": batch * (3 if advance_draft_positions else 1),
        "integer_minimums": 2 * batch if advance_draft_positions else 0,
        "producer_binding": None,
        "limitations": (
            "declared_state_scenario_not_observed_runtime_or_sampler_count_binding",
            "capacity_padding_stores_are_logical_addresses_not_dram_transactions",
            "grid_and_block_elements_do_not_bind_compiler_warps_or_actual_launch",
            "integer_counts_exclude_address_predicate_and_conversion_instructions",
            "slot_mapping_metadata_page_ownership_and_cache_rollback_are_separate_events",
        ),
    }



def glm_gpu_v2_slot_mapping_work(*, idx_mapping, query_start_loc, positions,
                               block_tables_by_group, kv_block_sizes, kernel_block_sizes,
                               mapping_enabled, slot_capacity, returned_tokens, cp_size):
    """Unpriced CP1 source roles/state of GPU-v2 compute_slot_mappings.

    BlockTables owns INT32 block numbers, INT64 slots/strides, UINT64 table
    pointers and BOOL mapping flags. The returned view does not limit the
    extra CTA's writes: padding starts at actual tokens and covers capacity.
    CP requires a different local-position mapping and is rejected here.
    Source masked position loads use zero for SIMD tails; CP1 block-number
    loads are unmasked, so partial 1024-element blocks also read row index0.
    Distinct source objects do not prove physical transactions or residency.
    """
    if any(type(value) is not int for value in (slot_capacity, returned_tokens, cp_size)):
        raise TypeError("slot mapping capacity/returned tokens/CP must be integers")
    if cp_size != 1:
        raise NotImplementedError("slot mapping source work currently requires CP1")
    vectors = (idx_mapping, query_start_loc, positions, block_tables_by_group,
               kv_block_sizes, kernel_block_sizes, mapping_enabled)
    if any(not isinstance(values, (tuple, list)) for values in vectors):
        raise TypeError("slot mapping requires explicit state/group vectors")
    batch, groups = len(idx_mapping), len(block_tables_by_group)
    if (not batch or len(query_start_loc) != batch + 1
            or any(len(values) != groups for values in (kv_block_sizes, kernel_block_sizes, mapping_enabled))):
        raise ValueError("slot mapping request/group vectors must match")
    if any(type(value) is not int for values in (idx_mapping, query_start_loc, positions) for value in values):
        raise TypeError("slot mapping request states must be integers")
    if (query_start_loc[0] != 0 or any(a > b for a, b in zip(query_start_loc, query_start_loc[1:]))
            or query_start_loc[-1] != len(positions)
            or not 0 <= len(positions) <= returned_tokens <= slot_capacity
            or any(value < 0 for value in (*idx_mapping, *positions))):
        raise ValueError("slot mapping requires valid query starts, positions and slice/capacity")
    if (any(type(value) is not bool for value in mapping_enabled)
            or any(type(value) is not int or not 0 < value < 2**31
                   for value in (*kv_block_sizes, *kernel_block_sizes))
            or any(kv_size % kernel_size for kv_size, kernel_size in zip(kv_block_sizes, kernel_block_sizes))):
        raise ValueError("slot mapping group sizes/flags must be explicit and kernel blocks must divide KV blocks")
    if not groups:
        return None  # source returns an empty group view without a kernel
    actual = len(positions)
    output, group_reads, unique_block_reads = [], [], 0
    for table, kernel_size, enabled in zip(block_tables_by_group, kernel_block_sizes, mapping_enabled):
        if (not isinstance(table, (tuple, list)) or not table
                or any(not isinstance(row, (tuple, list)) or not row for row in table)
                or len({len(row) for row in table}) != 1
                or any(type(value) is not int or not 0 <= value < 2**31 for row in table for value in row)
                or any(row >= len(table) for row in idx_mapping)):
            raise ValueError("slot mapping requires rectangular INT32 block tables and valid request rows")
        slots, reads, tail_reads, unique_reads = [], [], [], set()
        for request, row in enumerate(idx_mapping):
            start, end = query_start_loc[request:request + 2]
            request_reads = []
            for position in positions[start:end]:
                block_index = position // kernel_size if enabled else 0
                if block_index >= len(table[row]):
                    raise ValueError("slot mapping position exceeds declared block table")
                request_reads.append(block_index)
                unique_reads.add((row, block_index))
                slot = table[row][block_index] * kernel_size + position % kernel_size
                slots.append(slot if enabled else -1)
            reads.append(tuple(request_reads))
            tail = end > start and (end - start) % 1024 != 0
            tail_reads.append(0 if tail else None)
            if tail:
                unique_reads.add((row, 0))
        unique_block_reads += len(unique_reads)
        group_reads.append({"valid_lane_block_indices_by_request": tuple(reads),
                            "simd_tail_block_index_by_request": tuple(tail_reads),
                            "distinct_block_objects": tuple(sorted(unique_reads))})
        output.append(tuple(slots) + (-1,) * (slot_capacity - actual))

    def role(name, dtype, elements, access):
        widths = {"int32": 4, "int64": 8, "uint64": 8, "bool": 1}
        return {"role": name, "dtype": dtype, "elements": elements,
                "bytes": elements * widths[dtype], "access": access}

    return {
        "source": "vllm_0_30_gpu.BlockTables.compute_slot_mappings",
        "source_grid": (groups, batch + 1, 1), "source_block_elements": 1024,
        "source_threads": None, "launch": None, "seconds": None, "demand": None,
        "actual_tokens": actual, "returned_tokens": returned_tokens, "slot_capacity": slot_capacity,
        "slot_values_capacity": tuple(output), "group_reads": tuple(group_reads),
        "operands": (
            role("idx_mapping", "int32", batch, "read"),
            role("query_start_loc", "int32", batch + 1, "read"),
            role("positions", "int64", actual, "read"),
            role("block_table_ptrs", "uint64", groups, "read"),
            role("block_table_strides", "int64", groups, "read"),
            role("kv_block_sizes", "int32", groups, "read"),
            role("kernel_block_sizes", "int32", groups, "read"),
            role("mapping_enabled", "bool", groups, "read"),
            role("distinct_block_numbers", "int32", unique_block_reads, "read"),
            role("live_slot_ids", "int64", groups * actual, "write"),
            role("capacity_padding_slot_ids", "int64", groups * (slot_capacity - actual), "write")),
        "padding_writes_inside_returned_slice_bytes": groups * (returned_tokens - actual) * 8,
        "padding_writes_outside_returned_slice_bytes": groups * (slot_capacity - returned_tokens) * 8,
        "producer_binding": None,
        "limitations": (
            "declared_block_ids_and_requests_not_normal_allocator_or_runtime_evidence",
            "logical_distinct_objects_and_slot_stores_not_dram_transactions_or_local_services",
            "shared_input_arrays_are_counted_once_not_per_group_or_cta_reads",
            "SIMD_tail_index0_object_not_an_extra_scalar_or_transaction_count",
            "compiler_warps_and_actual_launch_unbound",
            "layer_views_alias_group_slots_not_additional_stores",
            "slot_mapping_is_separate_from_KV_insert_rejected_token_rollback_and_page_lifecycle",
        ),
    }



def glm_gpu_v2_metadata_refresh_plan(*, proposal_depth, metadata_update_supported_by_groups):
    """Source event dependencies for GLM's advancing, non-dummy eager loop.

    AutoRegressiveSpeculator.advance_draft_positions=True applies to GLM.
    _configure_fused_multi_step_decode chooses fused iff every attention
    group supports draft metadata updates. These explicit capability bits
    do not bind actual backend identity. Events include host/view/library
    and model roles, not a kernel launch count or a priced native graph.
    """
    if type(proposal_depth) is not int or proposal_depth < 1:
        raise ValueError("metadata refresh plan requires positive proposal depth")
    if (not isinstance(metadata_update_supported_by_groups, (tuple, list))
            or not metadata_update_supported_by_groups
            or any(type(value) is not bool for value in metadata_update_supported_by_groups)):
        raise ValueError("metadata refresh plan requires explicit attention-group capability flags")
    fused = proposal_depth > 1 and all(metadata_update_supported_by_groups)
    events = []

    def append(identifier, kind, predecessors, *, step=None, group=None, final_step=None):
        events.append({
            "event_id": identifier, "source_role": kind,
            "predecessors": tuple(dict.fromkeys(predecessors)),
            "step": step, "attention_group": group, "final_step": final_step,
            "launch": None, "demand": None, "seconds": None,
        })
        return identifier

    def initial_metadata(step, predecessor):
        slots = append(f"uniform_{step}_slot_mapping", "compute_slot_mappings", (predecessor,), step=step)
        views = append(f"uniform_{step}_layer_slot_views", "build_slot_mappings_by_layer", (slots,), step=step)
        return (append(f"uniform_{step}_metadata_build", "build_uniform_attn_metadata", (views,), step=step),)

    if proposal_depth > 1:
        last_update = append("prepare_decode_inputs", "prepare_decode_inputs", ("on_prefill_end",))
        metadata_frontier = initial_metadata(1, last_update) if fused else ()
        for step in range(1, proposal_depth):
            if not fused:
                metadata_frontier = initial_metadata(step, last_update)
            fill = append(f"uniform_{step}_current_step_fill", "current_draft_step.fill_",
                          (last_update, *metadata_frontier), step=step)
            forward = append(f"uniform_{step}_draft_forward", "draft_model_forward",
                             (fill, *metadata_frontier), step=step)
            sample = append(f"uniform_{step}_draft_sample", "sample_draft", (forward,), step=step)
            final = step == proposal_depth - 1
            last_update = append(f"uniform_{step}_input_update", "update_draft_inputs",
                                 (sample, forward), step=step, final_step=final)
            if fused and not final:
                slots = append(f"uniform_{step + 1}_slot_mapping", "compute_slot_mappings",
                               (last_update,), step=step + 1)
                group_updates = []
                previous_group_update = slots
                for group in range(len(metadata_update_supported_by_groups)):
                    previous_group_update = append(f"uniform_{step + 1}_group_{group}_metadata_update",
                        "update_draft_decode_metadata", (previous_group_update,), step=step + 1, group=group)
                    group_updates.append(previous_group_update)
                metadata_frontier = tuple(group_updates)
    return {
        "source": "vllm_0_30_AutoRegressiveSpeculator_advancing_eager",
        "proposal_depth": proposal_depth, "advance_draft_positions": True,
        "cudagraph_runtime_mode": "NONE", "dummy_run": False,
        "metadata_update_supported_by_groups": tuple(metadata_update_supported_by_groups),
        "execution_kind": "fused" if fused else "ordinary", "source_events": tuple(events),
        "actual_attention_backend_binding": None,
        "scope": "declared_capability_source_control_flow_not_kernel_call_count_or_priced_native",
        "limitations": (
            "current_step_fill_sampler_and_group_updates_do_not_bind_library_kernel_launches",
            "host_alias_and_backend_events_are_not_assumed_free",
            "prefill_on_prefill_end_predecessor_is_external_to_this_uniform_loop_subset",
            "EPLB_DP_sync_model_fusion_and_KV_page_lifecycle_not_covered",
            "FULL_graph_capture_replay_and_nonadvancing_Gemma4_are_outside_this_GLM_eager_scope",
        ),
    }



def glm_gpu_v2_sparse_backend_metadata_plan(*, proposal_depth, backend_name):
    """Conditional source constraint for NVIDIA sparse MLA draft builders.

    These vLLM0.30 builders inherit AttentionMetadataBuilder's False flag
    through SparseMLACommonMetadataBuilder without an override. One such
    draft attention group excludes fused advancing decode regardless of
    other groups. This does not choose/observe a backend or assert the
    total group count; the ordinary path has no per-group update callbacks.
    """
    builders = {
        "FLASH_ATTN_MLA_SPARSE": ("FlashAttnMLASparseMetadataBuilder",),
        "FLASHMLA_SPARSE": ("FlashMLASparseMetadataBuilder",),
        "FLASHINFER_MLA_SPARSE_SM90": (
            "FlashInferMLASparseSM90Builder", "FlashInferMLASparseMetadataBuilder"),
    }
    if not isinstance(backend_name, str):
        raise TypeError("sparse metadata source requires an explicit backend name")
    if backend_name not in builders:
        raise NotImplementedError("sparse backend metadata capability is not source-bound")
    plan = glm_gpu_v2_metadata_refresh_plan(
        proposal_depth=proposal_depth, metadata_update_supported_by_groups=(False,))
    plan["source_backend_constraint"] = {
        "declared_sparse_backend": backend_name,
        "builder_inheritance": (*builders[backend_name],
            "SparseMLACommonMetadataBuilder", "AttentionMetadataBuilder"),
        "supports_draft_decode_metadata_update": False,
        "capability_source": "vllm/v1/attention/backend.py:AttentionMetadataBuilder",
        "capability_scope": "one_known_sparse_group_suffices_not_total_group_count",
        "backend_selection_observed": False,
        "actual_group_membership": None,
        "source_runtime_requirements": "GLM advancing eager; at least one group uses declared unmodified 0.30 builder",
    }
    plan["scope"] = "conditional_sparse_backend_source_constraint_not_observed_runtime_or_priced_native"
    return plan



def glm_gpu_v2_mtp_storage_work(model, requests, runtime, *, prefill_context_parallel_size):
    """Source-only storage roles for the eager GLM GPU-v2 proposer subset.

    vLLM 0.30 autoregressive/speculator.py _prefill gathers sampled hidden,
    then copies it into persistent feedback storage. GLM nvidia/mtp.py
    returns (post_norm_hidden, post_norm_hidden), so no second gather is
    needed. MTPSpeculator.on_prefill_end compacts shared top-k via fancy
    indexing and assignment. These are logical object roles and lifetimes,
    never library launch counts, observed traffic, demand or prediction.
    """
    from .model_config import ModelConfig
    from .runtime_inputs import RuntimeInputs
    from src.serving.schedule.schedule_step import ScheduledRequest
    if not isinstance(model, ModelConfig) or not isinstance(runtime, RuntimeInputs):
        raise TypeError("GLM storage work requires typed model/runtime inputs")
    if (model.mtp_architecture != "glm_moe_dsa_mtp" or model.compute_dtype != "bf16"
            or model.num_mtp_layers != 1):
        raise NotImplementedError("GLM storage work requires the single-layer BF16 V32 source")
    if runtime.mtp_proposer_execution_source != "vllm_0_30_gpu_v2":
        raise ValueError("GLM storage work requires explicit GPU-v2 source selection")
    if type(prefill_context_parallel_size) is not int or prefill_context_parallel_size != 1:
        raise NotImplementedError("GLM storage work currently requires PCP1")
    if not isinstance(requests, (tuple, list)) or not requests:
        raise ValueError("GLM storage work requires a nonempty request sequence")
    if any(not isinstance(request, ScheduledRequest) for request in requests):
        raise TypeError("GLM storage work requires ScheduledRequest objects")
    depths = {request.mtp_next_proposed_tokens for request in requests}
    if len(depths) != 1 or next(iter(depths)) <= 0:
        raise ValueError("GLM storage work requires one positive proposal depth")
    if any(request.mtp_proposer_mode != "padded" for request in requests):
        raise ValueError("GLM storage work requires padded proposer queries")
    depth = next(iter(depths))
    retained_rows = _mtp_retained_last_rows(requests)
    batch, hidden = len(requests), model.hidden_size
    hidden_bytes = batch * hidden * model.dtype_bytes
    index_bytes = batch * 8  # last_token_indices is INT64 in the source.

    def operation(name, kind, source_role, destination_role, dtype, value_bytes,
                  predecessors, *, selected_rows=None, consumers=()):
        return {
            "operation_id": name, "operation": kind,
            "source_role": source_role, "destination_role": destination_role,
            "dtype": dtype, "read_value_bytes": value_bytes,
            "read_index_bytes": index_bytes if selected_rows is not None else 0,
            "write_value_bytes": value_bytes, "selected_rows": selected_rows,
            "predecessors": predecessors, "consumers": consumers,
            "launch": None, "seconds": None, "demand": None,
            "execution_support": "unsupported_launch_unbound",
        }

    operations = [
        operation("prefill_hidden_gather", "materialized_gather",
                  "draft_post_norm_hidden", "sample_hidden_temporary", "bf16",
                  hidden_bytes, ("draft_prefill_forward",), selected_rows=retained_rows,
                  consumers=("draft_sample", "prefill_feedback_copy")),
        operation("prefill_feedback_copy", "copy",
                  "sample_hidden_temporary", "persistent_feedback_hidden_front", "bf16",
                  hidden_bytes, ("prefill_hidden_gather", "draft_sample")),
    ]
    if model.index_share_for_mtp_iteration and depth > 1:
        topk_bytes = batch * model.index_topk * 4  # topk_indices_buffer is INT32.
        operations.extend((
            operation("topk_compaction_gather", "materialized_gather",
                      "draft_topk_buffer", "topk_compaction_temporary", "int32",
                      topk_bytes, ("draft_prefill_forward", "prefill_feedback_copy"),
                      selected_rows=retained_rows, consumers=("topk_front_copy",)),
            operation("topk_front_copy", "copy",
                      "topk_compaction_temporary", "draft_topk_buffer_front", "int32",
                      topk_bytes, ("topk_compaction_gather",), consumers=("uniform_draft_forward",)),
        ))
    # _generate_draft samples last_hidden[:num_reqs], a view, not a fancy
    # gather. update_draft_inputs copies feedback only before the final step.
    for step_index in range(1, depth - 1):
        operations.append(operation(f"uniform_{step_index}_feedback_update_role",
            "copy_role_within_update_draft_inputs", "uniform_post_norm_hidden_front",
            "persistent_feedback_hidden_front", "bf16", hidden_bytes,
            (f"uniform_{step_index}_draft_forward", f"uniform_{step_index}_draft_sample"),
            consumers=(f"uniform_{step_index + 1}_draft_forward",)))
    return {
        "scope": "GLM GPU-v2 PCP1 eager selected-hidden/compaction/feedback storage subset",
        "retained_rows": retained_rows, "padded_query_rows": sum(r.scheduled_tokens for r in requests),
        "sample_rows": batch, "proposal_depth": depth,
        "post_norm_logits_feedback_source_alias": True,
        "uniform_sampling_is_view": True,
        "final_uniform_step_feedback_update": False,
        "sampled_token_dtype_scenario": "int64",
        "sampled_token_producer_binding": None,
        "operations": tuple(operations),
        "uniform_updates": tuple(glm_gpu_v2_draft_update_work(
            num_requests=batch, hidden_size=hidden, step=step_index, proposal_depth=depth,
            sampled_token_dtype="int64", advance_draft_positions=True)
            for step_index in range(1, depth)),
        "limitations": (
            "source_tensor_roles_not_observed_dram_or_library_launch_counts",
            "actual_storage_addresses_strides_aliases_and_allocator_lifetimes_unbound",
            "indexed_temporary_and_front_copy_are_distinct_logical_roles",
            "feedback_update_role_is_within_fused_update_not_an_independent_launch",
            "sampler_rng_logits_prefill_token_position_slot_and_other_control_roles_not_covered",
            "no_rate_demand_or_duration_assigned_to_storage_work",
            "normal_gpu_v2_workload_remains_unsupported_until_launch_binding",
        ),
    }
