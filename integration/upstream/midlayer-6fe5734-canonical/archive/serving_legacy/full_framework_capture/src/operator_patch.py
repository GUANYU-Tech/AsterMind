"""Auto-installed operator-capture patch for vLLM V1 EngineCore child processes
(approach D). Companion to `_sidecar_patch.py`.

Captures the REAL per-step operator sequence (with exact shapes / comm volumes)
that vLLM issues during the model forward pass, by:

  1. wrapping `GPUModelRunner.execute_model` to delimit forward STEPS (one call =
     one step) and lazily install module hooks the first time `self.model` exists;
  2. registering `forward_pre_hook`s on the model's projection / attention / norm
     / MoE / collective modules to record op_type + real input shapes.

Records are appended as JSONL to the path in env `CUDA_OPERATOR_TRACE_PATH`.
Installed in the spawn-child via the same env+.pth+meta-path mechanism as the
scheduler sidecar (see sidecar_lifecycle.py).

This module is intentionally defensive: any capture failure is swallowed (the
real vLLM run must never crash because of tracing). A missing/empty operator
trace is detected downstream (the plugin run fails loudly there), not here.

CLEAN-ROOM: design from public vLLM/PyTorch APIs only (named_modules,
register_forward_pre_hook, tensor.shape/.dtype). No third-party code/naming.
"""
import importlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
# vllm-ascend overrides execute_model in its own NPUModelRunner (a subclass of
# the native GPUModelRunner). Patching only the native class does NOT delimit
# forward steps on Ascend because the subclass method shadows it. This OPTIONAL
# target patches NPUModelRunner.execute_model when vllm-ascend is present; on a
# native-CUDA vLLM the module is absent → never triggered (A800/C500 unchanged).
# This dispatches on "which vLLM runner module got imported", not on card name.
_NPU_RUNNER_MODULE = "vllm_ascend.worker.model_runner_v1"

# Per-process capture state.
_state = threading.local()
_GLOBAL = {"step_id": -1, "op_idx": 0, "hooks_installed": False,
           "path": None, "base_path": None}


def _emit(record: dict):
    path = _GLOBAL["path"]
    if not path:
        return
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Module classification: map a live nn.Module + its input to an OperatorRecord.
# Uses vLLM/PyTorch public class names; unknown modules are ignored (not all
# modules are model ops — only the performance-critical ones are recorded).
# ---------------------------------------------------------------------------

def _first_tensor(args, kwargs=None):
    import torch
    for a in args:
        if isinstance(a, torch.Tensor):
            return a
    if kwargs:
        for a in kwargs.values():
            if isinstance(a, torch.Tensor):
                return a
    return None


def _dtype_name(dt):
    """Short precision string from a torch.dtype ('torch.bfloat16' -> 'bfloat16')."""
    try:
        return str(dt).replace("torch.", "")
    except Exception:
        return ""


# QuantizeMethod class names that mean "not quantized" (weights kept at model
# dtype). Anything else on a linear/moe module is a real quant scheme.
_UNQUANT_METHODS = {
    "UnquantizedLinearMethod", "UnquantizedFusedMoEMethod",
    "UnquantizedEmbeddingMethod",
}


def _capture_quant(module, act_dtype):
    """Static per-layer (dtype, quant) read from a live linear/moe module.

    Reads the REAL precision the layer will compute at, from vLLM's already-built
    quant_method (the object `quant_config.get_quant_method(layer, prefix)`
    returned at model build) — NOT reconstructed from a kernel name. Returns
    (dtype_str, quant_dict|None):
      • quant_dict None  -> unquantized (weights at model dtype).
      • else {"scheme","weight_dtype","act_dtype","group_size"}.
    Fully defensive: any failure -> (act_dtype, None), so capture never breaks
    and the op just falls back to the model-dtype pricing path.
    """
    try:
        import torch
        qm = getattr(module, "quant_method", None)
        qm_name = type(qm).__name__ if qm is not None else ""
        if not qm_name or qm_name in _UNQUANT_METHODS:
            return act_dtype, None
        qc = getattr(qm, "quant_config", None)
        # Canonical scheme id (fp8 / compressed-tensors / awq / gptq / modelopt…).
        scheme = ""
        get_name = getattr(qc, "get_name", None)
        if callable(get_name):
            try:
                scheme = str(get_name())
            except Exception:
                scheme = ""
        if not scheme:
            # Fall back to the method class name (e.g. "Fp8LinearMethod").
            scheme = qm_name
        group_size = None
        for attr in ("group_size", "weight_group_size"):
            gs = getattr(qc, attr, None)
            if isinstance(gs, int):
                group_size = gs
                break
        # Weight dtype: the registered weight param (may be a packed int type for
        # w4a16 etc.). Best-effort; "" if the param is not yet materialized.
        weight_dtype = ""
        w = getattr(module, "weight", None)
        if isinstance(w, torch.Tensor):
            weight_dtype = _dtype_name(w.dtype)
        return act_dtype, {"scheme": scheme, "weight_dtype": weight_dtype,
                           "act_dtype": act_dtype, "group_size": group_size}
    except Exception:
        return act_dtype, None


def _classify_module(module):
    """Return (op_type, position) for a module we want to record, else None."""
    cls = type(module).__name__
    # Linear family (vLLM parallel linears + plain nn.Linear). Match by name
    # SUFFIX so vLLM-ascend's subclasses (AscendQKVParallelLinear,
    # AscendRowParallelLinear, AscendMergedColumnParallelLinear,
    # AscendColumnParallelLinear in vllm_ascend.ops.linear) are captured too —
    # exact-equality missed them, dropping ~all GEMM ops on Ascend (linear
    # captured 518 vs the ~164k real). This is data-driven dispatch by module
    # class name, not by card name.
    if cls.endswith("QKVParallelLinear"):
        return "linear", "qkv_proj"
    if cls.endswith("RowParallelLinear"):
        return "linear", "row"     # o_proj or down_proj (disambiguated by name path)
    if cls.endswith("MergedColumnParallelLinear") or cls.endswith("ColumnParallelLinear"):
        return "linear", "col"     # gate_up_proj or similar
    if cls.endswith("ParallelLMHead"):
        return "linear", "lm_head"
    # NOTE: ReplicatedLinear (MoE router gate) is intentionally NOT captured here:
    # the FusedMoE recipe (_moe_ffn_kernels) already emits the gate GEMM, so
    # capturing it separately would double-count. Dense models do not use it.
    if "RMSNorm" in cls:
        return "rms_norm", "norm"
    if "RotaryEmbedding" in cls:
        return "rotary", ""
    # Match ONLY the inner attention-compute module (vllm.attention.layer.Attention),
    # not the model-specific wrappers (Qwen3Attention / Qwen3MoeAttention / ...),
    # which also end in "Attention" and would double-count the attention op.
    if cls == "Attention":
        return "attention", ""
    if "SiluAndMul" in cls or "GeluAndMul" in cls or cls.endswith("AndMul"):
        return "activation", ""
    if "LogitsProcessor" in cls:
        # lm_head GEMM is invoked inside LogitsProcessor(__call__) -> not directly
        # hookable as ParallelLMHead in some models; capture it here instead.
        return "linear", "lm_head"
    if "FusedMoE" in cls or cls.endswith("MoE"):
        return "moe", ""
    return None


def _capture_routing(topk_ids):
    """Stash the real per-expert token histogram from a select_experts call.

    topk_ids: int tensor [num_tokens, top_k] of GLOBAL logical expert ids
    (EPLB-disabled, per vLLM docstring). We bincount over the global expert
    space; the downstream traffic model already maps to EP-local experts via
    expert assignment + grid, and a global histogram is the routing ground
    truth regardless of EP layout. Stored on thread-local state so the MoE
    forward post-hook (which fires right after) can attach it to the record.
    """
    try:
        import torch
        if not isinstance(topk_ids, torch.Tensor):
            return
        ids = topk_ids.detach().reshape(-1)
        if ids.numel() == 0:
            return
        n_exp = int(ids.max().item()) + 1
        counts = torch.bincount(ids.to(torch.int64), minlength=n_exp)
        _state.__dict__["last_tok_per_expert"] = counts.cpu().tolist()
    except Exception:
        pass


def _patch_routing_capture():
    """Patch the MoE routing entry points to capture the REAL per-expert
    routing histogram (topk_ids bincount). vLLM 0.18 moved routing out of
    FusedMoE into two backend-specific entry points (the old code hooked
    FusedMoE.select_experts, which no longer exists → silently no-op'd, so
    910B serving fell back to a uniform distribution — see
    docs/moe_tpe_format.md). Two points, one per backend:

      • GPU (A800): vllm.model_executor.layers.fused_moe.router.base_router.
        BaseRouter.select_experts — plain instance method; all router
        subclasses inherit it and none override it, so patching the class
        method covers every GPU routing call. topk_ids = result[1].
      • Ascend (910B): vllm_ascend.ops.fused_moe.experts_selector.select_experts
        — module-level free function. Each caller binds it by name
        (`from ...experts_selector import select_experts`), so patching the
        source module alone misses already-imported callers — rebind in every
        importer's namespace too (same by-name-binding issue as
        _patch_allreduce_binding).

    Both return (topk_weights, topk_ids); topk_ids are GLOBAL logical expert
    ids, shape [num_tokens, top_k] (EPLB-disabled) — exactly what
    _capture_routing bincounts. Idempotent (_operator_patched guard).
    """
    # ── GPU: BaseRouter.select_experts (instance method on the class) ──
    try:
        from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    except Exception:
        BaseRouter = None
    if BaseRouter is not None:
        orig = BaseRouter.__dict__.get("select_experts")
        if orig is not None and not getattr(orig, "_operator_patched", False):
            def _patched_gpu(self, *args, **kwargs):
                result = orig(self, *args, **kwargs)
                try:
                    if isinstance(result, tuple) and len(result) >= 2:
                        _capture_routing(result[1])
                except Exception:
                    pass
                return result
            _patched_gpu._operator_patched = True
            BaseRouter.select_experts = _patched_gpu

    # ── Ascend: experts_selector.select_experts (free function + rebind) ──
    try:
        import vllm_ascend.ops.fused_moe.experts_selector as _es
    except Exception:
        _es = None
    if _es is not None:
        orig = getattr(_es, "select_experts", None)
        if orig is not None and not getattr(orig, "_operator_patched", False):
            def _patched_asc(*args, **kwargs):
                result = orig(*args, **kwargs)
                try:
                    if isinstance(result, tuple) and len(result) >= 2:
                        _capture_routing(result[1])
                except Exception:
                    pass
                return result
            _patched_asc._operator_patched = True
            _es.select_experts = _patched_asc
            # Rebind every module that imported select_experts by name so the
            # patch actually intercepts the Ascend callers (fused_moe.apply,
            # AscendFusedMoE.forward_impl, w8a8/w4a8 quant methods).
            for _mod in list(sys.modules.values()):
                if getattr(_mod, "select_experts", None) is orig:
                    _mod.select_experts = _patched_asc


def _record_module(module, name):
    op = _classify_module(module)
    if op is None:
        return
    op_type, position = op

    def _hook(mod, args, kwargs):
        try:
            t = _first_tensor(args, kwargs)
            if t is None:
                return
            shape = tuple(int(x) for x in t.shape)
            tokens = shape[0] if len(shape) >= 1 else 1
            act_dtype = _dtype_name(getattr(t, "dtype", None))
            actual_op_type = op_type
            if op_type == "rms_norm":
                import torch
                residual = args[1] if len(args) >= 2 else (kwargs or {}).get("residual")
                if isinstance(residual, torch.Tensor):
                    actual_op_type = "fused_add_rms_norm"
            attrs = {"input_shape": list(shape)}
            # Precision: real activation dtype for every op; weight-quant scheme
            # only for weight-bearing ops (linear here; moe in the post-hook).
            rec_dtype = act_dtype
            rec_quant = None
            # op-specific attrs the recipe last hop needs
            if actual_op_type == "linear":
                rec_dtype, rec_quant = _capture_quant(mod, act_dtype)
                in_features = shape[-1]
                # vLLM parallel linears set output_size_per_partition to the
                # rank-local width (= output_size / tp_size for column-parallel,
                # = output_size for row-parallel). Use it so the captured shape
                # matches the actual per-rank GEMM. Fall back to output_size
                # for non-parallel modules (LogitsProcessor / plain nn.Linear).
                out_features = int(
                    getattr(mod, "output_size_per_partition", 0)
                    or getattr(mod, "output_size", 0)
                    or getattr(mod, "vocab_size", 0) or 0)
                attrs.update({"B": 1, "M": tokens, "N": out_features, "K": in_features})
            elif actual_op_type in ("rms_norm", "fused_add_rms_norm",
                                    "rotary", "kv_cache", "activation"):
                attrs.update({"batch": 1, "seq_len": tokens})
            elif actual_op_type == "attention":
                attrs.update({"batch": 1, "seq_len": tokens,
                              "is_decode": tokens <= _state.__dict__.get("decode_thresh", 1)})
            elif actual_op_type in ("moe",):
                attrs.update({"batch": 1, "seq_len": tokens, "is_decode": tokens <= 1})
                # Real routing histogram captured by the select_experts wrapper
                # during THIS module's forward. For a pre-hook it is not yet
                # available (see moe post-hook below); only attached there.
            rec = {
                "step_id": _GLOBAL["step_id"],
                "op_idx": _GLOBAL["op_idx"],
                "op_type": actual_op_type,
                "position": position,
                "layer_idx": _layer_idx_from_name(name),
                "attrs": attrs,
                "ts_us": time.perf_counter() * 1e6,
                "dtype": rec_dtype,
                "quant": rec_quant,
            }
            _GLOBAL["op_idx"] += 1
            _emit(rec)
        except Exception:
            pass  # never break the real forward

    def _moe_post_hook(mod, args, kwargs, output):
        # MoE routing (topk_ids) is only known AFTER select_experts runs inside
        # the module forward, so the MoE record is emitted from a post-hook (not
        # the pre-hook). The router gate is not a separately-hooked module, so no
        # other op claims an op_idx between this module's entry and exit —
        # op_idx ordering is preserved.
        try:
            t = _first_tensor(args, kwargs)
            if t is None:
                return
            shape = tuple(int(x) for x in t.shape)
            tokens = shape[0] if len(shape) >= 1 else 1
            act_dtype = _dtype_name(getattr(t, "dtype", None))
            rec_dtype, rec_quant = _capture_quant(mod, act_dtype)
            attrs = {"input_shape": list(shape),
                     "batch": 1, "seq_len": tokens, "is_decode": tokens <= 1}
            tpe = _state.__dict__.pop("last_tok_per_expert", None)
            if tpe is not None:
                attrs["tok_per_expert"] = tpe
            rec = {
                "step_id": _GLOBAL["step_id"],
                "op_idx": _GLOBAL["op_idx"],
                "op_type": "moe",
                "position": position,
                "layer_idx": _layer_idx_from_name(name),
                "attrs": attrs,
                "ts_us": time.perf_counter() * 1e6,
                "dtype": rec_dtype,
                "quant": rec_quant,
            }
            _GLOBAL["op_idx"] += 1
            _emit(rec)
        except Exception:
            pass  # never break the real forward

    if op_type == "moe":
        module.register_forward_hook(_moe_post_hook, with_kwargs=True)
    else:
        module.register_forward_pre_hook(_hook, with_kwargs=True)


def _layer_idx_from_name(name):
    # e.g. "model.layers.3.self_attn.qkv_proj" -> 3
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def _install_module_hooks(model):
    try:
        # Patch MoE routing capture once (class/module-level, not per-instance).
        # Must run AFTER the model is built so vllm_ascend's experts_selector
        # (and its by-name importers) are already imported and rebindable.
        _patch_routing_capture()
        for name, module in model.named_modules():
            _record_module(module, name)
        _GLOBAL["hooks_installed"] = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Collective-communication capture (TP all-reduce + EP all-to-all).
#
# TP all-reduce is a functional call (vllm.distributed.communication_op.
# tensor_model_parallel_all_reduce), not an nn.Module, so it is patched here to
# emit an `all_reduce` OperatorRecord with the real message bytes.
#
# EP all-to-all (MoE dispatch/combine) is a real method call on the EP
# GroupCoordinator (vllm.distributed.parallel_state.GroupCoordinator.dispatch /
# .combine), invoked from FusedMoE.forward_impl ONLY when vLLM actually routes
# tokens across EP ranks (do_naive_dispatch_combine, i.e. dp_size>1, or the
# pplx/deepep chunked paths). We patch those methods so an `all_to_all`
# OperatorRecord is emitted with the real message bytes and dispatch/combine
# position WHEN AND ONLY WHEN the call really happens. In the common TP/EP
# single-DP case vLLM does NOT call dispatch/combine (it uses expert_map + a
# final tensor_model_parallel_all_reduce, which we capture as `all_reduce`), so
# no all-to-all is emitted -- matching nsys, which shows zero all-to-all kernels
# there. The structural fabrication that previously wrapped every MoE op has
# been removed (see operator_records); all-to-all is now capture-only.
# ---------------------------------------------------------------------------

_COMM_MODULE = "vllm.distributed.communication_op"
# RowParallelLinear does `from ...communication_op import tensor_model_parallel_all_reduce`
# at import time, binding its own name; patch that binding too so the TP all-reduce
# in linear forward is captured.
_LINEAR_MODULE = "vllm.model_executor.layers.linear"
# EP all-to-all dispatch/combine live on GroupCoordinator in parallel_state.
_PSTATE_MODULE = "vllm.distributed.parallel_state"


def _make_all_reduce_wrapper(fn):
    def _patched_all_reduce(input_):
        try:
            import torch
            if isinstance(input_, torch.Tensor):
                shape = tuple(int(x) for x in input_.shape)
                tokens = shape[0] if shape else 1
                msg_bytes = input_.numel() * input_.element_size()
                _emit({
                    "step_id": _GLOBAL["step_id"], "op_idx": _GLOBAL["op_idx"],
                    "op_type": "all_reduce", "position": "tp",
                    "layer_idx": -1,
                    "attrs": {"batch": 1, "seq_len": tokens,
                              "msg_bytes": int(msg_bytes), "input_shape": list(shape)},
                    "dtype": _dtype_name(input_.dtype),
                })
                _GLOBAL["op_idx"] += 1
        except Exception:
            pass
        return fn(input_)
    _patched_all_reduce._operator_patched = True
    return _patched_all_reduce


def _patch_allreduce_binding(mod):
    fn = getattr(mod, "tensor_model_parallel_all_reduce", None)
    if fn is None or getattr(fn, "_operator_patched", False):
        return
    mod.tensor_model_parallel_all_reduce = _make_all_reduce_wrapper(fn)


def _emit_all_to_all(tensor, position):
    """Emit an `all_to_all` record for a real EP dispatch/combine call."""
    try:
        import torch
        if not isinstance(tensor, torch.Tensor):
            return
        shape = tuple(int(x) for x in tensor.shape)
        tokens = shape[0] if shape else 1
        msg_bytes = tensor.numel() * tensor.element_size()
        _emit({
            "step_id": _GLOBAL["step_id"], "op_idx": _GLOBAL["op_idx"],
            "op_type": "all_to_all", "position": position,
            "layer_idx": -1,
            "attrs": {"batch": 1, "seq_len": tokens,
                      "msg_bytes": int(msg_bytes), "input_shape": list(shape)},
            "dtype": _dtype_name(tensor.dtype),
        })
        _GLOBAL["op_idx"] += 1
    except Exception:
        pass


def _patch_group_coordinator(cls):
    """Patch GroupCoordinator.dispatch/combine to capture real EP all-to-all.

    These fire only when vLLM truly routes tokens across EP ranks (FusedMoE's
    do_naive_dispatch_combine / pplx / deepep paths). When they are not called
    (the common single-DP TP/EP case), nothing is emitted -- which matches nsys
    showing no all-to-all kernels there. dispatch records the input hidden_states
    volume; combine records the post-expert hidden_states volume.
    """
    disp = getattr(cls, "dispatch", None)
    if disp is not None and not getattr(disp, "_operator_patched", False):
        def _patched_dispatch(self, hidden_states, router_logits,
                              is_sequence_parallel=False):
            _emit_all_to_all(hidden_states, "dispatch")
            return disp(self, hidden_states, router_logits, is_sequence_parallel)
        _patched_dispatch._operator_patched = True
        cls.dispatch = _patched_dispatch

    comb = getattr(cls, "combine", None)
    if comb is not None and not getattr(comb, "_operator_patched", False):
        def _patched_combine(self, hidden_states, is_sequence_parallel=False):
            _emit_all_to_all(hidden_states, "combine")
            return comb(self, hidden_states, is_sequence_parallel)
        _patched_combine._operator_patched = True
        cls.combine = _patched_combine


def _patch_pstate_module(mod):
    cls = getattr(mod, "GroupCoordinator", None)
    if cls is not None:
        _patch_group_coordinator(cls)


# ---------------------------------------------------------------------------
# KV-transfer capture (prefill/decode disaggregation).
# In PD disaggregation the prefill instance sends its KV cache to the decode
# instance and the decode instance receives it. With the P2pNcclConnector this
# is NOT a model-forward kernel: the transfer is performed by the P2pNcclEngine,
# whose `send_tensor` (producer) and `recv_tensor` (consumer) move the real KV
# tensors (a per-request, per-layer KV slice). Neither goes through the patched
# collectives or a hookable nn.Module, so we patch these two engine methods to
# emit `kv_transfer` operator records carrying the REAL moved-tensor byte volume.
# Capture-only: a record is emitted only when a tensor is actually moved, so a
# non-disaggregated run emits nothing. We patch the engine (not the connector's
# save_kv_layer) because the connector sends an extracted slice, not the full
# paged kv_layer, so only the engine sees the true transferred bytes.
# ---------------------------------------------------------------------------

_P2P_ENGINE_MODULE = (
    "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine")


def _emit_kv_transfer(kv_tensor, position):
    """Emit a `kv_transfer` record for a real per-layer KV send/recv."""
    try:
        import torch
        if not isinstance(kv_tensor, torch.Tensor):
            return
        shape = tuple(int(x) for x in kv_tensor.shape)
        msg_bytes = kv_tensor.numel() * kv_tensor.element_size()
        _emit({
            "step_id": _GLOBAL["step_id"], "op_idx": _GLOBAL["op_idx"],
            "op_type": "kv_transfer", "position": position,  # "send" | "recv"
            "layer_idx": -1,
            "attrs": {"msg_bytes": int(msg_bytes), "input_shape": list(shape)},
            "dtype": _dtype_name(kv_tensor.dtype),
        })
        _GLOBAL["op_idx"] += 1
    except Exception:
        pass


def _patch_p2p_engine(mod):
    """Patch P2pNcclEngine.send_tensor/recv_tensor to capture real KV transfer.

    send_tensor (producer) receives the KV slice tensor as its 2nd positional arg;
    recv_tensor (consumer) returns the received tensor. We record the real moved
    bytes on each, only when an actual tensor is present (capture-only).
    """
    cls = getattr(mod, "P2pNcclEngine", None)
    if cls is None:
        return
    send = getattr(cls, "send_tensor", None)
    if send is not None and not getattr(send, "_operator_patched", False):
        def _patched_send(self, tensor_id, tensor, remote_address=None):
            result = send(self, tensor_id, tensor, remote_address)
            _emit_kv_transfer(tensor, "send")
            return result
        _patched_send._operator_patched = True
        cls.send_tensor = _patched_send

    recv = getattr(cls, "recv_tensor", None)
    if recv is not None and not getattr(recv, "_operator_patched", False):
        def _patched_recv(self, tensor_id, remote_address=None):
            out = recv(self, tensor_id, remote_address)
            tensor = out[0] if isinstance(out, tuple) and out else out
            _emit_kv_transfer(tensor, "recv")
            return out
        _patched_recv._operator_patched = True
        cls.recv_tensor = _patched_recv


# ---------------------------------------------------------------------------
# execute_model wrapper: one call == one forward step.
# ---------------------------------------------------------------------------

def _apply_runner_patch(runner_cls):
    if getattr(runner_cls.execute_model, "_operator_patched", False):
        return
    _orig = runner_cls.execute_model

    def _patched(self, *args, **kwargs):
        if not _GLOBAL["hooks_installed"]:
            _resolve_rank_path()   # distributed is up by the first forward step
            model = getattr(self, "model", None)
            if model is not None:
                _install_module_hooks(model)
        _GLOBAL["step_id"] += 1
        _GLOBAL["op_idx"] = 0
        return _orig(self, *args, **kwargs)

    _patched._operator_patched = True
    runner_cls.execute_model = _patched


class _RunnerFinder(importlib.abc.MetaPathFinder):
    # Core targets are always present in a vLLM run; the KV-connector targets are
    # OPTIONAL (only imported in PD disaggregation), so they are intercepted when
    # they appear but do NOT gate finder removal.
    _CORE_TARGETS = (_RUNNER_MODULE, _COMM_MODULE, _LINEAR_MODULE, _PSTATE_MODULE)
    _OPTIONAL_TARGETS = (_P2P_ENGINE_MODULE, _NPU_RUNNER_MODULE)
    _TARGETS = _CORE_TARGETS + _OPTIONAL_TARGETS

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _RunnerLoader(spec.loader, self, fullname)
        return spec


class _RunnerLoader(importlib.abc.Loader):
    def __init__(self, wrapped, finder, fullname):
        self.wrapped = wrapped
        self.finder = finder
        self.fullname = fullname

    def create_module(self, spec):
        if hasattr(self.wrapped, "create_module"):
            return self.wrapped.create_module(spec)
        return None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        if self.fullname == _RUNNER_MODULE and hasattr(module, "GPUModelRunner"):
            _apply_runner_patch(module.GPUModelRunner)
        elif self.fullname == _NPU_RUNNER_MODULE and hasattr(module, "NPUModelRunner"):
            # vllm-ascend: patch the subclass that actually overrides execute_model.
            _apply_runner_patch(module.NPUModelRunner)
        elif self.fullname == _COMM_MODULE:
            _patch_allreduce_binding(module)
        elif self.fullname == _LINEAR_MODULE:
            _patch_allreduce_binding(module)
        elif self.fullname == _PSTATE_MODULE:
            _patch_pstate_module(module)
        elif self.fullname == _P2P_ENGINE_MODULE:
            _patch_p2p_engine(module)
        # The finder stays installed for the whole process: the OPTIONAL
        # KV-connector target (p2p_nccl_engine) is imported LATE, during engine
        # init — well after the CORE targets — so removing the finder once the
        # core modules are present (the previous behavior) meant the connector
        # module was never intercepted in PD runs. The finder is a fast no-op for
        # non-target modules (one membership check), so keeping it is cheap.


def _resolve_rank_path():
    """Bind the output file to the worker's GLOBAL rank once distributed is up.

    Called at the first execute_model (after the process groups are initialized),
    so the file is named by rank (deterministic) rather than PID. Falls back to
    PID if rank cannot be resolved. The driver reads rank 0, matching the rank-0
    nsys comparison convention.
    """
    base = _GLOBAL["base_path"]
    rank = None
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        rank = None
    suffix = f"rank{rank}" if rank is not None else f"pid{os.getpid()}"
    _GLOBAL["path"] = f"{base}.{suffix}"


def install_operator_patch():
    path = os.environ.get("CUDA_OPERATOR_TRACE_PATH")
    if not path:
        return
    # The final per-rank file path is bound at the first forward step (see
    # _resolve_rank_path), once distributed process groups exist, so the driver
    # can deterministically read rank 0 (matching the rank-0 nsys comparison).
    _GLOBAL["base_path"] = path
    if _RUNNER_MODULE in sys.modules:
        mod = sys.modules[_RUNNER_MODULE]
        if hasattr(mod, "GPUModelRunner"):
            _apply_runner_patch(mod.GPUModelRunner)
    if _COMM_MODULE in sys.modules:
        _patch_allreduce_binding(sys.modules[_COMM_MODULE])
    if _LINEAR_MODULE in sys.modules:
        _patch_allreduce_binding(sys.modules[_LINEAR_MODULE])
    if _PSTATE_MODULE in sys.modules:
        _patch_pstate_module(sys.modules[_PSTATE_MODULE])
    # Optional KV-connector module (present only in PD disaggregation runs).
    if _P2P_ENGINE_MODULE in sys.modules:
        _patch_p2p_engine(sys.modules[_P2P_ENGINE_MODULE])
    if not all(m in sys.modules for m in _RunnerFinder._CORE_TARGETS):
        if not any(isinstance(f, _RunnerFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _RunnerFinder())


# Auto-install when imported (triggered by the .pth payload).
install_operator_patch()
