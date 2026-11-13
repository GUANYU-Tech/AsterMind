"""Compare analytical activation storage with existing startup records."""

from concurrent.futures import ThreadPoolExecutor
import ast
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import json
from pathlib import Path
import re
from time import perf_counter

from src.kernel.hardware_spec import HWSpec
from src.serving.kv_cache.activation_memory import (
    dense_decoder_mlp_activation_peak, symbolic_model_activation_peak, symbolic_startup_activation_peak,
    symbolic_v4_startup_activation_peak, symbolic_glm_startup_activation_peak,
)
from src.serving.kv_cache.kv_capacity import (
    load_runtime_memory_policy, project_symbolic_activation_policy, runtime_memory_policy_payload,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.kv_cache.model_residency import model_rank_residencies
from src.serving.workload.model_config import load_model_config


def compare_vllm030_worker_storage(upstream_root, *, tokens, max_requests, inputs_embeds_size, dtype):
    """Independent AST reference for unconditional known worker device arrays.

    Expected layouts use upstream expressions only. No source module/Tensor
    is executed, and no workload/lowering/predictor produces the reference.
    Block tables and conditional/runtime buffers are outside this scope.
    """
    from src.serving.workload.traced_forward import ForwardTrace
    from src.serving.workload.traced_sampling import VLLM030WorkerStorage
    root = Path(upstream_root)
    bindings = {"max_num_tokens": tokens, "max_num_reqs": max_requests, "inputs_embeds_size": inputs_embeds_size}
    names = dict(bindings)

    def value(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(value(child) for child in node.elts)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            return bindings[node.attr]
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max" and not node.keywords:
            return max(value(child) for child in node.args)
        raise ValueError("unsupported upstream worker storage metadata expression")

    expected, sources = {}, []
    for filename, class_name, prefix in (("gpu_model_runner.py", "GPUModelRunner", "profile."),
            ("gpu_input_batch.py", "InputBatch", "profile.input_batch.")):
        path = root / "vllm/v1/worker" / filename
        cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == class_name)
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        for statement in init.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target, call = statement.targets[0], statement.value
            if isinstance(target, ast.Name) and target.id == "arange_size":
                names[target.id] = value(call)
                continue
            if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == "self" and isinstance(call, ast.Call)):
                continue
            function = ast.unparse(call.func)
            maker = function == "self._make_buffer"
            if not maker and function not in ("torch.zeros", "torch.empty", "torch.full", "torch.ones"):
                continue
            keywords = {entry.arg: entry.value for entry in call.keywords}
            if not maker:
                if "device" not in keywords:
                    if "pin_memory" in keywords:
                        continue  # Pinned CPU tensors are not HBM copies.
                    raise ValueError("upstream worker tensor device is not explicit")
                device = keywords["device"]
                if isinstance(device, ast.Constant) and device.value == "cpu":
                    continue
                if ast.unparse(device) not in ("self.device", "device"):
                    raise ValueError("unsupported upstream worker tensor device expression")
            shape = tuple(value(arg) for arg in call.args)
            if len(shape) == 1 and isinstance(shape[0], tuple):
                shape = shape[0]
            item_dtype = {"self.dtype": dtype, "torch.int32": "int32", "torch.int64": "int64",
                "torch.float32": "fp32", "torch.float": "fp32", "torch.bool": "bool"}[ast.unparse(keywords["dtype"])]
            expected[prefix + target.attr] = {"shape": shape, "dtype": item_dtype, "source_line": statement.lineno}
        sources.append({"path": str(path), "symbol": class_name + ".__init__", "line": init.lineno})
        if class_name == "GPUModelRunner":
            maker = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_make_buffer")
            returned = next(node.value for node in maker.body if isinstance(node, ast.Return))
            if (not isinstance(returned, ast.Call) or ast.unparse(returned.func) != "CpuGpuBuffer"
                    or tuple(ast.unparse(arg) for arg in returned.args) != ("*size",)
                    or {entry.arg: ast.unparse(entry.value) for entry in returned.keywords}
                        != {"dtype": "dtype", "device": "self.device", "with_numpy": "numpy"}):
                raise ValueError("upstream worker buffer allocator no longer binds CpuGpuBuffer")
    utilities = root / "vllm/v1/utils.py"
    buffer_class = next(node for node in ast.parse(utilities.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "CpuGpuBuffer")
    gpu_assignment, = (node for node in ast.walk(buffer_class) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self.gpu" for target in node.targets))
    if ast.unparse(gpu_assignment.value) != "torch.zeros_like(self.cpu, device=device)":
        raise ValueError("CpuGpuBuffer no longer declares one same-layout device allocation")
    cpu_assignment, = (node for node in ast.walk(buffer_class) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self.cpu" for target in node.targets))
    call = cpu_assignment.value
    if not isinstance(call, ast.Call) or ast.unparse(call.func) != "torch.zeros":
        raise ValueError("CpuGpuBuffer source CPU layout is unknown")
    keywords = {entry.arg: entry.value for entry in call.keywords}
    if (tuple(ast.unparse(arg) for arg in call.args) != ("*size",)
            or ast.unparse(keywords["dtype"]) != "dtype"
            or not isinstance(keywords["device"], ast.Constant) or keywords["device"].value != "cpu"):
        raise ValueError("CpuGpuBuffer CPU/device layout forwarding changed")
    sources.append({"path": str(utilities), "symbol": "CpuGpuBuffer.__init__.gpu", "line": gpu_assignment.lineno})
    trace = ForwardTrace()
    owner = VLLM030WorkerStorage(trace, tokens=tokens, max_requests=max_requests, inputs_embeds_size=inputs_embeds_size, dtype=dtype)
    actual = {view.view_id: {"shape": view.shape, "dtype": view.dtype} for view in trace.views}
    checks = []
    for name in sorted(set(expected) | set(actual)):
        wanted = None if name not in expected else {key: expected[name][key] for key in ("shape", "dtype")}
        found = None if name not in actual else actual[name]
        checks.append({"buffer": name, "expected": wanted, "actual": found,
            "source_line": None if name not in expected else expected[name]["source_line"],
            "status": "matched" if wanted == found else "mismatched"})
    return {"scope": "unconditional runner and InputBatch sampling device arrays only; not full worker/native memory",
        "sources": sources, "checks": checks, "total": len(checks), "matched": sum(row["status"] == "matched" for row in checks),
        "failed": sum(row["status"] != "matched" for row in checks), "complete_worker_memory_accepted": False}


def assert_activation_alignment(predicted_bytes: int, observed_bytes: int) -> None:
    """User-requested 10% check; never adjusts the production estimate."""
    if type(predicted_bytes) is not int or predicted_bytes < 0 or type(observed_bytes) is not int or observed_bytes <= 0:
        raise ValueError("activation alignment requires nonnegative prediction and positive observation bytes")
    if abs(predicted_bytes - observed_bytes) * 10 > observed_bytes:
        raise AssertionError("analytical activation differs from startup record by more than10%")


def compare_dense_activation_records(model_path: str, policy_paths: tuple[str, ...], *, program="dense_mlp") -> dict[str, object]:
    """Read existing extracted startup records only at the evaluation boundary.

    The predictor receives model shape, parallelism and token capacity only.
    The initial program is a CUDA eager dense-MLP caller projection. It is
    deliberately compared to the complete recorded activation peak, so the
    report cannot hide missing attention/sampler/library allocations behind
    a successful partial-component check. Agreement alone does not prove
    complete liveness or a particular compiler/backend binding.
    """
    if not policy_paths:
        raise ValueError("activation comparison requires at least one existing policy record")
    if program == "dense_mlp":
        predict_peak = dense_decoder_mlp_activation_peak
    elif program == "symbolic_forward":
        predict_peak = symbolic_model_activation_peak
    else:
        raise ValueError("unknown activation source program")
    model = load_model_config(model_path)

    def compare(path):
        config = load_runtime_memory_policy(path)
        rows = []
        for policy in config.policies.values():
            if policy.binding.model_config_sha256 != model.config_sha256:
                raise ValueError("startup record model identity disagrees with analytical model")
            observed = next(reserve for reserve in policy.non_kv_reserves if reserve.kind == "activation")
            parallelism = ParallelConfig(tensor_parallel_size=policy.binding.tensor_parallel_size,
                pipeline_parallel_size=policy.binding.pipeline_parallel_size,
                enable_expert_parallel=policy.binding.expert_parallel_size > 1)
            fields = {"model": model_path, "runtime_memory_policy": str(Path(path).resolve()),
                "pool_name": policy.pool_name, "binding": asdict(policy.binding),
                "reference_kind": "existing_extracted_startup_record",
                "observed_activation_bytes": observed.bytes, "reference_provenance": asdict(observed.provenance),
                "reference_policy_evidence_status": policy.evidence_status}
            try:
                prediction = predict_peak(model, parallelism, tokens=policy.binding.max_batched_tokens)
            except NotImplementedError as error:
                rows.append({**fields, "status": "unsupported", "reason": str(error)})
                continue
            predicted = prediction["peak_bytes"]
            difference = abs(predicted - observed.bytes)
            try:
                assert_activation_alignment(predicted, observed.bytes)
            except AssertionError:
                status = "fail"
            else:
                status = "pass"
            rows.append({**fields, "status": status, "prediction": prediction,
                "absolute_error_bytes": difference, "relative_error": difference / observed.bytes,
                "complete_worker_memory_accepted": False})
        return rows

    with ThreadPoolExecutor(max_workers=min(8, len(policy_paths))) as executor:
        rows = [row for group in executor.map(compare, policy_paths) for row in group]
    return {"scope": program + " source projection versus complete startup activation records; not complete worker acceptance",
        "tolerance": "absolute(predicted-observed)/observed<=0.10", "cases": rows,
        "summary": {"total": len(rows), "pass": sum(row["status"] == "pass" for row in rows),
            "fail": sum(row["status"] == "fail" for row in rows),
            "unsupported": sum(row["status"] == "unsupported" for row in rows)},
        "all_models_migration_accepted": False}


def write_activation_comparison(model_path: str, policy_paths: tuple[str, ...], output: Path, *, program="dense_mlp") -> dict[str, object]:
    result = compare_dense_activation_records(model_path, policy_paths, program=program)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def parse_vllm030_startup_activation(log_text: str) -> dict[str, object]:
    """Separate graph estimate from the displayed peak using v0.30 semantics.

    gpu_worker.determine_available_memory assigns peak_activation_memory =
    transient_peak_headroom + cudagraph_memory_estimate_applied. The actual
    captured graph pool is a different value and must not be subtracted here.
    Accept one TP1 engine/summary only, and require the applied-estimate marker.
    Preserve the uncertainty of independently rounded GiB fields.
    """
    versions = re.findall(r"Initializing a V1 LLM engine \(v([^)]*)\)", log_text)
    if versions != ["0.30.0"]:
        raise ValueError("startup conversion requires exactly one vLLM 0.30.0 engine")
    if "CUDA graph memory profiling is enabled" not in log_text:
        raise ValueError("startup graph-estimate application is not established by the log")
    pools = re.findall(r"CUDA graph pool memory: ([0-9.]+) GiB \(actual\), ([0-9.]+) GiB \(estimated\)", log_text)
    peaks = re.findall(r"([0-9.]+) GiB for peak activation", log_text)
    modes = re.findall(r"enforce_eager=(True|False)", log_text)
    tensor_parallel = re.findall(r"tensor_parallel_size=([0-9]+)", log_text)
    if len(pools) != 1 or len(peaks) != 1 or len(modes) != 1 or tensor_parallel != ["1"]:
        raise ValueError("startup conversion requires one unambiguous pool/activation/execution summary")
    actual_text, estimate_text = pools[0]
    activation, estimate = Decimal(peaks[0]), Decimal(estimate_text)
    if activation <= 0 or estimate < 0 or estimate > activation:
        raise ValueError("startup activation and applied graph estimate are inconsistent")
    half_step = lambda number: Decimal(1).scaleb(number.as_tuple().exponent) / 2
    uncertainty = half_step(activation) + half_step(estimate)
    center = activation - estimate
    low, high = max(Decimal(0), center - uncertainty), center + uncertainty
    gib = Decimal(2**30)
    return {"vllm_version": versions[0], "enforce_eager": modes[0] == "True",
        "displayed_peak_activation_gib": peaks[0], "actual_graph_pool_gib": actual_text,
        "applied_graph_estimate_gib": estimate_text,
        "transient_peak_headroom_gib": str(center),
        "transient_peak_headroom_bytes_interval": (
            int((low * gib).to_integral_value(rounding=ROUND_FLOOR)),
            int((high * gib).to_integral_value(rounding=ROUND_CEILING))),
        "metric_source": "vllm/v1/worker/gpu_worker.py:determine_available_memory; peak_activation_memory=transient_peak_headroom+cudagraph_memory_estimate_applied",
        "scope": "evaluation-only interpretation of rounded logs; not a production reserve input"}


def compare_symbolic_startup_logs(model_path: str, log_paths: tuple[str, ...], *, tokens: int,
                                  max_requests: int, source_profile: str, hardware_config=None):
    """Compare startup headroom with a fully declared startup storage graph.

    This does not choose a provider, branch or formula to match an observation.
    It also reports whether a single deterministic number could satisfy every
    observation's 10% interval. Execution-state gaps remain explicit even when
    a number passes. The complete worker/profile program is still unaccepted.
    """
    if not log_paths or len(set(log_paths)) != len(log_paths):
        raise ValueError("startup comparison requires unique explicit log paths")
    startup = evaluate_symbolic_startup_storage(model_path, tokens=tokens, max_requests=max_requests,
        source_profile=source_profile, hardware_config=hardware_config)
    prediction = startup["prediction"]
    predicted = prediction["profile_storage_measurements"]["transient_source_headroom_bytes"]

    def compare(path):
        text = Path(path).read_text()
        parsed = parse_vllm030_startup_activation(text)
        reported_tokens = re.findall(r"max_num_batched_tokens['\"]?\s*[:=]\s*([0-9]+)", text)
        if not reported_tokens or {int(value) for value in reported_tokens} != {tokens}:
            raise ValueError("startup token capacity is missing or disagrees with the declared model profile")
        reported_requests = re.findall(r"max_num_seqs['\"]?\s*[:=]\s*([0-9]+)", text)
        if not reported_requests or {int(value) for value in reported_requests} != {max_requests}:
            raise ValueError("startup request capacity is missing or disagrees with the declared model profile")
        low, high = parsed["transient_peak_headroom_bytes_interval"]
        if low <= 0:
            numeric_status = "indeterminate"
        else:
            checks = []
            for observed in (low, high):
                try:
                    assert_activation_alignment(predicted, observed)
                except AssertionError:
                    checks.append(False)
                else:
                    checks.append(True)
            numeric_status = "pass" if all(checks) else "fail"
        # A prediction satisfying all rounded possibilities lies between
        # 90% of the upper observation and110% of the lower observation.
        interval = ((9 * high + 9) // 10, 11 * low // 10)
        return {"log_path": str(Path(path).resolve()), "observation": parsed,
            "predicted_bytes": predicted, "numeric_status": numeric_status,
            "prediction_interval_for_robust_ten_percent_bytes": interval,
            "enforce_eager_matches_source": parsed["enforce_eager"],
            "checkpoint_identity_aligned": False,
            "execution_configuration_aligned": False,
            "acceptance_status": "not_comparable",
            "unresolved_bindings": ("checkpoint/config contents", "native primitive/backend choices",
                "compiler lifetimes", "complete worker/profile storage"),
            "complete_worker_memory_accepted": False}

    with ThreadPoolExecutor(max_workers=min(8, len(log_paths))) as executor:
        rows = list(executor.map(compare, log_paths))
    common_low = max(row["prediction_interval_for_robust_ten_percent_bytes"][0] for row in rows)
    common_high = min(row["prediction_interval_for_robust_ten_percent_bytes"][1] for row in rows)
    return {"scope": "declared complete startup caller-storage headroom versus vLLM0.30 headroom; all raw observations retained",
        "model": model_path, "tokens": tokens, "max_requests": max_requests,
        "source_profile": source_profile, "prediction": prediction,
        "prediction_metric": "profile_storage_measurements.transient_source_headroom_bytes",
        "predicted_bytes": predicted,
        "cases": rows, "summary": {"total": len(rows),
            "pass": sum(row["numeric_status"] == "pass" for row in rows),
            "fail": sum(row["numeric_status"] == "fail" for row in rows),
            "indeterminate": sum(row["numeric_status"] == "indeterminate" for row in rows)},
        "acceptance_summary": {"total": len(rows), "pass": 0, "fail": 0, "not_comparable": len(rows)},
        "common_prediction_interval_bytes": None if common_low > common_high else (common_low, common_high),
        "all_observations_can_share_a_ten_percent_point": common_low <= common_high,
        "complete_worker_memory_accepted": False, "all_models_migration_accepted": False,
        "remaining_scope": ["compiled/IR execution differs from the declared eager source profile",
            "source-only native warmups do not bind a FlashInfer first call or actual library workspace/allocator state",
            "log-derived graph estimates never enter production activation prediction"]}


def _model_load_allocation_diagnostic(model, prediction, *, tensor_parallel_size, mtp_num_speculative_tokens,
        observed_increment_bytes, counter_reference):
    """Separate parameters, represented constructor buffers and a load counter.

    DeviceMemoryProfiler covers the complete load region. Its allocated-byte
    increase is not a pure parameter count, and this comparison never replaces
    a normal capacity reserve or assigns an unexplained remainder to an owner.
    """
    if (type(tensor_parallel_size) is not int or tensor_parallel_size != 1
            or type(mtp_num_speculative_tokens) is not int or mtp_num_speculative_tokens != 0):
        raise NotImplementedError("model-load diagnostic requires explicit TP1 and MTP0 source scope")
    if (observed_increment_bytes is None) != (counter_reference is None):
        raise ValueError("model-load observation requires both exact bytes and its source reference")
    if observed_increment_bytes is not None and (type(observed_increment_bytes) is not int
            or observed_increment_bytes < 0 or not isinstance(counter_reference, str) or not counter_reference):
        raise ValueError("model-load observation requires nonnegative integer bytes and an explicit source reference")
    residency, = model_rank_residencies(model, ParallelConfig(), worker_role="colocated", include_mtp=False)
    parameter_bytes = residency.weight_resident_bytes
    # Only model constructors belong to the measured loading region. Worker
    # input_ids/positions buffers are separate and must not be subtracted here.
    constructor_ids = {"model.rope-cache", "model.rope-interleaved-cache"}
    buffers = tuple(dict(item) for item in prediction["persistent_source_objects"]
        if item["allocation_id"] in constructor_ids)
    buffer_bytes = sum(item["bytes"] for item in buffers)
    if observed_increment_bytes is None:
        excess = unexplained = None
    else:
        excess = observed_increment_bytes - parameter_bytes
        unexplained = excess - buffer_bytes
        if unexplained < 0:
            raise ValueError("observed model-load increment is below its declared parameter and constructor-buffer payload")
    return {"tensor_parallel_size": tensor_parallel_size, "mtp_num_speculative_tokens": mtp_num_speculative_tokens,
        "physical_rank": residency.rank, "parameter_payload_bytes": parameter_bytes,
        "source_model_constructor_buffers": buffers, "source_model_constructor_buffer_bytes": buffer_bytes,
        "observed_model_loading_allocated_increase_bytes": observed_increment_bytes,
        "observed_counter_reference": counter_reference,
        "model_loading_increment_minus_parameter_bytes": excess,
        "unassigned_model_loading_increment_bytes": unexplained,
        "observed_counter_semantics": "allocated memory increase across the complete model-loading region; not pure parameters",
        "scope": "known source model buffers only; worker buffers, allocator and unrepresented runtime remain outside",
        "complete_model_load_memory_accepted": False, "normal_capacity_accepted": False}


def evaluate_symbolic_startup_storage(model_path, *, tokens, max_requests, source_profile,
        hardware_config=None, base_runtime_policy=None, tensor_parallel_size=1, source_max_model_len=None,
        source_use_deep_gemm=None, workspace_initial_capacity_bytes=None, indexer_max_logits_mb=None, expert_routing=None,
        framework_source_epoch=None, upstream_root=None, mtp_num_speculative_tokens=0,
        model_execution_version="vllm_0_30_0", source_lifetime_policy=None, source_compilation_options=None,
        observed_model_loading_allocated_increase_bytes=None, model_loading_counter_reference=None):
    """Report a fully declared storage scenario without inventing a log binding."""
    model = load_model_config(model_path)
    if model_execution_version not in ("vllm_0_11_0", "vllm_0_18_0", "vllm_0_30_0"):
        raise NotImplementedError("startup storage requires its version-bound source")
    ascend018 = source_profile == "vllm_ascend_0_18_no_metadata_head_only"
    load_scope_bound = (type(tensor_parallel_size) is int and tensor_parallel_size == 1
        and type(mtp_num_speculative_tokens) is int and mtp_num_speculative_tokens == 0)
    if ascend018 and not load_scope_bound and (observed_model_loading_allocated_increase_bytes is not None
            or model_loading_counter_reference is not None):
        raise NotImplementedError("model-load diagnostic requires explicit TP1 and MTP0 source scope")
    if not ascend018 and (observed_model_loading_allocated_increase_bytes is not None
            or model_loading_counter_reference is not None):
        raise ValueError("model-load counter comparison requires its explicit Ascend0.18 startup source")
    if (source_lifetime_policy is not None or source_compilation_options is not None) and not ascend018:
        raise ValueError("segment source options require the explicit Ascend0.18 source profile")
    if ascend018 != (model_execution_version == "vllm_0_18_0"):
        raise ValueError("Ascend0.18 startup requires its explicit head-only source profile and version")
    if ascend018 and (base_runtime_policy is not None or upstream_root is not None):
        raise ValueError("Ascend startup storage does not infer a runtime policy or CUDA worker inventory")
    if model_execution_version == "vllm_0_11_0" and (model.model_type not in ("qwen3", "llama")
            or source_profile not in ("cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise",
                "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first")
            or upstream_root is not None):
        raise NotImplementedError("startup0.11 supports explicit dense PyTorch-sort/FlashInfer0.4.1 storage;0.30 upstream comparison is not applicable")
    hardware = None
    if hardware_config is not None:
        hardware = HWSpec(str(hardware_config))
        if ascend018 and hardware.hardware_id != "ascend_910b":
            raise ValueError("Ascend0.18 startup requires the Ascend910B hardware identity")
        if not ascend018 and hardware.hardware_id not in ("nvidia_h200", "nvidia_a800"):
            raise NotImplementedError("CUDA sampler source storage requires a declared NVIDIA SM specification")
    started = perf_counter()
    projection = None
    v4_source = source_profile == "vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise"
    glm_source = source_profile in ("vllm_0_30_v32_bf16_sync_explicit_allreduce",
        "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    if type(mtp_num_speculative_tokens) is not int or mtp_num_speculative_tokens not in (0, 2):
        raise ValueError("startup source requires explicit zero or two speculative tokens")
    if not glm_source and mtp_num_speculative_tokens:
        raise ValueError("startup speculative passes require the GLM source")
    if base_runtime_policy is not None:
        if (v4_source or glm_source) and framework_source_epoch not in ("before_profile", "after_profile"):
            raise ValueError("V4/GLM policy requires an explicit framework source epoch; no complete runtime policy is inferred")
        if hardware is None:
            raise ValueError("derived runtime policy requires an explicit hardware specification")
        if not (v4_source or glm_source) and (tensor_parallel_size != 1 or any(value is not None for value in (
                source_max_model_len, source_use_deep_gemm, workspace_initial_capacity_bytes, indexer_max_logits_mb, expert_routing))):
            raise ValueError("dense startup source does not consume V4/TP workspace and routing options")
        config = load_runtime_memory_policy(base_runtime_policy)
        config.validate_scheduler_pools(("main",))
        base = config.policy_for_pool("main")
        if (tokens, max_requests) != (base.binding.max_batched_tokens, base.binding.max_requests):
            raise ValueError("startup source capacities disagree with the base policy binding")
        if (v4_source or glm_source) and source_max_model_len != base.binding.max_model_len:
            raise ValueError("V4/GLM startup maximum length must agree with the bound policy")
        policy, prediction = project_symbolic_activation_policy(base, model=model,
            parallelism=ParallelConfig(tensor_parallel_size=tensor_parallel_size), hardware=hardware,
            source_profile=source_profile, framework_source_epoch=framework_source_epoch,
            source_use_deep_gemm=source_use_deep_gemm, workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
            indexer_max_logits_mb=indexer_max_logits_mb, expert_routing=expert_routing,
            mtp_num_speculative_tokens=mtp_num_speculative_tokens, model_execution_version=model_execution_version)
        projection = {"base_policy_path": config.source_path, "base_policy_sha256": config.source_sha256,
            "evidence_status": policy.evidence_status, "payload": runtime_memory_policy_payload((policy,))}
    elif framework_source_epoch is not None:
        raise ValueError("framework source epoch requires an explicit base runtime policy")
    elif glm_source:
        if hardware is None:
            raise ValueError("GLM startup requires an explicit H200 specification")
        if source_use_deep_gemm is not None:
            raise ValueError("GLM startup does not consume the V4 DeepGEMM branch")
        if any(value is None for value in (source_max_model_len, workspace_initial_capacity_bytes,
                indexer_max_logits_mb, expert_routing)):
            raise ValueError("GLM startup requires explicit limits, workspace history, logits MiB and routing input")
        prediction = symbolic_glm_startup_activation_peak(model, ParallelConfig(tensor_parallel_size=tensor_parallel_size),
            hardware=hardware, pool_name="main", tokens=tokens, max_requests=max_requests,
            source_max_model_len=source_max_model_len, source_profile=source_profile,
            workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
            indexer_max_logits_mb=indexer_max_logits_mb, expert_routing=expert_routing,
            include_mtp=mtp_num_speculative_tokens > 0, num_speculative_tokens=mtp_num_speculative_tokens)
    elif v4_source:
        if hardware is None:
            raise ValueError("V4 startup requires an explicit H200 specification")
        if any(value is None for value in (source_max_model_len, source_use_deep_gemm,
                workspace_initial_capacity_bytes, indexer_max_logits_mb, expert_routing)):
            raise ValueError("V4 startup requires explicit limits, workspace history, logits MiB and routing input")
        prediction = symbolic_v4_startup_activation_peak(model, ParallelConfig(tensor_parallel_size=tensor_parallel_size),
            hardware=hardware, pool_name="main",
            tokens=tokens, max_requests=max_requests, source_max_model_len=source_max_model_len,
            source_use_deep_gemm=source_use_deep_gemm, workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
            indexer_max_logits_mb=indexer_max_logits_mb, source_profile=source_profile, expert_routing=expert_routing)
    elif tensor_parallel_size != 1 or any(value is not None for value in (source_max_model_len, source_use_deep_gemm,
            workspace_initial_capacity_bytes, indexer_max_logits_mb, expert_routing)):
        raise ValueError("dense startup source does not consume V4/TP workspace and routing options")
    else:
        if source_profile in ("cuda_eager_inplace_fa_no_metadata_pytorch_sort_fp32_noise",
                "cuda_eager_inplace_fa_no_metadata_flashinfer_0_4_1_top_k_first",
                "vllm_ascend_0_18_no_metadata_head_only"):
            num_compute_units = None
        elif source_profile == "cuda_eager_inplace_fa_no_metadata_native_triton_fp32_noise_cold_cache":
            if hardware is None:
                raise ValueError("Triton source profile requires an explicit hardware specification")
            num_compute_units = hardware.num_compute_units
        else:
            raise NotImplementedError("startup storage requires an explicit supported source profile")
        prediction = symbolic_startup_activation_peak(model, ParallelConfig(), tokens=tokens,
            max_requests=max_requests, source_profile=source_profile, num_compute_units=num_compute_units,
            model_execution_version=model_execution_version, source_lifetime_policy=source_lifetime_policy,
            source_compilation_options=source_compilation_options)
    result = {"model": model_path, "tokens": tokens, "max_requests": max_requests,
        "model_execution_version": model_execution_version, "source_lifetime_policy": source_lifetime_policy,
        "source_compilation_options": source_compilation_options,
        "prediction": prediction, "hardware_config": None if hardware_config is None else str(Path(hardware_config).resolve()),
        "runtime_policy_projection": projection,
        "cpu_trace_and_sweep_seconds": perf_counter() - started,
        "scope": "explicit source-only allocation scenario; no startup numeric/provider acceptance",
        "summary": {"source_programs_executed": 1, "normal_capacity_accepted": 0},
        "complete_worker_memory_accepted": False, "all_models_migration_accepted": False}
    if ascend018 and load_scope_bound:
        result["model_load_allocation_diagnostic"] = _model_load_allocation_diagnostic(model, prediction,
            tensor_parallel_size=tensor_parallel_size, mtp_num_speculative_tokens=mtp_num_speculative_tokens,
            observed_increment_bytes=observed_model_loading_allocated_increase_bytes,
            counter_reference=model_loading_counter_reference)
    elif ascend018:
        result["model_load_allocation_diagnostic_unavailable_reason"] = (
            "model-load counter/inventory diagnostic is bound only to TP1 MTP0; existing source execution is unchanged")
    if source_lifetime_policy is not None:
        measurements = prediction["profile_storage_measurements"]
        result["source_comparison_metric"] = {"name": "source_profile_peak_increase_bytes",
            "bytes": measurements["peak_source_storage_bytes"] - measurements["before_profile_source_bytes"],
            "reference_metric": "torch_peak_increase", "scope": "source-only vLLM0.18 peak-minus-before"}
    if upstream_root is not None:
        result["worker_storage_source_comparison"] = compare_vllm030_worker_storage(upstream_root,
            tokens=tokens, max_requests=max_requests, inputs_embeds_size=model.hidden_size, dtype=model.compute_dtype)
    return result


def write_symbolic_startup_storage_report(model_path, output, *, tokens, max_requests, source_profile,
        hardware_config=None, base_runtime_policy=None, derived_runtime_policy=None, **source_inputs):
    if (base_runtime_policy is None) != (derived_runtime_policy is None):
        raise ValueError("policy derivation requires both base and generated-policy paths")
    if base_runtime_policy is not None:
        inputs = tuple(Path(path).resolve() for path in (base_runtime_policy, model_path, hardware_config)
            if path is not None)
        targets = (output.resolve(), output.with_suffix(".md").resolve(), Path(derived_runtime_policy).resolve())
        if any(target in inputs for target in targets) or len(set(targets)) != len(targets):
            raise ValueError("generated policy/report paths must be distinct and preserve their source inputs")
    result = evaluate_symbolic_startup_storage(model_path, tokens=tokens,
        max_requests=max_requests, source_profile=source_profile, hardware_config=hardware_config,
        base_runtime_policy=base_runtime_policy, **source_inputs)
    if (derived_runtime_policy is not None and "worker_storage_source_comparison" in result
            and result["worker_storage_source_comparison"]["failed"]):
        raise ValueError("upstream worker storage comparison failed; refusing candidate policy export")
    if derived_runtime_policy is not None:
        derived_runtime_policy = Path(derived_runtime_policy)
        derived_runtime_policy.parent.mkdir(parents=True, exist_ok=True)
        derived_runtime_policy.write_text(json.dumps(result["runtime_policy_projection"]["payload"],
            indent=2, ensure_ascii=False) + "\n")
        load_runtime_memory_policy(derived_runtime_policy)
        result["derived_runtime_policy_path"] = str(derived_runtime_policy.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    prediction = result["prediction"]
    footprint = prediction["profile_storage_measurements"]
    lines = ["# 虚拟模型启动程序的存储峰值", "",
        "只执行Python元数据forward，不创建真实Tensor，不生成数值、权重或采样token。", "",
        "| 项目 | 结果 |", "|---|---|",
        f"| 模型配置 | {model_path} |",
        f"| 模型源码版本 | {result['model_execution_version']} |",
        f"| token容量 / 最大请求数 | {tokens} / {max_requests} |",
        f"| 声明的源码分支 | {source_profile} |",
        f"| 启动程序 | {prediction['source_program']} |",
        f"| 生命周期策略 | {prediction['lifetime_derivation']} |",
        f"| 激活storage峰值 | {prediction['peak_bytes']} B（{prediction['peak_bytes'] / 2**30:.6f} GiB） |",
        f"| profile范围源对象总峰值 | {footprint['peak_source_storage_bytes']} B |",
        f"| profile结束仍存活的源对象 | {footprint['after_profile_source_bytes']} B |",
        f"| 新增保留状态 / transient source headroom | {footprint['retained_source_growth_bytes']} / {footprint['transient_source_headroom_bytes']} B |",
        f"| 全rank图节点 / 峰值rank的storage记录 | {prediction['graph_nodes']} / {prediction['allocation_count']} |",
        f"| CPU建图及峰值扫描 | {result['cpu_trace_and_sweep_seconds'] * 1000:.3f} ms |",
        "| 正常容量 / 实际provider验收 | 未通过 / 未绑定 |", "",
        "同一次虚拟forward自动保留模型输出、gather结果及随机采样临时对象；按声明源码版本执行采样次数和对象释放顺序。峰值是单个物理rank独立live storage总量的最大值，不是组件或rank峰值相加。", "",
        "峰值时的独立存储：", "", "| storage | 字节 |", "|---|---:|"]
    lines.extend(f"| {item['allocation_id']} | {item['bytes']} |" for item in prediction["peak_live_allocations"])
    lines.extend(["", "逐物理rank的源对象峰值：", "", "| rank | 已表示激活峰值B | 范围源对象峰值B | 结束保留B | transient headroom B | 非驻留同时峰值B |", "|---|---:|---:|---:|---:|---:|"])
    for rank, record in prediction["ranks"].items():
        memory = record["profile_storage_measurements"]
        lines.append(f"| {rank} | {record['peak_bytes']} | {memory['peak_source_storage_bytes']} | {memory['after_profile_source_bytes']} | {memory['transient_source_headroom_bytes']} | {memory['nonretained_source_peak_bytes']} |")
    if "model_load_allocation_diagnostic" in result:
        load = result["model_load_allocation_diagnostic"]
        display = lambda value: "未提供/未归类" if value is None else str(value)
        lines.extend(["", "模型参数和加载counter（不改正常容量策略）：", "", "| 项目 | 字节 |", "|---|---:|",
            f"| 纯参数payload | {load['parameter_payload_bytes']} |",
            f"| 已表示的constructor持久buffer | {load['source_model_constructor_buffer_bytes']} |",
            f"| 实测整个model-load allocated增量 | {display(load['observed_model_loading_allocated_increase_bytes'])} |",
            f"| 尚未归类的model-load增量 | {display(load['unassigned_model_loading_increment_bytes'])} |",
            "", "加载counter包含该加载区间的参数及其他分配；不是纯权重。未归类差额不当作零或无来源reserve，worker/allocator/runtime总账仍未完成。"])
    lines.extend(["", "适用域是上表显式声明的eager/in-place、无attention metadata及native采样source profile。profile结束的保留状态由同一图的实际存活对象求和；transient headroom等于范围内总峰值减结束存活量，不能从所有时刻一律排除后来创建的cache。", "",
        "已有H200记录首轮使用FlashInfer并执行编译图；当前source-only场景不能当作其实际provider绑定，也不能用数值接近选择分支。0.30 Triton场景明确声明冷cache、SM数和两次native调用，0.11场景只有一次采样。显式FlashInfer0.4.1分支覆盖top-k-first的masked logits、softmax和INT32输出存储，不替仅记录FlashInfer名称的历史日志补造库版本；其他库workspace及编译后生命周期仍不在此声明范围。", "",
        "RoPE缓存和已预分配输入缓冲单列为persistent_source_objects；它们不是完整worker持久状态。尚未覆盖：" + "、".join(prediction["uncovered"]) + "。未知部分没有补零或从日志反推。", "",
        "该图只补充存储语义；缺少完整数学计价契约的storage-only节点被production lowering明确拒绝，不会伪装成零成本或通用elementwise计价。", ""])
    if source_profile == "vllm_0_30_v4_sync_no_metadata_pytorch_sort_fp32_noise":
        lines.extend(["V4复用普通模型forward、完整层栈及共同worker/sampler程序；仅attention helper替换为源码无metadata分支：跳过cache insert/compressor save/finish/score/topk/attention，保留输入、索引器query和输出投影，预留shared workspace并zero输出。C4不走短上下文捷径；UINT8 dummy logits的MiB及workspace入口历史均为显式条件输入。没有根据启动log选择分支或大小。", ""])
    if source_profile in ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce"):
        lines.extend(["GLM复用constructor尺寸的完整target、可选两次顺序MTP draft与共同worker/sampler存储程序；MTP选择见JSON source_inputs。路由和workspace历史是显式条件，并未证明所有路由的最大workspace；官方FP8驻留、稀疏索引组/推测状态、实际library/compiler/runtime/allocator仍保留独立缺口。此source-only导出不调用native allocator，不代表正常容量通过。", ""])
    if "worker_storage_source_comparison" in result:
        comparison = result["worker_storage_source_comparison"]
        lines.extend([f"独立upstream AST检查：已知worker/input-batch device buffers {comparison['matched']}/{comparison['total']}匹配，失败{comparison['failed']}。只覆盖无条件预分配的声明对象，不代表完整worker或native显存通过；逐项shape/dtype和源码位置见JSON。", ""])
    if derived_runtime_policy is not None:
        budget = prediction["activation_policy_budget"]
        lines.extend(["生成的现有schema2候选策略：[JSON](" + str(derived_runtime_policy.resolve()) + ")。activation=" + str(budget["bytes"])
            + "B，framework=" + str(budget["framework_bytes"]) + "B，基线framework明确对应" + budget["framework_source_epoch"]
            + "。结束驻留storage只计framework，非驻留同时峰值计activation；before_profile基线增加已表示驻留增长，after_profile基线不重复增加。二者组成声明源码程序的预算界，不是已观测点估计或未知warmup/backend的界。其他预留项继承输入，整体保持unverified_analytical_assumption，不是normal容量或实际runtime验收；基线文件未改写。", ""])
    output.with_suffix(".md").write_text("\n".join(lines))
    return result
