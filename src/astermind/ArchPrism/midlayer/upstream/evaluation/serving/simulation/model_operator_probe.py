"""Static operator evidence for independent source comparison, without gates."""

from collections.abc import Mapping
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from math import isfinite
import json
import hashlib
import re
import shlex
import subprocess
import sys

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor, predict
from src.kernel.kernel_spec import KernelKind
from src.kernel.predictors.gemm_block_fp8 import is_block_fp8_gemm
from src.kernel.predictors.fused_moe import dsv4_marlin_moe_source_work, dsv4_marlin_moe_roofline_work, dsv4_marlin_moe_scratch_work
from src.kernel.predictors.sparse_attn_indexer import dsv4_indexer_logical_work
from src.kernel.predictors.kv_cache_update import (
    dsv4_finish_logical_work, dsv4_swa_insert_tensor_objects,
    dsv4_partial_state_tensor_objects, dsv4_finish_partial_state_input,
)
from src.kernel.predictors.sparse_attn import dsv4_prefill_gather_logical_work, dsv4_attention_source_inputs, dsv4_decode_index_materialization_work, dsv4_decode_attention_index_inputs
from src.kernel.predictors.vector import (
    mhc_prenorm_logical_work, mhc_prenorm_roofline_work, mhc_stage_logical_work,
    dsv4_indexer_q_rope_quant_logical_work, dsv4_indexer_q_roofline_work,
    clamped_silu_logical_work, rmsnorm_logical_work,
    dsv4_moe_route_logical_work, dsv4_route_source_work, dsv32_moe_route_logical_work,
    dsv4_short_context_topk_source_work,
    dsv4_route_tensor_objects,
    kimi_moe_route_logical_work, vocab_parallel_embedding_logical_work,
    dsv32_draft_sampling_logical_work, vllm_logits_row_selection_logical_work,
    fp8_group_quant_source_work,
    vllm_logits_row_selection_roofline_bytes,
    vllm_logits_fp32_cast_logical_work, vllm_logits_gather_reorder_logical_work,
    vllm_plain_greedy_sampling_logical_work, vllm_plain_greedy_roofline_work,
    vllm_rejection_logits_materialization_logical_work,
    DSV32_DRAFT_MATERIALIZATION_VARIANTS, dsv32_draft_materialization_roofline_bytes,
    DSV32_DRAFT_INT32_BOOKKEEPING_VARIANTS, dsv32_draft_int32_bookkeeping_roofline_work)
from src.serving.deployment.deployment_config import (
    CollectiveConfig, DeploymentConfig, ParallelConfig, PipelineP2PConfig, WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kernel_lowering.execution_timeline import build_execution_timeline
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
from src.serving.kv_cache.model_residency import (
    dsv4_non_routed_loader_payload_bytes, glm_block_fp8_loader_payload_bytes,
    glm_block_fp8_constructor_buffer_bytes,
    glm_block_fp8_postload_parameter_bytes, GLM_SM90_DEEP_GEMM_STORAGE_SOURCE,
    glm_block_fp8_deepgemm_workspace_bytes,
    glm_block_fp8_deepgemm_call_allocation_bytes,
    dsv4_routed_loader_payload_bytes, kimi_k25_outer_loader_payload_bytes,
    model_rank_residencies)
from src.serving.workload.traced_sampling import (
    glm_gpu_v2_mtp_storage_work, glm_gpu_v2_prepare_decode_work,
    glm_gpu_v2_slot_mapping_work, glm_gpu_v2_metadata_refresh_plan,
    glm_gpu_v2_sparse_backend_metadata_plan)
from src.serving.workload.model_config import (
    load_model_config, GLM_BLOCK_FP8_CONFIG_SHA256S, is_validated_qwen3_moe_ep2_model_scope)
from src.serving.workload.runtime_inputs import (ExpertRoutingInput, RuntimeInputs, V32PaddedMTPInputs,
    vllm_0_30_source_model_inputs)
from src.serving.workload.traced_models import build_symbolic_workload


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"operator evidence cannot encode {type(value).__name__}")


def _snapshot():
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "uncommitted_changes": subprocess.check_output(
            ["git", "status", "--short"], text=True).splitlines(),
    }


def write_kimi_k3_nope_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-prefill")


def write_kimi_k3_nope_decode_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-decode")


def write_kimi_k3_nope_mixed_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-mixed")


def write_kimi_k3_kda_mixed_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "kda-mixed")


def write_kimi_k3_mixed_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "mixed-greedy-step")


def write_kimi_k3_kda_context_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "kda-context-prefill")


def write_kimi_k3_nope_context_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-context-prefill")


def write_kimi_k3_context_prefill_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "context-prefill-greedy-step")


def write_kimi_k3_nope_context_continuation_components(model_path, source_root, output, *, backend_supports_out):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-context-continuation",
                                      backend_supports_out=backend_supports_out)


def write_kimi_k3_context_continuation_greedy_step_components(model_path, source_root, output, *, backend_supports_out):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "context-continuation-greedy-step",
                                      backend_supports_out=backend_supports_out)


def write_kimi_k3_native_latent_output_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "native-latent-output")


def write_kimi_k3_marlin_routed_output_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "marlin-routed-output")


def write_kimi_k3_native_marlin_moe_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "native-marlin-moe")


def write_kimi_k3_nope_prefill_layer_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "nope-prefill-layer")


def write_kimi_k3_kda_fresh_prefill_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "kda-fresh-prefill")


def write_kimi_k3_dense_kda_prefill_layer_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "dense-kda-prefill-layer")


def write_kimi_k3_fresh_prefill_decoder_stack_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-decoder-stack")


def write_kimi_k3_fresh_prefill_logits_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-logits")


def write_kimi_k3_fresh_prefill_forward_logits_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-forward-logits")


def write_kimi_k3_single_decode_forward_logits_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "single-decode-forward-logits")


def write_kimi_k3_fresh_prefill_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "fresh-prefill-greedy-step")


def write_kimi_k3_single_decode_greedy_step_components(model_path, source_root, output):
    return _write_kimi_k3_source_chain(model_path, source_root, output, "single-decode-greedy-step")


def write_kimi_k3_situ_offline_compile(model_path, source_root, output):
    """Compile the unchanged activation TU; publish conditional static facts only.

    No GPU context/module, measured duration or production parameter is
    created. The task-local SDK supplies real Torch headers. The local
    CUDA12.2/SM90 experiment is explicitly not an observed runtime build.
    """
    import ast
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity

    model = load_model_config(model_path)
    if model.kimi_k3_semantics is None:
        raise ValueError("SiTU compiler probe requires the exact K3 config")
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    identity = vllm030_source_identity(source_root)
    directory = output.with_suffix("")
    directory.mkdir(parents=True, exist_ok=True)
    sdk = Path(".venv-vllm030/bin/python").absolute()
    # Inspect the installed module's location without importing Torch/CUDA.
    torch_root = Path(subprocess.check_output([
        str(sdk), "-c", "import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('torch').origin).parent)"
    ], text=True).strip())
    module = ast.parse((torch_root / "utils/cpp_extension.py").read_text())
    flags = None
    for node in module.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "COMMON_NVCC_FLAGS"
                                               for target in node.targets):
            flags = ast.literal_eval(node.value)
    if flags is None or not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError("Torch COMMON_NVCC_FLAGS is not a literal flag list")
    cmake = (source_root / "cmake/utils.cmake").read_text()
    removed = ("-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
               "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__")
    if "if (CUDA_VERSION VERSION_GREATER_EQUAL 12.0)" not in cmake or any(flag not in cmake for flag in removed):
        raise ValueError("verified source does not bind the CUDA>=12.0 Torch-flag removal")
    nvcc = Path("/usr/local/cuda-12.2/bin/nvcc")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 12.2," not in version:
        raise ValueError("this offline source experiment requires the declared CUDA12.2 toolchain")
    command = [str(nvcc), "--cubin", "-std=c++20", "-arch=sm_90", "-O3",
               "-DTORCH_TARGET_VERSION=0x020B000000000000ULL", "-DUSE_CUDA", "-DENABLE_FP8",
               *(flag for flag in flags if flag not in removed),
               f"-I{source_root / 'csrc'}", f"-I{torch_root / 'include'}",
               f"-I{torch_root / 'include/torch/csrc/api/include'}", "-o", str(directory / "activation.cubin"),
               str(source_root / "csrc/libtorch_stable/activation_kernels.cu")]
    with (directory / "compile.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    metadata = {"snapshot": _snapshot(), "source_identity": identity, "command": command,
                "compiler_version": version, "torch_headers": str(torch_root / "include"),
                "removed_flags": removed, "source_transform": None, "exit_code": result.returncode,
                "GPU_module_loaded_or_run": False, "actual_runtime_build_bound": False}
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if result.returncode != 0:
        raise RuntimeError(f"unchanged activation TU compilation failed; see {directory / 'compile.log'}")
    dump = subprocess.check_output([str(nvcc.parent / "cuobjdump"), "--dump-sass", str(directory / "activation.cubin")], text=True)
    (directory / "activation.sass").write_text(dump)
    instances = []
    for part in dump.split("Function : ")[1:]:
        symbol, body = part.split("\n", 1)
        if not symbol.startswith("_ZN4vllm19situ_and_mul_kernel"):
            continue
        opcodes = Counter(re.findall(r"/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s", body))
        instances.append({"symbol": symbol.strip(), "static_opcode_counts": dict(opcodes),
                          "magnitude_branch_compares": [line.strip() for line in body.splitlines()
                                                       if "FSETP.GE" in line and "0.60000002384185791016" in line]})
    bf16 = [instance for instance in instances if "BFloat16" in instance["symbol"]]
    if len(bf16) != 1 or not bf16[0]["magnitude_branch_compares"]:
        raise ValueError("offline BF16 ordinary SiTU magnitude-branch evidence not identified")
    metadata.update(component_count=len(instances), instances=instances,
        scope="local_CUDA12.2_SM90_default_math_complete_activation_TU_static_compiler_facts",
        dynamic_instruction_counts_inferred=False, provider_accepted=False, runtime_accepted=False,
        conditional_scalar_examples=[
            {"gate": 0, "up": 0, "tanh_arguments": [0, 0, 0], "magnitude_ge_0_6": [False, False, False]},
            {"gate": 4, "up": 25, "tanh_arguments": [1, 2, 1], "magnitude_ge_0_6": [True, True, True]}],
        scalar_example_scope="constructed finite BF16-representable source predicate examples; not checkpoint/model/GPU outputs",
        production_parameters_changed=False,
        conclusion="shape and three tanhf calls do not fix dynamic SFU demand; preserve unpriced boundary",
        limitations=["compiler flags derived for this local toolchain, not an observed installed vLLM runtime",
                     "static instruction counts include loops/alternate paths and are not per-element dynamic counts",
                     "input magnitude activity and actual toolchain/provider binding remain unobserved",
                     "no latency or issue rate inferred; no GPU module loaded or executed"])
    output.write_text(json.dumps(_json_value(metadata), indent=2) + "\n")
    return metadata


def _write_kimi_k3_source_chain(model_path, source_root, output, component, *, backend_supports_out=None):
    """One new source branch; do not rerun previously published components."""
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
    from src.serving.workload.kimi_k3 import (
        vllm030_kimi_k3_nope_fresh_prefill_workload, vllm030_kimi_k3_nope_decode_workload,
        vllm030_kimi_k3_nope_mixed_workload, vllm030_kimi_k3_triton_mixed_workload,
        vllm030_kimi_k3_mixed_forward_logits_workload, vllm030_kimi_k3_triton_context_prefill_workload,
        vllm030_kimi_k3_nope_context_prefill_workload, vllm030_kimi_k3_context_prefill_forward_logits_workload,
        vllm030_kimi_k3_tp1_native_latent_output_workload,
        vllm030_kimi_k3_marlin_routed_output_workload,
        vllm030_kimi_k3_native_marlin_moe_workload, vllm030_kimi_k3_nope_prefill_layer_workload,
        vllm030_kimi_k3_triton_fresh_prefill_workload, vllm030_kimi_k3_dense_fresh_prefill_layer_workload,
        vllm030_kimi_k3_fresh_prefill_decoder_stack_workload, vllm030_kimi_k3_fresh_prefill_logits_workload,
        vllm030_kimi_k3_fresh_prefill_forward_logits_workload,
        vllm030_kimi_k3_single_decode_forward_logits_workload, vllm030_kimi_k3_append_plain_greedy_sampler)
    from src.kernel.predictors.vector import (
        kimi_k3_nope_prefill_logical_work, kimi_k3_nope_decode_logical_work, kimi_k3_nope_mixed_logical_work,
        kimi_k3_nope_context_logical_work,
        kimi_k3_moe_route_logical_work,
        kimi_k3_situ_logical_work, kimi_k3_attention_residual_logical_work,
        kimi_k3_kda_prefill_logical_work, kimi_k3_kda_mixed_logical_work, kimi_k3_kda_context_prefill_logical_work,
        kimi_k3_kda_decode_logical_work, vllm_embedding_id_widen_logical_work,
        vllm_plain_greedy_sampling_logical_work)
    from src.kernel.predictors.fused_moe import kimi_k3_marlin_expert_logical_work
    from src.kernel.predictors.gemm import kimi_k3_nope_bmm_logical_work
    from src.kernel.predictors.flash_attn import kimi_k3_nope_context_attention_logical_work

    model = load_model_config(model_path)
    identity = vllm030_source_identity(Path(source_root))
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (WorkerPoolConfig("main", "colocated", "nvidia_h200"),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    if component in ("single-decode-forward-logits", "single-decode-greedy-step"):
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"input_ids": (1,), "sequence_length": 513, "state_pool_slots": 2,
                    "state_indices_by_layer": {layer: 1 for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: (512,) for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "cache_capacity_tokens": 1024, "shared_experts_stream_token_threshold": 256,
                    "aux_streams_available": True, "pool_name": "main", "rank": 0}
        step_id = "source-k3-single-decode-stack"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=1, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        graph = vllm030_kimi_k3_single_decode_forward_logits_workload(model, routing_by_layer=routes, **scenario)
        scope = "TP1_BF16_single_request_context513_embedding_93_layer_decode_and_logits_source_candidate"
        limitations = [
            "input token and every route/cache address are declared source inputs, not model or observed output",
            "one continuation token; independent logical per-layer cache namespaces, not physical shared-pool allocations",
            "explicit ordinary auxiliary streams; native CUDA streams/events and compiler/provider unbound",
            "router/down join precedes external shared/routed overlap; shared completion waits before latent transform",
            "NoPE short frontend retains QKV/gate slices, absorption,576 query/cache and V up-projection",
            "all92 histograms retain seeded-assumption provenance, not checkpoint logits",
            "sampler, metadata/index producers, normal capacity and normalized Serving bridge remain outside this graph"]
    elif component in ("fresh-prefill-decoder-stack", "fresh-prefill-forward-logits", "fresh-prefill-greedy-step"):
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"query_start_loc": (0, 512), "state_pool_slots": 2,
                    "state_indices_by_layer": {layer: 1 for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: tuple(range(512)) for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "cache_capacity_tokens": 512, "shared_experts_stream_token_threshold": 256,
                    "pool_name": "main", "rank": 0}
        step_id = "source-k3-fresh-prefill-stack"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=512, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        if component == "fresh-prefill-decoder-stack":
            graph = vllm030_kimi_k3_fresh_prefill_decoder_stack_workload(model, routing_by_layer=routes, **scenario)
        else:
            graph = vllm030_kimi_k3_fresh_prefill_forward_logits_workload(
                model, input_ids=tuple(range(512)), routing_by_layer=routes,
                **{key: value for key, value in scenario.items() if key != "query_start_loc"})
            scenario["input_ids"] = tuple(range(512))
        scope = "TP1_T512_fresh_93_layer_hybrid_decoder_stack_plus_output_AttnRes_source_candidate"
        limitations = [
            "input embedding hidden states external; excludes row selection, final norm, lm_head and sampler",
            "separate logical per-layer cache namespaces; declared slots are not actual shared native allocations",
            "all92 histograms retain seeded-assumption provenance, not checkpoint-computed logits or observed IDs",
            "one fresh full64-chunk sequence; no context/decode/mixed, MTP, auxiliary capture or distributed execution",
            "explicit Triton KDA/standard Marlin source candidates; actual backend/provider dispatch unbound",
            "chunk, SiTU, route, expert aggregate and NoPE sigmoid gate remain unpriced",
            "source node/aggregate counts are not GPU launch counts; source graph is not a complete model"]
        if component in ("fresh-prefill-forward-logits", "fresh-prefill-greedy-step"):
            scope = "TP1_T512_fresh_embedding_93_layer_stack_and_selected_row_logits_source_candidate"
            limitations[0] = "includes INT32-to-INT64 embedding IDs, BF16 embedding output and selected-row norm/head; excludes sampler"
    elif component == "fresh-prefill-logits":
        scenario = {"query_start_loc": (0, 512), "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_fresh_prefill_logits_workload(model, **scenario)
        scope = "TP1_T512_single_fresh_sequence_row_selection_final_RMSNorm_and_BF16_lm_head"
        limitations = [
            "external stack pre-norm hidden output; last-row selection precedes final norm and head",
            "one INT32 query boundary without speculation, prompt logprobs, pooling or overrides",
            "sampler, index producer/H2D and embedding input conversion are outside this component",
            "row selection logical-copy Roofline does not establish actual library/compiler launch",
            "generic analytical norm/head do not bind actual H200 provider"]
    elif component == "nope-decode":
        scenario = {"layer_index": 3, "slot_mapping": (512,), "kv_lens": (513,), "cache_capacity_tokens": 1024,
                    "cache_dtype": "bf16", "aux_stream_available": True, "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_nope_decode_workload(model, **scenario)
        scope = "TP1_BF16_NoPE_MLA_single_decode_sequence513_source_candidate"
        limitations = [
            "declared cache slots, sequence lengths and cache contents, not native Scheduler/metadata execution",
            "model source owns a shared auxiliary stream; T<512 slices QKV/gate with an output-gate join",
            "short _unquantized_gemm low-latency-vs-F.linear dispatch remains unbound",
            "BMM absorbs all Q rows; 576-wide query concat/cache insert precedes latent attention",
            "W_UV BMM writes the attention output directly, without a separate copy",
            "decode concat CUDA source uses3 cooperating warps per row at T<=64 and1 above64",
            "canonical analytical MLA decoder/split selector is not actual0.30 backend/provider execution",
            "compiled BF16 sigmoid output gate retains an unpriced materialization/compiler boundary"]
    elif component == "mixed-greedy-step":
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"input_ids": (1, *range(512)), "decode_kv_len": 513, "state_pool_slots": 3,
                    "state_indices_by_layer": {layer: (1, 2) for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: (512, *range(1024, 1536))
                                              for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "cache_capacity_tokens": 2048, "shared_experts_stream_token_threshold": 256,
                    "aux_streams_available": True, "pool_name": "main", "rank": 0}
        step_id = "source-k3-mixed-greedy"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=513, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        graph = vllm030_kimi_k3_mixed_forward_logits_workload(model, routing_by_layer=routes, **scenario)
        scope = "TP1_embedding_93_layer_mixed_one_decode_one_fresh512_and_two_selected_row_logits_source_candidate"
        limitations = [
            "declared input IDs and per-layer cache slots, not native Scheduler/metadata or computed cache values",
            "all92 expert histograms retain seeded-assumption provenance, not checkpoint logits or observed IDs",
            "KDA uses mixed chunk for decode1 and fresh512; NoPE shares frontend and runs prefill first",
            "row0 androw512 selected before two-row final norm/head; excludes sampler",
            "ordinary aux streams declared; no overlap discount or actual provider/backend execution",
            "chunk, SiTU, route, expert aggregate and NoPE output gate remain unpriced",
            "no context prefill, speculation, MTP, distributed or normal H200 capacity acceptance"]
    elif component in ("context-prefill-greedy-step", "context-continuation-greedy-step"):
        kinds = model.kimi_k3_semantics.layer_attention_types
        scenario = {"input_ids": tuple(range(512)), "context_length": 1024, "state_pool_slots": 2,
                    "state_indices_by_layer": {layer: 1 for layer, kind in enumerate(kinds) if kind == "kda"},
                    "slot_mapping_by_layer": {layer: tuple(range(1024, 1536))
                                              for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "context_block_tables_by_layer": {layer: tuple(range(16))
                                                      for layer, kind in enumerate(kinds) if kind == "mla_nope"},
                    "block_size": 64, "cache_capacity_tokens": 2048, "shared_experts_stream_token_threshold": 256,
                    "aux_streams_available": True, "prefill_source": "declared_fa3_bf16", "pool_name": "main", "rank": 0}
        if component == "context-continuation-greedy-step":
            if type(backend_supports_out) is not bool:
                raise ValueError("context continuation report requires explicit backend supports_out")
            scenario.update(context_chunk_lengths=(512, 512), backend_supports_out=backend_supports_out)
        step_id = "source-k3-context-greedy"
        declared = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0)
        routes = {layer: declared.resolve(
            step_id=step_id, layer_index=layer, moe_invocation_id=f"{step_id}:layer-{layer}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=512, num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
                  for layer in range(1, model.num_hidden_layers)}
        graph = vllm030_kimi_k3_context_prefill_forward_logits_workload(model, routing_by_layer=routes, **scenario)
        scope = "TP1_embedding_93_layer_initialized_context1024_new512_and_selected_row_logits_source_candidate"
        limitations = [
            "declared prefix cache/state contents, slots and input IDs, no native Scheduler context trace",
            "all92 route histograms retain seeded-assumption provenance, not checkpoint logits or observed IDs",
            "KDA reads existing conv/FP32 recurrent state; every NoPE layer gathers historical latent and merges two LSE partials",
            "single complete prefix chunk covers all new rows; no continuation chunks, empty slices, speculation or DCP",
            "canonical analytical FA3 reuse is explicit for new tokens, noncausal context backend remains unbound",
            "last query row511 selected before final norm/head; excludes sampler",
            "chunk/SiTU/route/expert/context attention/merge/output gate remain unpriced",
            "CPU MLA disables chunked prefill; source scenario is not actual Scheduler/normal capacity/provider acceptance"]
        if component == "context-continuation-greedy-step":
            scope += "__two512_context_chunks_explicit_supports_out"
            limitations.append("continuation merges in-place BF16 partial and FP32 LSE; first chunk copies depend on declared supports_out")
            limitations[3] = "one prefix request with two512 chunks; no empty slices, speculation or DCP"
    elif component in ("nope-context-prefill", "nope-context-continuation"):
        scenario = {"layer_index": 3, "slot_mapping": tuple(range(1024, 1536)), "context_length": 1024,
                    "context_block_table": tuple(range(16)), "block_size": 64, "cache_capacity_tokens": 2048,
                    "cache_dtype": "bf16", "prefill_source": "declared_fa3_bf16", "pool_name": "main", "rank": 0}
        if component == "nope-context-continuation":
            if type(backend_supports_out) is not bool:
                raise ValueError("context continuation report requires explicit backend supports_out")
            scenario.update(context_chunk_lengths=(512, 512), backend_supports_out=backend_supports_out)
        graph = vllm030_kimi_k3_nope_context_prefill_workload(model, **scenario)
        scope = "TP1_BF16_NoPE_one_complete_prefix1024_chunk_and_new512_source_candidate"
        limitations = [
            "declared cache blocks/slots/contents and source backend profile, not native metadata or observed provider",
            "one complete prefix chunk starts0 and covers all queries; no continuation/empty slices/DCP",
            "new causal FA3 analytical source includes FP32 LSE output in the same call",
            "context gather BF16-to-BF16 has no scale/token_to_seq read; latent cast aliases BF16 workspace",
            "history KV-B and key concat preserve strided V view; no context accumulator or output copy",
            "noncausal context attention exports tensor math/typed roles, backend softmax/tile/split remains unpriced",
            "merge LSE value branches/per-pack compiler work and BF16 output gate remain unpriced",
            "CPU platform disables MLA chunked prefill, so this source scenario is not a Scheduler context trace"]
        if component == "nope-context-continuation":
            scope += "__two512_context_chunks_explicit_supports_out"
            limitations[1] = "one prefix request split into two512 chunks, second a continuation; no empty slices/DCP"
            limitations[4] = "strided V aliases history KV-B; first output copy omitted only when declared supports_out"
            limitations.append("first chunk copies FP32 LSE; continuation merges in-place output+LSE, final suffix merge does not write LSE")
    elif component == "kda-context-prefill":
        scenario = {"layer_index": 0, "query_start_loc": (0, 128), "state_index": 1,
                    "state_pool_slots": 2, "has_initial_state": True,
                    "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_triton_context_prefill_workload(model, **scenario)
        scope = "TP1_Triton_KDA_one_initialized128_token_context_extension_source_candidate"
        limitations = [
            "declared initial state/index and full64-chunk query; prior length/cache values external, not a Scheduler trace",
            "three conv_fn calls read prior3 history values per channel; full query needs no short-history shift",
            "gather reads one FP32 pool row and writes one dense row; chunk final state scatters into the same pool slot",
            "multi-kernel chunk internal math/traffic remains unpriced, with no recurrence substitution",
            "not a complete hybrid decoder context graph, actual backend/provider or normal capacity acceptance"]
    elif component == "kda-mixed":
        scenario = {"layer_index": 0, "query_start_loc": (0, 1, 513), "state_indices": (1, 2),
                    "state_pool_slots": 3, "has_initial_state": (True, False),
                    "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_triton_mixed_workload(model, **scenario)
        scope = "TP1_Triton_KDA_one_initialized_decode_one_fresh512_shared_chunk_caller_source_candidate"
        limitations = [
            "declared state indices/flags and initial contents, not native metadata or numeric cache execution",
            "any prefill selects three conv_fn calls and chunk for all non-speculative rows, including decode1",
            "decode reads prior3 convolution values and rereads2 shifted values per channel; fresh has no old reads",
            "gather reads one FP32 state row and writes two dense rows, fresh zeroed; scatter writes both pools",
            "nine chunk tiles include one short decode segment; internal masked work/traffic remain unpriced",
            "no pure-decode recurrence substitution, speculation, prefix checkpoint or actual backend/provider binding"]
    elif component == "nope-mixed":
        scenario = {"layer_index": 3, "query_start_loc": (0, 1, 513), "decode_kv_len": 513,
                    "slot_mapping": (512, *range(1024, 1536)), "cache_capacity_tokens": 2048,
                    "cache_dtype": "bf16", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_nope_mixed_workload(model, **scenario)
        scope = "TP1_BF16_NoPE_MLA_one_decode513_one_fresh512_shared_frontend_source_candidate"
        limitations = [
            "declared decode-first slots and length, no native metadata/Scheduler or cached tensor execution",
            "T513 merged frontend/norm/QB and output gate/O-projection run once across both requests",
            "fresh prefill executes before decode absorption despite decode-first token order",
            "KV-B covers prefill512; absorption and W_UV cover decode1 only",
            "two fused epilogues insert disjoint cache slices; attention writes disjoint output slices without copy",
            "canonical analytical MLA selectors are not actual0.30 backend/provider execution",
            "compiled BF16 sigmoid output gate retains an unpriced materialization/compiler boundary"]
    elif component == "nope-prefill":
        scenario = {"layer_index": 3, "slot_mapping": tuple(range(512)), "cache_capacity_tokens": 512,
                    "cache_dtype": "bf16", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_nope_fresh_prefill_workload(model, **scenario)
        scope = "single_fresh_NoPE_MLA_T512_BF16_TP1_source_branch"
        limitations = [
            "merged frontend proven for T>=512; no short-input auxiliary stream dispatch claim",
            "one fresh sequence, no context chunks, MTP, DCP or graph padding",
            "canonical FA3 analytical selector reuse is not vLLM0.30 backend/provider execution",
            "compiled sigmoid gate preserves BF16 tensor boundary; intermediate storage remains unbound"]
    elif component == "native-latent-output":
        scenario = {"layer_index": 1, "rows": 8, "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_tp1_native_latent_output_workload(model, **scenario)
        scope = "TP1_native_MoERunner_output_tail_external_routed_latent_and_shared_output"
        limitations = [
            "external already combined BF16 routed latent and shared expert outputs; not a complete MoE call",
            "TP1 native runner only; no MegaMoE beta-add, TP tail tier, expert dispatch or routing assertion",
            "RMSNorm follows combined routed latent; ordinary up projection then distinct shared add",
            "all source-selected generic algorithms are analytical; actual backend/provider remains unbound"]
    elif component == "marlin-routed-output":
        scenario = {"layer_index": 1, "rows": 8, "pool_name": "main", "rank": 0}
        step_id = "source-k3-marlin-output"
        routing = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0).resolve(
            step_id=step_id, layer_index=1, moe_invocation_id=f"{step_id}:layer-1:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=scenario["rows"], num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
        graph = vllm030_kimi_k3_marlin_routed_output_workload(model, routing=routing, **scenario)
        scope = "TP1_Marlin_W4A16_SiTU_experts_plus_native_latent_output_tail_source_candidate"
        limitations = [
            "explicit seeded routing assumption, not computed from checkpoint logits or observed expert IDs",
            "external BF16 down-projected latent, selected IDs/weights and BF16 shared MLP output",
            "source aggregate includes alignment/W13/SiTU/weighted_W2/topk_sum; not one physical launch",
            "MXFP4 active weight payload/scales and live tensor math do not establish physical padded/scratch traffic",
            "SiTU compiler services, alignment/scratch and actual H200 backend/provider remain unbound"]
    elif component in ("kda-fresh-prefill", "dense-kda-prefill-layer"):
        scenario = {"query_start_loc": (0, 128), "state_index": 1, "state_pool_slots": 2,
                    "has_initial_state": False, "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32",
                    "pool_name": "main", "rank": 0}
        if component == "kda-fresh-prefill":
            scenario["layer_index"] = 0
            graph = vllm030_kimi_k3_triton_fresh_prefill_workload(model, **scenario)
            scope = "TP1_T128_fresh_single_sequence_Triton_KDA_source_candidate"
        else:
            graph = vllm030_kimi_k3_dense_fresh_prefill_layer_workload(model, **scenario)
            scope = "TP1_T128_fresh_KDA_dense_decoder_layer0_full_source_candidate"
        limitations = [
            "explicit Triton fresh full64-chunk candidate; no auto backend or model execution assertion",
            "one positive declared state slot, BF16 conv/FP32 recurrent cache; no prefix/MTP/RecoverSSM",
            "three Q/K/V convolutions, FP32 dense initial zeros, chunk result pool scatter and BF16 core copy retained",
            "chunk_h is BF16, final recurrent state is FP32; output overwrites post-conv V only after its earlier uses",
            "chunk internal buffer storage is not a complete arithmetic/traffic census or an assigned latency",
            "source state/index roles and copy projection do not bind actual library/compiler launches"]
    elif component in ("native-marlin-moe", "nope-prefill-layer"):
        layer_index = 1 if component == "native-marlin-moe" else 3
        scenario = {"layer_index": layer_index, "rows": 512, "pool_name": "main", "rank": 0,
                    "shared_experts_stream_token_threshold": 256}
        if component == "nope-prefill-layer":
            scenario.update(slot_mapping=tuple(range(scenario["rows"])), cache_capacity_tokens=512, cache_dtype="bf16")
        step_id = f"source-k3-{component}"
        routing = ExpertRoutingInput.seeded_assumption("K3-source-component-declared-routing", 0).resolve(
            step_id=step_id, layer_index=layer_index, moe_invocation_id=f"{step_id}:layer-{layer_index}:moe",
            schedule_step_sha256=hashlib.sha256(json.dumps(scenario, sort_keys=True).encode()).hexdigest(),
            tokens=scenario["rows"], num_experts=model.num_experts, topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256, pool_name="main", tensor_parallel_size=1,
            expert_parallel_size=1, expert_rank_group=(0,), expert_placement_strategy="linear")
        if component == "native-marlin-moe":
            graph = vllm030_kimi_k3_native_marlin_moe_workload(model, routing=routing, **scenario)
            scope = "TP1_T512_native_Marlin_MoE_original_hidden_to_output_source_candidate"
        else:
            graph = vllm030_kimi_k3_nope_prefill_layer_workload(
                model, routing=routing, **{key: value for key, value in scenario.items() if key != "rows"})
            scope = "TP1_T512_fresh_NoPE_decoder_layer3_full_source_candidate"
        limitations = [
            "explicit Marlin standard modular/no-DP-EP source program, not auto backend dispatch",
            "T512 exceeds model down-stream threshold256 and declared shared-stream threshold256",
            "original hidden input shared by router/down/shared gate; gate emits FP32 logits from BF16 A/B",
            "canonical seeded histogram remains an explicit assumption, not derived checkpoint logits",
            "shared SiTU, grouped top16 and expert aggregate remain unpriced; no generic activation substitution",
            "source call/aggregate counts are not a census of actual GPU launches",
            "NoPE layer case uses externally supplied AttnRes prefix/delta/block state and fresh BF16 cache slots"]
    else:
        raise ValueError("unknown K3 source component chain")
    if component in ("fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"):
        limitations = [line.replace("excludes sampler", "explicit plain greedy sampler appended below")
                       .replace("sampler, metadata/index producers", "metadata/index producers") for line in limitations]
        scenario["sampling_source"] = "vllm_0_30_plain_greedy_bf16"
        graph = vllm030_kimi_k3_append_plain_greedy_sampler(model, graph, sampling_source=scenario["sampling_source"])
        scope += "__plain_greedy_no_processors_penalties_masks_logprobs_or_thinking_budget"
        limitations.append("sampler profile is declared; no sampled values or native SamplingMetadata execution; argmax compiler/ID services unpriced")
    plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
    calls = []
    for call in plan.calls:
        spec = call.kernel_spec
        record = {"call_id": call.call_id, "dependencies": call.dependencies,
                  "kind": spec.kind.value, "variant": spec.variant, "dtype": spec.dtype,
                  "shape": dict(spec.shape), "attributes": dict(spec.attributes),
                  "execution_support": spec.execution_support, "actual_provider_bound": False,
                  "analytical_algorithm": None if spec.launch is None else spec.launch.algorithm}
        if spec.variant == "kimi_k3_nope_prefill":
            record["source_work"] = kimi_k3_nope_prefill_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_decode":
            record["source_work"] = kimi_k3_nope_decode_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_mixed":
            record["source_work"] = kimi_k3_nope_mixed_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_context":
            record["source_work"] = kimi_k3_nope_context_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_context_attention":
            record["source_work"] = kimi_k3_nope_context_attention_logical_work(spec)
        elif spec.variant == "kimi_k3_nope_bmm":
            record["source_work"] = kimi_k3_nope_bmm_logical_work(spec)
        elif spec.variant == "kimi_k3_marlin_experts":
            record["source_work"] = kimi_k3_marlin_expert_logical_work(spec)
        elif spec.variant == "kimi_k3_moe_route":
            record["source_work"] = kimi_k3_moe_route_logical_work(spec)
        elif spec.variant == "kimi_k3_situ":
            record["source_work"] = kimi_k3_situ_logical_work(spec)
        elif spec.variant == "kimi_k3_attn_res":
            record["source_work"] = kimi_k3_attention_residual_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_prefill":
            record["source_work"] = kimi_k3_kda_prefill_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_context_prefill":
            record["source_work"] = kimi_k3_kda_context_prefill_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_mixed":
            record["source_work"] = kimi_k3_kda_mixed_logical_work(spec)
        elif spec.variant == "kimi_k3_kda_decode":
            record["source_work"] = kimi_k3_kda_decode_logical_work(spec)
        elif spec.variant == "vllm_logits_row_select":
            record["source_work"] = vllm_logits_row_selection_logical_work(spec)
        elif spec.variant == "vllm_embedding_id_widen":
            record["source_work"] = vllm_embedding_id_widen_logical_work(spec)
        elif spec.variant == "vllm_plain_greedy_sample":
            record["source_work"] = vllm_plain_greedy_sampling_logical_work(spec)
        elif spec.variant == "vllm_logits_fp32_cast":
            record["source_work"] = vllm_logits_fp32_cast_logical_work(spec)
        if (spec.variant in ("kimi_k3_marlin_experts", "kimi_k3_moe_route", "kimi_k3_situ", "kimi_k3_nope_context_attention")
                or (spec.variant == "kimi_k3_nope_context" and spec.attributes["stage"] in ("merge", "context_partial_merge", "output_gate"))
                or (spec.variant in ("kimi_k3_kda_prefill", "kimi_k3_kda_mixed", "kimi_k3_kda_context_prefill") and spec.attributes["stage"] == "chunk")
                or (spec.variant in ("kimi_k3_nope_prefill", "kimi_k3_nope_decode", "kimi_k3_nope_mixed")
                    and spec.attributes["stage"] == "output_gate")
                or (spec.variant == "vllm_plain_greedy_sample" and spec.attributes["stage"] == "argmax")):
            try:
                predict(spec, hardware, method="roofline")
            except NotImplementedError as error:
                record.update(status="unpriced", roofline_seconds=None, reason=str(error))
            else:
                raise RuntimeError("K3 unbound SiTU/gate source cannot silently inherit a generic activation price")
        else:
            record.update(status="priced", roofline_seconds=predict(spec, hardware, method="roofline").t_total_s)
        calls.append(record)
    result = {
        "snapshot": _snapshot(), "source_identity": identity,
        "model_id": model.model_id, "model_revision": model.model_revision,
        "config_sha256": model.config_sha256,
        "scope": scope,
        "source_nodes": tuple({"node_id": node.node_id, "kind": node.kind.value,
                               "variant": node.variant, "dtype": node.dtype,
                               "shape": dict(node.shape), "attributes": dict(node.attributes)}
                              for node in graph.nodes),
        "source_edges": tuple({"source": edge.source, "destination": edge.destination, "kind": edge.kind}
                              for edge in graph.edges),
        "scenario": scenario, "calls": calls, "component_count": len(calls),
        "priced_calls": sum(call["status"] == "priced" for call in calls),
        "complete_moe_body_source_candidate": component in ("native-marlin-moe", "nope-prefill-layer"),
        "complete_decoder_layer_source_candidate": component in ("nope-prefill-layer", "dense-kda-prefill-layer"),
        "complete_decoder_stack_source_candidate": component in (
            "fresh-prefill-decoder-stack", "fresh-prefill-forward-logits", "single-decode-forward-logits",
            "fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"),
        "complete_forward_logits_source_candidate": component in (
            "fresh-prefill-forward-logits", "single-decode-forward-logits", "fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"),
        "complete_plain_greedy_model_source_candidate": component in ("fresh-prefill-greedy-step", "single-decode-greedy-step", "mixed-greedy-step", "context-prefill-greedy-step", "context-continuation-greedy-step"),
        "unpriced_calls": sum(call["status"] == "unpriced" for call in calls),
        "complete_model_workload": False, "runtime_accepted": False,
        "normal_capacity_accepted": False, "independent_reference_accepted": False,
        "limitations": limitations + [
            "full K3 prefill/decode/mixed hybrid decoder and normal Serving bridge remain incomplete"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), indent=2) + "\n")
    return result


def write_kimi_k3_source_components(model_path, source_root, output):
    """Evaluate independent source components, never a complete K3 graph."""
    from src.serving.backend.vllm_scheduler_process import vllm030_source_identity
    from src.serving.workload.kimi_k3 import (
        vllm030_kimi_k3_attention_residual_node, vllm030_kimi_k3_ordinary_mlp_workload,
        vllm030_kimi_k3_triton_decode_workload, vllm030_kimi_k3_dense_decode_layer_workload)
    from src.serving.workload.shape_inference import infer_gemm_dimensions
    from src.serving.workload.workload_graph import WorkloadGraph
    from src.kernel.predictors.vector import (
        kimi_k3_attention_residual_logical_work, kimi_k3_situ_logical_work, kimi_k3_kda_decode_logical_work)

    model = load_model_config(model_path)
    if model.attention_layout != "hybrid_kda_mla":
        raise ValueError("K3 source components require an exact hybrid config")
    identity = vllm030_source_identity(Path(source_root))
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (WorkerPoolConfig("main", "colocated", "nvidia_h200"),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts)
    positions = (
        "in_proj_qkvgfab", "f_b_proj", "fused_qkv_a_g_proj", "q_b_proj", "kv_b_proj", "o_proj",
        "gate_up_proj", "down_proj", "shared_experts.gate_up_proj", "shared_experts.down_proj",
        "experts.w13_weight", "experts.w2_weight", "routed_expert_down_proj", "router", "lm_head")
    dimensions = []
    for tp in (1, 8):
        for position in positions:
            dimensions.append({"tp": tp, "position": position, "rows": 8,
                               "m_n_k": infer_gemm_dimensions(model, position, 8, tp_size=tp)})
        for tier in ((None,) if tp == 1 else ("ALLREDUCE_OVERLAP", "COLUMN_PARALLEL")):
            dimensions.append({"tp": tp, "position": "routed_expert_up_proj", "rows": 8,
                               "explicit_source_tail_tier": tier,
                               "m_n_k": infer_gemm_dimensions(model, "routed_expert_up_proj", 8,
                                                               tp_size=tp, kimi_k3_tail_tier=tier)})
    stages = [(stage, layer) for layer in range(model.num_hidden_layers)
              for stage in ("pre_attention", "post_attention")] + [("output", None)]
    components = []
    # These are independent operator evaluations with externally supplied
    # tensors. No edges or combined model latency are invented between them.
    for rows in (1, 8, 256):
        for stage, layer in stages:
            node = vllm030_kimi_k3_attention_residual_node(
                model, rows=rows, stage=stage, layer_index=layer, pool_name="main", rank=0)
            plan = lower_workload(WorkloadGraph([node]), deployment, {"nvidia_h200": hardware})
            if len(plan.calls) != 1:
                raise RuntimeError("K3 residual component must retain one source call")
            spec = plan.calls[0].kernel_spec
            work = kimi_k3_attention_residual_logical_work(spec)
            prediction = predict(spec, hardware, method="roofline")
            components.append({"stage": stage, "layer_index": layer, "rows": rows,
                               "shape": dict(spec.shape), "attributes": dict(spec.attributes),
                               "source_work": work, "roofline_seconds": prediction.t_total_s,
                               "execution_support": spec.execution_support, "launch": None})
    mlp_cases = []
    for rows in (1, 8, 256):
        for role, layer in (("dense", 0), ("shared", model.num_dense_layers)):
            graph = vllm030_kimi_k3_ordinary_mlp_workload(
                model, rows=rows, layer_index=layer, role=role, pool_name="main", rank=0)
            plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
            calls = []
            for call in plan.calls:
                spec = call.kernel_spec
                record = {"call_id": call.call_id, "dependencies": call.dependencies,
                          "kind": spec.kind.value, "variant": spec.variant,
                          "dtype": spec.dtype, "shape": dict(spec.shape),
                          "execution_support": spec.execution_support,
                          "launch_policy": "analytical_launch", "actual_provider_bound": False}
                if spec.variant == "kimi_k3_situ":
                    work = kimi_k3_situ_logical_work(spec)
                    try:
                        predict(spec, hardware, method="roofline")
                    except NotImplementedError as error:
                        record.update(status="unpriced", roofline_seconds=None, reason=str(error), source_work=work)
                    else:
                        raise RuntimeError("K3 source tanhf cannot be silently priced")
                else:
                    prediction = predict(spec, hardware, method="roofline")
                    record.update(status="priced", roofline_seconds=prediction.t_total_s)
                calls.append(record)
            mlp_cases.append({"role": role, "layer_index": layer, "rows": rows,
                              "source_branch": "ordinary_TP1_KimiMLP_not_provider_dispatch",
                              "calls": calls, "unpriced_calls": sum(call["status"] == "unpriced" for call in calls)})
    def source_calls(graph):
        plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
        records = []
        for call in plan.calls:
            spec = call.kernel_spec
            record = {"call_id": call.call_id, "dependencies": call.dependencies,
                      "kind": spec.kind.value, "variant": spec.variant, "dtype": spec.dtype,
                      "shape": dict(spec.shape), "attributes": dict(spec.attributes),
                      "execution_support": spec.execution_support, "actual_provider_bound": False}
            if spec.variant == "kimi_k3_kda_decode":
                record["source_work"] = kimi_k3_kda_decode_logical_work(spec)
            elif spec.variant == "kimi_k3_attn_res":
                record["source_work"] = kimi_k3_attention_residual_logical_work(spec)
            elif spec.variant == "kimi_k3_situ":
                record["source_work"] = kimi_k3_situ_logical_work(spec)
            if spec.variant == "kimi_k3_situ":
                try:
                    predict(spec, hardware, method="roofline")
                except NotImplementedError as error:
                    record.update(status="unpriced", roofline_seconds=None, reason=str(error))
                else:
                    raise RuntimeError("K3 SiTU cannot inherit KDA source pricing")
            else:
                prediction = predict(spec, hardware, method="roofline")
                record.update(status="priced", roofline_seconds=prediction.t_total_s)
            records.append(record)
        return records

    kda_cases, first_layer_cases = [], []
    for rows in (1, 8):
        scenario = {"state_indices": tuple(range(1, rows + 1)), "state_pool_slots": rows + 1,
                    "conv_state_dtype": "bf16", "recurrent_state_dtype": "fp32", "pool_name": "main", "rank": 0}
        graph = vllm030_kimi_k3_triton_decode_workload(model, layer_index=0, **scenario)
        calls = source_calls(graph)
        kda_cases.append({"rows": rows, "layer_index": 0, "scenario": scenario, "calls": calls,
                          "phase": "decode", "source_branch": "eager_triton_TP1_not_observed_dispatch"})
        graph = vllm030_kimi_k3_dense_decode_layer_workload(model, **scenario)
        calls = source_calls(graph)
        for phase in ("decode1", "decode2"):
            first_layer_cases.append({"rows": rows, "layer_index": 0, "phase": phase,
                                      "scenario": scenario, "calls": calls,
                                      "unpriced_calls": sum(call["status"] == "unpriced" for call in calls),
                                      "state_policy": "same declared positive slots across phases, no actual cache values or Scheduler trace",
                                      "full_model": False, "runtime_accepted": False})
    result = {
        "snapshot": _snapshot(), "source_identity": identity,
        "model_id": model.model_id, "model_revision": model.model_revision,
        "config_sha256": model.config_sha256,
        "scope": "K3 source components plus ordinary TP1 Triton KDA decode and first dense decode-layer chains",
        "projection_dimensions": dimensions, "attention_residual_components": components,
        "ordinary_mlp_cases": mlp_cases,
        "ordinary_kda_decode_cases": kda_cases, "first_dense_decode_layer_cases": first_layer_cases,
        "component_count": len(components), "complete_model_workload": False,
        "runtime_accepted": False, "normal_capacity_accepted": False,
        "independent_reference_accepted": False,
        "limitations": [
            "PP1/SP1 BF16 residual components use externally supplied source tensors",
            "no KDA prefill/chunk/mixed or NoPE gated MLA decoder DAG, latent SiTU MoE or normal Serving bridge",
            "projection dimensions precede backend expert packing and do not price latent beta=1 C reads",
            "declared latent tail tier is not inferred from tokens or bound to runtime dispatch",
            "source launch metadata and logical bytes are not observed compiler/provider/DRAM evidence"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), indent=2) + "\n")
    return result


def _weight_residency_case(args):
    (model_path, tp, pp, vision_tensor_parallel_mode, model_execution_version,
     mtp_mode, constructor_buffer_limits, source_fp8_backend, source_moe_workspace_tokens) = args
    model = load_model_config(model_path)
    if constructor_buffer_limits is not None and model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S:
        raise ValueError("constructor buffer source limits currently require exact GLM block-FP8")
    if source_fp8_backend is not None and model.config_sha256 not in GLM_BLOCK_FP8_CONFIG_SHA256S:
        raise ValueError("post-load FP8 source backend currently requires exact GLM-5.3")
    if model.model_type != "glm_moe_dsa" and (
            model_execution_version is not None or mtp_mode is not None):
        raise ValueError("versioned MTP residency diagnostic currently requires GLM")
    capacity = HWSpec("configs/hardware/nvidia_h200.yaml").resource_by_role("dram").size_bytes
    if model.model_type == "kimi_k2":
        if vision_tensor_parallel_mode is None:
            raise ValueError("Kimi outer diagnostic requires explicit vision tensor parallel mode")
        parameters = kimi_k25_outer_loader_payload_bytes(
            model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
            vision_tensor_parallel_mode=vision_tensor_parallel_mode)
        return {
            "payload_scope": "kimi_active_outer_registered_parameters",
            "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
            "vision_tensor_parallel_mode": vision_tensor_parallel_mode,
            "outer_loader_parameter_bytes_per_rank": parameters,
            "outer_constructor_parameter_payload_bytes_per_rank": sum(parameters.values()),
            "normal_kv_capacity_accepted": False,
            "actual_outer_loader_or_runtime_binding": False,
            "scope": "explicit enabled-tower/no-offload BF16 parameter constructor; no text/KV/reserve accounting",
            "limitations": [
                "raw vt_* config metadata does not override VisionConfig constructor defaults",
                "parent config replaces projector mm_hidden_size equal to vision width with text width",
                "CompressedTensorsConfig is ignored by the outer tower/projector",
                "source vision mode is conditional; actual language_model_only/vision_chunk skip, TP mode and offload not bound",
                "nonpersistent time/rotary buffers, text weights, post-load aliases, scratch/reserves and allocator excluded",
            ],
        }
    if vision_tensor_parallel_mode is not None:
        raise ValueError("explicit vision tensor parallel mode applies only to Kimi outer diagnostic")
    if model.model_type == "deepseek_v4":
        payload = dsv4_routed_loader_payload_bytes(
            model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp))
        packed = payload["layers"] * (
            payload["w13_packed_bytes_per_layer"] + payload["w2_packed_bytes_per_layer"])
        scales = payload["layers"] * (
            payload["w13_ue8m0_bytes_per_layer"] + payload["w2_ue8m0_bytes_per_layer"])
        other_parameters = dsv4_non_routed_loader_payload_bytes(
            model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
            attention_backend="flashmla")
        return {
            "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
            "routed_loader_payload_per_rank": payload,
            "routed_packed_bytes_per_rank": packed,
            "routed_loader_scale_bytes_per_rank": scales,
            "non_routed_loader_parameter_bytes_per_rank": other_parameters,
            "non_routed_loader_bytes_per_rank": sum(other_parameters.values()),
            "constructor_parameter_payload_bytes_per_rank": packed + scales + sum(other_parameters.values()),
            "attention_backend_source_scenario": "flashmla",
            "actual_attention_backend_bound": False,
            "packed_payload_exceeds_physical_capacity": packed > capacity,
            "normal_kv_capacity_accepted": False,
            "scope": "conditional V4 EP-off aligned Marlin/FlashMLA source constructor parameter payload only; not post-load runtime residency",
            "limitations": [
                "non-MegaMoE FusedMoEFactory with Marlin is a source scenario, not observed provider selection",
                "registered source parameters are counted; actual checkpoint/loader/storage are not bound",
                "post-load conversion/aliases, rotary/topk/broadcast/scratch/workspaces are excluded",
                "target-off only; DSpark draft identity/weights are not included",
                "reserve/profile/allocator/Scheduler and actual available-KV remain unbound",
            ],
        }
    if model.model_type != "glm_moe_dsa":
        raise NotImplementedError("this Goal weight diagnostic requires exact GLM or V4 config")
    if model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S:
        if model_execution_version != "vllm_0_30_0" or mtp_mode != "off":
            raise ValueError("GLM block-FP8 payload requires explicit vLLM 0.30 and target-off")
        parallel = ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp)
        registered = glm_block_fp8_loader_payload_bytes(
            model, parallel, include_mla_absorption=False)
        absorbed = glm_block_fp8_loader_payload_bytes(
            model, parallel, include_mla_absorption=True)
        additions = {name: size for name, size in absorbed.items() if name not in registered}
        buffers = (None if constructor_buffer_limits is None else
                   glm_block_fp8_constructor_buffer_bytes(
                       model, parallel, max_batched_tokens=constructor_buffer_limits[0],
                       max_requests=constructor_buffer_limits[1]))
        postload = (None if source_fp8_backend is None else
                    glm_block_fp8_postload_parameter_bytes(
                        model, parallel, hardware_id="nvidia_h200",
                        linear_backend="deep_gemm", routed_backend="deep_gemm",
                        include_mla_absorption=True))
        workspace = (None if source_moe_workspace_tokens is None else
                     glm_block_fp8_deepgemm_workspace_bytes(
                         model, parallel, hardware_id="nvidia_h200",
                         tokens=source_moe_workspace_tokens))
        call_allocations = (None if source_moe_workspace_tokens is None else
                            glm_block_fp8_deepgemm_call_allocation_bytes(
                                model, parallel, hardware_id="nvidia_h200",
                                tokens=source_moe_workspace_tokens))
        return {
            "payload_scope": "glm_fp8_source_parameters",
            "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
            "registered_parameter_bytes_per_rank": registered,
            "registered_parameter_payload_bytes_per_rank": sum(registered.values()),
            "mla_absorption_parameter_bytes_per_rank": additions,
            "source_payload_with_mla_absorption_bytes_per_rank": sum(absorbed.values()),
            "constructor_buffer_source_limits": constructor_buffer_limits,
            "constructor_buffer_bytes_per_rank": buffers,
            "constructor_buffer_total_bytes_per_rank": None if buffers is None else sum(buffers.values()),
            "source_fp8_postload_backend": source_fp8_backend,
            "postload_parameter_bytes_per_rank": postload,
            "postload_parameter_payload_bytes_per_rank": None if postload is None else sum(postload.values()),
            "postload_source_binding": None if postload is None else GLM_SM90_DEEP_GEMM_STORAGE_SOURCE,
            "moe_workspace_source_request_bytes_per_rank": workspace,
            "moe_workspace_source_tokens": source_moe_workspace_tokens,
            "moe_call_allocation_bytes_per_rank": call_allocations,
            "moe_call_allocation_source": (None if call_allocations is None else {
                "source_binding_owner": "src.serving.backend.vllm_scheduler_process.vllm030_allocator_source_identity",
                "symbols": (
                    "fused_moe/modular_kernel.py:FusedMoEKernel.apply/_fused_experts",
                    "fused_moe/prepare_finalize/no_dp_ep.py:_quantize_input",
                    "fused_moe/utils.py:_fp8_quantize/count_expert_num_tokens",
                    "fused_moe/deep_gemm_utils.py:deepgemm_moe_permute",
                    "quantization/utils/fp8_utils.py:per_token_group_quant_fp8/silu_mul_per_token_group_quant_fp8_colmajor"),
                "scope": "declared unchanged vLLM0.30 source scenario, not loaded CUDA objects or provider selection",
            }),
            "moe_call_allocation_scope": (None if call_allocations is None else
                "unique tensor payloads outside the shared workspace for one declared call; "
                "sum is an allocation envelope, not a simultaneous live peak or total worker reserve"),
            "moe_workspace_source_request_scope": (None if workspace is None else
                "declared unchunked M=source-moe-workspace-tokens, BF16, no CPU expert counts, EP-off, default SM90 alignment, one lane/ubatch; shared by layers; not total worker workspace or activation peak"),
            "registered_payload_exceeds_physical_capacity": sum(registered.values()) > capacity,
            "model_execution_version": model_execution_version, "mtp_mode": "off",
            "normal_kv_capacity_accepted": False, "actual_loader_or_runtime_binding": False,
            "scope": "PP1 unreplicated embedding, unpadded FP8 constructors and ordinary CUDA/CP1 BF16 MLA absorption; source payload only",
            "limitations": [
                ("Backend FP8 weight/scale relayout, padding and persistent conversion objects remain unbound"
                 if postload is None else
                 "SM90 DeepGEMM registered parameter storage counted under explicit source profile; actual provider, other backend layouts and kernel persistent objects remain unbound"),
                ("Rotary/topk/index-group and other persistent buffers are excluded" if buffers is None else
                 "Explicit eager BF16 default-RoPE/topk/index-group source buffers counted; other backend/runtime persistent storage remains unbound"),
                "MTP, other activation/dispatch/library scratch, reserves, actual loading and allocator-ready residency are excluded",
                "Declared MLA absorption branch is not an observed provider or execution",
                "Normal capacity entry remains rejected; this is not available-KV evidence",
            ],
        }
    ranks = model_rank_residencies(
        model, ParallelConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        worker_role="colocated", include_mtp=mtp_mode != "off",
        model_execution_version=model_execution_version)
    return {
        "payload_scope": "glm_parameter_source_projection",
        "tp": tp, "pp": pp, "physical_hbm_bytes": capacity,
        "maximum_owner_weight_bytes": max(rank.weight_resident_bytes for rank in ranks),
        "rank_residencies": [asdict(rank) for rank in ranks],
        "model_execution_version": model_execution_version,
        "mtp_mode": "native" if mtp_mode is None else mtp_mode,
        "normal_kv_capacity_accepted": False,
        "actual_loader_or_runtime_binding": False,
        "scope": ("GLM vLLM 0.30 BF16 parameter source projection with MTP vocabulary aliases; unreplicated embedding source scenario, not normal runtime capacity"
                  if model_execution_version == "vllm_0_30_0" else
                  "existing independent draft-vocabulary inventory; v0.30 aliases and runtime reserves not bound"),
    }


def write_vllm030_model_identity_probe(model_path, source_root, output, *, draft_path=None):
    """Bind official target/draft metadata through genuine guarded CPU APIs.

    ModelConfig inspects the native registry without constructing a model or
    loading weights. This is identity/mode evidence, not a provider, Scheduler,
    capacity, full source graph or runtime acceptance result.
    """
    import tempfile
    from src.serving.backend.vllm_scheduler_process import (
        _install_cpu_device_guard, _verify_no_accelerator_access,
        vllm030_allocator_source_identity,
    )
    from src.serving.workload.model_config import _load_model_json

    model = load_model_config(model_path)
    if any(not isinstance(getattr(model, name), str) or not getattr(model, name) for name in (
            "model_id", "model_revision", "config_sha256", "model_source_url")):
        raise ValueError("model identity probe requires registered exact checkpoint provenance")
    identity = vllm030_allocator_source_identity(Path(source_root))
    guard = _install_cpu_device_guard()
    from vllm.config import ModelConfig as NativeModelConfig, ParallelConfig as NativeParallelConfig, SpeculativeConfig
    from vllm.model_executor.models import ModelRegistry

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def registry_binding(native):
        if len(native.architectures) != 1:
            raise ValueError("identity probe requires one explicit native architecture")
        architecture = native.architectures[0]
        if architecture not in ModelRegistry.models:
            raise ValueError("identity probe rejects a Transformers fallback or unregistered architecture")
        entry = ModelRegistry.models[architecture]
        return {"architecture": architecture, "module": entry.module_name, "class": entry.class_name,
                "inspected_architecture": native._model_info.architecture}

    with tempfile.TemporaryDirectory(prefix="model-identity-", dir=output.parent) as directory:
        target_dir = Path(directory) / "target"
        target_dir.mkdir()
        target_dir.joinpath("config.json").write_bytes(Path(model_path).read_bytes())
        target = NativeModelConfig(model=str(target_dir), skip_tokenizer_init=True, enforce_eager=True,
                                   dtype="bfloat16", max_model_len=128, runner="generate")
        target_binding = registry_binding(target)
        layers = target.hf_text_config.num_nextn_predict_layers
        if layers != model.num_mtp_layers:
            raise ValueError("actual native MTP layer count disagrees with the exact target config")
        native_on = None
        if layers == 0:
            if draft_path is not None:
                raise ValueError("zero-MTP target identity probe does not accept an invented draft")
        else:
            if model.model_type == "glm_moe_dsa":
                if draft_path is not None:
                    raise ValueError("GLM native MTP uses the target checkpoint configuration")
                method, draft_dir, tokens = "mtp", target_dir, 1
                draft_identity = {name: getattr(model, name) for name in (
                    "model_id", "model_revision", "config_sha256", "model_source_url")}
            elif model.model_type == "deepseek_v4":
                if draft_path is None:
                    raise ValueError("V4 native mode identity requires the exact official DSpark config")
                raw, sha, source_sha, canonical_sha, draft_id, revision, url, draft_scope = _load_model_json(str(draft_path))
                if (draft_id != model.model_id + "-DSpark" or draft_scope != "text"
                        or raw["model_type"] != "deepseek_v4"
                        or raw["architectures"] != ["DeepseekV4ForCausalLM"]):
                    raise ValueError("V4 DSpark draft identity does not match the exact target")
                method, tokens = "dspark", raw["dspark_block_size"]
                draft_dir = Path(directory) / "draft"
                draft_dir.mkdir()
                draft_dir.joinpath("config.json").write_bytes(Path(draft_path).read_bytes())
                draft_identity = dict(model_id=draft_id, model_revision=revision, config_sha256=sha,
                                      source_config_sha256=source_sha, canonical_config_sha256=canonical_sha,
                                      model_source_url=url)
            else:
                raise NotImplementedError("native mode identity is not bound for this target family")
            speculative = SpeculativeConfig(
                model=str(draft_dir), method=method, num_speculative_tokens=tokens,
                target_model_config=target,
                target_parallel_config=NativeParallelConfig(tensor_parallel_size=1, pipeline_parallel_size=1))
            draft = speculative.draft_model_config
            if draft is None:
                raise ValueError("native speculative config did not construct draft metadata")
            native_on = dict(method=speculative.method, registry=registry_binding(draft),
                             draft_identity=draft_identity, n_predict=draft.hf_config.n_predict,
                             num_speculative_tokens=speculative.num_speculative_tokens,
                             max_num_new_slots_for_drafting=speculative.max_num_new_slots_for_drafting,
                             parallel_drafting=speculative.parallel_drafting,
                             use_dspark=speculative.use_dspark())
        result = dict(snapshot=_snapshot(), source_identity=identity,
                      model_identity={name: getattr(model, name) for name in (
                          "model_id", "model_revision", "config_sha256", "model_source_url")},
                      target_registry=target_binding, configured_mtp_layers=layers,
                      applicable_modes=("off",) if layers == 0 else ("off", "native"),
                      native_mode=native_on, platform="guarded CPU metadata APIs; not H200 provider",
                      model_instantiated=False, weights_loaded=False, scheduler_executed=False,
                      normal_capacity_accepted=False, runtime_acceptance=False)
    _verify_no_accelerator_access(guard)
    output.write_text(json.dumps(result, indent=2) + "\n")


def write_model_weight_residency_diagnostic(
        model_path, layouts, output, *, vision_tensor_parallel_mode=None,
        model_execution_version=None, mtp_mode=None, constructor_buffer_limits=None,
        source_fp8_backend=None, source_moe_workspace_tokens=None):
    """Estimate which layouts can proceed to normal-capacity validation.

    Consume GLM residency, V4 parameters or explicit active Kimi outer parameters.
    No second formula, reserve assumption, debug block count or success gate.
    """
    if not layouts:
        raise ValueError("residency diagnostic requires at least one layout")
    if mtp_mode not in (None, "off", "native"):
        raise ValueError("unknown residency MTP mode")
    if source_fp8_backend not in (None, "deep_gemm_sm90"):
        raise ValueError("unsupported post-load FP8 source backend")
    if source_moe_workspace_tokens is not None:
        if type(source_moe_workspace_tokens) is not int or source_moe_workspace_tokens <= 0:
            raise ValueError("MoE workspace source tokens must be a positive integer")
        if source_fp8_backend != "deep_gemm_sm90":
            raise ValueError("MoE workspace source tokens require the explicit DeepGEMM SM90 profile")
    if constructor_buffer_limits is not None and (
            not isinstance(constructor_buffer_limits, tuple) or len(constructor_buffer_limits) != 2
            or any(type(value) is not int or value <= 0 for value in constructor_buffer_limits)):
        raise ValueError("constructor buffer limits require explicit positive batch-token and request limits")
    snapshot = _snapshot()
    with ProcessPoolExecutor(max_workers=len(layouts)) as executor:
        cases = list(executor.map(
            _weight_residency_case, [(model_path, tp, pp, vision_tensor_parallel_mode,
                                     model_execution_version, mtp_mode, constructor_buffer_limits,
                                     source_fp8_backend, source_moe_workspace_tokens)
                                    for tp, pp in layouts]))
    model_identity = load_model_config(model_path)
    payload = {"snapshot": snapshot, "model_config_path": model_path,
               "model_identity": {name: getattr(model_identity, name) for name in (
                   "model_id", "model_revision", "config_sha256", "model_source_url")},
               "status": "diagnostic_not_normal_capacity_acceptance", "cases": cases}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fp8_runner_source_links(steps):
    """Wire production source state ports; never infer a numeric runner state."""
    links, previous = [], {}
    for step in steps:
        cohorts = {}
        for node in step["nodes"]:
            attributes = node["attributes"]
            if "source_linear_io" not in attributes:
                continue
            history = attributes["source_linear_io"]["conditional_sdk_history"]
            rank = node["rank"]
            if rank not in cohorts:
                cohorts[rank] = []
            cohorts[rank].append(history)
        for rank in sorted(cohorts):
            history = cohorts[rank][0]
            if any(item != history for item in cohorts[rank]) or len(cohorts[rank]) != history["cohort_calls"]:
                raise ValueError("FP8 source state ports must describe the complete rank/model-call cohort")
            if history["invocation"] != step["phase"] or history["rank"] != rank:
                raise ValueError("FP8 source runner state cannot cross invocation or rank")
            links.append({"invocation": step["phase"], "rank": rank,
                "input_port": history["input_port"], "output_port": history["output_port"],
                "previous_output_port": previous[rank] if rank in previous else None,
                "initial_parameter": (None if rank in previous else
                    {"name": f"rank-{rank}:initial-sdk-H", "domain": history["entry_domain"], "value": None})})
            previous[rank] = history["output_port"]
    return links


def _compressor_state_source_links(steps):
    """Track request lineage, keeping physical admission/CoW events unknown."""
    writers = [node for step in steps for node in step["nodes"] if node["variant"] == "dsv4_compressor_state_cache"]
    if not writers:
        return []
    declared = ["source_cache_transition" in node["attributes"] for node in writers]
    if not all(declared):
        if any(declared):
            raise ValueError("compressor source lineage cannot omit some writer transitions")
        return None
    links, history, cohort = [], {}, None
    for step in steps:
        identities = [(node["rank"], node["attributes"]["source_dataflow"]["state_owner"]["layer"],
                       node["attributes"]["source_dataflow"]["state_owner"]["component"])
                      for node in step["nodes"] if node["variant"] == "dsv4_compressor_state_cache"]
        if len(set(identities)) != len(identities) or not identities:
            raise ValueError("compressor source lineage must cover each model cache exactly once")
        if cohort is None:
            cohort = set(identities)
        elif set(identities) != cohort:
            raise ValueError("compressor source lineage cannot drop or replace a model cache between steps")
        for node in sorted((node for node in step["nodes"] if node["variant"] == "dsv4_compressor_state_cache"), key=lambda node: node["node_id"]):
            transition = node["attributes"]["source_cache_transition"]
            cache = transition["cache"]
            if (transition["program"] != "vllm_0_30_compressor_partial_state_lineage" or transition["state_dtype"] != "fp32"
                    or type(cache["rank"]) is not int or cache["rank"] != node["rank"]
                    or type(cache["layer"]) is not int or cache["layer"] < 0
                    or node["dtype"] != "fp32" or transition["compress_ratio"] not in (4, 128)
                    or transition["compress_ratio"] != node["attributes"]["source_dataflow"]["compress_ratio"]
                    or cache != {"rank": node["rank"], **node["attributes"]["source_dataflow"]["state_owner"]}
                    or transition["input_port"] != node["node_id"] + ":state-entry"
                    or transition["output_port"] != node["node_id"] + ":state-exit"
                    or not node["node_id"].startswith(step["phase"] + ":")
                    or transition["physical_table_transition"] is not None or transition["prefix_cache_hit_observation"] is not None
                    or transition["actual_copy_completion_and_retained_content_bound"]):
                raise ValueError("compressor source lineage requires typed source ports, not unobserved physical/copy claims")
            updates = transition["request_updates"]
            ids = [update["request_id"] for update in updates]
            if len(set(ids)) != len(ids) or not ids or any(not isinstance(identity, str) or not identity for identity in ids):
                raise ValueError("compressor source lineage requires unique nonempty model requests")
            cursor = 0
            for update in updates:
                old, new = update["input_prefix"], update["output_prefix"]
                if (type(old) is not int or type(new) is not int or not 0 <= old < new
                        or update["write_positions"] != [old, new] or update["model_token_rows"] != [cursor, cursor + new - old]):
                    raise ValueError("compressor source lineage request spans do not cover its new model rows")
                cursor += new - old
            if cursor != node["shape"]["seq_len"]:
                raise ValueError("compressor source lineage must cover every writer row")
            identity = cache["rank"], cache["layer"], cache["component"]
            if identity not in history:
                history[identity] = {"exit": None, "requests": {}}
            prior = history[identity]
            requests = []
            for update in updates:
                request_id = update["request_id"]
                if request_id in prior["requests"]:
                    prefix = prior["requests"][request_id]
                    if prefix != update["input_prefix"]:
                        raise ValueError("compressor source lineage cannot skip or replay request prefixes")
                    initial = None
                else:
                    prefix = None
                    initial = {"prefix": [0, update["input_prefix"]], "values": None,
                        "prefix_cache_hit_observation": None, "admission_copy_transition": None}
                requests.append({"request_id": request_id, "previous_output_prefix": prefix,
                    "input_prefix": update["input_prefix"], "output_prefix": update["output_prefix"], "initial_context": initial})
                prior["requests"][request_id] = update["output_prefix"]
            links.append({"invocation": step["phase"], "source_node_id": node["node_id"], "cache": cache,
                "input_port": transition["input_port"], "output_port": transition["output_port"],
                "previous_output_port": prior["exit"], "requests": requests,
                "between_step_physical_transition": None, "retained_content_or_copy_completion_proven": False})
            prior["exit"] = transition["output_port"]
    return links


def normalize_model_operator_probe_source(model_path, target_sampling_source):
    """Select an existing logical scenario from an explicit versioned sampler.

    Model identity alone does not select a backend. Constructor history,
    phase IDs, topology and typed native inputs remain separately mandatory
    at the existing source-input and shared-forward boundaries. Sampler-less
    historical dense probes keep their LM-head-only default unchanged.
    """
    model = load_model_config(model_path)
    source = None
    if (model.model_type in ("qwen3", "llama") and not model.num_experts
            and model.compute_dtype == model.weight_dtype
            and target_sampling_source in (("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")
                if model.compute_dtype == "bf16" else ("vllm_0_30_plain_greedy_fp16",))
            and (model.compute_dtype == "bf16" or (model.compute_dtype == "fp16" and model.model_type == "qwen3"))):
        source = "cuda_eager_inplace"
    elif (model.model_type == "glm_moe_dsa" and model.compute_dtype == "bf16"
            and target_sampling_source in ("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16",
                "vllm_0_30_rejection_raw_logits_bf16")):
        if model.weight_dtype == "bf16":
            source = "vllm_0_30_v32_bf16_sync_explicit_allreduce"
        elif model.weight_dtype == "fp8":
            source = "vllm_0_30_v32_fp8_sync_explicit_allreduce"
    elif (model.model_type == "kimi_k2" and model.compute_dtype == "bf16"
            and model.weight_dtype == "mixed_int4_bf16"
            and target_sampling_source in ("vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")):
        source = "vllm_0_30_k2_6_int4_sync_explicit_allreduce"
    elif (model.model_type == "deepseek_v3" and model.compute_dtype == "bf16" and model.weight_dtype == "fp8"
            and target_sampling_source == "vllm_0_27_1_plain_greedy_bf16"):
        source = "vllm_0_27_1_deepseek_v3_sync_fp8"
    if source is None:
        return None, None
    return source, {
        "entrypoint": "static_probe_cli_versioned_sampler",
        "source_scenario": source,
        "target_sampling_source": target_sampling_source,
        "source_projection_only": True,
        "actual_provider_runtime_bound": False,
        "token_values_inferred": False,
    }


def build_model_operator_probe_source_inputs(*, symbolic_forward_source, embedding_token_ids_by_phase,
        source_max_model_len, source_max_batched_tokens, target_sampling_source,
        workspace_initial_capacity_bytes=None, glm_indexer_selection_source=None, glm_prefill_attention_source=None,
        source_max_requests=None, mtp_draft_sampling_source=None, model_source_inputs_by_phase=None):
    """Normalize caller-declared target/native source inputs for the shared probe.

    No token values, constructor limits, workspace history, source backend or
    cardinality declaration is inferred from a model/config or a duration.
    The existing RuntimeInputs boundary owns their complete validation.
    """
    glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    dsv3_source = symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
    if symbolic_forward_source == "cuda_eager_inplace":
        if (target_sampling_source not in (None, "vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16")
                or not isinstance(embedding_token_ids_by_phase, Mapping) or not embedding_token_ids_by_phase):
            raise ValueError("dense source probe requires explicit phase input ports and LM-head-only,0.30 FP32 cast or plain greedy source")
        if any(value is not None for value in (source_max_model_len, source_max_batched_tokens,
                workspace_initial_capacity_bytes, glm_indexer_selection_source, glm_prefill_attention_source,
                source_max_requests, mtp_draft_sampling_source, model_source_inputs_by_phase)):
            raise ValueError("dense source probe does not consume GLM/Kimi constructor or native draft inputs")
        return {phase: RuntimeInputs(ExpertRoutingInput.not_applicable(), "paged_varlen", "uniform",
            model_execution_version="vllm_0_30_0", symbolic_forward_source=symbolic_forward_source,
            target_sampling_source=target_sampling_source, embedding_token_ids=values)
            for phase, values in embedding_token_ids_by_phase.items()}
    if symbolic_forward_source not in (*glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce",
            "vllm_0_27_1_deepseek_v3_sync_fp8"):
        raise ValueError("ordinary source probe requires its explicit GLM/K2.6/DeepSeek-V3 forward source")
    if not isinstance(embedding_token_ids_by_phase, Mapping) or not embedding_token_ids_by_phase:
        raise ValueError("ordinary source probe requires explicit phase token IDs")
    if source_max_model_len is None or source_max_batched_tokens is None:
        raise ValueError("ordinary source probe requires explicit context and batch constructor limits")
    native = model_source_inputs_by_phase is not None
    if native:
        if (symbolic_forward_source not in glm_sources or not isinstance(model_source_inputs_by_phase, Mapping)
                or set(model_source_inputs_by_phase) != set(embedding_token_ids_by_phase)
                or any(not isinstance(state, V32PaddedMTPInputs) for state in model_source_inputs_by_phase.values())
                or mtp_draft_sampling_source != "vllm_0_30_legacy_full_logits_greedy"
                or type(source_max_requests) is not int or source_max_requests < 1
                or glm_prefill_attention_source != "force_sparse_mqa"):
            raise ValueError("GLM native source requires typed inputs for every phase, full-logits greedy, request capacity and sparse prefill source")
        if target_sampling_source != "vllm_0_30_rejection_raw_logits_bf16":
            raise ValueError("GLM native source requires its raw-logits rejection target chain")
    elif source_max_requests is not None or mtp_draft_sampling_source is not None:
        raise ValueError("native source fields require explicit per-phase V32 inputs")
    elif target_sampling_source not in ((None, "vllm_0_27_1_plain_greedy_bf16") if dsv3_source else
            (None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")):
        raise ValueError("ordinary source probe requires logits-only, a versioned FP32 cast or an explicit plain greedy target chain")
    if symbolic_forward_source in glm_sources:
        if workspace_initial_capacity_bytes is None or glm_indexer_selection_source is None:
            raise ValueError("GLM source probe requires explicit rank workspace history and indexer cardinality source")
    elif any(value is not None for value in (workspace_initial_capacity_bytes,
            glm_indexer_selection_source, glm_prefill_attention_source)):
        raise ValueError("K2.6/DeepSeek-V3 source probe does not consume GLM workspace/indexer options")
    return {phase: RuntimeInputs(
        ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0),
        "mla", "uniform", model_execution_version="vllm_0_27_1" if dsv3_source else "vllm_0_30_0",
        symbolic_forward_source=symbolic_forward_source, target_sampling_source=target_sampling_source,
        symbolic_source_max_model_len=source_max_model_len, symbolic_source_max_batched_tokens=source_max_batched_tokens,
        symbolic_source_max_requests=source_max_requests, mtp_draft_sampling_source=mtp_draft_sampling_source,
        model_source_inputs=None if not native else model_source_inputs_by_phase[phase],
        symbolic_workspace_initial_capacity_bytes=workspace_initial_capacity_bytes,
        glm_indexer_selection_source=glm_indexer_selection_source, glm_prefill_attention_source=glm_prefill_attention_source,
        embedding_token_ids=values) for phase, values in embedding_token_ids_by_phase.items()}


def _operator_probe_phase_requests(prefill_tokens, mtp_mode, *, decode_prefix=False):
    """One declared request fixture, shared by source inputs and execution."""
    if mtp_mode == "native":
        def request(request_id, phase, input_tokens, output_tokens, computed, accepted, verified):
            return ScheduledRequest(
                request_id, phase, input_tokens, output_tokens, computed,
                raw_model_output_tokens=accepted + 1,
                mtp_proposed_tokens=verified, mtp_verified_tokens=verified,
                mtp_accepted_tokens=accepted, mtp_proposer_tokens=2,
                mtp_next_proposed_tokens=2, mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="static_explicit_accept_reject_array")

        return (
            ("prefill", (request("request", "prefill", prefill_tokens, 0, 0, 0, 0),)),
            ("decode1", (request("request", "decode", 0, 2, prefill_tokens, 1, 1),)),
            ("decode2", (request("request", "decode", 0, 2, prefill_tokens + 2, 0, 1),)),
            ("mixed", (request("new-request", "prefill", prefill_tokens, 0, 0, 0, 0),
                       request("request", "decode", 0, 2, prefill_tokens + 3, 0, 1))),
        )
    if mtp_mode == "off":
        mixed = (ScheduledRequest("new-request", "prefill", prefill_tokens, 0, 0),
            ScheduledRequest("request", "decode", 0, 1, prefill_tokens + 2))
        return (
            ("prefill", (ScheduledRequest("request", "prefill", prefill_tokens, 0, 0),)),
            ("decode1", (ScheduledRequest("request", "decode", 0, 1, prefill_tokens),)),
            ("decode2", (ScheduledRequest("request", "decode", 0, 1, prefill_tokens + 1),)),
            ("mixed", tuple(sorted(mixed, key=lambda request: request.scheduled_tokens > 1)) if decode_prefix else mixed),
        )
    raise ValueError("unknown static MTP mode")


def _declared_glm_padded_probe_state(step, target_ids, *, source_id):
    """Independent IDs in incoming-request and source-packed row orders."""
    draft_counts = {request.mtp_proposer_tokens for request in step.requests}
    if len(draft_counts) != 1 or next(iter(draft_counts)) < 1:
        raise ValueError("synthetic padded probe needs one explicit positive proposer depth")
    depth = next(iter(draft_counts))
    packed, packed_ids, _ = vllm_0_30_source_model_inputs(step, embedding_token_ids=target_ids,
        prompt_lengths=None, num_speculative_tokens=depth)
    first_draft = [*packed_ids[1:], 97]
    cursor = 0
    for index, request in enumerate(packed.requests):
        retained = cursor + request.scheduled_tokens - request.mtp_verified_tokens + request.mtp_accepted_tokens - 1
        first_draft[retained] = 59 + index
        cursor += request.scheduled_tokens
    return V32PaddedMTPInputs(request_ids=tuple(request.request_id for request in step.requests),
        source_kind="declared_synthetic", source_id=source_id,
        draft_input_token_ids=(tuple(first_draft), *(tuple(127 + draft + index for index in range(len(step.requests)))
            for draft in range(depth - 1))),
        discarded_rows=(False,) * len(step.requests), block_size_tokens=16,
        parallel_drafting=False, use_local_argmax_reduction=False, probabilistic=False)


def _default_glm_native_probe_source(model, parallel, phase_requests, *, sampler, draft_sampler):
    """Declare standalone padded phases, never sampled IDs or runtime state."""
    if (model.model_type != "glm_moe_dsa" or parallel.tensor_parallel_size not in (1, 8)
            or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel):
        return None
    if sampler not in (None, "vllm_0_30_rejection_raw_logits_bf16"):
        raise ValueError("default GLM native fixture requires its raw-logits rejection source")
    if draft_sampler not in (None, "vllm_0_30_legacy_full_logits_greedy"):
        raise ValueError("default GLM native fixture requires its full-logits greedy draft source")
    ports = {phase: tuple(range(1, 1 + sum(request.scheduled_tokens for request in requests)))
        for phase, requests in phase_requests}
    states = {phase: _declared_glm_padded_probe_state(ScheduleStep(phase, "main", requests), ports[phase],
        source_id="default-static-independent-padded-input-scenario") for phase, requests in phase_requests}
    # Include every physical recurrent pass even for requests whose next
    # proposals will not be consumed. This is a declared fixture cap.
    context = max(request.computed_tokens + request.scheduled_tokens + len(states[phase].draft_input_token_ids) - 1
        for phase, requests in phase_requests for request in requests)
    batch = max(sum(request.scheduled_tokens for request in requests) for _, requests in phase_requests)
    requests_cap = max(len(requests) for _, requests in phase_requests)
    source = "vllm_0_30_v32_" + model.weight_dtype + "_sync_explicit_allreduce"
    inputs = build_model_operator_probe_source_inputs(symbolic_forward_source=source,
        embedding_token_ids_by_phase=ports, source_max_model_len=context, source_max_batched_tokens=batch,
        source_max_requests=requests_cap, target_sampling_source="vllm_0_30_rejection_raw_logits_bf16",
        workspace_initial_capacity_bytes=(0,) * parallel.worker_count,
        glm_indexer_selection_source="declared_causal_topk", glm_prefill_attention_source="force_sparse_mqa",
        mtp_draft_sampling_source="vllm_0_30_legacy_full_logits_greedy", model_source_inputs_by_phase=states)
    receipt = {"entrypoint": "default_static_glm_native_fixture", "source_scenario": source,
        "synthetic_fixture": True, "historical_default_equivalence_claimed": False,
        "phase_scope": "independent standalone phases, not linked sampled outputs or runtime buffer history",
        "token_input_scope": "declared sequential target IDs; packed global shift plus independent retained/recurrent draft IDs",
        "source_constructor_inputs": {"max_model_len": context, "max_batched_tokens": batch,
            "max_requests": requests_cap, "workspace_initial_capacity_bytes": (0,) * parallel.worker_count},
        "target_sampling_source": "vllm_0_30_rejection_raw_logits_bf16",
        "mtp_draft_sampling_source": "vllm_0_30_legacy_full_logits_greedy",
        "requested_target_sampling_source": sampler, "requested_draft_sampling_source": draft_sampler,
        "sampler_scope": "explicit synthetic rejection/full-logits-greedy source scenario; not observed dispatch",
        "source_projection_only": True, "actual_provider_runtime_bound": False, "normal_capacity_accepted": False}
    return inputs, receipt


def _default_glm4_target_probe_source(model, parallel, phase_requests, *, token_ports, sampler):
    """Correct the old bare GLM4 probe's0.30/MLA mix with its0.11 source."""
    if (model.model_type != "glm4_moe" or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel):
        return None
    if sampler not in (None, "vllm_0_11_plain_sampler_bf16", "vllm_0_11_plain_greedy_bf16"):
        raise ValueError("GLM4 probe requires an accurate0.11 output source")
    synthetic_ids = token_ports is None
    if synthetic_ids:
        token_ports = {phase: tuple(index % model.vocab_size for index in range(
            sum(request.scheduled_tokens for request in requests))) for phase, requests in phase_requests}
    routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
    inputs = {phase: RuntimeInputs(routing, "paged_varlen", "uniform", model_execution_version="vllm_0_11_0",
        symbolic_forward_source="vllm_0_11_glm4_sync", target_sampling_source=sampler,
        embedding_token_ids=ids) for phase, ids in token_ports.items()}
    return inputs, {"entrypoint": "default_static_glm4_target_fixture", "synthetic_fixture": True,
        "source_scenario": "H200_vLLM0.11_BF16_GLM4_TP1_PP1_eager",
        "model_source_profile": "vllm_0_11_glm4_sync",
        "source_correction": "old generic probe declared vLLM0.30/MLA; GLM4 uses0.11 GQA paged-varlen source",
        "historical_default_equivalence_claimed": False,
        "token_input_scope": "declared sequential IDs modulo vocabulary per phase" if synthetic_ids else "caller-provided IDs",
        "target_sampling_source": sampler, "source_projection_only": True,
        "actual_provider_runtime_bound": False, "normal_capacity_accepted": False}


def _default_target_probe_source(model, parallel, phase_requests, *, token_ports, sampler,
        v4_routing_source, v4_max_model_len, v4_max_batched_tokens, v4_indexer_selection,
        v4_use_deep_gemm):
    """Declare a reproducible synthetic static fixture, never runtime facts.

    The old bare probe supplied no constructor history or token values.
    These explicit choices define the new virtual source scenario; they
    are not an equivalence claim about the historical default or a provider.
    """
    tp, pp = parallel.tensor_parallel_size, parallel.pipeline_parallel_size
    glm = model.model_type == "glm_moe_dsa"
    kimi = model.model_type == "kimi_k2" and model.model_id == "moonshotai/Kimi-K2.6"
    v3, v4 = model.model_type == "deepseek_v3", model.model_type == "deepseek_v4"
    if not (glm or kimi or v3 or v4):
        return None
    if ((glm or kimi or v3) and tp not in (1, 8) or v4 and tp not in (4, 8)
            or parallel.enable_expert_parallel and (not (glm or kimi) or tp != 8)
            or pp != 1 and not (glm and tp == 8 and not parallel.enable_expert_parallel)):
        return None
    expected_sampler = "vllm_0_27_1_plain_greedy_bf16" if v3 else "vllm_0_30_plain_greedy_bf16"
    if sampler not in ((None, expected_sampler) if v3 else
            (None, expected_sampler, "vllm_0_30_plain_sampler_bf16")):
        return None
    if v4 and v4_routing_source not in (None, "vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text"):
        return None
    original_order = {phase: tuple(request.request_id for request in requests) for phase, requests in phase_requests}
    original_ports = token_ports
    if v3:
        repacked = []
        token_ports = None if original_ports is None else {}
        for phase, requests in phase_requests:
            ordered = tuple(sorted(requests, key=lambda request: request.scheduled_tokens > 1))
            repacked.append((phase, ordered))
            if original_ports is not None:
                cursor, parts = 0, {}
                for request in requests:
                    parts[request.request_id] = original_ports[phase][cursor:cursor + request.scheduled_tokens]
                    cursor += request.scheduled_tokens
                token_ports[phase] = tuple(token for request in ordered for token in parts[request.request_id])
        phase_requests = tuple(repacked)
    max_context = max(request.computed_tokens + request.scheduled_tokens
        for _, requests in phase_requests for request in requests)
    max_rows = max(sum(request.scheduled_tokens for request in requests) for _, requests in phase_requests)
    if token_ports is None:
        token_ports = {phase: tuple(index % model.vocab_size for index in range(
            sum(request.scheduled_tokens for request in requests))) for phase, requests in phase_requests}
    source = ("vllm_0_30_v32_bf16_sync_explicit_allreduce" if glm and model.weight_dtype == "bf16" else
        "vllm_0_30_v32_fp8_sync_explicit_allreduce" if glm else
        "vllm_0_30_k2_6_int4_sync_explicit_allreduce" if kimi else
        "vllm_0_27_1_deepseek_v3_sync_fp8" if v3 else "vllm_0_30_v4_sync_fp8_ds_mla")
    if v4:
        context = max_context if v4_max_model_len is None else v4_max_model_len
        batch = max_rows if v4_max_batched_tokens is None else v4_max_batched_tokens
        branch = False if v4_use_deep_gemm is None else v4_use_deep_gemm
        routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
        inputs = {phase: RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            symbolic_forward_source=source, target_sampling_source=sampler, embedding_token_ids=tuple(ids),
            symbolic_workspace_initial_capacity_bytes=(0,) * parallel.worker_count,
            v4_eager_metadata_source="vllm_0_30_eager_dict_all_valid",
            v4_routing_source="vllm_0_30_modular_cuda_top6_pure_text" if v4_routing_source is None else v4_routing_source,
            v4_source_max_model_len=context, v4_source_max_batched_tokens=batch,
            v4_source_use_deep_gemm=branch,
            v4_indexer_selection_source="declared_causal_topk" if v4_indexer_selection is None else v4_indexer_selection)
            for phase, ids in token_ports.items()}
    else:
        inputs = build_model_operator_probe_source_inputs(symbolic_forward_source=source,
            embedding_token_ids_by_phase=token_ports, source_max_model_len=max_context,
            source_max_batched_tokens=max_rows, target_sampling_source=sampler,
            workspace_initial_capacity_bytes=(0,) * parallel.worker_count if glm else None,
            glm_indexer_selection_source="declared_causal_topk" if glm else None)
    receipt = {"entrypoint": "default_static_target_fixture", "source_scenario": source,
        "synthetic_fixture": True, "historical_default_equivalence_claimed": False,
        "constructor_scope": "fixture minimum context/batch extents; not observed runtime configuration",
        "token_input_scope": "caller-provided IDs preserved per request" if original_ports is not None else
            "declared sequential IDs modulo vocabulary per phase; not model outputs or observed tokens",
        "phase_request_order_before": original_order,
        "phase_request_order_after": {phase: tuple(request.request_id for request in requests)
            for phase, requests in phase_requests},
        "source_constructor_inputs": {phase: {name: vars(runtime)[name] for name in (
            "symbolic_source_max_model_len", "symbolic_source_max_batched_tokens",
            "symbolic_workspace_initial_capacity_bytes", "glm_indexer_selection_source",
            "v4_source_max_model_len", "v4_source_max_batched_tokens", "v4_eager_metadata_source",
            "v4_routing_source", "v4_source_use_deep_gemm", "v4_indexer_selection_source")}
            for phase, runtime in inputs.items()},
        "workspace_scope": "declared cold source workspace per physical rank" if glm or v4 else "not selected by this fixture",
        "target_sampling_source": sampler, "source_projection_only": True,
        "actual_provider_runtime_bound": False, "normal_capacity_accepted": False}
    return phase_requests, inputs, receipt


def build_model_operator_probe(
    model_path: str,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
    *,
    mtp_mode: str = "off",
    prefill_tokens: int = 8,
    v4_eager_metadata: bool = False,
    v4_routing_source: str | None = None,
    pipeline_parallel_size: int = 1,
    pipeline_link: str | None = None,
    pipeline_layer_partition: str | None = None,
    source_graph_only: bool = False,
    mtp_draft_sampling_source: str | None = None,
    target_sampling_source: str | None = None,
    embedding_token_ids_by_phase: Mapping[str, tuple[int, ...]] | None = None,
    v4_source_max_model_len: int | None = None,
    v4_source_max_batched_tokens: int | None = None,
    v4_indexer_selection_source: str | None = None,
    v4_source_use_deep_gemm: bool | None = None,
    source_phase_requests: tuple[tuple[str, tuple[ScheduledRequest, ...]], ...] | None = None,
    source_runtime_inputs_by_phase: Mapping[str, RuntimeInputs] | None = None,
) -> dict[str, object]:
    """Run full layers for prefill, two decode steps, and a mixed step.

    This emits modeled facts, not independent reference expectations or a
    success gate. Native GLM-on uses explicit acceptance/rejection inputs;
    real Scheduler execution and final source alignment remain unverified.
    """
    if isinstance(prefill_tokens, bool) or not isinstance(prefill_tokens, int) or prefill_tokens < 1:
        raise ValueError("static probe prefill_tokens must be a positive integer")
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    model = load_model_config(model_path)
    if not isinstance(source_graph_only, bool):
        raise TypeError("source_graph_only must be boolean")
    if source_graph_only and (model.model_type != "deepseek_v4" or mtp_mode != "off"):
        raise NotImplementedError("this provenance-only graph capture requires V4 target-off")
    if type(pipeline_parallel_size) is not int or pipeline_parallel_size not in (1, 2):
        raise ValueError("static source probe supports PP1 or GLM target-off PP2")
    if pipeline_parallel_size == 2:
        if (model.model_type != "glm_moe_dsa" or mtp_mode != "off" or pipeline_link != "nvlink"
                or pipeline_layer_partition != "38,40"):
            raise NotImplementedError("PP2 source probe requires GLM target-off, explicit NVLink scenario and partition 38,40")
    elif pipeline_link is not None or pipeline_layer_partition is not None:
        raise ValueError("pipeline link applies only to PP2 source probe")
    if not isinstance(v4_eager_metadata, bool):
        raise TypeError("v4_eager_metadata must be boolean")
    if v4_eager_metadata and model.model_type != "deepseek_v4":
        raise ValueError("V4 eager source metadata requires a V4 model")
    if mtp_mode not in ("off", "native"):
        raise ValueError("unknown static MTP mode")
    if mtp_mode == "native" and model.model_type != "glm_moe_dsa":
        raise NotImplementedError("native static probe currently requires exact GLM V32 config")
    if mtp_draft_sampling_source is not None and mtp_mode != "native":
        raise ValueError("draft sampling source requires native probe")
    if target_sampling_source in ("vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16") and mtp_mode != "off":
        raise ValueError("plain target sampling source requires MTP-off probe")
    if target_sampling_source == "vllm_0_30_rejection_raw_logits_bf16" and mtp_mode != "native":
        raise ValueError("rejection target sampling source requires native probe")
    if source_runtime_inputs_by_phase is not None:
        if not isinstance(source_runtime_inputs_by_phase, Mapping):
            raise TypeError("source runtime inputs require a phase mapping")
        if (v4_eager_metadata or v4_routing_source is not None
                or mtp_draft_sampling_source is not None or target_sampling_source is not None
                or embedding_token_ids_by_phase is not None
                or v4_source_max_model_len is not None or v4_source_max_batched_tokens is not None
                or v4_indexer_selection_source is not None
                or v4_source_use_deep_gemm is not None):
            raise ValueError("source runtime inputs conflict with independent legacy runtime arguments")
    collectives = tuple(
        CollectiveConfig(
            operation, "ring", "nvlink", "ring", "main-collective-fabric",
            "analytical_roofline_v1", None)
        for operation in ("all_reduce", "all_gather"))
    parallel = ParallelConfig(
        tensor_parallel_size=tensor_parallel_size, pipeline_parallel_size=pipeline_parallel_size,
        enable_expert_parallel=enable_expert_parallel)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main", "colocated", "nvidia_h200", parallel,
                ranks=tuple(range(parallel.worker_count)),
                collectives=collectives, pipeline_p2p=(
                    (PipelineP2PConfig(0, 1, "send_recv", pipeline_link, "point_to_point",
                        "explicit_source_pipeline_fabric", "analytical_roofline_v1"),)
                    if pipeline_parallel_size == 2 else ())),)),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts,
        pipeline_layer_partition=(38, 40) if pipeline_parallel_size == 2 else None)
    routing = ExpertRoutingInput.seeded_assumption(
        "operator_support_static_seeded_routing_v1", 0)
    runtime = RuntimeInputs(
        routing, "paged_varlen" if model.model_type == "glm4_moe" else "mla", "uniform",
        model_execution_version="vllm_0_11_0" if model.model_type == "glm4_moe" else "vllm_0_30_0",
        symbolic_forward_source="vllm_0_11_glm4_sync" if model.model_type == "glm4_moe" else None,
        v4_eager_metadata_source="vllm_0_30_eager_dict_all_valid" if v4_eager_metadata else None,
        v4_source_max_model_len=v4_source_max_model_len,
        v4_source_max_batched_tokens=v4_source_max_batched_tokens,
        v4_source_use_deep_gemm=v4_source_use_deep_gemm,
        v4_routing_source=v4_routing_source)
    if mtp_draft_sampling_source is not None:
        runtime = replace(runtime, mtp_draft_sampling_source=mtp_draft_sampling_source)
    if target_sampling_source is not None:
        runtime = replace(runtime, target_sampling_source=target_sampling_source)
    pp_weight_projection = None
    if pipeline_parallel_size == 2:
        rank_weights = model_rank_residencies(model, parallel, worker_role="colocated", include_mtp=False)
        pp_weight_projection = {
            "scope": "existing owner target-off static weight projection; not normal allocator evidence",
            "maximum_rank_weight_bytes": max(rank.weight_resident_bytes for rank in rank_weights),
            "physical_hbm_spec_bytes": hardware.resource_by_role("dram").size_bytes,
            "rank_residencies": [asdict(rank) for rank in rank_weights],
            "reserve_activation_cache_runtime_allocation_bound": False,
        }
    phase_requests = _operator_probe_phase_requests(prefill_tokens, mtp_mode)
    if (source_phase_requests is None and source_runtime_inputs_by_phase is not None
            and source_runtime_inputs_by_phase and all(isinstance(inputs, RuntimeInputs)
                and inputs.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
                for inputs in source_runtime_inputs_by_phase.values())):
        # This static fixture declares model-input order, not a Scheduler or
        # persistent InputBatch reorder observation. V3's common source
        # requires query-length-one requests before longer prefill rows.
        # Supplied phase IDs are packed in this declared fixture order.
        phase_requests = _operator_probe_phase_requests(prefill_tokens, mtp_mode, decode_prefix=True)
    if source_phase_requests is not None:
        if (mtp_mode != "off" or (source_runtime_inputs_by_phase is None
                and (model.model_type != "deepseek_v4" or not v4_eager_metadata))):
            raise ValueError("explicit source phase requests require target-off and complete phase RuntimeInputs or V4 eager metadata")
        if (not isinstance(source_phase_requests, tuple) or not source_phase_requests
                or any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
                       or not isinstance(item[1], tuple) or not item[1]
                       or any(not isinstance(request, ScheduledRequest) for request in item[1]) for item in source_phase_requests)
                or len({item[0] for item in source_phase_requests}) != len(source_phase_requests)):
            raise ValueError("source phase requests require unique named nonempty ScheduledRequest tuples")
        phase_requests = source_phase_requests
    if embedding_token_ids_by_phase is not None:
        if not isinstance(embedding_token_ids_by_phase, Mapping):
            raise TypeError("embedding tokens require a phase mapping")
        if mtp_mode != "off":
            raise NotImplementedError("explicit embedding tokens require a lowered target-off source probe")
        if set(embedding_token_ids_by_phase) != {phase for phase, _ in phase_requests}:
            raise ValueError("embedding tokens require exactly the declared source probe phases")
    source_normalization = None
    if v4_indexer_selection_source is not None and v4_source_max_batched_tokens is None:
        raise ValueError("indexer source requires the explicit virtual batch constructor cap")
    if v4_source_max_batched_tokens is not None:
        if (model.model_type != "deepseek_v4" or model.compute_dtype != "bf16" or model.weight_dtype != "fp8"
                or tensor_parallel_size not in (4, 8) or pipeline_parallel_size != 1 or enable_expert_parallel
                or mtp_mode != "off" or not v4_eager_metadata
                or v4_routing_source not in ("vllm_0_30_modular_cuda_top6_pure_text", "vllm_0_30_triton_top6_pure_text")
                or embedding_token_ids_by_phase is None or v4_source_max_model_len is None
                or v4_source_use_deep_gemm is None or target_sampling_source not in (
                    None, "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_sampler_bf16")):
            raise ValueError("explicit V4 batch capacity requires the complete target-off eager source facts")
        # The persistent top-k buffer uses the caller's constructor capacity,
        # never a capacity inferred from these static scheduled rows.
        source_runtime_inputs_by_phase = {phase: replace(runtime,
            symbolic_forward_source="vllm_0_30_v4_sync_fp8_ds_mla",
            v4_indexer_selection_source=v4_indexer_selection_source,
            embedding_token_ids=embedding_token_ids_by_phase[phase]) for phase, _ in phase_requests}
        source_normalization = {"entrypoint": "historical_v4_probe_arguments",
            "source_scenario": f"H200_vLLM0.30_FP8_V4_TP{tensor_parallel_size}_PP1_eager",
            "target_sampling_source": target_sampling_source, "source_projection_only": True,
            "actual_provider_runtime_bound": False, "token_values_inferred": False}
    ordinary_dense = model.model_type in ("qwen3", "llama") and not model.num_experts
    ordinary_moe = model.model_type == "qwen3_moe" and model.num_experts in (2, 128)
    dense_tp2 = ordinary_dense and tensor_parallel_size == 2
    moe_ep2 = (ordinary_moe and tensor_parallel_size == 2 and enable_expert_parallel
        and is_validated_qwen3_moe_ep2_model_scope(model))
    moe_tp2 = ordinary_moe and tensor_parallel_size == 2 and not enable_expert_parallel
    if (source_runtime_inputs_by_phase is None and (ordinary_dense or ordinary_moe)
            and model.compute_dtype == model.weight_dtype and model.compute_dtype in ("bf16", "fp16")
            and (model.compute_dtype == "bf16" or model.model_type == "qwen3")
            and (tensor_parallel_size == 1 or dense_tp2 or moe_ep2 or moe_tp2) and pipeline_parallel_size == 1
            and (not enable_expert_parallel or moe_ep2)
            and mtp_mode == "off" and target_sampling_source in ((None, "vllm_0_30_plain_greedy_bf16",
                "vllm_0_30_plain_sampler_bf16") if model.compute_dtype == "bf16" else
                (None, "vllm_0_30_plain_greedy_fp16"))
            and not source_graph_only and not v4_eager_metadata
            and all(value is None for value in (v4_routing_source, v4_source_max_model_len,
                v4_source_use_deep_gemm, mtp_draft_sampling_source))):
        # Preserve the historical probe's H200/0.30 source scenario and
        # its explicitly chosen LM-head-only or greedy scope. No sampler preference, token
        # value, provider or constructor state is inferred from a checkpoint.
        ports = ({phase: None for phase, _ in phase_requests} if embedding_token_ids_by_phase is None
            else embedding_token_ids_by_phase)
        source_runtime_inputs_by_phase = build_model_operator_probe_source_inputs(
            symbolic_forward_source="cuda_eager_inplace", embedding_token_ids_by_phase=ports,
            source_max_model_len=None, source_max_batched_tokens=None, target_sampling_source=target_sampling_source)
        if ordinary_moe:
            # Reuse the probe's declared routing, never derive an expert
            # histogram or token values from the checkpoint/config.
            source_runtime_inputs_by_phase = {phase: replace(inputs, expert_routing=routing)
                for phase, inputs in source_runtime_inputs_by_phase.items()}
        source_normalization = {"entrypoint": "historical_moe_probe_arguments" if ordinary_moe else "historical_dense_probe_arguments",
            "source_scenario": ("H200_vLLM0.30_BF16_Qwen3_MoE_TP2_EP2_PP1_eager" if moe_ep2
                else "H200_vLLM0.30_BF16_Qwen3_MoE_TP2_EP1_PP1_eager" if moe_tp2
                else "H200_vLLM0.30_BF16_Qwen3_MoE_TP1_PP1_eager" if ordinary_moe
                else f"H200_vLLM0.30_{model.compute_dtype.upper()}_dense_TP{tensor_parallel_size}_PP1_eager"),
            "target_sampling_source": target_sampling_source, "source_projection_only": True,
            "actual_provider_runtime_bound": False, "token_values_inferred": False}
    if source_runtime_inputs_by_phase is None and mtp_mode == "off":
        fixture = _default_target_probe_source(model, parallel, phase_requests,
            token_ports=embedding_token_ids_by_phase, sampler=target_sampling_source,
            v4_routing_source=v4_routing_source, v4_max_model_len=v4_source_max_model_len,
            v4_max_batched_tokens=v4_source_max_batched_tokens, v4_indexer_selection=v4_indexer_selection_source,
            v4_use_deep_gemm=v4_source_use_deep_gemm)
        if fixture is not None:
            phase_requests, source_runtime_inputs_by_phase, source_normalization = fixture
        elif model.model_type == "glm4_moe":
            if any(value is not None for value in (v4_routing_source, v4_source_max_model_len,
                    v4_source_max_batched_tokens, v4_indexer_selection_source, v4_source_use_deep_gemm)):
                raise ValueError("GLM4 probe does not consume V4 source inputs")
            fixture = _default_glm4_target_probe_source(model, parallel, phase_requests,
                token_ports=embedding_token_ids_by_phase, sampler=target_sampling_source)
            if fixture is not None:
                source_runtime_inputs_by_phase, source_normalization = fixture
    if source_runtime_inputs_by_phase is None and mtp_mode == "native":
        if any(value is not None for value in (v4_routing_source, v4_source_max_model_len,
                v4_source_max_batched_tokens, v4_indexer_selection_source, v4_source_use_deep_gemm)):
            raise ValueError("GLM native fixture does not consume V4 source options")
        fixture = _default_glm_native_probe_source(model, parallel, phase_requests,
            sampler=target_sampling_source, draft_sampler=mtp_draft_sampling_source)
        if fixture is not None:
            source_runtime_inputs_by_phase, source_normalization = fixture
    if source_runtime_inputs_by_phase is None:
        raise NotImplementedError(
            "static probe arguments do not select a maintained common source: "
            f"model={model.model_type}, TP={tensor_parallel_size}, PP={pipeline_parallel_size}, "
            f"EP={enable_expert_parallel}, MTP={mtp_mode}, sampler={target_sampling_source!r}")
    if set(source_runtime_inputs_by_phase) != {phase for phase, _ in phase_requests}:
        raise ValueError("source runtime inputs require exactly the declared probe phases")
    from src.serving.workload.traced_models import bind_ordinary_forward_source
    normalized_inputs = {}
    for phase, requests in phase_requests:
        supplied = source_runtime_inputs_by_phase[phase]
        if not isinstance(supplied, RuntimeInputs):
            raise TypeError("every source phase requires RuntimeInputs")
        supplied = bind_ordinary_forward_source(model, deployment, supplied)
        if supplied.symbolic_forward_source is None:
            raise ValueError("every source phase requires complete facts for a shared symbolic forward source")
        normalized_inputs[phase] = supplied
        # Unknown IDs are real virtual input ports. The common source
        # builder owns their exact applicability; TP/EP fused embedding
        # and other value-dependent domains still require token values.
        if (supplied.embedding_token_ids is not None
                and len(supplied.embedding_token_ids) != sum(request.scheduled_tokens for request in requests)):
            raise ValueError("source embedding token IDs must cover every scheduled row")
    if any(normalized_inputs[phase] is not supplied for phase, supplied in source_runtime_inputs_by_phase.items()):
        source_normalization = {"entrypoint": "programmatic_phase_runtime_inputs",
            "source_projection_only": True, "actual_provider_runtime_bound": False,
            "token_values_inferred": False, "sampler_source_inferred": False}
    source_runtime_inputs_by_phase = normalized_inputs

    def runtime_field(name):
        values = tuple(vars(source_runtime_inputs_by_phase[phase])[name] for phase, _ in phase_requests)
        return values[0] if all(value == values[0] for value in values) else None

    static_lm_head_only = (mtp_mode == "off" and source_runtime_inputs_by_phase is not None and all(
        source_runtime_inputs_by_phase[phase].target_sampling_source is None for phase, _ in phase_requests))
    static_cast_only = (mtp_mode == "off" and source_runtime_inputs_by_phase is not None and all(
        source_runtime_inputs_by_phase[phase].target_sampling_source in (
            "vllm_0_11_plain_sampler_bf16", "vllm_0_30_plain_sampler_bf16")
        for phase, _ in phase_requests))
    static_output_scope = ("lm_head_only_no_sampler" if static_lm_head_only else
        "fp32_logits_only" if static_cast_only else None)
    results = []
    for phase, requests in phase_requests:
        phase_runtime = source_runtime_inputs_by_phase[phase]
        step = ScheduleStep(phase, "main", requests, attention_kv_layout=phase_runtime.attention_kv_layout)
        source_storage, source_graph = None, None

        def observe_source(source, observed_graph):
            nonlocal source_storage, source_graph
            from src.serving.kv_cache.activation_memory import symbolic_forward_storage_report
            if source_storage is not None:
                raise RuntimeError("static probe executed its source forward more than once")
            source_storage = symbolic_forward_storage_report(source, observed_graph,
                source_profile=phase_runtime.symbolic_forward_source)
            source_graph = observed_graph

        graph = build_symbolic_workload(step, model, deployment, launch_policy="analytical_launch",
            kv_cache_events=(), runtime_inputs=phase_runtime, hardware_by_id={"nvidia_h200": hardware},
            trace_observer=observe_source)
        if source_storage is None or source_graph is not graph:
            raise RuntimeError("static probe lost the storage facts of its actual source graph")
        if source_graph_only:
            results.append({
                "phase": phase,
                "source_runtime_inputs": asdict(phase_runtime),
                "source_activation_memory": source_storage,
                "scheduled_requests": [asdict(request) for request in requests],
                "nodes": [{
                    "node_id": node.node_id, "rank": node.rank,
                    "kind": node.kind.value, "variant": node.variant,
                    "shape": node.shape, "dtype": node.dtype,
                    "attributes": node.attributes,
                } for node in graph.nodes],
                "edges": [asdict(edge) for edge in graph.edges],
                "block_repeats": [asdict(repeat) for repeat in graph.block_repeats],
            })
            continue
        plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
        calls = []
        priced_predictions = {}
        for call in plan.calls:
            kernel = call.kernel_spec
            try:
                result = predict(kernel, hardware, method="roofline")
            except NotImplementedError as error:
                prediction = {"method": "roofline", "seconds": None, "status": "unpriced", "reason": str(error)}
            else:
                prediction = {"method": "roofline", "seconds": result.t_total_s,
                    "status": "priced", "scope": "logical_roofline_only_no_native_launch_acceptance"}
            resource = {"status": "public_resource_input_unavailable", "demand": None, "limitations": ()}
            try:
                if kernel.execution_support == "unsupported":
                    resource = {
                        "status": "launch_selector_unavailable",
                        "demand": None, "limitations": (kernel.unsupported_reason,),
                    }
                    if kernel.kind is KernelKind.VECTOR and kernel.variant in (
                            "mhc_prenorm", "mhc_prenorm_broadcast"):
                        work = mhc_prenorm_roofline_work(kernel)
                        resource["logical_scope"] = work['arithmetic_contract']
                        resource.update({"logical_work": work["logical_work"],
                                         "demand": work["demand"], "logical_bytes": work["logical_bytes"],
                                         "limitations": work["limitations"]})
                        resource["status"] = "source_logical_work_launch_unbound"
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_indexer_q_rope_quant":
                        work = dsv4_indexer_q_roofline_work(kernel)
                        resource["logical_scope"] = work['scope']
                        resource.update({"logical_work": work["logical_work"], "logical_bytes": work["logical_bytes"],
                                         "roofline_source_programs": {**work["source_programs"],
                                                                      "all_source_branches_memory_dominated": True}})
                        resource["status"] = "source_logical_work_launch_unbound"
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vllm_logits_gather_reorder":
                        work = vllm_logits_gather_reorder_logical_work(kernel)
                        resource["logical_scope"] = work['scope']
                        resource.update({"status": "source_logical_work_launch_unbound", "logical_work": work,
                            "logical_bytes": sum(operand["bytes"] for operand in work["operands"]), "demand": work["demand"]})
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant in (
                            "vllm_logits_fp32_cast", "vllm_rejection_logits_materialize"):
                        work = (vllm_logits_fp32_cast_logical_work(kernel)
                                if kernel.variant == "vllm_logits_fp32_cast"
                                else vllm_rejection_logits_materialization_logical_work(kernel))
                        resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                        resource.update({"status": "source_logical_work_launch_unbound",
                                         "logical_work": work,
                                         "logical_bytes": sum(operand["bytes"] for operand in work["operands"]),
                                         "limitations": work["limitations"]})
                        if kernel.variant == "vllm_logits_fp32_cast":
                            resource["demand"] = work["demand"]
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vllm_plain_greedy_sample":
                        work = vllm_plain_greedy_roofline_work(kernel)
                        resource["logical_scope"] = work['arithmetic_contract']
                        resource.update({"status": "source_logical_work_launch_unbound",
                                         "logical_work": work["logical_work"], "demand": work["demand"],
                                         "logical_bytes": work["logical_bytes"],
                                         "limitations": work["limitations"]})
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant in (
                            "dsv32_draft_argmax", "dsv32_draft_id_cast", "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"):
                        resource["logical_work"] = dsv32_draft_sampling_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                        if kernel.variant in DSV32_DRAFT_MATERIALIZATION_VARIANTS:
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = dsv32_draft_materialization_roofline_bytes(kernel)
                            resource["limitations"] = (
                                "typed_logical_materialization_bytes_not_observed_dram_traffic",
                                "zero_float_work_not_zero_integer_address_control_or_conversion_services",
                                "actual_library_launch_compiler_provider_and_runtime_unbound")
                        elif (kernel.variant in DSV32_DRAFT_INT32_BOOKKEEPING_VARIANTS
                              and (kernel.attributes["index_preparation_mode"] == "padded_spec"
                                   or (kernel.variant == "dsv32_draft_prepare_sample_indices"
                                       and kernel.attributes["index_preparation_mode"] == "ordinary_query_end"))):
                            logical_bytes, int32_ops = dsv32_draft_int32_bookkeeping_roofline_work(kernel)
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = logical_bytes
                            resource["logical_int32_operations"] = int32_ops
                            resource["limitations"] = (
                                "declared_metadata_source_mode_not_observed_scheduler_output",
                                "source_int32_arithmetic_comparison_select_not_compiled_instruction_count",
                                "logical_role_bytes_not_observed_dram_or_cache_traffic",
                                "address_control_library_launch_compiler_and_runtime_services_unbound")
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vllm_logits_row_select":
                        resource["logical_work"] = vllm_logits_row_selection_logical_work(kernel)
                        resource["demand"] = resource["logical_work"]["demand"]
                        resource["status"] = "source_logical_work_launch_unbound"
                        if kernel.attributes["index_source"] != "unbound_spec_metadata":
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = vllm_logits_row_selection_roofline_bytes(kernel)
                            resource["limitations"] = (
                                "typed_logical_copy_and_index_bytes_not_observed_dram_traffic",
                                "zero_mathematical_flops_not_zero_address_or_control_instructions",
                                "actual_provider_library_launch_compiler_and_runtime_unbound")
                    elif (kernel.kind is KernelKind.VECTOR and kernel.variant == "fp8_group_quant"
                          and kernel.attributes["source_program"] is not None):
                        work = fp8_group_quant_source_work(kernel)
                        resource.update({"status": "conditional_source_arithmetic_byte_baseline", "logical_work": work,
                            "logical_bytes": work["logical_bytes"], "source_read_write_bytes": work["source_read_write_bytes"],
                            "all_source_programs_memory_dominated": True})
                        resource["logical_scope"] = 'common ideal arithmetic/byte Roofline for all source alternatives; no provider, scale-mode or native issue binding'
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "vocab_parallel_embedding":
                        work = vocab_parallel_embedding_logical_work(kernel)
                        resource["logical_work"] = work
                        resource["status"] = "source_logical_work_launch_unbound"
                        if work["owned_token_rows"] is not None:
                            resource["logical_scope"] = 'logical_roofline_only_no_launch_binding'
                            resource["logical_bytes"] = sum(operand["bytes"] for operand in work["operands"])
                            resource["limitations"] = work["limitations"]
                        else:
                            resource["limitations"] = (*resource["limitations"], 'scheduled_token_ownership_unbound')
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv32_moe_route":
                        resource["logical_work"] = dsv32_moe_route_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                        resource["limitations"] = (*resource["limitations"], 'grouped_topk_selection_and_nonlinear_logical_service_unbound')
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "kimi_moe_route":
                        resource["logical_work"] = kimi_moe_route_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_moe_route":
                        resource["logical_work"] = dsv4_moe_route_logical_work(kernel)
                        resource["status"] = "source_logical_work_launch_unbound"
                elif kernel.execution_support == "supported":
                    pass
                else:
                    raise ValueError("unknown kernel execution support")
                if kernel.execution_support == "unsupported":
                    pass  # Logical pricing and physical launch support are separate contracts.
                elif (kernel.kind is KernelKind.SPARSE_ATTN_INDEXER
                      and kernel.variant == "dsa_index_reuse"):
                    resource = {
                        "status": "logical_buffer_alias_no_kernel_work",
                        "demand": None, "limitations": (),
                    }
                elif kernel.kind in (
                        KernelKind.COMMUNICATION, KernelKind.MEMCPY,
                        KernelKind.KV_OFFLOAD):
                    resource = {
                        "status": "transfer_uses_own_physical_contract",
                        "demand": None, "limitations": (),
                    }
                elif kernel.kind is KernelKind.GEMM and is_block_fp8_gemm(kernel):
                    from src.kernel.predictors.gemm_block_fp8 import block_fp8_gemm_roofline_work
                    work = block_fp8_gemm_roofline_work(kernel)
                    resource = {
                        "status": "block_fp8_template_exposes_no_resource_input",
                        "demand": None,
                        "logical_work": work,
                        "logical_bytes": work["logical_bytes"],
                        "limitations": ("public_resource_input_unavailable",),
                    }
                else:
                    inp, logical_bytes = create_predictor(
                        kernel.kind).resource_input(kernel, hardware)
                    resource = {
                        "status": "public_source_resource_input",
                        "demand": inp.demand_per_resource,
                        "logical_bytes": logical_bytes,
                        "modeled_dram_bytes": {
                            access: sum(node.bytes for path in inp.memory_paths
                                        if path.access_kind == access for node in path.nodes
                                        if node.resource_role == "dram")
                            for access in ("read", "write")},
                        "modeled_l2_bytes": {
                            access: sum(node.bytes for path in inp.memory_paths
                                        if path.access_kind == access for node in path.nodes
                                        if node.resource_role == "l2_cache")
                            for access in ("read", "write")},
                        "limitations": inp.limitations,
                    }
                    if kernel.kind is KernelKind.SPARSE_ATTN_INDEXER and kernel.variant in ("dsv4_indexer_prefill", "dsv4_indexer_decode"):
                        resource["indexer_logical_work"] = dsv4_indexer_logical_work(kernel)
                        resource["logical_scope"] = resource['indexer_logical_work']['scope']
                    if kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant in ("dsv4_sparse_prefill", "dsv4_sparse_decode"):
                        resource["attention_inputs"] = dsv4_attention_source_inputs(kernel)
                        if kernel.variant == "dsv4_sparse_decode":
                            resource["decode_attention_index_inputs"] = dsv4_decode_attention_index_inputs(kernel)
                    if kernel.kind is KernelKind.VECTOR and kernel.variant == "silu" and kernel.attributes["swiglu_limit"] is not None:
                        resource["logical_work"] = clamped_silu_logical_work(kernel)
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_short_context_topk":
                        resource["logical_work"] = dsv4_short_context_topk_source_work(kernel)
                    elif kernel.kind is KernelKind.VECTOR and kernel.variant == "dsv4_moe_route":
                        resource["route_source_work"] = dsv4_route_source_work(kernel)
                        resource["route_tensor_objects"] = dsv4_route_tensor_objects(kernel)
                    elif (model.model_type == "deepseek_v4" and kernel.kind is KernelKind.VECTOR
                          and kernel.variant == "rmsnorm" and kernel.dtype == "bf16"):
                        resource["logical_work"] = rmsnorm_logical_work(kernel)
                    elif kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant == "dsv4_sparse_prefill":
                        resource["logical_work"] = dsv4_prefill_gather_logical_work(kernel)
                        resource["logical_scope"] = 'logical_common_work_and_causal_candidate_capacity_projection_not_provider_or_latency_bound'
                    elif (kernel.kind is KernelKind.SPARSE_ATTN and kernel.variant == "dsv4_sparse_decode"
                          and kernel.shape["compress_ratio"] == 4):
                        resource["decode_index_materialization"] = dsv4_decode_index_materialization_work(kernel)
                        resource["logical_scope"] = 'logical_c4_index_materialization_and_attention_projection_not_complete_source_or_provider_service'
                    elif kernel.kind is KernelKind.KV_CACHE and kernel.variant == "dsv4_qnorm_rope_kv_insert":
                        resource["swa_insert_objects"] = dsv4_swa_insert_tensor_objects(kernel)
                    elif kernel.kind is KernelKind.KV_CACHE and kernel.variant == "dsv4_compressed_state_cache":
                        resource["finish_work"] = dsv4_finish_logical_work(kernel)
                        resource["partial_state_input"] = dsv4_finish_partial_state_input(kernel)
                    elif kernel.kind is KernelKind.KV_CACHE and kernel.variant == "dsv4_compressor_state_cache":
                        resource["partial_state_objects"] = dsv4_partial_state_tensor_objects(kernel)
                    elif (model.model_type == "deepseek_v4" and kernel.kind is KernelKind.FUSED_MOE
                          and kernel.variant == "routed_experts" and kernel.attributes["expert_parallel_size"] == 1
                          and kernel.attributes["swiglu_limit"] is not None):
                        resource["logical_work"] = dsv4_marlin_moe_source_work(kernel)
                        resource["marlin_roofline"] = dsv4_marlin_moe_roofline_work(kernel, hardware)
                        resource["marlin_scratch"] = dsv4_marlin_moe_scratch_work(kernel, hardware)
                        resource["logical_scope"] = resource['marlin_roofline']['scope']
                if kernel.kind is KernelKind.VECTOR and kernel.variant in (
                        "mhc_prenorm", "mhc_prenorm_broadcast", "mhc_post_prenorm",
                        "mhc_pre_norm", "mhc_pre_norm_broadcast", "mhc_post", "hc_head"):
                    resource["mhc_stage_work"] = mhc_stage_logical_work(kernel)
            except NotImplementedError as error:
                resource["limitations"] = (*resource["limitations"], str(error))
            if prediction["status"] == "priced":
                if "logical_scope" in resource:
                    prediction["scope"] = resource["logical_scope"]
                if resource["status"] == "logical_buffer_alias_no_kernel_work":
                    if result.t_total_s != 0:
                        raise ValueError("logical buffer alias must have zero public prediction duration")
                    prediction["status"] = "logical_alias"
            launch = kernel.launch
            if prediction["status"] in ("priced", "logical_alias"):
                priced_predictions[call.call_id] = result
            calls.append({
                "call_id": call.call_id, "rank": call.rank,
                "source_node_ids": call.source_node_ids,
                "dependencies": call.dependencies,
                "kind": kernel.kind.value, "variant": kernel.variant,
                "shape": kernel.shape, "dtype": kernel.dtype,
                "attributes": kernel.attributes,
                "execution_support": kernel.execution_support,
                "transfer": None if kernel.transfer is None else asdict(kernel.transfer),
                "launch": None if launch is None else {
                    "algorithm": launch.algorithm, "grid": launch.grid,
                    "block": launch.block,
                    "execution_variant": launch.execution_variant,
                    "parameters": launch.parameters,
                },
                "resource_input": resource,
                "prediction": prediction,
            })
        logical_timeline = None
        if len(priced_predictions) == len(plan.calls):
            timeline = build_execution_timeline(plan, priced_predictions)
            logical_timeline = {
                "duration_s": timeline.duration_s,
                "predicted_device_time_s": timeline.predicted_device_time_s,
                "call_count": len(plan.calls),
                "scope": "logical_roofline_kernel_DAG_no_host_dispatch_or_runtime_acceptance",
            }
        complete_source_chain = source_runtime_inputs_by_phase is not None and (
            phase_runtime.target_sampling_source == "vllm_0_30_plain_greedy_bf16"
            or (phase_runtime.symbolic_forward_source == "vllm_0_27_1_deepseek_v3_sync_fp8"
                and phase_runtime.target_sampling_source == "vllm_0_27_1_plain_greedy_bf16")
            or (phase_runtime.symbolic_forward_source == "vllm_0_11_glm4_sync"
                and phase_runtime.target_sampling_source == "vllm_0_11_plain_greedy_bf16")
            or (mtp_mode == "native" and isinstance(phase_runtime.model_source_inputs, V32PaddedMTPInputs)
                and phase_runtime.mtp_draft_sampling_source == "vllm_0_30_legacy_full_logits_greedy"
                and phase_runtime.target_sampling_source == "vllm_0_30_rejection_raw_logits_bf16"))
        results.append({
            "phase": phase,
            "source_runtime_inputs": asdict(phase_runtime),
            "source_activation_memory": source_storage,
            "scheduled_requests": [
                {**asdict(request), "positions": tuple(range(
                    request.computed_tokens,
                    request.computed_tokens + request.scheduled_tokens))}
                for request in requests],
            "logical_nodes": len(graph.nodes), "kernel_calls": len(calls),
            "unpriced_calls": sum(
                call["prediction"]["status"] == "unpriced" for call in calls),
            # Logical pricing and native launch support are independent.
            # The old native-proposer projection still omits source work;
            # an explicitly complete virtual target does not become partial
            # merely because its mathematical kernels have no native launch.
            "roofline_end_to_end": "source_projection_only" if pipeline_parallel_size == 2 else "partial" if (
                not complete_source_chain) or any(
                call["prediction"]["status"] == "unpriced" for call in calls
            ) else "priced",
            "modeled_calls_pricing": "partial" if any(
                call["prediction"]["status"] == "unpriced" for call in calls
            ) else "priced",
            "logical_roofline_timeline": logical_timeline,
            "nodes": [{
                "node_id": node.node_id, "rank": node.rank,
                "kind": node.kind.value, "variant": node.variant,
                "shape": node.shape, "dtype": node.dtype,
                "attributes": node.attributes,
                "predecessors": graph.predecessors(node.node_id),
            } for node in graph.nodes],
            "calls": calls,
        })
    if source_graph_only:
        return _json_value({
            "snapshot": _snapshot(), "model_config_path": model_path,
            "probe_workload_path": "symbolic_forward",
            "source_normalization": source_normalization,
            "static_output_scope": static_output_scope,
            "source_runtime_inputs_by_phase": {phase: asdict(source_runtime_inputs_by_phase[phase]) for phase, _ in phase_requests},
            "model_id": model.model_id, "model_revision": model.model_revision,
            "model_execution_version": runtime_field("model_execution_version"),
            "v4_eager_metadata_source": runtime_field("v4_eager_metadata_source"),
            "v4_source_max_model_len": runtime_field("v4_source_max_model_len"),
            "v4_source_use_deep_gemm": runtime_field("v4_source_use_deep_gemm"),
            "fp8_runner_source_links": _fp8_runner_source_links(results),
            "compressor_state_source_links": _compressor_state_source_links(results),
            "source_phase_requests_supplied": source_phase_requests is not None,
            "v4_routing_source": runtime_field("v4_routing_source"),
            "deployment": asdict(parallel), "mtp": "off",
            "scope": "workload source provenance graph only; no lowering, launch selection or prediction",
            "lowering_executed": False, "prediction_executed": False,
            "actual_runtime_or_loaded_weight_binding": False,
            "normal_kv_capacity_accepted": False,
            "steps": results,
        })
    return _json_value({
        "snapshot": _snapshot(), "model_config_path": model_path,
        "probe_workload_path": "symbolic_forward",
        "source_normalization": source_normalization,
        "static_output_scope": static_output_scope,
        "source_runtime_inputs_by_phase": {phase: asdict(source_runtime_inputs_by_phase[phase]) for phase, _ in phase_requests},
        "model_id": model.model_id, "model_revision": model.model_revision,
        "hardware_id": "nvidia_h200", "method": "roofline",
        "mtp": "on" if mtp_mode == "native" else "off",
        "prefill_tokens": prefill_tokens,
        "v4_eager_metadata_source": runtime_field("v4_eager_metadata_source"),
        "v4_source_max_model_len": runtime_field("v4_source_max_model_len"),
        "v4_source_use_deep_gemm": runtime_field("v4_source_use_deep_gemm"),
        "fp8_runner_source_links": _fp8_runner_source_links(results),
        "compressor_state_source_links": _compressor_state_source_links(results),
        "source_phase_requests_supplied": source_phase_requests is not None,
        "v4_routing_source": runtime_field("v4_routing_source"),
        "v4_routing_requirements": {
            "scope": "conditional routing source projection; hash priced only with explicit modular CUDA scenario",
            "assumptions": "modular CUDA; contiguous FP32 logits/bias; normalized top6; INT32 output IDs; no VL/padding; hash scenario also assumes INT32 input IDs/table without conversion",
            "actual_provider_compiler_runtime_bound": False,
            "normal_kv_capacity_accepted": False,
        } if runtime_field("v4_routing_source") is not None else None,
        "model_execution_version": runtime_field("model_execution_version"),
        "mtp_draft_sampling_source": runtime_field("mtp_draft_sampling_source"),
        "target_sampling_source": runtime_field("target_sampling_source"),
        "embedding_token_ids_by_phase": {phase: source_runtime_inputs_by_phase[phase].embedding_token_ids for phase, _ in phase_requests},
        "embedding_token_input_scope": (source_normalization["token_input_scope"]
            if source_normalization is not None and "token_input_scope" in source_normalization else
            "explicit caller source inputs in scheduled row order; not observed model samples or runtime trace"),
        "target_sampling_requirements": ({
            "scope": "declared all-greedy source chain; not observed SamplingMetadata or sampler/model execution",
            "requires": "no logprobs/logprob-token-ids, active logits processors, penalties, masks or thinking-budget mutation",
            "stages": ("BF16_to_FP32_logits", "FP32_argmax_INT64_ID", "INT64_to_INT32_output"),
            "aliases": ("sampled.long()", "output.unsqueeze(-1)"),
            "excluded_after_greedy_return": ("temperature", "min_p", "top_k_top_p", "random_sampling"),
            "actual_provider_runtime_bound": False,
        } if runtime_field("target_sampling_source") in (
            "vllm_0_11_plain_greedy_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_27_1_plain_greedy_bf16") else None),
        "native_weight_source_requirements": ({
            "scope": "explicit legacy full-logits source projection, not observed proposer dispatch",
            "decoder_weights": "exact per-module FP8/BF16 rules at the appended MTP layer prefix",
            "merge_projection": "BF16 nn.Linear outside quantizer selection",
            "draft_head": "llm_base_proposer._maybe_share_lm_head aliases target lm_head, preserving its BF16 exclusion",
            "actual_loading_provider_and_runtime_bound": False,
        } if mtp_mode == "native" and model.config_sha256 in GLM_BLOCK_FP8_CONFIG_SHA256S else None),
        "pipeline_source_requirements": {
            "VLLM_PP_LAYER_PARTITION": pipeline_layer_partition,
            "pipeline_link_scenario": pipeline_link,
            "physical_topology_observed": False,
            "normal_kv_capacity_accepted": False,
            "scheduler_runtime_accepted": False,
            "transport_semantics_accepted": False if pipeline_parallel_size == 2 else None,
            "modeled_boundary_reduction_order": (
                "receiver_stage_before_input_norm" if pipeline_parallel_size == 2 else None),
            "modeled_tp_partial_producer_scope": (
                "All modeled rank-local source math materialized on every TP rank, "
                "including replicated indexer/projections and TP-sharded attention/MLP/head; "
                "actual compiler/runtime/transport bindings remain incomplete"
                if pipeline_parallel_size == 2 else
                "The same complete virtual target program runs on each physical rank; "
                "logical pricing is recorded per call, independently of native launch support; "
                "actual compiler/runtime/provider and normal allocator remain unbound"),
            "kimi_ep_source_predicates": (
                {"dp_size": 1, "pcp_size": 1, "sp_size": 1,
                 "use_sequence_parallel_moe": False,
                 "use_all2all_kernels": False, "do_naive_dispatch_combine": False,
                 "source_symbol": "FusedMoEParallelConfig.use_all2all_kernels / MoERunner.do_naive_dispatch_combine",
                 "actual_backend_selection": None,
                 "scope": "explicit_non_sequence_parallel_source_scenario_not_observed_runtime"}
                if model.model_type == "kimi_k2" and tensor_parallel_size == 8
                and enable_expert_parallel else None),
            "v4_ffn_source_predicates": (
                {"use_mega_moe": False, "use_sequence_parallel": False,
                 "enable_expert_parallel": False, "ep_rank": 0,
                 "factory_expert_ownership": "all_experts_per_tensor_rank",
                 "factory_intermediate_per_rank": model.moe_intermediate_size // tensor_parallel_size,
                 "source_symbol": "DeepseekV4MoE._init_fused_moe_experts / FusedMoEParallelConfig.make",
                 "actual_backend_selection": None,
                 "parent_metadata_and_loaded_weight_binding_validated": False,
                 "scope": "non_MegaMoE_EP_off_source_branch_not_observed_runtime"}
                if model.model_type == "deepseek_v4" and tensor_parallel_size > 1
                and not enable_expert_parallel else None),
            "transport_semantics_gap": (
                "V32 leaves MLP hidden_states TP-partial (reduce_results=False); "
                "GPUWorker default TP shard/send/all-gather requires replicated inputs. "
                "No actual override or compiler transformation preserving partials is bound. "
                "The two payloads and wire bytes describe the source-default branch only; "
                "its priced calls do not prove a numerically valid PP model."
                if pipeline_parallel_size == 2 else None),
        },
        "pipeline_weight_projection": pp_weight_projection,
        "scope": "static modeled evidence; requires independent comparison",
        "on_mode_status": ("priced_declared_logical_source_chain" if source_runtime_inputs_by_phase is not None
            and all(step["roofline_end_to_end"] == "priced" for step in results) else "partial_source_projection")
            if mtp_mode == "native" else "not_executed_by_this_probe",
        "draft_architecture": "DeepseekV32MTPModel" if mtp_mode == "native" else None,
        "native_source_gaps": (
            ("exact_0_30_compiler_launch_binding_unavailable",
             "target_fusion_projection_actual_launch_and_slot_mapping_unvalidated",
             "independent_declared_draft_inputs_not_checkpoint_samples_or_runtime_buffer_observations",
             "actual_slot_mapping_and_normal_capacity_not_validated",
             "draft_backend_independent_autoselection_and_group_metadata_build_unbound",
             "runtime_draft_sampler_dispatch_not_bound") if mtp_mode == "native" else ()),
        "routing": "explicit_per_phase_runtime_inputs_not_observed",
        "deployment": asdict(parallel),
        "steps": results,
    })


def _glm_official_fp8_source_inputs(tp, mode):
    """Explicit synthetic comparison inputs; never model-generated outputs.

    Every phase is standalone, including empty workspace history. Padded
    first-pass stale/anchor IDs and recurrent argmax IDs are external scenario
    values; only the source's known non-anchor shift is preserved here.
    """
    routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
    runtimes = {}
    for phase, requests in _operator_probe_phase_requests(8, mode):
        rows = sum(request.scheduled_tokens for request in requests)
        target_ids = tuple(range(1, rows + 1))
        state = None
        if mode == "native":
            state = _declared_glm_padded_probe_state(ScheduleStep(phase, "main", requests), target_ids,
                source_id="official-family-independent-padded-input-scenario")
        runtimes[phase] = RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            symbolic_forward_source="vllm_0_30_v32_fp8_sync_explicit_allreduce",
            target_sampling_source="vllm_0_30_rejection_raw_logits_bf16" if mode == "native" else "vllm_0_30_plain_greedy_bf16",
            mtp_draft_sampling_source="vllm_0_30_legacy_full_logits_greedy" if mode == "native" else None,
            symbolic_source_max_model_len=128, symbolic_source_max_batched_tokens=128,
            symbolic_source_max_requests=2 if mode == "native" else None,
            symbolic_workspace_initial_capacity_bytes=(0,) * tp,
            glm_indexer_selection_source="declared_causal_topk",
            glm_prefill_attention_source="force_sparse_mqa" if mode == "native" else None,
            embedding_token_ids=target_ids, model_source_inputs=state)
    return runtimes


def _glm_official_fp8_probe(arguments):
    path, tp, mode = arguments
    return build_model_operator_probe(path, tp, False, mtp_mode=mode,
        source_runtime_inputs_by_phase=_glm_official_fp8_source_inputs(tp, mode))


def write_glm_official_fp8_family_comparison(output, *, tensor_parallel_size, mtp_mode):
    """Compare the official pair under identical declared source workloads."""
    identity_keys = frozenset((
        "model_id", "model_revision", "config_sha256", "source_config_sha256",
        "canonical_config_sha256", "extracted_config_sha256", "model_source_url", "model_source_revision"))

    def normalized(value):
        if isinstance(value, Mapping):
            return {key: normalized(item) for key, item in value.items() if key not in identity_keys}
        if isinstance(value, (tuple, list)):
            return [normalized(item) for item in value]
        return value

    paths = ("configs/model/glm-5.2-fp8.json", "configs/model/glm-5.3.json")
    models = tuple(load_model_config(path) for path in paths)
    with ProcessPoolExecutor(max_workers=2) as executor:
        probes = tuple(executor.map(_glm_official_fp8_probe,
                                    ((path, tensor_parallel_size, mtp_mode) for path in paths)))
    phases = []
    for first, second in zip(probes[0]["steps"], probes[1]["steps"], strict=True):
        phases.append({
            "phase": first["phase"], "logical_nodes": first["logical_nodes"], "kernel_calls": first["kernel_calls"],
            "unpriced_calls": first["unpriced_calls"], "roofline_end_to_end": first["roofline_end_to_end"],
            "graph_equal": normalized(first["nodes"]) == normalized(second["nodes"]),
            "lowered_calls_and_predictions_equal": normalized(first["calls"]) == normalized(second["calls"]),
            "scheduled_inputs_equal": first["scheduled_requests"] == second["scheduled_requests"],
            "unpriced_counts_equal": first["unpriced_calls"] == second["unpriced_calls"],
        })
    parallel = ParallelConfig(tensor_parallel_size=tensor_parallel_size)
    capacity = []
    for model in models:
        terms = {
            "loader_with_mla": glm_block_fp8_loader_payload_bytes(model, parallel, include_mla_absorption=True),
            "postload_with_mla": glm_block_fp8_postload_parameter_bytes(
                model, parallel, hardware_id="nvidia_h200", linear_backend="deep_gemm", routed_backend="deep_gemm",
                include_mla_absorption=True),
            "constructor_buffers": glm_block_fp8_constructor_buffer_bytes(model, parallel, max_batched_tokens=128, max_requests=2),
            "shared_workspace": glm_block_fp8_deepgemm_workspace_bytes(model, parallel, hardware_id="nvidia_h200", tokens=128),
            "kv_bytes_per_token_per_rank": model.kv_bytes_per_token_per_parallel_rank(tensor_parallel_size, 1),
        }
        capacity.append(terms)
    checks = {
        "separate_checkpoint_identities": models[0].model_id != models[1].model_id and models[0].config_sha256 != models[1].config_sha256,
        "shared_assumption_semantics": models[0].expert_routing_assumption_semantics_sha256 == models[1].expert_routing_assumption_semantics_sha256,
        "source_capacity_terms_equal": capacity[0] == capacity[1],
        "all_graphs_equal": all(phase["graph_equal"] for phase in phases),
        "all_lowering_and_prediction_records_equal": all(phase["lowered_calls_and_predictions_equal"] for phase in phases),
        "all_scheduled_inputs_equal": all(phase["scheduled_inputs_equal"] for phase in phases),
        "all_unpriced_counts_equal": all(phase["unpriced_counts_equal"] for phase in phases),
    }
    result = {
        "snapshot": _snapshot(), "status": "source_comparison_only_not_acceptance", "checks": checks,
        "failed": sum(not value for value in checks.values()), "tensor_parallel_size": tensor_parallel_size, "mtp_mode": mtp_mode,
        "hardware_id": "nvidia_h200", "source_execution": "vllm_0_30_0_eager_PP1_SP1_EP_off",
        "probe_workload_paths": [probe["probe_workload_path"] for probe in probes],
        "source_workload_assumptions": {
            "profile": "standalone_shared_virtual_target_frames",
            "max_model_len": 128, "max_batched_tokens": 128,
            "workspace_entry_bytes_per_rank": [0] * tensor_parallel_size,
            "workspace_scope": "declared fresh workspace per independent phase, not observed history or normal capacity",
            "token_ids": "explicit synthetic sequential IDs per phase, not checkpoint outputs",
            "indexer": "declared causal top-k cardinality",
            "sampler": "vllm_0_30_rejection_raw_logits_bf16" if mtp_mode == "native" else "vllm_0_30_plain_greedy_bf16",
            "padded_draft_inputs": "independent declared synthetic/stale IDs; no numerical draft/sampler execution" if mtp_mode == "native" else None,
            "prefill_attention_source": "force_sparse_mqa" if mtp_mode == "native" else None},
        "source_capacity_inventory_mtp_mode": "off",
        "identities": [{key: getattr(model, key) for key in ("model_id", "model_revision", "config_sha256", "model_source_url")} for model in models],
        "phase_comparisons": phases,
        "source_capacity_terms": [{"model_id": model.model_id, "terms": terms} for model, terms in zip(models, capacity, strict=True)],
        "runtime_acceptance": False, "normal_capacity_accepted": False,
        "limitations": [
            "same explicitly declared routing assumption; real replay/profile retains checkpoint binding",
            "capacity terms cover the same target-off source inventory/profile, not complete native or allocator-ready residency",
            "target-off logical pricing does not establish native-MTP migration, actual provider/runtime or normal capacity",
            "old derived-local per-tensor GLM5.2-FP8 artifacts are excluded from official acceptance",
            "main agent owns final acceptance; no old snapshot receipt inheritance"],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_value(result), ensure_ascii=False, indent=2) + "\n")
    return result


def require_logical_roofline_forward(payload):
    """Require a fully priced four-phase forward, never native acceptance."""
    if "static_output_scope" in payload and payload["static_output_scope"] == "lm_head_only_no_sampler":
        raise ValueError("static LM-head-only probe omits the sampler and cannot establish a complete logical request forward")
    if "static_output_scope" in payload and payload["static_output_scope"] == "fp32_logits_only":
        raise ValueError("static FP32 logits cast omits sampling and cannot establish a complete logical request forward")
    if ("source_runtime_inputs_by_phase" in payload and payload["source_runtime_inputs_by_phase"] is not None
            and any(runtime["target_sampling_source"] in (None, "vllm_0_11_plain_sampler_bf16", "vllm_0_30_plain_sampler_bf16")
                for runtime in payload["source_runtime_inputs_by_phase"].values())):
        raise ValueError("static logits-only source omits the sampler and cannot establish a complete logical request forward")
    if payload["mtp"] not in ("off", "on"):
        raise ValueError("complete logical forward check requires target-off or an explicit complete GLM MTP source")
    if payload["mtp"] == "on":
        if ("probe_workload_path" not in payload or payload["probe_workload_path"] != "symbolic_forward"
                or "source_runtime_inputs_by_phase" not in payload or payload["source_runtime_inputs_by_phase"] is None):
            raise ValueError("complete GLM MTP logical forward requires explicit per-phase common source inputs")
        for step in payload["steps"]:
            runtime = payload["source_runtime_inputs_by_phase"][step["phase"]]
            if (runtime["symbolic_forward_source"] not in (
                    "vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
                    or runtime["mtp_draft_sampling_source"] != "vllm_0_30_legacy_full_logits_greedy"
                    or runtime["target_sampling_source"] != "vllm_0_30_rejection_raw_logits_bf16"
                    or runtime["model_source_inputs"] is None or step["roofline_end_to_end"] != "priced"):
                raise ValueError("complete GLM MTP logical forward requires its paired typed target/proposer source chain")
    if tuple(step["phase"] for step in payload["steps"]) != ("prefill", "decode1", "decode2", "mixed"):
        raise ValueError("complete logical forward requires all four phases")
    for step in payload["steps"]:
        calls = step["calls"]
        if (type(step["kernel_calls"]) is not int or step["kernel_calls"] != len(calls)
                or not calls or len({call["call_id"] for call in calls}) != len(calls)):
            raise ValueError("complete logical forward call coverage is inconsistent")
        if step["roofline_end_to_end"] == "source_projection_only":
            raise ValueError("PP source projection cannot establish a complete logical forward")
        if (type(step["unpriced_calls"]) is not int or step["unpriced_calls"] != 0
                or step["modeled_calls_pricing"] != "priced"):
            raise ValueError("complete logical forward still has unpriced calls")
        for call in calls:
            prediction = call["prediction"]
            if prediction["method"] != "roofline" or prediction["status"] not in ("priced", "logical_alias"):
                raise ValueError("complete logical forward requires every call to have a Roofline price")
            seconds = prediction["seconds"]
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not isfinite(seconds) or seconds < 0):
                raise ValueError("complete logical forward has an invalid call duration")
        timeline = step["logical_roofline_timeline"]
        if timeline is None:
            raise ValueError("complete logical forward lacks a complete positive DAG timeline")
        duration = timeline["duration_s"]
        if (type(timeline["call_count"]) is not int or timeline["call_count"] != len(calls)
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not isfinite(duration) or duration <= 0):
            raise ValueError("complete logical forward lacks a complete positive DAG timeline")


def logical_roofline_summary(payload):
    """Compact user result from a completely priced logical execution DAG.

    Device durations come from the production timeline. Rank-local kernel
    durations are never summed to produce a model latency. This report does
    not upgrade source-only inputs to Scheduler, capacity or GPU acceptance.
    """
    require_logical_roofline_forward(payload)
    result = {field: payload[field] for field in (
        "snapshot", "model_config_path", "model_id", "model_revision", "hardware_id", "method",
        "mtp", "prefill_tokens", "deployment", "model_execution_version", "routing",
        "target_sampling_source", "target_sampling_requirements", "embedding_token_ids_by_phase",
        "embedding_token_input_scope", "v4_eager_metadata_source", "v4_source_max_model_len",
        "v4_source_use_deep_gemm", "v4_routing_source", "v4_routing_requirements")}
    if "source_runtime_inputs_by_phase" in payload:
        result["source_runtime_inputs_by_phase"] = payload["source_runtime_inputs_by_phase"]
        result["probe_workload_path"] = payload["probe_workload_path"]
    if "source_normalization" in payload:
        result["source_normalization"] = payload["source_normalization"]
    if "static_output_scope" in payload:
        result["static_output_scope"] = payload["static_output_scope"]
    result["scope"] = "static logical Roofline; declared inputs and routing, not model generation or Serving"
    result["steps"] = []
    for step in payload["steps"]:
        timeline = step["logical_roofline_timeline"]
        result["steps"].append({
            **{field: step[field] for field in ("phase", "scheduled_requests", "logical_nodes", "kernel_calls",
                "unpriced_calls", "roofline_end_to_end", "modeled_calls_pricing")},
            "logical_roofline_timeline": timeline,
            "latency_ms": timeline["duration_s"] * 1000,
            "native_unsupported_calls": sum(call["execution_support"] == "unsupported" for call in step["calls"]),
        })
    result["limitations"] = [
        "FLOPs/bytes baseline; instruction/control, compiler, cache and physical traffic costs are not fully modeled.",
        "Source/operator alignment remains incomplete; fully priced calls do not establish complete source coverage.",
        "No real Scheduler, normal KV capacity, QPS16, model-generated tokens or hardware accuracy accepted.",
        "Phase durations are individual model steps; mixed is a separate batch and must not be added to single-request latency.",
    ]
    return result


def write_model_operator_probe(
    model_path: str,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
    output: Path,
    *,
    mtp_mode: str = "off",
    prefill_tokens: int = 8,
    v4_eager_metadata: bool = False,
    v4_routing_source: str | None = None,
    pipeline_parallel_size: int = 1,
    pipeline_link: str | None = None,
    pipeline_layer_partition: str | None = None,
    source_graph_only: bool = False,
    mtp_draft_sampling_source: str | None = None,
    target_sampling_source: str | None = None,
    embedding_token_ids_by_phase: Mapping[str, tuple[int, ...]] | None = None,
    v4_source_max_model_len: int | None = None,
    v4_source_max_batched_tokens: int | None = None,
    v4_indexer_selection_source: str | None = None,
    v4_source_use_deep_gemm: bool | None = None,
    summary_only: bool = False,
    source_runtime_inputs_by_phase: Mapping[str, RuntimeInputs] | None = None,
    source_normalization: Mapping[str, object] | None = None,
) -> None:
    if type(summary_only) is not bool:
        raise TypeError("summary_only must be boolean")
    if source_normalization is not None:
        if not isinstance(source_normalization, Mapping) or "target_sampling_source" not in source_normalization:
            raise ValueError("source normalization requires the versioned sampler derivation")
        source, expected = normalize_model_operator_probe_source(model_path, source_normalization["target_sampling_source"])
        if (expected is None or dict(source_normalization) != expected
                or not isinstance(source_runtime_inputs_by_phase, Mapping) or not source_runtime_inputs_by_phase
                or any(not isinstance(runtime, RuntimeInputs) or runtime.symbolic_forward_source != source
                    or runtime.target_sampling_source != source_normalization["target_sampling_source"]
                    for runtime in source_runtime_inputs_by_phase.values())):
            raise ValueError("source normalization must match every declared phase and cannot promote provider acceptance")
    payload = build_model_operator_probe(
        model_path, tensor_parallel_size, enable_expert_parallel,
        mtp_mode=mtp_mode, prefill_tokens=prefill_tokens,
        v4_eager_metadata=v4_eager_metadata, v4_routing_source=v4_routing_source,
        v4_source_max_model_len=v4_source_max_model_len,
        v4_source_max_batched_tokens=v4_source_max_batched_tokens,
        v4_indexer_selection_source=v4_indexer_selection_source,
        v4_source_use_deep_gemm=v4_source_use_deep_gemm,
        pipeline_parallel_size=pipeline_parallel_size,
        pipeline_link=pipeline_link, pipeline_layer_partition=pipeline_layer_partition,
        source_graph_only=source_graph_only,
        mtp_draft_sampling_source=mtp_draft_sampling_source,
        target_sampling_source=target_sampling_source,
        embedding_token_ids_by_phase=embedding_token_ids_by_phase,
        source_runtime_inputs_by_phase=source_runtime_inputs_by_phase)
    if source_normalization is not None:
        if payload["source_normalization"] is None:
            payload["source_normalization"] = dict(source_normalization)
        else:
            payload["source_normalization"]["cli_source_derivation"] = dict(source_normalization)
    if summary_only:
        payload = logical_roofline_summary(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_glm_gpu_v2_storage_probe(model_path, tensor_parallel_size, output, *, prefill_context_parallel_size):
    """Export the production owner's unpriced source roles for an N3 case.

    The explicit acceptance array is the three-request source conservation
    scenario, not a measured acceptance probability. No model forward,
    launch selector, Scheduler, allocation or prediction executes here.
    """
    model = load_model_config(model_path)
    if tensor_parallel_size not in (1, 8) or type(tensor_parallel_size) is not int:
        raise NotImplementedError("GLM storage source probe covers TP1/TP8 only")
    requests = tuple(ScheduledRequest(
        f"request-{index}", "decode", 0, 4, 64,
        raw_model_output_tokens=accepted + 1,
        mtp_proposed_tokens=3, mtp_verified_tokens=3,
        mtp_accepted_tokens=accepted, mtp_proposer_tokens=3,
        mtp_next_proposed_tokens=3, mtp_proposer_mode="padded",
        mtp_acceptance_source_kind="explicit_assumption",
        mtp_acceptance_source_id="three_request_source_conservation_array")
        for index, accepted in enumerate((0, 2, 3)))
    runtime = RuntimeInputs(
        ExpertRoutingInput.seeded_assumption("unused_storage_only_routing", 0),
        "mla", "uniform", model_execution_version="vllm_0_30_0",
        mtp_proposer_execution_source="vllm_0_30_gpu_v2")
    work = glm_gpu_v2_mtp_storage_work(model, requests, runtime,
        prefill_context_parallel_size=prefill_context_parallel_size)
    # Separate explicit state cases, not metadata observed for the above
    # scheduled requests. Production owner derives all states/role bytes.
    prepare_scenario = {
        "target_seq_lens": (13, 16, 8), "num_rejected": (2, 0, 3),
        "positions": (10, 15, 4), "sample_src_positions": (11, 41, 90),
        "max_num_requests": 8, "max_model_len": 16, "proposal_depth": 3,
        "num_rejected_dtype": "int32",
    }
    prepare_cases = []
    for advance in (True, False):
        scenario = {**prepare_scenario, "advance_draft_positions": advance}
        prepare_cases.append({"input_scenario": scenario,
                              "work": glm_gpu_v2_prepare_decode_work(**scenario)})
    slot_scenario = {
        "idx_mapping": (0, 1, 2), "query_start_loc": (0, 1, 2, 3),
        "positions": (11, 15, 5),
        "block_tables_by_group": (
            ((10, 11, 12, 13), (20, 21, 22, 23), (30, 31, 32, 33)),
            ((40, 41, 42, 43), (50, 51, 52, 53), (60, 61, 62, 63))),
        "kv_block_sizes": (16, 16), "kernel_block_sizes": (4, 8),
        "mapping_enabled": (True, False), "slot_capacity": 16,
        "returned_tokens": 4, "cp_size": 1,
    }
    payload = {
        "snapshot": _snapshot(), "model_config_path": model_path,
        "model_type": model.model_type, "tensor_parallel_size": tensor_parallel_size,
        "prefill_context_parallel_size": prefill_context_parallel_size,
        "status": "source_storage_subset_not_full_native_or_priced_model",
        "independent_acceptance": False, "new_gpu_execution": False,
        "normal_gpu_v2_workload_supported": False,
        "scheduled_requests": [asdict(request) for request in requests],
        "logical_work": work,
        "prepare_decode_source_scenarios": prepare_cases,
        "slot_mapping_source_scenario": {
            "input_scenario": slot_scenario,
            "work": glm_gpu_v2_slot_mapping_work(**slot_scenario)},
        "metadata_refresh_source_scenarios": tuple(glm_gpu_v2_metadata_refresh_plan(
            proposal_depth=work["proposal_depth"], metadata_update_supported_by_groups=capabilities)
            for capabilities in ((True, True), (True, False))),
        "sparse_backend_metadata_source_constraints": tuple(glm_gpu_v2_sparse_backend_metadata_plan(
            proposal_depth=work["proposal_depth"], backend_name=backend)
            for backend in ("FLASH_ATTN_MLA_SPARSE", "FLASHMLA_SPARSE", "FLASHINFER_MLA_SPARSE_SM90")),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _rescore_formal_layer_shapes(comparison):
    """Replay external shape obligations on current facts; create no expectations."""
    records = []
    required = passed = failed = 0
    for record in comparison["records"]:
        if not record["checks"]:
            records.append({"model": record["model"], "status": record["status"],
                            "reason": record["reason"]})
            continue
        probe = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        phase_calls = {step["phase"]: step["calls"] for step in probe["steps"]}
        checks = []
        for obligation in record["checks"]:
            call_id = (f"{obligation['phase']}:layer-{obligation['layer']}"
                       f"{obligation['operation']}")
            actual = [call for call in phase_calls[obligation["phase"]]
                      if call["rank"] == record["rank"] and call["kind"] == "GEMM" and (
                          call["call_id"] == call_id
                          or call["call_id"].startswith(call_id + ":"))]
            expected = obligation["expected_shape"]
            ok = len(actual) == 1 and set(expected) <= set(actual[0]["shape"])
            if ok:
                ok = all(actual[0]["shape"][key] == value
                         for key, value in expected.items())
            checks.append({"call_id": call_id, "pass": ok})
        count = sum(check["pass"] for check in checks)
        records.append({"model": record["model"], "artifact": record["artifact"],
                        "snapshot": probe["snapshot"], "rank": record["rank"],
                        "passed": count, "failed": len(checks) - count,
                        "checks": checks})
        required += len(checks)
        passed += count
        failed += len(checks) - count
    if required != comparison["totals"]["required_shape_checks"]:
        raise ValueError("formal layer shape denominator changed during replay")
    return {"totals": {"required_shape_checks": required, "passed": passed, "failed": failed},
            "records": records, "independent_acceptance": False,
            "scope": "replay of existing external rank0 shape expectations; no new expectation, complete launch or runtime acceptance"}


def _rescore_v4_compressor_bytes(comparison):
    """Compare current facts to the external source object's byte lower bound."""
    records = []
    passed = failed = 0
    for record in comparison["records"]:
        probe = json.loads(Path(record["artifact"]).read_text(encoding="utf-8"))
        calls = {call["call_id"]: call for step in probe["steps"] for call in step["calls"]}
        checks = []
        for obligation in record["checks"]:
            call_id = (f"{obligation['phase']}:layer-{obligation['layer']}:"
                       f"{obligation['operation']}")
            call = calls[call_id]
            if obligation["operation"] in ("compressor-kv-score", "indexer-compressor-kv-score"):
                ok = (call["dtype"] == "bf16" and call["shape"]["dtype_bytes"] == 2
                      and call["attributes"]["output_dtype"] == "fp32"
                      and call["resource_input"]["logical_bytes"] >=
                      obligation["expected_unique_tensor_bytes"]["minimum_sum"])
            elif obligation["operation"] in ("compressor-state-update", "indexer-compressor-state-update"):
                expected_bytes = sum(obligation["expected_source_role_bytes_all_valid_projection"].values())
                arithmetic = obligation["expected_arithmetic"]
                demand = call["resource_input"]["demand"]
                ok = (call["dtype"] == "fp32" and call["shape"]["dtype_bytes"] == 4
                      and call["resource_input"]["logical_bytes"] == expected_bytes
                      and demand["compute_fma"] == arithmetic["compute_fma"]
                      and demand["compute_alu"] == arithmetic["score_plus_APE_additions"] + arithmetic["position_modulo"])
            else:
                raise ValueError("unknown formal V4 compressor operation")
            checks.append({"call_id": call_id, "pass": ok})
        count = sum(check["pass"] for check in checks)
        passed += count
        failed += len(checks) - count
        records.append({"model": record["model"], "snapshot": probe["snapshot"],
                        "passed": count, "failed": len(checks) - count, "checks": checks})
    if passed + failed != comparison["totals"]["checks"]:
        raise ValueError("formal V4 compressor denominator changed during replay")
    return {"records": records, "totals": {"checks": passed + failed, "passed": passed, "failed": failed},
            "independent_acceptance": False,
            "scope": "existing external source byte lower bound and dtype assertions only; actual launch, local services and full W3 remain unvalidated"}


def write_memory_ownership_handoff(output: Path) -> None:
    """Record the requested clean checkpoint release; no Goal gate is closed."""
    snapshot = _snapshot()
    if snapshot["uncommitted_changes"]:
        raise ValueError("memory ownership release requires a clean committed checkpoint")
    released = ("src/kernel/predictors/", "src/kernel/memory_access_builder.py",
                "tests/kernel/test_memory_access.py")
    record = {
        "status": "released_by_operator_support",
        "snapshot": snapshot,
        "dirty_files": [],
        "released_paths": [{"path": path, "writes_stopped": True,
                            "pending_patch": False} for path in released],
        "ongoing_write_tasks": [],
        "receiving_worktree": "../midlayer-wt-memory-unification-20261014",
        "new_writer_authorization": "main agent confirms release before enabling writes",
        "unfinished_interface_requests": [],
        "compatibility_requirements": [
            "Preserve BF16 A/B plus FP32 C for legal plain mixed-output Roofline; "
            "packed FP8 launch storage cannot satisfy that declaration.",
            "Existing production memory export explicitly rejects heterogeneous C, "
            "quantized/unknown launches and unsupported specialized epilogues. "
            "Do not export BF16 C for FP32 output; homogeneous defaults remain supported.",
            "V4 finish FP32 state reads, projection/save-partial roles, model clamps "
            "and Marlin weighted output are source mathematical contracts. "
            "They do not establish physical memory transactions, aliasing or runtime binding.",
            "Retain explicit unsupported/unpriced boundaries where storage objects, "
            "addresses, compiler or actual library launches cannot be proven.",
        ],
        "continuing_owner_scope": ["src/serving/", "src/kernel/kernel_spec.py",
                                   "src/kernel/launch_config/", "corresponding model evaluation/tests"],
        "future_predictor_requests": "send to main agent for a serial ownership window; "
                                     "operator-support will not reenter released paths",
        "goal_complete": False,
        "goal_status": "active_checkpoint",
        "goal_writes_stopped": False,
        "failed_gates": [f"W{i}" for i in range(1, 8)],
        "validation_scope": "ownership-only checkpoint; reuse original snapshot-bound results; "
                            "no full tests, GPU, merge, rebase or push",
        "regenerate_command": "PYTHONDONTWRITEBYTECODE=1 "
            "/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python "
            "scripts/evaluation/serving/report_operator_support_checkpoint.py "
            f"--output {shlex.quote(str(output))} --memory-ownership-handoff-only",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "memory_ownership_handoff.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _validate_grouped_topk_finite_precision_receipt(document):
    """Validate source witness fields before publishing any checkpoint files."""
    message = "formal source review finite-precision receipt"
    for key in ("accepted_scope", "source_reference_qualification"):
        if key not in document or not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f"{message} requires nonempty {key}")
    for key in ("examples", "findings", "limits"):
        if key not in document or not isinstance(document[key], list) or not document[key]:
            raise ValueError(f"{message} requires nonempty {key} list")
    if any(not isinstance(value, str) or not value.strip() for value in document["limits"]):
        raise ValueError(f"{message} limits must be nonempty strings")
    for finding in document["findings"]:
        if not isinstance(finding, dict) or any(
                key not in finding or not isinstance(finding[key], str) or not finding[key].strip()
                for key in ("item", "required_action")):
            raise ValueError(f"{message} findings require item and required_action strings")
    for example in document["examples"]:
        if not isinstance(example, dict):
            raise ValueError(f"{message} examples must be objects")
        for key in ("experts", "topk"):
            if key not in example or type(example[key]) is not int or example[key] <= 0:
                raise ValueError(f"{message} example requires positive {key}")
        if example["topk"] > example["experts"]:
            raise ValueError(f"{message} topk cannot exceed experts")
        for key in ("logits_all_zero", "same_value"):
            if key not in example or type(example[key]) is not bool:
                raise ValueError(f"{message} example requires boolean {key}")
        for key in ("correction_bias_all", "selection_score_fp32", "source_block_unbiased",
                    "source_warp_recovered_unbiased", "block_weight_each_fp32", "warp_weight_each_fp32"):
            if key not in example or type(example[key]) not in (int, float) or not isfinite(example[key]):
                raise ValueError(f"{message} example requires finite numeric {key}")
        for key in ("condition", "dispatch_transition"):
            if key not in example or not isinstance(example[key], str) or not example[key].strip():
                raise ValueError(f"{message} example requires nonempty {key}")
        if example["same_value"] is not (example["block_weight_each_fp32"] == example["warp_weight_each_fp32"]):
            raise ValueError(f"{message} same_value contradicts stored weights")
    if not any(example["same_value"] is False and
               example["block_weight_each_fp32"] != example["warp_weight_each_fp32"]
               for example in document["examples"]):
        raise ValueError(f"{message} requires a differing-output source witness")


_MHC_SM90_REVIEW_FILENAMES = (
    "mhc_prenorm_sm90_review.json", "mhc_prenorm_sm90_host_binary_review.json",
    "mhc_prenorm_sm90_descriptor_review.json", "mhc_prenorm_sm90_model_delta_review.json",
    "mhc_prenorm_sm90_tma_review.json", "mhc_prenorm_sm90_tma_model_delta_review.json",
    "mhc_prenorm_sm90_zero_stage_review.json", "mhc_prenorm_sm90_zero_stage_model_delta_review.json",
    "operator_sm90_formal_publication_review.json")


def _validate_mhc_sm90_review(filename, document):
    """Validate local-only publication boundaries, never replay its formulas."""
    if document["no_GPU_or_download_or_push"] is not True:
        raise ValueError("formal SM90 source receipt requires its offline boundary")
    for key in ("actual_provider_bound", "pricing_or_launch_assignment_accepted"):
        if key in document and document[key] is not False:
            raise ValueError("formal SM90 source receipt cannot promote local provider/pricing")
    if filename in ("mhc_prenorm_sm90_review.json", "mhc_prenorm_sm90_tma_review.json",
                    "mhc_prenorm_sm90_zero_stage_review.json"):
        snapshots = (document["candidate_start"], document["candidate_end"])
        scope = document["accepted_scope"]
        direct_key = ("direct_scopes_passed" if filename == "mhc_prenorm_sm90_zero_stage_review.json"
                      else "own_direct_scopes_passed")
        counts = (document[direct_key], document["candidate_failures"],
                  document["new_counter_categories"])
        limits = document["limits"]
    elif filename in ("mhc_prenorm_sm90_host_binary_review.json",
                      "mhc_prenorm_sm90_descriptor_review.json"):
        host = filename == "mhc_prenorm_sm90_host_binary_review.json"
        snapshots = (document["publication_snapshot"] if host else
                     document["observed_publication_head"],)
        scope = [document["source_basis"] if host else document["accepted_basis"]]
        counts = (document["new_counter_categories"],)
        limits = document["not_accepted"] if host else document["limits"]
        if not isinstance(document["conditionally_accepted"], dict) or not document["conditionally_accepted"]:
            raise ValueError("formal SM90 source receipt requires conditional facts")
        if (document["actual_provider_bound"] is not False or
                document["pricing_or_launch_assignment_accepted"] is not False):
            raise ValueError("formal SM90 source receipt cannot promote local provider/pricing")
    elif filename in ("mhc_prenorm_sm90_model_delta_review.json",
                      "mhc_prenorm_sm90_tma_model_delta_review.json",
                      "mhc_prenorm_sm90_zero_stage_model_delta_review.json"):
        snapshots = tuple(document[key]["commit"] for key in ("before_snapshot", "candidate_snapshot"))
        if filename == "mhc_prenorm_sm90_model_delta_review.json":
            count_key, calls_key = "conditional_metadata_checks", "conditional_calls"
            scope = [document["coverage_change"]]
        elif filename == "mhc_prenorm_sm90_tma_model_delta_review.json":
            count_key, calls_key = "metadata_checks", "conditional_source_calls"
            scope = document["limits"]
        else:
            count_key = calls_key = "nonempty_partition_checks"
            scope = [document["scope"]]
        counts = (document[count_key],)
        limits = document["limits"]
        records = document["records"]
        if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
            raise ValueError("formal SM90 source receipt requires original phase records")
        if any(not isinstance(item[calls_key], list) for item in records):
            raise ValueError("formal SM90 source receipt requires original conditional calls")
        if document[count_key] != sum(len(item[calls_key]) for item in records):
            raise ValueError("formal SM90 source receipt conditional count disagrees")
        if filename == "mhc_prenorm_sm90_model_delta_review.json" and any(
                item["runtime_or_provider_accepted"] is not False for item in records):
            raise ValueError("formal SM90 source receipt cannot promote phase runtime/provider")
        for record in records:
            for call in record[calls_key]:
                if filename == "mhc_prenorm_sm90_tma_model_delta_review.json":
                    metadata = call["expected_source_copy_metadata"]
                    if (metadata["actual_instruction_count_bound"] is not False or
                            metadata["dram_bytes_bound"] is not False):
                        raise ValueError("formal SM90 source receipt cannot promote source calls to traffic/instructions")
                elif filename == "mhc_prenorm_sm90_zero_stage_model_delta_review.json":
                    partition = call["K_stage_partition"]
                    if not isinstance(partition, list) or not partition or any(
                            type(value) is not int or value <= 0 for value in partition):
                        raise ValueError("formal SM90 source receipt requires nonempty K partitions")
        if type(document["all_other_artifact_content_equal"]) is not bool:
            raise ValueError("formal SM90 source receipt requires original equality flag")
    elif filename == "operator_sm90_formal_publication_review.json":
        snapshots = (document["publication_snapshot"]["commit"],)
        scope = [document["accepted_scope"]]
        counts = (document["receipts_checked"], document["lost_fields_or_relabels"])
        limits = document["limits"]
        records = document["records"]
        if not isinstance(records, list) or document["receipts_checked"] != len(records):
            raise ValueError("formal SM90 source receipt publication count disagrees")
        if any(type(record["full_original_receipt_equal"]) is not bool for record in records):
            raise ValueError("formal SM90 source receipt requires original publication flags")
        if document["lost_fields_or_relabels"] == 0 and any(
                not record["full_original_receipt_equal"] for record in records):
            raise ValueError("formal SM90 source receipt cannot hide publication losses")
    else:
        raise ValueError("unknown formal SM90 source receipt")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError("formal SM90 source receipt requires original snapshot identifiers")
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("formal SM90 source receipt requires nonnegative integer counts")
    if any(not isinstance(values, list) or not values or
           any(not isinstance(value, str) or not value for value in values) for values in (scope, limits)):
        raise ValueError("formal SM90 source receipt requires original scope and limitations")


def _validate_target_fused_embedding_review(document):
    """Keep source-only acceptance tied to the publisher's actual records."""
    message = "formal target embedding receipt"
    if document["no_GPU_or_download_or_push"] is not True:
        raise ValueError(f"{message} requires its offline boundary")
    if (not isinstance(document["source_reference"], str) or not document["source_reference"].strip()
            or not isinstance(document["remaining"], dict) or not document["remaining"]):
        raise ValueError(f"{message} requires its source reference and remaining limits")
    for field in ("direct_scopes_passed", "model_phases", "same_case_removed_calls", "candidate_failures"):
        if type(document[field]) is not int or document[field] < 0:
            raise ValueError(f"{message} requires nonnegative integer counts")
    snapshots = list(document["implementation_snapshots"])
    if not snapshots or not isinstance(document["direct_heads"], dict) or not document["direct_heads"]:
        raise ValueError(f"{message} requires original snapshot bindings")
    if set(document["direct_heads"]) != set(document["direct_dirty"]):
        raise ValueError(f"{message} requires matching head/dirty observation scopes")
    for scope, observation in document["direct_heads"].items():
        for boundary in ("before", "after"):
            snapshots.append(observation[boundary])
            changes = document["direct_dirty"][scope][boundary]
            if not isinstance(changes, list) or any(not isinstance(change, str) for change in changes):
                raise ValueError(f"{message} requires original dirty-file observations")
    records = document["records"]
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{message} requires original model records")
    for record in records:
        snapshots.extend(record[key]["commit"] for key in ("before_snapshot", "after_snapshot"))
        if not isinstance(record["phases"], list) or not record["phases"]:
            raise ValueError(f"{message} requires original phase records")
        for phase in record["phases"]:
            if any(not isinstance(phase[key], list) or any(not isinstance(value, str) for value in phase[key])
                   for key in ("removed", "changed")):
                raise ValueError(f"{message} requires original call-delta lists")
            if (type(phase["unpriced"]) is not int or phase["unpriced"] < 0
                    or (phase["unpriced"] > 0 and phase["roofline_end_to_end"] != "partial")):
                raise ValueError(f"{message} cannot promote an unpriced phase")
    if (document["model_phases"] != sum(len(record["phases"]) for record in records)
            or document["same_case_removed_calls"] != sum(
                len(phase["removed"]) for record in records for phase in record["phases"])):
        raise ValueError(f"{message} counts disagree with original records")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError(f"{message} requires original snapshot identifiers")
    if (not isinstance(document["limits"], list) or not document["limits"]
            or any(not isinstance(value, str) or not value for value in document["limits"])
            or not isinstance(document["resolved_pending_embedding"], str)
            or not document["resolved_pending_embedding"].strip()):
        raise ValueError(f"{message} requires its original scope and limitations")


def _validate_residual_norm_review(filename, document):
    """Preserve independent residual receipts without upgrading their scope."""
    message = "formal residual norm receipt"
    if (document["no_GPU_or_download_or_push"] is not True
            or document["all_other_call_and_step_content_equal"] is not True):
        raise ValueError(f"{message} requires original offline/delta qualifications")
    if (not isinstance(document["source_reference"], str) or not document["source_reference"]
            or not isinstance(document["snapshot_qualification"], str) or not document["snapshot_qualification"]
            or not isinstance(document["remaining"], dict) or not document["remaining"]
            or not isinstance(document["remaining"]["item"], str) or not document["remaining"]["item"].strip()
            or not isinstance(document["limits"], list) or not document["limits"]
            or any(not isinstance(limit, str) or not limit for limit in document["limits"])):
        raise ValueError(f"{message} requires source, snapshot and remaining limitations")
    snapshots = [document["implementation"]]

    def snapshot(value):
        snapshots.append(value["commit"])
        dirty(value["uncommitted_changes"])

    def dirty(values):
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{message} requires original dirty observations")

    heads, changes = document["direct_heads"], document["direct_dirty"]
    if not isinstance(heads, dict) or not heads or not isinstance(changes, dict) or set(heads) != set(changes):
        raise ValueError(f"{message} requires matching direct head/dirty observations")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{message} requires original phase records")
    if filename == "glm_residual_norm_review.json":
        snapshots.append(document["reporting_snapshot"])
        for scope, boundaries in heads.items():
            # The initial direct observation has only a captured before.
            # Its missing after must stay missing, as qualified by the receipt.
            if not isinstance(boundaries, dict) or "before" not in boundaries or set(boundaries) != set(changes[scope]):
                raise ValueError(f"{message} requires captured direct boundaries")
            for boundary, head in boundaries.items():
                snapshots.append(head)
                dirty(changes[scope][boundary])
        phases = []
        for record in records:
            snapshot(record["before_snapshot"])
            snapshot(record["candidate_snapshot"])
            if not isinstance(record["phases"], list) or not record["phases"]:
                raise ValueError(f"{message} requires original phase records")
            phases.extend(record["phases"])
        phase_count, changed_field = document["model_phase_count"], "changed_norms"
    elif filename == "kimi_residual_norm_review.json":
        for boundary, head in heads.items():
            snapshots.append(head)
            dirty(changes[boundary])
        for key in ("artifact_actual_snapshot", "before_snapshot", "after_snapshot"):
            snapshot(document[key])
        phases = records
        phase_count, changed_field = document["model_phases"], "changed_norm_calls"
    else:
        raise ValueError(f"{message} has an unknown source")
    counts = [document["direct_scopes_passed"], document["candidate_failures"],
              document["changed_norm_calls"], phase_count]
    for phase in phases:
        counts.extend(phase[key] for key in ("calls", changed_field, "added_logical_bytes", "unpriced"))
        if phase["roofline_end_to_end"] != "partial":
            raise ValueError(f"{message} cannot promote partial source phases")
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError(f"{message} requires nonnegative integer counts")
    if phase_count != len(phases) or document["changed_norm_calls"] != sum(phase[changed_field] for phase in phases):
        raise ValueError(f"{message} counts disagree with original records")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError(f"{message} requires original snapshot identifiers")


def _validate_target_sampling_review(document, delta):
    """Validate the original sampler receipt and its linked phase deltas."""
    message = "formal target sampling receipt"
    if (document["no_GPU_or_download_or_push"] is not True
            or delta["no_GPU_or_download_or_push"] is not True
            or delta["all_other_artifact_content_equal"] is not True):
        raise ValueError(f"{message} requires original offline/source qualifications")
    if (not isinstance(document["source_reference"], str) or not document["source_reference"]
            or document["source_reference"] != delta["source_reference"]):
        raise ValueError(f"{message} requires a consistent source reference")
    for values in (document["accepted_scope"], document["limits"], delta["limits"]):
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or not value for value in values):
            raise ValueError(f"{message} requires original scope and limits")
    if (not isinstance(document["snapshot_qualification"], str) or not document["snapshot_qualification"]
            or not isinstance(document["remaining"]["item"], str) or not document["remaining"]["item"]):
        raise ValueError(f"{message} requires snapshot and remaining qualifications")
    snapshots = list(document["implementations"])
    if not snapshots or set(document["direct_heads"]) != {"before", "after"} or set(document["direct_dirty"]) != {"before", "after"}:
        raise ValueError(f"{message} requires captured direct observations")
    for boundary in ("before", "after"):
        snapshots.append(document["direct_heads"][boundary])
        changes = document["direct_dirty"][boundary]
        if not isinstance(changes, list) or any(not isinstance(change, str) for change in changes):
            raise ValueError(f"{message} requires original dirty observations")
    checks = document["AST_scope_checks"]
    if not isinstance(checks, list) or not checks:
        raise ValueError(f"{message} requires original helper binding checks")
    for check in checks:
        snapshots.extend((check["implementation"], check["direct_snapshot"]))
        if (check["function_AST_unchanged"] is not True
                or check["implementation"] not in document["implementations"]
                or check["direct_snapshot"] not in document["direct_heads"].values()):
            raise ValueError(f"{message} cannot inherit changed helper acceptance")
    counts = [document[key] for key in ("direct_scopes_passed", "candidate_failures", "new_counter_categories")]
    records = delta["records"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"{message} requires original model records")
    phases = []
    for record in records:
        for key in ("before_snapshot", "candidate_snapshot"):
            snapshots.append(record[key]["commit"])
            changes = record[key]["uncommitted_changes"]
            if not isinstance(changes, list) or any(not isinstance(change, str) for change in changes):
                raise ValueError(f"{message} requires original artifact dirty state")
        if not isinstance(record["phases"], list) or not record["phases"]:
            raise ValueError(f"{message} requires original phase records")
        phases.extend(record["phases"])
    projections = 0
    for phase in phases:
        counts.extend(phase[key] for key in ("before_calls", "after_calls", "added_materializations",
                                            "logical_bytes", "before_unpriced", "after_unpriced"))
        indices = phase["newly_bound_INT64_hidden_selection"]
        if not isinstance(indices, list) or any(not isinstance(value, str) for value in indices):
            raise ValueError(f"{message} requires original conditional projection IDs")
        projections += len(indices)
        if phase["roofline_end_to_end"] != "partial":
            raise ValueError(f"{message} cannot promote partial phases")
        if phase["after_calls"] - phase["before_calls"] != phase["added_materializations"]:
            raise ValueError(f"{message} call counts disagree")
    counts.extend(delta[key] for key in ("model_phases", "added_materializations", "new_declared_INT64_hidden_projections"))
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError(f"{message} requires nonnegative integer counts")
    if (delta["model_phases"] != len(phases)
            or delta["added_materializations"] != sum(phase["added_materializations"] for phase in phases)
            or delta["new_declared_INT64_hidden_projections"] != projections):
        raise ValueError(f"{message} totals disagree with original phases")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{8,40}", value) is None for value in snapshots):
        raise ValueError(f"{message} requires original snapshot identifiers")


def _formal_source_delta_reviews(reference: Path):
    """Optional published source receipts, preserving each original binding.

    These documents are independent reference output, not current-tree or
    runtime acceptance. Counts and scopes are copied; no predictor replay
    or inferred snapshot equivalence is performed here.
    """
    filenames = (
        "glm_greedy_sampling_comparison.json", "glm_legacy_materialization_comparison.json",
        "glm_parameter_storage_review.json", "target_logits_row_review.json",
        "glm_first_input_review.json", "glm_padded_preparation_review.json",
        "glm_legacy_step_metadata_published_review.json", "glm_initial_slot_mapping_review.json",
        "grouped_topk_offline_compile_review.json", "grouped_topk_full_translation_unit_review.json",
        "grouped_topk_host_registration_review.json", "grouped_topk_finite_precision_reference.json",
        "operator_finite_precision_publication_review.json", "operator_model_roofline_paths_review.json",
        "glm_target_fused_embedding_review.json", "glm_residual_norm_review.json", "kimi_residual_norm_review.json",
        "target_sampling_materialization_review.json",
        *_MHC_SM90_REVIEW_FILENAMES)
    # Optional fields belong to the published per-document receipt schema;
    # absent metadata stays absent and cannot become current-head acceptance.
    metadata_fields = (
        "candidate", "snapshot", "accepted_scope", "scope", "source_scope", "binding_limit", "limits", "limitations",
        "before_head", "before_dirty", "after_head", "after_dirty", "test_start", "test_end_observation",
        "after_observation", "after_observation_is_not_test_end_snapshot",
        "helper_same_as_test_start_committed_base", "branch_qualification",
        "current_source_branches_match_committed740", "direct_summary",
        "direct_checks_reused", "passing_tests_repeated", "final_candidate_or_goal_accepted",
        "full_module_compile", "device_subset_compile", "kernels", "independent_static_observations",
        "result", "command", "source_transform", "header_source", "production_metadata_change",
        "historical_failure_qualification", "source_receipts", "independent_checks", "ELF_needed_libraries",
        "own_reader_corrections", "reused_worker_host_observation", "independently_loaded_library", "closes",
        "source_reference_qualification", "examples", "findings",
        "candidate_observation", "source_metadata_implementation", "receipt_validation_implementation",
        "source_reference", "metadata_checks", "published_receipt", "source_scenarios_preserved",
        "own_reader_correction", "remaining", "publication", "reader_correction")
    count_fields = ("checks", "failed", "immutable_comparison_checks", "immutable_comparison_failed",
                    "artifact_checks", "artifact_failed", "model_layout_mode_records", "phase_records",
                    "recount_mismatches")
    records = []
    for filename in filenames:
        path = reference / filename
        if not path.is_file():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError("formal source review must be a JSON object")
        if filename == "target_sampling_materialization_review.json":
            delta_path = Path(document["model_delta_review"])
            delta = json.loads(delta_path.read_text(encoding="utf-8"))
            _validate_target_sampling_review(document, delta)
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "model_delta_receipt": {"reference_artifact": str(delta_path.resolve()), "receipt": delta},
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename in ("glm_residual_norm_review.json", "kimi_residual_norm_review.json"):
            _validate_residual_norm_review(filename, document)
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename == "glm_target_fused_embedding_review.json":
            _validate_target_fused_embedding_review(document)
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename in _MHC_SM90_REVIEW_FILENAMES:
            _validate_mhc_sm90_review(filename, document)
            # Preserve the whole original receipt, including conditional
            # boxes, excluded bindings and original per-call evidence.
            records.append({"reference_artifact": str(path.resolve()), "receipt": document,
                            "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
            continue
        if filename == "grouped_topk_finite_precision_reference.json":
            _validate_grouped_topk_finite_precision_receipt(document)
        if filename == "operator_finite_precision_publication_review.json":
            candidate = document["candidate_observation"]
            if not isinstance(candidate, str) or re.fullmatch(r"[0-9a-f]{40}", candidate) is None:
                raise ValueError("finite-precision publication requires original candidate observation")
            checks = document["metadata_checks"]
            if (not isinstance(checks, list) or len(checks) != 2 or
                    any(not isinstance(check, dict) or type(check["block_path"]) is not bool
                        or not isinstance(check["metadata"], dict) for check in checks) or
                    {check["block_path"] for check in checks} != {True, False}):
                raise ValueError("finite-precision publication requires both metadata branches")
            published = document["published_receipt"]
            _validate_grouped_topk_finite_precision_receipt(published["receipt"])
            scenarios = document["source_scenarios_preserved"]
            if type(scenarios) is not int or scenarios != len(published["receipt"]["examples"]):
                raise ValueError("finite-precision publication scenario count disagrees with receipt")
        if filename == "operator_model_roofline_paths_review.json":
            source_records = document["records"]
            if not isinstance(source_records, list) or any(
                    not isinstance(record, dict) or not isinstance(record["phases"], list)
                    for record in source_records):
                raise ValueError("source-path publication requires original record/phase lists")
            if (document["model_layout_mode_records"] != len(source_records) or
                    document["phase_records"] != sum(len(record["phases"]) for record in source_records)):
                raise ValueError("source-path publication count disagrees with original records")
        receipt = {key: document[key] for key in metadata_fields if key in document}
        if not any(key in receipt for key in ("accepted_scope", "scope", "source_scope")):
            raise ValueError("formal source review requires its original acceptance scope")
        if filename == "grouped_topk_host_registration_review.json":
            checks = document["independent_checks"]
            if not isinstance(checks, dict) or any(type(value) is not bool for value in checks.values()):
                raise ValueError("formal source review host checks must be a boolean record map")
            receipt["checks"] = len(checks)
            receipt["failed"] = sum(not value for value in checks.values())
        for key in count_fields:
            if key in document:
                count = document[key]
                if filename == "grouped_topk_full_translation_unit_review.json" and key == "checks":
                    if not isinstance(count, list):
                        raise ValueError("formal source review full-TU checks must be a record list")
                    if any(not isinstance(record, dict) or type(record["pass"]) is not bool for record in count):
                        raise ValueError("formal source review full-TU check requires a boolean pass")
                    if sum(not record["pass"] for record in count) != document["failed"]:
                        raise ValueError("formal source review full-TU failed count disagrees with records")
                    receipt["check_records"] = count
                    receipt[key] = len(count)
                    continue
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(f"formal source review {key} must be a nonnegative integer")
                receipt[key] = count
        if "records" in document:
            if filename == "operator_model_roofline_paths_review.json":
                # Preserve each original phase flag, family and snapshot;
                # source_projection_only cannot be normalized to partial.
                receipt["records"] = document["records"]
            else:
                receipt["records"] = [{key: record[key] for key in
                                        ("snapshot", "artifact", "path", "tp", "failed") if key in record}
                                       for record in document["records"]]
        records.append({"reference_artifact": str(path.resolve()), "receipt": receipt,
                        "snapshot_policy": "original receipt binding only; no current-head or runtime inheritance"})
    return records


def _static_roofline_path_coverage(probe, artifact):
    """Read source-case pricing coverage without predicting or accepting it."""
    phases = []
    for step in probe["steps"]:
        calls = step["calls"]
        unpriced = []
        for call in calls:
            status = call["prediction"]["status"]
            if status == "unpriced":
                if call["prediction"]["seconds"] is not None:
                    raise ValueError("unpriced source call cannot carry predicted seconds")
                unpriced.append(call)
            elif status not in ("priced", "logical_alias"):
                raise ValueError("unknown source-case pricing status")
        if (type(step["kernel_calls"]) is not int or step["kernel_calls"] != len(calls)
                or type(step["unpriced_calls"]) is not int or step["unpriced_calls"] != len(unpriced)):
            raise ValueError("source-case call coverage disagrees with original artifact")
        phases.append({"phase": step["phase"], "calls": len(calls),
            "unpriced_calls": len(unpriced),
            "unpriced_families": dict(Counter(call["variant"] for call in unpriced)),
            "first_unpriced_call": None if not unpriced else unpriced[0]["call_id"],
            "roofline_end_to_end": step["roofline_end_to_end"]})
    return {"artifact": str(artifact), "snapshot": probe["snapshot"],
            "model_id": probe["model_id"], "model_revision": probe["model_revision"],
            "mtp": probe["mtp"], "deployment": probe["deployment"], "phases": phases,
            "runtime_or_normal_kv_accepted": False,
            "scope": "source-case coverage only; original identity, provider, routing and runtime qualifications retained"}


def _operator_probe_reproduction(probe, artifact):
    """Spell existing CLI inputs only when they reproduce every source fact.

    A historical artifact with missing constructor state or a programmatic
    input outside the existing CLI is retained as evidence, without an
    executable-looking command which would silently select the old graph.
    """
    def unavailable(reason):
        return {"command": None, "status": "inputs_not_representable", "reason": reason}

    if ("source_runtime_inputs_by_phase" not in probe
            or probe["source_runtime_inputs_by_phase"] is None):
        return unavailable("artifact has no complete phase RuntimeInputs; source/constructor inputs cannot be inferred")
    phases = probe["source_runtime_inputs_by_phase"]
    if not isinstance(phases, dict) or not phases:
        return unavailable("artifact has no named source phases")
    first = next(iter(phases.values()))
    source = first["symbolic_forward_source"]
    sampler = first["target_sampling_source"]
    mode = {"off": "off", "on": "native"}[probe["mtp"]]
    expected_requests = _operator_probe_phase_requests(probe["prefill_tokens"], mode,
        decode_prefix=source == "vllm_0_27_1_deepseek_v3_sync_fp8")
    expected_rows = {phase: _json_value([{**asdict(request), "positions": tuple(range(
        request.computed_tokens, request.computed_tokens + request.scheduled_tokens))}
        for request in requests]) for phase, requests in expected_requests}
    if (set(phases) != set(expected_rows) or len(probe["steps"]) != len(expected_rows)
            or any(step["phase"] not in expected_rows
                or step["scheduled_requests"] != expected_rows[step["phase"]] for step in probe["steps"])):
        return unavailable("custom phase requests are not expressible by the existing probe CLI fixture")
    ports = {phase: values["embedding_token_ids"] for phase, values in phases.items()}
    if any(ids is None for ids in ports.values()) and not all(ids is None for ids in ports.values()):
        return unavailable("mixed known/unknown phase token ports are not expressible by the existing CLI")
    command = [sys.executable, "scripts/evaluation/serving/report_model_operator_probe.py",
        "--model", str(probe["model_config_path"]), "--tensor-parallel-size",
        str(probe["deployment"]["tensor_parallel_size"]), "--prefill-tokens", str(probe["prefill_tokens"]),
        "--mtp-mode", mode, "--output", str(artifact)]
    if probe["deployment"]["enable_expert_parallel"]:
        command.append("--enable-expert-parallel")
    if sampler is not None:
        command += ["--target-sampling-source", sampler]
    pp = probe["deployment"]["pipeline_parallel_size"]
    if pp != 1:
        pipeline = probe["pipeline_source_requirements"]
        command += ["--pipeline-parallel-size", str(pp), "--pipeline-link", pipeline["pipeline_link_scenario"],
            "--pipeline-layer-partition", pipeline["VLLM_PP_LAYER_PARTITION"]]
    glm_sources = ("vllm_0_30_v32_bf16_sync_explicit_allreduce", "vllm_0_30_v32_fp8_sync_explicit_allreduce")
    if source == "vllm_0_11_glm4_sync":
        if mode != "off" or pp != 1:
            return unavailable("GLM4 probe CLI declares only its target-off TP1 source fixture")
        if any(ids is None for ids in ports.values()):
            return unavailable("GLM4 source CLI cannot reproduce unknown token values with a synthetic-ID fixture")
        fixture = _default_glm4_target_probe_source(load_model_config(probe["model_config_path"]),
            ParallelConfig(tensor_parallel_size=probe["deployment"]["tensor_parallel_size"],
                enable_expert_parallel=probe["deployment"]["enable_expert_parallel"]),
            expected_requests, token_ports={phase: None if ids is None else tuple(ids) for phase, ids in ports.items()},
            sampler=sampler)
        if fixture is None:
            return unavailable("GLM4 probe CLI cannot represent this parallel source fixture")
        expected, _ = fixture
    elif source == "vllm_0_30_v4_sync_fp8_ds_mla":
        if mode != "off" or first["v4_eager_metadata_source"] != "vllm_0_30_eager_dict_all_valid":
            return unavailable("V4 CLI requires the existing target-off eager dict source")
        command += ["--v4-eager-metadata", "--v4-routing-source", first["v4_routing_source"],
            "--v4-source-max-model-len", str(first["v4_source_max_model_len"]),
            "--v4-source-max-batched-tokens", str(first["v4_source_max_batched_tokens"])]
        branch = first["v4_source_use_deep_gemm"]
        if type(branch) is not bool:
            return unavailable("V4 artifact has no declared DeepGEMM process branch")
        command.append("--v4-source-use-deep-gemm" if branch else "--no-v4-source-use-deep-gemm")
        if first["v4_indexer_selection_source"] is not None:
            command += ["--v4-indexer-selection-source", first["v4_indexer_selection_source"]]
        routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
        expected = {phase: RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            symbolic_forward_source=source, target_sampling_source=sampler,
            embedding_token_ids=None if ids is None else tuple(ids),
            v4_eager_metadata_source=first["v4_eager_metadata_source"],
            v4_routing_source=first["v4_routing_source"], v4_source_max_model_len=first["v4_source_max_model_len"],
            v4_source_max_batched_tokens=first["v4_source_max_batched_tokens"],
            v4_source_use_deep_gemm=branch, v4_indexer_selection_source=first["v4_indexer_selection_source"])
            for phase, ids in ports.items()}
    elif source in ("cuda_eager_inplace", *glm_sources, "vllm_0_30_k2_6_int4_sync_explicit_allreduce",
            "vllm_0_27_1_deepseek_v3_sync_fp8"):
        if source != "cuda_eager_inplace" and sampler is None:
            return unavailable("existing ordinary probe CLI cannot combine explicit constructor inputs with a sampler-less source")
        if source == "cuda_eager_inplace" and sampler not in (
                None, "vllm_0_30_plain_sampler_bf16", "vllm_0_30_plain_greedy_bf16", "vllm_0_30_plain_greedy_fp16"):
            return unavailable("existing ordinary probe CLI cannot spell this model/sampler version")
        state = None
        if mode == "native":
            if source not in glm_sources:
                return unavailable("existing native CLI accepts only explicit GLM padded inputs")
            from .simulation_configuration import decode_typed_step_source_json
            state = {phase: decode_typed_step_source_json(json.dumps(values["model_source_inputs"]),
                V32PaddedMTPInputs) for phase, values in phases.items()}
            for phase, values in phases.items():
                command += ["--glm-mtp-phase-source-input", phase, json.dumps(values["model_source_inputs"])]
        if source != "cuda_eager_inplace":
            for flag, name in (("--source-max-model-len", "symbolic_source_max_model_len"),
                    ("--source-max-batched-tokens", "symbolic_source_max_batched_tokens"),
                    ("--source-max-requests", "symbolic_source_max_requests"),
                    ("--mtp-draft-sampling-source", "mtp_draft_sampling_source"),
                    ("--glm-indexer-selection-source", "glm_indexer_selection_source"),
                    ("--glm-prefill-attention-source", "glm_prefill_attention_source")):
                if first[name] is not None:
                    command += [flag, str(first[name])]
            if first["symbolic_workspace_initial_capacity_bytes"] is not None:
                command += ["--source-workspace-initial-capacity-bytes",
                    *(str(value) for value in first["symbolic_workspace_initial_capacity_bytes"])]
        expected = build_model_operator_probe_source_inputs(symbolic_forward_source=source,
            embedding_token_ids_by_phase={phase: None if ids is None else tuple(ids) for phase, ids in ports.items()},
            source_max_model_len=first["symbolic_source_max_model_len"],
            source_max_batched_tokens=first["symbolic_source_max_batched_tokens"], target_sampling_source=sampler,
            workspace_initial_capacity_bytes=(None if first["symbolic_workspace_initial_capacity_bytes"] is None
                else tuple(first["symbolic_workspace_initial_capacity_bytes"])),
            glm_indexer_selection_source=first["glm_indexer_selection_source"],
            glm_prefill_attention_source=first["glm_prefill_attention_source"],
            source_max_requests=first["symbolic_source_max_requests"],
            mtp_draft_sampling_source=first["mtp_draft_sampling_source"], model_source_inputs_by_phase=state)
        model = load_model_config(probe["model_config_path"])
        if source == "cuda_eager_inplace" and model.num_experts:
            routing = ExpertRoutingInput.seeded_assumption("operator_support_static_seeded_routing_v1", 0)
            expected = {phase: replace(value, expert_routing=routing) for phase, value in expected.items()}
    else:
        return unavailable(f"source {source!r} has no existing probe CLI spelling")
    expected = {phase: _json_value(asdict(value)) for phase, value in expected.items()}
    differences = sorted({name for phase, values in expected.items()
        for name in set(values) | set(phases[phase])
        if name not in values or name not in phases[phase] or phases[phase][name] != values[name]})
    if differences:
        return unavailable("existing CLI cannot preserve phase RuntimeInputs fields: " + ", ".join(differences))
    if all(ids is not None for ids in ports.values()):
        token_file = Path(artifact).with_name(Path(artifact).stem + "_embedding_token_ids.json")
        token_file.write_text(json.dumps(ports, indent=2) + "\n", encoding="utf-8")
        command += ["--embedding-token-ids", str(token_file)]
    return {"command": shlex.join(command), "status": "complete_source_inputs", "reason": None}


def write_operator_support_checkpoint(
        output: Path, reference: Path, tests_log: Path, recheck_log: Path | None,
        kernel_results: Path, *, current_probes: tuple[Path, ...] = (),
        source_storage_probes: tuple[Path, ...] = (),
        weight_diagnostics: tuple[Path, ...] = (),
        source_graph_probes: tuple[Path, ...] = ()) -> None:
    """Regenerate the Goal checkpoint from static facts and external review.

    A static priced call is not source alignment or real Scheduler evidence.
    This checkpoint deliberately leaves every full-scope gate open; it does
    not turn implementation smoke or missing evidence into acceptance.
    """
    # Validate optional published receipts before any derived output writes.
    formal_source_reviews = _formal_source_delta_reviews(reference)
    contract_path = reference / "source_contract.json"
    review_path = reference / "review.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    glm_capacity_path = reference / "glm_capacity_reference.json"
    glm_capacity = json.loads(glm_capacity_path.read_text(encoding="utf-8"))
    for case in glm_capacity["cases"]:
        if case["normal_capacity_accepted"] is not False:
            raise ValueError("GLM routed-weight lower bounds are not normal KV acceptance")
    glm_capacity_record = {"artifact": str(glm_capacity_path), **glm_capacity}
    weight_records = []
    for artifact in weight_diagnostics:
        diagnostic = json.loads(artifact.read_text(encoding="utf-8"))
        if diagnostic["status"] != "diagnostic_not_normal_capacity_acceptance":
            raise ValueError("weight diagnostic cannot claim normal capacity acceptance")
        for case in diagnostic["cases"]:
            if case["normal_kv_capacity_accepted"] is not False:
                raise ValueError("constructor parameter inventory is not normal capacity evidence")
            # payload_scope is an optional diagnostic metadata field. Older
            # V4 records keep their original schema/snapshot and normalize None.
            scope = case["payload_scope"] if "payload_scope" in case else None
            if scope not in (None, "kimi_active_outer_registered_parameters", "glm_parameter_source_projection"):
                raise ValueError("unsupported constructor parameter diagnostic scope")
            case["payload_scope"] = scope
            if scope == "kimi_active_outer_registered_parameters" and case["actual_outer_loader_or_runtime_binding"] is not False:
                raise ValueError("Kimi constructor source scenario cannot claim actual loader/runtime binding")
            if scope == "glm_parameter_source_projection" and case["actual_loader_or_runtime_binding"] is not False:
                raise ValueError("GLM parameter source projection cannot claim actual loader/runtime binding")
        weight_records.append({"artifact": str(artifact), **diagnostic})
    graph_records = []
    for artifact in source_graph_probes:
        graph = json.loads(artifact.read_text(encoding="utf-8"))
        for flag in ("lowering_executed", "prediction_executed",
                     "actual_runtime_or_loaded_weight_binding", "normal_kv_capacity_accepted"):
            if graph[flag] is not False:
                raise ValueError("source graph evidence must exclude lowering/pricing/runtime/capacity acceptance")
        graph_records.append({
            "artifact": str(artifact), "snapshot": graph["snapshot"],
            "model_id": graph["model_id"], "deployment": graph["deployment"],
            "v4_eager_metadata_source": graph["v4_eager_metadata_source"],
            "scope": graph["scope"],
            "phases": [{"phase": step["phase"], "nodes": len(step["nodes"]),
                        "edges": len(step["edges"])} for step in graph["steps"]],
        })
    layer_comparison_path = reference / "layer_stack_comparison.json"
    layer_comparison = json.loads(layer_comparison_path.read_text(encoding="utf-8"))
    layer_rescore = _rescore_formal_layer_shapes(layer_comparison)
    layer_rescore["expectation_reference"] = str(layer_comparison_path)
    layer_rescore_path = output / "layer_stack_rescore.json"
    layer_rescore_path.write_text(json.dumps(layer_rescore, indent=2) + "\n", encoding="utf-8")
    v4_comparison_path = reference / "v4_compressor_comparison.json"
    v4_comparison = json.loads(v4_comparison_path.read_text(encoding="utf-8"))
    v4_rescore = _rescore_v4_compressor_bytes(v4_comparison)
    v4_rescore["expectation_reference"] = str(v4_comparison_path)
    v4_rescore_path = output / "v4_compressor_rescore.json"
    v4_rescore_path.write_text(json.dumps(v4_rescore, indent=2) + "\n", encoding="utf-8")
    native_comparison_path = reference / "native_mtp_comparison.json"
    native_comparison = json.loads(native_comparison_path.read_text(encoding="utf-8"))
    kernel_matrix = json.loads(kernel_results.read_text(encoding="utf-8"))
    tests_summary = tests_log.read_text(encoding="utf-8").splitlines()[-1]
    if recheck_log is None:
        if not re.fullmatch(r"\d+ passed in .*", tests_summary):
            raise ValueError("a failing or incomplete test run requires explicit recheck evidence")
        recheck_summary = "本轮全维护通过，无失败项重跑"
    else:
        recheck_summary = recheck_log.read_text(encoding="utf-8").splitlines()[-1]
    if contract["new_gpu_execution"] is not False:
        raise ValueError("checkpoint expects an offline source reference")
    models = ("kimi-k2.6", "kimi-k3", "glm-5.2", "glm-5.2-fp8", "glm-5.3",
              "deepseek-v4-flash", "deepseek-v4-pro")
    alignment = []
    rows = []
    native_rows = []
    native_records = []
    for model in models:
        config = Path("configs/model") / f"{model}.json"
        if not config.is_file():
            alignment.append({"model": model, "status": "missing_exact_config"})
            rows.append(f"| {model} | 未执行 | 缺准确 checkpoint/config |")
            continue
        artifact = output / f"{model}_static_probe.json"
        if not artifact.is_file():
            alignment.append({"model": model, "status": "no_complete_model_probe"})
            rows.append(f"| {model} | 未执行完整模型 | 源码组件不等于完整静态支持 |")
            continue
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        current_model = load_model_config(str(config))
        if ("model_id" not in probe or "model_revision" not in probe
                or probe["model_id"] != current_model.model_id
                or probe["model_revision"] != current_model.model_revision):
            alignment.append({"model": model, "status": "historical_identity_excluded", "artifact": str(artifact)})
            rows.append(f"| {model} | 历史身份结果排除 | 不作为当前官方 checkpoint 验收 |")
            continue
        if Path(probe["model_config_path"]) != config or probe["mtp"] != "off":
            raise ValueError("static probe model/mode binding disagrees")
        phases = []
        for step in probe["steps"]:
            phases.append({
                "phase": step["phase"], "calls": step["kernel_calls"],
                "unpriced_calls": step["unpriced_calls"],
                "roofline_end_to_end": step["roofline_end_to_end"],
                "variant_counts": dict(Counter(call["variant"] for call in step["calls"])),
            })
        reproduction = _operator_probe_reproduction(probe, artifact)
        alignment.append({
            "model": model, "status": "modeled_facts_await_independent_comparison",
            "artifact": str(artifact), "snapshot": probe["snapshot"], "phases": phases,
            "command": reproduction["command"], "reproduction": reproduction,
        })
        counts = "/".join(str(phase["calls"]) for phase in phases)
        gaps = "/".join(str(phase["unpriced_calls"]) for phase in phases)
        rows.append(f"| {model} | {counts} calls；未计价 {gaps} | MTP-off 静态事实，待独立对齐 |")
    for name in ("glm-5.2_mtp_tp1", "glm-5.2_mtp"):
        artifact = output / f"{name}_static_probe.json"
        if not artifact.is_file():
            native_records.append({"artifact": str(artifact), "status": "not_executed"})
            native_rows.append(f"| {name} | 未执行 | 原生 on 证据缺失 |")
            continue
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        if (probe["mtp"] != "on" or probe["draft_architecture"] != "DeepseekV32MTPModel"
                or Path(probe["model_config_path"]) != Path("configs/model/glm-5.2.json")):
            raise ValueError("native GLM probe identity/mode disagrees")
        phases = [{"phase": step["phase"], "calls": step["kernel_calls"],
                   "unpriced_calls": step["unpriced_calls"]} for step in probe["steps"]]
        native_records.append({"artifact": str(artifact), "snapshot": probe["snapshot"],
                               "status": probe["on_mode_status"], "phases": phases,
                               "gaps": probe["native_source_gaps"]})
        counts = "/".join(str(phase["calls"]) for phase in phases)
        native_rows.append(f"| {name} | {counts} calls | partial：编译器/runtime、target 融合实际发射与slot、proposer 元数据和正常容量仍缺验收 |")
    current_records = []
    current_rows = []
    current_pipeline_scopes = []
    existing_paths = {}
    for artifact in current_probes:
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        # The last supplied artifact for each layout/mode retains its own
        # source scenario and snapshot; it is never current-head acceptance.
        path_key = (probe["model_id"], probe["mtp"],
                    probe["deployment"]["tensor_parallel_size"],
                    probe["deployment"]["pipeline_parallel_size"])
        existing_paths[path_key] = _static_roofline_path_coverage(probe, artifact)
        phases = [{"phase": step["phase"], "calls": step["kernel_calls"],
                   "unpriced_calls": step["unpriced_calls"],
                   "draft_sampling_stage_counts": dict(Counter(
                       call["variant"] for call in step["calls"]
                       if call["variant"] in ("dsv32_draft_argmax", "dsv32_draft_id_cast",
                                              "dsv32_draft_row_select", "dsv32_draft_buffer_copy", "dsv32_draft_token_stack", "dsv32_draft_id_scatter", "dsv32_draft_prepare_next_ids", "dsv32_draft_prepare_sample_indices", "dsv32_draft_seq_rollback", "dsv32_draft_step_metadata"))),
                   "draft_materialization_logical_roofline_calls": sum(
                       call["variant"] in DSV32_DRAFT_MATERIALIZATION_VARIANTS
                       and call["prediction"]["status"] == "priced"
                       and "scope" in call["prediction"]
                       and call["prediction"]["scope"] == "logical_roofline_only_no_launch_binding"
                       for call in step["calls"]),
                   "draft_int32_logical_roofline_calls": sum(
                       call["variant"] in DSV32_DRAFT_INT32_BOOKKEEPING_VARIANTS
                       and call["prediction"]["status"] == "priced"
                       and "scope" in call["prediction"]
                       and call["prediction"]["scope"] == "logical_roofline_only_no_launch_binding"
                       for call in step["calls"]),
                   "target_logits_selection_calls": sum(
                       call["variant"] == "vllm_logits_row_select" for call in step["calls"]),
                   "target_sampler_cast_calls": sum(
                       call["variant"] == "vllm_logits_fp32_cast" for call in step["calls"]),
                   "target_sampler_cast_logical_bytes": sum(
                       call["resource_input"]["logical_bytes"] for call in step["calls"]
                       if call["variant"] == "vllm_logits_fp32_cast"),
                   "rejection_materialization_stage_counts": dict(Counter(
                       call["attributes"]["materialization_stage"] for call in step["calls"]
                       if call["variant"] == "vllm_rejection_logits_materialize")),
                   "rejection_materialization_logical_bytes": sum(
                       call["resource_input"]["logical_bytes"] for call in step["calls"]
                       if call["variant"] == "vllm_rejection_logits_materialize"),
                   "target_logits_logical_roofline_calls": sum(
                       call["variant"] == "vllm_logits_row_select"
                       and call["prediction"]["status"] == "priced"
                       and "scope" in call["prediction"]
                       and call["prediction"]["scope"] == "logical_roofline_only_no_launch_binding"
                       for call in step["calls"]),
                   "target_logits_selection_output_bytes": sum(
                       operand["bytes"] for call in step["calls"]
                       if call["variant"] == "vllm_logits_row_select"
                       for operand in call["resource_input"]["logical_work"]["operands"]
                       if operand["role"] == "sample_hidden_states"),
                   "target_logits_selection_index_sources": dict(Counter(
                       call["attributes"]["index_source"] for call in step["calls"]
                       if call["variant"] == "vllm_logits_row_select")),
                   "target_fused_embed_norm_calls": sum(
                       call["variant"] == "dsv32_embed_norm"
                       and call["prediction"]["status"] == "priced"
                       for call in step["calls"]),
                   "residual_add_norm_calls": sum(
                       call["variant"] == "add_rmsnorm" for call in step["calls"]),
                   "residual_add_norm_logical_bytes": sum(
                       call["resource_input"]["logical_bytes"] for call in step["calls"]
                       if call["variant"] == "add_rmsnorm"),
                   "source_vocab_embedding_calls": sum(
                       call["variant"] == "vocab_parallel_embedding" for call in step["calls"]),
                   "source_vocab_embedding_partial_output_bytes": sum(
                       operand["bytes"] for call in step["calls"]
                       if call["variant"] == "vocab_parallel_embedding"
                       for operand in call["resource_input"]["logical_work"]["operands"]
                       if operand["role"] == "partial_output"),
                   "unpriced_calls_with_logical_work": sum(
                       call["prediction"]["status"] == "unpriced"
                       and "logical_work" in call["resource_input"]
                       for call in step["calls"]),
                   "roofline_end_to_end": step["roofline_end_to_end"],
                   "scheduled_requests": step["scheduled_requests"],
                   "rank0_c128_compress": [{
                       "call_id": call["call_id"], "shape": call["shape"],
                       "dtype": call["dtype"], "demand": call["resource_input"]["demand"],
                       "dependencies": call["dependencies"],
                   } for call in step["calls"] if call["rank"] == 0
                   and call["variant"] == "dsv4_compressed_state_cache"
                   and call["attributes"]["finish_stage"] == "compress"]}
                  for step in probe["steps"]]
        current_records.append({"artifact": str(artifact), "snapshot": probe["snapshot"],
                                "model": probe["model_id"], "mtp": probe["mtp"],
                                "deployment": probe["deployment"], "phases": phases,
                                "scope": "modeled source-layer facts; independent acceptance not inherited"})
        if "mtp_draft_sampling_source" in probe:
            current_records[-1]["mtp_draft_sampling_source"] = probe["mtp_draft_sampling_source"]
        if "target_sampling_source" in probe:
            current_records[-1]["target_sampling_source"] = probe["target_sampling_source"]
        if "v4_eager_metadata_source" in probe:
            current_records[-1]["v4_eager_metadata_source"] = probe["v4_eager_metadata_source"]
        if "v4_routing_source" in probe:
            current_records[-1]["v4_routing_source"] = probe["v4_routing_source"]
            current_records[-1]["v4_routing_requirements"] = probe["v4_routing_requirements"]
        if "pipeline_source_requirements" in probe:
            current_records[-1]["pipeline_source_requirements"] = probe["pipeline_source_requirements"]
            current_records[-1]["pipeline_weight_projection"] = probe["pipeline_weight_projection"]
            rank_scope = probe["pipeline_source_requirements"]["modeled_tp_partial_producer_scope"]
            if rank_scope is not None:
                current_pipeline_scopes.append(
                    f"{artifact.name}（{probe['snapshot']['commit'][:8]}）的源码计算覆盖自述：{rank_scope}。")
        counts = "/".join(str(phase["calls"]) for phase in phases)
        gaps = "/".join(str(phase["unpriced_calls"]) for phase in phases)
        logical_work = "/".join(str(phase["unpriced_calls_with_logical_work"]) for phase in phases)
        current_rows.append(f"| {artifact.name} | {probe['snapshot']['commit'][:8]} | {counts} calls；未计价 {gaps}；其中有逻辑数学/存储量 {logical_work} |")
        if probe["model_id"] in ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro"):
            current_rows[-1] = current_rows[-1][:-1] + (
                f"；V4 metadata source={probe['v4_eager_metadata_source']} |")
            if "v4_routing_source" in probe and probe["v4_routing_source"] is not None:
                current_rows[-1] = current_rows[-1][:-1] + (
                    f"；routing source={probe['v4_routing_source']}（实际provider未绑定） |")
    source_storage_records = []
    for artifact in source_storage_probes:
        probe = json.loads(artifact.read_text(encoding="utf-8"))
        if (probe["status"] != "source_storage_subset_not_full_native_or_priced_model"
                or probe["independent_acceptance"] is not False
                or probe["new_gpu_execution"] is not False
                or probe["normal_gpu_v2_workload_supported"] is not False):
            raise ValueError("source storage subset cannot claim native/runtime acceptance")
        # uniform_updates is the schema-defined optional source-work part;
        # old storage-only snapshots keep their original subset scope.
        work = probe["logical_work"]
        updates = work["uniform_updates"] if "uniform_updates" in work else ()
        if any(op["launch"] is not None or op["seconds"] is not None or op["demand"] is not None
               for op in (*work["operations"], *updates)):
            raise ValueError("source storage subset cannot assign launch, demand or duration")
        prepare_scenarios = (probe["prepare_decode_source_scenarios"]
                             if "prepare_decode_source_scenarios" in probe else ())
        if any(case["work"]["launch"] is not None or case["work"]["seconds"] is not None
               or case["work"]["demand"] is not None for case in prepare_scenarios):
            raise ValueError("prepare decode source scenarios cannot assign launch, demand or duration")
        slot_work = (probe["slot_mapping_source_scenario"]["work"]
                     if "slot_mapping_source_scenario" in probe else None)
        if slot_work is not None and any(slot_work[key] is not None for key in ("launch", "seconds", "demand")):
            raise ValueError("slot mapping source scenario cannot assign launch, demand or duration")
        refresh_plans = (probe["metadata_refresh_source_scenarios"]
                         if "metadata_refresh_source_scenarios" in probe else ())
        backend_plans = (probe["sparse_backend_metadata_source_constraints"]
                         if "sparse_backend_metadata_source_constraints" in probe else ())
        if any(event[key] is not None for plan in (*refresh_plans, *backend_plans) for event in plan["source_events"]
               for key in ("launch", "seconds", "demand")):
            raise ValueError("metadata refresh source events cannot assign launch, demand or duration")
        source_storage_records.append({"artifact": str(artifact), **probe})
    actuals = (
        "4 款本地绑定config可用于源码数学投影；V4来自既有H200 campaign，HF revision/checkpoint身份未验证；K2.7/K3/GLM5.3缺准确配置，DSpark draft字段与权重绑定缺失",
        "V4 QK、finish、eager短上下文与gather已有各自scoped复核；C128可见状态73788e5a、FFN gate/shared clamp3c354e6b、unpriced router stage0f4183d2已获正式scoped通过，保留原snapshot；inverse RoPE/padded scales30cb98ac已获正式3328/3328及6direct scoped通过；routed-core clamp40bd4b28/9e008651已获正式2496项及2direct scoped通过；Marlin W2 weight乘法d07fb71b已获正式1072项及2direct scoped通过；GPU-v2 storage42、metadata40、refresh110各自scoped通过；PP2归约位置92c3dfee正式52/52及2direct通过；f1479578的GLM PP1全TP/native数学图获独立scoped接受，route仍unpriced；新增finite CUDA route源程序保留条件与编译服务缺口，不继承实际provider/runtime接受；PP2默认partial传输/实际launch仍未闭合",
        "Kimi d7e4356a显式0.30图已分离BF16 A/B/C plain GateLinear与独立grouped sigmoid top8 producer，每阶段480 route unpriced；旧Kimi artifact保持原snapshot与历史0 unpriced标签，不证明当前完整定价；GLM V32 GateLinear已分离BF16 A/B与FP32 C，grouped routing仍unpriced；V4首层/大T DeepGEMM prenorm仍unpriced；实际backend未闭合，附加产物保留各阶段完整分母",
        "真实0.30 Scheduler缺合法NamespaceTool依赖；Kimi完整父模型skip/loaded residency未绑定，正常容量入口拒绝仅text_config账本；GLM BF16 TP1/PP1和TP8/PP1仅target routed权重已超过整卡HBM，off/native源码图均不具备正常容量；TP8/PP2未被该下界排除，但显式38,40+NVLink source候选保留未计价routing，TP-partial hidden默认shard/all-gather不证明数值正确，实际传输override/编译变换未绑定；Scheduler/native-on仍拒绝，物理拓扑/正常allocator/QPS16未验收；V4 normal residency未闭合",
        "GLM V32 有TP1/TP8原生on静态事实和GPU-v2源代码存储/状态子集：一次hidden gather及temporary/front copy，中间step更新ids/hidden/positions/seq_lens，final step只存draft token；prepare_decode及CP1 slot独立场景保留typed状态/完整capacity padding/禁用与SIMD tail读取，eager fused/ordinary事件按显式group capability声明选择并保留依赖；均未计价且待独立复核，实际backend capability未绑定；完整GPU-v2、INT32 library lowering及sampler/slot/runtime仍unsupported/unbound，V4 DSpark未闭合",
        "3a2f725d 独立源码复核 1904 fusion calls 无失配；独立 reference 内存重评分 5024/5024；正式文件依各自 snapshot 绑定，全目标 independent_acceptance=false"
        + ("；ordinary/rejection sampler 物化、条件 INT64 hidden projection 与场景拒绝边界已获正式 source-only 验收；routing/native 状态服务、完整 sampler、实际 metadata/provider/runtime/正常容量及完整 alignment 仍未通过"
           if any(Path(record["reference_artifact"]).name == "target_sampling_materialization_review.json"
                  for record in formal_source_reviews) else ""),
        f"维护测试：{tests_summary}；直接变更/失败项与依赖复查：{recheck_summary}；Kernel matrix 结果见日志；Serving QPS16 和原生 on runtime 验证尚缺；历史全测不冒充新生产提交全测",
    )
    gates = [{"gate": f"W{index}", "target": "GOAL.md Hard acceptance gates",
              "actual": actual, "status": "Fail"}
             for index, actual in enumerate(actuals, 1)]
    for filename, data in (
            ("operator_alignment.json", {"independent_acceptance": False,
                "source_contract": str(contract_path), "review": str(review_path),
                "review_acceptance": review["independent_acceptance"], "records": alignment,
                "native_records": native_records,
                "independent_glm_capacity_evidence": glm_capacity_record,
                "formal_layer_comparison": {
                    "path": str(layer_comparison_path),
                    "totals": layer_comparison["totals"],
                    "snapshot_binding": "each record retains its own reviewed snapshot; no inheritance across source deltas",
                },
                "current_v4_compressor_replay": {"path": str(v4_rescore_path),
                                                "totals": v4_rescore["totals"]},
                "current_layer_shape_replay": {"path": str(layer_rescore_path),
                                               "totals": layer_rescore["totals"]},
                "formal_native_comparison": {
                    "path": str(native_comparison_path),
                    "totals": native_comparison["totals"],
                    "snapshot_binding": "each record retains its own reviewed snapshot; no inheritance across source deltas",
                }}),
            ("gate.json", {"status": "Goal 未完成 — checkpoint", "gates": gates})):
        (output / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    handoff = {
        "status": "Goal 未完成 — checkpoint", "report_generation_snapshot": _snapshot(),
        "static_evidence": alignment,
        "current_static_evidence": current_records,
        "native_static_evidence": native_records,
        "source_storage_evidence": source_storage_records,
        "weight_diagnostic_evidence": weight_records,
        "source_graph_evidence": graph_records,
        "independent_glm_capacity_evidence": glm_capacity_record,
        "formal_source_delta_reviews": formal_source_reviews,
        "validation": {
            "tests_log": str(tests_log), "tests_terminal_summary": tests_summary,
            "recheck_log": None if recheck_log is None else str(recheck_log),
            "recheck_terminal_summary": recheck_summary,
            "kernel_results": str(kernel_results),
            "kernel_status": {key: value["status"] for key, value in kernel_matrix["kernel_matrix"].items()},
            "scope": "historical snapshots in log filenames retained; recheck runs only failed items and direct dependencies",
        },
        "failed_gates": [gate["gate"] for gate in gates],
        "blockers": [actuals[index] for index in (0, 3, 4)],
        "next_steps": [
            "Retain formal residual-norm and target sampler materialization receipts at their original snapshots, including dirty and source-only qualifications. Reuse all prior scopes without reopening unchanged findings; complete sampler/native/runtime and full W6 remain unaccepted.",
            "Preserve original vLLM0.30 PP baseline and source_projection_only. Source rank math, receiver reduction and BF16 payloads do not prove TP-partial preservation in GPUWorker's default transport. Do not assume an upstream patch/override or change wire bytes to claim acceptance; normal Serving remains fail-closed.",
            "Known typed materializations already have logical Roofline pricing without exact launches. Ordinary sampling closes only its mandatory BF16-to-FP32 stage. The explicit metadata-present raw rejection scenario additionally preserves bonus/target INT32 selectors, two distinct bonus casts and target FP32 clone. Optional outer logprobs, processors, sampling, rejection and output-ID conversion remain uncovered; conditional prefill index binding is not actual Scheduler evidence. TP-sharded embedding point reads need real valid IDs or per-rank owned rows; pointer alignment/grid/block are separate launch facts.",
            "Grouped routing retains source math and conditional CUDA12.2 SM90 EX2/RCP/shared-scratch evidence, plus finite-precision branch limitations. Point service pricing still needs defensible semantics for transcendental/packed64 work; do not invent a provider, infer instruction counts from static sites, use production durations or tune parameters.",
            "Keep precise external blockers with the main agent: missing accurate configs/DSpark assets, compatible NamespaceTool SDK and normal residency/allocator facts. Reuse bounded local searches and successful local Torch2.11 compilation evidence. Do not repeat unchanged asset/environment audits; continue on concrete source deltas, new inputs or an approved implementation path.",
        ],
        "push_status": "not pushed; no remote changes",
        "reproduce_checkpoint_command": shlex.join([
            sys.executable, "scripts/evaluation/serving/report_operator_support_checkpoint.py",
            "--output", str(output), "--reference", str(reference),
            "--tests-log", str(tests_log),
            *(["--recheck-log", str(recheck_log)] if recheck_log is not None else []),
            "--kernel-results", str(kernel_results)]),
    }
    if current_probes:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in current_probes
            for argument in ("--current-probe", str(artifact))])
    if source_storage_probes:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in source_storage_probes
            for argument in ("--source-storage-probe", str(artifact))])
    if weight_diagnostics:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in weight_diagnostics
            for argument in ("--weight-diagnostic", str(artifact))])
    if source_graph_probes:
        handoff["reproduce_checkpoint_command"] += " " + shlex.join([
            argument for artifact in source_graph_probes
            for argument in ("--source-graph-probe", str(artifact))])
    (output / "cli_handoff.json").write_text(json.dumps(handoff, indent=2) + "\n", encoding="utf-8")
    status = {
        "status": "goal_checkpoint", "goal_complete": False,
        "commit": handoff["report_generation_snapshot"]["commit"],
        "snapshot": handoff["report_generation_snapshot"],
        "working_on": "W1-W7 remain incomplete. Existing variants now preserve GLM/Kimi residual norm reads/writes and memo semantics. Explicit ordinary BF16-to-FP32 and metadata-present raw rejection sampler materializations have logical Roofline pricing without launches. Rejection bonus/target INT32 selectors, independent casts and FP32 clone retain distinct roles; neither scenario covers complete sampling or actual metadata. Source scenarios, prior formal receipts, actual provider/compiler/runtime, PP transport and normal capacity retain separate acceptance. Main owns linear integration; unchanged asset/environment audits are not a next action.",
        "failed_gates": handoff["failed_gates"],
        "validation": handoff["validation"],
        "current_static_evidence": current_records,
        "gate_checkpoint": gates,
        "source_storage_evidence": source_storage_records,
        "weight_diagnostic_evidence": weight_records,
        "source_graph_evidence": graph_records,
        "independent_glm_capacity_evidence": glm_capacity_record,
        "formal_source_delta_reviews": formal_source_reviews,
        "external_inputs_missing": ["actual H200 provider/capability and normal memory/reserve evidence", "V4 DSpark proposer/native runtime and dynamic state/sampling bindings"],
        "limitations": [actuals[index] for index in (1, 2, 3, 4, 5)],
        "ongoing_commands": [], "writes_stopped": False,
        "next_steps": handoff["next_steps"],
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    text = ["# Goal 未完成 — checkpoint", "",
        "目标仍为七模型 H200 + vLLM 0.30、纯文本、eager，共同 MTP-off 和原生 MTP-on。以下仅为本地离线静态事实，无 GPU 执行；priced 不代表与真实源码发射已对齐。", "",
        f"GPU-v2 source storage-only证据：{[record['artifact'] for record in source_storage_records]}。它们只记录producer/temporary/copy/feedback角色、typed logical bytes及依赖，launch/demand/seconds为None；不进入完整model calls分母，不代表INT32 public lowering或正常native/Serving已支持。新prepare_decode_source_scenarios是单独声明的状态场景，不是同文件scheduled_requests的实际metadata：depth1在producer之前return；其余先target seq减rejected，采样key总加1，仅advance=True推进并饱和positions/seq_lens；extra CTA按max_num_requests完整写query_start_loc和seq_lens尾部zeros，eager也保留capacity writes。prepare读取DraftModelSpeculator的INT64 draft_tokens，不等同sampler返回dtype；拒绝计数dtype为显式场景，真实producer、Triton warps、slot/page/cache与runtime仍未绑定；状态/role子集已在43bb0de3获正式scoped复核，实际native/runtime未验收。", "",
        f"本次 checkpoint 生成 snapshot：{handoff['report_generation_snapshot']['commit']}；Goal未完成，writes_stopped=false。平台Goal状态由调度器单独管理，本报告不推断active/usageLimited。下面附加证据各自绑定其生成 snapshot，不替换历史产物。", "",
        "slot_mapping_source_scenario是显式CP1请求/block-table场景，不是上述scheduled_requests的真实allocator输入。INT64 slot写覆盖group×完整capacity，返回num_tokens_padded切片不能删掉tail写；mapping_enabled=False仍读block0再写PAD_ID；1024元素SIMD尾部的position mask=0而block-number load无有效行mask，另列尾部block0源对象。group共享输入只按distinct object计，未推断transaction/DRAM/local服务；layer slot views不新增copy，slot映射与model KV insert、rejected rollback、page ownership分别保留。zero group没有kernel，CP>1显式unsupported，实际Triton/metadata/group/lifecycle尚未绑定。正式reference已确认d6bf2ca4终止轮native拒绝及off控制2direct，保留该snapshot；prepare/slot子集已获43bb0de3的40/40与6direct scoped验收，保留该snapshot。", "",
        "metadata_refresh_source_scenarios记录GLM advancing、非dummy、NONE/eager的uniform循环控制流；capability flags为显式源码场景，actual_attention_backend_binding=None，不用eager推断fused分支。所有attention group支持update时，首次构建metadata，后续非final input update之后刷新slots并依次update各group；存在不支持的group时，每一步在forward之前重建slots/layer views/metadata。depth1提前返回，不产生uniform事件；final update只写draft token，不产生下一轮slot刷新。current-step fill、model forward、sampler、input update、slot及host/view/backend events均不赋launch/demand/seconds，不作为kernel calls数或完整native定价，其他EPLB/DP/prefill/cache生命周期仍在边界之外。", "",
        "sparse_backend_metadata_source_constraints进一步约束源码分支：CUDA SM90 sparse候选FLASH_ATTN_MLA_SPARSE、FLASHMLA_SPARSE、FLASHINFER_MLA_SPARSE_SM90的builder均经SparseMLACommonMetadataBuilder继承AttentionMetadataBuilder.supports_draft_decode_metadata_update=False，未覆盖该值。只要一个draft attention group使用其中的未修改0.30 builder，advancing loop必选ordinary；无需假定其余group能力或总group数。它们是显式backend源码条件，backend_selection_observed=False、actual_group_membership=None，不选择或冒充已观测backend，不改变正常GPU-v2 unsupported边界。全True的generic事件场景保留为共享循环谓词检查，不能当作SM90 GLM分支证据；V4 SparseSWA及其True能力不用于GLM。该源码能力约束已在3cd6235a获正式scoped复核，实际backend选择仍未绑定。", "",
        "正式reference新增GPU-v2 metadata40/40及6direct，绑定43bb0de3；refresh110/110与generic3direct/sparse1direct分别绑定ae4fd782/3cd6235a或直接日志内的依赖复用说明。源码sparse builder False约束也已独立核实。它们接受typed roles、source host/readiness事件，不接受CUDA数据边、实际kernel数、stream/overlap、backend选择或native pricing；source updater之间的host顺序不证明物理数据依赖。原产物snapshot保留，PP2归约位置92c3dfee已获52/52及2direct scoped通过；后续producer覆盖变更不继承整树接受。", "",
        "Kimi-K2.6的既有d07fb71b H200 TP8/PP1/EP8静态产物可复用：prefill/decode1/decode2/mixed共3681/3622/3562/3864 modeled calls，未计价0/0/0/0；这不升级为实际发射/Scheduler验收。独立capacity routed下界与既有owner TP8静态权重均未排除H200 TP8/PP1，但reserve/profile/allocator未绑定，不能导出observed available-KV或正常blocks。现有Kimi TP8/PP2/EP8生产row只绑定A800；本轮未为H200另扩PP2，先保留较小TP8/PP1候选。兼容0.30 Scheduler的OpenAI NamespaceTool SDK仍缺，不能下载、stub或用0.11/debug blocks补齐W4；准确K2.7/K3/GLM5.3与DSpark配置仍是外部输入缺口。", "",
        "正常容量边界06c7398b已独立scoped通过：完整KimiK25 config的text_config只提供text子模型参数，未证明父模型vision_tower/mm_projector跳过或offload；无pixel_values、普通image/video limit0不等于vision_chunk skip，源码该自定义modality默认上限999。resolve_analytical_kv_capacity_plan在residency/native allocator前明确拒绝非'$' extraction，既有root完整模型路径保留；source-only text数学诊断仍可用。8573f835新增Kimi active outer构造参数账本，335唯一路径已获独立逐key/逐byte源码复核；该scope排除buffers/post-load/text/KV/reserves，实际mode/skip/offload/loader仍未绑定，因此不新增假设bool放行正常容量。direct边界与root容量/纯PP依赖3passed/3.67s不等于真实0.30正常容量通过。", "",
        "| 当前附加静态产物 | Snapshot | prefill/decode1/decode2/mixed |",
        "|---|---|---|", *current_rows, "",
        "V4 附加产物若声明 v4_eager_metadata_source=vllm_0_30_eager_dict_all_valid，表示明确的源码场景：eager、字典metadata、scheduled rows全部有效，非真实runner观测。C4短context以max_seq_len//4<=TOPK分派，保留projection/compressors、只填一次完整INT32 topk buffer，省去indexer Q/scoring。prefill原cache union只gather一次，写单BF16 workspace；query K/V role-use与该写足迹分开，物理读取重用及dequant/helper服务仍未绑定。默认未绑定场景保留原scoring投影；这不是INT32 materialized gather或完整native验收。", "",
        "| 模型 | prefill/decode1/decode2/mixed | 范围与缺口 |", "|---|---|---|", *rows, "",
        "| 原生 on 事实 | prefill/decode1/decode2/mixed | 范围与缺口 |", "|---|---|---|", *native_rows, "",
        "| Gate | Target | Actual | Status |", "|---|---|---|---|",
        *(f"| {gate['gate']} | 见 GOAL hard gate | {gate['actual']} | Fail |" for gate in gates), "",
        "V4 mHC 已拆出 broadcast prenorm、post/prenorm、BIG_FUSE、post 和 head。T≤32 的 regular post/pre 是两阶段，T>32 是三阶段；首层 broadcast 投影输入为 H，不能计成 4H 普通 GEMM。缺 DeepGEMM 绑定的调用保存完整依赖且 seconds=null；未计价项不能累加为完整端到端时延。已计价项是源码数学操作与 logical unique bytes，编译器归约、重复读和真实 DRAM 驻留仍需证据。Midlayer 明确拒绝本轮 mHC stages。", "",
        "新增本地源码输入：[有界 DeepGEMM 定位记录](deepgemm_bounded_source_inputs.json)。/share_data/yangxinhao/miniconda_yxh/envs/sglang_v2/lib/python3.12/site-packages/deep_gemm/VERSION 为 0.1.2，base 包为 0.1.0；这不是已选择的 vLLM provider 版本。sglang_v2 包的 include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh、common/math.cuh 和 mma/sm90.cuh 支持条件 source program：BF16 A 从 shared fragment 转 FP32 registers，FP32 B/C，TF32 WGMMA，M64/K64、128 math threads，24 输出在完整 single-N-tile 下至少 N32。完整 grid 的 padded fragment 数、平方/累加及每 CTA 512 次 subwarp additions 单独输出；数学 FP32 Torch 参考、TF32 source program 与实际执行分别保留。", "",
        "Opaque TMA descriptors 的 bounds/OOB fill、padded backing、selected N、TMA/total block threads 与 stages 仍未绑定。只读 binary 符号有 tf32_hc_prenorm_gemm host/FFI 入口，但构建路径 /deepgemm/csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp 没有对应本地文件。vLLM utils/deep_gemm.py 优先 external 再 vendored；现有 header 和五参数接口不证明版本/host config/compiler/provider 等价。未 import/load、使用 GPU、下载或修改共享环境；conditional_sm90_source_program 的 demand/seconds 为 None，prenorm 仍 unsupported。直接测试覆盖平方、累加及 synthetic M/K padding，正式 reference 对新 snapshot 的复核仍待交接。", "",
        "进一步的有界 host binary 证据：[原始命令及局部静态解读](deepgemm_sm90_host_binary_probe/result.json)。只读 reviewer 已核对本地 0.1.2 ELF 的完整 hyperconnection host 函数、launch-config helper 和两段 rodata：SM90 分支检查 N≤32、N%8=0、K%64=0；JIT 参数为 M64/K64、math128/TMA128；grid.x=ceil(M/64)*splits，block=(256,1,1)，launch dynamic shared 为 232448B。对 N24 的算术跟踪得到 blockN32、swizzle128、template-used shared204992B、stages12；template 使用量与 launch reserve 分别保留。这是局部 binary ABI/JIT/launch 构造复核，尚非正式 operator-reference 验收或实际发射；实际 vLLM 配置未绑定，launch/demand/seconds 继续 None。Header 本身不绑定 TMA descriptor fill；选中的 vLLM provider、缺失源码 provenance、compiler/device、occupancy、数值等价和时间仍未闭合。当前 H200 HWSpec Tensor Core variants 只有 fp16/bf16/fp8，不能借 BF16 速率计价 TF32；未修改硬件速率。", "",
        "同一 local binary 的 descriptor 续项也经限定只读复核：helper 原样保存 global bounds，并以 oobFill0/NONE 选择 CUDA 文档的零填充分支。A 编码 global[K,M]/box[64,64]；B 为 FP32 storage、TFLOAT32 tensor-map dtype，N24 时 global[K,24]/box[32,32]。原 B box[64,32] 的候选解读已纠正：helper 用 swizzle128/4B 覆盖 box 的 inner extent，不能直接套 caller 的 K64 参数。tma_copy.cuh 在默认 multicast1 下每 K64 用一个 A atom64 和两个 B atom32；新增 tma_load_source_calls 保存该源码调用计数，与编译指令、DRAM transactions/bytes 分开且不赋时延。M-tail 和 N24→N32 零填充仅在这个 local descriptor 条件下成立；泛化 header 的 descriptor 字段仍 None，synthetic K32 只检查 device-template 算术，不是 K%64 host ABI 的可达输入。D 的 split1/multi-split descriptor 形状已核对，device-store OOB 行为仍未声明。", "",
        "V4 sparse QK 已按独立发现修正为包含 RoPE 的 head_dim=512，Serving 与 Kernel evaluation 共用该 shape 底线。GLM 原生输入使用两个独立 RMSNorm 加 concat 的源码工作量；TP1 默认完整词表时合并 embedding gather，TP8 默认分片词表先 gather/all-reduce。Triton block 使用本地 3.4.0 CUDAOptions 默认值的显式解析投影，不能冒充已经绑定真实 0.30 编译器。Native artifacts 保留接受/拒绝数组和所有未闭合 source gaps；不是正常容量或真实 Scheduler 的完成证据。", "",
        "GLM 显式 0.30 路径在 sparse MHA/MLA 分派之前对全部 T 行做一次 Q absorption，prefill 和 mixed 均保留该调用，decode 不重复；legacy 路径保持原行数。源码版本纳入层模板 memo 签名。调用沿用 decode-q-down-proj ID 以便与历史比较定位，all_model_rows 属性明确其范围。正式 layer_stack comparison 的旧 snapshot 仍保留，不把局部实现或新 artifacts 当作 5024 项独立验收已经通过。", "",
        "GLM target 使用 dsv32_fused_norm_rope 与 dsv32_fused_q 两个 Roofline-only 源码投影，移除显式 0.30 路径中被它们吸收的独立 norm/rotary/cache 发射。前者保存两个 RMSNorm、index LayerNorm/RoPE/FP8 cache、BF16 materialized K 与 MLA cache，以及 full-indexer topk sentinel reset；shared 分支不清 topk。后者保存 BF16 Query RoPE、FP8 index Query 和 FP32 scaled index weights。H200 SM90 不满足 CuTeDSL SM100 条件；fused_q 的 Triton num_warps=1 有源码依据，norm 的 block128 仍是本地 Triton3.4 默认投影，未绑定 vLLM0.30 所需 Triton3.7.1。index LayerNorm gamma/beta 为显式 FP32，fused_q literal source ledger 包括独立 4B FP32 Q scale；两融合保留 index_rope_interleave=True 和会改变分支的 BF16 cache/quantize_mqa=False contract。全部 padded 行按 valid local slot 投影，真实 slot/owner/scoring skip、编译器和 runtime 仍未验收，unique bytes 不是实测 DRAM/cache hit。", "",
        f"当前 artifact 对正式 reference 既有期望的可复现 replay：{layer_rescore['totals']}，见 layer_stack_rescore.json；未添加期望、未改变分母。独立 reviewer 对 92fe1b04 确认原 156 项关闭，又对 3a2f725d 的 1904 fusion calls 和正式 comparison 脚本内存重评分复核通过。此 scoped 结论与正式 reference 文件更新、完整 source launch/runtime 验收分别记录。", "",
        f"V4 compressor 独立 reference 已增加与投影一一对应的 state obligations；原 612 项 projection 与新 612 项 state 共计 1224。现有独立期望重评分：{v4_rescore['totals']}，见 v4_compressor_rescore.json。BF16 A/B 保持 2B，GEMM output_dtype=fp32 明确 C 为 4B；同型默认 output_dtype=None 在 schema 边界规范化。混合输出仅 Roofline，actual injected launch、Midlayer 与 PipeWeave 拒绝；只计已知 global A/B/C 与 Tensor FLOPs，局部 SMEM/epilogue 服务未计价，不把其缺失写成 observed zero。下游 state update 读取 FP32 projection 和 FP32 APE（score半宽），读取 int64 position/slot，写 FP32 states；APE 跨 token reuse、slot地址/control、实际有效槽仍保留限制。首层 DeepGEMM unpriced 不因这些修复消失。", "",
        f"正式 layer_stack 检查：{layer_comparison['totals']}；native 局部检查：{native_comparison['totals']}。两者按原比较文件逐项保留各自 snapshot；当前产物不自动继承不同 snapshot 的通过结果。", "",
        "V4 新FFN候选依据正式 v4_ffn_reference.json 和上游 nvidia/model.py GateLinear(out_dtype=float32)、MoERunner gate调用、SiluAndMulWithClamp：hash只替换ID搜索，全部层保留gate投影；BF16 A/B、FP32 logits在plain mixed-output GEMM计价；shared activation保留swiglu_limit，gate上界clip、up双侧clip后SiLU/mul。clamp只支持Roofline逻辑数学/unique bytes，Midlayer/PipeWeave拒绝。独立router ID/weight producer作为unpriced节点接入，附源码数学及typed role reads；其provider、modular/monolithic融合位置和真实GateLinear/activation backend、编译launch仍未闭合，不将这些缺口计为observed zero；正式验收按新snapshot独立进行。C128 attention在显式eager metadata源码场景按每query的(position+1)//128计状态，默认仍为candidate upper bound；旧finish已通过scope不重开。", "",
        "正式 reference 已确认3c354e6b FFN 2496/2496及3direct+2clamp，73788e5a C128可见状态3direct scoped通过；新提交不自动继承整树接受。output projection 新候选依据 common/ops/fused_inv_rope_fp8_quant.py：每个local head执行inverse RoPE；每个live quant group计log2与exp2，FP32 scale buffer按16/4=4行对齐，padding行显式store zero；H200 grid=(aligned_T,local_heads),block32。只按live行计输入、RoPE/quant数学和FP8输出，按aligned行计FP32 scale写入。保留unique/role存储与compiler instructions、partner/repeated缓存读、实际交易和DeepGEMM einsum服务的区别；Midlayer拒绝未绑定的精细模型。", "",
        "正式 reference 对30cb98ac output projection 3328/3328及原4反例+2边界检查scoped通过；这不接受实际指令/访存/容量/native。正式 reference 对0f4183d2 unpriced route producer的2496 source-role/dependency检查与2direct scoped通过，仍不接受实际provider或modular/monolithic放置。routed-core 新候选依据 quantization/mxfp4.py、fused_moe/oracle/mxfp4.py make_mxfp4_moe_quant_config 与 fused_moe/activation.py：swiglu_limit传到gemm1_clamp_limit，SILU在activation前使用gate上界和up双侧clamp，alpha=1/beta=0。workload/lowering保留限制；Roofline按live routed rows计三个clamp、SiLU/mul普通数学及exp/reciprocal，维持既有BF16 materialization和Tensor GEMM工作；padding GEMM行不推导额外activation工作。默认None兼容既有unclamped/INT4聚合，显式clamp仅MXFP4窗口，Midlayer/PipeWeave拒绝；实际backend/编译指令/local服务仍未绑定。", "",
        "正式 reference 已确认40bd4b28/9e008651 routed-core clamp 2496/2496及2direct，d07fb71b Marlin weighted-output1072/1072及2direct scoped通过；不自动继承后续整树或runtime接受。Marlin 默认输出加权分支的静态数学缺口另行修正：marlin_moe.py W2将mul_topk_weights设为True，marlin_template.h store_result对valid row的BF16输出执行__hmul2，再由moe_sum/torch.sum归约top-k。现按live routed rows×hidden计selected-weight乘法，沿既有聚合ALU逻辑操作口径计数；不按padded GEMM行计该项，不将标量数学计数当packed指令吞吐或实际backend绑定。Tensor FLOPs和既有存储角色不变，INT4/MXFP4共享这一源码程序分支。", "",
        f"GLM 独立容量下界见 {glm_capacity_path}：只计BF16 target routed参数，TP1/PP1和TP8/PP1已超过配置H200全容量，draft/其他权重/cache/reserve只会增加占用。因此f1479578的TP8/PP1 off/native全rank源码图均为数学证据，不能提升为正常Serving case。下表TP8/PP2对应reference默认39,39分段（routed层36,39），未被该下界排除；该数值不是显式38,40源码候选的容量账本。完整post-load residency、传输与正常allocator仍未验收；下表直接读取独立reference，不复制生产或reference计算公式。", "",
        "| GLM BF16 layout | Routed layers/stage | Routed target lower bound/rank B | Configured HBM B | Excess B | Independent conclusion |",
        "|---|---|---:|---:|---:|---|",
        *[f"| TP{case['tp']}/PP{case['pp']} | {case['target_moe_layers_per_stage']} | {case['rank_routed_target_weight_lower_bound_bytes']} | {case['configured_H200_physical_bytes']} | {case['excess_bytes_if_positive']} | {case['conclusion']} |" for case in glm_capacity["cases"]], "",
        "GLM f1479578的TP8/PP1 off/native已获只读独立源码数学复核：all-rank actor、EP-off全专家I/TP、same-rank两跳topk reuse alias、单个group collective和memo replay通过。formal glm_native_tp_direct.log为2passed4.13s，保留该scope/snapshot；不接受正常容量、完整alignment或实际provider/compiler/runtime。3e6ca754的conditional finite CUDA source program也获只读独立scoped接受：sigmoid是tanhf表达式，T<=1024有两份FP32 shared score写入，T>1024另读selected bias并减法恢复；global role rereads与unique footprint分开，uint64 local-sort比较和warp reduction请求不转换为硬件指令或速率。非finite数据分支、实际tanhf/compare/reduction编译服务尚缺证据，demand/seconds仍None，route仍unpriced，不把源码计数写成Roofline端到端通过。", "",
        "Kimi d7e4356a的显式vLLM0.30 Gate/routing已获只读独立scoped复核。准确config不要求FP32 router输出，GateLinear.out_dtype=None走BF16 F.linear；投影以plain GEMM计价，独立kimi_moe_route读BF16 logits/FP32 bias，输出FP32 weights/INT32 IDs，同rank gate→route→已生成INT4 core依赖保留。CUDA shim直接透传BF16 scores，preprocess内部升FP32；torch branch的BF16 scoring/weights与FP32 bias选择/最后export分开记录，不造独立FP32 logits copy。384/8 CUDA tier的conditional source launch为T<=1024时grid=T/block256/shared3072B，否则grid=min(ceil(T/8),2048)/block256/shared0；均不赋给KernelSpec。每phase每rank60 routes，共480 unpriced。T8四phase calls从66d9ab0f的8420/8788/8728/9884变为8900/9268/9208/10364，仅新增480 required producer；两边Gate SFU均0，不声称减少多计SFU。T127同样480 routes；编译/provider/modular或monolithic/PDL/runtime/nonfinite/物理流量未绑定，完整Roofline仍Fail。直接依赖在kimi_v30_routing_failed_recheck.log有6passed，失败项最终在kimi_v30_routing_dtype_failed_recheck.log为3passed5.74s，原失败保留；旧source/参数账本复核保持各自snapshot。", "",
        "V4 mHC 545ea688的最终RMS fragment归约已获独立source-work复核：regular/broadcast的TileLang源码均在逐位置平方/局部累加后，归约gcd(1024,H)个FP32元素；沿既有Sinkhorn aggregate ALU口径补T*(gcd(1024,H)-1)逻辑加法。已有collapse、平方累加、RMSNorm两乘法和Sinkhorn轮次保持；compiler padded/shuffle/reduction instructions仍未绑定。原5positive直接依赖通过，2个fragment范围新项失败仅因test replace未同步elements alias；修正后只重跑2项，v4_mhc_rms_fragment_failed_recheck.log为2passed2.57s。生产数学变更触发545ea688五卡离线Kernel matrix，全部PASS；canonical矩阵不替代新mHC的硬件/完整Serving接受。首层mhc_pre_broadcast_tilelang强制use_tilelang_fallback=False，真实DeepGEMM实现缺失，不能使用torch参考或普通GEMM launch/速率替代。", "",
        "同场景8eab4256→545ea688 V4 T8 modular源码产物比较：Flash四phase的344个norms增加ALU 2815296/351912/351912/3167208；Pro四phase的976个norms增加ALU 7987584/998448/998448/8986032。两侧calls、FMA/SFU/逻辑bytes均一致；Flash仍4个、Pro仍8个DeepGEMM prenorm unpriced，不是完整静态端到端通过。实际launch/compiler/provider/normal容量/DSpark仍缺绑定，未使用duration或误差决定归约公式。", "",
        "V4 routed loader payload见deepseek-v4-flash_routed_loader_payload.json与deepseek-v4-pro_routed_loader_payload.json，各自保留生成snapshot。这是显式non-MegaMoE/EP-off/PP1且Marlin H256/I128已经对齐的源码场景：每rank保留全部experts，I按TP分片；packed uint8为每两个FP4值1B，UE8M0 scales每32个输入值1B。它只统计Mxfp4MoEMethod.create_weights的routed payload，不能替代post-load residency。Pro TP1 packed部分已超过H200全容量；FlashTP4与ProTP8未被这一部分排除，但不意味着正常容量通过。非routed权重、转换后scales/padding/alias/workspace、DSpark draft、runtime reserve/profile/allocator与实际provider仍未闭合；生产model_rank_residencies继续拒绝V4，未导出available-KV或blocks。", "",
        "V4 显式0.30 EP-off源码分支使用非MegaMoE FusedMoEFactory：保留TP，ep_rank=0，每rank拥有全部experts的I/TP分片；shared MLP也按TP分片。所有已建模rank-local输入、compressor/indexer、attention/head在每rank实例化，通信保持单个group调用。外层V4的n_local_experts元数据、实际loaded weights/provider及runtime仍未绑定。层模板签名区分layer0 broadcast/后续mhc_post_pre和hash/ordinary routing；两次memo构图与未缓存图的nodes/edges一致。该memo修复与静态source metadata场景差异分别记录：旧d07 T8声明eager valid metadata，新default场景为None；C4 default保留indexer-wq-b和phase-specific score-topk，eager短上下文只fill一次。跨场景不能把call数差额归因于memo，也不能用不同metadata声明作同口径对比。附加产物同时保留两个场景及各自snapshot，route producer和first broadcast prenorm仍显式unpriced。", "",
        "当前V4源码来源记录使用0.30真实的attn/ffn注册模块、compressor.fused_wkv_wgate权重和DeepseekV4Attention/Indexer/MLP调用点；mHC指向实际TileLang函数。private层模板权重类型明确routed=mxfp4、shared=fp8。该改动只修正来源标签和memo身份，不新增公式/速率/launch事实；旧f24静态产物保留旧snapshot及旧标签，不自动成为当前源码绑定验收。源码名称存在不证明实际实例、checkpoint权重地址、loader映射、compiler或backend已绑定。", "",
        "GLM BF16 TP8/PP2 target-off source候选使用显式VLLM_PP_LAYER_PARTITION=38,40；默认39,39在shared-indexer层开始第二段，guard拒绝没有本地full indexer的输入。vLLM0.30 V32 forward传递hidden_states/residual；PP scatter/send/receive/all-gather按两个BF16张量记账。新候选移除stage0 layer37末尾的MoE all-reduce，在接收端stage1 layer38 input norm之前归约hidden，norm另等residual gather；stage-tail block显式构建，不能memo/repeat旧归约尾部。新EP1 MoE图按FusedMoEParallelConfig的EP-off/TP规则遍历tensor group：每TP rank保留全部experts和相同router计数，intermediate为I/TP，ep_rank仍为0；shared MLP按TP分片，每lane的PP readiness连接对应partial producer。全部已建模的非MoE rank-local源码数学也已逐TP rank实例化：复制indexer/projection与分片attention/dense/head保留各自local shape，topk alias与compute依赖指向本rank，collective保持单个group调用并等待各rank producer；具体artifact覆盖范围见下述snapshot自述，历史产物不改标签。实际CUDA producer/data edge/fusion/provider均未绑定；默认partial传输风险未消失。正式reference的默认传输风险保留：V32 MLP reduce_results=False保留TP-partial hidden，GPUWorker默认分片send/all-gather需要复制输入；不同rank partial不能这样保留，后续all-reduce不能恢复丢失坐标。修正归约顺序不证明传输数值正确；actual override/编译变换未绑定。scope仅供构图与静态emission，不授予Scheduler接受；PP2 native-on含terminal verification拒绝。NVLink是显式拓扑场景，不是已观测16卡机器；正常容量/reserve/cache与runtime仍未验证，全部W gates保持Fail。", "",
        *[line for scope in current_pipeline_scopes for line in (scope, "")],
        f"独立参考：{contract_path}；review：{review_path}。operator_alignment.json 只保存当前事实和分母，未自产 reference，也未声明独立验收通过。", "",
        "每个静态 JSON 自带精确 commit 与 dirty 状态；不同 snapshot 的产物保留原标签，不冒充当前 HEAD 结果。命令、测试、Kernel matrix 与外部缺口处置见 cli_handoff.json。不得用 0.11/debug-blocks Serving 替代 W4，也不得以 classic MTP 替代 DSpark。",
    ]
    v4_weight_records = [record for record in weight_records
                         if all(case["payload_scope"] is None for case in record["cases"])]
    kimi_weight_records = [record for record in weight_records
                           if all(case["payload_scope"] == "kimi_active_outer_registered_parameters"
                                  for case in record["cases"])]
    glm_weight_records = [record for record in weight_records
                          if all(case["payload_scope"] == "glm_parameter_source_projection"
                                 for case in record["cases"])]
    if len(v4_weight_records) + len(kimi_weight_records) + len(glm_weight_records) != len(weight_records):
        raise ValueError("constructor diagnostic cannot mix source payload scopes")
    if glm_weight_records:
        text.extend(["", "## GLM 源码参数账本（正常容量未验收）", "",
            "full indexer的k_norm使用vLLM LayerNorm，weight和bias固定FP32；38,40两段分别有11/10份，每TP rank相对旧BF16账本增加5632/5120B。默认独立draft vocabulary账本保留；显式vllm_0_30_0 BF16场景按legacy和GPU-v2源码共享MTP LM head，PP1另共享embedding，PP>1保留draft embedding。三个draft RMSNorm与eh_proj/decoder继续独立，输入embedding假设VLLM_REPLICATE_EMBED关闭。参数别名和低于HBM均不证明真实loader/native/normal KV接受；原PP2传输、reserve/profile/allocator/Scheduler缺口不变。", "",
            "| Artifact / snapshot | TP | PP | MTP inventory | Version | Stage ranges | Maximum parameter B/rank | Physical HBM B |",
            "|---|---:|---:|---|---|---|---:|---:|"])
        for record in glm_weight_records:
            for case in record["cases"]:
                ranges = sorted({(rank["layer_start"], rank["layer_end"])
                                 for rank in case["rank_residencies"]})
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {case['tp']} | {case['pp']} | {case['mtp_mode']} | {case['model_execution_version']} | {ranges} | {case['maximum_owner_weight_bytes']} | {case['physical_hbm_bytes']} |")
        text.extend(["", "89fe70ba修正affine存储；glm_indexer_affine_direct.log为4passed3.47s。fe8c6e8c新增显式版本API与诊断：glm_v30_vocab_alias_direct.log为3failed3passed3.41s，初始guard误用了仅V4有值的expert_dtype；改为既有GLM weight_dtype校验后，仅三个失败alias case在glm_v30_vocab_alias_failed_recheck.log中3passed3.40s，复用原三个正向直接依赖。reviewer另指出报告入口缺runtime-binding标志校验；现对true或缺字段fail-closed，glm_parameter_binding_boundary_direct.log为2passed3.42s。此次未改kernel demand/launch，复用545ea688的五硬件Kernel matrix；未重跑全维护测试或真实Scheduler/QPS16。只读reviewer已核实两条proposer共享源码；新候选正式reference仍待其独立snapshot复核，不继承整树接受。", ""])
    embedding_records = [record for record in current_records
                         if any(phase["source_vocab_embedding_calls"] for phase in record["phases"])]
    if embedding_records:
        text.extend(["", "## CUDA vocab embedding 来源分支（未计价）", "",
            "2172e36a修正BF16 GLM vLLM0.30 TP>1 target/native的embedding：CUDA且UnquantizedEmbeddingMethod必选VocabParallelEmbedding的fused分支；target、legacy proposer和GPU-v2均传INT32 IDs，kernel内部标量widen不分配INT64输入copy。91eefd21按同一源码复用于Kimi-K2.6与V4的既有H2000.30分支：Kimi compressed-tensors只target Linear，VocabParallelEmbedding无matching weight scheme；V4 Fp8Config.get_quant_method对input embedding返回None，均保持BF16未量化embedding。每rank只读其owner weight rows，但所有rank都写完整BF16[T,H] partial output（非owner写零），随后仍是一次group all-reduce。逻辑GATHER/embedding接口保留，special lowering产出独立VECTOR/vocab_parallel_embedding；不借generic gather的flat launch或materialized_output=False定价。TP1 native继续使用既有folded embed_eh_norm。", "",
            "源码vocab_parallel_embedding_kernels.cu固定block256，grid_x=min(ceil((row_bytes/vector_bytes)/256),256)、grid_y=min(T,4*SM_count)；pointer/row alignment选择16/8/4/2/1B vector。H6144 BF16的五个conditional grid_x为3/6/12/24/48。actual IDs、每rank owner count、指针对齐及compiler/allocator未绑定，KernelSpec仍unsupported、launch=None、demand/seconds=None；source work保留完整output、INT32 input和conditional weight-read范围，不将source role bytes写成实测DRAM或cache hits。", "",
            "| Artifact / snapshot | Embedding calls（prefill/decode1/decode2/mixed） | 必需 BF16 partial-output 写 B（同序） |",
            "|---|---|---|"])
        for record in embedding_records:
            counts = "/".join(str(phase["source_vocab_embedding_calls"]) for phase in record["phases"])
            writes = "/".join(str(phase["source_vocab_embedding_partial_output_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {writes} |")
        text.extend(["", "同场景比较f1479578 PP1 off/native与3e6ca754 PP2 off→2172e36a：所有call IDs/dependencies、非embedding predictions不变；每phase分别替换8/24/8个generic gather，未计价数600→608、616→640、600→608。源反例已由只读reviewer独立确认；新候选实现复核因reviewer工具额度中止，正式reference待独立复核，不能继承全范围接受。glm_vocab_embedding_source_direct.log为5passed6.95s，PP2直接依赖为1passed7.00s；2172e36a五硬件离线Kernel matrix全部PASS、36.7s。没有GPU执行、真实Scheduler/QPS16或新全维护测试。", ""])
        text.extend(["91eefd21同场景对比Kimi d7e4356a与V4 modular545ea688：所有call IDs/dependencies、非embedding predictions不变，三款每phase分别替换8/4/8个generic gather，未计价480→488、4→8、8→16。kimi_v4_vocab_embedding_direct.log为8passed7.64s，含Kimi三个phase、FlashTP4/ProTP8全rank及原routing直接依赖；routing的数学与原独立scope没有被改写。该提交只扩展Serving来源分派，predictor唯一变更是docstring，pricing helper/KernelSpec/launch公式未变，复用2172e36a的Kernel matrix；没有新全维护/真实Scheduler/QPS16/GPU验证。完整父模型residency、V4 post-load及DSpark缺口保留，正式新snapshot验收仍待主agent/reference。", ""])
    target_selection_records = [record for record in current_records
                                if any(phase["target_logits_selection_calls"] for phase in record["phases"])]
    if target_selection_records:
        text.extend(["", "## vLLM target logits 行选择与逻辑 Roofline 限定", "",
            "vLLM0.30 GPUModelRunner.execute_model在最后PP stage先执行hidden_states[logits_indices]，再compute_logits；这是独立materialized BF16输出，不能把LM head读取少数行等同于省去该producer。源码ordinary query_start_loc为INT32；spec decode的cu_num_tokens由NumPy默认INT64 cumsum形成，_calc_spec_decode_metadata/async_tensor_h2d保留INT64索引。显式eager/H2000.30模型路径按scheduled request尾部选择verified draft与bonus行，target选择包含待验draft行，不使用proposer拒绝后的retained-row规则。", "",
            "MTP-disabled源码场景使用ordinary INT32；有verified draft行的源码场景使用spec INT64；仅MTP-active且零verified不能绑定scheduler dict是否为空，索引dtype/bytes保留None与每row4–8B范围。分支是schedule source scenario，非真实SchedulerOutput观测。每rank只读选中BF16源行并写完整sample_hidden_states；PP2仅最后stage出现，norm/select/head为same-rank数据边。已知索引宽度可经统一predict(...,method=roofline)按两份BF16行加索引的逻辑角色字节计价，无需grid/block；未知宽度仍拒绝点值。launch仍None、execution_support仍unsupported，Midlayer/Oracle/PipeWeave及精细resource_input仍拒绝；address/control、实际DRAM/cache、provider/library/compiler和正常runtime未验收。新prediction.scope明确logical_roofline_only_no_launch_binding，不能使完整phase或W gates通过。历史产物保持原未计价状态。", "",
            "| Artifact / snapshot | Selection calls（prefill/decode1/decode2/mixed） | BF16 materialized 输出 B（同序） | Index source（同序） | Logical Roofline only calls（同序） |",
            "|---|---|---|---|---|"])
        for record in target_selection_records:
            phases = record["phases"]
            counts = "/".join(str(phase["target_logits_selection_calls"]) for phase in phases)
            output_bytes = "/".join(str(phase["target_logits_selection_output_bytes"]) for phase in phases)
            index_sources = "/".join(str(phase["target_logits_selection_index_sources"]) for phase in phases)
            roofline_counts = "/".join(str(phase["target_logits_logical_roofline_calls"]) for phase in phases)
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {output_bytes} | {index_sources} | {roofline_counts} |")
        text.extend(["", "第一阶段源码边界：GLM V32非末PP stage返回未归约hidden_states与residual；普通eager GPUWorker使用空all_gather_tensors，GroupCoordinator按numel可整除TP时选择分片send/receive/all-gather。接口允许逐张量False override，但GPUWorker的SP分支仅设置residual，且V32非末stage明确拒绝SP。缺失的最小传输事实是sender与receiver一致的hidden_states override，或已证明等价的归约位置变换；不能把接口存在写成实际选择。embedding每rank条件weight读依赖owned_token_rows；已知IDs/owner计数后其逻辑字节无需pointer alignment，当前不能把0..T范围或TP平均数写成精确读值。", ""])
        text.extend(["", "target_logits_selection_glm_direct.log为9passed7.11s，涵盖三个index storage分支、fail-closed边界、TP1/TP8 target/native依赖及memo；target_logits_selection_kimi_v4_direct.log为7passed7.58s，含Kimi三phase、Flash/Pro全rank producer及原routing直接依赖；target_logits_selection_pp2_direct.log为1passed6.93s，只在最终stage ranks8–15生成选择节点。新源码math只增加已知copy角色及dtype下界，没有production duration调参；新snapshot正式reference验收待主agent协调，旧scoped通过不自动继承。", ""])
    fused_embed_records = [record for record in current_records
                         if any(phase["target_fused_embed_norm_calls"] for phase in record["phases"])]
    if fused_embed_records:
        text.extend(["", "## GLM TP1 embedding 源码逻辑 Roofline", "",
            "更正64471903：GLM V32 target的has_full_vocab_on_rank在TP1/unquantized时为True，Model.forward直接调用fused_embed_norm，绕过VocabParallelEmbedding.forward。新target融合读取INT32 IDs、BF16权重行和首层gamma，输出BF16 residual与normalized attention input；逻辑operand bytes为4T+6TH+2H，gamma object与每CTA重复读取另作区分。无全局INT64转换、无独立layer0 input norm；明确source grid=(T,)及num_warps=min(32,max(4,next_power_of_2(H)//512))，并非本机runtime/compiler实测。完整有效词表IDs下无需TP owner分布；native TP1 draft维持独立fused_embed_eh_norm分支。64471903历史产物与局部复核只覆盖leaf embedding，不能作为GLM模型分派验收。", "",
            "| Artifact / snapshot | Target fused calls（prefill/decode1/decode2/mixed） |", "|---|---|"])
        for record in fused_embed_records:
            counts = "/".join(str(phase["target_fused_embed_norm_calls"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} |")
        text.append("")
    residual_norm_records = [record for record in current_records
                             if record["model"] in ("zai-org/GLM-5.2", "moonshotai/Kimi-K2.6")
                             and any(phase["residual_add_norm_calls"] for phase in record["phases"])]
    if residual_norm_records:
        text.extend(["", "## GLM / Kimi residual-add RMSNorm", "",
            "V32主模型后续input norm、每层post-attention norm和final norm均通过fused_allreduce_rms_norm消费hidden_states与residual，更新residual并产生normed输出；TP1直接调用norm(hidden,residual)，TP>1的实际flashinfer融合或explicit all-reduce fallback选择仍未绑定。cc0c4218改用已有add_rmsnorm并补memo语义键，不更改predictor/硬件速率。首层无residual分支维持plain或embedding融合，MTP decoder首次residual=None仍plain，MTP post-attention改add，既有head add不变。Kimi-K2.6 parent构造DeepseekV2ForCausalLM，V2后续input/post-attention/final也都消费residual；88f488aa以相同已有add variant修正，保留V2源码符号，不假定V32的all-reduce融合或改变Kimi routing/EP/embedding。下面只统计各原snapshot实际记录的variant与public logical bytes，旧产物不会自动继承新语义；PP preserving transport与完整phase未计价缺口继续保留。", "",
            "| Artifact / snapshot | add_rmsnorm calls（prefill/decode1/decode2/mixed） | Logical bytes（同序） |",
            "|---|---|---|"])
        for record in residual_norm_records:
            counts = "/".join(str(phase["residual_add_norm_calls"]) for phase in record["phases"])
            byte_counts = "/".join(str(phase["residual_add_norm_logical_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {byte_counts} |")
        text.append("")
    target_sampling_records = [record for record in current_records
                               if "target_sampling_source" in record
                               and record["target_sampling_source"] == "vllm_0_30_plain_sampler_bf16"]
    if target_sampling_records:
        text.extend(["", "## 普通 target sampler 的必需 FP32 物化", "",
            "显式 target_sampling_source=vllm_0_30_plain_sampler_bf16 只覆盖普通 Sampler.forward 在 apply_logits_processors 前的 logits.to(torch.float32)：读取 LogitsProcessor 已完成 TP gather 和原词表 slice 的 BF16 logits，写独立 FP32 logits，逻辑字节为 6×rows×vocab。每个末 PP stage 的 TP rank 都保留自身 producer；该投影不需要库 launch，不改变 execution_support=unsupported，也不声明转换指令、物理 DRAM 或后续 sampler 已闭合。可选 raw_logits/logprobs 产生的额外物化、penalty/temperature/top-k/top-p、采样与输出 ID 转换仍未覆盖；MTP rejection 的 bonus/target indexing、clone 和 cast 是不同源码程序，当前场景明确拒绝 speculative 请求。默认 model-only 调用方保持原行为；没有 provider、正常容量、真实 Scheduler 或 W gate 接受提升。", "",
            "| Artifact / snapshot | Cast calls（prefill/decode1/decode2/mixed） | Logical bytes（同序） |",
            "|---|---|---|"])
        for record in target_sampling_records:
            counts = "/".join(str(phase["target_sampler_cast_calls"]) for phase in record["phases"])
            byte_counts = "/".join(str(phase["target_sampler_cast_logical_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {byte_counts} |")
        text.append("")
    rejection_records = [record for record in current_records
                         if "target_sampling_source" in record
                         and record["target_sampling_source"] == "vllm_0_30_rejection_raw_logits_bf16"]
    if rejection_records:
        text.extend(["", "## Rejection sampler 的条件物化 producer", "",
            "显式 target_sampling_source=vllm_0_30_rejection_raw_logits_bf16 声明 SpecDecodeMetadata 存在、BF16 全词表 logits 和非 processed logprobs 分支；该声明不是实际 Scheduler metadata 或 sampler 配置观测。bonus 行数是完整 batch 请求数，target 行数是 sum(mtp_verified_tokens)，均不使用 accepted 数缩减。RejectionSampler 分别索引 BF16 bonus/target；bonus 传给 Sampler 的 max_num_logprobs=-1/raw_logits override 强制产生独立 raw FP32 cast，之后 working cast 仍读取同一 BF16 bonus，不能用 FP32 alias 省掉。target 独立 cast 后 clone FP32 working logits，保护 raw logits 免受后续原地修改。", "",
            "metadata 的 bonus/target 索引由 cumsum_dtype=np.int32 产生，selection 每行读取 4B 索引；模型先前的 hidden logits_indices 仍来自累计 scheduled tokens 的 INT64 数组，不能混用两种角色。在本显式 metadata-present 场景中，即使 verified=0，hidden selection 的 index source 也已声明为 spec INT64；默认未绑定场景继续保留 4–8B bounds。zero target rows 只省去没有逻辑读写的 target selection/cast/clone，不赋予库实现零 launch 结论。", "",
            "每个末 stage TP rank 的源码角色总量为 bonus_rows×(16×vocab+4) + target_rows×(18×vocab+4) bytes。此处只计 typed 物化；bonus sampling、processors/constraints、rejection、可选外层 logprobs、metadata H2D、library/compiler/provider、实际 stream 与正常容量仍未覆盖。所有新增调用 launch=None、execution_support=unsupported，只支持逻辑 Roofline；完整阶段仍 partial。", "",
            "| Artifact / snapshot | Stage counts（prefill/decode1/decode2/mixed） | Logical bytes（同序） |",
            "|---|---|---|"])
        for record in rejection_records:
            counts = "/".join(str(phase["rejection_materialization_stage_counts"]) for phase in record["phases"])
            byte_counts = "/".join(str(phase["rejection_materialization_logical_bytes"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {counts} | {byte_counts} |")
        text.append("")
    sampling_records = [record for record in current_records
                        if "mtp_draft_sampling_source" in record
                        and record["mtp_draft_sampling_source"] is not None]
    if sampling_records:
        text.extend(["", "## GLM legacy greedy 源码场景（部分逻辑 Roofline）", "",
            "显式mtp_draft_sampling_source=vllm_0_30_legacy_full_logits_greedy选择源码full-logits greedy分支：默认BF16 LM head输出与TP全词表gather后argmax产生INT64 IDs，仅推进下一轮之前.int()生成INT32 IDs。首轮按query-end减rejected选择INT32行索引，sample hidden与recurrent hidden即使共享源tensor也分别产生BF16 materialized输出，positions产生独立INT64输出；后续[:batch_size]是view，不能重复算indexing。推进前input_ids与hidden_states分别写入buffer front，head依赖sample输出，下一轮norm依赖hidden copy。", "",
            "7aa185a1的产物仅有argmax/cast；87c0977b增加hidden/positions selection与buffer copy；后续topk compaction产物按各自snapshot列出，不重写旧产物。legacy propose在depth1提前返回之前执行compact_topk_indices，启用model.index_share_for_mtp_iteration时完整INT32[B,index_topk]先gather到temporary、再copy到front，两份读和两份写均保留容量；next-pass shared indexer依赖front producer。深度1仍产生这两个topk stage，多轮才产生positions/recurrent hidden选择及input-ID/hidden buffer copy。关闭sharing才省去topk compaction。GPU-v2的depth>1存储scope保留原结论，不共用该control判断。", "",
            "首轮set_inputs_first_pass的no-extra-slots分支先INT32 IDs整体shift(T-1)行，再按INT32 token_indices覆盖每请求的INT32 next ID，再copy完整INT64 positions与BF16 target post-norm hidden。CPU/padded prepare_next_token_ids输出均为INT32；padded准备已有条件源码角色，target sampling与backup H2D服务仍未计价。T1 shift为空，省去该memory操作；scatter/positions/hidden仍保留。当前源码场景要求整个scheduled batch进入proposer，部分proposer subset显式拒绝；method=mtp、parallel_drafting=False、pass_hidden=True、无M-RoPE/aux-hidden/padded-disable/extra-slots均是明确场景条件，不证明实际SpeculativeConfig。", "",
            "首轮set_forward_context调用_get_slot_mapping并把common target INT64 slots完整copy到proposer独立buffer；capture-disabled、无额外输入槽场景不增加tail PAD fill。后续_get_slot_mapping无input argument，返回每draft层共享的buffer slice/dict views，不能重复计copy或为host/view调用造GPU kernel。slot buffer由推进metadata写回，first copy与后续views分开。draft_config在_create_draft_vllm_config中独立选择speculative attention backend（None时autoselect），明确不继承target backend；build_per_group_and_layer_attn_metadata对实际每group调用build_for_drafting，每层仅共享该group对象。base方法默认build(common_prefix_len=0,fast_build=True)，具体builder/backend可能产生额外GPU work；实际group/provider未绑定时这些不能标为observed zero或完整发射。", "",
            "target Sampler与RejectionSampler输出均为INT32。padded prepare-next按discard/validity读sampled/backup IDs，固定写INT32 next IDs与valid counts；未知数据分支保留条件读取bounds，不从时延或raw计数猜discard位。每row至少一个ID读取，源码global bytes整体下界13B、上界13+4*sampled_width B，和独立parent backup H2D4B角色分开；BLOCK_SIZE_TOKENS=next_power_of_2(width)，grid=B，warps未绑定，不赋launch。prepare-sample-indices区分ordinary query-end减一、padded拒绝行计算和metadata未绑定；padded cu-prefix当前/前一元素共享一个INT32对象，不能重复计unique footprint。此producer连接首轮scatter与所有首轮hidden/position/topk选择；输入counts及selected indices仍为schedule source assumptions。", "",
            "多轮推进在循环之前只执行一次seq_lens-=num_rejected；metadata未绑定时presence与三份INT32读/写bytes保留条件范围，padded明确存在时即使rejected值为0也保留操作。每推进轮eagle_step_slot_mapping_metadata读取INT64 position、INT32 block-ID/seq_len共16B/真实请求，写INT64 position/slot与INT32 seq_len共20B；dummy只写8B slot。首轮input position来自materialized selector，后续为output-buffer view，alias只改变unique object extent，不删read/write role。cap exceed时position=0、seq_len=1、slot=PAD，仍读block0；源状态helper只消费显式cap/block/table/values，拒绝缺参数/非法形状，actual_allocator_binding=False。生产artifact的nominal KV长度不绑定真实runtime cap，实际slot/page/control数据继续未验收。", "",
            "多轮最终torch.stack(draft_token_ids_list,dim=1)读每一pass的独立INT64[B]结果，写materialized INT64[B,depth]，保留全部argmax依赖；不添加最终INT32反馈cast。depth1返回view，不产生stack写。source角色字节与argmax比较下界不证明实际DRAM/cache或PyTorch/library launch，全部新增调用unsupported、launch/demand/seconds=None，完整静态结果仍partial。准确SpeculativeConfig与index-sharing override、local/probabilistic/GPU-v2分支、heterogeneous vocabulary、watermarker/estimator、head dtype/scale/soft-cap override、target sampler/backup H2D及真实slot/metadata/allocator绑定仍缺。", "",
            "已知typed copy/row-select/scatter/INT64到INT32 cast/final stack可按逻辑read/write字节经统一Roofline计价，不需要grid/block。只将纯物化的数学浮点工作记为零，不声明integer conversion、address/control或编译服务为零；launch仍None且execution_support仍unsupported。argmax比较树、条件prepare和slot/state更新不在此闭合范围。历史产物保持原未计价状态，所有native阶段仍partial；PP1源码图不能越过已排除的正常容量门槛。", "",
            "已知ordinary/padded index preparation及明确present的padded rollback另按typed角色字节和源码INT32减/加/比较/选择计价；padded preparation包括tl.where的选择操作，不能省略。沿用既有H200 compute_alu的INT32 capacity，不改速率。未知metadata、absent rollback和slot/state仍拒绝，未给实际launch/compiler/native行为背书。", "",
            "| Artifact / snapshot | Source | 每phase源码stage counts | 每phase纯物化logical Roofline calls | 每phaseINT32 bookkeeping logical calls |",
            "|---|---|---|---|---|"])
        for record in sampling_records:
            counts = "/".join(str(phase["draft_sampling_stage_counts"]) for phase in record["phases"])
            logical_counts = "/".join(str(phase["draft_materialization_logical_roofline_calls"]) for phase in record["phases"])
            int32_counts = "/".join(str(phase["draft_int32_logical_roofline_calls"]) for phase in record["phases"])
            text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {record['mtp_draft_sampling_source']} | {counts} | {logical_counts} | {int32_counts} |")
        text.extend(["", "glm_legacy_greedy_source_direct.log为2failed3passed7.49s，失败来自两个ScheduledRequest fixture的raw_model_output_tokens=None；修复后失败项在glm_legacy_greedy_source_failed_recheck.log中2passed6.93s，复用原三个正向直接依赖。7aa185a1两款四phase native source probes正常完成，TP1/TP8分别80/664未计价调用每phase；五卡离线Kernel matrix全部PASS。后续selection/copy直接检查与最新matrix/probe以validation和上表snapshot为准。正式reference新snapshot复核待主agent协调，不能继承旧scope接受；真实Scheduler/QPS16、正常KV、GPU和收尾全维护测试未执行。", ""])
        text.extend(["selection/copy直接检查glm_legacy_feedback_direct.log为1failed4passed7.36s，四项正向含TP1/TP8 mixed rejected-row图及memo、depth1和既有typed sampler contract；失败来自新boundary fixture以normalized shape改dtype先触发dtype_bytes guard。修正fixture后glm_legacy_feedback_failed_recheck.log仍1failed3.26s，原因是supported fixture无LaunchConfig先触发公共guard；补完整synthetic launch以直接验证source-only边界后glm_legacy_feedback_boundary_recheck.log为1passed3.58s。只改fixture，复用原四项正向；没有放宽production contract或删除反例。", ""])
        text.extend(["legacy topk compaction直接差异在glm_legacy_topk_compaction_direct.log中4passed7.25s：TP1/TP8 mixed rejection、下一轮same-rank front producer、memo、完整INT32 temporary/copy字节、depth1及sharing关闭分支。KernelSpec复用已有source-only selection/copy variant，不扩展production KernelKind或通用INT32 lowering；此source subset与library/runtime门槛保持分离。", ""])
        text.extend(["final stack直接差异在glm_legacy_stack_direct.log中4passed7.16s，含TP1/TP8所有pass依赖、depth1不materialize及三pass typed byte边界。仅新增source-only VECTOR/dsv32_draft_token_stack，不扩展KernelKind，也不为unknown library launch分配速率或时延。", ""])
        text.extend(["first-pass input copies/scatter直接差异在glm_legacy_first_inputs_direct.log中4passed7.37s，涵盖typed角色字节、indexed-write边界、TP1/TP8 mixed rejection的(7,8)scatter、完整target norm依赖、memo、T1空shift和部分proposer subset拒绝。新增source-only VECTOR/dsv32_draft_id_scatter并复用copy variant；不扩KernelKind/通用INT32 lowering、不赋library launch或时延。", ""])
        text.extend(["padded next-ID/index准备直接差异在glm_legacy_padded_prepare_direct.log中4passed7.26s，含conditional branch reads、INT32输出、SIMD width/warps未绑定、cu-prefix unique/重复read区分与TP1/TP8 producer数据边；补整体13B下界及13+4*width B上界后仅相关helper与报告边界在glm_legacy_padded_prepare_bounds_report_direct.log中3passed4.23s，复用原三个图正向。备份H2D只列角色，实际sampler/slot/allocator/编译器未绑定；metadata未观测时不把ordinary或padded分支写成实际发射。", ""])
        text.extend(["legacy rollback/step metadata直接差异在glm_legacy_step_metadata_direct.log中5passed7.32s：typed mixed role bytes、padding与位置alias、cap reset/block0读取源状态、缺参数/不合法shape拒绝、TP1/TP8 data edges和depth1不更新；三pass位置buffer复用/一次rollback与报告边界在glm_legacy_step_alias_report_direct.log中3passed3.52s。旧GPU-v2状态scope保持原语义，本轮不继承或改写它。source-only VECTOR/dsv32_draft_seq_rollback/step_metadata均unpriced，不扩KernelKind或normal allocator支持；host metadata重建/CPU upper-bound、真实maxlen/blocks/slot/backend、compiler与runtime仍缺。", ""])
        text.extend(["首轮slot-mapping copy直接差异在glm_legacy_initial_slot_copy_direct.log中4passed7.52s，涵盖完整INT64 read/write、TP1/TP8 mixed/single-depth首轮only与memo。复用source-only copy variant，不扩KernelKind或通用INT64 lowering；actual slots/library launch未绑定。源码backend独立选择与host metadata建构缺口直接保留，不能继承target/GPU-v2 provider验收。", ""])
    if kimi_weight_records:
        text.extend(["", "## Kimi active outer constructor 参数（正常容量未验收）", "",
            "显式enabled-tower/no-offload源码场景，全部registered Parameter按BF16计；无图像输入不删除tower。VisionConfig默认参数与父config projector宽度归一化来自vLLM0.30源码。patch/norm与projector复制，vision权重依明确weights/data场景分片；heads不可整分时按源码保留全参数。仅此outer账本没有text权重、buffers、post-load、reserve或allocator，正常KV入口仍拒绝text_config，未推导available-KV/blocks。", "",
            "| Artifact / snapshot | TP | Vision source mode | Outer parameter paths | Outer registered B/rank |",
            "|---|---:|---|---:|---:|"])
        for record in kimi_weight_records:
            for case in record["cases"]:
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {case['tp']} | {case['vision_tensor_parallel_mode']} | {len(case['outer_loader_parameter_bytes_per_rank'])} | {case['outer_constructor_parameter_payload_bytes_per_rank']} |")
    if v4_weight_records:
        text.extend(["", "## V4 源码 constructor 参数账本（正常容量未验收）", "",
            "这些记录只统计显式FlashMLA/Marlin、EP-off/PP1、BF16默认dtype的注册参数，不包含post-load转换、alias、buffer、scratch和reserve。实际checkpoint/loader/provider没有绑定；source总量低于物理容量不能证明正常available-KV，生产V4容量入口仍拒绝。", "",
            "| Artifact / snapshot | TP | Routed packed B | Loader scales B | Non-routed B | Constructor total B | Physical HBM B |",
            "|---|---:|---:|---:|---:|---:|---:|"])
        for record in v4_weight_records:
            for case in record["cases"]:
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {case['tp']} | {case['routed_packed_bytes_per_rank']} | {case['routed_loader_scale_bytes_per_rank']} | {case['non_routed_loader_bytes_per_rank']} | {case['constructor_parameter_payload_bytes_per_rank']} | {case['physical_hbm_bytes']} |")
        text.extend(["", "646a9575的non-routed inventory已获独立scoped复核：Flash1330/Pro1894唯一registered Parameter路径与源码一致。该结论不包含post-load storage。首层finalize_mhc_broadcast_weights会另分配FP32 sum输出；DeepGemmFp8BlockScaledMMKernel经deepgemm_post_process_weight_scale_block把UE8M0转FP32后委托transform_sf_into_required_layout。wrapper并未给出扩展所分配的最终layout/storage/alias，不能以constructor总量替代最终resident bytes，不能推导正常KV blocks或available-KV。", ""])
    if formal_source_reviews:
        text.extend(["", "## 正式 reference 新交付的局部源码复核", "",
            "以下从独立reference发布文件读取原始checks、scope及binding限制，不重跑通过项、不改snapshot、不将dirty direct observation当clean整树验收。较早段落中当时待审/工具受限的描述保留为历史经过；本表是新发布的scoped结论。当前HEAD、actual compiler/provider/runtime、normal capacity及W1–W7仍未完成。", "",
            "| Reference receipt | 原始计数 | 原始 scope | Binding 限制 |",
            "|---|---|---|---|"])
        for record in formal_source_reviews:
            receipt = record["receipt"]
            name = Path(record["reference_artifact"]).name
            if name == "target_sampling_materialization_review.json":
                delta = record["model_delta_receipt"]["receipt"]
                counts = "; ".join(f"{key}={receipt[key]}" for key in ("direct_scopes_passed", "candidate_failures"))
                counts += "; " + "; ".join(f"{key}={delta[key]}" for key in (
                    "model_phases", "added_materializations", "new_declared_INT64_hidden_projections"))
                binding = (f"implementations={receipt['implementations']}; {receipt['snapshot_qualification']}; "
                           f"direct_heads={receipt['direct_heads']}; direct_dirty={receipt['direct_dirty']}")
                text.append(f"| [{name}]({record['reference_artifact']}) | {counts} | {'; '.join(receipt['accepted_scope'])} | {binding}; {'; '.join(receipt['limits'])} |")
                continue
            if name in ("glm_residual_norm_review.json", "kimi_residual_norm_review.json"):
                counts = "; ".join(f"{key}={receipt[key]}" for key in (
                    "direct_scopes_passed", "model_phase_count", "model_phases",
                    "changed_norm_calls", "candidate_failures") if key in receipt)
                binding = (f"implementation={receipt['implementation']}; {receipt['snapshot_qualification']}; "
                           f"direct_heads={receipt['direct_heads']}; direct_dirty={receipt['direct_dirty']}")
                text.append(f"| [{name}]({record['reference_artifact']}) | {counts} | {receipt['remaining']['item']} | {binding}; {'; '.join(receipt['limits'])} |")
                continue
            if name == "glm_target_fused_embedding_review.json":
                counts = "; ".join(f"{key}={receipt[key]}" for key in (
                    "direct_scopes_passed", "model_phases", "same_case_removed_calls", "candidate_failures"))
                binding = f"implementations={receipt['implementation_snapshots']}; direct_heads={receipt['direct_heads']}"
                text.append(f"| [{name}]({record['reference_artifact']}) | {counts} | {receipt['resolved_pending_embedding']} | {binding}; {'; '.join(receipt['limits'])} |")
                continue
            if name in _MHC_SM90_REVIEW_FILENAMES:
                if name in ("mhc_prenorm_sm90_review.json", "mhc_prenorm_sm90_tma_review.json",
                            "mhc_prenorm_sm90_zero_stage_review.json"):
                    scope = "; ".join(receipt["accepted_scope"])
                    binding = f"candidate_start={receipt['candidate_start']}; candidate_end={receipt['candidate_end']}"
                    limits = receipt["limits"]
                elif name == "mhc_prenorm_sm90_host_binary_review.json":
                    scope = receipt["source_basis"]
                    binding = f"publication_snapshot={receipt['publication_snapshot']}"
                    limits = receipt["not_accepted"]
                elif name == "mhc_prenorm_sm90_descriptor_review.json":
                    scope = receipt["accepted_basis"]
                    binding = f"observed_publication_head={receipt['observed_publication_head']}"
                    limits = receipt["limits"]
                elif name == "operator_sm90_formal_publication_review.json":
                    scope = receipt["accepted_scope"]
                    binding = f"publication_snapshot={receipt['publication_snapshot']['commit']}"
                    limits = receipt["limits"]
                else:
                    if name == "mhc_prenorm_sm90_model_delta_review.json":
                        scope = receipt["coverage_change"]
                    elif name == "mhc_prenorm_sm90_tma_model_delta_review.json":
                        scope = "; ".join(receipt["limits"])
                    else:
                        scope = receipt["scope"]
                    binding = f"original candidate_snapshot={receipt['candidate_snapshot']['commit']}"
                    limits = receipt["limits"]
                counts = "; ".join(f"{key}={receipt[key]}" for key in
                    ("own_direct_scopes_passed", "direct_scopes_passed", "candidate_failures",
                     "conditional_metadata_checks", "metadata_checks", "nonempty_partition_checks",
                     "receipts_checked", "lost_fields_or_relabels", "new_counter_categories") if key in receipt)
                path = record["reference_artifact"]
                text.append(f"| [{name}]({path}) | {counts} | {scope} | {binding}; {'; '.join(limits)} |")
                continue
            scope = (receipt["accepted_scope"] if "accepted_scope" in receipt else
                     receipt["scope"] if "scope" in receipt else receipt["source_scope"])
            counts = "; ".join(f"{key}={receipt[key]}" for key in
                               ("checks", "failed", "immutable_comparison_checks", "immutable_comparison_failed",
                                "artifact_checks", "artifact_failed", "model_layout_mode_records",
                                "phase_records", "recount_mismatches") if key in receipt)
            if "binding_limit" in receipt:
                binding = receipt["binding_limit"]
            elif "candidate" in receipt:
                binding = f"candidate={receipt['candidate']}; original artifact scope only"
            elif "snapshot" in receipt:
                binding = f"snapshot={receipt['snapshot']['commit']}; parameter/source scope only"
            elif "before_head" in receipt:
                binding = "; ".join(f"{key}={receipt[key]}" for key in
                                    ("before_head", "before_dirty", "after_head", "after_dirty")
                                    if key in receipt)
            elif "device_subset_compile" in receipt:
                compiled = receipt["device_subset_compile"]
                binding = (f"full module exit={receipt['full_module_compile']['exit_code']}; "
                           f"device subset exit={compiled['exit_code']}; "
                           f"GPU run={compiled['GPU_module_loaded_or_run']}; "
                           f"actual runtime build bound={compiled['actual_vLLM_runtime_build_bound']}")
            elif "production_metadata_change" in receipt:
                metadata = receipt["production_metadata_change"]
                binding = f"source metadata candidate={metadata['candidate']}; {metadata['scope']}; original compile scope only"
            elif "reused_worker_host_observation" in receipt:
                observation = receipt["reused_worker_host_observation"]
                binding = (f"worker observation reused; independent load={receipt['independently_loaded_library']}; "
                           f"actual runtime bound={observation['actual_vLLM_runtime_build_bound']}")
            elif "source_reference_qualification" in receipt:
                binding = receipt["source_reference_qualification"]
            elif "candidate_observation" in receipt:
                binding = f"candidate={receipt['candidate_observation']}; metadata/receipt scope only"
            else:
                binding = ", ".join(sorted({item["snapshot"]["commit"] for item in receipt["records"]}))
            path = record["reference_artifact"]
            text.append(f"| [{Path(path).name}]({path}) | {counts} | {scope} | {binding} |")
        for record in formal_source_reviews:
            receipt = record["receipt"]
            if "device_subset_compile" in receipt:
                text.extend(["", "离线编译资源与静态指令 sites（保留上述局部工具链/模板条件）：", "",
                    "| Device instance | Named score arrays B | Compiler static shared B | EX2 / RCP / TANH sites |",
                    "|---|---:|---:|---|"])
                for kernel in receipt["kernels"]:
                    text.append(f"| {kernel['demangled']} | {kernel['named_score_array_bytes_in_source']} | "
                                f"{kernel['compiler_reported_static_shared_bytes']} | "
                                f"{kernel['MUFU_EX2_sites']} / {kernel['MUFU_RCP_sites']} / {kernel['MUFU_TANH_sites']} |")
                text.extend(["", "这些sites不是动态执行次数；命名score数组不含编译器/cooperative-groups scratch，不能当总static shared或实际provider occupancy。当前源码logical-work将named_score_array_bytes_per_cta与未知总static_shared_bytes_per_cta=None分开；所有routing调用保持unpriced、launch=None，不从局部cubin推断H200服务率、provider或时延验收。", ""])
        text.extend(["", "本轮对主要未计价family的检查：GLM TP8 native808调用中616为grouped routing，Kimi496中480为routing，Flash modular12由embedding/prenorm/target selection各4个组成。grouped_topk_kernels.cu包含tanhf和packed64比较/归约；它经_moe_C_stable_libtorch及VLLM_GPU_FLAGS编译，common NVCC flags来自PyTorch。CMake里单独设置的use_fast_math属于具名Kimi kernels，不能据此给grouped tanhf绑定一条SFU指令或把INT32 ALU throughput用于整个uint64 compare/packing/归约。缺实际compiler/provider证据时继续保留unpriced，不以完整/分域duration或误差调参数。", ""])
    if graph_records:
        text.extend(["", "## V4 来源图（未执行 lowering 或定价）", "",
            "这些新图保留所有phase、TP rank的module_path/source_symbol/shape/dtype/attributes和完整edges，仅为正式reference提供新snapshot的来源比较输入。nodes数不是lowered kernel calls数；未计算时延，旧Roofline产物不改标签。", "",
            "| Artifact / snapshot | Phase | Workload nodes | Edges |",
            "|---|---|---:|---:|"])
        for record in graph_records:
            for phase in record["phases"]:
                text.append(f"| {record['artifact']} / {record['snapshot']['commit'][:8]} | {phase['phase']} | {phase['nodes']} | {phase['edges']} |")
        text.extend(["", "66ad4c85来源图已独立scoped复核：三个indexer来源修复分别覆盖Flash336/Pro960个对应节点，无stale MLA/LightningIndexer或self_attn/mlp标签；mHC/FFN/attention/head的全phase/rank来源通过。sparse QK所有节点保留包含RoPE的512总维度，Flash860/Pro2440个节点通过；未重复执行已过定价/launch/GPU测试。正式reference的Kimi TP/EP2695/2695绑定66d9ab0f，V4 TP/memo3772/3772绑定f24b7c56；这些数学/ownership/source scopes不接受实际provider、loaded weights、runtime、allocator或完整alignment。", ""])
    assessment_path = output / "unpriced_source_work_assessment.json"
    if assessment_path.is_file():
        assessment = json.loads(assessment_path.read_text(encoding="utf-8"))
        if (assessment["actual_runtime_bound"] is not False
                or assessment["production_pricing_changed"] is not False):
            raise ValueError("bounded source assessment cannot assert runtime or pricing acceptance")
        text.extend(["", "## 剩余未计价工作的一次有界源码判断", "",
            f"[完整字段与源码位置]({assessment_path.resolve()})。这是 worker 的解析判断，不是新的独立验收或时延预测；没有重复已通过测试、编译、环境或资产探测。", "",
            "| 范围 | 源码可确定 | 尚不可推出的点值/动态字段 | 本轮裁决 |",
            "|---|---|---|---|"])
        for family in assessment["families"]:
            text.append(f"| {family['family']} | {family['source_determined']} | {family['not_derivable']} | {family['decision']} |")
        text.extend(["", assessment["request_default_qualification"], "", assessment["pp_ruling"], ""])
    local_inputs_path = output / "bounded_local_inputs.json"
    if local_inputs_path.is_file():
        local_inputs = json.loads(local_inputs_path.read_text(encoding="utf-8"))
        text.extend(["", "## 历史有界本地输入核查（不代表当前资产状态）", "",
            f"结果与复现命令：[bounded_local_inputs.json]({local_inputs_path.resolve()})。仅检查声明的upstream/models、HF缓存及已知Conda环境；不宣称全机不存在资产。", "",
            "| 搜索根 | 存在 | config.json | draft JSON |", "|---|---|---|---|"])
        for search in local_inputs["searches"]:
            text.append(f"| {search['root']} | {search['exists']} | {search['config_paths']} | {search['draft_paths']} |")
        text.extend(["", f"离线环境：{local_inputs['offline_environment']}；包版本及路径：{local_inputs['offline_versions']}。",
            f"真实NamespaceTool导入结果：{local_inputs['NamespaceTool_import']}。",
            "以上保留原核查记录；当前七款准确配置已具备，K2.7-Code不在目标内。实际provider、normal容量及原生dispatch仍须分别验收，不能沿用历史缺配置结论。", ""])
        if "torch_sdk_environment_records" in local_inputs:
            text.extend(["已知Conda环境和base的真实Torch SDK：", "",
                "| SDK metadata | Version fields | ScalarType.h | CUDA stream / BLAS declarations |",
                "|---|---|---|---|"])
            for sdk in local_inputs["torch_sdk_environment_records"]:
                text.append(f"| {sdk['metadata']} | {sdk['fields']} | {sdk['ScalarType_header_present']} | "
                            f"{sdk['aoti_cuda_stream_declaration']} / {sdk['cuda_blas_handle_declaration']} |")
            text.extend(["", local_inputs["CUDA_API_scope_correction"], ""])
    module_result_path = output / "grouped_topk_cuda122_torch211_full_module" / "result.json"
    if module_result_path.is_file():
        module_result = json.loads(module_result_path.read_text(encoding="utf-8"))
        text.extend(["", "## 新发现的本地SDK完整模块编译核查", "",
            f"[命令与结果]({module_result_path.resolve()})；[编译日志]({(module_result_path.parent / 'compile.log').resolve()})。",
            f"Torch头文件来源：{module_result['torch_header_source']}；source_transform={module_result['source_transform']}；exit_code={module_result['exit_code']}；GPU run={module_result['GPU_module_loaded_or_run']}；actual runtime build bound={module_result['actual_vLLM_runtime_build_bound']}。",
            "旧离线命令遗漏USE_CUDA，导致两个受该宏控制的CUDA API声明不可见；这不能证明Torch2.11缺少这些API。它们确实存在于真实SDK，vLLM CMakeLists.txt为_moe_C_stable_libtorch设置USE_CUDA和Torch2.11 stable ABI目标。Python requirements的Torch2.13与C++模块的2.11 ABI目标分开记录；旧失败日志保留为命令缺项的历史证据，不再作为SDK缺API阻塞。", ""])
    for directory in ("grouped_topk_cuda122_torch211_use_cuda_full_module",
                      "grouped_topk_cuda122_torch211_cmake_definitions",
                      "grouped_topk_cuda122_torch211_cmake_flags"):
        corrected_result_path = output / directory / "result.json"
        if corrected_result_path.is_file():
            corrected = json.loads(corrected_result_path.read_text(encoding="utf-8"))
            text.extend(["", f"修正后的完整CUDA translation unit：[命令与结果]({corrected_result_path.resolve()})；"
                f"[编译日志]({(corrected_result_path.parent / 'compile.log').resolve()})。",
                f"exit_code={corrected['exit_code']}；cubin_present={corrected['cubin_present']}；"
                f"source_transform={corrected['source_transform']}；GPU run={corrected['GPU_module_loaded_or_run']}；"
                f"actual runtime build bound={corrected['actual_vLLM_runtime_build_bound']}。",
                "仅为本地CUDA12.2/SM90、真实Torch2.11头文件与显式编译旗标的离线编译证据，未链接或import vLLM扩展，未选择/运行H200 provider；不产生动态计数、速率、traffic或duration验收。四个旧device实例的正式scope不扩大。", ""])
            if "CUDA12_CMake_flag_filter_source" in corrected:
                text.extend([corrected["CUDA12_CMake_flag_filter_source"],
                    "CUDA>=12时vLLM明确移除Torch COMMON_NVCC_FLAGS的四个half/BF16禁用宏；之前未过滤旗标的编译失败是命令控制项，不是源码或SDK缺口。不按编译失败或时延猜旗标，目标定义及过滤规则均来自CMake源码。", ""])
    complete_instances_path = output / "grouped_topk_cuda122_torch211_cmake_flags" / "selected_instances.json"
    if complete_instances_path.is_file():
        complete_instances = json.loads(complete_instances_path.read_text(encoding="utf-8"))
        text.extend(["", f"完整translation unit的四个对应实例：[静态输出及复现命令]({complete_instances_path.resolve()})。",
            "| Device instance | Named score arrays B | Full-TU shared B | EX2 / RCP / TANH sites |",
            "|---|---:|---:|---|"])
        for kernel in complete_instances["records"]:
            text.append(f"| {kernel['demangled']} | {kernel['named_score_array_bytes_in_source']} | "
                        f"{kernel['resource_usage']['SHARED']} | {kernel['MUFU_EX2_static_sites']} / "
                        f"{kernel['MUFU_RCP_static_sites']} / {kernel['MUFU_TANH_static_sites']} |")
        accepted_tu = any(Path(record["reference_artifact"]).name == "grouped_topk_full_translation_unit_review.json"
                          and record["receipt"]["failed"] == 0 and record["receipt"]["checks"] > 0
                          for record in formal_source_reviews)
        qualification = ("完整TU静态输出已获正式reference局部复核，原始scope及四项检查见receipt表；不接受实际provider/动态服务或完整模型。"
                         if accepted_tu else "新增完整TU静态输出尚未独立验收。")
        text.extend(["", "本地完整TU中两个warp实例报告shared=1024B，而旧isolated subset报告0B；block仍为GLM3072B/Kimi4096B。这证明编译资源必须保留TU和旗标条件，不能把旧subset数字继承为完整模块或实际provider资源。"
                     + qualification + "source logical-work总shared保持None，所有routing服务继续unpriced。", ""])
    host_probe_path = output / "grouped_topk_torch211_host_registration_probe" / "host_load_result.json"
    if host_probe_path.is_file():
        host_probe = json.loads(host_probe_path.read_text(encoding="utf-8"))
        text.extend(["", "## Torch2.11 主机链接与注册边界", "",
            f"[链接结果与范围]({Path(host_probe['link_result']).resolve()})；"
            f"[主机加载观察及复现命令]({host_probe_path.resolve()})。load_exit_code={host_probe['load_exit_code']}。",
            "原始grouped_topk CUDA TU和host wrapper未改；仅选择vendor grouped_topk schema/TORCH_BOX注册语句，改用独立诊断namespace。这不是完整vLLM扩展，也不替代OpenAI SDK或实际provider。", ""])
        if host_probe["load_exit_code"] == 0:
            observation = host_probe["observation"]
            text.extend([f"真实Torch：{observation['torch_version']} / {observation['torch_source']}。",
                f"schema：`{observation['schema']}`。",
                f"CUDA dispatch registered={observation['CUDA_dispatch_registration']}；CPU dispatch registered={observation['CPU_dispatch_registration']}；"
                f"CUDA_VISIBLE_DEVICES='{observation['CUDA_VISIBLE_DEVICES']}'；PyTorch CUDA initialized before/after="
                f"{observation['pytorch_cuda_initialized_before']}/{observation['pytorch_cuda_initialized_after']}。",
                f"tensor/operator invoked={observation['tensor_created_or_operator_invoked']}；GPU kernel invoked={observation['GPU_kernel_invoked']}；"
                f"actual vLLM runtime bound={observation['actual_vLLM_runtime_build_bound']}。",
                "只确认真实本地SDK的单算子host ABI链接/注册；未调用CUDA kernel、不提供数值/traffic/demand/rate/duration，不扩大W1–W7或完整模型验收。", ""])
            host_review = [record for record in formal_source_reviews if
                           Path(record["reference_artifact"]).name == "grouped_topk_host_registration_review.json"]
            if host_review:
                receipt = host_review[0]["receipt"]
                text.extend([f"正式reference检查数/失败数：{receipt['checks']}/{receipt['failed']}；"
                    f"独立重新加载库={receipt['independently_loaded_library']}。原始scope：{receipt['accepted_scope']}。",
                    "这些检查覆盖源码选择、ELF/依赖/符号和既有raw观察，分母不是GPU执行或模型case；不继承完整vLLM/provider/runtime/时延验收。", ""])
        else:
            text.extend([f"主机加载失败：{host_probe['load_error']}。该失败保留为SDK边界证据。", ""])
    finite_precision_reviews = [record for record in formal_source_reviews if
        Path(record["reference_artifact"]).name == "grouped_topk_finite_precision_reference.json"]
    if finite_precision_reviews:
        receipt = finite_precision_reviews[0]["receipt"]
        text.extend(["", "## Grouped-topk 有限精度边界（源码反例）", "",
            f"正式reference：{finite_precision_reviews[0]['reference_artifact']}。原始scope：{receipt['accepted_scope']}。",
            f"验收限定：{receipt['source_reference_qualification']}。",
            "源码block读取保留的FP32 unbiased score，warp从已舍入的biased score减回bias。selection mask非空时，恢复和为零不会触发uniform fallback。以下直接列出已校验receipt的源码算术结果；不重新计算，不证明实际checkpoint或GPU缺陷。", "",
            "| E / K | Logits all zero | Bias | Selection FP32 | Block unbiased | Warp recovered | Block weight each | Warp weight each | Same value |",
            "|---|---|---|---|---|---|---|---|---|"])
        for example in receipt["examples"]:
            text.append(f"| {example['experts']} / {example['topk']} | {example['logits_all_zero']} | {example['correction_bias_all']} | {example['selection_score_fp32']} | {example['source_block_unbiased']} | {example['source_warp_recovered_unbiased']} | {example['block_weight_each_fp32']} | {example['warp_weight_each_fp32']} | {example['same_value']} |")
        text.append("")
        for example in receipt["examples"]:
            text.extend([f"条件：{example['condition']}。Dispatch限定：{example['dispatch_transition']}。", ""])
        for finding in receipt["findings"]:
            text.extend([f"Reference finding：{finding['item']}。Required action：{finding['required_action']}。", ""])
        text.extend(["GLM/Kimi conditional source contract显式记录不同恢复、selection validity、uniform fallback及epsilon归一化语义。实际checkpoint bias域、选中provider和native数值行为仍未绑定；CPU torch分支、离线编译和host注册均不能替代这些证据。运算/字节计数、launch assignment、HWSpec速率、demand/time未改，routing仍unpriced，W1–W7均未通过。", ""])
        text.extend([f"原始限制：{limit}" for limit in receipt["limits"]])
    publication_reviews = [record for record in formal_source_reviews if
        Path(record["reference_artifact"]).name == "operator_finite_precision_publication_review.json"]
    if publication_reviews:
        receipt = publication_reviews[0]["receipt"]
        text.extend(["", "", "## 有限精度限定的正式候选局部复核", "",
            f"正式reference：[publication review]({publication_reviews[0]['reference_artifact']})；"
            f"原候选={receipt['candidate_observation']}，metadata分支={len(receipt['metadata_checks'])}，"
            f"保留原始source scenario={receipt['source_scenarios_preserved']}，failed={receipt['failed']}。",
            f"原始接受范围：{receipt['accepted_scope']}。",
            "仅确认source数值限定和receipt发布；metadata分支/scenario数不是新增硬件、模型或数值验收分母。W6与全目标仍未通过，不向后续HEAD自动继承actual provider/runtime或pricing接受。", ""])
    if existing_paths:
        path_records = list(existing_paths.values())
        coverage_path = output / "existing_model_roofline_paths.json"
        coverage_path.write_text(json.dumps({
            "records": path_records, "goal_complete": False,
            "selection": "last explicitly supplied artifact per model/layout/mode; original source scenario and snapshot only",
            "actual_runtime_or_normal_capacity_accepted": False}, indent=2) + "\n", encoding="utf-8")
        text.extend(["", "## 已有配置模型的最小路径判断", "",
            f"[逐阶段未计价family与首个阻塞call]({coverage_path.resolve()})。复用原静态产物，不重跑已通过case；各行保持原snapshot、配置revision和source场景。未计价计数不代表时延误差，不能据其调整参数。",
            "| Artifact / original snapshot | 模型 / TP / PP / MTP | 未计价calls（prefill/decode1/decode2/mixed） | 首阶段缺口 |",
            "|---|---|---|---|"])
        for record in path_records:
            deployment = record["deployment"]
            phases = record["phases"]
            counts = "/".join(str(phase["unpriced_calls"]) for phase in phases)
            artifact = Path(record["artifact"])
            text.append(f"| [{artifact.name}]({artifact.resolve()}) / {record['snapshot']['commit'][:8]} | {record['model_id']} / {deployment['tensor_parallel_size']} / {deployment['pipeline_parallel_size']} / {record['mtp']} | {counts} | {phases[0]['unpriced_families']} |")
        text.extend(["", "Kimi-K2.6与GLM-5.2有准确配置和已实现源码路径；其局部阻塞分别来自grouped routing及embedding/target-indexing、native状态服务，不是另外三个模型配置缺失。GLM BF16 TP8/PP1与TP1/PP1有独立routed-weight容量排除证据；PP2未被排除，但仍是TP-partial传输未接受的source projection。",
            "V4 Flash是当前较小的条件源码case，但配置revision仍为existing-h200-campaign-config-revision-unverified，不能将它冒充准确checkpoint身份。其首层DeepGEMM prenorm和embedding/target-indexing缺源码到实际launch/provider的绑定；减少token数不能删掉这些必需producer。V4 Pro同样保留这些缺口及post-load residency/DSpark缺口。",
            "已有源码支持的数学FLOPs、typed bytes和条件launch继续有效，但当前没有一行证明完整真实Roofline/正常Serving闭合。真实0.30 Scheduler还受已记录的NamespaceTool SDK缺口约束；配置、源码场景、compiler/host注册、actual provider和正常KV验收分开。", ""])
        path_publications = [record for record in formal_source_reviews if
            Path(record["reference_artifact"]).name == "operator_model_roofline_paths_review.json"]
        if path_publications:
            receipt = path_publications[0]["receipt"]
            text.extend([f"正式独立发布统计复核：[原始receipt]({path_publications[0]['reference_artifact']})；"
                f"布局/模式records={receipt['model_layout_mode_records']}，phase records={receipt['phase_records']}，"
                f"recount mismatches={receipt['recount_mismatches']}。",
                f"原始scope：{receipt['accepted_scope']}。每个record的snapshot与source_projection_only/partial原标志保持；未增加operator pricing、实际runtime、normal KV或全Goal验收。", ""])
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
