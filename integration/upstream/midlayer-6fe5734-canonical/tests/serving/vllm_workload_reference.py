"""Independent pinned framework-source references for workload emission."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Mapping

from src.serving.workload.helper_launch import (
    RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED,
)


_VLLM_VERSION = "0.11.0"
_VLLM_SOURCE_COMMIT = "b8b302cde434df8c9289a2b465406b47ebab1c2d"
_PAGED_KV_CACHE_ATTRIBUTES = {
    "cache_layout": "paged",
    "cache_head_layout": "nhd",
    "kv_cache_dtype": "auto",
    "cache_storage_matches_input": True,
    "slot_mapping_all_valid": True,
    "vector_copy_base_alignment_bytes": 16,
}
_RESIDUAL_BOUNDARY_SOURCE_FRAGMENTS = (
    "if residual is None:",
    "residual = hidden_states",
    "hidden_states, residual = self.input_layernorm(",
    "hidden_states, residual = self.post_attention_layernorm(",
    "return hidden_states, residual",
)
_SOURCE_CONTRACT = (
    (
        "model_executor/models/qwen3.py",
        "ebcfa01f530ecd7c123cdddc6085d015dd809cb0a0189887a09f8a17a11f00ec",
        (
            ("Qwen3Attention", "__init__"),
            ("Qwen3Attention", "forward"),
            ("Qwen3DecoderLayer", "forward"),
            ("Qwen3ForCausalLM", "__init__"),
            ("Qwen3ForCausalLM", "compute_logits"),
        ),
    ),
    (
        "model_executor/models/qwen2.py",
        "ae57d84ad2a2cbc986538ce221d082f4662e169413988ddaaf4a48539dee1bff",
        (
            ("Qwen2MLP", "__init__"),
            ("Qwen2MLP", "forward"),
            ("Qwen2Model", "__init__"),
            ("Qwen2Model", "forward"),
        ),
    ),
    (
        "model_executor/layers/linear.py",
        "b2cc3cb18d161976a957f0b70af302291baa66aaed99229497674f266e1aac4d",
        (
            ("MergedColumnParallelLinear", "__init__"),
            ("QKVParallelLinear", "__init__"),
            ("RowParallelLinear", "__init__"),
            ("RowParallelLinear", "forward"),
        ),
    ),
    (
        "model_executor/layers/vocab_parallel_embedding.py",
        "f74cbf9882ae59449ce4713cd3ac24b77bdf5a585770399c5968b281f591a6e7",
        (
            ("VocabParallelEmbedding", "forward_native"),
            ("ParallelLMHead", "__init__"),
        ),
    ),
    (
        "model_executor/layers/layernorm.py",
        "878ae09c09a54a3565c740a99324946b99e60810ee8297be9850951759bbe022",
        (
            ("RMSNorm", "__init__"),
            ("RMSNorm", "forward_native"),
        ),
    ),
    (
        "model_executor/layers/activation.py",
        "5a514c782e1f5dedc3266a892a50003d102c4ab5e4aa36a769befce2c072ee68",
        (
            ("SiluAndMul", "__init__"),
            ("SiluAndMul", "forward_native"),
        ),
    ),
    (
        "model_executor/layers/rotary_embedding/__init__.py",
        "c2649d32e466a5593e254c5d00188e2fdd570d18fec7d39af86dcb8805debb85",
        (
            ("", "get_rope"),
        ),
    ),
    (
        "model_executor/layers/logits_processor.py",
        "3dc2eb8526b0ec49aae393022db488d9220be3fa507893d1d1bff9442529b56e",
        (
            ("LogitsProcessor", "forward"),
            ("LogitsProcessor", "_gather_logits"),
            ("LogitsProcessor", "_get_logits"),
        ),
    ),
    (
        "platforms/interface.py",
        "ff142312173c173dae20a8303cd10f9c29fa255862396655b6888d2fdb73fae5",
        (
            ("Platform", "use_all_gather"),
        ),
    ),
    (
        "attention/layer.py",
        "83136ff53e104322f5ab1425c19038d75c4ea8b06085b627af13211a39bc2653",
        (
            ("Attention", "__init__"),
            ("Attention", "forward"),
        ),
    ),
    (
        "v1/worker/gpu_model_runner.py",
        "ea6b7bfe4082349e80042df4801c7ef9e18c80d1c3879eeb6ac6e9c476303574",
        (
            ("GPUModelRunner", "_prepare_inputs"),
            ("GPUModelRunner", "execute_model"),
        ),
    ),
)

_LLAMA_SOURCE_CONTRACT = (
    (
        "model_executor/models/llama.py",
        "135cc5e748672057e1528b7f21c7a767c56d2ea4438b73ad60e3cfa3743d8613",
        (
            ("LlamaMLP", "__init__"),
            ("LlamaMLP", "forward"),
            ("LlamaAttention", "__init__"),
            ("LlamaAttention", "forward"),
            ("LlamaAttention", "_init_rotary_emb"),
            ("LlamaDecoderLayer", "__init__"),
            ("LlamaDecoderLayer", "forward"),
            ("LlamaModel", "__init__"),
            ("LlamaModel", "forward"),
            ("LlamaForCausalLM", "__init__"),
            ("LlamaForCausalLM", "forward"),
            ("LlamaForCausalLM", "compute_logits"),
        ),
    ),
)

_LLAMA_TP_SOURCE_CONTRACT = (
    (
        "model_executor/layers/linear.py",
        "b2cc3cb18d161976a957f0b70af302291baa66aaed99229497674f266e1aac4d",
        (
            ("MergedColumnParallelLinear", "__init__"),
            ("QKVParallelLinear", "__init__"),
            ("RowParallelLinear", "__init__"),
            ("RowParallelLinear", "forward"),
        ),
    ),
    (
        "model_executor/layers/vocab_parallel_embedding.py",
        "f74cbf9882ae59449ce4713cd3ac24b77bdf5a585770399c5968b281f591a6e7",
        (
            ("VocabParallelEmbedding", "forward_native"),
            ("ParallelLMHead", "__init__"),
        ),
    ),
    (
        "model_executor/layers/logits_processor.py",
        "3dc2eb8526b0ec49aae393022db488d9220be3fa507893d1d1bff9442529b56e",
        (
            ("LogitsProcessor", "forward"),
            ("LogitsProcessor", "_gather_logits"),
            ("LogitsProcessor", "_get_logits"),
        ),
    ),
    (
        "platforms/interface.py",
        "ff142312173c173dae20a8303cd10f9c29fa255862396655b6888d2fdb73fae5",
        (("Platform", "use_all_gather"),),
    ),
    (
        "v1/worker/gpu_model_runner.py",
        "ea6b7bfe4082349e80042df4801c7ef9e18c80d1c3879eeb6ac6e9c476303574",
        (
            ("GPUModelRunner", "_prepare_inputs"),
            ("GPUModelRunner", "execute_model"),
        ),
    ),
)

_LLAMA_TP_REQUIRED_SOURCE_FRAGMENTS = {
    "model_executor/layers/linear.py": (
        "self.num_heads = divide(self.total_num_heads, tp_size)",
        "if tp_size >= self.total_num_kv_heads:",
        "self.num_kv_heads = 1",
        "if self.reduce_results and self.tp_size > 1:",
        "output = tensor_model_parallel_all_reduce(output_parallel)",
    ),
    "model_executor/layers/vocab_parallel_embedding.py": (
        "output = tensor_model_parallel_all_reduce(output_parallel)",
    ),
    "model_executor/layers/logits_processor.py": (
        "if self.use_all_gather:",
        "logits = tensor_model_parallel_all_gather(logits)",
    ),
}

_LLAMA_REQUIRED_SOURCE_FRAGMENTS = (
    "head_dim = self.hidden_size // self.total_num_heads",
    "self.q_size = self.num_heads * self.head_dim",
    "self.kv_size = self.num_kv_heads * self.head_dim",
    "q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)",
    "q, k = self.rotary_emb(positions, q, k)",
    "attn_output = self.attn(q, k, v)",
    *_RESIDUAL_BOUNDARY_SOURCE_FRAGMENTS,
    "output_sizes=[intermediate_size] * 2",
    "self.act_fn = SiluAndMul()",
    "x, _ = self.down_proj(x)",
    "self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)",
    "self.lm_head = ParallelLMHead(",
    "logits = self.logits_processor(self.lm_head, hidden_states)",
)

_KIMI_MLA_0_11_SOURCE_CONTRACT = (
    (
        "model_executor/models/deepseek_v2.py",
        "f9483594c8ccf5af3a67ab6eebd0d55ffee8db7fc79f9a0f30f6db8448dc67ed",
        (
            ("DeepseekV2MLAAttention", "__init__"),
            ("DeepseekV2MLAAttention", "forward"),
        ),
    ),
    (
        "model_executor/layers/mla.py",
        "b509f0230356f851ef752070627382c541ba2757b7e9c6e68add3367162368b1",
        (("MultiHeadLatentAttention", "forward_native"),),
    ),
    (
        "v1/attention/backends/mla/common.py",
        "e23af74b27a0944348f13d1b41b38c9899f108f9a86de175f1706cdc8192f5b4",
        (
            ("MLACommonBaseImpl", "_v_up_proj"),
            ("MLACommonImpl", "_forward_prefill"),
            ("MLACommonImpl", "forward"),
        ),
    ),
    (
        "model_executor/models/registry.py",
        "66cc1cd3245239328c86af49dde46ac98bfefa3b5e52d038eb683c2d06076c25",
        (),
    ),
)

_KIMI_MLA_0_11_ATTENTION_SOURCE_FRAGMENTS = (
    "self.fused_qkv_a_proj = MergedColumnParallelLinear(",
    "[self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim]",
    "disable_tp=True)",
    "self.q_b_proj = ColumnParallelLinear(self.q_lora_rank,",
    "self.num_heads *\n                                                 "
    "self.qk_head_dim",
    "self.kv_b_proj = ColumnParallelLinear(",
    "self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)",
    "self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,",
)

_KIMI_OFFICIAL_SOURCE_EVIDENCE = {
    "revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
    "configuration_sha256": (
        "2bc5487efced291e0e6aa3b24aad1082bade4172aff94ea1b6f0ae20a1d6bb4f"
    ),
    "modeling_sha256": (
        "1fd8d198ff6ad69a5aec6fd85bf489d91ae2c432560b1e8ba7e34f710463c80a"
    ),
}

_GLM_5_2_SOURCE_EVIDENCE = {
    "framework": "transformers",
    "framework_version": "5.10.2",
    "source_commit": "0dad7b822255a0ae261ec45ae937371e859ffd1a",
    "configuration_sha256": (
        "3d72daa66b8a1bd703833b9b85a2b76e14fbd5158fe1238866d36e40b60225cc"
    ),
    "modeling_sha256": (
        "0560406b772637b451cedfaccbb1d5a6c73fd002a0ae05807e95af89ced44da7"
    ),
    "model_revision": "b4734de4facf877f85769a911abafc5283eab3d9",
}

_GLM_5_2_VLLM_SOURCE_EVIDENCE = {
    "framework": "vllm",
    "framework_version": "0.27.1",
    "source_commit": "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
    "python_requirement": ">=3.10,<3.15",
    "pytorch_requirement": "2.13.0",
    "transformers_requirement": ">=5.5.3",
    "source_files": (
        (
            "vllm/model_executor/models/registry.py",
            "f1dae30a951fda360151c8de9dd605d01c7138a69f70fe518286d57da69dc463",
        ),
        (
            "vllm/model_executor/models/deepseek_v2.py",
            "bcbd9cde689ffe3f5f271aff3a5e01351b05a68371192cac9d65cb5ffc6b4ba9",
        ),
        (
            "vllm/model_executor/layers/mla.py",
            "ccdbafa998b1a7f7db47527df85efdfa4bc3027f6ec769fd4c30b3aee963a81b",
        ),
        (
            "vllm/model_executor/layers/attention/mla_attention.py",
            "272c01ff03e52cdfaed6fc7bc2e3e5f641d35910ea0bf5365b928c9401c4df5f",
        ),
        (
            "vllm/model_executor/layers/attention/sparse_mla_attention.py",
            "1aa19b3dc2d74acd73d77a3d6b04c98b5e36585907308986098aff120a82d1cd",
        ),
        (
            "vllm/model_executor/layers/sparse_attn_indexer.py",
            "a6386326379dcf94439a0b020cd51a71ebfd0a36eaeb563417c821456b82a475",
        ),
        (
            "vllm/v1/attention/backends/mla/flashmla_sparse.py",
            "0544c431bbf311d7283fee7f34cd7ad8a2df7961380646f2c12db2e861608ee4",
        ),
        (
            "pyproject.toml",
            "bd895b6b28d30e395ac295ac8789fc44b8de1d039d3b13e6aa11c1e893757bd2",
        ),
    ),
}

_PP_ADDITIONAL_SOURCE_CONTRACT = (
    (
        "v1/worker/gpu_worker.py",
        "e06833e8a5247dea488f067078b9f64e1284ca9c97b248b279852b046e815256",
        (("Worker", "execute_model"),),
    ),
    (
        "distributed/parallel_state.py",
        "9a78fb28d380b5a188a1ac627a2e6877cacb185b47b40b5d9287e2d1f0afa24e",
        (
            ("", "_split_tensor_dict"),
            ("GroupCoordinator", "send_tensor_dict"),
            ("GroupCoordinator", "recv_tensor_dict"),
        ),
    ),
    (
        "distributed/utils.py",
        "2b608d7ddc8364285fcba416a0cc7e9faa43cdb1bc5f848199dbad12784f36a4",
        (("", "get_pp_indices"),),
    ),
    (
        "model_executor/models/utils.py",
        "a23f543890cd74c88476dcb448947010300b41123edad9320c4e0e9ad5b10e3a",
        (("", "make_layers"),),
    ),
)

_PP_REQUIRED_SOURCE_FRAGMENTS = {
    "v1/worker/gpu_worker.py": (
        ("all_gather_group=get_tp_group()", 2),
        ("all_gather_tensors = {", 1),
        ("not is_residual_scattered_for_sp(self.vllm_config,", 1),
    ),
    "distributed/parallel_state.py": (
        (
            "tensor = tensor.reshape(all_gather_size, -1)"
            "[all_gather_rank]",
            1,
        ),
        ("tensor = all_gather_group.all_gather(", 1),
        (
            "group_ranks = all_ranks.view(-1, "
            "tensor_model_parallel_size).unbind(0)",
            1,
        ),
        ("group_ranks = all_ranks.transpose(2, 3).reshape(", 1),
    ),
}

_EP_ADDITIONAL_SOURCE_CONTRACT = (
    (
        "model_executor/models/qwen3_moe.py",
        "895132b02d8b62c4ca8d0e6456a92bcb6491a99eb4efee2746a9f9ae9d562b7b",
        (
            ("Qwen3MoeAttention", "__init__"),
            ("Qwen3MoeAttention", "forward"),
            ("Qwen3MoeDecoderLayer", "__init__"),
            ("Qwen3MoeDecoderLayer", "forward"),
            ("Qwen3MoeSparseMoeBlock", "__init__"),
            ("Qwen3MoeSparseMoeBlock", "forward"),
        ),
    ),
    (
        "model_executor/layers/linear.py",
        "b2cc3cb18d161976a957f0b70af302291baa66aaed99229497674f266e1aac4d",
        (
            ("ReplicatedLinear", "__init__"),
            ("ReplicatedLinear", "forward"),
        ),
    ),
    (
        "model_executor/layers/fused_moe/layer.py",
        "40ed4930aeac0c04ec528b3c5eddd7ff8463a19ada211a46cd08c4db3bd06063",
        (
            ("", "determine_expert_map"),
            ("FusedMoE", "select_experts"),
            ("FusedMoE", "forward_native"),
            ("FusedMoE", "forward_impl"),
        ),
    ),
    (
        "model_executor/layers/fused_moe/fused_moe.py",
        "8be8477a653894194ef5c5095afc287b0409a51d980590a36dd43c8471f6740e",
        (("", "fused_topk"), ("", "fused_experts_impl")),
    ),
    (
        "model_executor/layers/fused_moe/moe_align_block_size.py",
        "ac0bd207c71de8db5a9d093cbcd2fb7cb9292f7ba5d2c69c5ff04121993c691f",
        (("", "moe_align_block_size"),),
    ),
    (
        "model_executor/layers/fused_moe/config.py",
        "eb284b1c00128827f83aebf061153605f9df0cf99f2bc6f4d95521db426a122d",
        (
            ("FusedMoEParallelConfig", "use_all2all_kernels"),
            ("FusedMoEParallelConfig", "make"),
        ),
    ),
    (
        "config/parallel.py",
        "33def3bb46ce10a43df3cd97879a4299b25e48beaceede8203e1941c43c415e1",
        (("ParallelConfig", "__post_init__"),),
    ),
    (
        "distributed/communication_op.py",
        "8a00815cda00849f1e668a11efd2a1a738189553f54d48273c5e42ec14a94891",
        (("", "tensor_model_parallel_all_reduce"),),
    ),
)

_EP_REQUIRED_SOURCE_FRAGMENTS = {
    "model_executor/models/qwen3_moe.py": (
        *((fragment, 1)
          for fragment in _RESIDUAL_BOUNDARY_SOURCE_FRAGMENTS),
        (
            "self.head_dim = head_dim or "
            "(hidden_size // self.total_num_heads)",
            1,
        ),
        ("self.q_size = self.num_heads * self.head_dim", 1),
        ("self.kv_size = self.num_kv_heads * self.head_dim", 1),
        (
            "self.o_proj = RowParallelLinear("
            "self.total_num_heads * self.head_dim,",
            1,
        ),
        ("head_dim=getattr(config, 'head_dim', None),", 1),
        (
            "final_hidden_states = self.experts(hidden_states=hidden_states,",
            1,
        ),
        ("self.gate = ReplicatedLinear(", 1),
    ),
    "model_executor/layers/linear.py": (
        ("class ReplicatedLinear(LinearBase):", 1),
    ),
    "model_executor/layers/fused_moe/layer.py": (
        (
            "do_naive_dispatch_combine: bool = (\n"
            "            self.dp_size > 1",
            1,
        ),
        ("states = self.maybe_all_reduce_tensor_model_parallel(", 1),
    ),
    "model_executor/layers/fused_moe/fused_moe.py": (
        ("num_tokens = hidden_states.size(0)", 2),
    ),
    "model_executor/layers/fused_moe/moe_align_block_size.py": (
        ("expert_ids = expert_map[expert_ids]", 1),
    ),
    "model_executor/layers/fused_moe/config.py": (
        ("return self.dp_size > 1 and self.use_ep", 1),
    ),
    "config/parallel.py": (
        ('expert_placement_strategy: ExpertPlacementStrategy = "linear"', 1),
    ),
    "distributed/communication_op.py": (
        ("def tensor_model_parallel_all_reduce(", 1),
    ),
}


def pinned_qwen3_source_evidence() -> dict[str, object]:
    """Verify exact source files and symbols without importing vLLM."""

    version = importlib.metadata.version("vllm")
    if version != _VLLM_VERSION:
        raise RuntimeError(
            f"Qwen3 source reference requires vLLM {_VLLM_VERSION}, "
            f"got {version}")
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the pinned vLLM package")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"vLLM source reference requires one package root: {locations!r}")
    package_root = Path(locations[0]).resolve()
    qwen2_source = (
        package_root / "model_executor/models/qwen2.py"
    ).read_text(encoding="utf-8")
    intermediate_fields = '["hidden_states", "residual"], config.hidden_size'
    if qwen2_source.count(intermediate_fields) != 1:
        raise RuntimeError(
            "pinned Qwen2 PP intermediate tensor fields changed")
    files = []
    for relative_path, expected_hash, required_symbols in _SOURCE_CONTRACT:
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"pinned vLLM workload source is missing: {source_path}")
        source_bytes = source_path.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned vLLM workload source changed: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}")
        tree = ast.parse(source_bytes, filename=str(source_path))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        top_level_functions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for class_name, method_name in required_symbols:
            if not class_name:
                if method_name not in top_level_functions:
                    raise RuntimeError(
                        f"pinned source {relative_path} lacks function "
                        f"{method_name}")
                continue
            if class_name not in classes:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks class {class_name}")
            methods = {
                node.name for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks symbol "
                    f"{class_name}.{method_name}")
        if relative_path == "model_executor/models/qwen3.py":
            source = source_bytes.decode("utf-8")
            for fragment in _RESIDUAL_BOUNDARY_SOURCE_FRAGMENTS:
                count = source.count(fragment)
                if count != 1:
                    raise RuntimeError(
                        "pinned Qwen3 residual boundary source changed: "
                        f"expected fragment once, got {count}: "
                        f"{fragment!r}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
            "symbols": tuple(
                (f"{class_name}.{method_name}"
                 if class_name else method_name)
                for class_name, method_name in required_symbols
            ),
        })
    scheduler_process = (
        Path(__file__).resolve().parents[2]
        / "src/serving/backend/vllm_scheduler_process.py"
    )
    scheduler_source = scheduler_process.read_text(encoding="utf-8")
    v1_assignment = 'os.environ["VLLM_USE_V1"] = "1"'
    if scheduler_source.count(v1_assignment) != 1:
        raise RuntimeError(
            "pinned Qwen3 TP reference requires the scheduler worker to force "
            "VLLM_USE_V1=1 exactly once")
    return {
        "vllm_version": version,
        "source_commit": _VLLM_SOURCE_COMMIT,
        "files": tuple(files),
        "midlayer_forces_vllm_v1": True,
        "residual_rmsnorm_boundaries": True,
    }


def pinned_llama_source_evidence() -> dict[str, object]:
    """Verify the exact Llama forward source without importing vLLM."""

    version = importlib.metadata.version("vllm")
    if version != _VLLM_VERSION:
        raise RuntimeError(
            f"Llama source reference requires vLLM {_VLLM_VERSION}, "
            f"got {version}")
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the pinned vLLM package")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"vLLM source reference requires one package root: {locations!r}")
    package_root = Path(locations[0]).resolve()
    files = []
    for relative_path, expected_hash, required_symbols in (
            _LLAMA_SOURCE_CONTRACT + _LLAMA_TP_SOURCE_CONTRACT):
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"pinned vLLM Llama source is missing: {source_path}")
        source_bytes = source_path.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned vLLM Llama source changed: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}")
        tree = ast.parse(source_bytes, filename=str(source_path))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        for class_name, method_name in required_symbols:
            if class_name not in classes:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks class {class_name}")
            methods = {
                node.name for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks symbol "
                    f"{class_name}.{method_name}")
        source = source_bytes.decode("utf-8")
        if relative_path == "model_executor/models/llama.py":
            for fragment in _LLAMA_REQUIRED_SOURCE_FRAGMENTS:
                count = source.count(fragment)
                if count != 1:
                    raise RuntimeError(
                        f"pinned source {relative_path} expected fragment "
                        f"once, got {count}: {fragment!r}")
        if relative_path in _LLAMA_TP_REQUIRED_SOURCE_FRAGMENTS:
            for fragment in _LLAMA_TP_REQUIRED_SOURCE_FRAGMENTS[
                    relative_path]:
                count = source.count(fragment)
                if count != 1:
                    raise RuntimeError(
                        f"pinned TP source {relative_path} expected fragment "
                        f"once, got {count}: {fragment!r}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
            "symbols": tuple(
                f"{class_name}.{method_name}"
                for class_name, method_name in required_symbols
            ),
        })
    scheduler_process = (
        Path(__file__).resolve().parents[2]
        / "src/serving/backend/vllm_scheduler_process.py"
    )
    scheduler_source = scheduler_process.read_text(encoding="utf-8")
    v1_assignment = 'os.environ["VLLM_USE_V1"] = "1"'
    if scheduler_source.count(v1_assignment) != 1:
        raise RuntimeError(
            "pinned Llama TP reference requires the scheduler worker to "
            "force VLLM_USE_V1=1 exactly once")
    return {
        "vllm_version": version,
        "source_commit": _VLLM_SOURCE_COMMIT,
        "files": tuple(files),
        "derived_head_dim": True,
        "fused_qkv_projection": True,
        "rotary_qk_then_attention": True,
        "residual_rmsnorm_boundaries": True,
        "fused_gate_up_silu_down": True,
        "parallel_lm_head": True,
        "tp_q_heads_partitioned": True,
        "tp_single_kv_head_replicated": True,
        "row_parallel_all_reduce": True,
        "vocab_embedding_all_reduce": True,
        "lm_head_all_gather": True,
        "midlayer_forces_vllm_v1": True,
    }


def pinned_kimi_mla_0_11_source_evidence() -> dict[str, object]:
    """Verify the exact vLLM source defining Kimi's executed MLA branches."""

    version = importlib.metadata.version("vllm")
    if version != _VLLM_VERSION:
        raise RuntimeError(
            f"Kimi MLA source reference requires vLLM {_VLLM_VERSION}, "
            f"got {version}")
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the pinned vLLM package")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            "vLLM Kimi MLA source reference requires one package root: "
            f"{locations!r}")
    package_root = Path(locations[0]).resolve()
    files = []
    mla_class_source = None
    for relative_path, expected_hash, required_symbols in (
            _KIMI_MLA_0_11_SOURCE_CONTRACT):
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"pinned vLLM Kimi MLA source is missing: {source_path}")
        source_bytes = source_path.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned vLLM Kimi MLA source changed: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}")
        tree = ast.parse(source_bytes, filename=str(source_path))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        for class_name, method_name in required_symbols:
            if class_name not in classes:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks class {class_name}")
            methods = {
                node.name for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks symbol "
                    f"{class_name}.{method_name}")
        source = source_bytes.decode("utf-8")
        if relative_path == "model_executor/models/deepseek_v2.py":
            if "DeepseekV2MLAAttention" not in classes:
                raise RuntimeError(
                    "pinned Kimi MLA source lacks DeepseekV2MLAAttention")
            mla_class_source = ast.get_source_segment(
                source, classes["DeepseekV2MLAAttention"])
            if mla_class_source is None:
                raise RuntimeError(
                    "cannot recover pinned DeepseekV2MLAAttention source")
            for fragment in _KIMI_MLA_0_11_ATTENTION_SOURCE_FRAGMENTS:
                count = mla_class_source.count(fragment)
                if count != 1:
                    raise RuntimeError(
                        "pinned Kimi MLA projection source changed: "
                        f"expected fragment once, got {count}: {fragment!r}")
            if "class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):" not in source:
                raise RuntimeError(
                    "pinned DeepseekV3 model no longer delegates to "
                    "DeepseekV2ForCausalLM")
        elif relative_path == "model_executor/models/registry.py":
            mapping = (
                '"DeepseekV3ForCausalLM": '
                '("deepseek_v2", "DeepseekV3ForCausalLM"),')
            if source.count(mapping) != 1:
                raise RuntimeError(
                    "pinned vLLM registry changed the DeepseekV3 mapping")
        elif relative_path == "model_executor/layers/mla.py":
            mla_forward = next(
                node for node in classes["MultiHeadLatentAttention"].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "forward_native")
            mla_forward_source = ast.get_source_segment(source, mla_forward)
            if mla_forward_source is None:
                raise RuntimeError(
                    "cannot recover pinned MultiHeadLatentAttention.forward_native")
            for fragment in (
                    "qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]",
                    "q = self.q_b_proj(q_c)[0]",
                    "attn_out = self.mla_attn(",
                    "return self.o_proj(attn_out)[0]"):
                if fragment not in mla_forward_source:
                    raise RuntimeError(
                        "pinned MLA layer execution branch changed: "
                        f"{fragment!r}")
            if "self.kv_b_proj(" in mla_forward_source:
                raise RuntimeError(
                    "pinned MLA layer unexpectedly executes kv_b_proj outside "
                    "the attention backend")
        elif relative_path == "v1/attention/backends/mla/common.py":
            def method_node(class_name, method_name):
                method = next(
                    node for node in classes[class_name].body
                    if isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == method_name)
                return method

            def method_source(class_name, method_name):
                method = method_node(class_name, method_name)
                segment = ast.get_source_segment(source, method)
                if segment is None:
                    raise RuntimeError(
                        "cannot recover pinned MLA method source: "
                        f"{class_name}.{method_name}")
                return segment

            prefill_source = method_source(
                "MLACommonImpl", "_forward_prefill")
            forward_source = method_source("MLACommonImpl", "forward")
            v_up_source = method_source("MLACommonBaseImpl", "_v_up_proj")
            for fragment in (
                    "self.kv_b_proj(kv_c_normed)",
                    "self._run_prefill_new_tokens("):
                if fragment not in prefill_source:
                    raise RuntimeError(
                        "pinned MLA prefill execution branch changed: "
                        f"{fragment!r}")
            for fragment in (
                    "self._forward_prefill(",
                    "torch.bmm(decode_q_nope, self.W_UK_T",
                    "self._forward_decode(",
                    "self._v_up_proj(attn_out"):
                if fragment not in forward_source:
                    raise RuntimeError(
                        "pinned MLA mixed/decode execution branch changed: "
                        f"{fragment!r}")
            forward_method = method_node("MLACommonImpl", "forward")
            forward_calls_kv_b_proj = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "kv_b_proj"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                for node in ast.walk(forward_method))
            if forward_calls_kv_b_proj:
                raise RuntimeError(
                    "pinned MLA decode branch unexpectedly executes kv_b_proj")
            if "torch.bmm(x, self.W_UV" not in v_up_source:
                raise RuntimeError(
                    "pinned MLA decode v-up projection branch changed")
        else:
            raise AssertionError(
                f"unhandled Kimi MLA source file: {relative_path!r}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
            "symbols": tuple(
                f"{class_name}.{method_name}"
                for class_name, method_name in required_symbols
            ),
        })
    if mla_class_source is None:
        raise RuntimeError("pinned Kimi MLA class source was not inspected")
    return {
        "vllm_version": version,
        "source_commit": _VLLM_SOURCE_COMMIT,
        "files": tuple(files),
        "architecture": "DeepseekV3ForCausalLM",
        "implementation": "DeepseekV2MLAAttention",
        "runtime_layer": "MultiHeadLatentAttention.forward_native",
        "runtime_backend": "MLACommonImpl.forward",
        "prefill_executes_kv_b_proj": True,
        "decode_executes_kv_b_proj": False,
        "decode_executes_absorbed_q_down": True,
        "decode_executes_v_up": True,
        "fused_qkv_a_projection": True,
        "fused_qkv_a_tensor_parallel": "replicated_disable_tp",
        "q_b_tensor_parallel": "column_parallel",
        "kv_b_tensor_parallel": "column_parallel",
        "o_projection_tensor_parallel": "row_parallel",
        "official_model_revision": (
            _KIMI_OFFICIAL_SOURCE_EVIDENCE["revision"]),
        "official_configuration_sha256": (
            _KIMI_OFFICIAL_SOURCE_EVIDENCE["configuration_sha256"]),
        "official_modeling_sha256": (
            _KIMI_OFFICIAL_SOURCE_EVIDENCE["modeling_sha256"]),
    }


def pinned_kimi_mla_source_evidence(
    vllm_checkout: Path,
) -> dict[str, object]:
    """Verify Kimi MLA against the fixed vLLM 0.27.1 source checkout."""

    if not isinstance(vllm_checkout, Path):
        raise TypeError("vllm_checkout must be a Path")
    expected_commit = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
    actual_commit = subprocess.check_output(
        ("git", "-C", str(vllm_checkout), "rev-parse", "HEAD"),
        text=True,
    ).strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            "Kimi MLA source checkout changed: expected "
            f"{expected_commit}, got {actual_commit}")
    source_contract = (
        (
            "vllm/model_executor/models/deepseek_v2.py",
            "bcbd9cde689ffe3f5f271aff3a5e01351b05a68371192cac9d65cb5ffc6b4ba9",
            (
                ("DeepseekV2MLAAttention", "__init__"),
                ("DeepseekV2MLAAttention", "forward"),
                ("DeepseekV2MoE", "__init__"),
                ("DeepseekV2MoE", "forward"),
            ),
            (
                "DeepSeekV2FusedQkvAProjLinear(",
                "self.kv_b_proj = ColumnParallelLinear(",
                "self.o_proj = RowParallelLinear(",
                "is_neox_style=False",
                '"deepseek_yarn"',
                "class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):",
            ),
        ),
        (
            "vllm/model_executor/layers/mla.py",
            "ccdbafa998b1a7f7db47527df85efdfa4bc3027f6ec769fd4c30b3aee963a81b",
            (("MultiHeadLatentAttentionWrapper", "forward"),),
            (
                "qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]",
                "q = q_proj_layer(q_proj_input)[0]",
                "if self.indexer and self.is_sparse and not self.skip_topk:",
                "attn_out = self.mla_attn(",
                "return self.o_proj(attn_out)[0]",
            ),
        ),
        (
            "vllm/model_executor/layers/attention/mla_attention.py",
            "272c01ff03e52cdfaed6fc7bc2e3e5f641d35910ea0bf5365b928c9401c4df5f",
            (
                ("MLAAttention", "forward_impl"),
                ("MLAAttention", "_v_up_proj"),
                ("MLACommonBaseImpl", "_compute_prefill_context"),
                ("MLACommonImpl", "forward_mqa"),
            ),
            (
                "num_mqa_tokens = attn_metadata.num_decode_tokens",
                "self.impl.forward_mha(",
                "torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)",
                "kv_nope = self.kv_b_proj(kv_c_normed)[0].view(",
                "torch.bmm(x, self.W_UV, out=out.transpose(0, 1))",
            ),
        ),
        (
            "vllm/model_executor/models/registry.py",
            "f1dae30a951fda360151c8de9dd605d01c7138a69f70fe518286d57da69dc463",
            (),
            (
                '"DeepseekV3ForCausalLM": ("deepseek_v2", '
                '"DeepseekV3ForCausalLM"),',
            ),
        ),
    )
    files = []
    for relative_path, expected_hash, required_symbols, fragments in (
            source_contract):
        source_path = vllm_checkout / relative_path
        source_bytes = source_path.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned vLLM Kimi source changed: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}")
        tree = ast.parse(source_bytes, filename=str(source_path))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        for class_name, method_name in required_symbols:
            if class_name not in classes:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks class {class_name}")
            methods = {
                node.name for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks symbol "
                    f"{class_name}.{method_name}")
        source = source_bytes.decode("utf-8")
        for fragment in fragments:
            if fragment not in source:
                raise RuntimeError(
                    f"pinned Kimi MLA source fragment changed in "
                    f"{relative_path}: {fragment!r}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
        })
    return {
        "vllm_version": "0.27.1",
        "source_commit": actual_commit,
        "files": tuple(files),
        "architecture": "DeepseekV3ForCausalLM",
        "implementation": "DeepseekV2MLAAttention",
        "runtime_layer": "MultiHeadLatentAttentionWrapper.forward",
        "runtime_backend": "MLAAttention.forward_impl",
        "prefill_executes_kv_b_proj": True,
        "decode_executes_kv_b_proj": False,
        "decode_executes_absorbed_q_down": True,
        "decode_executes_v_up": True,
        "fused_qkv_a_projection": True,
        "fused_qkv_a_tensor_parallel": "replicated",
        "q_b_tensor_parallel": "column_parallel",
        "kv_b_tensor_parallel": "column_parallel",
        "o_projection_tensor_parallel": "row_parallel",
        "official_model_revision": (
            _KIMI_OFFICIAL_SOURCE_EVIDENCE["revision"]),
        "official_configuration_sha256": (
            _KIMI_OFFICIAL_SOURCE_EVIDENCE["configuration_sha256"]),
        "official_modeling_sha256": (
            _KIMI_OFFICIAL_SOURCE_EVIDENCE["modeling_sha256"]),
    }


def kimi_k2_6_mla_projection_reference(
    model_path: Path,
    *,
    token_count: int,
) -> tuple[dict[str, object], ...]:
    """Derive TP=1 Kimi MLA projection shapes independently from config."""

    if not isinstance(model_path, Path):
        raise TypeError("model_path must be a Path")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("Kimi MLA token_count must be an integer")
    if token_count <= 0:
        raise ValueError("Kimi MLA token_count must be positive")
    provenance = json.loads(
        (model_path / "model_provenance.json").read_text(encoding="utf-8"))
    if provenance["model_id"] != "moonshotai/Kimi-K2.6":
        raise ValueError("Kimi MLA reference requires moonshotai/Kimi-K2.6")
    if provenance["revision"] != (
            "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"):
        raise ValueError("Kimi MLA reference requires the pinned model revision")
    root = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    if root["model_type"] != "kimi_k25":
        raise ValueError("Kimi MLA reference requires model_type=kimi_k25")
    if root["architectures"] != ["KimiK25ForConditionalGeneration"]:
        raise ValueError("Kimi MLA reference requires the Kimi-K2.5 container")
    text = root["text_config"]
    if text["model_type"] != "kimi_k2":
        raise ValueError("Kimi MLA reference requires text model kimi_k2")
    if text["architectures"] != ["DeepseekV3ForCausalLM"]:
        raise ValueError(
            "Kimi MLA reference requires DeepseekV3ForCausalLM text")
    if root["dtype"] != "bfloat16" or text["dtype"] != "bfloat16":
        raise ValueError("Kimi MLA reference currently requires BF16 compute")
    quantization = text["quantization_config"]
    group = quantization["config_groups"]["group_0"]
    if group["targets"] != ["Linear"]:
        raise ValueError("Kimi MLA reference requires Linear quantization")
    if "re:.*self_attn.*" not in quantization["ignore"]:
        raise ValueError(
            "Kimi MLA reference requires BF16 self-attention weights")

    fields = {
        name: text[name]
        for name in (
            "hidden_size", "num_attention_heads", "q_lora_rank",
            "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim",
        )
    }
    for name, value in fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Kimi MLA config {name} must be positive")
    hidden = fields["hidden_size"]
    heads = fields["num_attention_heads"]
    q_lora = fields["q_lora_rank"]
    kv_lora = fields["kv_lora_rank"]
    qk_nope = fields["qk_nope_head_dim"]
    qk_rope = fields["qk_rope_head_dim"]
    value_head = fields["v_head_dim"]
    shapes = (
        (
            "fused_qkv_a_proj",
            (token_count, q_lora + kv_lora + qk_rope, hidden),
            "DeepseekV2MLAAttention.fused_qkv_a_proj:"
            "MergedColumnParallelLinear(disable_tp=True)",
        ),
        (
            "q_b_proj",
            (token_count, heads * (qk_nope + qk_rope), q_lora),
            "DeepseekV2MLAAttention.q_b_proj:ColumnParallelLinear",
        ),
        (
            "kv_b_proj",
            (token_count, heads * (qk_nope + value_head), kv_lora),
            "DeepseekV2MLAAttention.kv_b_proj:ColumnParallelLinear",
        ),
        (
            "o_proj",
            (token_count, hidden, heads * value_head),
            "DeepseekV2MLAAttention.o_proj:RowParallelLinear",
        ),
    )
    return tuple({
        "position": position,
        "gemm_mnk": gemm_mnk,
        "compute_dtype": "bf16",
        "weight_dtype": "bf16",
        "source_symbol": source_symbol,
        "evidence_status": "validated_static_source_semantics",
    } for position, gemm_mnk, source_symbol in shapes)


def pinned_glm_5_2_source_evidence() -> dict[str, object]:
    """Return pinned Transformers execution and vLLM source-audit evidence."""

    return {
        "framework": _GLM_5_2_VLLM_SOURCE_EVIDENCE["framework"],
        "framework_version": (
            _GLM_5_2_VLLM_SOURCE_EVIDENCE["framework_version"]),
        "source_commit": _GLM_5_2_VLLM_SOURCE_EVIDENCE["source_commit"],
        "model_revision": _GLM_5_2_SOURCE_EVIDENCE["model_revision"],
        "model_config_sha256": (
            "185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"),
        "transformers_version": (
            _GLM_5_2_SOURCE_EVIDENCE["framework_version"]),
        "transformers_source_commit": (
            _GLM_5_2_SOURCE_EVIDENCE["source_commit"]),
        "transformers_configuration_relative_path": (
            "src/transformers/models/glm_moe_dsa/"
            "configuration_glm_moe_dsa.py"),
        "transformers_configuration_sha256": (
            _GLM_5_2_SOURCE_EVIDENCE["configuration_sha256"]),
        "transformers_modeling_relative_path": (
            "src/transformers/models/glm_moe_dsa/"
            "modeling_glm_moe_dsa.py"),
        "transformers_modeling_sha256": (
            _GLM_5_2_SOURCE_EVIDENCE["modeling_sha256"]),
        "vllm_python_requirement": (
            _GLM_5_2_VLLM_SOURCE_EVIDENCE["python_requirement"]),
        "vllm_pytorch_requirement": (
            _GLM_5_2_VLLM_SOURCE_EVIDENCE["pytorch_requirement"]),
        "vllm_transformers_requirement": (
            _GLM_5_2_VLLM_SOURCE_EVIDENCE["transformers_requirement"]),
        "vllm_source_files": (
            _GLM_5_2_VLLM_SOURCE_EVIDENCE["source_files"]),
        "architecture": "GlmMoeDsaForCausalLM",
        "vllm_registration": (
            "registry.GlmMoeDsaForCausalLM->deepseek_v2.GlmMoeDsaForCausalLM"),
        "vllm_configuration_status": "registered_source_audited",
        "earliest_registered_vllm_version": "0.16.0",
        "earliest_registered_vllm_commit": (
            "89a77b10846fd96273cce78d86d2556ea582d26e"),
        "vllm_selection_reason": (
            "0.27.1 implements cross-layer shared-indexer reuse"),
        "attention_implementation": "DeepseekV2MLAAttention",
        "indexer_implementation": "Indexer/SparseAttnIndexer",
        "moe_implementation": "DeepseekV2MoE/FusedMoE",
        "qkv_a_tensor_parallel": "replicated_fused",
        "q_b_tensor_parallel": "column_parallel",
        "kv_b_tensor_parallel": "column_parallel",
        "o_projection_tensor_parallel": "row_parallel",
        "main_rope_layout": "interleaved",
        "indexer_rope_layout": "interleaved",
        "indexer_cache": "fp8_e4m3_key_with_ue8m0_scale",
        "attention_cache": "compressed_bf16_mla",
        "shared_index_status": "shared_topk_buffer_reuse",
        "mtp_execution_status": (
            "vllm_0_27_1_has_no_registered_glm_moe_dsa_mtp_model"),
        "transformers_execution_branch_validated": True,
        "transformers_execution_validation": (
            "tiny_a800_bf16_prefill_decode_dynamic_indexed_cache"),
        "validation_python_version": "3.12.9",
        "validation_pytorch_version": "2.7.1+cu126",
        "validation_transformers_version": "5.10.2",
        "validation_device": "NVIDIA A800-SXM4-80GB",
        "validation_model_dtype": "bf16",
        "validation_model_config": {
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "kv_lora_rank": 8,
            "q_lora_rank": 16,
            "qk_rope_head_dim": 8,
            "qk_nope_head_dim": 8,
            "v_head_dim": 8,
            "index_topk": 4,
            "index_head_dim": 8,
            "index_n_heads": 2,
            "indexer_types": ("full", "shared"),
            "mlp_layer_types": ("dense", "sparse"),
            "num_nextn_predict_layers": 0,
        },
        "validation_prefill_input_ids": ((1, 3, 5),),
        "validation_decode_input_ids": ((7,),),
        "validation_logits_shapes": ((1, 3, 32), (1, 1, 32)),
        "validation_logits_dtype": "bf16",
        "validation_cache_sequence_length": 4,
        "validation_indexer_modes": ("full", "shared"),
        "validation_cache_layer_type": "DynamicCache",
        "validation_full_indexer_cache_shape": (1, 4, 8),
        "vllm_source_audited": True,
        "vllm_execution_branch_validated": False,
        "runtime_contract_status": "vllm_blocked_pytorch_2_13_required",
        "evidence_status": (
            "transformers_a800_executed_vllm_source_audited"),
    }


def glm_5_2_dsa_projection_reference(
    model_path: Path,
    *,
    token_count: int,
    tp_size: int = 1,
) -> tuple[dict[str, object], ...]:
    """Derive GLM-5.2 DSA projection shapes from the official fixture."""

    if not isinstance(model_path, Path):
        raise TypeError("GLM-5.2 model_path must be a Path")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("GLM-5.2 token_count must be an integer")
    if token_count <= 0:
        raise ValueError("GLM-5.2 token_count must be positive")
    if isinstance(tp_size, bool) or not isinstance(tp_size, int):
        raise TypeError("GLM-5.2 tp_size must be an integer")
    if tp_size <= 0:
        raise ValueError("GLM-5.2 tp_size must be positive")
    provenance = json.loads(
        (model_path / "model_provenance.json").read_text(encoding="utf-8"))
    if provenance["model_id"] != "zai-org/GLM-5.2":
        raise ValueError("GLM DSA reference requires zai-org/GLM-5.2")
    if provenance["revision"] != _GLM_5_2_SOURCE_EVIDENCE[
            "model_revision"]:
        raise ValueError("GLM DSA reference requires the pinned model revision")
    config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    if config["model_type"] != "glm_moe_dsa":
        raise ValueError("GLM DSA reference requires model_type=glm_moe_dsa")
    if config["architectures"] != ["GlmMoeDsaForCausalLM"]:
        raise ValueError(
            "GLM DSA reference requires GlmMoeDsaForCausalLM")
    if config["dtype"] != "bfloat16":
        raise ValueError("GLM DSA reference requires BF16 model weights")
    fields = {
        name: config[name]
        for name in (
            "hidden_size", "num_attention_heads", "q_lora_rank",
            "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "index_n_heads", "index_head_dim",
        )
    }
    for name, value in fields.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"GLM DSA config {name} must be positive")
    hidden = fields["hidden_size"]
    heads = fields["num_attention_heads"]
    q_lora = fields["q_lora_rank"]
    kv_lora = fields["kv_lora_rank"]
    qk_nope = fields["qk_nope_head_dim"]
    qk_rope = fields["qk_rope_head_dim"]
    value_head = fields["v_head_dim"]
    index_heads = fields["index_n_heads"]
    index_head = fields["index_head_dim"]
    sharded_dimensions = (
        heads * (qk_nope + qk_rope),
        heads * (qk_nope + value_head),
        heads * value_head,
    )
    if any(dimension % tp_size for dimension in sharded_dimensions):
        raise ValueError(
            "GLM DSA projection dimensions must divide tp_size")
    shapes = (
        (
            "fused_qkv_a_proj",
            (token_count, q_lora + kv_lora + qk_rope, hidden),
            "DeepseekV2MLAAttention.fused_qkv_a_proj:replicated",
        ),
        (
            "q_b_proj",
            (token_count, heads * (qk_nope + qk_rope) // tp_size, q_lora),
            "DeepseekV2MLAAttention.q_b_proj:colwise",
        ),
        (
            "kv_b_proj",
            (token_count, heads * (qk_nope + value_head) // tp_size, kv_lora),
            "DeepseekV2MLAAttention.kv_b_proj:colwise",
        ),
        (
            "o_proj",
            (token_count, hidden, heads * value_head // tp_size),
            "DeepseekV2MLAAttention.o_proj:rowwise",
        ),
        (
            "indexer_wq_b",
            (token_count, index_heads * index_head, q_lora),
            "Indexer.wq_b:replicated",
        ),
        (
            "indexer_wk_weights_proj",
            (token_count, index_head + index_heads, hidden),
            "Indexer.wk_weights_proj:MergedColumnParallelLinear(disable_tp=True)",
        ),
    )
    references = []
    for position, gemm_mnk, source_symbol in shapes:
        references.append({
            "position": position,
            "gemm_mnk": gemm_mnk,
            "compute_dtype": "bf16",
            "weight_dtype": "bf16",
            "source_symbol": source_symbol,
            "evidence_status": "vllm_0_27_1_source_audited",
        })
    return tuple(references)


def target_model_layer_manifest_reference(
    model_path: Path,
    *,
    phase: str,
    layer_index: int,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> dict[str, object]:
    """Derive one representative target layer manifest from fixed config."""

    if not isinstance(model_path, Path):
        raise TypeError("target model_path must be a Path")
    if phase not in ("prefill", "decode", "mixed"):
        raise ValueError(f"unsupported target reference phase: {phase!r}")
    if (not query_lengths or len(query_lengths) != len(kv_lengths)
            or any(length <= 0 for length in query_lengths + kv_lengths)):
        raise ValueError(
            "target reference requires aligned positive query/KV lengths")
    if phase == "prefill" and len(query_lengths) != 1:
        raise ValueError("prefill reference requires one request")
    if phase == "decode" and any(length != 1 for length in query_lengths):
        raise ValueError("decode reference requires unit query lengths")
    if phase == "mixed" and query_lengths != (3, 1):
        raise ValueError("mixed reference is fixed to query lengths (3, 1)")
    provenance = json.loads(
        (model_path / "model_provenance.json").read_text(encoding="utf-8"))
    root = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    model_id = provenance["model_id"]
    if model_id == "moonshotai/Kimi-K2.6":
        if provenance["revision"] != _KIMI_OFFICIAL_SOURCE_EVIDENCE[
                "revision"]:
            raise ValueError("Kimi manifest requires the pinned revision")
        config = root["text_config"]
        if config["model_type"] != "kimi_k2":
            raise ValueError("Kimi manifest requires kimi_k2 text config")
        if layer_index == 0:
            layer_type = "dense"
        elif layer_index == config["first_k_dense_replace"]:
            layer_type = "moe"
        else:
            raise ValueError(
                "Kimi manifest layer must be representative dense layer 0 "
                "or first MoE layer")
        attention_layout = "mla"
        indexer_type = None
        router_dtype = "bf16"
        routed_weight_dtype = "int4"
        rotary_layout = "interleaved_deepseek_yarn"
        rope_attributes = {
            "rope_layout": "interleaved",
            "rope_type": "deepseek_yarn",
            "rope_theta": config["rope_theta"],
            "rope_factor": config["rope_scaling"]["factor"],
            "rope_original_max_position_embeddings": (
                config["rope_scaling"][
                    "original_max_position_embeddings"]),
            "rope_beta_fast": config["rope_scaling"]["beta_fast"],
            "rope_beta_slow": config["rope_scaling"]["beta_slow"],
            "rope_mscale": config["rope_scaling"]["mscale"],
            "rope_mscale_all_dim": config["rope_scaling"][
                "mscale_all_dim"],
        }
    elif model_id == "zai-org/GLM-5.2":
        if provenance["revision"] != _GLM_5_2_SOURCE_EVIDENCE[
                "model_revision"]:
            raise ValueError("GLM manifest requires the pinned revision")
        config = root
        if config["model_type"] != "glm_moe_dsa":
            raise ValueError("GLM manifest requires glm_moe_dsa config")
        layer_type = config["mlp_layer_types"][layer_index]
        indexer_type = config["indexer_types"][layer_index]
        if (layer_type, indexer_type) not in (
                ("dense", "full"), ("sparse", "shared")):
            raise ValueError(
                "GLM manifest requires a full-indexer Dense or "
                "shared-indexer MoE representative")
        if layer_type == "sparse":
            layer_type = "moe"
        elif layer_type == "dense":
            layer_type = "dense"
        else:
            raise AssertionError(
                f"unhandled GLM representative layer type: {layer_type!r}")
        attention_layout = "dsa_mla"
        router_dtype = "fp32"
        routed_weight_dtype = "bf16"
        rotary_layout = "interleaved_default"
        rope_attributes = {
            "rope_layout": "interleaved",
            "rope_type": "default",
            "rope_theta": config["rope_parameters"]["rope_theta"],
            "rope_factor": None,
            "rope_original_max_position_embeddings": None,
            "rope_beta_fast": None,
            "rope_beta_slow": None,
            "rope_mscale": None,
            "rope_mscale_all_dim": None,
        }
    else:
        raise ValueError(f"unsupported target model ID: {model_id!r}")

    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    q_lora = config["q_lora_rank"]
    kv_lora = config["kv_lora_rank"]
    qk_nope = config["qk_nope_head_dim"]
    qk_rope = config["qk_rope_head_dim"]
    value_head = config["v_head_dim"]
    token_count = sum(query_lengths)
    prefill_indices = tuple(
        index for index, length in enumerate(query_lengths)
        if phase in ("prefill", "mixed") and length > 1)
    decode_indices = tuple(
        index for index, length in enumerate(query_lengths)
        if phase in ("decode", "mixed") and length == 1)
    nodes = []

    def add(
        suffix: str,
        kind: str,
        variant: str,
        dtype: str,
        shape: dict[str, int],
        predecessors: tuple[str, ...],
        attributes: dict[str, object] | None = None,
    ) -> None:
        nodes.append({
            "suffix": suffix,
            "kind": kind,
            "variant": variant,
            "dtype": dtype,
            "shape": shape,
            "predecessors": predecessors,
            "attributes": {} if attributes is None else attributes,
        })

    add(
        "input-norm", "norm", "rmsnorm", "bf16",
        {"rows": token_count, "hidden": hidden},
        ("__previous_layer__",),
    )
    add(
        "fused-qkv-a-proj", "linear", "gemm", "bf16",
        {"b": 1, "m": token_count,
         "n": q_lora + kv_lora + qk_rope, "k": hidden},
        ("input-norm",),
    )
    add(
        "q-a-norm", "norm", "rmsnorm", "bf16",
        {"rows": token_count, "hidden": q_lora},
        ("fused-qkv-a-proj",),
    )
    add(
        "q-b-proj", "linear", "gemm", "bf16",
        {"b": 1, "m": token_count,
         "n": heads * (qk_nope + qk_rope), "k": q_lora},
        ("q-a-norm",),
    )
    add(
        "kv-a-norm", "norm", "rmsnorm", "bf16",
        {"rows": token_count, "hidden": kv_lora},
        ("fused-qkv-a-proj",),
    )
    add(
        "main-rotary", "rotary", "rotary", "bf16",
        {
            "elements": token_count * (heads + 1) * qk_rope,
            "seq_len": token_count,
            "num_query_heads": heads,
            "num_key_heads": 1,
            "head_dim": qk_rope,
        },
        ("q-b-proj", "fused-qkv-a-proj"),
        {"layout": rotary_layout, **rope_attributes},
    )
    if attention_layout == "mla":
        cache_variant = "mla_kv_cache"
    elif attention_layout == "dsa_mla":
        cache_variant = "mla_kv_cache"
    else:
        raise AssertionError(
            f"unhandled target attention layout: {attention_layout!r}")
    add(
        "compressed-kv-update", "kv_update", cache_variant, "bf16",
        {
            "elements": token_count * (kv_lora + qk_rope),
            "seq_len": token_count,
            "compressed_dim": kv_lora + qk_rope,
        },
        ("main-rotary", "kv-a-norm"),
        {
            "cache_layout": "paged_mla",
            "cache_write_semantics": "append",
        },
    )

    topk_source = None
    if attention_layout == "dsa_mla":
        index_heads = config["index_n_heads"]
        index_head = config["index_head_dim"]
        selected = min(config["index_topk"], max(kv_lengths))
        if indexer_type == "full":
            add(
                "indexer-wq-b", "linear", "gemm", "bf16",
                {"b": 1, "m": token_count,
                 "n": index_heads * index_head, "k": q_lora},
                ("q-a-norm",),
            )
            add(
                "indexer-wk-weights-proj", "linear", "gemm", "bf16",
                {"b": 1, "m": token_count,
                 "n": index_head + index_heads, "k": hidden},
                ("input-norm",),
            )
            add(
                "indexer-k-norm", "norm", "layernorm", "bf16",
                {"rows": token_count, "hidden": index_head},
                ("indexer-wk-weights-proj",),
                {"affine": True, "epsilon": 1e-5},
            )
            add(
                "indexer-rotary", "rotary", "rotary", "bf16",
                {
                    "elements": token_count * (index_heads + 1) * qk_rope,
                    "seq_len": token_count,
                    "num_query_heads": index_heads,
                    "num_key_heads": 1,
                    "head_dim": qk_rope,
                },
                ("indexer-wq-b", "indexer-k-norm"),
                {"layout": "interleaved_default", **rope_attributes},
            )
            add(
                "indexer-k-update", "kv_update", "dsa_fp8_index_cache",
                "uint8",
                {
                    "elements": token_count * (index_head + 4),
                    "seq_len": token_count,
                    "key_dim": index_head,
                    "scale_bytes_per_token": 4,
                },
                ("indexer-rotary",),
                {
                    "cache_layout": "paged_mla_indexer",
                    "cache_write_semantics": "append",
                    "quantization": "fp8_e4m3_with_ue8m0_scale",
                },
            )
            topk_source = "indexer-score-topk"
            add(
                topk_source, "eager_attn", f"dsa_indexer_{phase}",
                "int32",
                {
                    "batch": len(query_lengths),
                    "q_len": max(query_lengths),
                    "kv_len": max(kv_lengths),
                    "num_query_heads": index_heads,
                    "head_dim": index_head,
                    "topk": selected,
                },
                ("indexer-k-update", "indexer-wk-weights-proj"),
                {
                    "causal": True,
                    "score_accumulator_dtype": "fp32",
                    "score_function": "weighted_dot_product_topk",
                    "cache_layout": "paged_mla_indexer",
                    "cache_quantization": "fp8_e4m3_with_ue8m0_scale",
                    "q_lens": query_lengths,
                    "kv_lens": kv_lengths,
                    "request_phases": tuple(
                        "decode" if length == 1 else "prefill"
                        for length in query_lengths),
                    "topk_capacity": config["index_topk"],
                    "topk_index_dtype": "int32",
                    "topk_sentinel": -1,
                },
            )
        elif indexer_type == "shared":
            topk_source = "indexer-topk-reuse"
            add(
                topk_source, "gather", "dsa_index_reuse", "int32",
                {"elements": token_count * selected},
                ("__prior_full_indexer__",),
                {
                    "reuse_semantics": "alias",
                    "topk_capacity": config["index_topk"],
                    "topk_index_dtype": "int32",
                },
            )
        else:
            raise AssertionError(f"unexpected indexer type: {indexer_type}")

    attention_outputs = []
    if prefill_indices:
        prefill_tokens = sum(query_lengths[index] for index in prefill_indices)
        prefill_kv_lengths = tuple(kv_lengths[index] for index in prefill_indices)
        kv_b_suffix = "prefill-kv-b-proj"
        add(
            kv_b_suffix, "linear", "gemm", "bf16",
            {
                "b": 1,
                "m": sum(prefill_kv_lengths),
                "n": heads * (qk_nope + value_head),
                "k": kv_lora,
            },
            ("compressed-kv-update",),
        )
        if attention_layout == "mla":
            attention_suffix = "prefill-mla-attention"
            attention_kind = "flash_attn"
            attention_variant = "mla_prefill"
        else:
            attention_suffix = "prefill-dsa-attention"
            attention_kind = "eager_attn"
            attention_variant = "dsa_sparse_prefill"
        attention_predecessors = ["compressed-kv-update"]
        if topk_source is not None:
            attention_predecessors.append(topk_source)
        attention_predecessors.append(kv_b_suffix)
        attention_shape = {
            "batch": len(prefill_indices),
            "q_len": max(query_lengths[index] for index in prefill_indices),
            "kv_len": max(prefill_kv_lengths),
            "qk_head_dim": qk_nope + qk_rope,
            "v_head_dim": value_head,
            "num_query_heads": heads,
            "kv_lora_rank": kv_lora,
        }
        if attention_layout == "dsa_mla":
            del attention_shape["kv_lora_rank"]
            attention_shape["topk"] = min(
                config["index_topk"], max(prefill_kv_lengths))
        attention_attributes = {
            "causal": True,
            "kv_layout": "mla",
            "kv_dist": "uniform",
            "q_lens": tuple(query_lengths[index] for index in prefill_indices),
            "kv_lens": prefill_kv_lengths,
        }
        if attention_layout == "dsa_mla":
            attention_attributes.update({
                "indexer_type": indexer_type,
                "selected_kv_lens": tuple(
                    min(config["index_topk"], length)
                    for length in prefill_kv_lengths),
                "topk_capacity": config["index_topk"],
                "topk_index_dtype": "int32",
            })
        add(
            attention_suffix, attention_kind, attention_variant, "bf16",
            attention_shape, tuple(attention_predecessors),
            attention_attributes,
        )
        attention_outputs.append(attention_suffix)
    if decode_indices:
        decode_tokens = sum(query_lengths[index] for index in decode_indices)
        decode_kv_lengths = tuple(kv_lengths[index] for index in decode_indices)
        add(
            "decode-q-down-proj", "linear", "batched_gemm", "bf16",
            {
                "b": heads,
                "m": decode_tokens,
                "n": kv_lora,
                "k": qk_nope,
            },
            ("main-rotary",),
            {
                "batch_layout": "strided",
                "a_batch_stride_elements": decode_tokens * qk_nope,
                "b_batch_stride_elements": qk_nope * kv_lora,
                "c_batch_stride_elements": decode_tokens * kv_lora,
                "weight_dtype": "bf16",
                "weight_group_size": 0,
            },
        )
        if attention_layout == "mla":
            attention_suffix = "decode-mla-attention"
            attention_kind = "flash_attn"
            attention_variant = "mla_decode"
        else:
            attention_suffix = "decode-dsa-attention"
            attention_kind = "eager_attn"
            attention_variant = "dsa_sparse_decode"
        attention_predecessors = ["compressed-kv-update"]
        if topk_source is not None:
            attention_predecessors.append(topk_source)
        attention_predecessors.append("decode-q-down-proj")
        attention_shape = {
            "batch": len(decode_indices),
            "q_len": max(query_lengths[index] for index in decode_indices),
            "kv_len": max(decode_kv_lengths),
            "qk_head_dim": kv_lora + qk_rope,
            "v_head_dim": kv_lora,
            "num_query_heads": heads,
            "kv_lora_rank": kv_lora,
        }
        if attention_layout == "dsa_mla":
            del attention_shape["kv_lora_rank"]
            attention_shape["topk"] = min(
                config["index_topk"], max(decode_kv_lengths))
        attention_attributes = {
            "causal": True,
            "kv_layout": "mla",
            "kv_dist": "uniform",
            "q_lens": tuple(query_lengths[index] for index in decode_indices),
            "kv_lens": decode_kv_lengths,
        }
        if attention_layout == "dsa_mla":
            attention_attributes.update({
                "indexer_type": indexer_type,
                "selected_kv_lens": tuple(
                    min(config["index_topk"], length)
                    for length in decode_kv_lengths),
                "topk_capacity": config["index_topk"],
                "topk_index_dtype": "int32",
            })
        add(
            attention_suffix, attention_kind, attention_variant, "bf16",
            attention_shape, tuple(attention_predecessors),
            attention_attributes,
        )
        add(
            "decode-v-up-proj", "linear", "batched_gemm", "bf16",
            {"b": heads, "m": decode_tokens, "n": value_head, "k": kv_lora},
            (attention_suffix,),
            {
                "batch_layout": "strided",
                "a_batch_stride_elements": decode_tokens * kv_lora,
                "b_batch_stride_elements": kv_lora * value_head,
                "c_batch_stride_elements": decode_tokens * value_head,
                "weight_dtype": "bf16",
                "weight_group_size": 0,
            },
        )
        attention_outputs.append("decode-v-up-proj")

    add(
        "o-proj", "linear", "gemm", "bf16",
        {"b": 1, "m": token_count,
         "n": hidden, "k": heads * value_head},
        tuple(attention_outputs),
    )
    add(
        "post-attention-norm", "norm", "rmsnorm", "bf16",
        {"rows": token_count, "hidden": hidden},
        ("o-proj",),
    )
    if layer_type == "dense":
        intermediate = config["intermediate_size"]
        add(
            "gate-up", "linear", "gemm", "bf16",
            {"b": 1, "m": token_count,
             "n": 2 * intermediate, "k": hidden},
            ("post-attention-norm",),
        )
        add(
            "silu-mul", "activation", "silu", "bf16",
            {"elements": token_count * intermediate,
             "tok": token_count, "intermediate": intermediate},
            ("gate-up",),
        )
        add(
            "down-proj", "linear", "gemm", "bf16",
            {"b": 1, "m": token_count, "n": hidden, "k": intermediate},
            ("silu-mul",),
        )
    else:
        moe_intermediate = config["moe_intermediate_size"]
        experts = config["n_routed_experts"]
        topk = config["num_experts_per_tok"]
        if routed_weight_dtype == "int4":
            weight_group_size = 32
        elif routed_weight_dtype == "bf16":
            weight_group_size = 0
        else:
            raise AssertionError(
                "unhandled routed expert weight dtype: "
                f"{routed_weight_dtype!r}")
        add(
            "router", "router", "gemm", router_dtype,
            {"b": 1, "m": token_count, "n": experts, "k": hidden},
            ("post-attention-norm",),
            {
                "score_function": config["scoring_func"],
                "topk_method": config["topk_method"],
                "expert_group_count": config["n_group"],
                "topk_group_count": config["topk_group"],
                "normalize_topk_probability": config["norm_topk_prob"],
                "weight_dtype": "bf16",
                "weight_group_size": 0,
            },
        )
        add(
            "routed-experts", "fused_moe", "full", "bf16",
            {
                "tokens": token_count,
                "experts": experts,
                "topk": topk,
                "hidden": hidden,
                "intermediate": moe_intermediate,
            },
            ("router", "post-attention-norm"),
            {
                "routed_scaling_factor": config["routed_scaling_factor"],
                "routed_weight_dtype": routed_weight_dtype,
                "weight_group_size": weight_group_size,
            },
        )
        add(
            "shared-experts:gate-up", "linear", "gemm", "bf16",
            {"b": 1, "m": token_count,
             "n": 2 * moe_intermediate, "k": hidden},
            ("post-attention-norm",),
        )
        add(
            "shared-experts:silu-mul", "activation", "silu", "bf16",
            {"elements": token_count * moe_intermediate,
             "tok": token_count, "intermediate": moe_intermediate},
            ("shared-experts:gate-up",),
        )
        add(
            "shared-experts:down-proj", "linear", "gemm", "bf16",
            {"b": 1, "m": token_count,
             "n": hidden, "k": moe_intermediate},
            ("shared-experts:silu-mul",),
        )
        add(
            "shared-add", "elementwise", "add", "bf16",
            {"elements": token_count * hidden},
            ("shared-experts:down-proj", "routed-experts"),
        )

    return {
        "schema_version": 1,
        "model_id": model_id,
        "phase": phase,
        "layer_index": layer_index,
        "layer_type": layer_type,
        "indexer_type": indexer_type,
        "nodes": tuple(nodes),
    }


def pinned_qwen3_pp_source_evidence() -> dict[str, object]:
    """Verify the additional pinned sources that define PP data movement."""

    base = pinned_qwen3_source_evidence()
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the pinned vLLM package")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"vLLM PP source reference requires one package root: "
            f"{locations!r}")
    package_root = Path(locations[0]).resolve()
    files = []
    for relative_path, expected_hash, required_symbols in (
            _PP_ADDITIONAL_SOURCE_CONTRACT):
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"pinned vLLM PP source is missing: {source_path}")
        source_bytes = source_path.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned vLLM PP source changed: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}")
        tree = ast.parse(source_bytes, filename=str(source_path))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        top_level_functions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for class_name, method_name in required_symbols:
            if not class_name:
                if method_name not in top_level_functions:
                    raise RuntimeError(
                        f"pinned source {relative_path} lacks function "
                        f"{method_name}")
                continue
            if class_name not in classes:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks class {class_name}")
            methods = {
                node.name for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks symbol "
                    f"{class_name}.{method_name}")
        if relative_path in _PP_REQUIRED_SOURCE_FRAGMENTS:
            source = source_bytes.decode("utf-8")
            for fragment, expected_count in _PP_REQUIRED_SOURCE_FRAGMENTS[
                    relative_path]:
                actual_count = source.count(fragment)
                if actual_count != expected_count:
                    raise RuntimeError(
                        "pinned PP/TP source fragment changed in "
                        f"{relative_path}: {fragment!r}; expected "
                        f"{expected_count}, got {actual_count}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
            "symbols": tuple(
                (f"{class_name}.{method_name}"
                 if class_name else method_name)
                for class_name, method_name in required_symbols
            ),
        })
    return {
        **base,
        "files": base["files"] + tuple(files),
        "pipeline_data_path": (
            "Worker.execute_model->GroupCoordinator.send_tensor_dict/"
            "recv_tensor_dict"),
        "pipeline_control_metadata_latency": "unmodeled",
        "pipeline_intermediate_tensor_fields": (
            "hidden_states", "residual"),
        "pipeline_tp_wire_partition": (
            "reshape(tp_world_size,-1)[tp_rank]"),
        "pipeline_tp_receiver_reassembly": (
            "tp_group.all_gather(dim=0)->original_shape"),
        "pipeline_residual_all_gather_when_sp_disabled": True,
        "parallel_rank_layout": "DP_x_PP_x_TP",
    }


def pinned_qwen3_moe_ep_source_evidence() -> dict[str, object]:
    """Verify pinned sources that define the TP=2, DP=1 EP workload."""

    base = pinned_qwen3_source_evidence()
    package_spec = importlib.util.find_spec("vllm")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise RuntimeError("cannot locate the pinned vLLM package")
    locations = tuple(package_spec.submodule_search_locations)
    if len(locations) != 1:
        raise RuntimeError(
            f"vLLM EP source reference requires one package root: "
            f"{locations!r}")
    package_root = Path(locations[0]).resolve()
    files = []
    for relative_path, expected_hash, required_symbols in (
            _EP_ADDITIONAL_SOURCE_CONTRACT):
        source_path = package_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"pinned vLLM EP source is missing: {source_path}")
        source_bytes = source_path.read_bytes()
        actual_hash = hashlib.sha256(source_bytes).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"pinned vLLM EP source changed: {relative_path}; "
                f"expected {expected_hash}, got {actual_hash}")
        tree = ast.parse(source_bytes, filename=str(source_path))
        classes = {
            node.name: node for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        top_level_functions = {
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for class_name, method_name in required_symbols:
            if not class_name:
                if method_name not in top_level_functions:
                    raise RuntimeError(
                        f"pinned source {relative_path} lacks function "
                        f"{method_name}")
                continue
            if class_name not in classes:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks class {class_name}")
            methods = {
                node.name for node in classes[class_name].body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if method_name not in methods:
                raise RuntimeError(
                    f"pinned source {relative_path} lacks symbol "
                    f"{class_name}.{method_name}")
        source = source_bytes.decode("utf-8")
        for fragment, expected_count in _EP_REQUIRED_SOURCE_FRAGMENTS[
                relative_path]:
            actual_count = source.count(fragment)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"pinned EP source fragment changed in {relative_path}: "
                    f"{fragment!r}; expected {expected_count}, got "
                    f"{actual_count}")
        files.append({
            "relative_path": relative_path,
            "sha256": actual_hash,
            "symbols": tuple(
                (f"{class_name}.{method_name}"
                 if class_name else method_name)
                for class_name, method_name in required_symbols
            ),
        })
    return {
        "vllm_version": base["vllm_version"],
        "source_commit": base["source_commit"],
        "files": tuple(files),
        "full_input_rows_per_rank": "hidden_states.size(0)",
        "expert_placement_strategy": "linear",
        "dispatch_requires_dp_gt_1": True,
        "output_collective": "tensor_model_parallel_all_reduce",
        "alignment_uses_global_experts_before_expert_map": True,
        "attention_head_dim_source": (
            "config.head_dim_or_hidden_size_div_total_heads"),
        "attention_o_projection_input": "total_num_heads_times_head_dim",
        "residual_rmsnorm_boundaries": True,
    }


def qwen3_moe_tp2_attention_reference(
    model_path: Path,
    *,
    step_id: str,
    layer_index: int,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, int],
    resource_id: str,
) -> dict[str, object]:
    """Derive the Qwen3MoE TP=2 attention subgraph from pinned source."""

    if not isinstance(model_path, Path):
        raise TypeError("model_path must be a Path")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("step_id must be non-empty")
    if (isinstance(layer_index, bool) or not isinstance(layer_index, int)
            or layer_index < 0):
        raise ValueError("layer_index must be a non-negative integer")
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in query_lengths)):
        raise ValueError("query_lengths must contain positive integers")
    if (not isinstance(kv_lengths, tuple)
            or len(kv_lengths) != len(query_lengths)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in kv_lengths)):
        raise ValueError(
            "kv_lengths must align with query_lengths and be positive")
    if any(kv_length < query_length for query_length, kv_length in zip(
            query_lengths, kv_lengths)):
        raise ValueError(
            "Qwen3MoE KV lengths cannot be shorter than query lengths")
    if (not isinstance(physical_ranks, tuple)
            or len(physical_ranks) != 2
            or any(isinstance(rank, bool) or not isinstance(rank, int)
                   or rank < 0 for rank in physical_ranks)
            or len(set(physical_ranks)) != 2):
        raise ValueError(
            "Qwen3MoE TP=2 reference requires two unique physical ranks")
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError(
            "Qwen3MoE TP=2 reference resource_id must be non-empty")

    config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    if config["model_type"] != "qwen3_moe":
        raise ValueError(
            "Qwen3MoE source reference requires model_type=qwen3_moe")
    if config["torch_dtype"] != "bfloat16":
        raise ValueError(
            "Qwen3MoE source reference currently requires BF16")
    hidden = config["hidden_size"]
    query_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    layers = config["num_hidden_layers"]
    for name, value in (
        ("hidden_size", hidden),
        ("num_attention_heads", query_heads),
        ("num_key_value_heads", kv_heads),
        ("head_dim", head_dim),
        ("num_hidden_layers", layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Qwen3MoE config {name} must be positive")
    if layer_index >= layers:
        raise ValueError("layer_index exceeds the Qwen3MoE layer count")
    if query_heads % 2:
        raise ValueError(
            "Qwen3MoE TP=2 reference requires even query heads")
    if kv_heads >= 2:
        if kv_heads % 2:
            raise ValueError(
                "Qwen3MoE TP=2 reference requires even KV heads")
        local_kv_heads = kv_heads // 2
    else:
        local_kv_heads = 1
    local_query_heads = query_heads // 2
    token_count = sum(query_lengths)
    dtype_bytes = 2
    prefix = f"layer-{layer_index}"
    nodes = []
    edges = []
    previous_rank_node_ids: tuple[str, ...] = ()

    def add_ranked(
        suffix: str,
        workload_kind: str,
        workload_shape: Mapping[str, int],
        workload_variant: str,
        kernel_kind: str,
        kernel_variant: str,
        kernel_shape: Mapping[str, int],
        source_symbol: str,
        *,
        workload_attributes: Mapping[str, object] | None = None,
        evidence_status: str = "validated_static_source_semantics",
        kernel_call_count: int = 1,
        principal_call_suffix: str = "",
    ) -> tuple[str, str]:
        nonlocal previous_rank_node_ids
        attributes = (
            {} if workload_attributes is None else dict(workload_attributes)
        )
        node_ids = []
        for logical_rank, physical_rank in enumerate(physical_ranks):
            node_id = f"{step_id}:{prefix}:{suffix}:tp-rank-{logical_rank}"
            nodes.append({
                "node_id": node_id,
                "workload_kind": workload_kind,
                "workload_shape": dict(workload_shape),
                "workload_variant": workload_variant,
                "workload_dtype": "bf16",
                "workload_pool_name": "main",
                "workload_rank": physical_rank,
                "workload_attributes": dict(attributes),
                "kernel_kind": kernel_kind,
                "kernel_variant": kernel_variant,
                "kernel_shape": dict(kernel_shape),
                "kernel_call_count": kernel_call_count,
                "kernel_call_id": f"{node_id}{principal_call_suffix}",
                "kernel_pool_name": "main",
                "kernel_rank": physical_rank,
                "kernel_resource": "compute",
                "source_symbol": source_symbol,
                "evidence_status": evidence_status,
            })
            if previous_rank_node_ids:
                predecessor = (
                    previous_rank_node_ids[logical_rank]
                    if len(previous_rank_node_ids) == 2
                    else previous_rank_node_ids[0]
                )
                edges.append((predecessor, node_id, "data"))
            node_ids.append(node_id)
        previous_rank_node_ids = tuple(node_ids)
        return tuple(node_ids)

    norm_variant = "rmsnorm" if layer_index == 0 else "add_rmsnorm"
    add_ranked(
        "input-norm", "norm",
        {"rows": token_count, "hidden": hidden},
        norm_variant, "VECTOR", norm_variant,
        {"rows": token_count, "hidden": hidden, "tok": token_count,
         "dtype_bytes": dtype_bytes},
        "Qwen3MoeDecoderLayer.forward:input_layernorm:RMSNorm",
    )
    local_qkv = (
        local_query_heads + 2 * local_kv_heads) * head_dim
    add_ranked(
        "qkv", "linear",
        {"b": 1, "m": token_count, "n": local_qkv, "k": hidden},
        "gemm", "GEMM", "gemm",
        {"b": 1, "m": token_count, "n": local_qkv, "k": hidden,
         "dtype_bytes": dtype_bytes},
        "Qwen3MoeAttention.qkv_proj:QKVParallelLinear",
        workload_attributes=_gemm_workload_attributes(
            "qkv_proj", 2, expert_parallel_size=2),
    )
    add_ranked(
        "q-norm", "norm",
        {"rows": token_count * local_query_heads, "hidden": head_dim},
        "rmsnorm", "VECTOR", "rmsnorm",
        {"rows": token_count * local_query_heads, "hidden": head_dim,
         "tok": token_count * local_query_heads,
         "dtype_bytes": dtype_bytes},
        "Qwen3MoeAttention.q_norm:RMSNorm",
        kernel_call_count=2,
        principal_call_suffix=":1",
    )
    add_ranked(
        "k-norm", "norm",
        {"rows": token_count * local_kv_heads, "hidden": head_dim},
        "rmsnorm", "VECTOR", "rmsnorm",
        {"rows": token_count * local_kv_heads, "hidden": head_dim,
         "tok": token_count * local_kv_heads,
         "dtype_bytes": dtype_bytes},
        "Qwen3MoeAttention.k_norm:RMSNorm",
        kernel_call_count=2,
        principal_call_suffix=":1",
    )
    rotary_elements = (
        token_count * (local_query_heads + local_kv_heads) * head_dim)
    add_ranked(
        "rotary", "rotary",
        {"elements": rotary_elements, "seq_len": token_count,
         "num_query_heads": local_query_heads,
         "num_kv_heads": local_kv_heads, "head_dim": head_dim},
        "rotary", "VECTOR", "rotary",
        {"elements": rotary_elements, "element_count": rotary_elements,
         "seq_len": token_count, "num_query_heads": local_query_heads,
         "num_kv_heads": local_kv_heads, "head_dim": head_dim,
         "dtype_bytes": dtype_bytes},
        "Qwen3MoeAttention.rotary_emb:get_rope",
    )
    kv_elements = token_count * local_kv_heads * head_dim
    add_ranked(
        "kv-update", "kv_update",
        {"elements": kv_elements, "seq_len": token_count,
         "num_kv_heads": local_kv_heads, "head_dim": head_dim},
        "paged_kv_cache", "KV_CACHE", "paged_kv_cache",
        {"elements": kv_elements, "element_count": kv_elements,
         "seq_len": token_count, "num_kv_heads": local_kv_heads,
         "head_dim": head_dim, "dtype_bytes": dtype_bytes},
        "Attention.forward:stores_key_and_value_in_kv_cache",
        evidence_status="projected_logical_decomposition",
    )
    add_ranked(
        "attention", "flash_attn",
        {"batch": len(query_lengths), "q_len": max(query_lengths),
         "kv_len": max(kv_lengths), "head_dim": head_dim,
         "num_query_heads": local_query_heads,
         "num_kv_heads": local_kv_heads},
        "prefill", "FLASH_ATTN", "prefill",
        {"batch": len(query_lengths), "q_len": max(query_lengths),
         "kv_len": max(kv_lengths), "head_dim": head_dim,
         "num_query_heads": local_query_heads,
         "num_kv_heads": local_kv_heads, "dtype_bytes": dtype_bytes},
        "Qwen3MoeAttention.attn:Attention",
        workload_attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": query_lengths,
            "kv_lens": kv_lengths,
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "analytical_launch",
        },
        evidence_status="projected_backend_selection",
        # The backend-accepted output buffer is zero-filled
        # (Attention.forward use_output) ahead of the FA program.
        kernel_call_count=2,
        principal_call_suffix=":1",
    )
    local_attention_width = local_query_heads * head_dim
    o_projection_node_ids = add_ranked(
        "o-proj", "linear",
        {"b": 1, "m": token_count, "n": hidden,
         "k": local_attention_width},
        "gemm", "GEMM", "gemm",
        {"b": 1, "m": token_count, "n": hidden,
         "k": local_attention_width, "dtype_bytes": dtype_bytes},
        "Qwen3MoeAttention.o_proj:RowParallelLinear",
        workload_attributes=_gemm_workload_attributes(
            "o_proj", 2, expert_parallel_size=2),
    )
    collective_bytes = token_count * hidden * dtype_bytes
    collective_attributes = {
        "algorithm": "ring",
        "bytes": collective_bytes,
        "bytes_semantics": "per_rank_input",
        "backend": "analytical_roofline_v1",
        "backend_config_sha256": None,
        "direction": "collective",
        "link": "nvlink",
        "topology": "ring",
        "participants": physical_ranks,
        "operation": "all_reduce",
        "source_rank": None,
        "destination_rank": None,
        "ring_order": physical_ranks,
        "reduction": "sum",
        "resource_id": resource_id,
    }
    collective_id = f"{step_id}:{prefix}:o-proj:tp-all-reduce"
    nodes.append({
        "node_id": collective_id,
        "workload_kind": "collective",
        "workload_shape": {},
        "workload_variant": None,
        "workload_dtype": None,
        "workload_pool_name": "main",
        "workload_rank": physical_ranks[0],
        "workload_attributes": dict(collective_attributes),
        "kernel_kind": "COMMUNICATION",
        "kernel_variant": None,
        "kernel_shape": {},
        "kernel_transfer": dict(collective_attributes),
        "kernel_call_count": 1,
        "kernel_call_id": collective_id,
        "kernel_pool_name": "main",
        "kernel_rank": physical_ranks[0],
        "kernel_resource": resource_id,
        "source_symbol": "RowParallelLinear.forward:all_reduce",
        "evidence_status": (
            "validated_static_source_semantics_projected_latency"),
    })
    for node_id in o_projection_node_ids:
        edges.append((node_id, collective_id, "barrier"))
    previous_rank_node_ids = (collective_id,)
    add_ranked(
        "post-attn-norm", "norm",
        {"rows": token_count, "hidden": hidden},
        "add_rmsnorm", "VECTOR", "add_rmsnorm",
        {"rows": token_count, "hidden": hidden, "tok": token_count,
         "dtype_bytes": dtype_bytes},
        "Qwen3MoeDecoderLayer.forward:post_attention_layernorm:RMSNorm",
    )
    return {
        "nodes": tuple(nodes),
        "edges": tuple(edges),
        "topological_order": tuple(item["node_id"] for item in nodes),
        "validation_scope": "qwen3_moe_bf16_tp2_attention_vllm_v1",
        "physical_ranks": physical_ranks,
        "resource_id": resource_id,
        "layer_index": layer_index,
        "token_count": token_count,
    }


def qwen3_single_rank_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    """Derive Qwen3 TP=PP=1 forward shapes from fixed source semantics."""

    if not isinstance(model_path, Path):
        raise TypeError("model_path must be a Path")
    if forward_scope not in ("prefill", "decode", "mixed"):
        raise ValueError(
            "Qwen3 source reference forward_scope must be prefill, decode, "
            "or mixed")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("step_id must be non-empty")
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in query_lengths)):
        raise ValueError("query_lengths must contain positive integers")
    if (not isinstance(kv_lengths, tuple)
            or len(kv_lengths) != len(query_lengths)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in kv_lengths)):
        raise ValueError(
            "kv_lengths must align with query_lengths and be positive")
    if any(kv_length < query_length for query_length, kv_length in zip(
            query_lengths, kv_lengths)):
        raise ValueError("Qwen3 KV lengths cannot be shorter than query lengths")
    if (forward_scope == "decode"
            and any(length != 1 for length in query_lengths)):
        raise ValueError(
            "decode Qwen3 source reference requires one query row per request")
    if forward_scope == "mixed" and not (
            any(length == 1 for length in query_lengths)
            and any(length > 1 for length in query_lengths)):
        raise ValueError(
            "mixed Qwen3 source reference requires decode and prefill rows")
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["model_type"] != "qwen3":
        raise ValueError("Qwen3 source reference requires model_type=qwen3")
    if config["torch_dtype"] != "bfloat16":
        raise ValueError("Qwen3 source reference currently requires BF16")
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    query_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    for name, value in (
        ("hidden_size", hidden),
        ("intermediate_size", intermediate),
        ("num_attention_heads", query_heads),
        ("num_key_value_heads", kv_heads),
        ("head_dim", head_dim),
        ("num_hidden_layers", layers),
        ("vocab_size", vocab),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Qwen3 config {name} must be positive")
    if query_heads % kv_heads:
        raise ValueError("Qwen3 KV heads must divide query heads exactly")
    token_count = sum(query_lengths)
    dtype_bytes = 2
    nodes = []

    def add(
        suffix: str,
        workload_kind: str,
        workload_shape: Mapping[str, int],
        workload_variant: str,
        kernel_kind: str,
        kernel_variant: str,
        kernel_shape: Mapping[str, int],
        source_symbol: str,
        evidence_status: str = "validated_static_source_semantics",
        workload_attributes: Mapping[str, object] | None = None,
    ) -> None:
        attributes = (
            {} if workload_attributes is None else dict(workload_attributes)
        )
        nodes.append({
            "node_id": f"{step_id}:{suffix}",
            "workload_kind": workload_kind,
            "workload_shape": dict(workload_shape),
            "workload_variant": workload_variant,
            "workload_dtype": "bf16",
            "workload_pool_name": "main",
            "workload_rank": 0,
            "workload_attributes": attributes,
            "kernel_kind": kernel_kind,
            "kernel_variant": kernel_variant,
            "kernel_shape": dict(kernel_shape),
            "source_symbol": source_symbol,
            "evidence_status": evidence_status,
        })

    add(
        "token-embedding", "gather",
        {"elements": token_count * hidden, "rows": token_count,
         "hidden": hidden},
        "embedding", "VECTOR", "gather",
        {"rows": token_count, "hidden": hidden,
         "element_count": token_count * hidden,
         "dtype_bytes": dtype_bytes},
        "Qwen2Model.embed_tokens:VocabParallelEmbedding.forward_native",
    )
    for layer in range(layers):
        prefix = f"layer-{layer}"
        add(
            f"{prefix}:input-norm", "norm",
            {"rows": token_count, "hidden": hidden},
            "rmsnorm" if layer == 0 else "add_rmsnorm",
            "VECTOR", "rmsnorm" if layer == 0 else "add_rmsnorm",
            {"rows": token_count, "hidden": hidden,
             "dtype_bytes": dtype_bytes},
            "Qwen3DecoderLayer.forward:RMSNorm",
        )
        qkv_n = (query_heads + 2 * kv_heads) * head_dim
        add(
            f"{prefix}:qkv", "linear",
            {"b": 1, "m": token_count, "n": qkv_n, "k": hidden},
            "gemm", "GEMM", "gemm",
            {"b": 1, "m": token_count, "n": qkv_n, "k": hidden,
             "dtype_bytes": dtype_bytes},
            "Qwen3Attention.qkv_proj:QKVParallelLinear",
        )
        heads_by_name = {"q": query_heads, "k": kv_heads}
        for name in ("q", "k"):
            heads = heads_by_name[name]
            add(
                f"{prefix}:{name}-norm", "norm",
                {"rows": token_count * heads, "hidden": head_dim},
                "rmsnorm", "VECTOR", "rmsnorm",
                {"rows": token_count * heads, "hidden": head_dim,
                 "dtype_bytes": dtype_bytes},
                f"Qwen3Attention.{name}_norm:RMSNorm",
            )
        rotary_elements = token_count * (query_heads + kv_heads) * head_dim
        add(
            f"{prefix}:rotary", "rotary",
            {"elements": rotary_elements, "seq_len": token_count,
             "num_query_heads": query_heads, "num_kv_heads": kv_heads,
             "head_dim": head_dim},
            "rotary", "VECTOR", "rotary",
            {"element_count": rotary_elements, "seq_len": token_count,
             "num_query_heads": query_heads, "num_kv_heads": kv_heads,
             "head_dim": head_dim, "dtype_bytes": dtype_bytes},
            "Qwen3Attention.rotary_emb:get_rope",
        )
        kv_elements = token_count * kv_heads * head_dim
        add(
            f"{prefix}:kv-update", "kv_update",
            {"elements": kv_elements, "seq_len": token_count,
             "num_kv_heads": kv_heads, "head_dim": head_dim},
            "paged_kv_cache", "KV_CACHE", "paged_kv_cache",
            {"element_count": kv_elements, "seq_len": token_count,
             "num_kv_heads": kv_heads, "head_dim": head_dim,
             "dtype_bytes": dtype_bytes},
            "Attention.forward:stores_key_and_value_in_kv_cache",
            "projected_logical_decomposition",
            workload_attributes=_PAGED_KV_CACHE_ATTRIBUTES,
        )
        if forward_scope == "decode":
            attention_variant = "decode"
        elif forward_scope in ("prefill", "mixed"):
            attention_variant = "prefill"
        else:
            raise RuntimeError(
                f"unhandled Qwen3 forward scope: {forward_scope!r}")
        add(
            f"{prefix}:attention", "flash_attn",
            {"batch": len(query_lengths), "q_len": max(query_lengths),
             "kv_len": max(kv_lengths), "head_dim": head_dim,
             "num_query_heads": query_heads, "num_kv_heads": kv_heads},
            attention_variant, "FLASH_ATTN", attention_variant,
            {"batch": len(query_lengths), "q_len": max(query_lengths),
             "kv_len": max(kv_lengths), "head_dim": head_dim,
             "num_query_heads": query_heads, "num_kv_heads": kv_heads,
             "dtype_bytes": dtype_bytes},
            "Qwen3Attention.attn:Attention",
            "projected_backend_selection",
            {
                "causal": True,
                "kv_layout": "paged_varlen",
                "kv_dist": "uniform",
                "q_lens": query_lengths,
                "kv_lens": kv_lengths,
                "fa_ksteps": None,
                "fa_eff_mfrac": 1.0,
                "launch_policy": "analytical_launch",
                "split_policy": "serving",
                "graph_runtime_mode": "NONE",
            },
        )
        add(
            f"{prefix}:o-proj", "linear",
            {"b": 1, "m": token_count, "n": hidden,
             "k": query_heads * head_dim},
            "gemm", "GEMM", "gemm",
            {"b": 1, "m": token_count, "n": hidden,
             "k": query_heads * head_dim,
             "dtype_bytes": dtype_bytes},
            "Qwen3Attention.o_proj:RowParallelLinear",
        )
        add(
            f"{prefix}:post-attn-norm", "norm",
            {"rows": token_count, "hidden": hidden},
            "add_rmsnorm", "VECTOR", "add_rmsnorm",
            {"rows": token_count, "hidden": hidden,
             "dtype_bytes": dtype_bytes},
            "Qwen3DecoderLayer.forward:RMSNorm(residual)",
        )
        add(
            f"{prefix}:gate-up", "linear",
            {"b": 1, "m": token_count, "n": 2 * intermediate,
             "k": hidden},
            "gemm", "GEMM", "gemm",
            {"b": 1, "m": token_count, "n": 2 * intermediate,
             "k": hidden, "dtype_bytes": dtype_bytes},
            "Qwen2MLP.gate_up_proj:MergedColumnParallelLinear",
        )
        activation_elements = token_count * intermediate
        add(
            f"{prefix}:activation", "activation",
            {"elements": activation_elements, "tok": token_count,
             "intermediate": intermediate},
            "silu", "VECTOR", "silu",
            {"element_count": activation_elements, "tok": token_count,
             "intermediate": intermediate, "dtype_bytes": dtype_bytes},
            "Qwen2MLP.act_fn:SiluAndMul",
        )
        add(
            f"{prefix}:down", "linear",
            {"b": 1, "m": token_count, "n": hidden,
             "k": intermediate},
            "gemm", "GEMM", "gemm",
            {"b": 1, "m": token_count, "n": hidden,
             "k": intermediate, "dtype_bytes": dtype_bytes},
            "Qwen2MLP.down_proj:RowParallelLinear",
        )
    add(
        "final-norm", "norm", {"rows": token_count, "hidden": hidden},
        "add_rmsnorm", "VECTOR", "add_rmsnorm",
        {"rows": token_count, "hidden": hidden,
         "dtype_bytes": dtype_bytes},
        "Qwen2Model.norm:RMSNorm",
    )
    logits_rows = len(query_lengths)
    add(
        "logits-boundary-gather", "gather",
        {"elements": logits_rows * hidden, "rows": logits_rows,
         "hidden": hidden},
        "logits_indices", "VECTOR", "gather",
        {"rows": logits_rows, "hidden": hidden,
         "element_count": logits_rows * hidden,
         "dtype_bytes": dtype_bytes},
        "vllm.v1.worker.gpu_model_runner.sample_hidden_states",
        workload_attributes={
            "gather_factor": 1,
            "index_count": logits_rows,
            "index_bytes": 8,
            "predicate": "per-step target logit row selection",
            "scattered": True,
            "source_symbol": (
                "vllm.v1.worker.gpu_model_runner.sample_hidden_states"),
        },
    )
    add(
        "lm-head", "lm_head",
        {"b": 1, "m": len(query_lengths), "n": vocab, "k": hidden},
        "gemv" if len(query_lengths) == 1 else "gemm",
        "GEMM", "gemv" if len(query_lengths) == 1 else "gemm",
        {"b": 1, "m": len(query_lengths), "n": vocab, "k": hidden,
         "dtype_bytes": dtype_bytes},
        "GPUModelRunner._prepare_inputs.logits_indices:last-row-per-request"
        "->Qwen3ForCausalLM.compute_logits:ParallelLMHead",
    )
    if forward_scope == "prefill":
        # The A800 prefill token-embedding gather is a versioned measured
        # zero (the launch lands outside the step capture window), so the
        # reference omits it exactly like the observation-suppressed
        # emission; decode and mixed keep it.
        nodes = [
            item for item in nodes
            if item["node_id"] != f"{step_id}:token-embedding"
        ]
    return tuple(nodes)


def qwen3_tp2_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, int],
    resource_id: str,
) -> dict[str, object]:
    """Derive a TP=2 rank-local DAG from pinned vLLM source semantics."""

    if (not isinstance(physical_ranks, tuple)
            or len(physical_ranks) != 2
            or any(isinstance(rank, bool) or not isinstance(rank, int)
                   or rank < 0 for rank in physical_ranks)
            or len(set(physical_ranks)) != 2):
        raise ValueError("TP=2 reference requires two unique physical ranks")
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("TP=2 reference resource_id must be non-empty")
    base = qwen3_single_rank_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
    )
    config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    query_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    vocab = config["vocab_size"]
    if query_heads % 2:
        raise ValueError("TP=2 Qwen3 reference requires even query heads")
    if kv_heads >= 2:
        if kv_heads % 2:
            raise ValueError("TP=2 Qwen3 reference requires even KV heads")
        local_kv_heads = kv_heads // 2
    else:
        local_kv_heads = 1
    local_query_heads = query_heads // 2
    if intermediate % 2:
        raise ValueError("TP=2 Qwen3 reference requires even intermediate size")
    local_intermediate = intermediate // 2
    padded_vocab = ((vocab + 63) // 64) * 64
    if padded_vocab % 2:
        raise ValueError("TP=2 padded vocabulary must be even")
    local_vocab = padded_vocab // 2
    token_count = sum(query_lengths)
    dtype_bytes = 2
    nodes = []
    edges = []
    previous_rank_node_ids: tuple[str, ...] = ()

    for base_item in base:
        item = dict(base_item)
        suffix = item["node_id"][len(step_id) + 1:]
        workload_shape = dict(item["workload_shape"])
        kernel_shape = dict(item["kernel_shape"])
        if suffix.endswith(":qkv"):
            local_qkv = (
                local_query_heads + 2 * local_kv_heads) * head_dim
            workload_shape["n"] = local_qkv
            kernel_shape["n"] = local_qkv
        elif suffix.endswith(":q-norm"):
            rows = token_count * local_query_heads
            workload_shape["rows"] = rows
            kernel_shape["rows"] = rows
        elif suffix.endswith(":k-norm"):
            rows = token_count * local_kv_heads
            workload_shape["rows"] = rows
            kernel_shape["rows"] = rows
        elif suffix.endswith(":rotary"):
            elements = (
                token_count * (local_query_heads + local_kv_heads) * head_dim)
            for shape in (workload_shape, kernel_shape):
                shape["elements" if shape is workload_shape else "element_count"] = (
                    elements)
                shape["num_query_heads"] = local_query_heads
                shape["num_kv_heads"] = local_kv_heads
        elif suffix.endswith(":kv-update"):
            elements = token_count * local_kv_heads * head_dim
            workload_shape["elements"] = elements
            workload_shape["num_kv_heads"] = local_kv_heads
            kernel_shape["element_count"] = elements
            kernel_shape["num_kv_heads"] = local_kv_heads
        elif suffix.endswith(":attention"):
            workload_shape["num_query_heads"] = local_query_heads
            workload_shape["num_kv_heads"] = local_kv_heads
            kernel_shape["num_query_heads"] = local_query_heads
            kernel_shape["num_kv_heads"] = local_kv_heads
        elif suffix.endswith(":o-proj"):
            local_attention_width = local_query_heads * head_dim
            workload_shape["k"] = local_attention_width
            kernel_shape["k"] = local_attention_width
        elif suffix.endswith(":gate-up"):
            workload_shape["n"] = 2 * local_intermediate
            kernel_shape["n"] = 2 * local_intermediate
        elif suffix.endswith(":activation"):
            elements = token_count * local_intermediate
            workload_shape["elements"] = elements
            workload_shape["intermediate"] = local_intermediate
            kernel_shape["element_count"] = elements
            kernel_shape["intermediate"] = local_intermediate
        elif suffix.endswith(":down"):
            workload_shape["k"] = local_intermediate
            kernel_shape["k"] = local_intermediate
        elif suffix == "lm-head":
            workload_shape["n"] = local_vocab
            kernel_shape["n"] = local_vocab

        local_node_ids = []
        for logical_rank, physical_rank in enumerate(physical_ranks):
            local_item = dict(item)
            local_node_id = (
                f"{step_id}:{suffix}:tp-rank-{logical_rank}")
            local_item["node_id"] = local_node_id
            local_item["workload_shape"] = dict(workload_shape)
            local_item["kernel_shape"] = dict(kernel_shape)
            local_item["workload_rank"] = physical_rank
            nodes.append(local_item)
            if previous_rank_node_ids:
                edges.append((
                    previous_rank_node_ids[logical_rank],
                    local_node_id,
                    "data",
                ))
            local_node_ids.append(local_node_id)

        if (suffix == "token-embedding"
                or suffix.endswith(":o-proj")
                or suffix.endswith(":down")):
            operation = "all_reduce"
            collective_bytes = token_count * hidden * dtype_bytes
            reduction = "sum"
            if suffix == "token-embedding":
                source_symbol = (
                    "VocabParallelEmbedding.forward_native:all_reduce")
            elif suffix.endswith(":o-proj") or suffix.endswith(":down"):
                source_symbol = "RowParallelLinear.forward:all_reduce"
            else:
                raise AssertionError(
                    f"unclassified all-reduce source suffix: {suffix!r}")
        elif suffix == "lm-head":
            operation = "all_gather"
            collective_bytes = (
                workload_shape["m"] * local_vocab * dtype_bytes)
            reduction = None
            source_symbol = (
                "LogitsProcessor._gather_logits:VLLM_USE_V1_all_gather")
        else:
            operation = None
        if operation is None:
            previous_rank_node_ids = tuple(local_node_ids)
            continue
        collective_id = (
            f"{step_id}:{suffix}:tp-{operation.replace('_', '-')}")
        attributes = {
            "algorithm": "ring",
            "bytes": collective_bytes,
            "bytes_semantics": "per_rank_input",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "direction": "collective",
            "link": "nvlink",
            "topology": "ring",
            "participants": physical_ranks,
            "operation": operation,
            "source_rank": None,
            "destination_rank": None,
            "ring_order": physical_ranks,
            "reduction": reduction,
            "resource_id": resource_id,
        }
        nodes.append({
            "node_id": collective_id,
            "workload_kind": "collective",
            "workload_shape": {},
            "workload_variant": None,
            "workload_dtype": None,
            "workload_pool_name": "main",
            "workload_rank": physical_ranks[0],
            "workload_attributes": attributes,
            "kernel_kind": "COMMUNICATION",
            "kernel_variant": None,
            "kernel_shape": {},
            "kernel_transfer": attributes,
            "source_symbol": source_symbol,
            "evidence_status": "validated_static_source_semantics",
        })
        for local_node_id in local_node_ids:
            edges.append((local_node_id, collective_id, "barrier"))
        previous_rank_node_ids = (collective_id, collective_id)

    return {
        "nodes": tuple(nodes),
        "edges": tuple(edges),
        "topological_order": tuple(item["node_id"] for item in nodes),
        "validation_scope": "qwen3_bf16_tp2_vllm_v1",
        "physical_ranks": physical_ranks,
        "resource_id": resource_id,
    }


def llama_tp2_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, int],
    resource_id: str,
) -> dict[str, object]:
    """Derive the Llama TP=2 DAG from pinned vLLM source semantics."""

    if not isinstance(model_path, Path):
        raise TypeError("Llama TP=2 reference model_path must be a Path")
    if forward_scope not in ("prefill", "decode", "mixed"):
        raise ValueError(
            "Llama TP=2 reference forward_scope must be prefill, decode, "
            "or mixed")
    if not isinstance(step_id, str) or not step_id:
        raise ValueError("Llama TP=2 reference step_id must be non-empty")
    if (not isinstance(query_lengths, tuple) or not query_lengths
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in query_lengths)):
        raise ValueError(
            "Llama TP=2 query_lengths must contain positive integers")
    if (not isinstance(kv_lengths, tuple)
            or len(kv_lengths) != len(query_lengths)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in kv_lengths)):
        raise ValueError(
            "Llama TP=2 kv_lengths must align with query_lengths")
    if any(kv_length < query_length for query_length, kv_length in zip(
            query_lengths, kv_lengths)):
        raise ValueError(
            "Llama TP=2 KV lengths cannot be shorter than query lengths")
    if (forward_scope == "decode"
            and any(length != 1 for length in query_lengths)):
        raise ValueError(
            "Llama TP=2 decode requires one query row per request")
    if forward_scope == "mixed" and not (
            any(length == 1 for length in query_lengths)
            and any(length > 1 for length in query_lengths)):
        raise ValueError(
            "Llama TP=2 mixed scope requires decode and prefill rows")
    if (not isinstance(physical_ranks, tuple)
            or len(physical_ranks) != 2
            or any(isinstance(rank, bool) or not isinstance(rank, int)
                   or rank < 0 for rank in physical_ranks)
            or len(set(physical_ranks)) != 2):
        raise ValueError(
            "Llama TP=2 reference requires two unique physical ranks")
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError(
            "Llama TP=2 reference resource_id must be non-empty")

    config_path = model_path / "config.json"
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if config_sha256 != (
            "ad0330f06f0e3d4defab454b7bfdff2ed3e89d2a6e7ae4679483e0cd1bfe3512"):
        raise ValueError(
            "Llama TP=2 reference requires the exact source-backed config; "
            f"got SHA256 {config_sha256}")
    config = json.loads(config_bytes)
    if config["model_type"] != "llama":
        raise ValueError("Llama TP=2 reference requires model_type=llama")
    if config["torch_dtype"] != "bfloat16":
        raise ValueError("Llama TP=2 reference requires BF16")
    if config["tie_word_embeddings"] is not False:
        raise NotImplementedError(
            "Llama TP=2 reference does not cover tied embeddings")
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    query_heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    for name, value in (
        ("hidden_size", hidden),
        ("intermediate_size", intermediate),
        ("num_attention_heads", query_heads),
        ("num_key_value_heads", kv_heads),
        ("num_hidden_layers", layers),
        ("vocab_size", vocab),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Llama config {name} must be positive")
    if hidden % query_heads:
        raise ValueError(
            "Llama TP=2 hidden size must divide query heads exactly")
    if query_heads % 2:
        raise ValueError("Llama TP=2 requires even query heads")
    if kv_heads >= 2:
        if kv_heads % 2:
            raise ValueError(
                "Llama TP=2 requires even non-replicated KV heads")
        local_kv_heads = kv_heads // 2
    else:
        local_kv_heads = 1
    if intermediate % 2:
        raise ValueError("Llama TP=2 requires even intermediate size")
    head_dim = hidden // query_heads
    local_query_heads = query_heads // 2
    local_intermediate = intermediate // 2
    padded_vocab = ((vocab + 63) // 64) * 64
    if padded_vocab % 2:
        raise ValueError("Llama TP=2 padded vocabulary must be even")
    local_vocab = padded_vocab // 2
    token_count = sum(query_lengths)
    dtype_bytes = 2
    nodes = []
    edges = []
    previous_rank_node_ids: tuple[str, ...] = ()

    def add_ranked(
        suffix: str,
        workload_kind: str,
        workload_shape: Mapping[str, int],
        workload_variant: str,
        kernel_kind: str,
        kernel_variant: str,
        kernel_shape: Mapping[str, int],
        source_symbol: str,
        *,
        workload_attributes: Mapping[str, object] | None = None,
        evidence_status: str = "validated_static_source_semantics",
        collective_operation: str | None = None,
        collective_bytes: int | None = None,
        kernel_call_count: int = 1,
        principal_call_suffix: str = "",
    ) -> None:
        nonlocal previous_rank_node_ids
        if (collective_operation is None) != (collective_bytes is None):
            raise ValueError(
                "Llama TP=2 collective operation and bytes must align")
        local_node_ids = []
        attributes = (
            {} if workload_attributes is None
            else dict(workload_attributes)
        )
        for logical_rank, physical_rank in enumerate(physical_ranks):
            node_id = f"{step_id}:{suffix}:tp-rank-{logical_rank}"
            item = {
                "node_id": node_id,
                "workload_kind": workload_kind,
                "workload_shape": dict(workload_shape),
                "workload_variant": workload_variant,
                "workload_dtype": "bf16",
                "workload_pool_name": "main",
                "workload_rank": physical_rank,
                "workload_attributes": dict(attributes),
                "kernel_kind": kernel_kind,
                "kernel_variant": kernel_variant,
                "kernel_shape": dict(kernel_shape),
                "source_symbol": source_symbol,
                "evidence_status": evidence_status,
                "kernel_call_count": kernel_call_count,
            }
            if principal_call_suffix:
                item["kernel_call_id"] = f"{node_id}{principal_call_suffix}"
            nodes.append(item)
            if previous_rank_node_ids:
                edges.append((
                    previous_rank_node_ids[logical_rank],
                    node_id,
                    "data",
                ))
            local_node_ids.append(node_id)
        if collective_operation is None:
            previous_rank_node_ids = tuple(local_node_ids)
            return
        if collective_operation == "all_reduce":
            reduction = "sum"
        elif collective_operation == "all_gather":
            reduction = None
        else:
            raise ValueError(
                "Llama TP=2 reference received unknown collective")
        collective_id = (
            f"{step_id}:{suffix}:tp-"
            f"{collective_operation.replace('_', '-')}"
        )
        transfer = {
            "algorithm": "ring",
            "bytes": collective_bytes,
            "bytes_semantics": "per_rank_input",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "direction": "collective",
            "link": "nvlink",
            "topology": "ring",
            "participants": physical_ranks,
            "operation": collective_operation,
            "source_rank": None,
            "destination_rank": None,
            "ring_order": physical_ranks,
            "reduction": reduction,
            "resource_id": resource_id,
        }
        nodes.append({
            "node_id": collective_id,
            "workload_kind": "collective",
            "workload_shape": {},
            "workload_variant": None,
            "workload_dtype": None,
            "workload_pool_name": "main",
            "workload_rank": physical_ranks[0],
            "workload_attributes": dict(transfer),
            "kernel_kind": "COMMUNICATION",
            "kernel_variant": None,
            "kernel_shape": {},
            "kernel_transfer": dict(transfer),
            "source_symbol": source_symbol,
            "evidence_status": (
                "validated_static_source_semantics_projected_latency"),
            "kernel_call_count": 1,
        })
        for local_node_id in local_node_ids:
            edges.append((local_node_id, collective_id, "barrier"))
        previous_rank_node_ids = (collective_id, collective_id)

    hidden_bytes = token_count * hidden * dtype_bytes
    add_ranked(
        "token-embedding",
        "gather",
        {
            "elements": token_count * hidden,
            "rows": token_count,
            "hidden": hidden,
        },
        "embedding",
        "VECTOR",
        "gather",
        {
            "rows": token_count,
            "hidden": hidden,
            "element_count": token_count * hidden,
            "dtype_bytes": dtype_bytes,
        },
        "VocabParallelEmbedding.forward_native:all_reduce",
        workload_attributes={
            "gather_factor": 1,
            "index_count": token_count,
            "index_bytes": 8,
            "predicate": "per-step scheduler input embedding",
            "scattered": True,
        },
        collective_operation="all_reduce",
        collective_bytes=hidden_bytes,
    )
    for layer in range(layers):
        prefix = f"layer-{layer}"
        add_ranked(
            f"{prefix}:input-norm",
            "norm",
            {"rows": token_count, "hidden": hidden},
            "rmsnorm" if layer == 0 else "add_rmsnorm",
            "VECTOR",
            "rmsnorm" if layer == 0 else "add_rmsnorm",
            {
                "rows": token_count,
                "hidden": hidden,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaDecoderLayer.forward:input_layernorm",
        )
        local_qkv = (
            local_query_heads + 2 * local_kv_heads) * head_dim
        add_ranked(
            f"{prefix}:qkv",
            "linear",
            {
                "b": 1,
                "m": token_count,
                "n": local_qkv,
                "k": hidden,
            },
            "gemm",
            "GEMM",
            "gemm",
            {
                "b": 1,
                "m": token_count,
                "n": local_qkv,
                "k": hidden,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaAttention.qkv_proj:QKVParallelLinear",
            workload_attributes=_gemm_workload_attributes("qkv_proj", 2),
        )
        rotary_elements = token_count * (
            local_query_heads + local_kv_heads) * head_dim
        add_ranked(
            f"{prefix}:rotary",
            "rotary",
            {
                "elements": rotary_elements,
                "seq_len": token_count,
                "num_query_heads": local_query_heads,
                "num_kv_heads": local_kv_heads,
                "head_dim": head_dim,
            },
            "rotary",
            "VECTOR",
            "rotary",
            {
                "element_count": rotary_elements,
                "seq_len": token_count,
                "num_query_heads": local_query_heads,
                "num_kv_heads": local_kv_heads,
                "head_dim": head_dim,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaAttention.rotary_emb",
        )
        kv_elements = token_count * local_kv_heads * head_dim
        add_ranked(
            f"{prefix}:kv-update",
            "kv_update",
            {
                "elements": kv_elements,
                "seq_len": token_count,
                "num_kv_heads": local_kv_heads,
                "head_dim": head_dim,
            },
            "paged_kv_cache",
            "KV_CACHE",
            "paged_kv_cache",
            {
                "element_count": kv_elements,
                "seq_len": token_count,
                "num_kv_heads": local_kv_heads,
                "head_dim": head_dim,
                "dtype_bytes": dtype_bytes,
            },
            "Attention.forward:stores_key_and_value_in_kv_cache",
            evidence_status="projected_logical_decomposition",
            workload_attributes=_PAGED_KV_CACHE_ATTRIBUTES,
        )
        if forward_scope == "decode":
            attention_variant = "decode"
        elif forward_scope in ("prefill", "mixed"):
            attention_variant = "prefill"
        else:
            raise RuntimeError(
                f"unhandled Llama TP=2 scope: {forward_scope!r}")
        add_ranked(
            f"{prefix}:attention",
            "flash_attn",
            {
                "batch": len(query_lengths),
                "q_len": max(query_lengths),
                "kv_len": max(kv_lengths),
                "head_dim": head_dim,
                "num_query_heads": local_query_heads,
                "num_kv_heads": local_kv_heads,
            },
            attention_variant,
            "FLASH_ATTN",
            attention_variant,
            {
                "batch": len(query_lengths),
                "q_len": max(query_lengths),
                "kv_len": max(kv_lengths),
                "head_dim": head_dim,
                "num_query_heads": local_query_heads,
                "num_kv_heads": local_kv_heads,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaAttention.attn:Attention",
            workload_attributes={
                "causal": True,
                "kv_layout": "paged_varlen",
                "kv_dist": "uniform",
                "q_lens": query_lengths,
                "kv_lens": kv_lengths,
                "fa_ksteps": None,
                "fa_eff_mfrac": 1.0,
                "launch_policy": "analytical_launch",
                "split_policy": "serving",
                "graph_runtime_mode": "NONE",
            },
            evidence_status="projected_backend_selection",
            # The backend-accepted output buffer is zero-filled
            # (Attention.forward use_output) ahead of the FA program.
            kernel_call_count=2,
            principal_call_suffix=":1",
        )
        add_ranked(
            f"{prefix}:o-proj",
            "linear",
            {
                "b": 1,
                "m": token_count,
                "n": hidden,
                "k": local_query_heads * head_dim,
            },
            "gemm",
            "GEMM",
            "gemm",
            {
                "b": 1,
                "m": token_count,
                "n": hidden,
                "k": local_query_heads * head_dim,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaAttention.o_proj:RowParallelLinear",
            workload_attributes=_gemm_workload_attributes("o_proj", 2),
            collective_operation="all_reduce",
            collective_bytes=hidden_bytes,
        )
        add_ranked(
            f"{prefix}:post-attn-norm",
            "norm",
            {"rows": token_count, "hidden": hidden},
            "add_rmsnorm",
            "VECTOR",
            "add_rmsnorm",
            {
                "rows": token_count,
                "hidden": hidden,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaDecoderLayer.forward:post_attention_layernorm",
        )
        add_ranked(
            f"{prefix}:gate-up",
            "linear",
            {
                "b": 1,
                "m": token_count,
                "n": 2 * local_intermediate,
                "k": hidden,
            },
            "gemm",
            "GEMM",
            "gemm",
            {
                "b": 1,
                "m": token_count,
                "n": 2 * local_intermediate,
                "k": hidden,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaMLP.gate_up_proj:MergedColumnParallelLinear",
            workload_attributes=_gemm_workload_attributes(
                "gate_up_proj", 2),
        )
        activation_elements = token_count * local_intermediate
        add_ranked(
            f"{prefix}:activation",
            "activation",
            {
                "elements": activation_elements,
                "tok": token_count,
                "intermediate": local_intermediate,
            },
            "silu",
            "VECTOR",
            "silu",
            {
                "element_count": activation_elements,
                "tok": token_count,
                "intermediate": local_intermediate,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaMLP.act_fn:SiluAndMul",
        )
        add_ranked(
            f"{prefix}:down",
            "linear",
            {
                "b": 1,
                "m": token_count,
                "n": hidden,
                "k": local_intermediate,
            },
            "gemm",
            "GEMM",
            "gemm",
            {
                "b": 1,
                "m": token_count,
                "n": hidden,
                "k": local_intermediate,
                "dtype_bytes": dtype_bytes,
            },
            "LlamaMLP.down_proj:RowParallelLinear",
            workload_attributes=_gemm_workload_attributes("down_proj", 2),
            collective_operation="all_reduce",
            collective_bytes=hidden_bytes,
        )
    add_ranked(
        "final-norm",
        "norm",
        {"rows": token_count, "hidden": hidden},
        "add_rmsnorm",
        "VECTOR",
        "add_rmsnorm",
        {
            "rows": token_count,
            "hidden": hidden,
            "dtype_bytes": dtype_bytes,
        },
        "LlamaModel.norm:RMSNorm",
    )
    logits_rows = len(query_lengths)
    add_ranked(
        "logits-boundary-gather",
        "gather",
        {"elements": logits_rows * hidden, "rows": logits_rows,
         "hidden": hidden},
        "logits_indices",
        "VECTOR",
        "gather",
        {"rows": logits_rows, "hidden": hidden,
         "element_count": logits_rows * hidden,
         "dtype_bytes": dtype_bytes},
        "vllm.v1.worker.gpu_model_runner.sample_hidden_states",
        workload_attributes={
            "gather_factor": 1,
            "index_count": logits_rows,
            "index_bytes": 8,
            "predicate": "per-step target logit row selection",
            "scattered": True,
            "source_symbol": (
                "vllm.v1.worker.gpu_model_runner.sample_hidden_states"),
        },
    )
    add_ranked(
        "lm-head",
        "lm_head",
        {"b": 1, "m": logits_rows, "n": local_vocab, "k": hidden},
        "gemv" if logits_rows == 1 else "gemm",
        "GEMM",
        "gemv" if logits_rows == 1 else "gemm",
        {
            "b": 1,
            "m": logits_rows,
            "n": local_vocab,
            "k": hidden,
            "dtype_bytes": dtype_bytes,
        },
        "LlamaForCausalLM.compute_logits:ParallelLMHead:all_gather",
        workload_attributes=_gemm_workload_attributes("lm_head", 2),
        collective_operation="all_gather",
        collective_bytes=logits_rows * local_vocab * dtype_bytes,
    )
    return {
        "nodes": tuple(nodes),
        "edges": tuple(edges),
        "topological_order": tuple(item["node_id"] for item in nodes),
        "validation_scope": "llama_bf16_tp2_vllm_v1",
        "physical_ranks": physical_ranks,
        "resource_id": resource_id,
        "local_query_heads": local_query_heads,
        "local_kv_heads": local_kv_heads,
    }


def llama_af_tp2_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    attention_ranks: tuple[int, int],
    ffn_ranks: tuple[int, int],
    attention_resource_id: str,
    ffn_resource_id: str,
    activation_resource_id: str,
) -> dict[str, object]:
    """Derive the exact one-layer Llama AF+TP2 DAG from source semantics."""

    if (not isinstance(ffn_ranks, tuple)
            or len(ffn_ranks) != 2
            or any(isinstance(rank, bool) or not isinstance(rank, int)
                   or rank < 0 for rank in ffn_ranks)
            or len(set(ffn_ranks)) != 2):
        raise ValueError(
            "Llama AF+TP2 reference requires two unique FFN ranks")
    if set(attention_ranks).intersection(ffn_ranks):
        raise ValueError(
            "Llama AF+TP2 Attention and FFN ranks must be disjoint")
    for name, value in (
        ("attention_resource_id", attention_resource_id),
        ("ffn_resource_id", ffn_resource_id),
        ("activation_resource_id", activation_resource_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Llama AF+TP2 {name} must be a non-empty string")

    base = llama_tp2_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        physical_ranks=attention_ranks,
        resource_id=attention_resource_id,
    )
    if any(f"{step_id}:layer-1:" in item["node_id"]
           for item in base["nodes"]):
        raise NotImplementedError(
            "Llama AF+TP2 reference is restricted to the exact one-layer "
            "fixture")

    if forward_scope == "prefill":
        # The A800 prefill token-embedding gather is a versioned measured
        # zero (launch placement outside the capture window), so the
        # reference drops it together with its all-reduce and re-anchors the
        # first residual transfer on the pool's last in-window producer,
        # matching the observation-suppressed emission contract.
        embedding_ids = frozenset({
            f"{step_id}:token-embedding:tp-rank-0",
            f"{step_id}:token-embedding:tp-rank-1",
            f"{step_id}:token-embedding:tp-all-reduce",
        })
        base = dict(base)
        base["nodes"] = tuple(
            item for item in base["nodes"]
            if item["node_id"] not in embedding_ids)
        base["edges"] = tuple(
            edge for edge in base["edges"]
            if edge[0] not in embedding_ids
            and edge[1] not in embedding_ids)

    hidden = next(
        item["workload_shape"]["hidden"]
        for item in base["nodes"]
        if item["node_id"] == f"{step_id}:layer-0:input-norm:tp-rank-0"
    )
    rows = sum(query_lengths)
    payload_bytes = rows * hidden * 2
    ffn_fragments = (
        ":layer-0:post-attn-norm:",
        ":layer-0:gate-up:",
        ":layer-0:activation:",
        ":layer-0:down:",
    )

    def physical_rank(node_id: str, ranks: tuple[int, int]) -> int:
        if node_id.endswith(":tp-rank-0"):
            return ranks[0]
        if node_id.endswith(":tp-rank-1"):
            return ranks[1]
        return ranks[0]

    def move_to_ffn(source: Mapping[str, object]) -> dict[str, object]:
        item = dict(source)
        node_id = item["node_id"]
        rank = physical_rank(node_id, ffn_ranks)
        item["workload_pool_name"] = "ffn"
        item["workload_rank"] = rank
        if item["kernel_kind"] == "COMMUNICATION":
            attributes = dict(item["workload_attributes"])
            attributes["participants"] = ffn_ranks
            attributes["ring_order"] = ffn_ranks
            attributes["resource_id"] = ffn_resource_id
            item["workload_attributes"] = attributes
            transfer = dict(item["kernel_transfer"])
            transfer["participants"] = ffn_ranks
            transfer["ring_order"] = ffn_ranks
            transfer["resource_id"] = ffn_resource_id
            item["kernel_transfer"] = transfer
            item["kernel_resource"] = ffn_resource_id
        else:
            item["kernel_resource"] = "compute"
        item["kernel_call_id"] = node_id
        item["kernel_pool_name"] = "ffn"
        item["kernel_rank"] = rank
        return item

    def keep_on_attention(source: Mapping[str, object]) -> dict[str, object]:
        item = dict(source)
        node_id = item["node_id"]
        rank = item["workload_rank"]
        item["workload_pool_name"] = "attention"
        source_call_id = item.get("kernel_call_id")
        item["kernel_call_id"] = (
            source_call_id
            if source_call_id is not None
            and source_call_id.startswith(node_id)
            else node_id)
        item["kernel_pool_name"] = "attention"
        item["kernel_rank"] = rank
        if item["kernel_kind"] == "COMMUNICATION":
            item["kernel_resource"] = attention_resource_id
        else:
            item["kernel_resource"] = "compute"
        return item

    def activation_transfer(
        boundary_index: int,
        tensor_name: str,
        lane_index: int,
        source_pool: str,
        destination_pool: str,
        source_ranks: tuple[int, int],
        destination_ranks: tuple[int, int],
    ) -> dict[str, object]:
        source_rank = source_ranks[lane_index]
        destination_rank = destination_ranks[lane_index]
        node_id = (
            f"{step_id}:af-transfer-{boundary_index}:"
            f"{tensor_name.replace('_', '-')}:tp-rank-{lane_index}"
        )
        transfer = {
            "algorithm": "send_recv",
            "bytes": payload_bytes,
            "bytes_semantics": "payload",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "direction": "point_to_point",
            "link": "nvlink",
            "topology": "point_to_point",
            "participants": (source_rank, destination_rank),
            "operation": "send_recv",
            "source_rank": source_rank,
            "destination_rank": destination_rank,
            "ring_order": (),
            "reduction": None,
            "resource_id": activation_resource_id,
        }
        attributes = dict(transfer)
        attributes["destination_pool"] = destination_pool
        attributes["payload_tensor"] = {
            "name": tensor_name,
            "shape": (rows, hidden),
            "dtype": "bf16",
            "bytes": payload_bytes,
        }
        return {
            "node_id": node_id,
            "workload_kind": "af_activation_transfer",
            "workload_shape": {},
            "workload_variant": None,
            "workload_dtype": None,
            "workload_pool_name": source_pool,
            "workload_rank": source_rank,
            "workload_attributes": attributes,
            "kernel_kind": "COMMUNICATION",
            "kernel_variant": None,
            "kernel_shape": {},
            "kernel_transfer": transfer,
            "kernel_call_count": 1,
            "kernel_call_id": node_id,
            "kernel_pool_name": source_pool,
            "kernel_rank": source_rank,
            "kernel_resource": activation_resource_id,
            "source_symbol": (
                "LlamaDecoderLayer.forward:hidden_states_residual_boundary"),
            "evidence_status": (
                "validated_static_source_semantics_projected_latency"),
        }

    first_boundary_nodes = tuple(
        activation_transfer(
            0,
            tensor_name,
            lane_index,
            "attention",
            "ffn",
            attention_ranks,
            ffn_ranks,
        )
        for lane_index in range(2)
        for tensor_name in ("hidden_states", "residual")
    )
    if forward_scope == "prefill":
        first_boundary_nodes = tuple(
            dict(
                node,
                workload_attributes=dict(
                    node["workload_attributes"],
                    residual_anchor=RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED,
                ),
            ) if ":residual:" in node["node_id"] else node
            for node in first_boundary_nodes
        )
    second_boundary_nodes = tuple(
        activation_transfer(
            1,
            tensor_name,
            lane_index,
            "ffn",
            "attention",
            ffn_ranks,
            attention_ranks,
        )
        for lane_index in range(2)
        for tensor_name in ("hidden_states", "residual")
    )
    o_collective = f"{step_id}:layer-0:o-proj:tp-all-reduce"
    down_collective = f"{step_id}:layer-0:down:tp-all-reduce"
    nodes = []
    for source in base["nodes"]:
        node_id = source["node_id"]
        if any(fragment in node_id for fragment in ffn_fragments):
            nodes.append(move_to_ffn(source))
        else:
            nodes.append(keep_on_attention(source))
        if node_id == o_collective:
            nodes.extend(first_boundary_nodes)
        elif node_id == down_collective:
            nodes.extend(second_boundary_nodes)

    token_embedding = f"{step_id}:token-embedding:tp-all-reduce"
    first_residual_sources = (
        (o_collective, o_collective)
        if forward_scope == "prefill"
        else (token_embedding, token_embedding)
    )
    post_norm = tuple(
        f"{step_id}:layer-0:post-attn-norm:tp-rank-{lane_index}"
        for lane_index in range(2)
    )
    final_norm = tuple(
        f"{step_id}:final-norm:tp-rank-{lane_index}"
        for lane_index in range(2)
    )

    def boundary_edges(
        boundary_index: int,
        hidden_source: str,
        residual_sources: tuple[str, str],
        consumers: tuple[str, str],
    ) -> tuple[tuple[str, str, str], ...]:
        result = []
        for lane_index in range(2):
            hidden_transfer = (
                f"{step_id}:af-transfer-{boundary_index}:"
                f"hidden-states:tp-rank-{lane_index}")
            residual_transfer = (
                f"{step_id}:af-transfer-{boundary_index}:"
                f"residual:tp-rank-{lane_index}")
            result.append((hidden_source, hidden_transfer, "ready"))
            result.append((
                residual_sources[lane_index],
                residual_transfer,
                "ready",
            ))
        for lane_index in range(2):
            hidden_transfer = (
                f"{step_id}:af-transfer-{boundary_index}:"
                f"hidden-states:tp-rank-{lane_index}")
            residual_transfer = (
                f"{step_id}:af-transfer-{boundary_index}:"
                f"residual:tp-rank-{lane_index}")
            result.append((hidden_transfer, consumers[lane_index], "data"))
            result.append((residual_transfer, consumers[lane_index], "data"))
        return tuple(result)

    first_edges = boundary_edges(
        0,
        o_collective,
        first_residual_sources,
        post_norm,
    )
    second_edges = boundary_edges(
        1,
        down_collective,
        post_norm,
        final_norm,
    )
    edges = []
    inserted_first = False
    inserted_second = False
    for edge in base["edges"]:
        source, destination, _ = edge
        if source == o_collective and destination in post_norm:
            if not inserted_first:
                edges.extend(first_edges)
                inserted_first = True
            continue
        if source == down_collective and destination in final_norm:
            if not inserted_second:
                edges.extend(second_edges)
                inserted_second = True
            continue
        edges.append(edge)
    if not inserted_first or not inserted_second:
        raise RuntimeError(
            "Llama AF+TP2 reference did not replace both AF boundaries")

    indegree = {item["node_id"]: 0 for item in nodes}
    successors = {item["node_id"]: [] for item in nodes}
    for source, destination, _ in edges:
        indegree[destination] += 1
        successors[source].append(destination)
    ready = [item["node_id"] for item in nodes
             if indegree[item["node_id"]] == 0]
    topological_order = []
    while ready:
        current = ready.pop(0)
        topological_order.append(current)
        for successor in successors[current]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(topological_order) != len(nodes):
        raise RuntimeError("Llama AF+TP2 reference must be acyclic")

    return {
        "nodes": tuple(nodes),
        "edges": tuple(edges),
        "topological_order": tuple(topological_order),
        "validation_scope": "llama_bf16_af_tp2_vllm_v1",
        "attention_ranks": attention_ranks,
        "ffn_ranks": ffn_ranks,
        "attention_resource_id": attention_resource_id,
        "ffn_resource_id": ffn_resource_id,
        "activation_resource_id": activation_resource_id,
        "activation_tensors": ("hidden_states", "residual"),
        "activation_transfers_per_step": 8,
        "activation_bytes_per_tensor": payload_bytes,
    }


def _qwen3_uniform_pp_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, ...],
    resource_id: str,
) -> dict[str, object]:
    """Derive a validated uniform PP DAG from pinned vLLM semantics."""

    if not isinstance(physical_ranks, tuple):
        raise ValueError("uniform PP reference physical_ranks must be a tuple")
    pipeline_size = len(physical_ranks)
    if (pipeline_size < 2
            or any(isinstance(rank, bool) or not isinstance(rank, int)
                   or rank < 0 for rank in physical_ranks)
            or len(set(physical_ranks)) != pipeline_size):
        raise ValueError(
            "uniform PP reference requires at least two unique physical ranks")
    if not isinstance(resource_id, str) or not resource_id:
        raise ValueError("uniform PP reference resource_id must be non-empty")
    base = qwen3_single_rank_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
    )
    config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    if config["torch_dtype"] != "bfloat16":
        raise ValueError("uniform PP reference requires BF16")
    if config["tie_word_embeddings"] is not False:
        raise NotImplementedError(
            "uniform PP reference does not cover tied embedding/LM-head "
            "weights")
    num_layers = config["num_hidden_layers"]
    if num_layers % pipeline_size:
        raise ValueError(
            "uniform PP reference requires decoder layers divisible by PP")
    hidden = config["hidden_size"]
    token_count = sum(query_lengths)
    tensor_bytes = token_count * hidden * 2
    layers_per_stage = num_layers // pipeline_size
    nodes = []
    edges = []
    previous_node_id = None
    previous_stage = None

    def stage_for_suffix(suffix: str) -> int:
        if suffix == "token-embedding":
            return 0
        if suffix in ("final-norm", "logits-boundary-gather", "lm-head"):
            return pipeline_size - 1
        if not suffix.startswith("layer-"):
            raise ValueError(
                f"unclassified PP reference node suffix: {suffix!r}")
        layer_prefix = suffix.split(":", 1)[0]
        layer_index = int(layer_prefix.removeprefix("layer-"))
        return min(layer_index // layers_per_stage, pipeline_size - 1)

    for base_item in base:
        item = dict(base_item)
        suffix = item["node_id"][len(step_id) + 1:]
        stage = stage_for_suffix(suffix)
        if previous_stage is not None and stage != previous_stage:
            if stage != previous_stage + 1:
                raise RuntimeError(
                    "uniform PP reference encountered a non-adjacent forward "
                    "stage boundary")
            for tensor_name in ("hidden_states", "residual"):
                node_id = (
                    f"{step_id}:pp-{previous_stage}-to-{stage}:"
                    f"{tensor_name.replace('_', '-')}")
                attributes = {
                    "algorithm": "send_recv",
                    "bytes": tensor_bytes,
                    "bytes_semantics": "payload",
                    "backend": "analytical_roofline_v1",
                    "backend_config_sha256": None,
                    "direction": "point_to_point",
                    "link": "nvlink",
                    "topology": "point_to_point",
                    "participants": (
                        physical_ranks[previous_stage], physical_ranks[stage]),
                    "operation": "send_recv",
                    "source_rank": physical_ranks[previous_stage],
                    "destination_rank": physical_ranks[stage],
                    "ring_order": (),
                    "reduction": None,
                    "resource_id": resource_id,
                    "source_stage": previous_stage,
                    "destination_stage": stage,
                    "payload_tensor": {
                        "name": tensor_name,
                        "shape": (token_count, hidden),
                        "dtype": "bf16",
                        "bytes": tensor_bytes,
                    },
                }
                physical_transfer = {
                    name: attributes[name]
                    for name in (
                        "algorithm",
                        "bytes",
                        "bytes_semantics",
                        "backend",
                        "backend_config_sha256",
                        "direction",
                        "link",
                        "topology",
                        "participants",
                        "operation",
                        "source_rank",
                        "destination_rank",
                        "ring_order",
                        "reduction",
                        "resource_id",
                    )
                }
                nodes.append({
                    "node_id": node_id,
                    "workload_kind": "p2p",
                    "workload_shape": {},
                    "workload_variant": None,
                    "workload_dtype": None,
                    "workload_pool_name": "main",
                    "workload_rank": physical_ranks[previous_stage],
                    "workload_attributes": attributes,
                    "kernel_kind": "COMMUNICATION",
                    "kernel_variant": None,
                    "kernel_shape": {},
                    "kernel_transfer": physical_transfer,
                    "source_symbol": (
                        "GroupCoordinator.send_tensor_dict:"
                        f"tensor[{tensor_name}]"),
                    "evidence_status": (
                        "validated_static_source_semantics_projected_latency"),
                })
                if previous_node_id is not None:
                    edges.append((previous_node_id, node_id, "ready"))
                previous_node_id = node_id
        item["workload_rank"] = physical_ranks[stage]
        nodes.append(item)
        if previous_node_id is not None:
            edges.append((previous_node_id, item["node_id"], "data"))
        previous_node_id = item["node_id"]
        previous_stage = stage

    if pipeline_size == 2:
        validation_scope = "qwen3_bf16_pp2_vllm_v1"
    elif pipeline_size == 3:
        validation_scope = "qwen3_bf16_pp3_vllm_v1"
    else:
        validation_scope = "qwen3_bf16_uniform_pp_vllm_v1"
    return {
        "nodes": tuple(nodes),
        "edges": tuple(edges),
        "topological_order": tuple(item["node_id"] for item in nodes),
        "validation_scope": validation_scope,
        "physical_ranks": physical_ranks,
        "resource_id": resource_id,
        "pipeline_tensor_order": ("hidden_states", "residual"),
        "high_level_send_tensor_dict_calls_per_step": pipeline_size - 1,
        "physical_tensor_sends_per_step": 2 * (pipeline_size - 1),
    }


def qwen3_pp2_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, int],
    resource_id: str,
) -> dict[str, object]:
    return _qwen3_uniform_pp_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        physical_ranks=physical_ranks,
        resource_id=resource_id,
    )


def qwen3_pp3_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, int, int],
    resource_id: str,
) -> dict[str, object]:
    return _qwen3_uniform_pp_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        physical_ranks=physical_ranks,
        resource_id=resource_id,
    )


def qwen3_uniform_pp_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_ranks: tuple[int, ...],
    resource_id: str,
) -> dict[str, object]:
    return _qwen3_uniform_pp_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        physical_ranks=physical_ranks,
        resource_id=resource_id,
    )


def qwen3_tp2_pp2_forward_reference(
    model_path: Path,
    *,
    forward_scope: str,
    step_id: str,
    query_lengths: tuple[int, ...],
    kv_lengths: tuple[int, ...],
    physical_rank_groups: tuple[tuple[int, int], tuple[int, int]],
    tensor_resource_id: str,
    pipeline_resource_id: str,
) -> dict[str, object]:
    """Derive the exact TP=2, PP=2 DAG from pinned vLLM semantics."""

    if (not isinstance(physical_rank_groups, tuple)
            or len(physical_rank_groups) != 2
            or any(not isinstance(group, tuple) or len(group) != 2
                   for group in physical_rank_groups)):
        raise ValueError(
            "TP=2, PP=2 reference requires two two-rank TP groups")
    physical_ranks = tuple(
        rank for group in physical_rank_groups for rank in group)
    if (any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in physical_ranks)
            or len(set(physical_ranks)) != 4):
        raise ValueError(
            "TP=2, PP=2 reference requires four unique physical ranks")
    for name, value in (
            ("tensor_resource_id", tensor_resource_id),
            ("pipeline_resource_id", pipeline_resource_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
    config = json.loads(
        (model_path / "config.json").read_text(encoding="utf-8"))
    if config["torch_dtype"] != "bfloat16":
        raise ValueError("TP=2, PP=2 reference requires BF16")
    if config["tie_word_embeddings"] is not False:
        raise NotImplementedError(
            "TP=2, PP=2 reference does not cover tied embeddings")
    num_layers = config["num_hidden_layers"]
    if num_layers % 2:
        raise ValueError(
            "TP=2, PP=2 reference requires an even decoder layer count")
    hidden = config["hidden_size"]
    token_count = sum(query_lengths)
    tensor_elements = token_count * hidden
    if tensor_elements % 2:
        raise ValueError(
            "pipeline tensor elements must divide the TP all-gather size")
    wire_elements = tensor_elements // 2
    wire_bytes = wire_elements * 2
    stage_split = num_layers // 2
    base = qwen3_tp2_forward_reference(
        model_path,
        forward_scope=forward_scope,
        step_id=step_id,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        physical_ranks=physical_rank_groups[0],
        resource_id=tensor_resource_id,
    )

    def stage_for_node_id(node_id: str) -> int:
        suffix = node_id[len(step_id) + 1:]
        if suffix.startswith("token-embedding:"):
            return 0
        if (suffix.startswith("final-norm:")
                or suffix.startswith("logits-boundary-gather:")
                or suffix.startswith("lm-head:")):
            return 1
        if not suffix.startswith("layer-"):
            raise ValueError(
                f"unclassified TP=2, PP=2 node suffix: {suffix!r}")
        layer_prefix = suffix.split(":", 1)[0]
        layer_index = int(layer_prefix.removeprefix("layer-"))
        return 0 if layer_index < stage_split else 1

    stage_by_node_id = {
        item["node_id"]: stage_for_node_id(item["node_id"])
        for item in base["nodes"]
    }
    cross_stage_edges = tuple(
        edge for edge in base["edges"]
        if stage_by_node_id[edge[0]] != stage_by_node_id[edge[1]]
    )
    if (not cross_stage_edges
            or any((stage_by_node_id[source], stage_by_node_id[destination])
                   != (0, 1)
                   for source, destination, _ in cross_stage_edges)):
        raise RuntimeError(
            "TP=2 reference does not expose one forward PP boundary")
    boundary_sources = tuple(dict.fromkeys(
        source for source, _, _ in cross_stage_edges))
    boundary_destinations = tuple(dict.fromkeys(
        destination for _, destination, _ in cross_stage_edges))
    if len(boundary_sources) != 1 or len(boundary_destinations) != 2:
        raise RuntimeError(
            "TP=2, PP=2 boundary requires one replicated source frontier "
            "and two destination-rank compute nodes")

    nodes = []
    inserted_boundary = False
    for base_item in base["nodes"]:
        stage = stage_by_node_id[base_item["node_id"]]
        if stage == 1 and not inserted_boundary:
            for tensor_name in ("hidden_states", "residual"):
                for lane_index in range(2):
                    source_rank = physical_rank_groups[0][lane_index]
                    destination_rank = physical_rank_groups[1][lane_index]
                    node_id = (
                        f"{step_id}:pp-0-to-1:"
                        f"{tensor_name.replace('_', '-')}:"
                        f"tp-rank-{lane_index}")
                    attributes = {
                        "algorithm": "send_recv",
                        "bytes": wire_bytes,
                        "bytes_semantics": "payload",
                        "backend": "analytical_roofline_v1",
                        "backend_config_sha256": None,
                        "direction": "point_to_point",
                        "link": "nvlink",
                        "topology": "point_to_point",
                        "participants": (source_rank, destination_rank),
                        "operation": "send_recv",
                        "source_rank": source_rank,
                        "destination_rank": destination_rank,
                        "ring_order": (),
                        "reduction": None,
                        "resource_id": pipeline_resource_id,
                        "source_stage": 0,
                        "destination_stage": 1,
                        "payload_tensor": {
                            "name": tensor_name,
                            "shape": (wire_elements,),
                            "dtype": "bf16",
                            "bytes": wire_bytes,
                        },
                    }
                    physical_transfer = {
                        name: attributes[name]
                        for name in (
                            "algorithm", "bytes", "bytes_semantics",
                            "backend", "backend_config_sha256", "direction",
                            "link", "topology", "participants", "operation",
                            "source_rank", "destination_rank", "ring_order",
                            "reduction", "resource_id",
                        )
                    }
                    nodes.append({
                        "node_id": node_id,
                        "workload_kind": "p2p",
                        "workload_shape": {},
                        "workload_variant": None,
                        "workload_dtype": None,
                        "workload_pool_name": "main",
                        "workload_rank": source_rank,
                        "workload_attributes": attributes,
                        "kernel_kind": "COMMUNICATION",
                        "kernel_variant": None,
                        "kernel_shape": {},
                        "kernel_transfer": physical_transfer,
                        "source_symbol": (
                            "GroupCoordinator.send_tensor_dict:"
                            f"reshape(TP,-1)[{lane_index}]:{tensor_name}"),
                        "evidence_status": (
                            "validated_static_source_semantics_"
                            "projected_latency"),
                    })
            for tensor_name in ("hidden_states", "residual"):
                collective_id = (
                    f"{step_id}:pp-0-to-1:"
                    f"{tensor_name.replace('_', '-')}:tp-all-gather")
                collective_attributes = {
                    "algorithm": "ring",
                    "bytes": wire_bytes,
                    "bytes_semantics": "per_rank_input",
                    "backend": "analytical_roofline_v1",
                    "backend_config_sha256": None,
                    "direction": "collective",
                    "link": "nvlink",
                    "topology": "ring",
                    "participants": physical_rank_groups[1],
                    "operation": "all_gather",
                    "source_rank": None,
                    "destination_rank": None,
                    "ring_order": physical_rank_groups[1],
                    "reduction": None,
                    "resource_id": tensor_resource_id,
                }
                nodes.append({
                    "node_id": collective_id,
                    "workload_kind": "collective",
                    "workload_shape": {},
                    "workload_variant": None,
                    "workload_dtype": None,
                    "workload_pool_name": "main",
                    "workload_rank": physical_rank_groups[1][0],
                    "workload_attributes": collective_attributes,
                    "kernel_kind": "COMMUNICATION",
                    "kernel_variant": None,
                    "kernel_shape": {},
                    "kernel_transfer": collective_attributes,
                    "source_symbol": (
                        "GroupCoordinator.recv_tensor_dict:"
                        f"all_gather_group.all_gather:{tensor_name}"),
                    "evidence_status": (
                        "validated_static_source_semantics_projected_latency"),
                })
            inserted_boundary = True

        item = dict(base_item)
        if item["workload_kind"] == "collective":
            attributes = dict(item["workload_attributes"])
            attributes["participants"] = physical_rank_groups[stage]
            attributes["ring_order"] = physical_rank_groups[stage]
            attributes["resource_id"] = tensor_resource_id
            item["workload_attributes"] = attributes
            transfer = dict(item["kernel_transfer"])
            transfer["participants"] = physical_rank_groups[stage]
            transfer["ring_order"] = physical_rank_groups[stage]
            transfer["resource_id"] = tensor_resource_id
            item["kernel_transfer"] = transfer
            item["workload_rank"] = physical_rank_groups[stage][0]
        else:
            lane_index = int(item["node_id"].rsplit(
                ":tp-rank-", 1)[1])
            item["workload_rank"] = physical_rank_groups[stage][lane_index]
        nodes.append(item)
    if not inserted_boundary:
        raise RuntimeError("TP=2, PP=2 reference did not insert its boundary")

    boundary_edges = []
    source_frontier = boundary_sources[0]
    for tensor_name in ("hidden_states", "residual"):
        p2p_ids = []
        for lane_index in range(2):
            p2p_id = (
                f"{step_id}:pp-0-to-1:"
                f"{tensor_name.replace('_', '-')}:tp-rank-{lane_index}")
            boundary_edges.append((source_frontier, p2p_id, "ready"))
            p2p_ids.append(p2p_id)
    for tensor_name in ("hidden_states", "residual"):
        p2p_ids = tuple(
            f"{step_id}:pp-0-to-1:{tensor_name.replace('_', '-')}:"
            f"tp-rank-{lane_index}"
            for lane_index in range(2)
        )
        collective_id = (
            f"{step_id}:pp-0-to-1:"
            f"{tensor_name.replace('_', '-')}:tp-all-gather")
        for p2p_id in p2p_ids:
            boundary_edges.append((p2p_id, collective_id, "barrier"))
    boundary_collectives = tuple(
        f"{step_id}:pp-0-to-1:{tensor_name.replace('_', '-')}:"
        "tp-all-gather"
        for tensor_name in ("hidden_states", "residual")
    )
    for destination in boundary_destinations:
        for collective_id in boundary_collectives:
            boundary_edges.append((collective_id, destination, "data"))
    edges = []
    inserted_boundary_edges = False
    for edge in base["edges"]:
        if stage_by_node_id[edge[0]] != stage_by_node_id[edge[1]]:
            if not inserted_boundary_edges:
                edges.extend(boundary_edges)
                inserted_boundary_edges = True
            continue
        edges.append(edge)
    if not inserted_boundary_edges:
        raise RuntimeError(
            "TP=2, PP=2 reference did not replace its cross-stage edges")

    return {
        "nodes": tuple(nodes),
        "edges": tuple(edges),
        "topological_order": tuple(item["node_id"] for item in nodes),
        "validation_scope": "qwen3_bf16_tp2_pp2_vllm_v1",
        "physical_rank_groups": physical_rank_groups,
        "tensor_resource_id": tensor_resource_id,
        "pipeline_resource_id": pipeline_resource_id,
        "pipeline_tensor_order": ("hidden_states", "residual"),
        "physical_tensor_sends_per_step": 4,
        "destination_all_gathers_per_step": 2,
        "pipeline_wire_elements_per_rank": wire_elements,
        "pipeline_wire_bytes_per_rank": wire_bytes,
    }


def compare_tp2_reference_to_emission(
    reference: Mapping[str, object],
    graph,
    plan,
) -> dict[str, object]:
    """Compare a complete TP=2 rank-local DAG and lowering without timing."""

    expected_nodes = tuple(reference["nodes"])
    strict_workload_attributes = reference["validation_scope"] in (
        "llama_bf16_tp2_vllm_v1",
        "llama_bf16_af_tp2_vllm_v1",
    )
    expected_by_id = {item["node_id"]: item for item in expected_nodes}
    actual_by_id = {node.node_id: node for node in graph.nodes}
    missing = tuple(sorted(set(expected_by_id) - set(actual_by_id)))
    extra = tuple(sorted(set(actual_by_id) - set(expected_by_id)))
    calls_by_node_id = {node_id: [] for node_id in actual_by_id}
    for call in plan.calls:
        for node_id in call.source_node_ids:
            if node_id not in calls_by_node_id:
                raise ValueError(
                    f"TP KernelCall references unknown node {node_id!r}")
            calls_by_node_id[node_id].append(call)
    expected_predecessors = {node_id: [] for node_id in expected_by_id}
    for source, destination, _ in reference["edges"]:
        expected_predecessors[destination].append(source)
    matched = []
    mismatched = []
    for node_id in sorted(set(expected_by_id).intersection(actual_by_id)):
        expected = expected_by_id[node_id]
        node = actual_by_id[node_id]
        differences = []
        for field, actual_value, expected_value in (
            ("workload_kind", node.kind.value, expected["workload_kind"]),
            ("workload_variant", node.variant,
             expected["workload_variant"]),
            ("workload_shape", dict(node.shape), expected["workload_shape"]),
            ("workload_dtype", node.dtype, expected["workload_dtype"]),
            ("workload_pool_name", node.pool_name,
             expected["workload_pool_name"]),
            ("workload_rank", node.rank, expected["workload_rank"]),
        ):
            if actual_value != expected_value:
                differences.append({
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                })
        expected_attributes = {
            name: value
            for name, value in expected["workload_attributes"].items()
            if name != "launch_policy"
        }
        actual_attributes = {
            name: value
            for name, value in node.attributes.items()
            if name != "launch_policy"
        }
        if (strict_workload_attributes
                and actual_attributes != expected_attributes):
            differences.append({
                "field": "workload_attributes",
                "expected": expected_attributes,
                "actual": actual_attributes,
            })
        elif not strict_workload_attributes:
            for attribute, expected_value in expected_attributes.items():
                if attribute not in node.attributes:
                    actual_value = None
                else:
                    actual_value = node.attributes[attribute]
                if actual_value == expected_value:
                    continue
                differences.append({
                    "field": f"workload_attributes.{attribute}",
                    "expected": expected_value,
                    "actual": actual_value,
                })
        calls = calls_by_node_id[node_id]
        if "kernel_call_count" in expected:
            expected_call_count = expected["kernel_call_count"]
            if len(calls) != expected_call_count:
                differences.append({
                    "field": "kernel_call_count",
                    "expected": expected_call_count,
                    "actual": len(calls),
                })
        elif not calls:
            differences.append({
                "field": "kernel_call_count",
                "expected": ">=1",
                "actual": 0,
            })
        if calls:
            matching_calls = tuple(
                call for call in calls
                if (call.kernel_spec.kind.value == expected["kernel_kind"]
                    and (expected["kernel_kind"] == "COMMUNICATION"
                         or (call.kernel_spec.variant
                             == expected["kernel_variant"]
                             and all(
                                 call.kernel_spec.shape.get(name) == value
                                 for name, value
                                 in expected["kernel_shape"].items()))))
            )
            if len(matching_calls) != 1:
                differences.append({
                    "field": "principal_kernel_call_count",
                    "expected": 1,
                    "actual": len(matching_calls),
                })
                principal_call = None
            else:
                principal_call = matching_calls[0]
        else:
            principal_call = None
        if principal_call is not None:
            for expected_field, actual_value in (
                ("kernel_call_id", principal_call.call_id),
                ("kernel_pool_name", principal_call.pool_name),
                ("kernel_rank", principal_call.rank),
                ("kernel_resource", principal_call.resource),
            ):
                if (expected_field in expected
                        and actual_value != expected[expected_field]):
                    differences.append({
                        "field": expected_field,
                        "expected": expected[expected_field],
                        "actual": actual_value,
                    })
            spec = principal_call.kernel_spec
            if spec.kind.value != expected["kernel_kind"]:
                differences.append({
                    "field": "kernel_kind",
                    "expected": expected["kernel_kind"],
                    "actual": spec.kind.value,
                })
            if expected["kernel_kind"] == "COMMUNICATION":
                transfer = spec.transfer
                for field, expected_value in expected[
                        "kernel_transfer"].items():
                    actual_value = getattr(transfer, field)
                    if actual_value != expected_value:
                        differences.append({
                            "field": f"kernel_transfer.{field}",
                            "expected": expected_value,
                            "actual": actual_value,
                        })
            else:
                for field, actual_value, expected_value in (
                    ("kernel_variant", spec.variant,
                     expected["kernel_variant"]),
                    ("kernel_dtype", spec.dtype, "bf16"),
                ):
                    if actual_value != expected_value:
                        differences.append({
                            "field": field,
                            "expected": expected_value,
                            "actual": actual_value,
                        })
                for shape_field, expected_value in expected[
                        "kernel_shape"].items():
                    if shape_field not in spec.shape:
                        actual_value = None
                    else:
                        actual_value = spec.shape[shape_field]
                    if actual_value != expected_value:
                        differences.append({
                            "field": f"kernel_shape.{shape_field}",
                            "expected": expected_value,
                            "actual": actual_value,
                        })
        if differences:
            mismatched.append({
                "node_id": node_id,
                "differences": tuple(differences),
            })
        else:
            matched.append(node_id)

    actual_edges = tuple(
        (edge.source, edge.destination, edge.kind) for edge in graph.edges)
    topology_mismatched = []
    if actual_edges != reference["edges"]:
        topology_mismatched.append({
            "field": "workload_edges",
            "expected": reference["edges"],
            "actual": actual_edges,
        })
    actual_order = graph.topological_order()
    if actual_order != reference["topological_order"]:
        topology_mismatched.append({
            "field": "workload_topological_order",
            "expected": reference["topological_order"],
            "actual": actual_order,
        })
    final_call_by_node = {
        node_id: calls[-1].call_id
        for node_id, calls in calls_by_node_id.items()
        if calls
    }
    dependency_mismatches = []
    for node_id in sorted(set(expected_by_id).intersection(actual_by_id)):
        calls = calls_by_node_id[node_id]
        if not calls:
            continue
        predecessors = expected_predecessors[node_id]
        if any(predecessor not in final_call_by_node
               for predecessor in predecessors):
            continue
        expected_external = tuple(
            final_call_by_node[predecessor] for predecessor in predecessors)
        if calls[0].dependencies != expected_external:
            dependency_mismatches.append({
                "call_id": calls[0].call_id,
                "expected": expected_external,
                "actual": calls[0].dependencies,
            })
        for previous, current in zip(calls, calls[1:]):
            if current.dependencies != (previous.call_id,):
                dependency_mismatches.append({
                    "call_id": current.call_id,
                    "expected": (previous.call_id,),
                    "actual": current.dependencies,
                })
    if dependency_mismatches:
        topology_mismatched.append({
            "field": "kernel_dependencies",
            "differences": tuple(dependency_mismatches),
        })
    failed = bool(missing or extra or mismatched or topology_mismatched)
    scope = reference["validation_scope"]
    if scope == "llama_bf16_tp2_vllm_v1":
        failed_status = "failed_llama_tp2_emission_alignment"
        validated_status = (
            "validated_llama_tp2_static_source_semantics_projected_latency")
    elif scope == "llama_bf16_af_tp2_vllm_v1":
        failed_status = "failed_llama_af_tp2_emission_alignment"
        validated_status = (
            "validated_llama_af_tp2_static_source_semantics_"
            "projected_latency")
    elif scope == "qwen3_bf16_tp2_vllm_v1":
        failed_status = "failed_tp2_emission_alignment"
        validated_status = (
            "validated_tp2_static_source_semantics_projected_latency")
    elif scope == "qwen3_bf16_pp2_vllm_v1":
        failed_status = "failed_pp2_emission_alignment"
        validated_status = (
            "validated_pp2_static_source_semantics_projected_latency")
    elif scope == "qwen3_bf16_pp3_vllm_v1":
        failed_status = "failed_pp3_emission_alignment"
        validated_status = (
            "validated_pp3_static_source_semantics_projected_latency")
    elif scope == "qwen3_bf16_uniform_pp_vllm_v1":
        failed_status = "failed_uniform_pp_emission_alignment"
        validated_status = (
            "validated_uniform_pp_static_source_semantics_projected_latency")
    elif scope == "qwen3_bf16_tp2_pp2_vllm_v1":
        failed_status = "failed_tp2_pp2_emission_alignment"
        validated_status = (
            "validated_tp2_pp2_static_source_semantics_projected_latency")
    elif scope == "qwen3_moe_bf16_tp2_attention_vllm_v1":
        failed_status = "failed_qwen3_moe_tp2_attention_alignment"
        validated_status = (
            "validated_qwen3_moe_tp2_attention_static_source_semantics_"
            "projected_latency")
    else:
        raise ValueError(
            f"unknown rank-local validation scope: {scope!r}")
    return {
        "validation_status": failed_status if failed else validated_status,
        "expected_node_count": len(expected_nodes),
        "actual_node_count": len(graph.nodes),
        "matched": tuple(matched),
        "missing": missing,
        "extra": extra,
        "mismatched": tuple(mismatched),
        "topology_mismatched": tuple(topology_mismatched),
        "node_coverage": (
            len(matched) / len(expected_nodes) if expected_nodes else 0.0),
        "unvalidated": (
            "binary_kernel_fusion",
            "launch_grid_block_and_tiling",
            "kernel_duration",
            "communication_duration",
            "pipeline_control_metadata_latency",
        ),
    }


def compare_reference_to_emission(
    reference: tuple[dict[str, object], ...],
    graph,
    plan,
) -> dict[str, object]:
    """Return explicit matched/missing/extra/mismatched diagnostics."""

    expected_by_id = {item["node_id"]: item for item in reference}
    actual_by_id = {node.node_id: node for node in graph.nodes}
    missing = tuple(sorted(set(expected_by_id) - set(actual_by_id)))
    extra = tuple(sorted(set(actual_by_id) - set(expected_by_id)))
    calls_by_node_id = {node_id: [] for node_id in actual_by_id}
    for call in plan.calls:
        for node_id in call.source_node_ids:
            if node_id not in calls_by_node_id:
                raise ValueError(
                    f"KernelCall references unknown node {node_id!r}")
            calls_by_node_id[node_id].append(call)
    matched = []
    mismatched = []
    semantic_field_count = sum(
        10 + len(item["kernel_shape"])
        + len(item["workload_attributes"])
        for item in reference
    )
    matched_semantic_field_count = 0
    for node_id in sorted(set(expected_by_id).intersection(actual_by_id)):
        expected = expected_by_id[node_id]
        node = actual_by_id[node_id]
        differences = []
        node_fields = (
            ("workload_kind", node.kind.value, expected["workload_kind"]),
            ("workload_variant", node.variant,
             expected["workload_variant"]),
            ("workload_shape", dict(node.shape),
             expected["workload_shape"]),
            ("workload_dtype", node.dtype, expected["workload_dtype"]),
            ("workload_pool_name", node.pool_name,
             expected["workload_pool_name"]),
            ("workload_rank", node.rank, expected["workload_rank"]),
        )
        for field, actual_value, expected_value in node_fields:
            if actual_value == expected_value:
                matched_semantic_field_count += 1
            else:
                differences.append({
                    "field": field,
                    "expected": expected_value,
                    "actual": actual_value,
                })
        for attribute, expected_value in expected[
                "workload_attributes"].items():
            if attribute not in node.attributes:
                differences.append({
                    "field": f"workload_attributes.{attribute}",
                    "expected": expected_value,
                    "actual": None,
                })
            elif node.attributes[attribute] == expected_value:
                matched_semantic_field_count += 1
            else:
                differences.append({
                    "field": f"workload_attributes.{attribute}",
                    "expected": expected_value,
                    "actual": node.attributes[attribute],
                })
        calls = calls_by_node_id[node_id]
        matching_calls = tuple(
            call for call in calls
            if (call.kernel_spec.kind.value == expected["kernel_kind"]
                and call.kernel_spec.variant == expected["kernel_variant"]
                and all(
                    call.kernel_spec.shape.get(name) == value
                    for name, value in expected["kernel_shape"].items()))
        )
        if len(matching_calls) == 1:
            matched_semantic_field_count += 1
            spec = matching_calls[0].kernel_spec
            kernel_fields = (
                ("kernel_kind", spec.kind.value, expected["kernel_kind"]),
                ("kernel_variant", spec.variant,
                 expected["kernel_variant"]),
                ("kernel_dtype", spec.dtype, "bf16"),
            )
            for field, actual_value, expected_value in kernel_fields:
                if actual_value == expected_value:
                    matched_semantic_field_count += 1
                else:
                    differences.append({
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    })
            for shape_field, expected_value in expected[
                    "kernel_shape"].items():
                if shape_field not in spec.shape:
                    differences.append({
                        "field": f"kernel_shape.{shape_field}",
                        "expected": expected_value,
                        "actual": None,
                    })
                elif spec.shape[shape_field] == expected_value:
                    matched_semantic_field_count += 1
                else:
                    differences.append({
                        "field": f"kernel_shape.{shape_field}",
                        "expected": expected_value,
                        "actual": spec.shape[shape_field],
                    })
        else:
            differences.append({
                "field": "principal_kernel_call_count",
                "expected": 1,
                "actual": len(matching_calls),
            })
        if differences:
            mismatched.append({
                "node_id": node_id,
                "differences": tuple(differences),
            })
        else:
            matched.append(node_id)
    status_counts = {}
    for item in reference:
        status = item["evidence_status"]
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1
    matched_status_counts = {}
    for node_id in matched:
        status = expected_by_id[node_id]["evidence_status"]
        if status in matched_status_counts:
            matched_status_counts[status] += 1
        else:
            matched_status_counts[status] = 1
    comparison_coverage = (
        matched_semantic_field_count / semantic_field_count
        if semantic_field_count else 0.0
    )
    if "validated_static_source_semantics" in matched_status_counts:
        validated_node_count = matched_status_counts[
            "validated_static_source_semantics"]
    else:
        validated_node_count = 0
    validated_node_coverage = (
        validated_node_count / len(reference) if reference else 0.0
    )
    expected_order = tuple(item["node_id"] for item in reference)
    actual_order = graph.topological_order()
    expected_edges = tuple(
        (source, destination, "data")
        for source, destination in zip(
            expected_order, expected_order[1:])
    )
    actual_edges = tuple(
        (edge.source, edge.destination, edge.kind)
        for edge in graph.edges
    )
    topology_mismatched = []
    for field, expected_value, actual_value in (
        ("workload_topological_order", expected_order, actual_order),
        ("workload_edges", expected_edges, actual_edges),
    ):
        if expected_value != actual_value:
            topology_mismatched.append({
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
            })
    final_call_by_node = {
        node_id: calls[-1].call_id
        for node_id, calls in calls_by_node_id.items()
        if calls
    }
    dependency_mismatches = []
    for index, node_id in enumerate(expected_order):
        calls = calls_by_node_id[node_id]
        if not calls:
            continue
        expected_external = (
            () if index == 0
            else (final_call_by_node[expected_order[index - 1]],)
        )
        if calls[0].dependencies != expected_external:
            dependency_mismatches.append({
                "call_id": calls[0].call_id,
                "expected": expected_external,
                "actual": calls[0].dependencies,
            })
        for previous, current in zip(calls, calls[1:]):
            if current.dependencies != (previous.call_id,):
                dependency_mismatches.append({
                    "call_id": current.call_id,
                    "expected": (previous.call_id,),
                    "actual": current.dependencies,
                })
    if dependency_mismatches:
        topology_mismatched.append({
            "field": "kernel_dependencies",
            "differences": tuple(dependency_mismatches),
        })
    has_alignment_failure = bool(
        missing or extra or mismatched or topology_mismatched)
    return {
        "validation_status": (
            "failed_emission_alignment"
            if has_alignment_failure else
            "validated_shapes_with_projected_attention_decomposition"),
        "expected_node_count": len(reference),
        "actual_node_count": len(graph.nodes),
        "matched": tuple(matched),
        "missing": missing,
        "extra": extra,
        "mismatched": tuple(mismatched),
        "topology_mismatched": tuple(topology_mismatched),
        "semantic_field_count": semantic_field_count,
        "matched_semantic_field_count": matched_semantic_field_count,
        "comparison_field_coverage": comparison_coverage,
        "validated_static_node_count": validated_node_count,
        "validated_static_node_coverage": validated_node_coverage,
        "evidence_status_counts": status_counts,
        "matched_evidence_status_counts": matched_status_counts,
        "unvalidated": (
            "binary_kernel_fusion",
            "attention_backend_runtime_selection",
            "launch_grid_block_and_tiling",
            "kernel_duration",
        ),
    }

def _gemm_workload_attributes(position, tensor_parallel_size,
                              expert_parallel_size=1):
    return {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": position,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "launch_policy": "analytical_launch",
    }
