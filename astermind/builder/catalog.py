"""Scene options supported by the loaded native Cell adapter, not file count."""

import json
import math
from pathlib import Path

import yaml

HARDWARE = {
    "nvidia_a800": "NVIDIA A800",
    "nvidia_h200": "NVIDIA H200",
    "ascend_910b": "Ascend 910B",
    "metax_c500": "MetaX C500",
}
MODEL_NAMES = {
    "glm-5.2": "GLM-5.2",
    "qwen3-4b": "Qwen3 4B",
    "qwen3-8b": "Qwen3 8B",
    "qwen3-32b": "Qwen3 32B",
    "qwen3-32b-fp16": "Qwen3 32B FP16",
    "qwen3-30b-a3b": "Qwen3 30B A3B",
    "llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
}
COMPUTE_MODES = (
    {"id": "midlayer", "name": "MidLayer（原生 KernelPlan）"},
    {"id": "roofline", "name": "Roofline（原生 KernelPlan）"},
    {"id": "fast_mode", "name": "Fast Mode"},
)
# Models the form can launch without a separately supplied weight projection.
FAST_EDITOR_MODELS = frozenset(MODEL_NAMES)
FAST_EDITOR_GPU_LIMITS = {"nvidia_h200": 8, "nvidia_a800": 8, "ascend_910b": 8, "metax_c500": 4}


def cell_member_counts(dataset):
    value = json.loads(
        (Path(dataset) / "materialized_input.json").read_text(encoding="utf-8")
    )
    groups = {}
    for node in value["network"]["nodes"]:
        if node["role"] == "compute":
            identity = node.get("cell_id") or node["node_id"]
            groups[identity] = groups.get(identity, 0) + 1
    if not groups:
        raise ValueError("所选物理数据没有计算 Cell")
    return groups


def parallel_choices(model, hardware_id):
    """Match this adapter's BF16 colocated native rank-emission contract."""
    if hardware_id not in HARDWARE or model.get("torch_dtype") != "bfloat16":
        return []
    kind = model.get("model_type")
    if kind not in {"qwen3", "qwen3_moe", "llama"}:
        return []
    if kind == "qwen3_moe":
        return (
            [dict(tp=2, pp=1, expert_parallel=True)]
            if hardware_id == "nvidia_a800"
            else []
        )
    result = [dict(tp=1, pp=1, expert_parallel=False)]
    if hardware_id == "nvidia_a800" and kind == "qwen3":
        result.append(dict(tp=2, pp=1, expert_parallel=False))
        layers = model.get("num_hidden_layers", 0)
        if not model.get("tie_word_embeddings", False) and layers % 2 == 0:
            result.append(dict(tp=2, pp=2, expert_parallel=False))
        if not model.get("tie_word_embeddings", False):
            result.extend(
                dict(tp=1, pp=pp, expert_parallel=False)
                for pp in range(2, layers + 1)
                if layers % pp == 0
            )
    return result


def scene_catalog(root=None):
    if root is None:
        from astermind.common.bootstrap import activate

        root = activate()
    root = Path(root)
    hardware = [
        dict(id=identity, name=name, file=f"configs/hardware/{identity}.yaml")
        for identity, name in HARDWARE.items()
        if (root / f"configs/hardware/{identity}.yaml").is_file()
    ]
    for row in hardware:
        profile = (
            yaml.load(
                (root / row["file"]).read_text(encoding="utf-8"),
                Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
            )
            or {}
        )
        link = profile.get("interconnect", {})
        bandwidth = link.get("nvlink_unidir_bw_GB_s")
        row["gpu_link_bandwidth_gbps"] = (
            bandwidth * 8
            if isinstance(bandwidth, (int, float))
            and math.isfinite(bandwidth)
            and bandwidth > 0
            else None
        )
        row["gpu_link_latency_ns"] = link.get("nvlink_signal_latency_ns") or 0
        row["fast_gpu_limit_per_satellite"] = FAST_EDITOR_GPU_LIMITS[row["id"]]
    models = []
    for source in sorted((root / "configs/model").glob("*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        dtype = data.get("dtype") or data.get("torch_dtype")
        choices = {row["id"]: parallel_choices(data, row["id"]) for row in hardware}
        available = any(choices.values())
        reason = (
            ""
            if available
            else (
                "当前原生调度器尚未接入此精度"
                if dtype != "bfloat16"
                else "当前 Cell 运行器尚未接入此模型架构"
            )
        )
        models.append(
            dict(
                id=f"configs/model/{source.name}",
                name=MODEL_NAMES.get(source.stem, source.stem),
                dtype=dtype,
                available=available,
                reason=reason,
                max_context=data.get("max_position_embeddings", 131072),
                experts=data.get("num_experts", data.get("n_routed_experts", 0)),
                choices=choices,
                fast_hardware=[row["id"] for row in hardware] if source.stem in FAST_EDITOR_MODELS else [],
            )
        )
    return dict(
        models=models,
        hardware=hardware,
        compute_modes=COMPUTE_MODES,
        defaults=dict(
            model="configs/model/qwen3-8b.json", hardware="nvidia_a800", tp=2, pp=2
        ),
        other_axes=dict(dp=False, sp=False, cp=False),  # Detailed mode selectors.
        source_model_count=len(models),
        source_hardware_count=len(list((root / "configs/hardware").glob("*.yaml"))),
    )


def validate_deployment_choice(deployment, root=None):
    catalog = scene_catalog(root)
    model = next(
        (row for row in catalog["models"] if row["id"] == deployment.model), None
    )
    hardware = next(
        (row for row in catalog["hardware"] if row["id"] == deployment.hardware_id),
        None,
    )
    if model is None:
        raise ValueError("所选模型档案不存在")
    if hardware is None or deployment.hardware != hardware["file"]:
        raise ValueError("硬件档案与所选 GPU 型号不一致")
    choice = dict(
        tp=deployment.tp, pp=deployment.pp, expert_parallel=deployment.expert_parallel
    )
    batch = deployment.compute_mode == "fast_mode"
    if batch:
        if hardware["id"] not in model["fast_hardware"]:
            raise ValueError("该模型需要通过高级 JSON 配置其 Fast Mode 来源和容量")
        if deployment.gpus_per_satellite > hardware["fast_gpu_limit_per_satellite"]:
            raise ValueError(f"此编辑器的 {hardware['name']} 单高速域最多配置 {hardware['fast_gpu_limit_per_satellite']} 张卡")
        if deployment.tp * deployment.pp * deployment.dp > 64:
            raise ValueError("场景编辑器的 Fast Mode 已验收至每 Cell 64 卡，更大部署请使用高级 JSON")
        if deployment.wide_ep_enabled != (deployment.dp > 1) or (deployment.dp > 1 and not deployment.expert_parallel):
            raise ValueError("DP 大于 1 需要显式开启专家并行和 WideEP")
        if deployment.max_batched_tokens < deployment.max_model_len:
            raise ValueError("此场景的 vLLM 0.30 未启用分块预填充，单批 token 上限须不小于上下文上限")
    elif deployment.dp != 1 or deployment.wide_ep_enabled or deployment.fast_glm_dsa_score is not None or choice not in model["choices"].get(hardware["id"], []):
        raise ValueError("当前计算模式的场景预设未提供此模型、硬件与 TP / PP / DP / EP 组合")
    if deployment.max_model_len > model["max_context"]:
        raise ValueError(f"上下文长度不能超过模型上限 {model['max_context']}")
    if (deployment.expert_parallel or batch and model["experts"]) and deployment.expert_routing_seed is None:
        raise ValueError("MoE 专家路由需要显式随机种子")
    if not model["experts"] and deployment.expert_routing_seed is not None:
        raise ValueError("稠密模型不使用专家路由种子")
    return hardware


def validate_fast_layout(deployment, root):
    """Run the native plan and formula admission without starting a Scheduler."""
    from astermind.engine.compute.runtime import MidlayerCellConfig, create_midlayer_runtime
    from astermind.engine.compute.fast_mode.hardware import Hardware
    from astermind.engine.compute.fast_mode.predictor import Predictor
    from astermind.engine.compute.factory import standard_vllm030_budgets
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model

    config = MidlayerCellConfig(
        Path(root), model_file=deployment.model, hardware_file=deployment.hardware,
        hardware_id=deployment.hardware_id, scheduler_adapter="vllm_0_30", compute_mode="fast_mode",
        tensor_parallel_size=deployment.tp, pipeline_parallel_size=deployment.pp, data_parallel_size=deployment.dp,
        enable_expert_parallel=deployment.expert_parallel, expert_routing_seed=deployment.expert_routing_seed,
        fast_glm_dsa_score=deployment.fast_glm_dsa_score,
        max_model_len=deployment.max_model_len, max_batched_tokens=deployment.max_batched_tokens,
        max_requests=deployment.max_requests,
    )
    _, engine, _ = create_midlayer_runtime(config, with_scheduler=False)
    model = engine.model
    hardware = engine.hardware_by_id[config.hardware_id]
    ranges = pipeline_layer_ranges_for_model(model, deployment.pp)
    parallel = engine.deployment.worker_pools["main"].parallel_plan.config
    if min(standard_vllm030_budgets(model, hardware, parallel, tuple(b-a for a, b in ranges))) <= 0:
        raise ValueError("权重与预留超过本部署的逐 rank HBM 预算")
    profile = Hardware.from_hardware_spec(hardware, compute_dtype=model.compute_dtype)
    raw = json.loads((config.root / config.model_file).read_text(encoding="utf-8-sig"))
    Predictor(Path(config.model_file).stem, raw, deployment.tp, deployment.dp,
              parallel.expert_parallel_size, ranges, hardware=profile,
              glm_dsa_bf16_timing=config.fast_glm_dsa_score == "bf16_timing_proxy")
