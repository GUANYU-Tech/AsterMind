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
    "qwen3-4b": "Qwen3 4B",
    "qwen3-8b": "Qwen3 8B",
    "qwen3-32b": "Qwen3 32B",
    "qwen3-30b-a3b": "Qwen3 30B A3B",
    "llama-3.1-8b-instruct": "Llama 3.1 8B Instruct",
}


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
    models = []
    for source in sorted((root / "configs/model").glob("*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        choices = {row["id"]: parallel_choices(data, row["id"]) for row in hardware}
        available = any(choices.values())
        reason = (
            ""
            if available
            else (
                "当前原生调度器尚未接入此精度"
                if data.get("torch_dtype") != "bfloat16"
                else "当前 Cell 运行器尚未接入此模型架构"
            )
        )
        models.append(
            dict(
                id=f"configs/model/{source.name}",
                name=MODEL_NAMES.get(source.stem, source.stem),
                dtype=data.get("torch_dtype"),
                available=available,
                reason=reason,
                max_context=data.get("max_position_embeddings", 131072),
                experts=data.get("num_experts", data.get("n_routed_experts", 0)),
                choices=choices,
            )
        )
    return dict(
        models=models,
        hardware=hardware,
        defaults=dict(
            model="configs/model/qwen3-8b.json", hardware="nvidia_a800", tp=2, pp=2
        ),
        other_axes=dict(dp=False, sp=False, cp=False),
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
    if model is None or not model["available"]:
        raise ValueError("所选模型尚未接入当前 Cell 运行器")
    if hardware is None or deployment.hardware != hardware["file"]:
        raise ValueError("硬件档案与所选 GPU 型号不一致")
    choice = dict(
        tp=deployment.tp, pp=deployment.pp, expert_parallel=deployment.expert_parallel
    )
    if choice not in model["choices"].get(hardware["id"], []):
        raise ValueError("当前模型、硬件与 TP / PP / EP 组合不受原生 Cell 运行器支持")
    if deployment.max_model_len > model["max_context"]:
        raise ValueError(f"上下文长度不能超过模型上限 {model['max_context']}")
    if deployment.expert_parallel and deployment.expert_routing_seed is None:
        raise ValueError("MoE 专家路由需要显式随机种子")
    if not deployment.expert_parallel and deployment.expert_routing_seed is not None:
        raise ValueError("稠密模型不使用专家路由种子")
    return hardware
