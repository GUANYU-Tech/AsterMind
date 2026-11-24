"""Expose and validate the bundled execution capabilities used by both editors."""
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path

from paths import REPOSITORY, RUNTIME


def memory_policies():
    """Return named files the operator has made available to the web service."""
    directory = (RUNTIME / 'memory-policies').resolve()
    policies = {
        file.name: file.resolve()
        for file in sorted(directory.glob('*.json'))
        if file.is_file() and file.resolve().is_relative_to(directory)
    }
    example = REPOSITORY / 'tests/L2/deepseek-v4-h200-memory.json'
    if example.is_file():
        policies.setdefault('deepseek-v4-h200-memory.json', example.resolve())
    return policies


def resolve_memory_policy(value):
    if not value:
        return None
    policies = memory_policies()
    path = policies.get(value)
    if path is None:
        path = next((file for file in policies.values() if value == str(file)), None)
    if path is None:
        raise ValueError('请选择已登记的显存策略。将策略 JSON 放入 viz/.runtime/memory-policies 后刷新页面。')
    return str(path)


def require_fast_scheduler():
    python = os.environ.get('ORBITServe_VLLM030_PYTHON')
    source = os.environ.get('ORBITServe_VLLM030_SOURCE')
    if not python or not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError('Fast Mode 需要可执行的 vLLM 0.30 Python。请检查服务端 ORBITServe_VLLM030_PYTHON。')
    if not source or not (Path(source) / 'vllm' / '__init__.py').is_file():
        raise ValueError('Fast Mode 需要固定版本的 vLLM 0.30 源码。请检查服务端 ORBITServe_VLLM030_SOURCE。')


def compute_catalog(upstream):
    from astermind.builder.catalog import scene_catalog
    from astermind.engine.comm.registered_profiles import LINKS

    catalog = scene_catalog(upstream)
    catalog['memory_policies'] = [
        {'id': name, 'file': str(path)} for name, path in memory_policies().items()
    ]
    catalog['collective_algorithms'] = collective_options()
    for row in catalog['hardware']:
        row['su_technology'] = LINKS.get(row['id'])
    for model in catalog['models']:
        config = json.loads((upstream / model['id']).read_text(encoding='utf-8-sig'))
        source = config.get('text_config', config)
        quantization = source.get('quantization_config') or config.get('quantization_config') or {}
        model['quantization'] = quantization.get('quant_method')
    return catalog


def collective_options():
    from astermind.builder.config.communication import (
        ALGORITHMS, DEFAULT_ALGORITHMS, LAYER_ALGORITHMS, LAYERED_COLLECTIVES,
    )

    scopes = {scope: {} for scope in ('single_su', 'intra_cross_su', 'inter_satellite')}
    for scope, rules in scopes.items():
        for operation, default in DEFAULT_ALGORITHMS.items():
            choices = ALGORITHMS.get(operation, frozenset()) - {'hierarchical'}
            if scope != 'inter_satellite' and operation in LAYERED_COLLECTIVES:
                choices |= LAYER_ALGORITHMS
            if choices:
                rules[operation] = dict(choices=sorted(choices), default=default if default in choices else sorted(choices)[0])
    return scopes


def collective_policy(rows):
    options = collective_options()
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'scope', 'operation', 'algorithm'}:
            raise ValueError('集合通信规则需要范围、操作和算法')
        scope, op, algorithm = row['scope'], row['operation'], row['algorithm']
        if algorithm not in options.get(scope, {}).get(op, {}).get('choices', ()):
            raise ValueError(f'集合通信不支持 {scope} / {op} / {algorithm}')
        if op in result.setdefault(scope, {}):
            raise ValueError(f'集合通信范围和操作重复：{scope} / {op}')
        result[scope][op] = algorithm
    return result


def su_groups(raw, cards, limit):
    if not isinstance(raw, (list, tuple)) or any(type(n) is not int or not 1 <= n <= limit for n in raw):
        raise ValueError(f'每个 SU 组需要 1 到 {limit} 张卡')
    if raw and sum(raw) != cards:
        raise ValueError('SU 分组卡数之和必须等于每星 GPU 数')
    return tuple(raw) or tuple(min(limit, cards - start) for start in range(0, cards, limit))


def validate_fast_su_placement(hardware_id, devices, hierarchies):
    from astermind.common.hardware_capabilities import PORTABLE_BATCH_HARDWARE

    if hardware_id not in PORTABLE_BATCH_HARDWARE:
        return
    ranks = set(devices)
    for owner, hierarchy in hierarchies.items():
        occupied = sum(not ranks.isdisjoint(group) for group in hierarchy['su_groups'])
        if occupied > 1:
            raise ValueError(f'{hardware_id} Fast Mode 要求 {owner} 的计算 Rank 位于同一 SU 域')


def scheduler_adapter(mode, model, dp, *, pd=False):
    if pd:
        return 'vllm_0_11'
    return 'vllm_0_30' if mode == 'fast_mode' or dp > 1 or model.startswith(('glm-', 'deepseek-', 'kimi-')) else 'auto'


def resolve_scheduling(options, model, longest, upstream):
    """Resolve an editor's automatic fields while retaining explicit values."""
    context = options.get('max_model_len')
    if context is None:
        context = min(model['max_context'], 131072, max(128, longest))
    if context > model['max_context'] or longest > context:
        raise ValueError(f'最长请求需要 {longest} Token，调度上限 {context} Token，模型上限 {model["max_context"]} Token')
    batch_tokens = options.get('max_batched_tokens')
    if batch_tokens is None:
        batch_tokens = context * options['max_requests']
    mode = options['compute_mode']
    adapter = options['scheduler_adapter']
    from astermind.builder.catalog import fast_kv_block_size
    from types import SimpleNamespace

    block = options.get('kv_block_size_tokens') or 16
    if mode == 'fast_mode' or Path(options['model']).stem.startswith('deepseek-v4-'):
        block = fast_kv_block_size(SimpleNamespace(model=options['model'], max_model_len=context))
    count = options.get('num_kv_blocks')
    if count is None and adapter != 'vllm_0_30':
        count = hbm_kv_block_count(options, block, upstream)
        if (count - 1) * block < context:
            raise ValueError(f'显存不足，KV 容量无法容纳 {context} Token 的上下文')
    return dict(max_model_len=context, max_batched_tokens=batch_tokens,
                kv_block_size_tokens=block, num_kv_blocks=count or 64)


def prepare_deployment(options, upstream, *, largest_su=None, native_detail=False):
    from astermind.builder.catalog import validate_deployment_choice
    from astermind.builder.config.deployment import SceneDeployment

    deployment = SceneDeployment.model_validate(options)
    if deployment.compute_mode == 'fast_mode':
        domain = deployment if largest_su is None else deployment.model_copy(update={'gpus_per_satellite': largest_su})
        hardware = validate_deployment_choice(domain, upstream)
    elif native_detail:
        catalog = compute_catalog(upstream)
        if not any(row['id'] == deployment.model for row in catalog['models']):
            raise ValueError('请选择后端目录中的模型')
        hardware = next((row for row in catalog['hardware']
                         if (row['id'], row['file']) == (deployment.hardware_id, deployment.hardware)), None)
        if hardware is None:
            raise ValueError('硬件档案与所选 GPU 型号不一致')
        _validate_detail_options(deployment.model_dump_json(exclude={'communication'}), str(upstream))
    else:
        hardware = validate_deployment_choice(deployment, upstream)
    if deployment.long_prefill_token_threshold and deployment.enable_chunked_prefill is not True:
        raise ValueError('长 Prefill 阈值需要开启分块 Prefill')
    if deployment.compute_mode == 'fast_mode':
        # The cache key includes the policy content. Server operators can replace
        # a policy between validation and submission without reusing a stale check.
        policy = resolve_memory_policy(deployment.memory_policy_file)
        deployment = deployment.model_copy(update={'memory_policy_file': policy})
        validation = deployment.model_dump_json(exclude={'communication'})
        fingerprint = sha256(Path(policy).read_bytes()).hexdigest() if policy else None
        _validate_fast_options(validation, str(upstream), fingerprint)
    elif deployment.memory_policy_file:
        raise ValueError('此场景表单的显存策略用于 Fast Mode')
    return deployment, hardware


@lru_cache(maxsize=128)
def _validate_detail_options(options, upstream):
    from astermind.builder.config.deployment import SceneDeployment
    from astermind.engine.compute.runtime import MidlayerCellConfig, create_midlayer_runtime

    value = SceneDeployment.model_validate_json(options)
    try:
        create_midlayer_runtime(MidlayerCellConfig(
            Path(upstream), model_file=value.model, hardware_file=value.hardware, hardware_id=value.hardware_id,
            tensor_parallel_size=value.tp, pipeline_parallel_size=value.pp, data_parallel_size=value.dp,
            enable_expert_parallel=value.expert_parallel, expert_routing_seed=value.expert_routing_seed,
            compute_mode=value.executor_compute_mode, scheduler_adapter=value.scheduler_adapter,
            max_model_len=value.max_model_len, max_batched_tokens=value.max_batched_tokens, max_requests=value.max_requests,
            num_kv_blocks=value.num_kv_blocks, kv_block_size_tokens=value.kv_block_size_tokens,
            enable_prefix_caching=value.prefix_caching, enable_chunked_prefill=value.enable_chunked_prefill,
            long_prefill_token_threshold=value.long_prefill_token_threshold,
        ), with_scheduler=False)
    except NotImplementedError as error:
        raise ValueError(f'Detail Mode 不支持当前配置：{error}') from error


@lru_cache(maxsize=128)
def _validate_fast_options(options, upstream, policy_fingerprint):
    from astermind.builder.catalog import validate_fast_layout
    from astermind.builder.config.deployment import SceneDeployment

    try:
        validate_fast_layout(SceneDeployment.model_validate_json(options), Path(upstream))
    except NotImplementedError as error:
        raise ValueError(f'Fast Mode 不支持当前配置：{error}') from error


@lru_cache(maxsize=64)
def detail_prediction_method(model_file, hardware_id, upstream):
    from src.kernel.hardware_spec import HWSpec
    from src.serving.workload.model_config import load_model_config

    model = load_model_config(upstream / model_file)
    # The source support registry identifies the model/card pairs with a Roofline projection.
    contract = {
        'deepseek_v3': 'deepseek_v3_portable_roofline',
        'deepseek_v4': 'deepseek_v4_portable_roofline',
        'kimi_k2': 'kimi_k2_portable_roofline',
        'kimi_linear': 'kimi_k3_portable_roofline',
    }.get(model.model_type)
    if model.model_type == 'glm_moe_dsa':
        contract = {
            'bf16': 'glm_bf16_portable_roofline',
            'fp8': 'glm_fp8_weight_only_roofline',
        }.get(model.weight_dtype)
    return 'roofline' if contract and HWSpec.source_contract_supported(contract, hardware_id) else 'midlayer'


def hbm_kv_block_count(options, block_tokens, upstream):
    from src.kernel.hardware_spec import HWSpec
    from src.serving.kv_cache.model_residency import model_rank_residencies
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.workload.model_config import load_model_config

    model = load_model_config(upstream / options['model'])
    hbm_bytes = HWSpec(upstream / options['hardware']).resource_by_role('dram').size_bytes
    # Match the standard native HBM allowance for models without a dedicated policy.
    usable_bytes = int(hbm_bytes * .9) - 2 * 1024**3
    residencies = model_rank_residencies(model, ParallelConfig(
        tensor_parallel_size=options['tp'], pipeline_parallel_size=options['pp'],
        enable_expert_parallel=options['expert_parallel']), worker_role='colocated')
    counts = []
    for rank in residencies:
        block_bytes = block_tokens * model.kv_bytes_per_token_for_layer_range(
            options['tp'], rank.layer_start, rank.layer_end)
        counts.append((usable_bytes - rank.weight_resident_bytes) // block_bytes)
    count = min(counts)
    if count < 2:
        raise ValueError('显存不足：扣除权重和运行预留后，剩余显存无法容纳 KV 缓存池')
    return count
