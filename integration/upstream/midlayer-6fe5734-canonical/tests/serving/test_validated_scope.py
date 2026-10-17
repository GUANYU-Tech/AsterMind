"""Pin the validated target-model scope table and its three gate wirings.

``src.serving.deployment.validated_scope`` is the single source for the
validated target-model deployment topologies; the workload construction gate,
the workload/kernel emission gate, and the vLLM Scheduler scope gate all query
it.  These tests pin the truth table the gates share -- including the
historical granularity drift the table records but does not adjudicate:

* the construction gate carries no hardware pin on the A800 production and
  PD EP8 rows and never inspects the compute dtype;
* the emission gate alone pins the pool size, requires BF16 compute, gates
  the A800 production rows behind the allow flag, and derives the
  DeepSeek-V3 aggregate TP from the config identity (32-layer is TP4-only
  at emission while construction accepts it at TP8);
* the Scheduler gate knows roles instead of deployments or hardware, pins
  the named config identities, and deliberately over-approximates (the
  PP2/PP4 rows accept every named target model in any role, and the PD EP8
  row accepts every named target model).
"""

import dataclasses

import pytest

from src.serving.backend.vllm import (
    VLLMSchedulerConfig,
    _validate_ep_scheduler_model_scope,
    _validate_native_scheduler_scope,
)
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.deployment.validated_scope import (
    TARGET_MODEL_PRODUCTION_PARALLELISM,
    is_validated_target_construction_scope,
    is_validated_target_scheduler_scope,
    validated_target_emission_scope,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
from src.serving.simulation import validate_workload_emission_parallelism
from src.serving.workload.build_workload import build_workload
from src.serving.workload.model_config import (
    TARGET_MODEL_CONFIG_SHA256,
    TARGET_MODEL_CONFIG_SHA256_BY_TYPE,
    is_named_target_model_identity,
    load_model_config,
)
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs


def _kimi():
    return load_model_config("tests/serving/fixtures/kimi_k2_6_official")


def _glm():
    return load_model_config("tests/serving/fixtures/glm_5_2_official")


def _glm_fp8():
    return load_model_config("configs/model/glm-5.2-fp8.json")


def _dsv3():
    return load_model_config("configs/model/deepseek-v3.json")


def _dsv3_l32():
    return load_model_config("configs/model/deepseek-v3-l32.json")


def _qwen3():
    return load_model_config("tests/serving/fixtures/qwen3_scheduler")


_MODEL_LOADERS = {
    "kimi": _kimi,
    "glm": _glm,
    "glm_fp8": _glm_fp8,
    "dsv3": _dsv3,
    "dsv3_l32": _dsv3_l32,
    "qwen3": _qwen3,
}


def test_target_config_sha_sets_match_loaded_model_identities():
    kimi, glm, glm_fp8 = _kimi(), _glm(), _glm_fp8()
    dsv3, dsv3_l32 = _dsv3(), _dsv3_l32()
    assert TARGET_MODEL_CONFIG_SHA256 == frozenset((
        kimi.config_sha256,
        glm.config_sha256,
        glm_fp8.config_sha256,
        dsv3.config_sha256,
        dsv3_l32.config_sha256,
    ))
    assert TARGET_MODEL_CONFIG_SHA256_BY_TYPE == {
        "kimi_k2": (kimi.config_sha256,),
        "glm_moe_dsa": (glm.config_sha256, glm_fp8.config_sha256),
        "deepseek_v3": (dsv3.config_sha256, dsv3_l32.config_sha256),
    }
    assert TARGET_MODEL_PRODUCTION_PARALLELISM == {
        "kimi_k2": (8, 2),
        "glm_moe_dsa": (8, 4),
    }


def test_named_target_identity_is_flat_across_target_models():
    kimi, glm, dsv3, dsv3_l32 = _kimi(), _glm(), _dsv3(), _dsv3_l32()
    assert is_named_target_model_identity("kimi_k2", kimi.config_sha256)
    assert is_named_target_model_identity("glm_moe_dsa", glm.config_sha256)
    assert is_named_target_model_identity(
        "glm_moe_dsa", _glm_fp8().config_sha256)
    assert is_named_target_model_identity("deepseek_v3", dsv3.config_sha256)
    assert is_named_target_model_identity(
        "deepseek_v3", dsv3_l32.config_sha256)
    # Deliberately flat: the Scheduler's historical gate accepts any pinned
    # config identity with any target model_type.
    assert is_named_target_model_identity("kimi_k2", glm.config_sha256)
    assert is_named_target_model_identity("glm_moe_dsa", dsv3.config_sha256)
    assert not is_named_target_model_identity("kimi_k2", "0" * 64)
    assert not is_named_target_model_identity("qwen3", kimi.config_sha256)
    assert not is_named_target_model_identity(
        "qwen3", _qwen3().config_sha256)


@pytest.mark.parametrize(
    "model_key,mode,hardware_id,tp,pp,expert_parallel,expert_parallel_size,"
    "accepted",
    (
        # The TP=PP=EP=1 fixture row accepts every target model on any
        # hardware in either mode; non-target models have no row.
        ("kimi", "colocated", "nvidia_a800", 1, 1, False, 1, True),
        ("dsv3", "pd_disagg", "ascend_910b", 1, 1, False, 1, True),
        ("qwen3", "colocated", "nvidia_a800", 1, 1, False, 1, False),
        # The A800 colocated production rows: Kimi PP2, GLM PP4, both EP8.
        ("kimi", "colocated", "nvidia_a800", 8, 2, True, 8, True),
        ("glm", "colocated", "nvidia_a800", 8, 4, True, 8, True),
        # Drift: these rows predate the hardware pin, so construction accepts
        # the same topology on other hardware while emission rejects it.
        ("kimi", "colocated", "nvidia_h200", 8, 2, True, 8, True),
        ("kimi", "colocated", "ascend_910b", 8, 2, True, 8, True),
        ("glm", "colocated", "nvidia_h200", 8, 4, True, 8, True),
        # The per-model production rows are not interchangeable.
        ("kimi", "colocated", "nvidia_a800", 8, 4, True, 8, False),
        ("glm", "colocated", "nvidia_a800", 8, 2, True, 8, False),
        ("kimi", "pd_disagg", "nvidia_a800", 8, 2, True, 8, False),
        # The PD TP8/PP1/EP8 row covers Kimi and GLM on either hardware --
        # and, as drift, construction does not pin that hardware either.
        ("kimi", "pd_disagg", "nvidia_h200", 8, 1, True, 8, True),
        ("glm", "pd_disagg", "ascend_910b", 8, 1, True, 8, True),
        # DeepSeek-V3 has no EP8 production row at all.
        ("dsv3", "pd_disagg", "nvidia_a800", 8, 1, True, 8, False),
        ("dsv3", "colocated", "nvidia_h200", 8, 1, True, 8, False),
        # The H200 EP1 rows pin their hardware.
        ("glm", "colocated", "nvidia_h200", 8, 1, False, 1, True),
        ("glm", "colocated", "nvidia_a800", 8, 1, False, 1, False),
        ("glm_fp8", "pd_disagg", "nvidia_h200", 8, 1, False, 1, True),
        ("kimi", "colocated", "nvidia_h200", 8, 1, True, 8, True),
        ("kimi", "colocated", "nvidia_a800", 8, 1, True, 8, False),
        ("dsv3", "colocated", "nvidia_h200", 8, 1, False, 1, True),
        ("dsv3", "pd_disagg", "nvidia_h200", 8, 1, False, 1, True),
        # Drift: construction accepts the 32-layer DeepSeek-V3 identity at
        # TP8; emission is TP4-only for it.
        ("dsv3_l32", "colocated", "nvidia_h200", 8, 1, False, 1, True),
        ("dsv3_l32", "colocated", "nvidia_h200", 4, 1, False, 1, True),
        ("dsv3", "colocated", "nvidia_h200", 4, 1, False, 1, False),
        ("dsv3_l32", "colocated", "nvidia_a800", 4, 1, False, 1, False),
    ),
)
def test_construction_scope_truth_table(
        model_key, mode, hardware_id, tp, pp, expert_parallel,
        expert_parallel_size, accepted):
    model = _MODEL_LOADERS[model_key]()
    assert is_validated_target_construction_scope(
        model,
        deployment_mode=mode,
        hardware_id=hardware_id,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel=expert_parallel,
        expert_parallel_size=expert_parallel_size,
    ) is accepted


@pytest.mark.parametrize(
    "model_key,mode,hardware_id,single_pool,pool_ranks,workers,rank_local,"
    "allow,tp,pp,expert_parallel,expert_parallel_size,expected_row",
    (
        # pool_ranks/workers None mean the matched TP*PP pool.
        ("kimi", "colocated", "nvidia_a800", True, None, None, True, True,
         8, 2, True, 8, "kimi_a800_colocated_tp8_pp2_ep8"),
        # The A800 production rows sit behind the caller's allow flag.
        ("kimi", "colocated", "nvidia_a800", True, None, None, True, False,
         8, 2, True, 8, None),
        # Drift: emission pins the hardware the construction gate ignores.
        ("kimi", "colocated", "nvidia_h200", True, None, None, True, True,
         8, 2, True, 8, None),
        # The pool must hold exactly TP*PP ranks and workers.
        ("kimi", "colocated", "nvidia_a800", True, 8, 8, True, True,
         8, 2, True, 8, None),
        # Emission requires rank-local plans.
        ("kimi", "colocated", "nvidia_a800", True, None, None, False, True,
         8, 2, True, 8, None),
        # Colocated rows are single-pool; PD rows are not.
        ("kimi", "colocated", "nvidia_a800", False, None, None, True, True,
         8, 2, True, 8, None),
        ("glm", "colocated", "nvidia_a800", True, None, None, True, True,
         8, 4, True, 8, "glm_a800_colocated_tp8_pp4_ep8"),
        ("kimi", "pd_disagg", "nvidia_h200", False, None, None, True, False,
         8, 1, True, 8, "target_pd_tp8_pp1_ep8"),
        ("glm", "pd_disagg", "nvidia_a800", False, None, None, True, False,
         8, 1, True, 8, "target_pd_tp8_pp1_ep8"),
        # DeepSeek-V3 has no EP8 emission row.
        ("dsv3", "pd_disagg", "nvidia_h200", False, None, None, True, False,
         8, 1, True, 8, None),
        ("glm", "colocated", "nvidia_h200", True, None, None, True, False,
         8, 1, False, 1, "glm_h200_colocated_tp8_pp1_ep1"),
        ("glm_fp8", "pd_disagg", "nvidia_h200", False, None, None, True,
         False, 8, 1, False, 1, "glm_h200_pd_tp8_pp1_ep1"),
        ("kimi", "colocated", "nvidia_h200", True, None, None, True, False,
         8, 1, True, 8, "kimi_h200_colocated_tp8_pp1_ep8"),
        ("kimi", "colocated", "nvidia_a800", True, None, None, True, False,
         8, 1, True, 8, None),
        ("dsv3", "colocated", "nvidia_h200", True, None, None, True, False,
         8, 1, False, 1, "dsv3_h200_colocated_tp8_pp1_ep1"),
        # Drift: the 32-layer identity is excluded from the TP8 aggregate
        # row at emission, so it runs TP4-only there.
        ("dsv3_l32", "colocated", "nvidia_h200", True, None, None, True,
         False, 8, 1, False, 1, None),
        ("dsv3_l32", "colocated", "nvidia_h200", True, None, None, True,
         False, 4, 1, False, 1, "dsv3_h200_colocated_tp4_pp1_ep1_l32"),
        ("dsv3", "colocated", "nvidia_h200", True, None, None, True, False,
         4, 1, False, 1, None),
        ("dsv3", "pd_disagg", "nvidia_h200", False, None, None, True, False,
         8, 1, False, 1, "dsv3_h200_pd_tp8_pp1_ep1"),
        # Non-target models have no emission row; their slices stay with the
        # caller's own branches.
        ("qwen3", "colocated", "nvidia_a800", True, None, None, True, True,
         2, 1, False, 1, None),
    ),
)
def test_emission_scope_truth_table(
        model_key, mode, hardware_id, single_pool, pool_ranks, workers,
        rank_local, allow, tp, pp, expert_parallel, expert_parallel_size,
        expected_row):
    model = _MODEL_LOADERS[model_key]()
    row = validated_target_emission_scope(
        model,
        deployment_mode=mode,
        hardware_id=hardware_id,
        single_pool=single_pool,
        pool_rank_count=tp * pp if pool_ranks is None else pool_ranks,
        worker_count=tp * pp if workers is None else workers,
        is_rank_local=rank_local,
        allow_production_pipeline=allow,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel=expert_parallel,
        expert_parallel_size=expert_parallel_size,
    )
    assert (row.name if row is not None else None) == expected_row


def test_emission_scope_requires_bf16_compute():
    model = dataclasses.replace(_dsv3(), compute_dtype="fp8")
    assert validated_target_emission_scope(
        model,
        deployment_mode="colocated",
        hardware_id="nvidia_h200",
        single_pool=True,
        pool_rank_count=8,
        worker_count=8,
        is_rank_local=True,
        allow_production_pipeline=False,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
    ) is None


@pytest.mark.parametrize(
    "model_key,role,tp,pp,expert_parallel,accepted",
    (
        # The PP2/PP4 rows accept every named target model in any role.
        ("kimi", "colocated", 8, 2, True, True),
        ("kimi", "prefill", 8, 2, True, True),
        ("kimi", "attention", 8, 4, True, True),
        ("glm", "decode", 8, 4, True, True),
        # Over-approximation, pinned: no validated DeepSeek-V3 PP2/EP8
        # deployment exists, yet the Scheduler row accepts the named model.
        ("dsv3", "colocated", 8, 2, True, True),
        ("kimi", "prefill", 8, 1, True, True),
        # Over-approximation, pinned: the PD EP8 row names every target
        # model, DeepSeek-V3 included.
        ("dsv3", "decode", 8, 1, True, True),
        ("dsv3", "colocated", 8, 1, True, False),
        ("glm", "colocated", 8, 1, True, False),
        ("kimi", "colocated", 8, 1, True, True),
        ("kimi", "prefill", 8, 1, False, False),
        ("glm", "colocated", 8, 1, False, True),
        ("glm", "prefill", 8, 1, False, True),
        ("glm", "attention", 8, 1, False, False),
        ("dsv3", "colocated", 8, 1, False, True),
        ("dsv3", "decode", 8, 1, False, True),
        ("dsv3_l32", "colocated", 4, 1, False, True),
        ("dsv3_l32", "prefill", 4, 1, False, False),
        # TP4 is pinned to the 32-layer identity.
        ("dsv3", "colocated", 4, 1, False, False),
        # The named-identity pin: a drifted config digest closes every row,
        # while the flat named set accepts cross-model identities.
        ("kimi_drift", "colocated", 8, 2, True, False),
        ("kimi_glm_sha", "colocated", 8, 2, True, True),
        ("qwen3", "colocated", 8, 2, True, False),
        # The target scope starts at TP4; smaller plans take other gates.
        ("kimi", "colocated", 2, 1, True, False),
    ),
)
def test_scheduler_scope_truth_table(
        model_key, role, tp, pp, expert_parallel, accepted):
    model = _MODEL_LOADERS[model_key]() if model_key in _MODEL_LOADERS else None
    if model_key == "kimi_drift":
        model = dataclasses.replace(_kimi(), config_sha256="0" * 64)
    elif model_key == "kimi_glm_sha":
        model = dataclasses.replace(
            _kimi(), config_sha256=_glm().config_sha256)
    assert is_validated_target_scheduler_scope(
        model,
        role=role,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        expert_parallel=expert_parallel,
    ) is accepted


def _collectives():
    return tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "main-collective-fabric",
            "analytical_roofline_v1",
            None,
        )
        for operation in ("all_reduce", "all_gather"))


def _pipeline_p2p(pipeline_size):
    return tuple(
        PipelineP2PConfig(
            stage,
            stage + 1,
            "send_recv",
            "nvlink",
            "point_to_point",
            "main-pipeline-fabric",
            "analytical_roofline_v1",
        )
        for stage in range(pipeline_size - 1))


def _colocated_target_deployment(
        model, hardware_id, tensor_parallel_size, pipeline_parallel_size,
        expert_parallel):
    return build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    enable_expert_parallel=expert_parallel,
                ),
                ranks=tuple(range(
                    tensor_parallel_size * pipeline_parallel_size)),
                collectives=_collectives(),
                pipeline_p2p=_pipeline_p2p(pipeline_parallel_size),
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def _allow_all_emission(deployment, model):
    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )


def test_kimi_a800_production_scope_emits_with_allow_flag():
    model = _kimi()
    deployment = _colocated_target_deployment(
        model, "nvidia_a800", 8, 2, True)
    _allow_all_emission(deployment, model)


def test_kimi_a800_production_scope_requires_allow_flag():
    model = _kimi()
    deployment = _colocated_target_deployment(
        model, "nvidia_a800", 8, 2, True)
    with pytest.raises(NotImplementedError, match="Kimi-K2.6 PP2"):
        validate_workload_emission_parallelism(
            deployment,
            model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=False,
            allow_single_pool_tp2_pp2=True,
        )


def test_dsv3_l32_h200_tp4_scope_emits():
    model = _dsv3_l32()
    deployment = _colocated_target_deployment(
        model, "nvidia_h200", 4, 1, False)
    _allow_all_emission(deployment, model)


def test_dsv3_l32_tp8_emission_is_rejected():
    model = _dsv3_l32()
    deployment = _colocated_target_deployment(
        model, "nvidia_h200", 8, 1, False)
    with pytest.raises(NotImplementedError, match="unsupported rank-local"):
        _allow_all_emission(deployment, model)


def _target_prefill_step_workload(model, deployment):
    return build_workload(
        ScheduleStep(
            "target-model",
            "main",
            (ScheduledRequest("request", "prefill", 4, 0, 0),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(
            KVCacheEvent(
                "request", KVCacheAction.ALLOCATE, "hbm", 0, 16, 1024),
            KVCacheEvent(
                "request", KVCacheAction.MISS, "hbm", 0, 1, 64),
            KVCacheEvent(
                "request", KVCacheAction.HIT, "hbm", 0, 1, 64),
            KVCacheEvent(
                "request", KVCacheAction.RELEASE, "hbm", 0, 16, 1024),
        ),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )


def test_kimi_a800_production_scope_constructs_logical_workload():
    model = _kimi()
    deployment = _colocated_target_deployment(
        model, "nvidia_a800", 8, 2, True)
    graph = _target_prefill_step_workload(model, deployment)
    assert graph.node("target-model:token-embedding").attributes[
        "model_semantics_contract"] == "kimi_k2_6_vllm_0_27_1_logical_v1"


def test_construction_hardware_drift_is_visible_across_gates():
    # Drift, pinned end to end: construction accepts the Kimi A800
    # production topology on H200 hardware; emission rejects the same
    # deployment.
    model = _kimi()
    deployment = _colocated_target_deployment(
        model, "nvidia_h200", 8, 2, True)
    _target_prefill_step_workload(model, deployment)
    with pytest.raises(
            NotImplementedError,
            match="rank-local EP workload/kernel emission"):
        _allow_all_emission(deployment, model)


def test_native_scheduler_scope_admits_kimi_a800_production_layout():
    _validate_native_scheduler_scope(
        BackendConfig("main", "colocated", 128, 1, 8,
                      scheduler_pipeline_depth=1),
        VLLMSchedulerConfig(
            "configs/model/kimi-k2.6.json",
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            enable_expert_parallel=True,
        ),
    )


def test_native_scheduler_scope_rejects_unvalidated_kimi_tp8_ep1():
    with pytest.raises(
            NotImplementedError,
            match="only TP=1 or independently validated TP=2"):
        _validate_native_scheduler_scope(
            BackendConfig("main", "colocated", 128, 1, 8,
                          scheduler_pipeline_depth=1),
            VLLMSchedulerConfig(
                "configs/model/kimi-k2.6.json",
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
            ),
        )


def test_ep_scheduler_model_scope_reads_production_parallelism_table():
    kimi = _kimi()
    # Kimi's own production row and the TP8/PP1 row return without error.
    _validate_ep_scheduler_model_scope(
        VLLMSchedulerConfig(
            "configs/model/kimi-k2.6.json",
            tensor_parallel_size=8,
            pipeline_parallel_size=2,
            enable_expert_parallel=True,
        ),
        kimi,
    )
    _validate_ep_scheduler_model_scope(
        VLLMSchedulerConfig(
            "configs/model/kimi-k2.6.json",
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
        ),
        kimi,
    )
    # PP4 is GLM's production layout, not Kimi's: the gate falls through to
    # the Qwen3MoE slice and fails closed.
    with pytest.raises(NotImplementedError, match="BF16 Qwen3MoE"):
        _validate_ep_scheduler_model_scope(
            VLLMSchedulerConfig(
                "configs/model/kimi-k2.6.json",
                tensor_parallel_size=8,
                pipeline_parallel_size=4,
                enable_expert_parallel=True,
            ),
            kimi,
        )
    # DeepSeek-V3 has no production EP row at all.
    with pytest.raises(
            AssertionError, match="unhandled Scheduler model type"):
        _validate_ep_scheduler_model_scope(
            VLLMSchedulerConfig(
                "configs/model/deepseek-v3.json",
                tensor_parallel_size=8,
                pipeline_parallel_size=1,
                enable_expert_parallel=True,
            ),
            _dsv3(),
        )
