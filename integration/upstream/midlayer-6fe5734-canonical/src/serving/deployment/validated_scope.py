"""The validated target-model deployment scopes as one declarative table.

Three production gates historically restated overlapping subsets of the same
validated deployment facts:

* ``src.serving.workload.build_workload`` -- the target logical workload
  construction gate;
* ``src.serving.simulation.validate_workload_emission_parallelism`` -- the
  rank-local workload/kernel emission gate;
* ``src.serving.backend.vllm._validate_native_scheduler_scope`` -- the vLLM
  Scheduler control-plane gate.

The gates do not enforce identical granularity.  The differences are
recorded here as explicit per-row fields instead of being re-derived at each
call site; this module documents the drift without adjudicating it:

* The construction gate predates the hardware pins on the A800 production
  and PD EP8 rows: it accepts those topologies on any hardware and never
  inspects the compute dtype (``construction_checks_hardware=False``).  The
  H200 rows carry hardware pins in every gate.
* The emission gate is the only gate that sees pools, so it alone pins the
  pool size (ranks/workers == TP*PP), requires BF16 compute, and gates the
  A800 production rows behind the caller's ``allow_single_pool_tp1_pp``
  flag.  It also derives the DeepSeek-V3 aggregate TP from the config
  identity: the 32-layer variant is TP4-only at emission, while the
  construction gate accepts that identity at TP8 as well.
* The Scheduler gate knows roles instead of deployments or hardware: it pins
  the five pinned config identities (``TARGET_MODEL_CONFIG_SHA256``) and its
  rows deliberately over-approximate -- the PP2/PP4 rows accept every named
  target model in any role, and the PD EP8 row accepts every named target
  model.  Combinations no other gate validates are failed closed afterwards
  by the narrower per-model checks in ``_construct_native_scheduler`` and
  ``_validate_ep_scheduler_model_scope``.
"""

from dataclasses import dataclass

from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    ModelConfig,
    is_named_target_model_identity,
)


@dataclass(frozen=True)
class ValidatedScope:
    """One validated target-model deployment topology.

    The fields through ``hardware_ids`` state the validated fact itself;
    ``config_sha256`` pins the one identity-sensitive variant (DeepSeek-V3
    TP4 is validated only for the 32-layer local config).  The remaining
    fields record which axes each consuming gate actually enforces, so the
    historical granularity drift is data rather than call-site folklore.
    """

    name: str
    scope_kind: str
    model_types: tuple[str, ...]
    deployment_modes: tuple[str, ...]
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel: bool
    expert_parallel_size: int
    hardware_ids: tuple[str, ...]
    config_sha256: str | None
    # Construction gate: False on the rows that predate the hardware pin.
    construction_checks_hardware: bool
    # Emission gate: False when the gate never consults this row (the
    # fixture layout is accepted for every single-rank pool up front).
    emission_gate: bool
    # Emission gate: the A800 production rows additionally require the
    # caller's allow_single_pool_tp1_pp flag.
    emission_requires_allow_flag: bool
    # Emission gate: identities excluded on this row (the TP8 DeepSeek-V3
    # aggregate row excludes the 32-layer identity, which is TP4-only at
    # emission; the construction gate accepts it at TP8 too).
    emission_excludes_config_sha256: tuple[str, ...]
    # Scheduler gate: None when the Scheduler never accepts this topology.
    # Where the Scheduler accepts a wider model set than the validated one,
    # scheduler_model_types records the Scheduler's own (over-approximate)
    # set; scheduler_roles=None means the Scheduler row is role-free.
    scheduler_model_types: tuple[str, ...] | None
    scheduler_roles: tuple[str, ...] | None
    scheduler_requires_named_identity: bool

    def __post_init__(self) -> None:
        for field in ("name", "scope_kind"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ValidatedScope.{field} must be non-empty")
        for field in ("model_types", "deployment_modes", "hardware_ids"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or not value or any(
                    not isinstance(item, str) or not item for item in value):
                raise TypeError(
                    f"ValidatedScope.{field} must be a non-empty tuple of "
                    "non-empty strings")
        for field in (
                "tensor_parallel_size", "pipeline_parallel_size",
                "expert_parallel_size"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"ValidatedScope.{field} must be an integer")
            if value < 1:
                raise ValueError(f"ValidatedScope.{field} must be positive")
        for field in (
                "expert_parallel", "construction_checks_hardware",
                "emission_gate", "emission_requires_allow_flag",
                "scheduler_requires_named_identity"):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"ValidatedScope.{field} must be boolean")
        if self.config_sha256 is not None and (
                not isinstance(self.config_sha256, str)
                or len(self.config_sha256) != 64
                or any(character not in "0123456789abcdef"
                       for character in self.config_sha256)):
            raise ValueError(
                "ValidatedScope.config_sha256 must be a lowercase SHA256 or "
                "null")
        if not isinstance(self.emission_excludes_config_sha256, tuple):
            raise TypeError(
                "ValidatedScope.emission_excludes_config_sha256 must be a "
                "tuple")
        if self.scheduler_model_types is not None and (
                not isinstance(self.scheduler_model_types, tuple)
                or not self.scheduler_model_types):
            raise TypeError(
                "ValidatedScope.scheduler_model_types must be a non-empty "
                "tuple or null")
        if self.scheduler_roles is not None and (
                not isinstance(self.scheduler_roles, tuple)
                or not self.scheduler_roles):
            raise TypeError(
                "ValidatedScope.scheduler_roles must be a non-empty tuple "
                "or null")


_TARGET_MODEL_TYPES = ("kimi_k2", "glm_moe_dsa", "deepseek_v3")

_VALIDATED_TARGET_SCOPES = (
    # The TP=PP=EP=1 fixture layout: accepted for the fixed target models in
    # colocated and PD deployments on any hardware.  The emission gate never
    # consults this row because it accepts every single-rank pool up front,
    # and the Scheduler gate handles TP=1 outside the target scope.
    ValidatedScope(
        name="fixture_tp1_pp1_ep1",
        scope_kind="fixture",
        model_types=_TARGET_MODEL_TYPES,
        deployment_modes=("colocated", "pd_disagg"),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_a800", "nvidia_h200"),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=False,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=None,
        scheduler_roles=None,
        scheduler_requires_named_identity=False,
    ),
    # Source-backed A800 production layout: one colocated TP=EP=8 pool,
    # PP2 for Kimi-K2.6.
    ValidatedScope(
        name="kimi_a800_colocated_tp8_pp2_ep8",
        scope_kind="a800_production",
        model_types=("kimi_k2",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_a800",),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=True,
        emission_requires_allow_flag=True,
        emission_excludes_config_sha256=(),
        scheduler_model_types=_TARGET_MODEL_TYPES,
        scheduler_roles=None,
        scheduler_requires_named_identity=True,
    ),
    # Source-backed A800 production layout: one colocated TP=EP=8 pool,
    # PP4 for GLM-5.2.
    ValidatedScope(
        name="glm_a800_colocated_tp8_pp4_ep8",
        scope_kind="a800_production",
        model_types=("glm_moe_dsa",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=4,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_a800",),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=True,
        emission_requires_allow_flag=True,
        emission_excludes_config_sha256=(),
        scheduler_model_types=_TARGET_MODEL_TYPES,
        scheduler_roles=None,
        scheduler_requires_named_identity=True,
    ),
    # Kimi-K2.6 / GLM-5.2 PD production layout: one TP=EP=8 pool per role,
    # PP1, on A800 or H200.
    ValidatedScope(
        name="target_pd_tp8_pp1_ep8",
        scope_kind="production_pd",
        model_types=("kimi_k2", "glm_moe_dsa"),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_a800", "nvidia_h200"),
        config_sha256=None,
        construction_checks_hardware=False,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=_TARGET_MODEL_TYPES,
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
    # GLM-5.2 H200 aggregate: one colocated TP8/PP1/EP1 pool.
    ValidatedScope(
        name="glm_h200_colocated_tp8_pp1_ep1",
        scope_kind="h200_aggregate",
        model_types=("glm_moe_dsa",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("glm_moe_dsa",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # GLM-5.2 H200 PD: one TP8/PP1/EP1 pool per role.
    ValidatedScope(
        name="glm_h200_pd_tp8_pp1_ep1",
        scope_kind="h200_pd_ep1",
        model_types=("glm_moe_dsa",),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("glm_moe_dsa",),
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
    # Kimi-K2.6 H200 InferenceX-style colocated layout: one TP8/PP1 pool
    # with EP8 (EP=TP).  W4A16 routed-expert residency closes on 8x141GB
    # HBM, so the PP2 A800 production scope is not the only feasible Kimi
    # colocated layout.
    ValidatedScope(
        name="kimi_h200_colocated_tp8_pp1_ep8",
        scope_kind="h200_colocated_ep8",
        model_types=("kimi_k2",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=True,
        expert_parallel_size=8,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("kimi_k2",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # DeepSeek-V3 H200 InferenceX-style aggregate: one colocated
    # TP8/PP1/EP1 pool (FP8 weights TP8-sharded fit one H200 per rank).
    # The emission gate derives the aggregate TP from the config identity,
    # so it excludes the 32-layer variant here; the construction gate
    # accepts that identity at TP8 as well.
    ValidatedScope(
        name="dsv3_h200_colocated_tp8_pp1_ep1",
        scope_kind="h200_aggregate",
        model_types=("deepseek_v3",),
        deployment_modes=("colocated",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(DEEPSEEK_V3_L32_CONFIG_SHA256,),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=True,
    ),
    # The 32-layer local DeepSeek-V3 variant measured at TP4 on four H200s:
    # one colocated TP4/PP1/EP1 pool running the same rank-local DAG.  The
    # Scheduler row pins the config identity but not the named-target set.
    ValidatedScope(
        name="dsv3_h200_colocated_tp4_pp1_ep1_l32",
        scope_kind="h200_aggregate",
        model_types=("deepseek_v3",),
        deployment_modes=("colocated",),
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=DEEPSEEK_V3_L32_CONFIG_SHA256,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("colocated",),
        scheduler_requires_named_identity=False,
    ),
    # DeepSeek-V3 H200 PD: one TP8/PP1/EP1 pool per role.  The PD transfer
    # payload is the compressed MLA kv_c||k_pe cache only (no DSA index
    # cache), TP-replicated, so the rank-pair transfer is EP-independent.
    ValidatedScope(
        name="dsv3_h200_pd_tp8_pp1_ep1",
        scope_kind="h200_pd_ep1",
        model_types=("deepseek_v3",),
        deployment_modes=("pd_disagg",),
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel=False,
        expert_parallel_size=1,
        hardware_ids=("nvidia_h200",),
        config_sha256=None,
        construction_checks_hardware=True,
        emission_gate=True,
        emission_requires_allow_flag=False,
        emission_excludes_config_sha256=(),
        scheduler_model_types=("deepseek_v3",),
        scheduler_roles=("prefill", "decode"),
        scheduler_requires_named_identity=True,
    ),
)

# The source-backed A800 production parallel layout per target model,
# derived from the table rows so the Scheduler EP gate and the workload
# gates cannot restate it separately.  The Scheduler EP gate additionally
# accepts the TP8/PP1 layout (the production_pd and h200_colocated_ep8
# rows) for both models.
TARGET_MODEL_PRODUCTION_PARALLELISM = {
    row.model_types[0]: (
        row.tensor_parallel_size, row.pipeline_parallel_size)
    for row in _VALIDATED_TARGET_SCOPES
    if row.scope_kind == "a800_production"
}


def _scope_integer(value, field: str, gate: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{gate} scope {field} must be an integer")
    if value < 1:
        raise ValueError(f"{gate} scope {field} must be positive")


def _scope_common_fields(
    gate: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
) -> None:
    _scope_integer(tensor_parallel_size, "tensor_parallel_size", gate)
    _scope_integer(pipeline_parallel_size, "pipeline_parallel_size", gate)
    if not isinstance(expert_parallel, bool):
        raise TypeError(f"{gate} scope expert_parallel must be boolean")


def is_validated_target_construction_scope(
    model: ModelConfig,
    *,
    deployment_mode: str,
    hardware_id: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
    expert_parallel_size: int,
) -> bool:
    """Return whether the target logical workload construction gate accepts.

    This is exactly the scope disjunction of
    ``build_workload._build_target_model_logical_workload``: it checks the
    model/config identity against the rows, the deployment mode and the
    parallel axes, and the hardware pin only on the rows that carry one for
    this gate.  Pool size, rank locality, and the single-pool colocated rule
    remain the caller's own structural checks.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("target construction scope requires a ModelConfig")
    for label, value in (
            ("deployment_mode", deployment_mode),
            ("hardware_id", hardware_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"target construction scope {label} must be non-empty")
    _scope_common_fields(
        "target construction",
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel,
    )
    _scope_integer(
        expert_parallel_size, "expert_parallel_size", "target construction")
    for row in _VALIDATED_TARGET_SCOPES:
        if model.model_type not in row.model_types:
            continue
        if deployment_mode not in row.deployment_modes:
            continue
        if (tensor_parallel_size, pipeline_parallel_size) != (
                row.tensor_parallel_size, row.pipeline_parallel_size):
            continue
        if expert_parallel is not row.expert_parallel:
            continue
        if expert_parallel_size != row.expert_parallel_size:
            continue
        if (row.config_sha256 is not None
                and model.config_sha256 != row.config_sha256):
            continue
        if row.construction_checks_hardware and (
                hardware_id not in row.hardware_ids):
            continue
        return True
    return False


def validated_target_emission_scope(
    model: ModelConfig,
    *,
    deployment_mode: str,
    hardware_id: str,
    single_pool: bool,
    pool_rank_count: int,
    worker_count: int,
    is_rank_local: bool,
    allow_production_pipeline: bool,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
    expert_parallel_size: int,
) -> ValidatedScope | None:
    """Return the validated emission scope one target pool matches, if any.

    This is exactly the target-model branch sequence of
    ``simulation.validate_workload_emission_parallelism``: BF16 compute and
    rank locality up front, then the rows with their pool-size
    (ranks/workers == TP*PP), single-pool colocated rule, hardware pins, the
    ``allow_single_pool_tp1_pp`` flag on the A800 production rows, and the
    32-layer exclusion on the TP8 DeepSeek-V3 aggregate row.  Single-rank
    fixture pools and the non-target Qwen3/Llama slices remain the caller's
    own branches.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("target emission scope requires a ModelConfig")
    for label, value in (
            ("deployment_mode", deployment_mode),
            ("hardware_id", hardware_id)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"target emission scope {label} must be non-empty")
    for label, value in (
            ("single_pool", single_pool),
            ("is_rank_local", is_rank_local),
            ("allow_production_pipeline", allow_production_pipeline)):
        if not isinstance(value, bool):
            raise TypeError(f"target emission scope {label} must be boolean")
    _scope_common_fields(
        "target emission",
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel,
    )
    _scope_integer(expert_parallel_size, "expert_parallel_size",
                   "target emission")
    _scope_integer(pool_rank_count, "pool_rank_count", "target emission")
    _scope_integer(worker_count, "worker_count", "target emission")
    if model.compute_dtype != "bf16" or not is_rank_local:
        return None
    for row in _VALIDATED_TARGET_SCOPES:
        if not row.emission_gate:
            continue
        if model.model_type not in row.model_types:
            continue
        if deployment_mode not in row.deployment_modes:
            continue
        if (tensor_parallel_size, pipeline_parallel_size) != (
                row.tensor_parallel_size, row.pipeline_parallel_size):
            continue
        if expert_parallel is not row.expert_parallel:
            continue
        if expert_parallel_size != row.expert_parallel_size:
            continue
        if hardware_id not in row.hardware_ids:
            continue
        if (row.config_sha256 is not None
                and model.config_sha256 != row.config_sha256):
            continue
        if model.config_sha256 in row.emission_excludes_config_sha256:
            continue
        if row.emission_requires_allow_flag and not allow_production_pipeline:
            continue
        if "colocated" in row.deployment_modes and not single_pool:
            continue
        pool_size = row.tensor_parallel_size * row.pipeline_parallel_size
        if pool_rank_count != pool_size or worker_count != pool_size:
            continue
        return row
    return None


def is_validated_target_scheduler_scope(
    model: ModelConfig,
    *,
    role: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    expert_parallel: bool,
) -> bool:
    """Return whether the vLLM Scheduler gate's target scope accepts.

    This is exactly the ``target_scope`` disjunction of
    ``vllm._validate_native_scheduler_scope``: the Scheduler knows roles
    rather than deployments or hardware, pins the named config identities on
    every row except the 32-layer DeepSeek-V3 TP4 row, and deliberately
    over-approximates (the PP2/PP4 rows accept every named target model in
    any role; the PD EP8 row accepts every named target model).  The
    narrower per-model checks in ``_construct_native_scheduler`` and
    ``_validate_ep_scheduler_model_scope`` fail closed afterwards.
    """

    if not isinstance(model, ModelConfig):
        raise TypeError("target scheduler scope requires a ModelConfig")
    if not isinstance(role, str) or not role:
        raise ValueError("target scheduler scope role must be non-empty")
    _scope_common_fields(
        "target scheduler",
        tensor_parallel_size,
        pipeline_parallel_size,
        expert_parallel,
    )
    named = is_named_target_model_identity(
        model.model_type, model.config_sha256)
    for row in _VALIDATED_TARGET_SCOPES:
        if row.scheduler_model_types is None:
            continue
        if model.model_type not in row.scheduler_model_types:
            continue
        if row.scheduler_roles is not None and role not in row.scheduler_roles:
            continue
        if (tensor_parallel_size, pipeline_parallel_size) != (
                row.tensor_parallel_size, row.pipeline_parallel_size):
            continue
        if expert_parallel is not row.expert_parallel:
            continue
        if row.scheduler_requires_named_identity and not named:
            continue
        if (row.config_sha256 is not None
                and model.config_sha256 != row.config_sha256):
            continue
        return True
    return False
