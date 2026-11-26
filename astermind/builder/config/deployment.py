"""Logical model placement and native Scheduler configuration."""
from typing import Annotated, Literal
from pydantic import AliasChoices, Field, StrictBool, model_validator
from astermind.common.contracts import Contract, ComputeModeConfig, Identifier, Nanoseconds

Positive = Annotated[int, Field(strict=True, gt=0)]

class FastRoofline(Contract):
    tensor_percent: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)] = 70
    hbm_percent: Annotated[float, Field(gt=0, le=100, allow_inf_nan=False)] = 96


class Parallelism(Contract):
    # Native physical worker count is TP * PP * DP. EP reuses stage-local ranks.
    dp: Positive = 1
    tp: Positive = 1
    sp: Positive = 1
    pp: Positive = 1
    ep: Positive = 1

    @property
    def workers(self):
        return self.tp * self.pp * self.dp

    @model_validator(mode="after")
    def validate_native_axes(self):
        if self.sp != 1:
            raise ValueError("SP is not supported by the current Cell integration")
        if self.ep not in (1, self.tp * self.dp):
            raise ValueError("EP must use the stage-local TP x DP group")
        return self


class Placement(Contract):
    rank: Annotated[int, Field(strict=True, ge=0)]
    device: Identifier
    pool: str = "main"


class ServingCell(Contract):
    id: Identifier
    model: str = "configs/model/qwen3-4b.json"
    hardware: str = "configs/hardware/nvidia_a800.yaml"
    hardware_id: str = "nvidia_a800"
    parallel: Parallelism = Parallelism()
    placements: tuple[Placement, ...]
    ingress_device: Identifier
    output_device: Identifier
    max_model_len: Positive = 128
    max_batched_tokens: Positive | None = None
    max_requests: Positive = 4
    num_kv_blocks: Positive = 64
    kv_block_size_tokens: Positive = 16
    scheduler_adapter: Literal["auto", "vllm_0_11", "vllm_0_30", "vllm_0_31"] = "auto"
    enable_chunked_prefill: StrictBool | None = Field(
        default=None, validation_alias=AliasChoices("enable_chunked_prefill", "chunked_prefill"),
    )
    long_prefill_token_threshold: Annotated[int, Field(strict=True, ge=0)] = 0
    prefix_caching: bool = False
    native_expert_parallel: bool = False
    wide_ep_enabled: bool = False
    expert_routing_seed: Annotated[int, Field(strict=True, ge=0)] | None = None
    deployment: Literal["colocated", "pd_disagg"] = "colocated"
    host_launch_profile: str | None = None
    memory_policy_file: str | None = None
    watermark: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0
    # Egress is a physical role; it never changes the logical rank graph.
    egress_mode: Literal["single_gateway", "multi_gateway", "distributed"] = "single_gateway"
    external_terminals: tuple[Identifier, ...]
    execution_mode: Literal["accelerated", "fast", "reference"] = "accelerated"
    graph_backend: Literal["midlayer"] = "midlayer"
    fast_glm_dsa_score: Literal["bf16_timing_proxy"] | None = None
    fast_roofline: FastRoofline | None = None
    fast_kv_pool_blocks: Annotated[int, Field(strict=True, ge=2)] | None = None
    fast_link_contention: StrictBool = False
    detail_prediction_method: Literal["midlayer", "roofline"] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_scheduler_adapter_name(cls, value):
        if isinstance(value, dict) and value.get("scheduler_adapter") == "upstream_exact":
            value = dict(value)
            value["scheduler_adapter"] = "vllm_0_11"
        return value

    @property
    def expert_parallel_enabled(self):
        return self.native_expert_parallel or self.parallel.ep > 1

    @model_validator(mode="after")
    def validate_placement(self):
        if self.long_prefill_token_threshold and self.enable_chunked_prefill is not True:
            raise ValueError("long_prefill_token_threshold requires enable_chunked_prefill=true")
        if self.parallel.dp > 1 and not self.wide_ep_enabled:
            raise ValueError("multiple DP replicas in one Cell require wide_ep_enabled=true")
        if self.wide_ep_enabled and (self.parallel.dp <= 1 or not self.expert_parallel_enabled
                                    or self.parallel.ep != self.parallel.tp*self.parallel.dp):
            raise ValueError("WideEP requires DP>1 and the declared stage-local TP x DP expert group")
        ranks = [p.rank for p in self.placements]
        workers = self.parallel.workers
        if self.deployment == "pd_disagg":
            if self.prefix_caching or self.watermark or self.memory_policy_file:
                raise ValueError("PD smoke uses explicit KV capacity and no session cache")
            if any(p.pool != ("prefill" if p.rank < workers else "decode") for p in self.placements):
                raise ValueError("PD placement requires prefill ranks followed by decode ranks")
            workers *= 2
        elif any(p.pool != "main" for p in self.placements):
            raise ValueError("colocated Cell placements use the main pool")
        if sorted(ranks) != list(range(workers)):
            raise ValueError("placements must cover every logical rank exactly once")
        if len({p.device for p in self.placements}) != len(ranks):
            raise ValueError("logical ranks require exclusive physical GPUs")
        if not self.external_terminals or len(set(self.external_terminals)) != len(self.external_terminals):
            raise ValueError("external terminals must be nonempty and distinct")
        return self


class SceneDeployment(ComputeModeConfig):
    fast_roofline: FastRoofline | None = None
    fast_kv_pool_blocks: Annotated[int, Field(strict=True, ge=2)] | None = None
    fast_link_contention: StrictBool = False
    model: str = "configs/model/qwen3-8b.json"
    hardware: str = "configs/hardware/nvidia_a800.yaml"
    hardware_id: str = "nvidia_a800"
    gpus_per_satellite: Annotated[int, Field(strict=True, ge=1, le=64)] = 1
    tp: Positive = 2
    pp: Positive = 2
    dp: Positive = 1
    expert_parallel: bool = False
    wide_ep_enabled: bool = False
    fast_glm_dsa_score: Literal["bf16_timing_proxy"] | None = None
    expert_routing_seed: Annotated[int, Field(strict=True, ge=0)] | None = None
    max_requests: Annotated[int, Field(strict=True, ge=1, le=1024)] = 4
    max_model_len: Annotated[int, Field(strict=True, ge=2, le=131072)] = 128
    max_batched_tokens: Positive | None = None
    num_kv_blocks: Positive = 64
    kv_block_size_tokens: Positive = 16
    scheduler_adapter: Literal["auto", "vllm_0_11", "vllm_0_30", "vllm_0_31"] = "auto"
    memory_policy_file: str | None = None
    prefix_caching: bool = False
    enable_chunked_prefill: StrictBool | None = Field(
        default=None, validation_alias=AliasChoices("enable_chunked_prefill", "chunked_prefill"),
    )
    long_prefill_token_threshold: Annotated[int, Field(strict=True, ge=0)] = 0
    gpu_link_bandwidth_gbps: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1600
    gpu_link_latency_ns: Annotated[int, Field(strict=True, ge=0)] = 200
    communication: dict = Field(default_factory=dict)


# Preserve public serialization identities across the file-only move.
Parallelism.__module__ = 'astermind.builder.config'
Placement.__module__ = 'astermind.builder.config'
ServingCell.__module__ = 'astermind.builder.config'

# Preserve public serialization identities across the file-only move.
SceneDeployment.__module__ = 'astermind.builder.scene'
