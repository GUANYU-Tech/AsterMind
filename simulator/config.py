"""Independent physical resources, logical deployment and serving inputs."""
from pathlib import Path
from typing import Literal, Annotated
import json

from pydantic import Field, model_validator
from network_sim.model import Contract, Identifier, Nanoseconds

Positive = Annotated[int, Field(strict=True, gt=0)]


class Parallelism(Contract):
    # Native physical worker count is TP * PP. EP reuses TP ranks and is
    # selected through native_expert_parallel, not another multiplicative axis.
    dp: Positive = 1
    tp: Positive = 1
    sp: Positive = 1
    pp: Positive = 1
    ep: Positive = 1

    @property
    def workers(self):
        return self.tp * self.pp

    @model_validator(mode="after")
    def validate_native_axes(self):
        if (self.dp, self.sp, self.ep) != (1, 1, 1):
            raise ValueError("this M-L Cell entry supports DP=SP=EP=1; native EP reuses TP ranks")
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
    max_batched_tokens: Positive = 32
    max_requests: Positive = 4
    num_kv_blocks: Positive = 64
    kv_block_size_tokens: Positive = 16
    prefix_caching: bool = False
    native_expert_parallel: bool = False
    expert_routing_seed: Annotated[int, Field(strict=True, ge=0)] | None = None
    host_launch_profile: str | None = None
    memory_policy_file: str | None = None
    watermark: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0
    # Egress is a physical role; it never changes the logical rank graph.
    egress_mode: Literal["single_gateway", "multi_gateway", "distributed"] = "single_gateway"
    external_terminals: tuple[Identifier, ...]
    graph_backend: Literal["midlayer"] = "midlayer"

    @model_validator(mode="after")
    def validate_placement(self):
        ranks = [p.rank for p in self.placements]
        if sorted(ranks) != list(range(self.parallel.workers)):
            raise ValueError("placements must cover every logical rank exactly once")
        if len({p.device for p in self.placements}) != len(ranks):
            raise ValueError("logical ranks require exclusive physical GPUs")
        if not self.external_terminals or len(set(self.external_terminals)) != len(self.external_terminals):
            raise ValueError("external terminals must be nonempty and distinct")
        return self


class Request(Contract):
    id: Identifier
    arrival_ns: Nanoseconds = 0
    user_device: Identifier
    input_tokens: Positive
    output_tokens: Positive = 1
    input_bytes: Nanoseconds = 0
    output_bytes_per_token: Nanoseconds = 4
    cell: Identifier | None = None
    session_id: str | None = None
    context_id: str | None = None
    parent_context_id: str | None = None
    prefix_hit_tokens: Nanoseconds = 0

    @model_validator(mode="after")
    def validate_history(self):
        if self.prefix_hit_tokens > self.input_tokens:
            raise ValueError("prefix hit exceeds prompt")
        if (self.context_id or self.parent_context_id) and not self.session_id:
            raise ValueError("context identity requires session_id")
        return self


class Migration(Contract):
    id: Identifier
    after_request: Identifier
    source_cell: Identifier
    target_cell: Identifier
    session_id: str
    context_id: str
    mode: Literal["copy", "move"] = "copy"


class OrbitConfiguration(Contract):
    # The existing O-S module produces the canonical, hash-verified trace.
    trace_manifest: str | None = None
    samples: tuple[dict, ...] = ()
    elements_file: str | None = None
    simulation_epoch_utc: str | None = None
    sample_step_ns: Positive = 1_000_000_000
    mode: Literal["per_member", "reference_offsets"] = "per_member"
    reference_satellite: Identifier | None = None
    offsets_itrs_km: dict[str, tuple[float, float, float]] = {}
    max_isl_range_km: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10000
    min_ground_elevation_deg: Annotated[float, Field(ge=-90, le=90)] = 0

    @model_validator(mode="after")
    def validate_source(self):
        if sum((self.trace_manifest is not None, bool(self.samples), self.elements_file is not None)) != 1:
            raise ValueError("supply one orbit source: canonical manifest, samples, or SGP4 elements")
        if self.elements_file and not self.simulation_epoch_utc:
            raise ValueError("SGP4 propagation needs simulation_epoch_utc")
        if self.mode == "reference_offsets" and not self.reference_satellite:
            raise ValueError("reference-offset formation needs a reference satellite")
        if self.mode == "per_member" and (self.reference_satellite or self.offsets_itrs_km):
            raise ValueError("independent propagation cannot contain reference offsets")
        return self


class LinkParameters(Contract):
    kind: Literal["manual", "islsim"] = "manual"
    distance_km: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    fso: dict = {}


class SimulationConfig(Contract):
    schema_version: Literal[1] = 1
    # Ground resources are explicit in the outer model. The current network
    # adapter uses an owner container with no orbit; it never propagates it.
    physical: dict
    ground_owners: tuple[Identifier, ...] = ()
    satellite_roles: dict[Identifier, Literal["compute", "communication", "mixed"]] = {}
    cells: tuple[ServingCell, ...]
    requests: tuple[Request, ...]
    migrations: tuple[Migration, ...] = ()
    backend: Literal["flowsim", "ns3"] = "flowsim"
    horizon_ns: Positive = 1_000_000_000
    packet_bytes: Positive = 1400
    orbit: OrbitConfiguration | None = None
    link_parameters: dict[str, LinkParameters] = {}

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("trace_manifest", "elements_file"):
            value = data.get("orbit", {}).get(key)
            if value and not Path(value).is_absolute():
                data["orbit"][key] = str((path.parent / value).resolve())
        return cls.model_validate(data)

    def topology(self):
        from .resources import compile_topology
        return compile_topology(self)

    @model_validator(mode="after")
    def validate_ids(self):
        for entries in (self.cells, self.requests):
            ids = [entry.id for entry in entries]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate Cell/request ID")
        if not self.cells:
            raise ValueError("at least one serving Cell required")
        known = {c.id for c in self.cells}
        if any(r.cell is not None and r.cell not in known for r in self.requests):
            raise ValueError("request selects an unknown Cell")
        if any(r.arrival_ns > self.horizon_ns for r in self.requests):
            raise ValueError("request arrival exceeds simulation horizon")
        if len({m.id for m in self.migrations}) != len(self.migrations):
            raise ValueError("duplicate migration ID")
        for migration in self.migrations:
            if migration.after_request not in {r.id for r in self.requests}:
                raise ValueError("unknown migration trigger request")
            if migration.source_cell not in known or migration.target_cell not in known or migration.source_cell == migration.target_cell:
                raise ValueError("migration needs two distinct known Cells")
        return self
