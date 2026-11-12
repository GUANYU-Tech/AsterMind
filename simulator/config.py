"""Independent physical resources, logical deployment and serving inputs."""

import json
from pathlib import Path
from typing import Annotated, Literal

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
            raise ValueError(
                "this M-L Cell entry supports DP=SP=EP=1; native EP reuses TP ranks"
            )
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
    egress_mode: Literal["single_gateway", "multi_gateway", "distributed"] = (
        "single_gateway"
    )
    external_terminals: tuple[Identifier, ...]
    graph_backend: Literal["midlayer"] = "midlayer"

    @model_validator(mode="after")
    def validate_placement(self):
        ranks = [p.rank for p in self.placements]
        if sorted(ranks) != list(range(self.parallel.workers)):
            raise ValueError("placements must cover every logical rank exactly once")
        if len({p.device for p in self.placements}) != len(ranks):
            raise ValueError("logical ranks require exclusive physical GPUs")
        if not self.external_terminals or len(set(self.external_terminals)) != len(
            self.external_terminals
        ):
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


class CellLinkAddition(Contract):
    source: Identifier
    target: Identifier


class SimulationConfig(Contract):
    schema_version: Literal[1, "orbitserve.native_cells.v1"] = 1
    # Ground resources are explicit in the outer model. The current network
    # adapter uses an owner container with no orbit; it never propagates it.
    physical: dict
    ground_owners: tuple[Identifier, ...] = ()
    satellite_roles: dict[Identifier, Literal["compute", "communication", "mixed"]] = (
        Field(default_factory=dict)
    )
    cells: tuple[ServingCell, ...]
    requests: tuple[Request, ...]
    migrations: tuple[Migration, ...] = ()
    backend: Literal["flowsim", "ns3", "simgrid"] = "flowsim"
    timeline_mode: Literal["orbitserve", "simgrid"] = "orbitserve"
    communication_mode: Literal[
        "flowsim", "protobridge_satellite", "protobridge_cell"
    ] = "flowsim"
    protobridge_startup_us: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    horizon_ns: Positive = 1_000_000_000
    packet_bytes: Literal[1400] = 1400
    record_packets: bool = False
    physical_dataset: str | None = None
    physical_network: dict | str | None = None
    physical_link_bindings: dict[str, str] = Field(default_factory=dict)
    additional_cell_links: tuple[CellLinkAddition, ...] = ()
    disabled_physical_links: tuple[Identifier, ...] = ()
    runtime_network: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def scene_default_communication(cls, value):
        # Numeric-v1 files retain their historical transport behavior. New
        # scene-editor documents default to the fewest FlowSim events.
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "orbitserve.native_cells.v1"
            and "communication_mode" not in value
        ):
            return {**value, "communication_mode": "protobridge_cell"}
        return value

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ("physical_dataset", "physical_network"):
            value = data.get(key)
            if isinstance(value, str) and not Path(value).is_absolute():
                data[key] = str((path.parent / value).resolve())
        return cls.model_validate(data)

    def topology(self):
        from .resources import compile_topology

        return compile_topology(self)

    @model_validator(mode="after")
    def validate_ids(self):
        if self.timeline_mode == 'simgrid' and self.backend != 'simgrid':
            raise ValueError('SimGrid timeline requires the SimGrid communication backend')
        if self.backend == 'simgrid' and self.timeline_mode != 'simgrid':
            raise ValueError('native SimGrid communication requires timeline_mode=simgrid')
        if self.communication_mode != "flowsim" and self.backend not in {"flowsim", "simgrid"}:
            raise ValueError(
                "ProtoBridge modes require FlowSim or SimGrid for external communication"
            )
        from orbitserve.runtime import RUNTIME_NETWORK_FIELDS

        if set(self.runtime_network) - set(RUNTIME_NETWORK_FIELDS):
            raise ValueError(
                "runtime_network may only override runtime faults/recovery fields"
            )
        if self.physical.get("link_policy") is not None:
            raise ValueError(
                "use the canonical physical_network allocator for acquisition and terminal policy"
            )
        if self.physical_dataset is not None and self.physical_network is not None:
            raise ValueError("supply physical_dataset or physical_network, not both")
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
            if (
                migration.source_cell not in known
                or migration.target_cell not in known
                or migration.source_cell == migration.target_cell
            ):
                raise ValueError("migration needs two distinct known Cells")
        return self
