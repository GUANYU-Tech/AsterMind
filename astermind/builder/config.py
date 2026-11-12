"""Independent physical resources, logical deployment and serving inputs."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from astermind.common.contracts import Contract, Identifier, Nanoseconds
from astermind.engine.comm.frame_surrogate import CommPolicy

Positive = Annotated[int, Field(strict=True, gt=0)]


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
    max_batched_tokens: Positive = 32
    max_requests: Positive = 4
    num_kv_blocks: Positive = 64
    kv_block_size_tokens: Positive = 16
    scheduler_adapter: Literal["auto", "vllm_0_11", "vllm_0_30", "vllm_0_31"] = "auto"
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
    before_request: Identifier | None = None
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
    compute_mode: Literal["midlayer", "roofline"] = "midlayer"
    communication: CommPolicy = Field(default_factory=CommPolicy)
    horizon_ns: Positive = 1_000_000_000
    physical_dataset: str | None = None
    physical_network: dict | str | None = None
    physical_link_bindings: dict[str, str] = Field(default_factory=dict)
    additional_cell_links: tuple[CellLinkAddition, ...] = ()
    disabled_physical_links: tuple[Identifier, ...] = ()
    runtime_network: dict = Field(default_factory=dict)

    @property
    def backend(self):
        """Read-only label for the unchanged historical report renderer."""
        return 'runtime_comm'

    @property
    def timeline_mode(self):
        """Read-only clock label; neither label is a configurable old mode."""
        return 'orbitserve'

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
        from astermind.builder.resources import compile_topology

        return compile_topology(self)

    def predecessor(self, request):
        # Ported from Communication 9c08c9c ConfiguredScheduler.predecessor;
        # Cell selection remains owned by the existing Engine policy.
        if request.parent_context_id is None:
            return None
        for parent in self.requests:
            # Retain the current native history key when context_id is omitted.
            if (parent.session_id, parent.context_id or parent.id) == (
                request.session_id, request.parent_context_id
            ):
                return parent.id
        raise ValueError("prerequisite requires an explicit parent request in the workload")

    @model_validator(mode="after")
    def validate_ids(self):
        from astermind.engine import RUNTIME_NETWORK_FIELDS

        if set(self.runtime_network) - (set(RUNTIME_NETWORK_FIELDS) - {'transport_model','failure_policy','max_transfer_retries'}):
            raise ValueError(
                "runtime_network accepts faults only; use communication for execution/retry policy"
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
        if set(self.communication.cell_policies) - known:
            raise ValueError("communication policy refers to an unknown Cell")
        # Communication 9c08c9c SimulationConfig.validate_ids, adapted to the
        # retained flat config and explicit receiving Cell (no default_cell).
        dependencies = {r.id: [] for r in self.requests}
        contexts = [(r.session_id, r.context_id or r.id) for r in self.requests if r.session_id]
        if len(contexts) != len(set(contexts)):
            raise ValueError("duplicate session context")
        for request in self.requests:
            parent = self.predecessor(request)
            if parent is not None:
                dependencies[request.id].append(parent)
        for migration in self.migrations:
            if migration.before_request in dependencies:
                dependencies[migration.before_request].append(migration.after_request)

        def visit(node, active, done):
            if node in active:
                raise ValueError("cyclic KV prerequisites")
            if node in done:
                return
            for parent in dependencies.get(node, ()):
                visit(parent, active | {node}, done)
            done.add(node)

        done = set()
        for node in dependencies:
            visit(node, set(), done)
        requests = {r.id: r for r in self.requests}
        for migration in self.migrations:
            if migration.before_request is not None:
                if (migration.before_request not in requests
                        or migration.before_request == migration.after_request):
                    raise ValueError("KV prerequisite requires a distinct known receiving request")
                request = requests[migration.before_request]
                if request.cell != migration.target_cell:
                    raise ValueError("KV destination must match explicit receiving request Cell")
                if (request.session_id != migration.session_id
                        or request.parent_context_id != migration.context_id):
                    raise ValueError("KV history must match receiving request parent context")
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
