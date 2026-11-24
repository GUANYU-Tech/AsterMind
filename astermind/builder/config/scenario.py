"""Simulation entry configuration and cross-configuration constraints."""
import json
from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field, model_validator
from astermind.common.contracts import Contract, ComputeModeConfig, Identifier, Nanoseconds
from .deployment import Positive, ServingCell
from .requests import Request, Migration
from .communication import CommPolicy
from astermind.common.hardware_capabilities import FAST_MODE_HARDWARE, PORTABLE_BATCH_HARDWARE

class CellLinkAddition(Contract):
    source: Identifier
    target: Identifier


class RoutingConfig(Contract):
    """Policy selection only; the engine owns routing refresh boundaries."""
    policy: str = Field(default="min_hops", pattern=r"^[a-z][a-z0-9_-]{0,127}$")


CellMetric = Literal["load", "distance", "arrival"]


class CellPriority(Contract):
    """Rank reachable Cells by the user's ordered criteria, then by Cell ID."""
    primary: CellMetric
    secondary: CellMetric | None = None

    @model_validator(mode="after")
    def distinct_criteria(self):
        if self.primary == self.secondary:
            raise ValueError("Cell 选择的首选和次选条件必须不同")
        return self


class SimulationConfig(ComputeModeConfig):
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
    communication: CommPolicy = Field(default_factory=CommPolicy)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    cell_selection: Literal['least_loaded', 'nearest', 'estimated_arrival'] | CellPriority = 'least_loaded'
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
        from .scenario_input import SCHEMA, lower_scenario

        if data.get("schema_version") == SCHEMA:
            from astermind.engine.policies.routing import get_routing_policy

            config = cls.model_validate(lower_scenario(data, path.parent))
            get_routing_policy(config.routing.policy)
            return config
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
        if self.compute_mode == "fast_mode":
            if self.migrations:
                raise ValueError("fast_mode does not support KV migrations")
            portable_scene = any(cell.hardware_id in PORTABLE_BATCH_HARDWARE for cell in self.cells)
            for cell in self.cells:
                if cell.detail_prediction_method is not None:
                    raise ValueError("per-Cell detail prediction requires Detail Mode")
                if cell.hardware_id not in FAST_MODE_HARDWARE:
                    raise ValueError(f"fast_mode has no hardware profile for {cell.hardware_id}")
                if cell.deployment != "colocated" or cell.parallel.sp != 1:
                    raise ValueError(f"fast_mode requires colocated SP1 in Cell {cell.id}")
                override = self.communication.cell_policies.get(cell.id)
                intra_mode = (override and override.intra_mode) or self.communication.intra_mode
                if intra_mode != "surrogate":
                    raise ValueError(f"fast_mode requires surrogate intra communication in Cell {cell.id}")
                if cell.parallel.dp > 1 and self.communication.ep_strategy != "direct":
                    raise ValueError(f"fast_mode requires direct EP in Cell {cell.id}")
                if (cell.hardware_id in PORTABLE_BATCH_HARDWARE
                        and cell.scheduler_adapter != "vllm_0_30"):
                    raise ValueError(f"{cell.hardware_id} fast_mode requires vllm_0_30 in Cell {cell.id}")
                if portable_scene:
                    comm_hardware = (override and override.hardware_id) or self.communication.default_hardware_id
                    for placement in cell.placements:
                        actual = self.communication.gpu_hardware.get(placement.device, comm_hardware)
                        if actual != cell.hardware_id:
                            raise ValueError(f"Cell {cell.id} GPU {placement.device} uses {actual} for Comm, expected {cell.hardware_id}")
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


# Preserve public serialization identities across the file-only move.
CellLinkAddition.__module__ = 'astermind.builder.config'
SimulationConfig.__module__ = 'astermind.builder.config'
