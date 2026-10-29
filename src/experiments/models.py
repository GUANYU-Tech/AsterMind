"""Experiment-level composition of the current public module contracts."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbitserve.architecture import HardwareSpec
from orbitserve.common import to_jsonable
from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.serving import ServingSpec
from orbitserve.workload import WorkloadBundleRef

from .constellation_cost import CostingSpec


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    physical_network: dict[str, Any] | None = None
    hardware: HardwareSpec | None = None
    serving: ServingSpec = field(default_factory=ServingSpec)
    workload: WorkloadBundleRef = field(default_factory=WorkloadBundleRef)
    runtime_strategy: RuntimeStrategySpec = field(default_factory=RuntimeStrategySpec)
    costing: CostingSpec = field(default_factory=CostingSpec)
    communication_hardware: HardwareSpec | None = None
    ground_baseline: dict | None = None
    dse: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    base_dir: str | None = None
    physical_dataset: str | None = None

    @classmethod
    def from_dict(cls, data, *, base_dir=None):
        allowed = set(cls.__dataclass_fields__) | {"schema_version"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                "Unsupported experiment fields: "
                + ", ".join(sorted(unknown))
                + "; use physical_network with current SGP4 constellation inputs"
            )
        network = deepcopy(data.get("physical_network"))
        if network is None and data.get('physical_dataset'):
            from .physical_dataset import load_dataset
            folder, _, _ = load_dataset(Path(base_dir or '.') / data['physical_dataset'], verify_exports=False)
            network = json.loads((folder / 'materialized_input.json').read_text(encoding='utf-8'))
        if data.get("ground_baseline") is None and (
            network is None or data.get("hardware") is None
        ):
            raise ValueError(
                "physical_network and hardware are required for satellite experiments"
            )
        if data.get("ground_baseline") is not None and network is not None:
            raise ValueError(
                "ground_baseline and physical_network are separate experiment modes"
            )
        network = network or {}
        network_base = Path(base_dir or ".")
        if isinstance(network, str):
            path = network_base / network
            network = json.loads(path.read_text(encoding="utf-8"))
            network_base = path.parent
        # Resolve file-backed sources at the file that declared them, including
        # per-member compute TLE input, without running physical calculations.
        from .network_inputs import snapshot_network_input
        network = snapshot_network_input(network, network_base)
        if network:
            from orbitserve.runtime import NetworkSpec, candidate_links

            from .network_inputs import materialize_network_input

            if data.get('physical_dataset'):
                from .physical_dataset import dataset_physical_input
                concrete = dataset_physical_input(data['physical_dataset'], network, base_dir or '.')
            else:
                concrete = materialize_network_input(network, network_base)
            allowed_network = {
                "schema_version",
                "time",
                "elements",
                "constellations",
                "network",
                "ground_ingresses",
                "route_probes",
                "initialization_provenance",
                "scenario_type",
            }
            if set(network) - allowed_network:
                raise ValueError(
                    "Unsupported physical network fields: "
                    + ", ".join(sorted(set(network) - allowed_network))
                )
            candidate_links(NetworkSpec.from_value(concrete["network"]))
        return cls(
            experiment_id=str(data["experiment_id"]),
            physical_network=network or None,
            hardware=HardwareSpec.from_dict(data["hardware"])
            if data.get("hardware")
            else None,
            ground_baseline=deepcopy(data.get("ground_baseline")),
            communication_hardware=HardwareSpec.from_dict(
                data["communication_hardware"]
            )
            if data.get("communication_hardware")
            else None,
            serving=ServingSpec.from_dict(data.get("serving")),
            workload=WorkloadBundleRef.from_dict(data.get("workload")),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                data.get("runtime_strategy")
            ),
            costing=CostingSpec.from_dict(data.get("costing")),
            dse=deepcopy(data.get("dse", {})),
            description=str(data.get("description", "")),
            base_dir=base_dir,
            physical_dataset=str(data['physical_dataset']) if data.get('physical_dataset') else None,
        )

    def to_dict(self):
        return to_jsonable(self)


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    manifest: dict
    architecture: dict
    serving: dict
    network: dict
    runtime: Any
    cost: dict
    dse: dict
    module_status: dict
    warnings: tuple[str, ...] = ()

    def to_dict(self):
        return {**to_jsonable(self), "runtime": self.runtime.to_dict()}
