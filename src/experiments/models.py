"""Experiment manifest and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orbitserve.common import to_jsonable
from orbitserve.orbit import ConstellationDesign
from orbitserve.experiments.constellation_cost import ConstellationCostResult, CostingSpec
from orbitserve.dse.search import DSESpec, DSEStudyResult
from orbitserve.architecture import HardwareSpec
from orbitserve.common import PerformanceMetrics, SLOMetrics
from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.serving import ServingSpec
from orbitserve.orbit import SimulationConfig
from orbitserve.workload import WorkloadBundleRef


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    constellation: ConstellationDesign
    costing: CostingSpec
    workload: WorkloadBundleRef = field(default_factory=WorkloadBundleRef)
    serving: ServingSpec = field(default_factory=ServingSpec)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    scheduler: RuntimeStrategySpec = field(default_factory=lambda: RuntimeStrategySpec.from_value(None))
    dse: DSESpec = field(default_factory=DSESpec)
    hardware: HardwareSpec | None = None
    description: str = ""
    base_dir: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: str | None = None) -> "ExperimentSpec":
        return cls(
            experiment_id=str(data["experiment_id"]),
            description=str(data.get("description", "")),
            constellation=ConstellationDesign.from_dict(data["constellation"]),
            costing=CostingSpec.from_dict(data.get("costing")),
            hardware=HardwareSpec.from_dict(data["hardware"]) if data.get("hardware") else None,
            workload=WorkloadBundleRef.from_dict(data.get("workload")),
            serving=ServingSpec.from_dict(data.get("serving")),
            simulation=SimulationConfig.from_dict(data.get("simulation")),
            scheduler=RuntimeStrategySpec.from_dict(data.get("scheduler")),
            dse=DSESpec.from_dict(data.get("dse")),
            base_dir=base_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "constellation": self.constellation.to_dict(),
            "costing": self.costing.to_dict(),
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "workload": self.workload.to_dict(),
            "serving": self.serving.to_dict(),
            "simulation": self.simulation.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "dse": self.dse.to_dict(),
            "base_dir": self.base_dir,
        }


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    manifest: dict[str, Any]
    constellation_architecture: dict[str, Any]
    cost: ConstellationCostResult
    performance: PerformanceMetrics
    slo: SLOMetrics
    dse: DSEStudyResult
    module_status: dict[str, str]
    warnings: tuple[str, ...]
    workload_generation: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_kind": "fixed_design_simulation",
            "experiment_id": self.experiment_id,
            "manifest": to_jsonable(self.manifest),
            "constellation_architecture": to_jsonable(self.constellation_architecture),
            "cost": self.cost.to_dict(),
            "performance": self.performance.to_dict(),
            "slo": self.slo.to_dict(),
            "dse": self.dse.to_dict(),
            "module_status": to_jsonable(self.module_status),
            "warnings": list(self.warnings),
            "workload_generation": _jsonable_result(self.workload_generation),
        }


def _jsonable_result(value: Any) -> Any:
    if value is not None and hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    return to_jsonable(value)
