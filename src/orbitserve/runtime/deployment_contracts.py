"""Executable serving-deployment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common import to_jsonable

@dataclass(frozen=True)
class RuntimeDeployment:
    deployment_id: str
    role: str = "colocated"
    member_satellites: tuple[str, ...] = ()
    entry_satellite_ids: tuple[str, ...] = ()
    member_energy_weights: dict[str, float] = field(default_factory=dict)
    placement_scope: str = "aggregate_logical_pool"
    prefill_tokens_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    request_throughput_per_s: float | None = None
    per_group_prefill_tokens_per_s: float | None = None
    per_group_decode_tokens_per_s: float | None = None
    kv_cache_capacity_tokens: int | None = None
    kv_cache_bytes_per_token: float | None = None
    satellites_per_deployment: int | None = None
    stage_satellite_count: int | None = None
    internal_comm_accounting: str = "embedded"
    pd_mode: str | None = None
    placement_model: str = "aggregate_logical_pool"
    selection_policy: str = "fixed_default_deployment"
    plane_index: int | None = None
    cell_index: int | None = None
    cluster_id: str | None = None
    compute_cell_id: str | None = None
    cluster_member_indices: tuple[int, ...] = ()
    paired_deployment_ids: tuple[str, ...] = ()
    paired_deployment_scopes: dict[str, str] = field(default_factory=dict)
    paired_deployment_rules: dict[str, str] = field(default_factory=dict)
    valid_for_paper_main_results: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def has_compute_capacity(self) -> bool:
        return bool(
            self.prefill_tokens_per_s
            and self.prefill_tokens_per_s > 0
            and self.decode_tokens_per_s
            and self.decode_tokens_per_s > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class RuntimeDeploymentCatalog:
    deployments: tuple[RuntimeDeployment, ...] = ()
    placement_model: str = "aggregate_logical_pool"
    selection_policy: str = "fixed_default_deployment"
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def default_deployment(self) -> RuntimeDeployment | None:
        return self.deployments[0] if self.deployments else None

    @property
    def prefill_deployments(self) -> tuple[RuntimeDeployment, ...]:
        return self.deployments_for_role("prefill")

    @property
    def decode_deployments(self) -> tuple[RuntimeDeployment, ...]:
        return self.deployments_for_role("decode")

    @property
    def colocated_deployments(self) -> tuple[RuntimeDeployment, ...]:
        return self.deployments_for_role("colocated")

    def deployments_for_role(self, role: str) -> tuple[RuntimeDeployment, ...]:
        return tuple(deployment for deployment in self.deployments if deployment.role == role)

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(
            {
                "deployments": self.deployments,
                "placement_model": self.placement_model,
                "selection_policy": self.selection_policy,
                "warnings": self.warnings,
                "provenance": self.provenance,
            }
        )
        payload["default_deployment"] = (
            self.default_deployment.to_dict() if self.default_deployment is not None else None
        )
        payload["prefill_deployments"] = [item.to_dict() for item in self.prefill_deployments]
        payload["decode_deployments"] = [item.to_dict() for item in self.decode_deployments]
        payload["colocated_deployments"] = [item.to_dict() for item in self.colocated_deployments]
        return payload
