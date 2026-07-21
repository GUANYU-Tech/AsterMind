"""Executable serving-deployment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable

@dataclass(frozen=True)
class RuntimeDeployment:
    deployment_id: str
    member_satellites: tuple[str, ...] = ()
    entry_satellite_ids: tuple[str, ...] = ()
    prefill_tokens_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    request_throughput_per_s: float | None = None
    kv_cache_capacity_tokens: int | None = None
    kv_cache_bytes_per_token: float | None = None
    satellites_per_deployment: int | None = None
    internal_comm_accounting: str = "embedded"
    placement_model: str = "aggregate_logical_pool"
    selection_policy: str = "fixed_default_deployment"
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
        return payload
