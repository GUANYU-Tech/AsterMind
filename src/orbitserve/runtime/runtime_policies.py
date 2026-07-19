"""Runtime policy implementations for the event evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import RuntimeAction, RuntimePredictionEnvelope, RuntimeStrategySpec


class RuntimePolicy(Protocol):
    name: str

    def select_action(self, state: Any, request: dict[str, Any], candidates: tuple[Any, ...]) -> RuntimeAction:
        """Select a runtime action for a request."""


@dataclass(frozen=True)
class LocalOnlyRuntimePolicy:
    name: str = "local_only"

    def select_action(self, state: Any, request: dict[str, Any], candidates: tuple[Any, ...]) -> RuntimeAction:
        del state
        deployment = candidates[0] if candidates else None
        request_id = _optional_str(request.get("request_id"))
        if deployment is None:
            return RuntimeAction(
                action_id=f"{request_id or 'request'}:reject_no_deployment",
                request_id=request_id,
                action_type="reject_drop",
                prediction=RuntimePredictionEnvelope(
                    bottleneck_reason="no_deployment",
                    warnings=("no runtime deployment candidate is available",),
                ),
                provenance={"policy": self.name},
            )
        return RuntimeAction(
            action_id=f"{request_id or 'request'}:local_compute",
            request_id=request_id,
            action_type="local_compute",
            serving_deployment=deployment.deployment_id,
            prefill_deployment=deployment.deployment_id,
            decode_deployment=deployment.deployment_id,
            prediction=RuntimePredictionEnvelope(evaluated=False, warnings=()),
            provenance={
                "policy": self.name,
                "deployment_selection": deployment.selection_policy,
                "placement_model": deployment.placement_model,
                "nearest_semantics": "not_available_without_access_trace",
            },
        )


def get_runtime_policy(strategy: RuntimeStrategySpec | dict[str, Any] | Any) -> RuntimePolicy | None:
    runtime_strategy = (
        strategy
        if isinstance(strategy, RuntimeStrategySpec)
        else RuntimeStrategySpec.from_value(strategy)
    )
    if runtime_strategy.policy == "local_only":
        return LocalOnlyRuntimePolicy()
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = ["RuntimePolicy", "LocalOnlyRuntimePolicy", "get_runtime_policy"]
