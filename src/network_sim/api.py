"""Public interface shared by the checkout and packaged layouts."""

from .comm.model import CompletionEvent, FailureEvent, TransferRequest
from .comm.runtime import CommunicationRuntime
from .physical.model import LinkPolicyConfig, Scenario
from .physical.topology import PhysicalTopology

__all__ = [
    "Scenario", "TransferRequest", "PhysicalTopology", "CommunicationRuntime",
    "CompletionEvent", "FailureEvent", "LinkPolicyConfig",
]
