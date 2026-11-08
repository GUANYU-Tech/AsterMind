"""Compatibility imports for the public physical and transfer contracts."""

from .comm.model import CompletionEvent, FailureEvent, FlowStatus, TransferRequest
from .physical.model import (
    AvailabilityWindow, Bandwidth, Cell, Contract, Device, Identifier, Link,
    LinkPolicyConfig, Nanoseconds, OSReferenceLayout, Satellite, Scenario, Template,
)

__all__ = [
    "AvailabilityWindow", "Bandwidth", "Cell", "CompletionEvent", "Contract",
    "Device", "FailureEvent", "FlowStatus", "Identifier", "Link",
    "LinkPolicyConfig", "Nanoseconds", "OSReferenceLayout", "Satellite",
    "Scenario", "Template", "TransferRequest",
]
