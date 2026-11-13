"""Compatibility imports for physical resource configuration."""
from typing import Annotated, Literal
from pydantic import Field
from astermind.common.contracts import Contract, Identifier, Nanoseconds, Bandwidth
from astermind.builder.config.resources import (
    Cell,
    Satellite,
    Device,
    AvailabilityWindow,
    OSReferenceLayout,
    LinkPolicyConfig,
    Link,
    Template,
    Scenario,
)

__all__ = ['Cell', 'Satellite', 'Device', 'AvailabilityWindow', 'OSReferenceLayout', 'LinkPolicyConfig', 'Link', 'Template', 'Scenario']
