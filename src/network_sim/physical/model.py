"""Physical topology and link configuration; time is integer nanoseconds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9_.:/-]+$")]
Nanoseconds = Annotated[int, Field(strict=True, ge=0, le=2**64 - 1)]
Bandwidth = Annotated[float, Field(ge=1e-6, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Cell(Contract):
    id: Identifier
    satellites: tuple[Identifier, ...]


class Satellite(Contract):
    id: Identifier
    # Static geometry only. None means unspecified, not a fabricated orbit.
    ecef_position_km: tuple[float, float, float] | None = None


class Device(Contract):
    id: Identifier
    satellite: Identifier
    kind: Literal["gpu", "switch", "terminal"]
    capacity_gbps: Bandwidth | None = None
    external_access: Annotated[bool, Field(strict=True)] = False
    max_external_links: Annotated[int, Field(strict=True, ge=0)] = 1

    @model_validator(mode="after")
    def validate_device(self):
        if self.kind == "gpu" and self.capacity_gbps is not None:
            raise ValueError("GPU access bandwidth belongs on links, not a forwarding gate")
        if self.kind != "terminal" and self.external_access:
            raise ValueError("external_access is a terminal capability")
        return self


class AvailabilityWindow(Contract):
    start_ns: Nanoseconds
    end_ns: Nanoseconds

    @model_validator(mode="after")
    def validate_window(self):
        if self.end_ns <= self.start_ns:
            raise ValueError("availability window must have start < end")
        return self


class OSReferenceLayout(Contract):
    """Explicit Walker reference relationships, not inferred orbital geometry."""
    plane_by_node: dict[Identifier, Annotated[int, Field(strict=True, ge=0)]]
    same_plane_neighbor_pairs: tuple[tuple[Identifier, Identifier], ...] = ()
    adjacent_plane_pairs: tuple[tuple[int, int], ...] = ()

    @model_validator(mode="after")
    def validate_pairs(self):
        for a, b in self.same_plane_neighbor_pairs:
            if a >= b or a not in self.plane_by_node or b not in self.plane_by_node:
                raise ValueError("O-S neighbor pairs need known IDs in canonical order")
            if self.plane_by_node[a] != self.plane_by_node[b]:
                raise ValueError("same-plane pair must be in the same plane")
        for a, b in self.adjacent_plane_pairs:
            if a >= b or a not in self.plane_by_node.values() or b not in self.plane_by_node.values():
                raise ValueError("adjacent planes need known plane IDs in canonical order")
        return self


class LinkPolicyConfig(Contract):
    mode: Literal["simple_keep", "os_reference"] = "simple_keep"
    check_interval_ns: Annotated[int, Field(strict=True, gt=0, le=2**64 - 1)]
    initial_acquisition_ns: Nanoseconds
    reacquisition_ns: Nanoseconds
    os_terminal_count: Annotated[int, Field(strict=True, gt=0)] = 4
    os_layout: OSReferenceLayout | None = None

    @model_validator(mode="after")
    def validate_layout_mode(self):
        if self.os_layout is not None and self.mode != "os_reference":
            raise ValueError("os_layout requires os_reference mode")
        return self


class Link(Contract):
    id: Identifier
    source: Identifier
    target: Identifier
    bandwidth_gbps: Bandwidth
    latency_ns: Nanoseconds = 0
    bidirectional: Annotated[bool, Field(strict=True)] = True
    enabled: Annotated[bool, Field(strict=True)] = True
    # None = always available; () = never available. Windows are [start, end).
    availability_windows: tuple[AvailabilityWindow, ...] | None = None
    distance_km: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def validate_windows(self):
        windows = self.availability_windows or ()
        if any(a.end_ns > b.start_ns for a, b in zip(windows, windows[1:])):
            raise ValueError("availability windows must be sorted and disjoint")
        return self


class Template(Contract):
    """Convenience generator over existing devices; never infers model groups."""

    id: Identifier
    kind: Literal["fc", "ring", "switch"]
    devices: tuple[Identifier, ...]
    switch: Identifier | None = None
    bandwidth_gbps: Bandwidth
    latency_ns: Nanoseconds = 0


class Scenario(Contract):
    schema_version: Literal[1] = 1
    cells: tuple[Cell, ...]
    satellites: tuple[Satellite, ...]
    devices: tuple[Device, ...]
    links: tuple[Link, ...] = ()
    templates: tuple[Template, ...] = ()
    link_policy: LinkPolicyConfig | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class OrbitConfiguration(Contract):
    # The existing O-S module produces the canonical, hash-verified trace.
    trace_manifest: str | None = None
    samples: tuple[dict, ...] = ()
    elements_file: str | None = None
    simulation_epoch_utc: str | None = None
    sample_step_ns: Annotated[int, Field(strict=True, gt=0)] = 1_000_000_000
    mode: Literal["per_member", "reference_offsets"] = "per_member"
    reference_satellite: Identifier | None = None
    offsets_itrs_km: dict[str, tuple[float, float, float]] = {}
    max_isl_range_km: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10000
    min_ground_elevation_deg: Annotated[float, Field(ge=-90, le=90)] = 0

    @model_validator(mode="after")
    def validate_source(self):
        if sum((self.trace_manifest is not None, bool(self.samples), self.elements_file is not None)) != 1:
            raise ValueError("supply one orbit source: canonical manifest, samples, or SGP4 elements")
        if self.elements_file and not self.simulation_epoch_utc:
            raise ValueError("SGP4 propagation needs simulation_epoch_utc")
        if self.mode == "reference_offsets" and not self.reference_satellite:
            raise ValueError("reference-offset formation needs a reference satellite")
        if self.mode == "per_member" and (self.reference_satellite or self.offsets_itrs_km):
            raise ValueError("independent propagation cannot contain reference offsets")
        return self


class LinkParameters(Contract):
    kind: Literal["manual", "islsim"] = "manual"
    distance_km: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    fso: dict = {}


