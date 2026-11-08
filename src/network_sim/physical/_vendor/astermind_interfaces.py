from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InterfaceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtensibleInterfaceModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class Workload(InterfaceModel):
    output_directory: Path
    workload_base: Path
    communication_group: Path
    et_files: tuple[Path, ...]
    rank_ids: tuple[int, ...]
    chakra_schema_version: str


class SatelliteState(InterfaceModel):
    satellite_id: str
    name: str
    timestamp: datetime
    teme_position_km: tuple[float, float, float] | None = None
    teme_velocity_km_s: tuple[float, float, float] | None = None
    # Synthetic formations are fixed spherical-Earth snapshots, not TLE orbits.
    ecef_position_km: tuple[float, float, float] | None = None
    latitude_deg: float
    longitude_deg: float
    altitude_km: float

    @model_validator(mode="after")
    def validate_position_frame(self) -> SatelliteState:
        if self.teme_position_km is None and self.ecef_position_km is None:
            raise ValueError("satellite state requires a TEME or ECEF position")
        return self


class CandidateInterSatelliteLink(InterfaceModel):
    source_satellite: str
    target_satellite: str
    distance_km: float = Field(gt=0)

    @model_validator(mode="after")
    def reject_self_link(self) -> CandidateInterSatelliteLink:
        if self.source_satellite == self.target_satellite:
            raise ValueError("inter-satellite links cannot be self-links")
        return self


class EnrichedInterSatelliteLink(CandidateInterSatelliteLink):
    bandwidth_gbps: float = Field(ge=0)
    latency_ns: int = Field(ge=0)
    snr_db: float
    modulation: str
    bit_error_rate: float = Field(ge=0, le=1)
    feasible: bool
    limiting_reason: str = ""
    baud_rate_ghz: float = Field(default=0, ge=0)
    raw_bandwidth_gbps: float = Field(default=0, ge=0)
    terminal_limited: bool = False


class OrbitSnapshot(InterfaceModel):
    timestamp: datetime
    satellites: list[SatelliteState]
    links: list[CandidateInterSatelliteLink]


class NetworkNode(InterfaceModel):
    id: int = Field(ge=0)
    kind: Literal["accelerator", "switch", "isl_terminal"]
    satellite_id: str
    local_index: int | None = Field(default=None, ge=0)
    accelerator_id: int | None = Field(default=None, ge=0)


class NetworkLink(InterfaceModel):
    id: str
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    level: Literal["intra_satellite", "inter_satellite"]
    bandwidth_gbps: float = Field(gt=0)
    latency_ns: int = Field(ge=0)
    distance_km: float | None = Field(default=None, gt=0)
    snr_db: float | None = None
    bit_error_rate: float = Field(default=0.0, ge=0, le=1)
    modulation: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def reject_self_link(self) -> NetworkLink:
        if self.source == self.target:
            raise ValueError("network links cannot be self-links")
        return self


class Topology(InterfaceModel):
    timestamp: datetime
    satellites: list[SatelliteState]
    nodes: list[NetworkNode]
    links: list[NetworkLink]
    accelerator_to_satellite: dict[int, str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_local_references(self) -> Topology:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("topology node IDs must be unique")
        node_id_set = set(node_ids)
        link_ids: set[str] = set()
        undirected_edges: set[tuple[int, int]] = set()
        for link in self.links:
            if link.id in link_ids:
                raise ValueError(f"duplicate topology link ID: {link.id}")
            link_ids.add(link.id)
            if link.source not in node_id_set or link.target not in node_id_set:
                raise ValueError(f"link {link.id} references an unknown node")
            edge = tuple(sorted((link.source, link.target)))
            if edge in undirected_edges:
                raise ValueError(f"duplicate undirected topology edge: {edge}")
            undirected_edges.add(edge)
        accelerator_nodes = {
            node.accelerator_id: node
            for node in self.nodes
            if node.kind == "accelerator"
        }
        if None in accelerator_nodes:
            raise ValueError("accelerator nodes must define accelerator_id")
        if set(accelerator_nodes) != set(self.accelerator_to_satellite):
            raise ValueError(
                "accelerator_to_satellite must cover every accelerator node"
            )
        for accelerator_id, satellite_id in self.accelerator_to_satellite.items():
            if accelerator_nodes[accelerator_id].satellite_id != satellite_id:
                raise ValueError("accelerator_to_satellite contains a mismatch")
        return self


class SimulationResult(ExtensibleInterfaceModel):
    status: Literal["success", "failed"]
    simulator: Literal["ASTRA-SIM"] = "ASTRA-SIM"
    network_backend: Literal["ns-3", "FlowSim"] = "ns-3"
    metrics: dict[str, Any]
    derived_metrics: dict[str, Any] = Field(default_factory=dict)
    versions: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
