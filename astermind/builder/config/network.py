"""Physical network inputs. Roles and terminal capabilities belong to Runtime."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

from astermind.common import to_jsonable


@dataclass(frozen=True)
class NetworkNodeSpec:
    node_id: str
    role: str
    isl_terminals: int = 0
    sgl_terminals: int = 0
    plane: int | None = None
    slot: int | None = None
    cell_id: str | None = None
    cell_ingress: bool = False
    compute_target: bool = True
    shell_id: str | None = None
    # Hardware generation is independent of orbital shell/Walker configuration.
    radio_profile: str | None = None
    radio_capacity_bps: dict[str, float] = field(default_factory=dict)
    capacity_assumptions: tuple[str, ...] = ()
    ground_kind: str = 'gateway'
    optical_terminal_bandwidth_bps: float | None = None
    terminal_assumptions: tuple[str, ...] = ()
    sgl_profiles: dict[str, str] = field(default_factory=dict)
    # Concurrent ground-node associations, independent of aggregate RF rates.
    max_user_links: int | None = None
    max_gateway_links: int | None = None

    def __post_init__(self) -> None:
        for key in ('max_user_links', 'max_gateway_links'):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{key} must be a nonnegative integer or null')
            if value is not None and self.role == 'ground':
                raise ValueError(f'{key} is a satellite association limit')
        from astermind.engine.setup.radio_capacity import STARLINK_RADIO_PROFILES, CAPACITY_FIELDS
        if not set(self.sgl_profiles) <= {'user_uplink', 'user_downlink', 'feeder_uplink', 'feeder_downlink'} or any(not isinstance(v, str) or not v for v in self.sgl_profiles.values()):
            raise ValueError('sgl_profiles requires user/feeder uplink/downlink profile IDs')
        object.__setattr__(self, 'sgl_profiles', dict(self.sgl_profiles))
        if self.ground_kind not in {'user', 'gateway'}:
            raise ValueError('ground_kind must be user or gateway')
        if self.radio_profile is not None and self.radio_profile not in STARLINK_RADIO_PROFILES:
            raise ValueError('radio_profile must name a versioned Starlink profile')
        if self.radio_profile and self.role != 'communication':
            raise ValueError('Starlink radio_profile requires a communication satellite')
        if self.radio_profile and self.isl_terminals > STARLINK_RADIO_PROFILES[self.radio_profile]['terminals']['optical_count']:
            raise ValueError('isl_terminals exceeds selected Starlink optical terminal count; use a custom scenario for other hardware')
        if self.optical_terminal_bandwidth_bps is not None:
            if not math.isfinite(self.optical_terminal_bandwidth_bps) or self.optical_terminal_bandwidth_bps <= 0:
                raise ValueError('optical_terminal_bandwidth_bps must be finite and positive')
            if not self.terminal_assumptions:
                raise ValueError('per-direction optical ceiling requires explicit terminal_assumptions')
        if isinstance(self.terminal_assumptions, str) or any(not isinstance(s, str) or not s.strip() for s in self.terminal_assumptions):
            raise ValueError('terminal_assumptions requires nonempty strings')
        object.__setattr__(self, 'terminal_assumptions', tuple(self.terminal_assumptions))
        if not set(self.radio_capacity_bps) <= CAPACITY_FIELDS or any(
            not math.isfinite(v) or v <= 0 for v in self.radio_capacity_bps.values()
        ):
            raise ValueError('radio_capacity_bps requires known positive finite aggregate capacities')
        if isinstance(self.capacity_assumptions, str) or any(not isinstance(s, str) or not s.strip() for s in self.capacity_assumptions):
            raise ValueError('capacity_assumptions requires a sequence of nonempty strings')
        if self.radio_capacity_bps and not self.capacity_assumptions:
            raise ValueError('radio capacity overrides require capacity_assumptions')
        object.__setattr__(self, 'radio_capacity_bps', dict(self.radio_capacity_bps))
        object.__setattr__(self, 'capacity_assumptions', tuple(self.capacity_assumptions))
        if not self.node_id or self.role not in {"ground", "communication", "compute"}:
            raise ValueError(
                "network nodes require an ID and ground/communication/compute role"
            )
        for key in ("isl_terminals", "sgl_terminals"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("cell_ingress", "compute_target"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("plane", "slot"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{key} must be a nonnegative integer")


@dataclass(frozen=True)
class LinkProfile:
    bandwidth_bps: float
    setup_time_s: float
    max_distance_km: float | None = None
    # Piecewise capacity multipliers, applied at absolute simulation times.
    capacity_schedule: tuple[tuple[float, float], ...] = ()
    capacity_model: str = "fixed"
    channel_bandwidth_hz: float | None = None
    frequency_hz: float | None = None
    transmit_power_w: float | None = None
    transmit_gain_db: float = 0.0
    receive_gain_db: float = 0.0
    noise_temperature_k: float = 290.0
    system_loss_db: float = 0.0
    implementation_efficiency: float = 1.0
    sun_exclusion_angle_deg: float | None = None
    wavelength_m: float | None = None
    transmit_aperture_m: float | None = None
    receive_aperture_m: float | None = None
    transmit_aperture_efficiency: float = 0.6
    receive_aperture_efficiency: float = 0.6
    beam_divergence_rad: float | None = None
    required_photons_per_bit: float | None = None
    pointing_loss_db: float = 0.0
    atmospheric_loss_db: float = 0.0
    parameter_sources: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    # Optional directional RF budget profiles; geometry/setup remain on this profile.
    uplink_profile: str | None = None
    downlink_profile: str | None = None
    # Portable TX/RX/channel input; local physical preparation binds source hashes.
    islsim_config: dict[str, Any] = field(default_factory=dict)
    islsim_source_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("bandwidth_bps", "setup_time_s"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.bandwidth_bps == 0:
            raise ValueError("bandwidth_bps must be positive")
        if self.max_distance_km is not None and (
            not math.isfinite(self.max_distance_km) or self.max_distance_km <= 0
        ):
            raise ValueError("max_distance_km must be finite and positive")
        schedule = tuple(tuple(row) for row in self.capacity_schedule)
        if any(
            len(row) != 2
            or not all(math.isfinite(v) for v in row)
            or row[0] < 0
            or not 0 <= row[1] <= 1
            for row in schedule
        ):
            raise ValueError(
                "capacity_schedule requires nonnegative finite time/multiplier pairs"
            )
        if any(a[0] >= b[0] for a, b in zip(schedule, schedule[1:])):
            raise ValueError("capacity_schedule must have strictly increasing times")
        object.__setattr__(self, "capacity_schedule", schedule)
        if self.capacity_model not in {"fixed", "free_space_shannon", "optical_photon_budget", "islsim"}:
            raise ValueError("unsupported capacity_model")
        for key in ("transmit_gain_db", "receive_gain_db", "system_loss_db"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"invalid {key}")
        if (
            not math.isfinite(self.implementation_efficiency)
            or not 0 < self.implementation_efficiency <= 1
        ):
            raise ValueError("implementation_efficiency must be in (0,1]")
        if self.capacity_model == "free_space_shannon":
            for key in (
                "channel_bandwidth_hz",
                "frequency_hz",
                "transmit_power_w",
                "noise_temperature_k",
            ):
                value = getattr(self, key)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"free_space_shannon requires positive {key}")
        if self.sun_exclusion_angle_deg is not None and (
            not math.isfinite(self.sun_exclusion_angle_deg)
            or not 0 <= self.sun_exclusion_angle_deg <= 180
        ):
            raise ValueError("invalid sun_exclusion_angle_deg")
        for key in ("wavelength_m", "transmit_aperture_m", "receive_aperture_m", "beam_divergence_rad", "required_photons_per_bit"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("transmit_aperture_efficiency", "receive_aperture_efficiency"):
            if not math.isfinite(getattr(self, key)) or not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        for key in ("pointing_loss_db", "atmospheric_loss_db"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.capacity_model == "optical_photon_budget":
            for key in ("wavelength_m", "transmit_aperture_m", "receive_aperture_m", "transmit_power_w", "required_photons_per_bit"):
                value = getattr(self, key)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"optical_photon_budget requires positive {key}")
        for key in ("parameter_sources", "assumptions"):
            values = getattr(self, key)
            if isinstance(values, str) or any(not isinstance(v, str) for v in values):
                raise ValueError(f"{key} must be a sequence of strings")
            object.__setattr__(self, key, tuple(values))
        if self.capacity_model == 'islsim':
            from astermind.engine.setup.islsim_config import prepare_config
            object.__setattr__(self, 'islsim_config', prepare_config(self.islsim_config))
            object.__setattr__(self, 'islsim_source_sha256', dict(self.islsim_source_sha256))
            object.__setattr__(self, '_islsim_binding', None)

    def bandwidth_at(self, time_s: float) -> float:
        multiplier = 1.0
        for start, value in self.capacity_schedule:
            if start > time_s:
                break
            multiplier = value
        return self.bandwidth_bps * multiplier

    def attributes(self, distance_km: float, time_s: float) -> dict:
        from astermind.engine.setup.link_budget import calculate_budget

        result = calculate_budget(self, distance_km)
        multiplier = self.bandwidth_at(time_s) / self.bandwidth_bps
        return {**result, "bandwidth_bps": result["budget_bandwidth_bps"] * multiplier}



@dataclass(frozen=True)
class NetworkSpec:
    nodes: tuple[NetworkNodeSpec, ...]
    link_profiles: dict[str, LinkProfile]
    communication_topology: str = "grid"
    wrap_planes: bool = True
    wrap_slots: bool = True
    cross_plane_offset: int = 0
    cell_topology: str = "ring"
    inter_cell_links: bool = False
    ground_roles: tuple[str, ...] = ("communication",)
    explicit_pairs: tuple[tuple[str, str], ...] = ()
    ground_access_policy: str = "nearest_distance"
    communication_shells: tuple[dict, ...] = ()
    cell_topologies: dict[str, str] = field(default_factory=dict)
    cell_link_overrides: dict[str, list[dict]] = field(default_factory=dict)
    transport_model: str = "runtime_comm"
    # Unexpected outages: revealed only at start_s to online Runtime routing.
    faults: tuple[dict, ...] = ()
    failure_policy: str = "fail"
    max_transfer_retries: int = 8
    ground_matching: str = 'greedy'
    satellite_access_policy: str = 'nearest_distance'
    cell_egress_policies: dict[str, str] = field(default_factory=dict)
    communication_grid_shape: tuple[int, int] | None = None
    topology_repair: str = 'none'
    handover_policy: str = 'retain_valid'
    cross_layer_access_policy: str = 'nearest_distance'
    fault_timing: str = 'continuous'
    random_faults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from astermind.engine.setup.faults import validate_random_faults
        if self.fault_timing not in {'boundary', 'continuous'}:
            raise ValueError('fault_timing must be boundary or continuous')
        validate_random_faults(self.random_faults)
        object.__setattr__(self, 'random_faults', deepcopy(self.random_faults))
        if self.handover_policy not in {'retain_valid','quality_first'}:
            raise ValueError('unsupported handover_policy')
        if self.cross_layer_access_policy not in {'nearest_distance','highest_capacity','longest_available'}:
            raise ValueError('unsupported cross_layer_access_policy')
        if self.topology_repair not in {'none', 'ronchetti_intra_orbit'}:
            raise ValueError('unsupported topology_repair')
        if self.communication_grid_shape is not None:
            shape = tuple(self.communication_grid_shape)
            if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError('communication_grid_shape requires two positive integers')
            object.__setattr__(self, 'communication_grid_shape', shape)
        if self.ground_matching not in {'greedy', 'deferred_acceptance'}:
            raise ValueError('unsupported ground_matching')
        policies = {'nearest_distance', 'longest_available', 'highest_elevation', 'highest_capacity'}
        if self.satellite_access_policy not in policies:
            raise ValueError('unsupported satellite_access_policy')
        if any(v not in {'single', 'multi', 'free'} for v in self.cell_egress_policies.values()):
            raise ValueError('unsupported cell egress policy')
        object.__setattr__(self, 'cell_egress_policies', dict(self.cell_egress_policies))
        if self.failure_policy not in {"fail", "retry_next_sample"}:
            raise ValueError("unsupported failure_policy")
        if type(self.max_transfer_retries) is not int or self.max_transfer_retries < 0:
            raise ValueError("max_transfer_retries must be a nonnegative integer")
        if self.transport_model != "runtime_comm":
            raise ValueError("retired transport_model; use runtime_comm and SimulationConfig.communication")
        nodes = tuple(
            n if isinstance(n, NetworkNodeSpec) else NetworkNodeSpec(**n)
            for n in self.nodes
        )
        nodes = tuple(replace(n, cell_ingress=True)
                      if n.cell_id and self.cell_egress_policies.get(n.cell_id) == 'free' else n
                      for n in nodes)
        from astermind.engine.setup.islsim import freeze_profiles
        profile_values = freeze_profiles({'nodes': self.nodes, 'link_profiles': self.link_profiles})['link_profiles']
        profiles = {
            k: v if isinstance(v, LinkProfile) else LinkProfile(**v)
            for k, v in profile_values.items()
        }
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "link_profiles", profiles)
        owners = {node.node_id: node.cell_id for node in nodes if node.role == 'compute'}
        if not set(self.cell_link_overrides) <= set(owners.values()):
            raise ValueError('cell_link_overrides references an unknown Cell')
        for cell, rows in self.cell_link_overrides.items():
            pairs = set()
            for edge in rows:
                if set(edge) != {'source', 'target', 'profile_id'} or edge['profile_id'] not in profiles:
                    raise ValueError('Cell links require source, target and a registered profile')
                a, b = edge['source'], edge['target']
                pair = frozenset((a, b))
                if a == b or owners.get(a) != cell or owners.get(b) != cell or pair in pairs:
                    raise ValueError(f'Cell {cell} has invalid or repeated link endpoints')
                pairs.add(pair)
        for node in nodes:
            for name in node.sgl_profiles.values():
                if name not in profiles or profiles[name].uplink_profile or profiles[name].downlink_profile:
                    raise ValueError('node sgl_profiles must reference an existing nonrecursive profile')
        for profile in profiles.values():
            for name in (profile.uplink_profile, profile.downlink_profile):
                if name is not None and (name not in profiles or
                    profiles[name].uplink_profile or profiles[name].downlink_profile):
                    raise ValueError('directional RF profiles must reference a nonrecursive link profile')
        object.__setattr__(self, "ground_roles", tuple(self.ground_roles))
        object.__setattr__(
            self, "explicit_pairs", tuple(tuple(pair) for pair in self.explicit_pairs)
        )
        if len({n.node_id for n in nodes}) != len(nodes):
            raise ValueError("duplicate network node ID")
        for fault in self.faults:
            if (not {"target_type", "target_id", "start_s", "end_s"} <= set(fault)
                or not set(fault) <= {"target_type", "target_id", "start_s", "end_s", "source"}
                or fault.get('source', 'scheduled') not in {'scheduled', 'random', 'manual'}
                or fault["target_type"] not in {"node", "link"}
                or not isinstance(fault["target_id"], str)
                or not all(math.isfinite(fault[k]) for k in ("start_s", "end_s"))
                or not 0 <= fault["start_s"] < fault["end_s"]):
                raise ValueError("fault requires node/link target and finite start_s < end_s")
            if fault["target_type"] == "node" and fault["target_id"] not in {n.node_id for n in nodes}:
                raise ValueError("unknown fault node")
        object.__setattr__(self, "faults", tuple(dict(f) for f in self.faults))
        object.__setattr__(
            self,
            "communication_shells",
            tuple(dict(row) for row in self.communication_shells),
        )
        shell_ids = [row["shell_id"] for row in self.communication_shells]
        if len(set(shell_ids)) != len(shell_ids):
            raise ValueError("duplicate communication shell ID")
        if self.communication_shells:
            actual = {n.shell_id for n in nodes if n.role == "communication"}
            if actual != set(shell_ids):
                raise ValueError(
                    "communication nodes must belong to exactly one configured shell"
                )
        cell_topologies = {"ring", "chain", "fully_connected"}
        if any(v not in cell_topologies for v in self.cell_topologies.values()):
            raise ValueError("unsupported per-cell topology")
        if self.communication_topology not in {"grid", "honeycomb", "explicit"}:
            raise ValueError(
                "communication_topology must be grid, honeycomb, or explicit"
            )
        if self.ground_access_policy not in policies:
            raise ValueError(
                "unsupported ground_access_policy"
            )
        if self.cell_topology not in cell_topologies:
            raise ValueError("physical cell_topology must be ring, chain, or fully_connected")
        for key in (
            "wrap_planes",
            "wrap_slots",
            "inter_cell_links",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if type(self.cross_plane_offset) is not int:
            raise ValueError("cross_plane_offset must be an integer")
        if not set(self.ground_roles) <= {"compute", "communication"}:
            raise ValueError("invalid ground_roles")
        cells: dict[str, list[NetworkNodeSpec]] = {}
        for node in nodes:
            if node.cell_id:
                if node.role != "compute":
                    raise ValueError("only compute satellites may belong to a cell")
                cells.setdefault(node.cell_id, []).append(node)
        for cell, members in cells.items():
            mode = self.cell_egress_policies.get(cell, 'multi')
            if mode == 'single' and sum(n.cell_ingress for n in members) != 1:
                raise ValueError(f'cell {cell}: single egress requires exactly one ingress member')
            if sum(n.compute_target for n in members) != 1:
                raise ValueError(
                    f"cell {cell} requires exactly one logical compute target"
                )
            if mode != 'free' and not any(n.cell_ingress for n in members):
                raise ValueError(f"cell {cell} requires an ingress member")
        if set(self.cell_egress_policies) - cells.keys():
            raise ValueError('cell_egress_policies references unknown cell')

    def external_member(self, node: NetworkNodeSpec) -> bool:
        return (not node.cell_id or node.cell_ingress
                or self.cell_egress_policies.get(node.cell_id) == 'free')

    @classmethod
    def from_value(cls, value: Any) -> NetworkSpec | None:
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("network must be a mapping")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

# Preserve public serialization identities across the file-only move.
NetworkNodeSpec.__module__ = 'astermind.engine.setup.network_spec'
LinkProfile.__module__ = 'astermind.engine.setup.network_spec'
NetworkSpec.__module__ = 'astermind.engine.setup.network_spec'
