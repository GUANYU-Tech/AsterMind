"""Constellation architecture dataclasses and lightweight analysis helpers.

The model keeps the original shell-based manifest shape working, while adding
the clustered compute-cell design vector described in the research notes.  The
analysis here is intentionally a fast, auditable surrogate: it derives geometry,
topology, coverage, shadow geometry checks, and feasibility flags, but it does not
pretend to replace STK calibration, dynamic link scheduling, or queueing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

from ..common import to_jsonable


EARTH_RADIUS_KM = 6371.0
EARTH_MU_KM3_S2 = 398600.4418
EARTH_ROTATION_RAD_PER_S = 7.2921159e-5
C_LIGHT_KM_PER_S = 299792.458

def _pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalise_mode(value: Any) -> str:
    return str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")


def _nonnegative_finite_number(value: Any, *, field_path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_path} must be a finite non-negative number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field_path} must be a finite non-negative number")
    return numeric


def _nonnegative_integer(value: Any, *, field_path: str) -> int | None:
    numeric = _nonnegative_finite_number(value, field_path=field_path)
    if numeric is None:
        return None
    if not numeric.is_integer():
        raise ValueError(f"{field_path} must be a non-negative integer")
    return int(numeric)


def _validate_gateway_profile(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("constellation.gateway_profile must be a mapping")
    profile = dict(value)
    if "bandwidth_bps" in profile:
        _nonnegative_finite_number(
            profile["bandwidth_bps"],
            field_path="constellation.gateway_profile.bandwidth_bps",
        )
    if "terminal_count_per_satellite" in profile:
        terminal_count = _nonnegative_integer(
            profile["terminal_count_per_satellite"],
            field_path="constellation.gateway_profile.terminal_count_per_satellite",
        )
        if terminal_count is not None:
            profile["terminal_count_per_satellite"] = terminal_count
    return profile


def _validate_orbital_set_satellite_profile(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("orbital_set.satellite_profile must be a mapping")
    profile = dict(value)
    if "link_profiles" in profile:
        raise ValueError(
            "orbital_set.satellite_profile.link_profiles is no longer supported; "
            "define ISL physical parameters in orbital_set.cluster"
        )
    return profile


def orbital_period_s(altitude_km: float, *, earth_radius_km: float = EARTH_RADIUS_KM) -> float:
    """Return circular-orbit period for the lightweight DSE surrogate."""

    a_km = earth_radius_km + altitude_km
    mean_motion_rad_s = math.sqrt(EARTH_MU_KM3_S2 / (a_km**3))
    return 2.0 * math.pi / mean_motion_rad_s


def mean_motion_rad_s(altitude_km: float, *, earth_radius_km: float = EARTH_RADIUS_KM) -> float:
    a_km = earth_radius_km + altitude_km
    return math.sqrt(EARTH_MU_KM3_S2 / (a_km**3))


def coverage_central_angle_deg(
    altitude_km: float,
    min_elevation_deg: float,
    *,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> float:
    """Maximum geocentric angle visible above a ground elevation mask."""

    a_km = earth_radius_km + altitude_km
    eps = math.radians(min_elevation_deg)
    ratio = earth_radius_km / a_km
    inner = max(0.0, 1.0 - (ratio**2) * (math.cos(eps) ** 2))
    cos_theta = ratio * (math.cos(eps) ** 2) + math.sin(eps) * math.sqrt(inner)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_theta))))


def line_of_sight_ok(
    r1_km: tuple[float, float, float],
    r2_km: tuple[float, float, float],
    *,
    h_clear_km: float = 0.0,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> bool:
    """Generic segment-to-Earth line-of-sight test for two ECI positions."""

    dx = r2_km[0] - r1_km[0]
    dy = r2_km[1] - r1_km[1]
    dz = r2_km[2] - r1_km[2]
    denom = dx * dx + dy * dy + dz * dz
    if denom <= 0:
        return True
    lambda_star = -((r1_km[0] * dx + r1_km[1] * dy + r1_km[2] * dz) / denom)
    lambda_star = max(0.0, min(1.0, lambda_star))
    px = r1_km[0] + lambda_star * dx
    py = r1_km[1] + lambda_star * dy
    pz = r1_km[2] + lambda_star * dz
    distance_to_earth_center = math.sqrt(px * px + py * py + pz * pz)
    return distance_to_earth_center > earth_radius_km + h_clear_km


@dataclass(frozen=True)
class ConstellationShell:
    """Legacy shell shape kept for old manifests and notebooks.

    In the clustered architecture model, ``sats_per_plane`` maps to
    ``clusters_per_plane`` and each compute cell contains one satellite.
    """

    name: str
    orbit_family: str
    altitude_km: float
    inclination_deg: float
    plane_count: int
    sats_per_plane: int
    min_elevation_deg: float = 25.0
    phase_factor: int = 0
    ltan: str | None = None

    @property
    def satellite_count(self) -> int:
        return self.plane_count * self.sats_per_plane

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstellationShell":
        return cls(
            name=str(data["name"]),
            orbit_family=str(_pick(data, "orbit_family", "walker")),
            altitude_km=float(data["altitude_km"]),
            inclination_deg=float(data["inclination_deg"]),
            plane_count=int(data["plane_count"]),
            sats_per_plane=int(data["sats_per_plane"]),
            min_elevation_deg=float(_pick(data, "min_elevation_deg", 25.0)),
            phase_factor=int(_pick(data, "phase_factor", 0)),
            ltan=data.get("ltan"),
        )

    def to_orbital_set(self) -> "OrbitalSet":
        return OrbitalSet(
            name=self.name,
            altitude_km=self.altitude_km,
            inclination_deg=self.inclination_deg,
            orbit_type=self.orbit_family,
            orbital_set_type="legacy_shell",
            walker_type="delta" if self.orbit_family == "walker" else self.orbit_family,
            planes=self.plane_count,
            clusters_per_plane=self.sats_per_plane,
            walker_phase=self.phase_factor,
            min_elevation_deg=self.min_elevation_deg,
            ltan=self.ltan,
            cluster=FormationCluster(),
            provenance={"source": "legacy_shell", "sats_per_plane_interpreted_as": "clusters_per_plane"},
        )

    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload["satellite_count"] = self.satellite_count
        return payload


@dataclass(frozen=True)
class FormationCluster:
    """Geometry of a co-orbiting satellite formation; no link topology is encoded here."""

    satellites_per_cluster: int = 1
    grid_shape: tuple[int, int] = (1, 1)
    formation_frame: str = "lvlh"
    pitch_m: float = 0.0
    pitch_x_m: float | None = None
    pitch_y_m: float | None = None
    cluster_radius_m: float | None = None
    d_keepout_m: float = 0.0
    sigma_navigation_m: float = 0.0
    sigma_control_m: float = 0.0
    delta_v_maint_per_year_m_s: float | None = None
    station_keeping_propellant_mass_kg: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "FormationCluster":
        data = data or {}
        removed = sorted(set(data) & {
            "topology", "tile_shape", "tile_center_stride", "inter_tile_bridge_policy",
            "short_isl_profile", "same_plane_long_isl_policy", "cross_plane_isl_policy",
        })
        if removed:
            raise ValueError(
                "orbit cluster no longer accepts link-topology fields "
                f"{removed}; configure static terminals in architecture and select visible links in runtime"
            )
        grid_shape = _parse_shape(_pick(data, "grid_shape", None))
        satellites_per_cluster = _optional_int(data.get("satellites_per_cluster"))
        if satellites_per_cluster is None:
            satellites_per_cluster = grid_shape[0] * grid_shape[1]
        if satellites_per_cluster != grid_shape[0] * grid_shape[1]:
            raise ValueError(
                "cluster.satellites_per_cluster must equal grid_shape rows*cols "
                f"({satellites_per_cluster} != {grid_shape[0] * grid_shape[1]})"
            )
        return cls(
            satellites_per_cluster=satellites_per_cluster,
            grid_shape=grid_shape,
            formation_frame=str(_pick(data, "formation_frame", "lvlh")),
            pitch_m=float(_pick(data, "pitch_m", _pick(data, "formation_pitch_m", 0.0))),
            pitch_x_m=_optional_float(data.get("pitch_x_m")),
            pitch_y_m=_optional_float(data.get("pitch_y_m")),
            cluster_radius_m=_optional_float(data.get("cluster_radius_m")),
            d_keepout_m=float(_pick(data, "d_keepout_m", 0.0)),
            sigma_navigation_m=float(_pick(data, "sigma_navigation_m", 0.0)),
            sigma_control_m=float(_pick(data, "sigma_control_m", 0.0)),
            delta_v_maint_per_year_m_s=_optional_float(data.get("delta_v_maint_per_year_m_s")),
            station_keeping_propellant_mass_kg=_optional_float(data.get("station_keeping_propellant_mass_kg")),
        )

    @property
    def resolved_pitch_x_m(self) -> float:
        return self.pitch_x_m if self.pitch_x_m is not None else self.pitch_m

    @property
    def resolved_pitch_y_m(self) -> float:
        return self.pitch_y_m if self.pitch_y_m is not None else self.pitch_m

    @property
    def offsets_lvlh_m(self) -> tuple[tuple[float, float, float], ...]:
        rows, cols = self.grid_shape
        return tuple(
            ((row - (rows - 1) / 2.0) * self.resolved_pitch_x_m,
             (col - (cols - 1) / 2.0) * self.resolved_pitch_y_m, 0.0)
            for row in range(rows) for col in range(cols)
        )

    @property
    def computed_cluster_radius_m(self) -> float:
        return max((math.dist((0.0, 0.0, 0.0), item) for item in self.offsets_lvlh_m), default=0.0)

    @property
    def effective_cluster_radius_m(self) -> float:
        return max(self.cluster_radius_m or 0.0, self.computed_cluster_radius_m)

    def formation_analysis(self) -> dict[str, Any]:
        offsets = self.offsets_lvlh_m
        distances = [math.dist(offsets[i], offsets[j]) for i in range(len(offsets)) for j in range(i + 1, len(offsets))]
        minimum = min(distances) if distances else None
        required = self.d_keepout_m + self.sigma_navigation_m + self.sigma_control_m
        warnings: list[str] = []
        if self.cluster_radius_m is not None and self.cluster_radius_m < self.computed_cluster_radius_m:
            warnings.append("cluster.cluster_radius_m is smaller than grid-derived radius; using conservative maximum")
        if self.formation_frame.lower() != "lvlh":
            warnings.append("only LVLH fixed-grid formation analysis is implemented")
        return {
            "grid_shape": list(self.grid_shape), "satellites_per_cluster": self.satellites_per_cluster,
            "formation_frame": self.formation_frame, "pitch_x_m": self.resolved_pitch_x_m,
            "pitch_y_m": self.resolved_pitch_y_m, "computed_cluster_radius_m": self.computed_cluster_radius_m,
            "effective_cluster_radius_m": self.effective_cluster_radius_m,
            "formation_min_pair_distance_m": minimum, "formation_required_keepout_margin_m": required,
            "formation_keepout_feasible": True if minimum is None else minimum >= required,
            "station_keeping": {"delta_v_maint_per_year_m_s": self.delta_v_maint_per_year_m_s,
                "station_keeping_propellant_mass_kg": self.station_keeping_propellant_mass_kg, "mode": "sensitivity_only"},
            "warnings": warnings,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "satellites_per_cluster": self.satellites_per_cluster, "grid_shape": list(self.grid_shape),
            "formation_frame": self.formation_frame, "pitch_m": self.pitch_m,
            "pitch_x_m": self.pitch_x_m, "pitch_y_m": self.pitch_y_m,
            "cluster_radius_m": self.cluster_radius_m, "d_keepout_m": self.d_keepout_m,
            "sigma_navigation_m": self.sigma_navigation_m, "sigma_control_m": self.sigma_control_m,
            "delta_v_maint_per_year_m_s": self.delta_v_maint_per_year_m_s,
            "station_keeping_propellant_mass_kg": self.station_keeping_propellant_mass_kg,
        }


@dataclass(frozen=True)
class CustomOrbitalPlane:
    """Optional per-plane override for custom orbital-set descriptions."""

    index: int
    altitude_km: float | None = None
    inclination_deg: float | None = None
    raan_deg: float | None = None
    u0_deg: float | None = None
    clusters_per_plane: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomOrbitalPlane":
        return cls(
            index=int(_pick(data, "index", _pick(data, "plane_index", 0))),
            altitude_km=_optional_float(data.get("altitude_km")),
            inclination_deg=_optional_float(data.get("inclination_deg")),
            raan_deg=_optional_float(data.get("raan_deg", data.get("Omega_deg"))),
            u0_deg=_optional_float(data.get("u0_deg")),
            clusters_per_plane=_optional_int(data.get("clusters_per_plane")),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ResolvedOrbitalPlane:
    """Public per-plane propagation parameters resolved from an ``OrbitalSet``."""

    orbital_set_name: str
    plane_index: int
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    u0_deg: float
    clusters_per_plane: int
    min_elevation_deg: float
    orbit_type: str
    orbital_set_type: str
    walker_type: str | None = None
    ltan: str | None = None
    raan_pattern_or_ltan: str | None = None
    epoch: str | None = None
    custom_override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class OrbitalSet:
    name: str
    altitude_km: float
    inclination_deg: float
    orbit_type: str = "circular"
    orbital_set_type: str = "walker_delta"
    walker_type: str | None = "delta"
    raan_span_deg: float | None = None
    raan_spacing_mode: str = "periodic"
    selected_plane_indices: tuple[int, ...] | None = None
    selected_cell_indices: dict[int, tuple[int, ...]] = field(default_factory=dict)
    raan_pattern_or_ltan: str | None = None
    planes: int = 1
    clusters_per_plane: int = 1
    walker_phase: int = 0
    epoch: str | None = None
    min_elevation_deg: float = 25.0
    ltan: str | None = None
    raan0_deg: float = 0.0
    u0_deg: float = 0.0
    custom_planes: tuple[CustomOrbitalPlane, ...] = ()
    cluster: FormationCluster = field(default_factory=FormationCluster)
    satellite_profile: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "satellite_profile",
            _validate_orbital_set_satellite_profile(self.satellite_profile),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrbitalSet":
        cluster = FormationCluster.from_dict(data.get("cluster"))
        satellite_profile = _validate_orbital_set_satellite_profile(data.get("satellite_profile"))
        planes = int(_pick(data, "planes", _pick(data, "plane_count", 1)))
        clusters_per_plane = int(_pick(data, "clusters_per_plane", _pick(data, "sats_per_plane", 1)))
        if planes <= 0:
            raise ValueError("orbital_set.planes must be positive")
        if clusters_per_plane <= 0:
            raise ValueError("orbital_set.clusters_per_plane must be positive")
        selected_planes = _parse_optional_int_tuple(data.get("selected_plane_indices"))
        selected_cells = _parse_selected_cells(data.get("selected_cell_indices"))
        orbit_type = str(_pick(data, "orbit_type", _pick(data, "orbit_family", "circular")))
        orbital_set_type = str(_pick(data, "orbital_set_type", _pick(data, "shell_type", "walker_delta")))
        walker_type = data.get("walker_type")
        if walker_type is None:
            mode = _normalise_mode(orbital_set_type)
            if "star" in mode:
                walker_type = "star"
            elif "walker" in mode or "shell" in mode:
                walker_type = "delta"
        return cls(
            name=str(data["name"]),
            altitude_km=float(data["altitude_km"]),
            inclination_deg=float(data["inclination_deg"]),
            orbit_type=orbit_type,
            orbital_set_type=orbital_set_type,
            walker_type=str(walker_type) if walker_type is not None else None,
            raan_span_deg=_optional_float(data.get("raan_span_deg")),
            raan_spacing_mode=str(_pick(data, "raan_spacing_mode", "periodic")),
            selected_plane_indices=selected_planes,
            selected_cell_indices=selected_cells,
            raan_pattern_or_ltan=data.get("raan_pattern_or_ltan"),
            planes=planes,
            clusters_per_plane=clusters_per_plane,
            walker_phase=int(_pick(data, "walker_phase", _pick(data, "phase_factor", 0))),
            epoch=data.get("epoch"),
            min_elevation_deg=float(_pick(data, "min_elevation_deg", 25.0)),
            ltan=data.get("ltan"),
            raan0_deg=float(_pick(data, "raan0_deg", _pick(data, "Omega0_deg", 0.0))),
            u0_deg=float(_pick(data, "u0_deg", 0.0)),
            custom_planes=tuple(
                CustomOrbitalPlane.from_dict(item)
                for item in (data.get("custom_planes") or data.get("planes_detail") or ())
            ),
            cluster=cluster,
            satellite_profile=satellite_profile,
            provenance={"source": "orbital_set", **dict(data.get("provenance") or {})},
        )

    @property
    def effective_plane_indices(self) -> tuple[int, ...]:
        if self.selected_plane_indices is not None:
            return self.selected_plane_indices
        if self.custom_planes:
            return tuple(plane.index for plane in self.custom_planes)
        return tuple(range(self.planes))

    @property
    def effective_plane_count(self) -> int:
        return len(self.effective_plane_indices)

    def effective_cell_indices(self, plane_index: int) -> tuple[int, ...]:
        if plane_index in self.selected_cell_indices:
            return self.selected_cell_indices[plane_index]
        return tuple(range(self._clusters_per_plane_for(plane_index)))

    @property
    def compute_cell_count(self) -> int:
        return sum(len(self.effective_cell_indices(plane)) for plane in self.effective_plane_indices)

    @property
    def satellite_count(self) -> int:
        return self.compute_cell_count * self.cluster.satellites_per_cluster

    @property
    def semimajor_axis_km(self) -> float:
        return EARTH_RADIUS_KM + self.altitude_km

    @property
    def mean_motion_rad_s(self) -> float:
        return mean_motion_rad_s(self.altitude_km)

    @property
    def orbital_period_s(self) -> float:
        return 2.0 * math.pi / self.mean_motion_rad_s

    @property
    def resolved_raan_span_deg(self) -> float:
        if self.raan_span_deg is not None:
            return self.raan_span_deg
        if self.walker_type and _normalise_mode(self.walker_type) == "star":
            return 180.0
        return 360.0

    @property
    def raan_degrees(self) -> tuple[float, ...]:
        return tuple(self._raan_for_plane(index) for index in self.effective_plane_indices)

    def iter_resolved_planes(self) -> tuple[ResolvedOrbitalPlane, ...]:
        """Return deterministic per-plane propagation parameters.

        This helper is the public boundary for downstream trace materializers
        that need plane-level RAAN, altitude, inclination, and cell count
        without depending on private placement helpers.
        """

        planes: list[ResolvedOrbitalPlane] = []
        for plane_index in self.effective_plane_indices:
            custom = self._custom_plane(plane_index)
            planes.append(
                ResolvedOrbitalPlane(
                    orbital_set_name=self.name,
                    plane_index=plane_index,
                    altitude_km=(
                        custom.altitude_km
                        if custom and custom.altitude_km is not None
                        else self.altitude_km
                    ),
                    inclination_deg=(
                        custom.inclination_deg
                        if custom and custom.inclination_deg is not None
                        else self.inclination_deg
                    ),
                    raan_deg=self._raan_for_plane(plane_index),
                    u0_deg=self._u0_for_plane(plane_index),
                    clusters_per_plane=self._clusters_per_plane_for(plane_index),
                    min_elevation_deg=self.min_elevation_deg,
                    orbit_type=self.orbit_type,
                    orbital_set_type=self.orbital_set_type,
                    walker_type=self.walker_type,
                    ltan=self.ltan,
                    raan_pattern_or_ltan=self.raan_pattern_or_ltan,
                    epoch=self.epoch,
                    custom_override=custom is not None,
                )
            )
        return tuple(planes)

    def cell_argument_latitude_deg(self, plane_index: int, cell_index: int) -> float:
        clusters = self._clusters_per_plane_for(plane_index)
        full_cell_count = max(1, sum(self._clusters_per_plane_for(index) for index in self.effective_plane_indices))
        walker_offset = 360.0 * self.walker_phase * plane_index / full_cell_count
        return self._u0_for_plane(plane_index) + 360.0 * cell_index / clusters + walker_offset

    def _custom_plane_map(self) -> dict[int, CustomOrbitalPlane]:
        return {plane.index: plane for plane in self.custom_planes}

    def _custom_plane(self, plane_index: int) -> CustomOrbitalPlane | None:
        return self._custom_plane_map().get(plane_index)

    def _clusters_per_plane_for(self, plane_index: int) -> int:
        custom = self._custom_plane(plane_index)
        if custom and custom.clusters_per_plane is not None:
            return custom.clusters_per_plane
        return self.clusters_per_plane

    def _raan_for_plane(self, plane_index: int) -> float:
        custom = self._custom_plane(plane_index)
        if custom and custom.raan_deg is not None:
            return custom.raan_deg
        if self.planes == 1:
            return self.raan0_deg
        span = self.resolved_raan_span_deg
        if self.raan_spacing_mode == "inclusive_span" and self.planes > 1:
            return self.raan0_deg + plane_index * span / (self.planes - 1)
        return self.raan0_deg + plane_index * span / self.planes

    def _u0_for_plane(self, plane_index: int) -> float:
        custom = self._custom_plane(plane_index)
        if custom and custom.u0_deg is not None:
            return custom.u0_deg
        return self.u0_deg

    def analysis(self) -> dict[str, Any]:
        cluster_analysis = self.cluster.formation_analysis()
        a_km = self.semimajor_axis_km
        period = self.orbital_period_s
        velocity_km_s = math.sqrt(EARTH_MU_KM3_S2 / a_km)
        same_plane: dict[str, Any]
        if self.clusters_per_plane > 1:
            centroid_distance_km = 2.0 * a_km * math.sin(math.pi / self.clusters_per_plane)
            arc_distance_km = 2.0 * math.pi * a_km / self.clusters_per_plane
            min_sat_distance_km = max(0.0, centroid_distance_km - 2.0 * self.cluster.effective_cluster_radius_m / 1000.0)
            center_angle_rad = 2.0 * math.pi / self.clusters_per_plane
            los_angle_limit_rad = 2.0 * math.acos(min(1.0, EARTH_RADIUS_KM / a_km))
            same_plane = {
                "adjacent_centroid_distance_km": centroid_distance_km,
                "adjacent_arc_distance_km": arc_distance_km,
                "adjacent_min_satellite_distance_km_conservative": min_sat_distance_km,
                "adjacent_propagation_delay_s": centroid_distance_km / C_LIGHT_KM_PER_S,
                "adjacent_center_angle_deg": math.degrees(center_angle_rad),
                "adjacent_los_feasible_angle_check": center_angle_rad <= los_angle_limit_rad,
                "average_ring_hop_count_estimate": self.clusters_per_plane / 4.0,
            }
        else:
            same_plane = {
                "adjacent_centroid_distance_km": None,
                "adjacent_arc_distance_km": None,
                "adjacent_min_satellite_distance_km_conservative": None,
                "adjacent_propagation_delay_s": None,
                "adjacent_center_angle_deg": None,
                "adjacent_los_feasible_angle_check": None,
                "average_ring_hop_count_estimate": None,
            }
        theta_max = coverage_central_angle_deg(self.altitude_km, self.min_elevation_deg)
        beta_no_eclipse_rad = math.asin(min(1.0, EARTH_RADIUS_KM / a_km))
        eclipse_half_angle_rad = math.acos(math.sqrt(max(0.0, 1.0 - (EARTH_RADIUS_KM / a_km) ** 2)))
        eclipse_duration_beta0_s = 2.0 * eclipse_half_angle_rad / self.mean_motion_rad_s
        warnings = list(cluster_analysis.get("warnings", ()))
        if self.custom_planes:
            warnings.append(
                "custom orbital planes are represented for placement/counting; high-fidelity per-plane ephemeris "
                "and cross-plane time series are not evaluated"
            )
        if self.orbit_type == "dawn_dusk_sso" and not self.ltan and not self.raan_pattern_or_ltan:
            warnings.append("dawn_dusk_sso should configure LTAN or raan_pattern_or_ltan for reproducible sunlight geometry")
        if self.selected_plane_indices is not None:
            warnings.append("partial Walker/orbital-set coverage must be reported separately from full-shell averages")
        warnings.extend(
            [
                "STK-calibrated ephemeris, visibility, and shadow time series are not implemented in this lightweight analysis",
                "dynamic port scheduling, FOR/pointing, acquisition, and queueing are not evaluated",
            ]
        )
        return {
            "name": self.name,
            "orbit": {
                "altitude_km": self.altitude_km,
                "semimajor_axis_km": a_km,
                "inclination_deg": self.inclination_deg,
                "orbit_type": self.orbit_type,
                "orbital_set_type": self.orbital_set_type,
                "walker_type": self.walker_type,
                "raan_span_deg": self.resolved_raan_span_deg,
                "raan_spacing_mode": self.raan_spacing_mode,
                "raan_deg": list(self.raan_degrees),
                "custom_planes": [plane.to_dict() for plane in self.custom_planes],
                "ltan": self.ltan,
                "epoch": self.epoch,
                "mean_motion_rad_s": self.mean_motion_rad_s,
                "orbital_period_s": period,
                "orbital_velocity_km_s": velocity_km_s,
            },
            "placement": {
                "configured_planes": self.planes,
                "selected_plane_indices": list(self.effective_plane_indices),
                "effective_plane_count": self.effective_plane_count,
                "clusters_per_plane": self.clusters_per_plane,
                "compute_cell_count": self.compute_cell_count,
                "satellites_per_cluster": self.cluster.satellites_per_cluster,
                "satellite_count": self.satellite_count,
                "walker_phase": self.walker_phase,
                "sample_argument_latitude_deg": [
                    {
                        "plane_index": plane,
                        "cell_index": cell,
                        "u_deg": self.cell_argument_latitude_deg(plane, cell),
                    }
                    for plane in self.effective_plane_indices[: min(3, self.effective_plane_count)]
                    for cell in self.effective_cell_indices(plane)[: min(3, len(self.effective_cell_indices(plane)))]
                ],
            },
            "cluster": cluster_analysis,
            "same_plane": same_plane,
            "coverage": {
                "evaluated": False,
                "footprint_check_evaluated": True,
                "min_elevation_deg": self.min_elevation_deg,
                "theta_max_deg": theta_max,
                "single_cell_ground_footprint_area_km2_spherical_cap": 2.0
                * math.pi
                * (EARTH_RADIUS_KM**2)
                * (1.0 - math.cos(math.radians(theta_max))),
                "eci_ecef_model": "simplified GMST rotation required downstream; not evaluated here",
                "visible_cell_count": None,
                "sgl_pass_duration_s": None,
                "handover_opportunity_count": None,
            },
            "cross_plane_links": {
                "evaluated": False,
                "nearest_distance_time_series": None,
                "los_feasible_fraction": None,
                "optical_for_pointing_feasible_fraction": None,
                "handover_rate": None,
            },
            "shadow": {
                "evaluated": False,
                "geometry_check_evaluated": True,
                "beta_angle_no_eclipse_threshold_deg": math.degrees(beta_no_eclipse_rad),
                "eclipse_duration_beta0_s": eclipse_duration_beta0_s,
                "sunlit_fraction_beta0": max(0.0, 1.0 - eclipse_duration_beta0_s / period),
                "cluster_shadow_boundary_skew_s_conservative": (
                    2.0 * self.cluster.effective_cluster_radius_m / 1000.0 / velocity_km_s
                ),
                "model": "cylindrical-shadow geometry check",
            },
            "warnings": tuple(dict.fromkeys(warnings)),
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "altitude_km": self.altitude_km,
            "inclination_deg": self.inclination_deg,
            "orbit_type": self.orbit_type,
            "orbital_set_type": self.orbital_set_type,
            "walker_type": self.walker_type,
            "raan_span_deg": self.raan_span_deg,
            "raan_spacing_mode": self.raan_spacing_mode,
            "selected_plane_indices": list(self.selected_plane_indices) if self.selected_plane_indices else None,
            "selected_cell_indices": {
                str(key): list(value) for key, value in self.selected_cell_indices.items()
            },
            "raan_pattern_or_ltan": self.raan_pattern_or_ltan,
            "planes": self.planes,
            "clusters_per_plane": self.clusters_per_plane,
            "walker_phase": self.walker_phase,
            "epoch": self.epoch,
            "min_elevation_deg": self.min_elevation_deg,
            "ltan": self.ltan,
            "raan0_deg": self.raan0_deg,
            "u0_deg": self.u0_deg,
            "custom_planes": [plane.to_dict() for plane in self.custom_planes],
            "cluster": self.cluster.to_dict(),
            "satellite_profile": to_jsonable(self.satellite_profile),
            "compute_cell_count": self.compute_cell_count,
            "satellite_count": self.satellite_count,
            "provenance": to_jsonable(self.provenance),
        }


@dataclass(frozen=True)
class ConstellationDesign:
    name: str
    shells: tuple[ConstellationShell, ...] = ()
    orbital_sets: tuple[OrbitalSet, ...] = ()
    gateway_profile: dict[str, Any] = field(default_factory=dict)
    application_profile: str = "ChatLLM inference"
    safety: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "gateway_profile", _validate_gateway_profile(self.gateway_profile))

    @property
    def effective_orbital_sets(self) -> tuple[OrbitalSet, ...]:
        if self.orbital_sets:
            return self.orbital_sets
        return tuple(shell.to_orbital_set() for shell in self.shells)

    @property
    def compute_cell_count(self) -> int:
        return sum(orbital_set.compute_cell_count for orbital_set in self.effective_orbital_sets)

    @property
    def satellite_count(self) -> int:
        return sum(orbital_set.satellite_count for orbital_set in self.effective_orbital_sets)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self.architecture_analysis()["warnings"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConstellationDesign":
        if "isl_topology" in data:
            raise ValueError("constellation.isl_topology was removed; runtime selects links from orbit visibility and architecture terminal limits")
        shells = tuple(ConstellationShell.from_dict(item) for item in data.get("shells", ()))
        orbital_sets = tuple(OrbitalSet.from_dict(item) for item in data.get("orbital_sets", ()))
        if not shells and not orbital_sets:
            raise ValueError("constellation must contain shells or orbital_sets")
        return cls(
            name=str(data["name"]),
            shells=shells,
            orbital_sets=orbital_sets,
            gateway_profile=_validate_gateway_profile(data.get("gateway_profile")),
            application_profile=str(_pick(data, "application_profile", "ChatLLM inference")),
            safety=dict(data.get("safety") or {}),
        )

    def energy_design_condition(self) -> dict[str, float]:
        """Return conservative beta-zero orbit conditions for power-system sizing."""

        shadows = [item.analysis()["shadow"] for item in self.effective_orbital_sets]
        eclipse_s = max(float(item["eclipse_duration_beta0_s"]) for item in shadows)
        sunlit_fraction = min(float(item["sunlit_fraction_beta0"]) for item in shadows)
        return {
            "eclipse_duration_min": eclipse_s / 60.0,
            "sunlit_fraction": sunlit_fraction,
        }

    def architecture_analysis(self) -> dict[str, Any]:
        orbital_set_analyses = [orbital_set.analysis() for orbital_set in self.effective_orbital_sets]
        warnings: list[str] = []
        for item in orbital_set_analyses:
            warnings.extend(item.get("warnings", ()))
        safety_screening = _safety_screening(self.effective_orbital_sets, self.safety)
        warnings.extend(safety_screening["warnings"])
        if self.shells and not self.orbital_sets:
            warnings.append(
                "legacy shells are interpreted as one-satellite compute cells; use orbital_sets for clustered formations"
            )
        return {
            "evaluated": True,
            "model": "lightweight_constellation_architecture_surrogate",
            "application_profile": self.application_profile,
            "orbital_set_count": len(self.effective_orbital_sets),
            "compute_cell_count": self.compute_cell_count,
            "satellite_count": self.satellite_count,
            "orbital_sets": orbital_set_analyses,
            "energy_design_condition": self.energy_design_condition(),
            "safety_screening": safety_screening,
            "gateway_profile": to_jsonable(self.gateway_profile),
            "provenance": {
                "source": "docs/research/parameters/constellation_architecture_model.md",
                "high_fidelity_calibration": "not_evaluated",
                "dynamic_routing_queueing": "not_evaluated",
            },
            "warnings": list(dict.fromkeys(warnings)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shells": [shell.to_dict() for shell in self.shells],
            "orbital_sets": [orbital_set.to_dict() for orbital_set in self.effective_orbital_sets],
            "gateway_profile": to_jsonable(self.gateway_profile),
            "application_profile": self.application_profile,
            "compute_cell_count": self.compute_cell_count,
            "satellite_count": self.satellite_count,
            "architecture_analysis": self.architecture_analysis(),
        }


def _parse_shape(value: Any) -> tuple[int, int]:
    if value is None:
        return (1, 1)
    if isinstance(value, str):
        parts = value.lower().replace(" ", "").split("x")
        if len(parts) != 2:
            raise ValueError(f"grid shape must look like 'MxN', got {value!r}")
        rows, cols = int(parts[0]), int(parts[1])
    else:
        rows, cols = value
        rows, cols = int(rows), int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError("grid shape dimensions must be positive")
    return (rows, cols)


def _parse_optional_shape(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    return _parse_shape(value)


def _parse_optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _parse_selected_cells(value: Any) -> dict[int, tuple[int, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("selected_cell_indices must be a mapping from plane index to cell indices")
    return {int(key): tuple(int(item) for item in items) for key, items in value.items()}


def _safety_screening(orbital_sets: tuple[OrbitalSet, ...], safety: dict[str, Any]) -> dict[str, Any]:
    altitude_threshold = float(safety.get("altitude_separation_threshold_km", 20.0))
    conjunction_threshold = float(safety.get("conjunction_threshold_km", 1.0))
    pair_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for idx, first in enumerate(orbital_sets):
        for second in orbital_sets[idx + 1 :]:
            altitude_delta = abs(first.altitude_km - second.altitude_km)
            inclination_delta = abs(first.inclination_deg - second.inclination_deg)
            screening_required = altitude_delta < altitude_threshold and inclination_delta > 1e-6
            lower_bound_km = max(0.0, altitude_delta - (first.cluster.effective_cluster_radius_m + second.cluster.effective_cluster_radius_m) / 1000.0)
            risk = screening_required and lower_bound_km < conjunction_threshold
            if screening_required:
                warnings.append(
                    f"orbital sets {first.name}/{second.name} have close altitudes and different inclinations; "
                    "high-fidelity conjunction screening is required"
                )
            pair_results.append(
                {
                    "orbital_set_a": first.name,
                    "orbital_set_b": second.name,
                    "altitude_delta_km": altitude_delta,
                    "inclination_delta_deg": inclination_delta,
                    "screening_required": screening_required,
                    "min_cross_shell_sat_distance_lower_bound_km": lower_bound_km,
                    "conjunction_event_count": None,
                    "required_altitude_separation_km": altitude_threshold if risk else None,
                    "risk_flag": risk,
                }
            )
    return {
        "evaluated": True,
        "method": "altitude/inclination trigger plus conservative lower bound",
        "altitude_separation_threshold_km": altitude_threshold,
        "conjunction_threshold_km": conjunction_threshold,
        "pairs": pair_results,
        "warnings": warnings,
    }
