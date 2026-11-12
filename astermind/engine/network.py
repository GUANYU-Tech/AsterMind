"""Physical network state shared by OrbitServe and AsterMind.

Network produces immutable frames.  It does not own a simulation clock and it
does not execute transfers.  Communication reads one frame at the current
global time, selects a route, and delegates timing to exactly one backend.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class NetworkNodeState:
    node_id: str
    satellite_id: str
    role: str = "compute"
    state: str = "up"
    isl_terminals: int = 0
    sgl_terminals: int = 0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id or not self.satellite_id:
            raise ValueError("network node and satellite IDs must be nonempty")
        if self.state not in {"up", "down"}:
            raise ValueError("network node state must be up or down")
        if type(self.isl_terminals) is not int or self.isl_terminals < 0:
            raise ValueError("isl_terminals must be a nonnegative integer")
        if type(self.sgl_terminals) is not int or self.sgl_terminals < 0:
            raise ValueError("sgl_terminals must be a nonnegative integer")


@dataclass(frozen=True)
class CandidateTopologyLink:
    link_id: str
    source_node_id: str
    target_node_id: str
    profile_id: str
    kind: str = "isl"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.link_id or not self.source_node_id or not self.target_node_id:
            raise ValueError("candidate link identifiers must be nonempty")
        if self.source_node_id == self.target_node_id:
            raise ValueError("candidate links cannot be self-links")
        if not self.profile_id:
            raise ValueError("candidate link requires a Link Profile")


@dataclass(frozen=True)
class CandidateTopologyGraph:
    node_ids: tuple[str, ...]
    links: tuple[CandidateTopologyLink, ...]

    def __post_init__(self) -> None:
        if len(self.node_ids) != len(set(self.node_ids)):
            raise ValueError("candidate topology node IDs must be unique")
        known = set(self.node_ids)
        identifiers: set[str] = set()
        edges: set[tuple[str, str]] = set()
        for link in self.links:
            if link.link_id in identifiers:
                raise ValueError(f"duplicate candidate link ID: {link.link_id}")
            identifiers.add(link.link_id)
            if link.source_node_id not in known or link.target_node_id not in known:
                raise ValueError(
                    f"candidate link references an unknown node: {link.link_id}"
                )
            edge = tuple(sorted((link.source_node_id, link.target_node_id)))
            if edge in edges:
                raise ValueError(f"duplicate candidate topology edge: {edge}")
            edges.add(edge)


@dataclass(frozen=True)
class LinkProfile:
    profile_id: str
    bandwidth_bps: float
    propagation_delay_s: float = 0.0
    max_distance_km: float | None = None
    capacity_model: str = "fixed"
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("Link Profile ID must be nonempty")
        if not math.isfinite(self.bandwidth_bps) or self.bandwidth_bps <= 0:
            raise ValueError("Link Profile bandwidth must be finite and positive")
        if not math.isfinite(self.propagation_delay_s) or self.propagation_delay_s < 0:
            raise ValueError("Link Profile delay must be finite and nonnegative")
        if self.max_distance_km is not None and (
            not math.isfinite(self.max_distance_km) or self.max_distance_km <= 0
        ):
            raise ValueError("Link Profile maximum distance must be positive")
        if self.capacity_model not in {
            "fixed",
            "free_space_shannon",
            "optical_photon_budget",
            "islsim",
        }:
            raise ValueError(f"unsupported Link Profile model: {self.capacity_model}")


@dataclass(frozen=True)
class LinkDecision:
    eligible: bool
    reason: str = ""
    distance_km: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.eligible and not self.reason:
            raise ValueError("an ineligible link must record a reason")


class ConnectionPolicy(Protocol):
    """Choose preferred candidates and terminal allocation order."""

    def order(
        self,
        candidates: tuple[CandidateTopologyLink, ...],
        node_states: Mapping[str, NetworkNodeState],
        previous_frame: NetworkStateFrame | None,
    ) -> tuple[CandidateTopologyLink, ...]: ...


class LinkEligibility(Protocol):
    """Evaluate visibility, distance, terminals, node state and faults."""

    def evaluate(
        self,
        candidate: CandidateTopologyLink,
        profile: LinkProfile,
        node_states: Mapping[str, NetworkNodeState],
        time_s: float,
    ) -> LinkDecision: ...


class PreserveCandidateOrder:
    def order(
        self,
        candidates: tuple[CandidateTopologyLink, ...],
        node_states: Mapping[str, NetworkNodeState],
        previous_frame: NetworkStateFrame | None,
    ) -> tuple[CandidateTopologyLink, ...]:
        return candidates


class BasicLinkEligibility:
    """Conservative baseline; geometry-specific checks remain injectable."""

    def evaluate(
        self,
        candidate: CandidateTopologyLink,
        profile: LinkProfile,
        node_states: Mapping[str, NetworkNodeState],
        time_s: float,
    ) -> LinkDecision:
        source = node_states[candidate.source_node_id]
        target = node_states[candidate.target_node_id]
        if source.state != "up" or target.state != "up":
            return LinkDecision(False, "node_down")
        if candidate.attributes.get("faulted"):
            return LinkDecision(False, "injected_fault")
        distance = candidate.attributes.get("distance_km")
        if distance is not None:
            distance = float(distance)
            if (
                profile.max_distance_km is not None
                and distance > profile.max_distance_km
            ):
                return LinkDecision(False, "distance_limit", distance)
        if candidate.kind == "isl" and (
            source.isl_terminals < 1 or target.isl_terminals < 1
        ):
            return LinkDecision(False, "isl_terminal_unavailable", distance)
        return LinkDecision(True, distance_km=distance)


@dataclass(frozen=True)
class EstablishedLink:
    link_id: str
    source_node_id: str
    target_node_id: str
    kind: str
    bandwidth_bps: float
    propagation_delay_s: float
    profile_id: str
    distance_km: float | None = None
    shared_capacity_pools: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkState:
    candidate: CandidateTopologyLink
    state: str
    reason: str
    established: EstablishedLink | None = None

    def __post_init__(self) -> None:
        if self.state not in {"established", "down"}:
            raise ValueError("link state must be established or down")
        if self.state == "established" and self.established is None:
            raise ValueError("established link state requires link attributes")
        if self.state == "down" and not self.reason:
            raise ValueError("down link state requires a reason")


@dataclass(frozen=True)
class NetworkStateFrame:
    start_time_s: float
    valid_until_s: float | None
    node_states: Mapping[str, NetworkNodeState]
    candidate_topology_graph: CandidateTopologyGraph
    link_states: tuple[LinkState, ...]
    shared_capacity_pools_bps: Mapping[str, float] = field(default_factory=dict)
    revision: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_time_s) or self.start_time_s < 0:
            raise ValueError("Network State Frame start must be nonnegative")
        if self.valid_until_s is not None and (
            not math.isfinite(self.valid_until_s)
            or self.valid_until_s <= self.start_time_s
        ):
            raise ValueError("Network State Frame requires a nonempty interval")
        if set(self.node_states) != set(self.candidate_topology_graph.node_ids):
            raise ValueError("frame node states must match candidate topology nodes")
        candidate_ids = {link.link_id for link in self.candidate_topology_graph.links}
        if {state.candidate.link_id for state in self.link_states} != candidate_ids:
            raise ValueError("frame requires one Link State per candidate link")

    @property
    def established_topology_graph(self) -> tuple[EstablishedLink, ...]:
        return tuple(
            state.established
            for state in self.link_states
            if state.established is not None
        )

    def contains(self, time_s: float) -> bool:
        return self.start_time_s <= time_s and (
            self.valid_until_s is None or time_s < self.valid_until_s
        )


class NetworkFrameSource(Protocol):
    def frame_at(self, time_s: float) -> NetworkStateFrame: ...


class RuntimeNetworkFrames:
    """Read the host's current physical state without allocating links again.

    Frames are instantaneous views. Routing, acquisition, directional budgets,
    observed faults and in-flight rate updates remain owned by NetworkState.
    This must not query exported future audit frames during a causal run.
    """

    def __init__(self, host):
        self.host = host

    def frame_at(self, time_s: float) -> NetworkStateFrame:
        if time_s != self.host.now:
            raise ValueError(
                "Runtime frames may only be read at the host's current time"
            )
        network = self.host.net
        nodes = {
            key: NetworkNodeState(
                key,
                key,
                node.role,
                "down"
                if any(
                    f["target_type"] == "node" and f["target_id"] == key
                    for f in network.active_faults.values()
                )
                else "up",
                node.isl_terminals,
                node.sgl_terminals,
            )
            for key, node in network.nodes.items()
        }
        candidates, states = [], []
        for edge in network.edges:
            candidate = CandidateTopologyLink(
                edge["link_id"],
                edge["endpoint_a"],
                edge["endpoint_b"],
                edge["kind"],
                edge["kind"],
            )
            candidates.append(candidate)
            current = network.link_at(edge["link_id"], time_s)
            established = None
            if current["state"] == "UP":
                directions = network.directions(edge, time_s)
                established = EstablishedLink(
                    candidate.link_id,
                    candidate.source_node_id,
                    candidate.target_node_id,
                    candidate.kind,
                    current["bandwidth_bps"],
                    current["propagation_delay_s"],
                    candidate.profile_id,
                    current.get("distance_km"),
                    attributes={"directions": directions},
                )
            states.append(
                LinkState(
                    candidate,
                    "established" if established else "down",
                    current.get("reason") or "not_established",
                    established,
                )
            )
        return NetworkStateFrame(
            time_s,
            math.nextafter(time_s, math.inf),
            nodes,
            CandidateTopologyGraph(tuple(nodes), tuple(candidates)),
            tuple(states),
            dict(network.pool_capacities),
            network.revision,
        )


class StaticNetworkFrames:
    def __init__(self, frame: NetworkStateFrame):
        if frame.start_time_s != 0 or frame.valid_until_s is not None:
            raise ValueError("static frame must cover [0, infinity)")
        self.frame = frame

    def frame_at(self, time_s: float) -> NetworkStateFrame:
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("network query time must be nonnegative")
        return self.frame


class FrameSequence:
    def __init__(self, frames: Iterable[NetworkStateFrame]):
        self.frames = tuple(frames)
        if not self.frames or self.frames[0].start_time_s != 0:
            raise ValueError("Network State Frames must start at time zero")
        for current, following in zip(self.frames, self.frames[1:]):
            if current.valid_until_s != following.start_time_s:
                raise ValueError("Network State Frames must be contiguous")
        self._starts = tuple(frame.start_time_s for frame in self.frames)

    def frame_at(self, time_s: float) -> NetworkStateFrame:
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("network query time must be nonnegative")
        index = bisect_right(self._starts, time_s) - 1
        if index < 0 or not self.frames[index].contains(time_s):
            raise LookupError(f"no Network State Frame covers {time_s}")
        return self.frames[index]


class OrbitEnvironmentFrames:
    """Adapt OrbitServe ephemeris/contact windows to Network State Frames.

    The environment remains the owner of orbit propagation and candidate
    contact generation.  This adapter applies Link Connection and exposes the
    immutable frame consumed by Communication.
    """

    def __init__(
        self,
        environment_state: Any,
        *,
        default_isl_bandwidth_bps: float,
        default_isl_terminals: int = 2,
        connection: LinkConnection | None = None,
    ):
        if (
            not math.isfinite(default_isl_bandwidth_bps)
            or default_isl_bandwidth_bps <= 0
        ):
            raise ValueError("default ISL bandwidth must be finite and positive")
        if type(default_isl_terminals) is not int or default_isl_terminals < 1:
            raise ValueError("default ISL terminal count must be positive")
        self.environment_state = environment_state
        self.default_isl_bandwidth_bps = float(default_isl_bandwidth_bps)
        self.default_isl_terminals = default_isl_terminals
        self.connection = connection or LinkConnection(
            PreserveCandidateOrder(), BasicLinkEligibility()
        )
        self._revision = 0
        self._last_frame: NetworkStateFrame | None = None

    def frame_at(self, time_s: float) -> NetworkStateFrame:
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("network query time must be nonnegative")
        catalog = tuple(getattr(self.environment_state, "satellite_catalog", ()) or ())
        satellite_ids = {
            str(row["satellite_id"])
            for row in catalog
            if row.get("satellite_id") is not None
        }
        windows = tuple(getattr(self.environment_state, "isl_windows", ()) or ())
        for row in windows:
            satellite_ids.update((str(row["endpoint_a"]), str(row["endpoint_b"])))
        node_states = {
            satellite_id: NetworkNodeState(
                satellite_id,
                satellite_id,
                "compute",
                "up",
                self.default_isl_terminals,
            )
            for satellite_id in sorted(satellite_ids)
        }
        active = [
            dict(row)
            for row in windows
            if float(row["start_s"]) <= time_s < float(row["end_s"])
        ]
        candidates: list[CandidateTopologyLink] = []
        profiles: dict[str, LinkProfile] = {}
        seen_edges: set[tuple[str, str]] = set()
        for index, row in enumerate(
            sorted(
                active,
                key=lambda item: (
                    str(item.get("link_id") or ""),
                    str(item["endpoint_a"]),
                    str(item["endpoint_b"]),
                ),
            )
        ):
            source, target = str(row["endpoint_a"]), str(row["endpoint_b"])
            edge = tuple(sorted((source, target)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            link_id = str(row.get("link_id") or f"isl:{source}:{target}")
            profile_id = f"orbit-window:{link_id}:{index}"
            delay = row.get("propagation_delay_s")
            if delay is None:
                low = float(row.get("propagation_delay_min_s", 0.0))
                high = float(row.get("propagation_delay_max_s", low))
                delay = (low + high) / 2.0
            distance = row.get("distance_km")
            if distance is None and row.get("distance_min_km") is not None:
                low = float(row["distance_min_km"])
                high = float(row.get("distance_max_km", low))
                distance = (low + high) / 2.0
            bandwidth = float(
                row.get("bandwidth_bps") or self.default_isl_bandwidth_bps
            )
            profiles[profile_id] = LinkProfile(
                profile_id,
                bandwidth,
                float(delay),
                capacity_model=str(row.get("capacity_model") or "fixed"),
                attributes={
                    key: value
                    for key, value in row.items()
                    if key not in {"endpoint_a", "endpoint_b"}
                },
            )
            candidates.append(
                CandidateTopologyLink(
                    link_id,
                    source,
                    target,
                    profile_id,
                    "isl",
                    {
                        "distance_km": distance,
                        "start_s": float(row["start_s"]),
                        "end_s": float(row["end_s"]),
                        "shared_capacity_pools": tuple(
                            row.get("shared_capacity_pools", ())
                        ),
                    },
                )
            )
        boundaries = sorted(
            {
                float(value)
                for row in windows
                for value in (row["start_s"], row["end_s"])
                if float(value) > time_s
            }
        )
        valid_until = boundaries[0] if boundaries else None
        graph = CandidateTopologyGraph(tuple(sorted(node_states)), tuple(candidates))
        self._revision += 1
        frame = build_network_frame(
            start_time_s=float(time_s),
            valid_until_s=valid_until,
            node_states=node_states,
            candidate_graph=graph,
            profiles=profiles,
            connection=self.connection,
            previous_frame=self._last_frame,
            revision=self._revision,
        )
        self._last_frame = frame
        return frame


class LinkConnection:
    """Apply Connection Policy, Link Eligibility and terminal limits."""

    def __init__(
        self,
        policy: ConnectionPolicy,
        eligibility: LinkEligibility,
    ):
        self.policy = policy
        self.eligibility = eligibility

    def establish(
        self,
        graph: CandidateTopologyGraph,
        node_states: Mapping[str, NetworkNodeState],
        profiles: Mapping[str, LinkProfile],
        *,
        time_s: float,
        previous_frame: NetworkStateFrame | None = None,
    ) -> tuple[LinkState, ...]:
        ordered = self.policy.order(graph.links, node_states, previous_frame)
        if {link.link_id for link in ordered} != {link.link_id for link in graph.links}:
            raise ValueError(
                "Connection Policy must return every candidate exactly once"
            )
        terminal_use: dict[str, int] = {node_id: 0 for node_id in node_states}
        results: list[LinkState] = []
        for candidate in ordered:
            profile = profiles[candidate.profile_id]
            decision = self.eligibility.evaluate(
                candidate, profile, node_states, time_s
            )
            if decision.eligible and candidate.kind == "isl":
                for node_id in (candidate.source_node_id, candidate.target_node_id):
                    if terminal_use[node_id] >= node_states[node_id].isl_terminals:
                        decision = LinkDecision(False, "isl_terminal_limit")
                        break
            if not decision.eligible:
                results.append(LinkState(candidate, "down", decision.reason))
                continue
            if candidate.kind == "isl":
                terminal_use[candidate.source_node_id] += 1
                terminal_use[candidate.target_node_id] += 1
            established = EstablishedLink(
                candidate.link_id,
                candidate.source_node_id,
                candidate.target_node_id,
                candidate.kind,
                profile.bandwidth_bps,
                profile.propagation_delay_s,
                profile.profile_id,
                decision.distance_km,
                tuple(candidate.attributes.get("shared_capacity_pools", ())),
                {**profile.attributes, **candidate.attributes, **decision.attributes},
            )
            results.append(LinkState(candidate, "established", "", established))
        by_id = {state.candidate.link_id: state for state in results}
        return tuple(by_id[link.link_id] for link in graph.links)


def build_network_frame(
    *,
    start_time_s: float,
    valid_until_s: float | None,
    node_states: Mapping[str, NetworkNodeState],
    candidate_graph: CandidateTopologyGraph,
    profiles: Mapping[str, LinkProfile],
    connection: LinkConnection,
    previous_frame: NetworkStateFrame | None = None,
    shared_capacity_pools_bps: Mapping[str, float] | None = None,
    revision: int = 0,
) -> NetworkStateFrame:
    link_states = connection.establish(
        candidate_graph,
        node_states,
        profiles,
        time_s=start_time_s,
        previous_frame=previous_frame,
    )
    return NetworkStateFrame(
        start_time_s,
        valid_until_s,
        dict(node_states),
        candidate_graph,
        link_states,
        dict(shared_capacity_pools_bps or {}),
        revision,
    )


__all__ = [
    "BasicLinkEligibility",
    "CandidateTopologyGraph",
    "CandidateTopologyLink",
    "ConnectionPolicy",
    "EstablishedLink",
    "FrameSequence",
    "LinkConnection",
    "LinkDecision",
    "LinkEligibility",
    "LinkProfile",
    "LinkState",
    "NetworkFrameSource",
    "NetworkNodeState",
    "NetworkStateFrame",
    "OrbitEnvironmentFrames",
    "PreserveCandidateOrder",
    "RuntimeNetworkFrames",
    "StaticNetworkFrames",
    "build_network_frame",
]
