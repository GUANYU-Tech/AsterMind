"""Free configuration -> AsterMind canonical graph -> native FlowSim graph."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import UTC, datetime

from .model import Link, Scenario


@dataclass(frozen=True)
class CompiledEdge:
    source: str
    target: str
    bandwidth_gbps: float
    latency_ns: int
    resource: str


class PhysicalTopology:
    """Owns physical identities; GPUs are endpoints, not implicit relay nodes."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.devices = {item.id: item for item in scenario.devices}
        self.satellites = {item.id: item for item in scenario.satellites}
        if len(self.devices) != len(scenario.devices):
            raise ValueError("duplicate device ID")
        if len(self.satellites) != len(scenario.satellites):
            raise ValueError("duplicate satellite ID")
        if len({c.id for c in scenario.cells}) != len(scenario.cells):
            raise ValueError("duplicate Cell ID")
        self.satellite_to_cell = {}
        for cell in scenario.cells:
            if not cell.satellites:
                raise ValueError(f"Cell {cell.id} has no satellites")
            for sat in cell.satellites:
                if sat not in self.satellites or sat in self.satellite_to_cell:
                    raise ValueError(f"unknown or repeated Cell member {sat}")
                self.satellite_to_cell[sat] = cell.id
        if set(self.satellite_to_cell) != set(self.satellites):
            raise ValueError("each satellite must belong to exactly one Cell")
        policy = scenario.link_policy
        if (
            policy is not None
            and policy.os_layout is not None
            and set(policy.os_layout.plane_by_node) != set(self.satellites)
        ):
            raise ValueError(
                "O-S reference layout must name every configured satellite"
            )
        for device in self.devices.values():
            if device.satellite not in self.satellites:
                raise ValueError(f"unknown satellite for device {device.id}")
        self.links = tuple(self._expand_links())
        self._validate_links()
        self.device_numbers = {
            identifier: index for index, identifier in enumerate(sorted(self.devices))
        }
        self.native_nodes, self.native_edges, self.node_device = self._compile()
        self.edge_resource = {
            (e.source, e.target): e.resource for e in self.native_edges
        }

    def _expand_links(self):
        yield from self.scenario.links
        template_ids = set()
        for template in self.scenario.templates:
            if template.id in template_ids:
                raise ValueError(f"duplicate template ID {template.id}")
            template_ids.add(template.id)
            members = template.devices
            if len(set(members)) != len(members) or not members:
                raise ValueError("template members must be nonempty and unique")
            if any(member not in self.devices for member in members):
                raise ValueError("template references unknown device")
            if template.kind == "switch":
                if (
                    template.switch not in self.devices
                    or self.devices[template.switch].kind != "switch"
                ):
                    raise ValueError("switch template needs an explicit Switch device")
                if template.switch in members:
                    raise ValueError("Switch cannot be its own template member")
                pairs = [(member, template.switch) for member in members]
            elif template.kind == "fc":
                if template.switch is not None:
                    raise ValueError("fc template does not accept switch")
                pairs = [
                    (a, b) for i, a in enumerate(members) for b in members[i + 1 :]
                ]
            else:
                if template.switch is not None:
                    raise ValueError("ring template does not accept switch")
                pairs = (
                    list(
                        dict.fromkeys(
                            tuple(sorted((members[i], members[(i + 1) % len(members)])))
                            for i in range(len(members))
                        )
                    )
                    if len(members) > 1
                    else []
                )
            for index, (source, target) in enumerate(pairs):
                yield Link(
                    id=f"{template.id}.{index}",
                    source=source,
                    target=target,
                    bandwidth_gbps=template.bandwidth_gbps,
                    latency_ns=template.latency_ns,
                )

    def _validate_links(self):
        identifiers, pairs, external_counts = set(), set(), {}
        for link in self.links:
            if link.id in identifiers:
                raise ValueError(f"duplicate link ID {link.id}")
            identifiers.add(link.id)
            if link.source not in self.devices or link.target not in self.devices:
                raise ValueError(f"unknown endpoint on link {link.id}")
            if link.source == link.target:
                raise ValueError("self links are unsupported")
            pair = tuple(sorted((link.source, link.target)))
            if pair in pairs:
                raise ValueError(
                    "duplicate physical pair: use distinct terminals for separate links"
                )
            pairs.add(pair)
            a, b = self.devices[link.source], self.devices[link.target]
            if link.availability_windows is not None and (
                (
                    self.scenario.link_policy is None
                    and not self.scenario.dynamic_external
                )
                or a.satellite == b.satellite
            ):
                raise ValueError(
                    "availability windows require dynamic inter-satellite link policy"
                )
            if a.satellite != b.satellite:
                if a.kind != "terminal" or b.kind != "terminal":
                    raise ValueError("inter-satellite links must join two terminals")
                if (
                    self.satellite_to_cell[a.satellite]
                    != self.satellite_to_cell[b.satellite]
                ) and (not a.external_access or not b.external_access):
                    raise ValueError(
                        "inter-Cell link requires external_access at both terminals"
                    )
                if link.enabled:
                    for terminal in (a, b):
                        external_counts[terminal.id] = (
                            external_counts.get(terminal.id, 0) + 1
                        )
        for identifier, count in (
            external_counts.items()
            if self.scenario.link_policy is None and not self.scenario.dynamic_external
            else ()
        ):
            if count > self.devices[identifier].max_external_links:
                raise ValueError(f"terminal {identifier} exceeds max_external_links")

    @property
    def canonical(self):
        return self._canonical_topology()

    def _canonical_topology(self):
        from astermind.interfaces import (
            NetworkLink,
            NetworkNode,
            SatelliteState,
            Topology,
        )

        # Reuse AsterMind's graph and reference validators. Cell membership and
        # shared capacities stay in the wrapper because upstream has no fields.
        stamp = datetime(1970, 1, 1, tzinfo=UTC)
        numbers = {
            identifier: index for index, identifier in enumerate(sorted(self.devices))
        }
        self.device_numbers = numbers
        states = []
        for satellite in self.satellites.values():
            position = satellite.ecef_position_km
            if position is None:
                continue  # AsterMind's satellite list only contains real supplied geometry.
            if (
                not all(math.isfinite(x) for x in position)
                or math.dist(position, (0, 0, 0)) == 0
            ):
                raise ValueError("ECEF position must be finite and nonzero")
            x, y, z = position
            radius = math.sqrt(x * x + y * y + z * z)
            states.append(
                SatelliteState(
                    satellite_id=satellite.id,
                    name=satellite.id,
                    timestamp=stamp,
                    ecef_position_km=position,
                    latitude_deg=math.degrees(math.asin(z / radius)),
                    longitude_deg=math.degrees(math.atan2(y, x)),
                    altitude_km=radius - 6371,
                )
            )
        kinds = {"gpu": "accelerator", "switch": "switch", "terminal": "isl_terminal"}
        return Topology(
            timestamp=stamp,
            satellites=states,
            nodes=[
                NetworkNode(
                    id=numbers[d.id],
                    kind=kinds[d.kind],
                    satellite_id=d.satellite,
                    accelerator_id=numbers[d.id] if d.kind == "gpu" else None,
                )
                for d in self.devices.values()
            ],
            links=[
                NetworkLink(
                    id=l.id,
                    source=numbers[l.source],
                    target=numbers[l.target],
                    level="intra_satellite"
                    if self.devices[l.source].satellite
                    == self.devices[l.target].satellite
                    else "inter_satellite",
                    bandwidth_gbps=l.bandwidth_gbps,
                    latency_ns=l.latency_ns,
                    enabled=l.enabled,
                )
                for l in self.links
            ],
            accelerator_to_satellite={
                numbers[d.id]: d.satellite
                for d in self.devices.values()
                if d.kind == "gpu"
            },
            metadata={
                "static": self.scenario.link_policy is None,
                "geometry_mode": "static_supplied_or_unspecified",
                "link_state_scope": "candidate_configuration",
                "geometry_unspecified": [
                    s.id for s in self.satellites.values() if s.ecef_position_km is None
                ],
                "cells": [c.model_dump() for c in self.scenario.cells],
            },
        )

    def _compile(self):
        nodes, edges, node_device = [], [], {}
        self.ingress, self.egress = {}, {}
        for identifier, device in self.devices.items():
            number = self.device_numbers[identifier]
            split = device.kind == "gpu" or device.capacity_gbps is not None
            entrance = f"n{number}.in" if split else f"n{number}"
            exit_node = f"n{number}.out" if split else entrance
            self.ingress[identifier], self.egress[identifier] = entrance, exit_node
            for node in dict.fromkeys((entrance, exit_node)):
                nodes.append(node)
                node_device[node] = identifier
            if device.capacity_gbps is not None:
                edges.append(
                    CompiledEdge(
                        entrance,
                        exit_node,
                        device.capacity_gbps,
                        0,
                        f"device:{identifier}",
                    )
                )
            # GPU has no in->out bridge: it cannot be silently used as a router.
        for link in self.links:
            if not link.enabled:
                continue
            directions = [(link.source, link.target)]
            if link.bidirectional:
                directions.append((link.target, link.source))
            for source, target in directions:
                edges.append(
                    CompiledEdge(
                        self.egress[source],
                        self.ingress[target],
                        link.bandwidth_gbps,
                        link.latency_ns,
                        f"link:{link.id}:{source}->{target}",
                    )
                )
        return tuple(nodes), tuple(edges), node_device

    def endpoints(self, source, target):
        if source not in self.devices or target not in self.devices:
            raise ValueError("unknown transfer endpoint")
        if source == target:
            raise ValueError(
                "local operations belong to the local compute/memory model"
            )
        # Starting/ending on a capacity-limited terminal also consumes its gate.
        start = (
            self.egress[source]
            if self.devices[source].kind == "gpu"
            else self.ingress[source]
        )
        end = (
            self.ingress[target]
            if self.devices[target].kind == "gpu"
            else self.egress[target]
        )
        return start, end

    def physical_path(self, native_path):
        result = []
        for node in native_path:
            device = self.node_device[node]
            if not result or result[-1] != device:
                result.append(device)
        return result

    def resources(self, native_path):
        return [self.edge_resource[(a, b)] for a, b in itertools.pairwise(native_path)]

    def view(self, level="devices", identifier=None):
        """Three views of one physical graph, no synthetic reachability edges."""
        if level == "devices":
            if identifier is not None and identifier not in self.satellites:
                raise ValueError("unknown satellite")
            allowed = {
                d.id
                for d in self.devices.values()
                if identifier is None or d.satellite == identifier
            }
            return {
                "devices": [
                    d.model_dump() for d in self.devices.values() if d.id in allowed
                ],
                "links": [
                    l.model_dump()
                    for l in self.links
                    if l.source in allowed and l.target in allowed
                ],
                "external_links": [
                    l.model_dump()
                    for l in self.links
                    if (l.source in allowed) != (l.target in allowed)
                ],
            }
        if level == "satellites":
            if identifier is not None and identifier not in {
                c.id for c in self.scenario.cells
            }:
                raise ValueError("unknown Cell")
            allowed = {
                s
                for s, c in self.satellite_to_cell.items()
                if identifier is None or c == identifier
            }
            return {
                "satellites": [
                    s.model_dump() for s in self.satellites.values() if s.id in allowed
                ],
                "terminals": [
                    d.model_dump()
                    for d in self.devices.values()
                    if d.kind == "terminal" and d.satellite in allowed
                ],
                "links": [
                    l.model_dump()
                    for l in self.links
                    if self.devices[l.source].satellite
                    != self.devices[l.target].satellite
                    and self.devices[l.source].satellite in allowed
                    and self.devices[l.target].satellite in allowed
                ],
                "external_links": [
                    l.model_dump()
                    for l in self.links
                    if (self.devices[l.source].satellite in allowed)
                    != (self.devices[l.target].satellite in allowed)
                ],
            }
        if level == "cells":
            return {
                "cells": [c.model_dump() for c in self.scenario.cells],
                "links": [
                    l.model_dump()
                    for l in self.links
                    if self.satellite_to_cell[self.devices[l.source].satellite]
                    != self.satellite_to_cell[self.devices[l.target].satellite]
                ],
            }
        raise ValueError("level must be devices, satellites or cells")
