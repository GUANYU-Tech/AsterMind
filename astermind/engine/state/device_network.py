"""Device graph projected onto one authoritative orbital NetworkState.

GPU ports are endpoints, switch/terminal gates are shared resources. External
links read the orbital allocator and directional budgets at the actual time;
no physical capacity or acquisition clock is duplicated by this projection.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from collections.abc import Sequence
from math import inf, isfinite
from types import SimpleNamespace
from astermind.engine.clock import Time

from astermind.engine.setup.network_spec import LinkProfile, NetworkNodeSpec
from astermind.engine.state.network import C_KM_PER_S, NetworkState


class ProjectedLinkHistory(Sequence):
    def __init__(self, history, edge, end_s, count):
        self.history, self.edge, self.end_s, self.count = history, edge, end_s, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        row = self.history[index]
        return {**row, **self.edge, "end_s": min(row["end_s"], self.end_s)}


class DeviceNetworkState(NetworkState):
    def __init__(
        self,
        topology,
        horizon_s,
        backend,
        *,
        physical=None,
        bindings=None,
        ground_owners=(),
        prune_terminal_checks=False,
        indexed_fast_routes=False,
    ):
        self.topology, self.physical = topology, physical
        self.indexed_fast_routes = indexed_fast_routes
        self.disabled_device_links = set()
        self.failed_device_links = set()
        self.start_s = physical.start_s if physical is not None else 0.0
        # Static links can have an open interval. The end is not an event time.
        self.end_s = inf if horizon_s is None else Time(horizon_s)
        if self.start_s != 0 or (physical is not None and self.end_s > physical.end_s):
            raise ValueError(
                "device serving horizon must lie inside physical data starting at zero"
            )
        if physical is not None and set(topology.satellites) != set(physical.nodes):
            raise ValueError(
                "device owners must exactly match physical satellite/ground IDs"
            )
        self.nodes = {n: NetworkNodeSpec(n, "compute") for n in topology.native_nodes}
        self.owner = {
            n: topology.devices[d].satellite for n, d in topology.node_device.items()
        }
        self.ground_owners = set(ground_owners)
        if physical is not None:
            self.ground_owners.update(
                n.node_id for n in physical.spec.nodes if n.role == "ground"
            )
        self._directions, self._pairs, self._meta, profiles = {}, {}, {}, {}
        self._physical_ids = {}
        self._bindings = {}
        used_physical = set()
        bindings = dict(bindings or {})
        links = {link.id: link for link in topology.links}
        unknown = set(bindings) - links.keys()
        if unknown:
            raise ValueError(f"unknown device link bindings: {sorted(unknown)}")
        physical_by_pair = {}
        if physical is not None:
            for edge in physical.edges:
                physical_by_pair.setdefault(
                    frozenset((edge["endpoint_a"], edge["endpoint_b"])), []
                ).append(edge)
        edges = []

        def add(a, b, metadata):
            pair = tuple(sorted((a, b)))
            if pair not in self._pairs:
                identity = f"device-edge-{len(edges)}"
                row = {
                    "link_id": identity,
                    "endpoint_a": pair[0],
                    "endpoint_b": pair[1],
                    "kind": "device_gate" if metadata.get("gate") else "device_link",
                    "profile_id": identity,
                }
                self._pairs[pair] = row
                self._meta[identity] = metadata
                edges.append(row)
                profiles[identity] = LinkProfile(metadata["bandwidth_bps"], 0.0)
            row = self._pairs[pair]
            self._directions[row["link_id"], a, b] = metadata
            if metadata.get("physical_link_id"):
                self._physical_ids.setdefault(metadata["physical_link_id"], set()).add(
                    row["link_id"]
                )

        for identifier, device in topology.devices.items():
            if device.capacity_gbps is not None and (not indexed_fast_routes or device.kind == "switch"):
                add(
                    topology.ingress[identifier],
                    topology.egress[identifier],
                    {
                        "gate": True,
                        "bandwidth_bps": device.capacity_gbps * 1e9,
                        "latency_s": 0.0,
                        "resource_id": f"device:{identifier}",
                        "owners": (device.satellite,),
                    },
                )
        for link in topology.links:
            if not link.enabled:
                continue
            a, b = topology.devices[link.source], topology.devices[link.target]
            metadata = {
                "bandwidth_bps": link.bandwidth_gbps * 1e9,
                "latency_s": Time.from_ns(link.latency_ns),
                "resource_id": f"link:{link.id}",
                "device_link_id": link.id,
                "owners": (a.satellite, b.satellite),
                "schedule": tuple(getattr(link, "capacity_schedule_ns", ())),
            }
            if a.satellite != b.satellite and physical is not None:
                if link.availability_windows is not None:
                    raise ValueError(
                        "physical contact windows must come from the verified physical dataset"
                    )
                candidates = physical_by_pair.get(
                    frozenset((a.satellite, b.satellite)), ()
                )
                selected = bindings.get(link.id)
                if selected is None and len(candidates) == 1:
                    selected = candidates[0]["link_id"]
                if selected is None or selected not in {
                    e["link_id"] for e in candidates
                }:
                    raise ValueError(
                        f"external link {link.id} requires an unambiguous physical link binding"
                    )
                if selected in used_physical:
                    raise ValueError(
                        "one physical link cannot be copied into multiple device links"
                    )
                used_physical.add(selected)
                metadata["physical_link_id"] = selected
                self._bindings[link.id] = selected
            elif link.id in bindings:
                raise ValueError(
                    "physical bindings require an external link and a physical dataset"
                )
            directions = [(link.source, link.target)]
            if link.bidirectional:
                directions.append((link.target, link.source))
            for source, target in directions:
                add(
                    topology.egress[source],
                    topology.ingress[target],
                    {
                        **metadata,
                        "source_owner": topology.devices[source].satellite,
                        "target_owner": topology.devices[target].satellite,
                    },
                )
        self.edges = tuple(edges)
        if physical is not None and used_physical != set(physical.edge_by_id):
            raise ValueError(
                "device links must cover every physical candidate link exactly once"
            )
        self.edge_by_id = {e["link_id"]: e for e in edges}
        base = physical.spec if physical is not None else SimpleNamespace()
        self.spec = SimpleNamespace(
            **{
                **vars(base),
                "nodes": tuple(self.nodes.values()),
                "transport_model": backend,
                "link_profiles": profiles,
                "fault_timing": getattr(base, "fault_timing", "continuous"),
            }
        )
        self.pool_capacities = (
            dict(physical.pool_capacities) if physical is not None else {}
        )
        self.calendars, self.shared_load, self.shared_pool_load = {}, {}, {}
        self._project()
        self._validate_terminal_allocation(prune=prune_terminal_checks)

    def _validate_terminal_allocation(self, *, prune=False):
        if self.physical is None:
            return
        links = self.topology.links
        candidates = None
        if prune:
            # A terminal with no more possible incident links than slots
            # cannot overflow in any frame, even if every link is active.
            bound = tuple(link for link in links if link.id in self._bindings)
            counts = Counter(terminal for link in bound for terminal in (link.source, link.target))
            candidates = {terminal for terminal, count in counts.items()
                          if count > self.topology.devices[terminal].max_external_links}
            if not candidates:
                return
            links = tuple(link for link in bound
                          if link.source in candidates or link.target in candidates)
        checks = sorted(
            {
                self.start_s,
                *(
                    t
                    for rows in self.physical.link_times.values()
                    for t in rows
                    if self.start_s <= t < self.end_s
                ),
            }
        )
        for time_s in checks:
            used = {}
            for link in links:
                physical_id = self._bindings.get(link.id)
                if (
                    physical_id is None
                    or self.physical.link_at(physical_id, time_s)["state"] == "DOWN"
                ):
                    continue
                for terminal in (link.source, link.target):
                    if candidates is not None and terminal not in candidates:
                        continue
                    used[terminal] = used.get(terminal, 0) + 1
                    if (
                        used[terminal]
                        > self.topology.devices[terminal].max_external_links
                    ):
                        raise ValueError(
                            f"physical allocation exceeds device terminal slots: {terminal}"
                        )

    def _project(self):
        physical = self.physical
        self.revision = physical.revision if physical is not None else 0
        self.active_faults = {}
        if physical is not None:
            for index, fault in physical.active_faults.items():
                if fault["target_type"] == "node":
                    for node, owner in self.owner.items():
                        if owner == fault["target_id"]:
                            self.active_faults[index, node] = {
                                **fault,
                                "target_id": node,
                            }
                else:
                    self.active_faults[index] = fault
        boundaries = {self.start_s, self.end_s}
        indexed = (self.indexed_fast_routes and physical is not None
                   and physical.physical_attributes.link_store)
        if physical is not None:
            boundaries.update(t for t in physical.environment_times if t < self.end_s)
            changes = (indexed.boundaries if indexed and indexed.boundaries is not None
                       else (t for rows in physical.link_times.values() for t in rows))
            boundaries.update(t for t in changes if t < self.end_s)
            boundaries.update(
                t
                for times in physical.attribute_boundaries.values()
                for t in times
                if t < self.end_s
            )
        for data in self._directions.values():
            boundaries.update(
                Time.from_ns(at) for at, _ in data.get("schedule", ()) if Time.from_ns(at) < self.end_s
            )
        ordered = tuple(sorted(boundaries))
        self.environment_times = ordered if isfinite(self.end_s) else ordered[:-1]
        self.frames = [
            {"time_s": a, "end_s": b, "revision": self.revision}
            for a, b in zip(ordered, ordered[1:])
        ]
        self.frame_times = [f["time_s"] for f in self.frames]
        self.link_frames, self.link_times, self.events = {}, {}, []
        for edge in self.edges:
            metadata = self._meta[edge["link_id"]]
            physical_id = metadata.get("physical_link_id")
            if physical_id and indexed:
                times = physical.link_times[physical_id]
                self.link_frames[edge["link_id"]] = ProjectedLinkHistory(
                    physical.link_frames[physical_id], edge, self.end_s, bisect_left(times, self.end_s))
                self.link_times[edge["link_id"]] = times
                continue
            if physical_id:
                rows = [
                    {**row, **edge, "end_s": min(row["end_s"], self.end_s)}
                    for row in physical.link_frames[physical_id]
                    if row["time_s"] < self.end_s
                ]
            else:
                rows = [
                    {
                        **edge,
                        "time_s": self.start_s,
                        "end_s": self.end_s,
                        "state": "UP",
                        "reason": "declared_device_resource",
                        "setup_start_s": 0.0,
                        "ready_s": 0.0,
                    }
                ]
            self.link_frames[edge["link_id"]] = rows
            self.link_times[edge["link_id"]] = tuple(row["time_s"] for row in rows)
        if physical is not None:
            self.events = [
                {**event, "link_id": device_id, "revision": self.revision}
                for event in physical.events
                if event["time_s"] <= self.end_s
                for device_id in sorted(self._physical_ids.get(event["link_id"], ()))
            ]

    def metadata(self, edge, source=None, target=None):
        if source is not None:
            return self._directions.get((edge["link_id"], source, target))
        return self._meta[edge["link_id"]]

    def use_link_state_window(self, time_s):
        if self.indexed_fast_routes:
            self.physical.use_link_state_window(time_s)

    def pools_for_hop(self, source, target):
        edge = self._pairs.get(tuple(sorted((source, target))))
        data = self.metadata(edge, source, target) if edge is not None else None
        if not data or not data.get("physical_link_id"):
            return ()
        return self.physical.pools_for_hop(data["source_owner"], data["target_owner"])

    def allows_hop(self, source, target):
        edge = self._pairs.get(tuple(sorted((source, target))))
        return (
            edge is not None and (edge["link_id"], source, target) in self._directions
        )

    def route_node_allowed(self, node, source, target):
        owner = self.owner[node]
        return owner not in self.ground_owners or owner in {
            self.owner[source],
            self.owner[target],
        }

    def transport_options(self):
        return {
            "shared_pools": self.pool_capacities,
            "link_pools": {
                key: self.pools_for_hop(key[1], key[2]) for key in self._directions
            },
        }

    def sample_time(self, time_s):
        return (
            self.physical.sample_time(time_s) if self.physical is not None else time_s
        )

    def distance(self, edge, time_s):
        data = self.metadata(edge)
        if data.get("physical_link_id"):
            return self.physical.distance(
                self.physical.edge_by_id[data["physical_link_id"]], time_s
            )
        return data["latency_s"] * C_KM_PER_S

    def directional_attributes(self, edge, source, target, time_s):
        data = self.metadata(edge, source, target)
        if data is None:
            return {"bandwidth_bps": 0.0, "direction_blocked": True}
        rate = data["bandwidth_bps"]
        for at, multiplier in data.get("schedule", ()):
            if Time.from_ns(at) <= time_s:
                rate = data["bandwidth_bps"] * multiplier
        if data.get("physical_link_id"):
            original = self.physical.edge_by_id[data["physical_link_id"]]
            attrs = self.physical.directional_attributes(
                original, data["source_owner"], data["target_owner"], time_s
            )
            return {**attrs, "bandwidth_bps": min(rate, attrs["bandwidth_bps"])}
        return {
            "bandwidth_bps": rate,
            "capacity_model": "declared_device_link",
            "budget_evaluated": True,
        }

    def link_at(self, link_id, time_s):
        edge = self.edge_by_id[link_id]
        data = self.metadata(edge)
        if data.get("physical_link_id"):
            state = self.physical.link_at(data["physical_link_id"], time_s)
        else:
            state = self.link_frames[link_id][0]
        state = {
            **state,
            **edge,
            "bandwidth_bps": state.get("bandwidth_bps", data["bandwidth_bps"]),
            "propagation_delay_s": self.distance(edge, time_s) / C_KM_PER_S,
        }
        if self.faulted(edge):
            state.update(state="DOWN", reason="injected_fault")
        if data.get('device_link_id') in self.disabled_device_links:
            state.update(state='DOWN', reason='manual_device_link_disconnected')
        return state

    def _edge_transfer(
        self, link_id, source, target, time_s, payload_bytes, allow_wait
    ):
        if self.metadata(self.edge_by_id[link_id], source, target) is None:
            return None
        return super()._edge_transfer(
            link_id, source, target, time_s, payload_bytes, allow_wait
        )

    def faulted(self, edge, time_s=None):
        if self.metadata(edge).get('device_link_id') in self.failed_device_links:
            return True
        if self.physical is None:
            return False
        data = self.metadata(edge)
        if data.get("physical_link_id") and self.physical.faulted(
            self.physical.edge_by_id[data["physical_link_id"]], time_s
        ):
            return True
        faults = self.physical.active_faults.values()
        return any(
            f["target_type"] == "node" and f["target_id"] in data["owners"]
            for f in faults
        )

    def observe_faults(self, observations, time_s):
        if self.physical is None:
            raise ValueError("runtime physical faults require a physical dataset")
        self.physical.observe_faults(observations, time_s)
        self._project()
        self._validate_terminal_allocation()

    def position(self, node, time_s):
        owner = self.owner[node]
        if self.physical is not None:
            return self.physical.position(owner, time_s)
        return self.topology.satellites[owner].ecef_position_km
