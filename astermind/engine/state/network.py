"""Event-timed terminal allocation and physical link attributes over Orbit traces."""

from __future__ import annotations

import bisect
import heapq
import itertools
import math
from copy import copy, deepcopy
from typing import Any

from astermind.engine.association import select_ground_links
from astermind.engine.topology import candidate_links
from astermind.engine.setup.network_spec import NetworkSpec
from astermind.engine.setup.radio_capacity import capacity_pools, directional_profile, hop_pools, radio_parameters

C_KM_PER_S = 299792.458


class NetworkUnavailable(ValueError):
    pass


class NetworkState:
    """Immutable topology timeline plus directed, full-duplex transmission calendars.

    Geometry is held at the last SGP4 sample. Acquisition and traffic events
    retain exact event times; this class never interpolates physical positions.
    """

    def __init__(self, trace: Any, spec: NetworkSpec, gateways=(), sun_positions=(), *, physical_state=None, _defer_compile=False):
        self.physical_attributes = None
        prepared = None
        if physical_state is not None:
            from astermind.engine.setup.sun_geometry import SUN_GEOMETRY_VERSION
            from astermind.engine.state.physical_dataset import PhysicalAttributes, physical_spec
            self.physical_attributes = PhysicalAttributes(physical_state)
            prepared = self.physical_attributes.metadata
            from astermind.engine.setup.faults import physical_network_settings
            if prepared['trace_id'] != trace.trace_id or physical_network_settings(prepared['network']) != physical_spec(spec):
                raise ValueError('physical dataset does not match trace/network configuration')
            if (any(p.sun_exclusion_angle_deg is not None for p in spec.link_profiles.values())
                    and prepared.get('sun_geometry_version') != SUN_GEOMETRY_VERSION):
                raise ValueError('physical dataset uses obsolete Sun visibility; regenerate physical data in step one')
            sun_positions = [{'time_s': t, 'position_itrs_km': p} for t, p in prepared['sun_positions']]
        self.trace, self.spec = trace, spec
        self.nodes = {n.node_id: n for n in spec.nodes}
        self.pool_capacities = capacity_pools(self.nodes)
        self.edges = candidate_links(spec)
        for edge in self.edges:
            hop_pools(self.nodes, edge['endpoint_a'], edge['endpoint_b'])
            hop_pools(self.nodes, edge['endpoint_b'], edge['endpoint_a'])
        if prepared is None and not sun_positions and any(
            p.sun_exclusion_angle_deg is not None for p in spec.link_profiles.values()
        ):
            from astermind.engine.setup.network_geometry import sun_positions_from_trace

            sun_positions = sun_positions_from_trace(trace)
        self.sun_positions = sorted(
            (float(row["time_s"]), tuple(row["position_itrs_km"]))
            for row in sun_positions
        )
        if (
            any(
                p.sun_exclusion_angle_deg is not None
                for p in spec.link_profiles.values()
            )
            and not self.sun_positions
        ):
            raise ValueError("Sun exclusion requires a time-aligned Sun ephemeris")
        self.start_s, self.end_s = (
            float(trace.time["start_s"]),
            float(trace.time["end_s"]),
        )
        self.samples: dict[str, list[tuple[float, tuple[float, ...]]]] = {}
        for row in trace.ephemeris_samples:
            self.samples.setdefault(str(row["satellite_id"]), []).append(
                (float(row["time_s"]), tuple(float(x) for x in row["position_itrs_km"]))
            )
        satellite_ids = {n.node_id for n in spec.nodes if n.role != "ground"}
        if satellite_ids != {str(n["satellite_id"]) for n in trace.satellite_catalog}:
            raise ValueError(
                "network satellite IDs must exactly match the physical Orbit catalog"
            )
        self.ground_positions = {}
        for row in gateways:
            node_id = str(row.get("ingress_id") or row.get("gateway_id") or "")
            self.ground_positions[node_id] = ground_position_km(row)
        for node in spec.nodes:
            if node.role == "ground":
                if node.node_id not in self.ground_positions:
                    raise ValueError(f"ground coordinates missing: {node.node_id}")
            else:
                rows = sorted(self.samples.get(node.node_id, ()))
                if not rows or rows[0][0] > self.start_s or rows[-1][0] < self.end_s:
                    raise ValueError(
                        f"ephemeris does not cover network horizon: {node.node_id}"
                    )
                if len({t for t, _ in rows}) != len(rows):
                    raise ValueError("duplicate ephemeris time")
                self.samples[node.node_id] = rows
        self.sample_times = {
            node: tuple(t for t, _ in rows) for node, rows in self.samples.items()
        }
        self.environment_times = tuple(
            sorted(
                {
                    self.start_s,
                    self.end_s,
                    *(t for times in self.sample_times.values() for t in times),
                }
            )
        )
        self.sun_times = tuple(t for t, _ in self.sun_positions)
        self.windows: dict[tuple[str, str], list[dict]] = {}
        for row in trace.isl_windows:
            key = tuple(sorted((str(row["endpoint_a"]), str(row["endpoint_b"]))))
            self.windows.setdefault(key, []).append(row)
        for row in trace.sgl_windows:
            key = tuple(sorted((str(row["ingress_id"]), str(row["satellite_id"]))))
            self.windows.setdefault(key, []).append(row)
        # Adaptive access pairs without any contact in this horizon cannot be
        # selected. Keep fixed ISLs as topology, even when physically blocked.
        self.candidate_edge_ids = {edge['link_id'] for edge in self.edges}
        self.edges = tuple(edge for edge in self.edges if edge['kind'] not in {'sgl', 'cross_layer'} or
                           any(float(w['start_s']) < self.end_s and float(w['end_s']) > self.start_s
                               for w in self.windows.get((edge['endpoint_a'], edge['endpoint_b']), ())))
        self._set_attribute_boundaries()
        self.frames: list[dict] = []
        self.events: list[dict] = []
        self.calendars: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
        self.shared_load: dict[tuple[str, str, str], int] = {}
        self.shared_pool_load = {}
        self.active_faults = {}
        self.observed_faults = {}
        self.fault_boundaries = set()
        self.revision = 0
        self.manual_disabled_links = set()
        self.added_edge_ids = set()
        from astermind.engine.setup.faults import validate_fault_boundaries
        validate_fault_boundaries(spec, self.environment_times)
        self._compile_pending = _defer_compile and prepared is None
        if self._compile_pending:
            self.link_frames, self.link_times = {}, {}
            self.allocation_checks = {'passed': True, 'max_pool_usage': {}}
        elif prepared is None:
            self._compile()
        else:
            if prepared['ground_positions'] != {k: list(v) for k, v in self.ground_positions.items()}:
                raise ValueError('physical dataset ground positions differ from runtime gateways')
            for name in ('frames', 'events', 'link_frames', 'allocation_checks'):
                setattr(self, name, prepared[name])
            self.edge_by_id = {edge['link_id']: edge for edge in self.edges}
            self.link_times = {key: [row['time_s'] for row in rows] for key, rows in self.link_frames.items()}
        if any(f['target_type'] == 'link' and f['target_id'] not in self.candidate_edge_ids for f in spec.faults):
            raise ValueError("unknown fault link")
        self.frame_times = [f["time_s"] for f in self.frames]

    def _set_attribute_boundaries(self):
        self.attribute_boundaries = {}
        if self.physical_attributes is not None:
            for edge in self.edges:
                profiles = (self.spec.link_profiles[edge.get('profile_id',edge['kind'])],
                            *(directional_profile(self.nodes,self.spec.link_profiles,edge,source)
                              for source in (edge['endpoint_a'],edge['endpoint_b'])))
                boundaries = {t for profile in profiles for t,_ in profile.capacity_schedule}
                boundaries.update(float(w[k]) for w in self.windows.get((edge['endpoint_a'],edge['endpoint_b']),())
                                  for k in ('start_s','end_s'))
                self.attribute_boundaries[edge['link_id']] = tuple(sorted(boundaries))
    def save_physical_state(self, path, *, workers=1, progress=None):
        """Persist initial acquisition state and backend-independent budgets."""
        from astermind.engine.state.physical_dataset import write_physical_state
        if self.observed_faults or self.calendars or self.shared_load or self.shared_pool_load:
            raise ValueError('physical preparation requires a fresh network without runtime observations/traffic')
        write_physical_state(self, path, workers=workers, progress=progress)

    @staticmethod
    def open_physical_attributes(path):
        """Open immutable attributes for offline snapshots without a full trace."""
        from astermind.engine.state.physical_dataset import PhysicalAttributes
        return PhysicalAttributes(path)

    def position(self, node_id: str, time_s: float) -> tuple[float, ...]:
        if not self.start_s <= time_s <= self.end_s:
            raise NetworkUnavailable("position_time_outside_trace")
        if node_id in self.ground_positions:
            return self.ground_positions[node_id]
        rows = self.samples[node_id]
        index = bisect.bisect_right(self.sample_times[node_id], time_s) - 1
        if index < 0 or time_s > rows[-1][0]:
            raise NetworkUnavailable("ephemeris_time_uncovered")
        return rows[index][1]

    def sample_time(self, time_s):
        return self.environment_times[
            max(0, bisect.bisect_right(self.environment_times, time_s) - 1)
        ]

    def _saved_attributes(self, edge, time_s):
        # Internal read-only access keeps distance lookups cheap. External readers
        # receive independent values through PhysicalAttributes.at().
        return self.physical_attributes._cached_at(edge['link_id'], self._attribute_time(edge, time_s))

    def _attribute_time(self, edge, time_s):
        # Fractional arrivals in the same physical interval share one cache
        # key. Exact capacity/contact boundaries are still kept separately.
        boundaries = self.attribute_boundaries.get(edge['link_id'], ())
        index = bisect.bisect_right(boundaries,time_s)-1
        canonical = max(self.sample_time(time_s), boundaries[index] if index>=0 else self.start_s)
        return canonical

    def distance(self, edge: dict, time_s: float) -> float:
        if self.physical_attributes is not None and edge['link_id'] not in self.added_edge_ids:
            return self._saved_attributes(edge,time_s)['geometry']['distance_km']
        return math.dist(
            self.position(edge["endpoint_a"], time_s),
            self.position(edge["endpoint_b"], time_s),
        )

    def sun_separation(self, edge: dict, time_s: float, *, receiver_id=None) -> float | None:
        """Use direct solar incidence at each receiver, never through Earth.

        The shared bidirectional link retains its conservative exclusion policy:
        if either receiver can see the Sun inside the exclusion cone, the pair
        is unavailable. A caller may inspect one receiving direction explicitly.
        """
        from astermind.engine.setup.sun_geometry import receiver_sun_separation

        if receiver_id is not None and receiver_id not in (edge['endpoint_a'], edge['endpoint_b']):
            raise ValueError('solar receiver must be a link endpoint')
        if not self.sun_positions:
            return None
        index = max(
            0, bisect.bisect_right(self.sun_times, self.sample_time(time_s)) - 1
        )
        _, sun = self.sun_positions[index]
        angles = []
        a, b = (
            self.position(edge["endpoint_a"], time_s),
            self.position(edge["endpoint_b"], time_s),
        )
        for identity, observer, transmitter in ((edge['endpoint_a'], a, b), (edge['endpoint_b'], b, a)):
            if receiver_id is not None and identity != receiver_id:
                continue
            angle = receiver_sun_separation(observer, transmitter, sun)
            if angle is not None:
                angles.append(angle)
        return min(angles) if angles else 180.0

    def fork(self) -> NetworkState:
        result = copy(self)
        result.calendars = {key: list(value) for key, value in self.calendars.items()}
        result.shared_load = dict(self.shared_load)
        result.shared_pool_load = dict(self.shared_pool_load)
        result.active_faults = dict(self.active_faults)
        result.observed_faults = {k: dict(v) for k, v in self.observed_faults.items()}
        result.fault_boundaries = set(self.fault_boundaries)
        result.manual_disabled_links = set(self.manual_disabled_links)
        result.added_edge_ids = set(self.added_edge_ids)
        result.windows = dict(self.windows)
        if hasattr(self, '_route_cache'):
            result._route_cache = self._route_cache.fork()
        return result

    def observe_fault(self, index, fault, time_s, *, restored=False):
        """Reallocate from observed outages, never from their future repair time.

        Release is immediate. Boundary-mode acquisition starts at this boundary;
        unaffected contacts keep their original acquisition clock. Replaying only
        observed boundaries makes both forecasts and forks causal.
        """
        self.observe_faults(((index, fault, restored),), time_s)

    def observe_faults(self, observations, time_s):
        """Apply one boundary's complete outage batch before reallocating once."""
        if self.spec.fault_timing == 'boundary' and not any(math.isclose(time_s, t, rel_tol=0, abs_tol=1e-8) for t in self.environment_times):
            raise ValueError('故障与恢复只能在时间片边界生效')
        for index, fault, restored in observations:
            if restored:
                self.active_faults.pop(index, None)
                self.observed_faults[index]['end_s'] = time_s
            else:
                self.active_faults[index] = fault
                # The simulation horizon is not an observed repair. Keep the
                # outage active even at its exact end until a repair is seen.
                self.observed_faults[index] = {**fault, 'end_s': math.inf}
        self.fault_boundaries.add(time_s)
        self.revision += 1
        self._compile(from_s=time_s)
        self.frame_times = [f['time_s'] for f in self.frames]

    def faulted(self, edge, time_s=None):
        faults = (self.active_faults.values() if time_s is None else
                  (f for f in (*self.spec.faults, *self.observed_faults.values()) if f['start_s'] <= time_s < f['end_s']))
        return any((f['target_type'] == 'link' and f['target_id'] == edge['link_id']) or
                   (f['target_type'] == 'node' and f['target_id'] in
                    (edge['endpoint_a'], edge['endpoint_b'])) for f in faults)

    def fault_observations(self):
        return sorted((t, index, restored, fault) for index, fault in enumerate(self.spec.faults)
                      for t, restored in ((fault['start_s'], False), (fault['end_s'], True))
                      if self.start_s <= t <= self.end_s)

    def audit_timeline(self):
        """Replay all observations for export; never use this as an online oracle."""
        result = self.fork()
        for time_s, index, restored, fault in self.fault_observations():
            result.observe_fault(index, fault, time_s, restored=restored)
        return result

    def _compile(self, from_s=None) -> None:
        """Store only per-link changes; never copy the whole graph per event."""
        compile_start = self.start_s if from_s is None else from_s
        old_frames = getattr(self, 'link_frames', {})
        seed = {}
        if from_s is not None:
            for link_id, history in old_frames.items():
                i = bisect.bisect_right(self.link_times[link_id], compile_start) - 1
                if i >= 0 and history[i]['state'] != 'DOWN':
                    seed[link_id] = {k: history[i][k] for k in ('setup_start_s', 'ready_s')}
        self.events = [dict(e) for e in self.events if e['time_s'] < compile_start]
        self.link_frames = {edge['link_id']: [dict(row) for row in old_frames.get(edge['link_id'], ())
                                            if row['time_s'] < compile_start] for edge in self.edges}
        self.edge_by_id = {edge["link_id"]: edge for edge in self.edges}
        self.allocation_checks = {"passed": True, "max_pool_usage": {}}
        queue = []
        # Prepare one edge at a time, retaining only changes of physical
        # eligibility, not edge_count × sample_count attribute dictionaries.
        base_times = self.environment_times
        for t in base_times[:-1]:
            if t >= compile_start:
                heapq.heappush(queue, (t, -1, "", None))
        for link_id, state in seed.items():
            if compile_start < state['ready_s'] < self.end_s:
                heapq.heappush(queue, (state['ready_s'], 1, link_id, None))
        for edge in self.edges:
            link_id = edge["link_id"]
            profile = self.spec.link_profiles[edge.get("profile_id", edge["kind"])]
            windows = self.windows.get((edge["endpoint_a"], edge["endpoint_b"]), ())
            ticks = {compile_start, self.end_s}
            ticks.update(self.fault_boundaries)
            ticks.update(next((t for t in base_times if t > b), self.end_s)
                         for b in self.fault_boundaries)
            for window in windows:
                low = max(self.start_s, float(window["start_s"]))
                high = min(self.end_s, float(window["end_s"]))
                ticks.update((low, high))
                ticks.update(
                    base_times[
                        bisect.bisect_right(base_times, low) : bisect.bisect_left(
                            base_times, high
                        )
                    ]
                )
            changes = []
            for t in sorted(t for t in ticks if compile_start <= t < self.end_s):
                attr = self._geometry_attributes(edge, t)
                if not changes or changes[-1]["reason"] != attr["reason"]:
                    changes.append({"time_s": t, **attr})
            for i, attr in enumerate(changes):
                attr["available_until_s"] = (
                    changes[i + 1]["time_s"] if i + 1 < len(changes) else self.end_s
                )
                heapq.heappush(queue, (attr["time_s"], 0, link_id, attr))
            schedules = (profile, *(directional_profile(self.nodes, self.spec.link_profiles, edge, source)
                                   for source in (edge['endpoint_a'], edge['endpoint_b'])))
            for t in sorted({t for p in schedules for t, _ in p.capacity_schedule}):
                if compile_start < t < self.end_s:
                    heapq.heappush(queue, (t, 2, link_id, None))
        eligible = {}
        states = seed
        latest = {}
        all_times = {self.start_s, self.end_s, compile_start,
                     *(f['time_s'] for f in self.frames if f['time_s'] < compile_start)}
        while queue:
            t = queue[0][0]
            batch = []
            while queue and queue[0][0] == t:
                batch.append(heapq.heappop(queue))
            if t >= self.end_s:
                break
            all_times.add(t)
            changed = set()
            refresh = any(kind == -1 for _, kind, _, _ in batch)
            for _, kind, link_id, attr in batch:
                if kind == -1:
                    continue
                if kind == 0:
                    latest[link_id] = attr
                    changed.add(link_id)
                    if attr["reason"] is None:
                        eligible[link_id] = attr
                    else:
                        eligible.pop(link_id, None)
                elif kind == 2 or (
                    link_id in states and states[link_id]["ready_s"] == t
                ):
                    changed.add(link_id)
            # Only currently eligible edges participate in allocation. Refresh
            # their sampled weights without retaining a dense attribute history.
            if refresh or any(kind == 2 for _, kind, _, _ in batch):
                for link_id, old in tuple(eligible.items()):
                    attr = {
                        **old,
                        **self._geometry_attributes(self.edge_by_id[link_id], t),
                    }
                    eligible[link_id] = latest[link_id] = attr
            for link_id in list(states):
                if link_id not in eligible:
                    states.pop(link_id)
                    changed.add(link_id)
            previous_dynamic = {}
            if self.spec.handover_policy == 'quality_first' and t not in self.fault_boundaries:
                previous_dynamic = {k:states.pop(k) for k in list(states)
                                    if latest[k]['kind'] in {'sgl','cross_layer'}}
            retained_sgl = [latest[k] for k in states if latest[k]['kind'] == 'sgl']
            selected_sgl = select_ground_links(
                self, [e for k,e in eligible.items() if e['kind'] == 'sgl' and k not in states],
                t, retained=retained_sgl)
            used = {}
            for link_id in states:
                edge = latest[link_id]
                pool = "sgl" if edge["kind"] == "sgl" else "isl"
                for node in (edge["endpoint_a"], edge["endpoint_b"]):
                    used[node, pool] = used.get((node, pool), 0) + 1

            def priority(edge, t=t):
                score = edge['distance_km']
                policy = (self.spec.cross_layer_access_policy if edge['kind']=='cross_layer'
                          else self.spec.ground_access_policy if edge['kind']=='sgl' else 'nearest_distance')
                if policy == 'longest_available':
                    score = -(edge['available_until_s']-t)
                elif edge['kind']=='cross_layer' and policy=='highest_capacity':
                    a,b=edge['endpoint_a'],edge['endpoint_b']
                    score=-min(self.directional_attributes(edge,a,b,t)['bandwidth_bps'],
                               self.directional_attributes(edge,b,a,t)['bandwidth_bps'])
                return (1.0 if edge.get('topology_patch') else 0.0,score,edge['distance_km'],edge['link_id'])

            for edge in sorted(eligible.values(), key=priority):
                link_id = edge["link_id"]
                if link_id in states:
                    continue
                if edge['kind'] == 'sgl' and link_id not in selected_sgl:
                    continue
                pool = "sgl" if edge["kind"] == "sgl" else "isl"
                endpoints = edge["endpoint_a"], edge["endpoint_b"]
                if any(
                    used.get((n, pool), 0)
                    >= getattr(self.nodes[n], pool + "_terminals")
                    for n in endpoints
                ):
                    if (
                        not self.link_frames[link_id]
                        or self.link_frames[link_id][-1]["state"] != "DOWN"
                    ):
                        changed.add(link_id)
                    continue
                old = previous_dynamic.get(link_id)
                profile = self.spec.link_profiles[edge.get("profile_id", edge["kind"])]
                states[link_id] = old or {
                    "setup_start_s": t,
                    "ready_s": t + profile.setup_time_s,
                }
                if old is None:
                    changed.add(link_id)
                    if t < states[link_id]["ready_s"] < self.end_s:
                        heapq.heappush(queue, (states[link_id]['ready_s'], 1, link_id, None))
                for n in endpoints:
                    key = n, pool
                    used[key] = used.get(key, 0) + 1
                    if used[key] > getattr(self.nodes[n], pool + "_terminals"):
                        raise RuntimeError("terminal over-allocation")
                    name = n + ":" + pool
                    self.allocation_checks["max_pool_usage"][name] = max(
                        used[key], self.allocation_checks["max_pool_usage"].get(name, 0)
                    )
            changed.update(k for k in previous_dynamic if k not in states)
            for link_id in changed:
                attr = {
                    **latest[link_id],
                    **self._geometry_attributes(self.edge_by_id[link_id], t),
                }
                state = states.get(link_id)
                record = {
                    **attr,
                    **(state or {}),
                    "time_s": t,
                    "state": "UP"
                    if state and state["ready_s"] <= t
                    else "SETTING_UP"
                    if state
                    else "DOWN",
                    "reason": attr["reason"]
                    if attr["reason"] is not None
                    else None
                    if state
                    else "terminal_unavailable",
                }
                history = self.link_frames[link_id]
                prior = history[-1] if history else {}
                def signature(edge):
                    return (edge.get('state'), edge.get('reason'), edge.get('setup_start_s'))
                if signature(record) != signature(prior):
                    self.events.append(
                        {
                            k: record.get(k)
                            for k in (
                                "time_s",
                                "link_id",
                                "state",
                                "reason",
                                "setup_start_s",
                                "ready_s",
                            )
                        }
                    )
                if history and history[-1]["time_s"] == t:
                    history[-1] = record
                else:
                    history.append(record)
        for link_id, history in self.link_frames.items():
            for i, row in enumerate(history):
                row["end_s"] = (
                    history[i + 1]["time_s"] if i + 1 < len(history) else self.end_s
                )
            if history and history[-1]["state"] != "DOWN":
                self.events.append(
                    {
                        "time_s": self.end_s,
                        "link_id": link_id,
                        "state": "DOWN",
                        "reason": "horizon_end",
                    }
                )
        self.events.sort(key=lambda e: (e["time_s"], e["link_id"]))
        self.link_times = {
            key: [row["time_s"] for row in history]
            for key, history in self.link_frames.items()
        }
        ordered = sorted(all_times)
        self.frames = [{"time_s": a, "end_s": b} for a, b in itertools.pairwise(ordered)]

    def _geometry_attributes(self, edge, time_s):
        if self.physical_attributes is not None and edge['link_id'] not in self.added_edge_ids:
            result = self.physical_attributes.geometry_at(edge['link_id'], self._attribute_time(edge, time_s))
            if any(f['start_s'] <= time_s < f['end_s'] and
                   ((f['target_type'] == 'link' and f['target_id'] == edge['link_id']) or
                    (f['target_type'] == 'node' and f['target_id'] in (edge['endpoint_a'], edge['endpoint_b'])))
                   for f in self.observed_faults.values()):
                result['reason'] = 'injected_fault'
            if edge['link_id'] in self.manual_disabled_links:
                result['reason'] = 'manual_disconnected'
            return result
        profile = self.spec.link_profiles[edge.get("profile_id", edge["kind"])]
        windows = self.windows.get((edge["endpoint_a"], edge["endpoint_b"]), ())
        window = next(
            (w for w in windows if float(w["start_s"]) <= time_s < float(w["end_s"])),
            None,
        )
        distance = self.distance(edge, time_s)
        probe = time_s
        reason = None
        if window is None:
            reason = "geometry_blocked"
        elif (
            profile.max_distance_km is not None
            and self.distance(edge, probe) > profile.max_distance_km
        ):
            reason = "out_of_range"
        if reason is None and edge.get('topology_patch'):
            a = self.position(edge['endpoint_a'], time_s)
            b = self.position(edge['endpoint_b'], time_s)
            delta = tuple(y-x for x,y in zip(a,b))
            length2 = sum(x*x for x in delta)
            fraction = max(0., min(1., -sum(x*y for x,y in zip(a,delta)) / length2)) if length2 else 0.
            clearance_radius = math.sqrt(sum((x+fraction*y)**2 for x,y in zip(a,delta)))
            if clearance_radius < 6478.135:
                reason = 'patch_atmosphere_blocked'
        angle = (
            self.sun_separation(edge, probe)
            if reason is None and profile.sun_exclusion_angle_deg is not None
            else None
        )
        if (
            reason is None
            and profile.sun_exclusion_angle_deg is not None
            and angle < profile.sun_exclusion_angle_deg
        ):
            reason = "sun_exclusion"
        evaluate_budget = reason is None
        if any(f['start_s'] <= time_s < f['end_s'] and
               ((f['target_type'] == 'link' and f['target_id'] == edge['link_id']) or
                (f['target_type'] == 'node' and f['target_id'] in
                 (edge['endpoint_a'], edge['endpoint_b'])))
               for f in self.observed_faults.values()):
            reason = 'injected_fault'
        attrs = (self._budget_attributes(profile, distance, time_s) if evaluate_budget else
                 {'bandwidth_bps': 0.0, 'budget_bandwidth_bps': 0.0,
                      'capacity_model': profile.capacity_model, 'rate_scope': 'per_direction',
                      'budget_evaluated': False})
        attrs['bandwidth_bps'] = min([attrs['bandwidth_bps'], *self.optical_limits(edge)])
        if reason is None and attrs.get('link_feasible') is False:
            reason = 'link_budget_infeasible'
        if edge['link_id'] in self.manual_disabled_links:
            reason = 'manual_disconnected'
        return {
            **edge,
            "distance_km": distance,
            "propagation_delay_s": distance / C_KM_PER_S,
            **attrs,
            "attributes_time_s": self.sample_time(time_s),
            "sun_separation_deg": angle,
            "geometry_end_s": float(window["end_s"]) if window else None,
            "reason": reason,
        }

    def snapshot(self, time_s, *, include_inactive=True):
        if not self.start_s <= time_s <= self.end_s:
            raise NetworkUnavailable("network_time_outside_trace")
        links = []
        for edge in self.edges:
            history = self.link_frames[edge["link_id"]]
            i = bisect.bisect_right(self.link_times[edge["link_id"]], time_s) - 1
            if i < 0:
                continue
            record = history[i]
            if not include_inactive and record["state"] == "DOWN":
                continue
            attributes = self._geometry_attributes(edge, time_s)
            links.append(
                {
                    **record,
                    **{k: v for k, v in attributes.items() if k not in {'reason'}},
                    'directions': self.directions(edge, time_s, geometry=attributes),
                    'bandwidth_semantics': 'base_budget_use_directions_for_effective_capacity',
                    **(
                        {"state": "DOWN", "reason": "horizon_end"}
                        if time_s == self.end_s
                        else {}
                    ),
                    **({"state": "DOWN", "reason": "injected_fault"}
                       if self.faulted(edge, time_s) else {}),
                }
            )
        i = bisect.bisect_right(self.frame_times, time_s) - 1
        valid_until = min(self.frames[min(i, len(self.frames) - 1)]['end_s'],
                          next((t for t in self.environment_times if t > time_s), self.end_s))
        valid_until = min(valid_until, *(t for f in self.spec.faults for t in (f['start_s'], f['end_s']) if t > time_s), self.end_s)
        return {
            "time_s": time_s,
            "valid_until_s": valid_until,
            "nodes": self._snapshot_nodes(time_s),
            "links": links,
            'shared_capacity_pools_bps': dict(self.pool_capacities),
        }

    def _snapshot_nodes(self, time_s):
        if not hasattr(self, '_snapshot_catalog'):
            self._snapshot_catalog = tuple(
                {**n.__dict__, 'radio_parameters': radio_parameters(n)}
                for n in sorted(self.nodes.values(), key=lambda n: n.node_id))
        return [dict(n if getattr(self, '_share_snapshot_metadata', False) else deepcopy(n),
                     position_itrs_km=self.position(n['node_id'], time_s),
                     state='DOWN' if any(f['target_type'] == 'node' and f['target_id'] == n['node_id']
                     and f['start_s'] <= time_s < f['end_s'] for f in (*self.spec.faults, *self.observed_faults.values())) else 'UP')
                for n in self._snapshot_catalog]

    def directions(self, edge, time_s, *, geometry=None):
        if geometry is not None and geometry.get('budget_evaluated') is False:
            return [{'source_node_id': a, 'target_node_id': b, 'bandwidth_bps': 0.0,
                         'budget_bandwidth_bps': 0.0, 'budget_evaluated': False,
                         'capacity_model': directional_profile(self.nodes, self.spec.link_profiles, edge, a).capacity_model,
                         'rate_scope': 'per_direction', 'shared_pools': list(hop_pools(self.nodes,a,b))}
                    for a,b in ((edge['endpoint_a'],edge['endpoint_b']), (edge['endpoint_b'],edge['endpoint_a']))]
        return [dict(source_node_id=a, target_node_id=b,
                     **self.directional_attributes(edge, a, b, time_s),
                     shared_pools=list(hop_pools(self.nodes, a, b)))
                for a, b in ((edge['endpoint_a'], edge['endpoint_b']),
                             (edge['endpoint_b'], edge['endpoint_a']))]

    def directional_attributes(self, edge, source, target, time_s):
        if self.physical_attributes is not None and edge['link_id'] not in self.added_edge_ids:
            row = self.physical_attributes.direction_at(edge['link_id'], self._attribute_time(edge, time_s), source, target)
            return {k: v for k, v in row.items() if k not in {'source_node_id', 'target_node_id', 'shared_pools'}}
        profile = directional_profile(self.nodes, self.spec.link_profiles, edge, source)
        # No physical budget exists outside a geometric contact window. These
        # compact zero-rate records retain blocked-state audit information.
        if not any(float(w['start_s']) <= time_s < float(w['end_s']) for w in
                   self.windows.get((edge['endpoint_a'], edge['endpoint_b']), ())):
            return {'bandwidth_bps': 0.0, 'budget_bandwidth_bps': 0.0,
                        'capacity_model': profile.capacity_model, 'rate_scope': 'per_direction',
                        'budget_evaluated': False}
        result = self._budget_attributes(profile, self.distance(edge, time_s), time_s)
        pools = hop_pools(self.nodes, source, target)
        result['bandwidth_bps'] = min([result['bandwidth_bps']] +
                                     [self.pool_capacities[k] for k in pools] +
                                     self.optical_limits(edge))
        return result

    def _budget_attributes(self, profile, distance, time_s):
        cached = getattr(self, '_last_budget', None)
        if cached is None or cached[0] is not profile or cached[1:3] != (distance, time_s):
            cached = profile, distance, time_s, profile.attributes(distance, time_s)
            self._last_budget = cached
        return dict(cached[3])

    def optical_limits(self, edge):
        return [self.nodes[n].optical_terminal_bandwidth_bps for n in (edge['endpoint_a'], edge['endpoint_b'])
                if edge['kind'] != 'sgl' and self.nodes[n].optical_terminal_bandwidth_bps is not None]

    def reservation_keys(self, leg):
        return ((leg['link_id'], leg['source_node_id'], leg['target_node_id']),
                *((k, '', '') for k in self.pools_for_hop(leg['source_node_id'], leg['target_node_id'])))

    def pools_for_hop(self, source, target):
        return hop_pools(self.nodes, source, target)

    def reserve(self, leg):
        for key in self.reservation_keys(leg):
            self.calendars.setdefault(key, []).append((leg['start_s'], leg['finish_s']))
            self.calendars[key].sort()

    def link_at(self, link_id, time_s):
        edge = self.edge_by_id[link_id]
        i = bisect.bisect_right(self.link_times[link_id], time_s) - 1
        if i < 0:
            raise NetworkUnavailable("link_time_uncovered")
        record = self.link_frames[link_id][i]
        result = {**record, **self._geometry_attributes(edge, time_s)}
        if result['state'] == 'DOWN' and result['reason'] is None:
            result['reason'] = record['reason']
        if self.faulted(edge, time_s):
            result.update(state="DOWN", reason="injected_fault")
        return result

    def _edge_transfer(
        self,
        link_id: str,
        source: str,
        target: str,
        time_s: float,
        payload_bytes: float,
        allow_wait: bool,
    ):
        frames = self.link_frames[link_id]
        # Outages are learned online. Future fault schedules are not route oracles.
        if self.faulted(self.edge_by_id[link_id]):
            return None
        times = self.link_times[link_id]
        pools = self.pools_for_hop(source, target)
        occupied = sorted(row for key in ((link_id, source, target), *((k, '', '') for k in pools))
                          for row in self.calendars.get(key, ()))
        cursor = time_s
        while cursor < self.end_s:
            index = max(0, bisect.bisect_right(times, cursor) - 1)
            frame = frames[index]
            edge = frame
            if edge["state"] != "UP":
                if not allow_wait:
                    return None
                cursor = frame["end_s"]
                continue
            busy = next(((a, b) for a, b in occupied if b > cursor), None)
            if busy and busy[0] <= cursor:
                cursor = busy[1]
                continue
            start, remaining, pieces = cursor, payload_bytes * 8.0, []
            while cursor < self.end_s:
                i = bisect.bisect_right(times, cursor) - 1
                current = frames[i]
                current_edge = current
                if current_edge["state"] != "UP" or current_edge.get(
                    "setup_start_s"
                ) != edge.get("setup_start_s"):
                    break
                next_sample = self.environment_times[
                    min(
                        bisect.bisect_right(self.environment_times, cursor),
                        len(self.environment_times) - 1,
                    )
                ]
                stop = min(
                    current["end_s"], next_sample, busy[0] if busy else self.end_s
                )
                bandwidth = self.directional_attributes(current_edge, source, target, cursor)['bandwidth_bps']
                if self.spec.transport_model != "queue":
                    # Route forecasts use the currently observed contender count.
                    # Actual completions come exclusively from the native engine.
                    bandwidth /= 1 + self.shared_load.get((link_id, source, target), 0)
                    bandwidth = min([bandwidth] + [self.pool_capacities[k] /
                                    (1 + self.shared_pool_load.get(k, 0)) for k in pools])
                duration = (remaining / bandwidth) if bandwidth > 0 else math.inf
                finish = stop if not math.isfinite(duration) else min(cursor + duration, stop)
                complete = bandwidth > 0 and duration <= stop - cursor
                sent = (
                    remaining
                    if complete
                    else min(remaining, max(0.0, finish - cursor) * bandwidth)
                )
                pieces.append(
                    {
                        "start_s": cursor,
                        "finish_s": finish,
                        "bandwidth_bps": bandwidth,
                        "bytes": sent / 8,
                        'distance_km': self.distance(edge, cursor),
                        'attributes_time_s': self.sample_time(cursor),
                    }
                )
                remaining -= sent
                cursor = finish
                if remaining <= 1e-7:
                    # Reuse the exact same finish-time geometry lookup for the
                    # propagation estimate and returned leg; never reuse a
                    # start-time value across a physical boundary.
                    finish_distance = self.distance(edge, cursor)
                    propagation = finish_distance / C_KM_PER_S
                    arrival = cursor + propagation
                    if arrival > self.end_s:
                        return None
                    return {
                        **edge,
                        "source_node_id": source,
                        "target_node_id": target,
                        "payload_bytes": payload_bytes,
                        'distance_km': finish_distance,
                        'start_distance_km': self.distance(edge, start),
                        'distance_time_s': cursor,
                        'attributes_time_s': self.sample_time(cursor),
                        "start_s": start,
                        "finish_s": cursor,
                        "arrival_s": arrival,
                        "propagation_delay_s": propagation,
                        "transmission_s": cursor - start,
                        "queue_wait_s": start - time_s,
                        'contact_ready_s': edge['ready_s'],
                        'acquisition_wait_s': max(0., edge['ready_s'] - max(time_s, edge['setup_start_s'])),
                        "segments": pieces,
                    }
                if busy and cursor >= busy[0]:
                    cursor = busy[1]
                    break
            else:
                return None
            if not allow_wait:
                return None
            if cursor <= start:
                return None
        return None

    def transfer(
        self,
        source: str,
        target: str,
        time_s: float,
        payload_bytes: float,
        policy: str = "earliest_arrival",
        *,
        commit: bool = False,
        allow_wait: bool = True,
        kinds: set[str] | None = None,
        avoid_nodes: set[str] | None = None,
        recompute: bool = False,
    ) -> tuple[float, tuple[dict, ...]]:
        """Reuse a selected path; recompute current leg timing on every call.

        Policies choose the initial path. Traffic alone does not change it.
        Physical changes invalidate only cached paths using changed links.
        ``recompute=True`` bypasses persistence for an explicit fresh search.
        A path that cannot carry this payload also falls back to a fresh search.
        """
        if policy not in {"min_hops", "min_propagation", "earliest_arrival"}:
            raise ValueError("unsupported routing policy")
        if not math.isfinite(payload_bytes) or payload_bytes < 0:
            raise ValueError("payload_bytes must be finite and nonnegative")
        if (
            source not in self.nodes
            or target not in self.nodes
            or not self.start_s <= time_s < self.end_s
        ):
            raise NetworkUnavailable("route_endpoint_or_time_unavailable")
        from astermind.engine.state.route_cache import PersistentRouteCache

        if not hasattr(self, '_route_cache'):
            self._route_cache = PersistentRouteCache()
        route_cache = self._route_cache
        route_key = route_cache.key(source, target, policy, allow_wait, kinds, avoid_nodes)
        selected = None if recompute else route_cache.get(self, route_key, time_s)
        if selected is not None:
            _, hops = selected
            arrival, replay = time_s, []
            for link_id, first, second in hops:
                leg = self._edge_transfer(
                    link_id, first, second, arrival, payload_bytes, allow_wait
                )
                if leg is None:
                    break
                replay.append(leg)
                arrival = leg['arrival_s']
            else:
                route_cache.hits += 1
                if commit:
                    for leg in replay:
                        self.reserve(leg)
                return arrival, tuple(replay)
            # A long transfer may no longer fit its contact window. Search for
            # a feasible alternative without caching failed timing predictions.
            route_cache.replay_failures += 1
        route_cache.searches += 1
        adjacency = self._routing_adjacency(kinds)
        queue = [(0.0, time_s, (source,), 0, ())]
        labels: dict[str, list[tuple[float, float]]] = {}
        serial = 0
        while queue:
            cost, arrival, nodes, _, path = heapq.heappop(queue)
            node = nodes[-1]
            if node == target:
                if not recompute:
                    route_cache.remember(self, route_key, path, time_s)
                if commit:
                    for leg in path:
                        self.reserve(leg)
                return arrival, path
            prior = labels.setdefault(node, [])
            if any(c <= cost and t <= arrival for c, t in prior):
                continue
            prior[:] = [(c, t) for c, t in prior if not (cost <= c and arrival <= t)]
            prior.append((cost, arrival))
            for neighbor, link_id in adjacency.get(node, ()):
                if (
                    neighbor in nodes
                    or neighbor in (avoid_nodes or ())
                    or not self.route_node_allowed(neighbor, source, target)
                ):
                    continue
                leg = self._edge_transfer(
                    link_id, node, neighbor, arrival, payload_bytes, allow_wait
                )
                if leg is None:
                    continue
                extra = (
                    1.0
                    if policy == "min_hops"
                    else leg["propagation_delay_s"]
                    if policy == "min_propagation"
                    else leg["arrival_s"] - arrival
                )
                serial += 1
                heapq.heappush(
                    queue,
                    (
                        cost + extra,
                        leg["arrival_s"],
                        (*nodes, neighbor),
                        serial,
                        (*path, leg),
                    ),
                )
        raise NetworkUnavailable("no_feasible_dynamic_route")

    def _routing_adjacency(self, kinds):
        """Cache only structural neighbors; every edge cost is evaluated live."""
        epoch = (id(self.edges), self.revision)
        if getattr(self, '_adjacency_epoch', None) != epoch:
            self._adjacency_epoch = epoch
            self._adjacency_cache = {}
        key = None if kinds is None else frozenset(kinds)
        if key not in self._adjacency_cache:
            adjacency = {}
            for edge in self.edges:
                if kinds is not None and edge['kind'] not in kinds:
                    continue
                a, b = edge['endpoint_a'], edge['endpoint_b']
                adjacency.setdefault(a, []).append((b, edge['link_id']))
                adjacency.setdefault(b, []).append((a, edge['link_id']))
            # Preserve the original deterministic search order and bound the
            # number of retained kind filters. Forks may share these read-only
            # neighbor tuples; costs, loads and calendars are never cached here.
            if len(self._adjacency_cache) >= 16:
                self._adjacency_cache = {}
            self._adjacency_cache[key] = {
                node: tuple(sorted(rows)) for node, rows in adjacency.items()}
        return self._adjacency_cache[key]

    def route_cache_info(self):
        """Report path reuse separately from per-hop transport work."""
        from astermind.engine.state.route_cache import PersistentRouteCache

        cache = getattr(self, '_route_cache', None)
        return (cache if cache is not None else PersistentRouteCache()).info()

    def route_node_allowed(self, node, source, target):
        return self.nodes[node].role != "ground" or node == target


def ground_position_km(row: dict) -> tuple[float, float, float]:
    lat, lon = float(row["lat_deg"]), float(row["lon_deg"])
    altitude = float(row.get("altitude_km", 0.0))
    if (
        not all(math.isfinite(v) for v in (lat, lon, altitude))
        or not -90 <= lat <= 90
        or not -180 <= lon <= 180
    ):
        raise ValueError("invalid ground coordinates")
    lat, lon = math.radians(lat), math.radians(lon)
    a, f = 6378.137, 1 / 298.257223563
    eccentricity2 = f * (2 - f)
    radius = a / math.sqrt(1 - eccentricity2 * math.sin(lat) ** 2)
    return (
        (radius + altitude) * math.cos(lat) * math.cos(lon),
        (radius + altitude) * math.cos(lat) * math.sin(lon),
        (radius * (1 - eccentricity2) + altitude) * math.sin(lat),
    )
