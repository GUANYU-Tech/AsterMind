"""Chronological request execution; predictions never publish future mutable state."""

from __future__ import annotations

import heapq
import math
from dataclasses import replace
from collections import defaultdict, deque
from types import SimpleNamespace

from .contracts import RuntimeEngineResult
from .decision import build_candidate_selector
from .eligibility import build_compute_eligibility_set
from ..metrics.records import RuntimeRequestRecord
from ..setup.kv_capacity import derive_kv_cache_capacity_tokens
from ..setup.serving_time import prefill_suffix_time_s, decode_time_s
from ..state.energy import (
    advance_satellite_energy,
    apply_request_energy,
    energy_window_at,
    EnergyWindowError,
)
from ..state.kv_cache import local_kv_reuse_tokens, commit_session_kv, touch_kv_entry
from ..state.network import NetworkState, NetworkUnavailable
from ..setup.radio_capacity import hop_pools
from ..state.runtime_state import RuntimeState

ISL_KINDS = {
    "communication_isl",
    "cross_layer",
    "cell_isl",
    "inter_cell",
}


def run_event_engine(spec):
    # Kept local to avoid a module cycle with the prediction helpers.
    from .simulator import (
        normalize_runtime_request,
        _build_energy_trace,
        _all_uncompleted,
    )

    trace = spec.environment_state
    if trace is None:
        return _all_uncompleted(spec, "environment_state_missing")
    if hasattr(trace, "trace_manifest") and not trace.trace_manifest.evaluated:
        return _all_uncompleted(spec, "orbit_trace_not_evaluated")
    serving = spec.serving_profile
    if serving is None or not serving.has_compute_capacity:
        return _all_uncompleted(spec, "serving_capacity_unavailable")
    from ..setup.faults import realize_runtime_faults
    from .topology import candidate_links
    boundaries = sorted({float(row['time_s']) for row in trace.ephemeris_samples})
    runtime_network = realize_runtime_faults(spec.network, boundaries, candidate_links(spec.network))
    network = NetworkState(trace, runtime_network, spec.gateways, physical_state=spec.physical_state_ref)
    try:
        energy, initial_soc = _build_energy_trace(trace, spec.energy_profile)
        capacity = derive_kv_cache_capacity_tokens(spec.constellation, serving)
    except ValueError as error:
        return _all_uncompleted(spec, str(error))
    catalog = tuple(
        {
            **row,
            "compute_capable": row.get("compute_capable") is not False
            and network.nodes[str(row["satellite_id"])].role == "compute"
            and network.nodes[str(row["satellite_id"])].compute_target,
        }
        for row in trace.satellite_catalog
    )
    environment = SimpleNamespace(
        **{
            key: getattr(trace, key)
            for key in (
                "time",
                "ephemeris_samples",
                "sgl_windows",
                "isl_windows",
                "sunlight_windows",
                "trace_id",
                "trace_model",
            )
        },
        satellite_catalog=catalog,
        network=network,
    )
    eligible = build_compute_eligibility_set(
        satellite_catalog=catalog
    ).eligible_compute_satellite_ids
    if not eligible:
        return _all_uncompleted(spec, "eligible_compute_set_empty")
    state = RuntimeState()
    for node in catalog:
        state.compute_node_for(
            str(node["satellite_id"]),
            initial_soc=spec.energy_profile.get("node_profiles", {})
            .get(str(node["satellite_id"]), {})
            .get("initial_soc", initial_soc),
            max_concurrency=serving.max_concurrency,
        )
    for node in eligible:
        state.kv_cache_for(node, capacity_tokens=capacity)
    engine = _EventEngine(spec, environment, energy, state, eligible)
    requests = [normalize_runtime_request(row) for row in spec.requests]
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("duplicate request_id")
    by_id = {request["request_id"]: request for request in requests}

    def validate_dependency(rid, visiting, visited):
        if rid in visiting:
            raise ValueError("request dependency cycle")
        if rid in visited:
            return
        visiting.add(rid)
        request = by_id[rid]
        if (
            request.get("arrival_time_mode", "absolute")
            == "after_dependency_completion"
        ):
            predecessor = request.get("depends_on_request_id")
            if predecessor not in by_id:
                raise ValueError("unknown request predecessor")
            validate_dependency(predecessor, visiting, visited)
        visiting.remove(rid)
        visited.add(rid)

    visited = set()
    for rid in by_id:
        validate_dependency(rid, set(), visited)
    for request in sorted(
        requests, key=lambda r: (r["arrival_time_s"], r["request_id"])
    ):
        if not math.isfinite(request["arrival_time_s"]):
            raise ValueError("request arrival_time_s must be finite")
        engine.jobs[request["request_id"]] = {
            "request": request,
            "path": [],
            "ledger": [],
            "done": False,
            "first_done_s": None,
            "output_done_s": None,
            "copied_tokens": 0,
            "copied_from": None,
        }
        if (
            request.get("arrival_time_mode", "absolute")
            == "after_dependency_completion"
        ):
            gap = float(request.get("inter_turn_gap_s", 0.0))
            if not math.isfinite(gap) or gap < 0:
                raise ValueError("inter_turn_gap_s must be finite and nonnegative")
            engine.followers[request["depends_on_request_id"]].append(
                request["request_id"]
            )
        else:
            engine.push(
                max(0.0, request["arrival_time_s"]),
                30,
                "request_arrival",
                request["request_id"],
            )
    for frame in network.frames:
        engine.push(frame["time_s"], 0, "network_update", None)
    for event in network.events:
        engine.push(event["time_s"], 0, "link_state", None, event)
    boundaries = sorted({float(row["time_s"]) for row in trace.ephemeris_samples})
    for time in boundaries:
        engine.push(time, 0, "topology_sample", None)
        if engine.controller and time < engine.end:
            engine.push(time, 5, 'control_boundary', None)
    fault_batches = defaultdict(list)
    for index, fault in enumerate(network.spec.faults):
        if network.spec.fault_timing == 'boundary':
            fault_batches[fault['start_s']].append((index, fault, False))
            fault_batches[fault['end_s']].append((index, fault, True))
        else:
            engine.push(fault['start_s'], -2, "fault_start", None, (index, fault))
            engine.push(fault['end_s'], -2, "fault_end", None, (index, fault))
    for time_s, observations in sorted(fault_batches.items()):
        engine.push(time_s, -2, 'fault_batch', None, observations)
    engine.run()
    ordered = sorted(
        engine.jobs.values(),
        key=lambda j: (j["request"]["arrival_time_s"], j["request"]["request_id"]),
    )
    return RuntimeEngineResult(
        records=tuple(j["record"] for j in ordered),
        decision_traces=tuple(
            j.get(
                "decision",
                {"request_id": j["request"]["request_id"], "candidate_count": 0},
            )
            for j in ordered
        ),
        state_summary={
            **state.to_dict(),
            "event_log": engine.log,
            "network": {
                "events": network.events,
                "transport": engine.transport_provenance,
                "directed_reservations": [
                    {
                        "link_id": key[0],
                        "source": key[1],
                        "target": key[2],
                        "intervals": rows,
                    }
                    for key, rows in sorted(network.calendars.items())
                ],
            },
        },
        provenance={
            "runtime_engine": "chronological_discrete_event_runtime",
            "event_order": "time_priority_insertion_sequence",
            "network_admission_order": "per_hop_ready_event",
            "communication_queue_model": {
                'simgrid': 'simgrid_cm02_shared_directed_hops',
                'fluid': 'orbitserve_equal_share_fluid_directed_hops',
                'ns3': 'ns3_point_to_point_packet_round_robin',
            }.get(engine.net.spec.transport_model)
            if engine.transport
            else "directed_full_duplex_reservation_calendar",
            "network_transport": engine.transport_provenance,
            "network_rate_update_during_execution": bool(engine.transport),
            "network_forecast_model": "current_contender_count_snapshot"
            if engine.transport
            else "directed_reservation_calendar",
            "compute_queue_model": "ready_order_fifo_concurrency_slots",
            "serving_rate_model": "admission_snapshot_fixed_rate",
            "internal_collectives": "serving_embedded_with_physical_connectivity_gate",
            "fault_semantics": ('boundary_fault_buffered_sender_retry_same_boundary'
                                if network.spec.fault_timing == 'boundary' else
                                "continuous_fault_buffered_sender_retry_on_observed_change"),
            "realized_faults": list(network.spec.faults),
            "compute_energy_distribution": "equal_work_share_across_physical_cell_members",
            "rate_update_during_execution": False,
            "kv_publication": "compute_completion_event",
            "kv_cache_capacity_tokens": capacity,
            "kv_capacity_model": "architecture_storage_bytes_per_model_token",
        },
    )


class _EventEngine:
    def __init__(self, spec, environment, energy, state, eligible):
        from .control import current_controller
        self.controller = current_controller()
        self.spec, self.env, self.energy, self.state, self.eligible = (
            spec,
            environment,
            energy,
            state,
            eligible,
        )
        self.net, self.serving = environment.network, spec.serving_profile
        self.end = min(
            float(environment.time["end_s"]),
            float(spec.time_horizon_s) if spec.time_horizon_s is not None else math.inf,
        )
        self.jobs, self.heap, self.log = {}, [], []
        self.followers = defaultdict(list)
        self.active_compute = {}
        self.waiting = defaultdict(deque)
        self.sequence = 0
        self.now = 0.0
        self.transport = None
        self.shared_messages = {}
        self.shared_capacities = {}
        self.messages = {}
        self.waiting_retries = {}
        self.transport_provenance = {
            "backend": "queue",
            "sharing_scope": "exclusive_directed_hop",
        }
        if self.net.spec.transport_model == "simgrid":
            from .simgrid_transport import SimGridTransport

            self.transport = SimGridTransport(
                self.net.nodes, self.net.edges, self.net.spec.link_profiles
            )
            self.transport_provenance = self.transport.provenance
        elif self.net.spec.transport_model in {'fluid', 'ns3'}:
            if self.net.spec.transport_model == 'fluid':
                from .fluid_transport import FluidTransport as Transport
            else:
                from .ns3_transport import Ns3Transport as Transport
            self.transport = Transport(self.net.nodes, self.net.edges, self.net.spec.link_profiles)
            self.transport_provenance = self.transport.provenance

    def push(self, time, priority, kind, request_id=None, payload=None):
        if time < self.now - 1e-9:
            raise RuntimeError("event scheduled in the past")
        self.sequence += 1
        heapq.heappush(
            self.heap, (float(time), priority, self.sequence, kind, request_id, payload)
        )

    def fail(self, job, reason):
        if job["done"]:
            return
        from .simulator import _uncompleted

        if self.transport:
            self.cancel_shared(job["request"]["request_id"], reason)
        for identity, message in tuple(self.messages.items()):
            if message['request_id'] == job['request']['request_id']:
                message['cancelled'] = True
                self.messages.pop(identity, None)
        job["done"] = True
        self.active_compute.pop(job["request"]["request_id"], None)
        if "node" in job:
            queue = self.state.satellite_for(job["node"]).compute_queue
            queue.reservations[:] = [
                replace(row, finish_s=self.now)
                if row.request_id == job["request"]["request_id"]
                and row.start_s <= self.now < row.finish_s
                else row
                for row in queue.reservations
            ]
            self.push(self.now, 70, "admit", None, job["node"])
        job["record"] = _uncompleted(job["request"], reason).with_updates(
            energy_ledger=tuple(job["ledger"]), selected_path=tuple(job["path"])
        )
        self.log.append(
            {
                "time_s": self.now,
                "kind": "request_failed",
                "request_id": job["request"]["request_id"],
                "reason": reason,
            }
        )
        for rid in self.followers.get(job["request"]["request_id"], ()):
            self.fail(self.jobs[rid], "dependency_uncompleted")

    def run(self):
        try:
            self._run()
            if self.controller:
                self.controller.finished(self)
        finally:
            if self.transport:
                self.transport.close()

    def _run(self):
        while self.heap:
            if self.controller and hasattr(self.controller, 'poll'):
                self.controller.poll(self)
            if self.transport:
                target = min(self.heap[0][0], self.end)
                if target > self.transport.now:
                    self.advance_shared(target)
                    continue
            time, priority, sequence, kind, request_id, payload = heapq.heappop(
                self.heap
            )
            if time > self.end:
                break
            self.now = time
            if kind in {'link_state', 'network_update'} and (payload or {}).get('revision', 0) != self.net.revision:
                continue
            job = self.jobs.get(request_id)
            if job and job["done"]:
                continue
            if isinstance(payload, dict) and payload.get('cancelled') and kind != 'retry_transfer':
                continue
            details = {}
            if kind in {
                "transfer_queued",
                "transmit_start",
                "transmit_complete",
                "hop_arrival",
                "bandwidth_update",
            }:
                leg = payload["leg"]
                details = {
                    k: leg.get(k)
                    for k in (
                        "transfer_id",
                        "link_id",
                        "source_node_id",
                        "target_node_id",
                        "payload_bytes",
                        "stage",
                        "start_s",
                        "finish_s",
                        "arrival_s",
                        "ready_s",
                        "queue_wait_s",
                        'contact_ready_s',
                        'acquisition_wait_s',
                        "transmission_s",
                        "propagation_delay_s",
                        "transport_model",
                    )
                }
                if kind == "bandwidth_update":
                    details.update(payload["allocation"])
            if job and job.get("node"):
                details["compute_node_id"] = job["node"]
            if isinstance(payload, dict):
                details.update({k: payload[k] for k in ('phase', 'pipeline_stage', 'operation', 'members', 'bytes', 'attempt') if k in payload})
                details.update({k: v for k, v in payload.get('context', {}).items() if k != 'origin'})
            self.log.append(
                {
                    "time_s": time,
                    "kind": kind,
                    **details,
                    "request_id": request_id,
                    **(
                        {"link_id": payload["link_id"], "state": payload["state"], "reason": payload.get('reason')}
                        if kind == "link_state"
                        else {}
                    ),
                }
            )
            if kind in {"link_state", "topology_sample", "network_update"}:
                self.wake_retries()
                if self.transport:
                    self.refresh_shared()
                else:
                    for message in tuple(self.messages.values()):
                        leg = message['leg']
                        if not leg['start_s'] <= self.now < leg['finish_s']:
                            continue
                        current = self.net.link_at(leg['link_id'], self.now)
                        if current['state'] != 'UP' or current.get('setup_start_s') != leg.get('setup_start_s'):
                            self.interrupt_transfer(self.jobs[message['request_id']], message,
                                                    'network_contact_lost_during_transmission')
                if kind == "network_update":
                    for active in tuple(self.active_compute.values()):
                        if self.now < active[
                            "decode_done_s"
                        ] and not self.cell_available(active["node"]):
                            self.fail(
                                active, "compute_cell_disconnected_during_service"
                            )
                if kind == "link_state":
                    edge = self.net.link_frames[payload["link_id"]][0]
                    if edge["kind"] == "cell_isl" and payload["state"] == "DOWN":
                        for active in tuple(self.active_compute.values()):
                            if (
                                not active["done"]
                                and "prefill_start_s" in active
                                and active["prefill_start_s"]
                                <= time
                                < active["decode_done_s"]
                                and not self.cell_available(active["node"])
                            ):
                                self.fail(
                                    active, "compute_cell_disconnected_during_service"
                                )
                continue
            try:
                getattr(self, "on_" + kind)(job, payload)
            except (NetworkUnavailable, EnergyWindowError, ValueError) as error:
                if job is None:
                    raise
                self.fail(job, str(error))
        self.now = self.end
        for job in self.jobs.values():
            if not job["done"]:
                self.fail(job, "simulation_horizon_exceeded")

    def on_control_boundary(self, unused, payload):
        self.controller.checkpoint(self)

    def wake_retries(self):
        """Retry against a newly observed graph, including mid-sample repairs."""
        pending, self.waiting_retries = self.waiting_retries, {}
        for message in pending.values():
            if not self.jobs[message['request_id']]['done']:
                self.push(self.now, 25, 'retry_transfer', message['request_id'], message)

    def interrupt_unavailable(self, reason):
        for message in tuple(self.messages.values()):
            leg = message['leg']
            if self.net.link_at(leg['link_id'], self.now)['state'] == 'DOWN':
                self.interrupt_transfer(self.jobs[message['request_id']], message, reason)
        for job in tuple(self.active_compute.values()):
            if not self.cell_available(job['node']):
                self.fail(job, 'compute_cell_disconnected_during_service')

    def on_request_arrival(self, job, unused):
        from .simulator import _select_access, _preview_candidate, _decision

        request = job["request"]
        if not float(self.env.time["start_s"]) <= request["arrival_time_s"] < self.end:
            raise ValueError("request_time_outside_trace_horizon")
        access = _select_access(
            state=self.state,
            request=request,
            simulation_spec=self.spec,
            environment_state=self.env,
            energy_trace=self.energy,
            eligible_satellite_ids=self.eligible,
            sgl_bandwidth_bps=0.0,
            horizon_s=self.end,
        )
        candidates, forecasts = [], {}
        for node in self.eligible:
            try:
                forecast = _preview_candidate(
                    state=self.state,
                    request=request,
                    strategy=self.spec.strategy,
                    environment_state=self.env,
                    energy_trace=self.energy,
                    serving=self.serving,
                    access=access,
                    compute_satellite_id=node,
                    isl_bandwidth_bps=0.0,
                )
            except (NetworkUnavailable, EnergyWindowError, ValueError):
                continue
            candidate = forecast["candidate"]
            candidates.append(candidate)
            forecasts[candidate.candidate_id] = forecast
        selected = build_candidate_selector(self.spec.strategy).select_action(
            {"candidates": tuple(c.to_dict() for c in candidates)}, request
        )
        job["decision"] = _decision(request, selected, candidates)
        if not selected.evaluated:
            raise ValueError("no_feasible_compute_candidate")
        job.update(
            access=access,
            node=forecasts[selected.selected_candidate_id][
                "candidate"
            ].compute_satellite_id,
            selection=selected,
        )
        self.log_route_plan(job, forecasts[selected.selected_candidate_id]['inbound_path'], 'input', 'initial')
        self.send(
            job,
            request["ingress_id"],
            access.satellite_id,
            request["input_bytes"],
            {"sgl"},
            "input",
            "access_done",
        )

    def log_route_plan(self, job, path, stage, reason):
        self.log.append(dict(time_s=self.now, kind='route_plan', request_id=job['request']['request_id'],
                             stage=stage, reason=reason, semantics='forecast_not_transmitted',
                             path=[{k: leg[k] for k in ('link_id','source_node_id','target_node_id')} for leg in path]))

    def send(self, job, source, target, size, kinds, stage, completion, visited=(), context=None):
        if source == target:
            self.push(self.now, 40, completion, job["request"]["request_id"], context)
            return
        _, route = self.net.transfer(
            source,
            target,
            self.now,
            size,
            self.spec.strategy.routing_policy,
            kinds=kinds,
            avoid_nodes=set(visited),
        )
        if not route:
            raise NetworkUnavailable("empty_physical_route")
        if (context or {}).get('attempt', 0):
            self.log_route_plan(job, route, stage, 'recovery')
        leg = dict(route[0])
        # Commit only the hop now eligible for transmission, never later hops,
        # compute admission, downlinks or future KV state.
        if not self.transport:
            self.net.reserve(leg)
        else:
            leg.update(finish_s=None, arrival_s=None, transmission_s=None, segments=[])
        leg["transport_model"] = self.net.spec.transport_model
        leg['shared_pools'] = list(hop_pools(self.net.nodes, source, leg['target_node_id']))
        leg["stage"] = stage
        leg["ready_s"] = self.now
        leg["transfer_id"] = f"{job['request']['request_id']}:{self.sequence}"
        if stage == "kv_copy":
            leg["link_kind"], leg["kind"] = leg["kind"], "kv_copy"
        message = dict(
            request_id=job['request']['request_id'],
            leg=leg,
            target=target,
            size=size,
            kinds=kinds,
            stage=stage,
            completion=completion,
            visited=(*visited, source),
            context=context or {},
            origin=(context or {}).get('origin', source),
        )
        message['context'] = {**message['context'], 'origin': message['origin']}
        self.messages[leg['transfer_id']] = message
        rid = job["request"]["request_id"]
        self.push(self.now, 9, "transfer_queued", rid, message)
        self.push(leg["start_s"], 10, "transmit_start", rid, message)
        if not self.transport:
            self.push(leg["finish_s"], 15, "transmit_complete", rid, message)
            self.push(leg["arrival_s"], 20, "hop_arrival", rid, message)

    def update_shared_forecast(self):
        self.net.shared_load = {}
        self.net.shared_pool_load = {}
        for row in self.transport.active.values():
            leg = row["leg"]
            key = leg["link_id"], leg["source_node_id"], leg["target_node_id"]
            self.net.shared_load[key] = self.net.shared_load.get(key, 0) + 1
            for pool in hop_pools(self.net.nodes, key[1], key[2]):
                self.net.shared_pool_load[pool] = self.net.shared_pool_load.get(pool, 0) + 1

    def cancel_shared(self, request_id, reason):
        identities = [
            identity
            for identity, message in self.shared_messages.items()
            if message["request_id"] == request_id and identity in self.transport.active
        ]
        self.transport.cancel(identities)
        for identity in identities:
            message = self.shared_messages.pop(identity)
            leg = message["leg"]
            leg["cancelled_at_s"] = self.now
            self.log.append(
                {
                    "time_s": self.now,
                    "kind": "transfer_cancelled",
                    "request_id": request_id,
                    "transfer_id": identity,
                    "link_id": leg["link_id"],
                    "reason": reason,
                    "bytes_sent": sum(s["bytes"] for s in leg["segments"]),
                }
            )
        self.update_shared_forecast()

    def refresh_shared(self):
        capacities = {}
        for identity, row in tuple(self.transport.active.items()):
            if identity not in self.transport.active:
                continue
            leg = row["leg"]
            current = self.net.link_at(leg["link_id"], self.now)
            if (
                current["state"] != "UP"
                or current.get("setup_start_s") != leg["contact_setup_start_s"]
            ):
                self.interrupt_transfer(self.jobs[self.shared_messages[identity]['request_id']], self.shared_messages[identity], "network_contact_lost_during_transmission")
                continue
            value = self.net.directional_attributes(current, leg['source_node_id'], leg['target_node_id'], self.now)['bandwidth_bps']
            if self.shared_capacities.get(row["key"]) != value:
                capacities[row["key"]] = value
        self.transport.configure(capacities)
        self.shared_capacities.update(capacities)

    def next_sample(self):
        return next((t for t in self.net.environment_times if t > self.now + 1e-12), self.end)

    def interrupt_transfer(self, job, message, reason):
        if message.get('cancelled'):
            return
        leg = message['leg']
        identity = leg['transfer_id']
        old_interval = (leg['start_s'], leg['finish_s'])
        serialized = leg.get('finish_s') is not None and leg['finish_s'] <= self.now
        if self.transport:
            self.transport.cancel([identity] if identity in self.transport.active else [])
            self.shared_messages.pop(identity, None)
            self.update_shared_forecast()
        sent = sum(max(0, min(self.now, s['finish_s']) - s['start_s']) * s['bandwidth_bps'] / 8
                   for s in leg.get('segments', ())) if self.now >= leg['start_s'] else 0
        sent = min(leg['payload_bytes'], sent)
        actual_segments = []
        for segment in leg.get('segments', ()):
            finish = min(self.now, segment['finish_s'])
            if finish > segment['start_s']:
                actual_segments.append({**segment, 'finish_s': finish,
                    'bytes': (finish - segment['start_s']) * segment['bandwidth_bps'] / 8})
        leg['segments'] = actual_segments
        message['cancelled'] = True
        self.messages.pop(identity, None)
        leg.update(cancelled_at_s=self.now, cancellation_reason=reason,
                   bytes_discarded=sent, arrival_s=None,
                   finish_s=old_interval[1] if serialized else max(leg['start_s'], self.now),
                   transmission_s=leg['transmission_s'] if serialized else max(0, self.now - leg['start_s']))
        if not self.transport:
            for key in self.net.reservation_keys(leg):
                intervals = self.net.calendars.get(key, [])
                if old_interval in intervals:
                    intervals.remove(old_interval)
                    if old_interval[0] < self.now:
                        intervals.append((old_interval[0], min(old_interval[1], self.now)))
                    intervals.sort()
        self.log.append(dict(time_s=self.now, kind='transfer_cancelled', request_id=message['request_id'],
                             transfer_id=identity, link_id=leg['link_id'], reason=reason,
                             bytes_sent=sent, bytes_discarded=sent, payload_bytes=leg['payload_bytes'],
                             source_node_id=leg['source_node_id'], target_node_id=leg['target_node_id'],
                             resume_node_id=leg['source_node_id'], retained_payload_bytes=message['size'],
                             loss_semantics='interrupted_hop_discard_completed_hops_retained'))
        attempt = message['context'].get('attempt', 0) + 1
        if self.net.spec.failure_policy == 'fail' or attempt > self.net.spec.max_transfer_retries:
            self.fail(job, reason)
            return
        resume_node = leg['source_node_id']
        if any(f['target_type']=='node' and f['target_id']==resume_node for f in self.net.active_faults.values()):
            self.fail(job, 'forwarding_buffer_lost_on_node_failure')
            return
        context = {**message['context'], 'attempt': attempt}
        self.push(self.now, 25, 'retry_transfer', message['request_id'],
                  {**message, 'context': context, 'attempt': attempt, 'resume_node':resume_node})

    def on_retry_transfer(self, job, message):
        if message['completion'] == 'access_done':
            from .simulator import _select_access
            try:
                access = _select_access(state=self.state,
                    request={**job['request'], 'arrival_time_s': self.now},
                    simulation_spec=self.spec, environment_state=self.env,
                    energy_trace=self.energy, eligible_satellite_ids=self.eligible,
                    sgl_bandwidth_bps=0., horizon_s=self.end)
                message['target'] = access.satellite_id
                job['access'] = access
            except NetworkUnavailable:
                pass
        self.log.append(dict(time_s=self.now, kind='route_recovery', request_id=message['request_id'],
                             stage=message['stage'], attempt=message['attempt'],
                             source_node_id=message['resume_node'], target_node_id=message['target'],
                             original_source_node_id=message['origin'],
                             recovery_scope='current_buffered_node', retransmit_bytes=message['size']))
        try:
            self.send(job, message['resume_node'], message['target'], message['size'], message['kinds'],
                      message['stage'], message['completion'], context=message['context'])
        except NetworkUnavailable:
            if message['attempt'] >= self.net.spec.max_transfer_retries or self.now >= self.end:
                self.fail(job, 'retry_route_unavailable')
            else:
                message['attempt'] += 1
                message['context']['attempt'] = message['attempt']
                self.waiting_retries[message['leg']['transfer_id']] = message

    def on_fault_start(self, unused, payload):
        index, fault = payload
        self.on_fault_batch(unused, ((index, fault, False),))

    def on_fault_end(self, unused, payload):
        index, fault = payload
        self.on_fault_batch(unused, ((index, fault, True),))

    def on_fault_batch(self, unused, observations):
        from ..setup.faults import fault_step_details
        self.net.observe_faults(observations, self.now)
        self.schedule_reallocation()
        failed_nodes = set()
        for index, fault, restored in observations:
            self.log.append(dict(time_s=self.now, kind='fault_restored' if restored else 'fault_injected',
                                 **fault, **fault_step_details(fault, self.net.environment_times)))
            if not restored and fault['target_type'] == 'node':
                failed_nodes.add(fault['target_id'])
        for message in tuple(self.messages.values()):
            if self.net.faulted(message['leg']):
                kind = 'node' if failed_nodes.intersection((message['leg']['source_node_id'],message['leg']['target_node_id'])) else 'link'
                self.interrupt_transfer(self.jobs[message['request_id']], message, 'injected_' + kind + '_failure')
        for job in tuple(self.active_compute.values()):
            if failed_nodes.intersection(self.members(job['node'])):
                self.fail(job, 'compute_member_failed_no_checkpoint')
        for identity, message in tuple(self.waiting_retries.items()):
            if message['resume_node'] in failed_nodes:
                self.waiting_retries.pop(identity, None)
                self.fail(self.jobs[message['request_id']], 'forwarding_buffer_lost_on_node_failure')
        self.wake_retries()

    def schedule_reallocation(self):
        # Replaced graph versions must not accumulate future events or wake
        # route waiters after they have become obsolete.
        self.heap[:] = [e for e in self.heap if e[3] not in {'link_state', 'network_update'}
                        or (e[5] or {}).get('revision', 0) == self.net.revision]
        heapq.heapify(self.heap)
        for event in self.net.events:
            if event['time_s'] >= self.now:
                self.push(event['time_s'], 0, 'link_state', None,
                          {**event, 'revision': self.net.revision})
        for frame in self.net.frames:
            if frame['time_s'] > self.now:
                self.push(frame['time_s'], 0, 'network_update', None, {'revision': self.net.revision})

    def advance_shared(self, target):
        response = self.transport.advance_until(target)
        self.now = response["time_s"]
        start = response["interval_start_s"]
        duration = self.now - start
        contenders = defaultdict(int)
        for identity in response["bytes_sent"]:
            contenders[
                self.transport.resource(self.shared_messages[identity]["leg"])
            ] += 1
        for identity, sent in response["bytes_sent"].items():
            message = self.shared_messages[identity]
            leg = message["leg"]
            if duration > 0:
                segment = {
                    "start_s": start,
                    "finish_s": self.now,
                    "bytes": sent,
                    "bandwidth_bps": sent * 8 / duration,
                    "link_capacity_bps": self.shared_capacities[
                        self.transport.resource(leg)
                    ],
                    "concurrent_flows": contenders[self.transport.resource(leg)],
                    'distance_km': self.net.distance(leg, start),
                    'attributes_time_s': self.net.sample_time(start),
                }
                leg["segments"].append(segment)
                self.push(
                    self.now,
                    14,
                    "bandwidth_update",
                    message["request_id"],
                    {
                        **message,
                        "allocation": {
                            "interval_start_s": start,
                            "interval_end_s": self.now,
                            "allocated_bandwidth_bps": segment["bandwidth_bps"],
                            "bytes_sent": sent,
                            "link_capacity_bps": segment["link_capacity_bps"],
                        },
                    },
                )
        for identity in response["completed"]:
            message = self.shared_messages[identity]
            leg = message["leg"]
            propagation = self.net.distance(leg, self.now) / 299792.458
            leg.update(
                finish_s=self.now,
                arrival_s=self.now + propagation,
                transmission_s=self.now - leg["start_s"],
                propagation_delay_s=propagation,
                distance_km=self.net.distance(leg, self.now),
                distance_time_s=self.now,
                attributes_time_s=self.net.sample_time(self.now),
            )
            self.push(self.now, 15, "transmit_complete", message["request_id"], message)
            self.push(
                leg["arrival_s"], 20, "hop_arrival", message["request_id"], message
            )
        self.update_shared_forecast()

    def on_bandwidth_update(self, job, message):
        # Native byte-progress observation over the stated elapsed interval.
        pass

    def debit(self, job, node_id, *, prompt=0, output=0, size=0, stage):
        node = self.state.satellite_for(node_id)
        advance_satellite_energy(node, self.energy, self.now)
        window = energy_window_at(self.energy, node_id, self.now)
        update = apply_request_energy(
            node,
            prompt_tokens=prompt,
            output_tokens=output,
            link_bytes=size,
            energy_window=window,
            event_time_s=self.now,
            metadata={
                "request_id": job["request"]["request_id"],
                "energy_stage": stage,
            },
        )
        job["ledger"].append(update["ledger_entry"])
        if update["energy_violation"]:
            raise ValueError("energy_unavailable")
        return update

    def on_transfer_queued(self, job, message):
        # The next hop is now eligible; its reserved service may start later.
        pass

    def on_transmit_complete(self, job, message):
        # Link serialization ends here; propagation continues until hop_arrival.
        if self.transport:
            self.shared_messages.pop(message["leg"]["transfer_id"], None)

    def on_transmit_start(self, job, message):
        leg = message["leg"]
        current = self.net.link_at(leg['link_id'], self.now)
        if current['state'] != 'UP' or current.get('setup_start_s') != leg.get('setup_start_s'):
            self.interrupt_transfer(job, message, 'link_unavailable_at_transmission_start')
            return
        if self.transport:
            if current["state"] != "UP":
                raise NetworkUnavailable("link_unavailable_at_transmission_start")
            leg["contact_setup_start_s"] = current.get("setup_start_s")
            message["request_id"] = job["request"]["request_id"]
            if leg["payload_bytes"] > 0:
                key = self.transport.resource(leg)
                capacity = self.net.directional_attributes(current, leg['source_node_id'], leg['target_node_id'], self.now)['bandwidth_bps']
                self.transport.configure({key: capacity})
                self.shared_capacities[key] = capacity
                self.shared_messages[leg["transfer_id"]] = message
                self.transport.start(leg["transfer_id"], leg, leg["payload_bytes"])
                if capacity == 0:
                    self.transport.configure({key: 0})
                self.update_shared_forecast()
            else:
                propagation = current["propagation_delay_s"]
                leg.update(
                    finish_s=self.now,
                    arrival_s=self.now + propagation,
                    transmission_s=0.0,
                    propagation_delay_s=propagation,
                )
                self.push(
                    self.now,
                    15,
                    "transmit_complete",
                    job["request"]["request_id"],
                    message,
                )
                self.push(
                    leg["arrival_s"],
                    20,
                    "hop_arrival",
                    job["request"]["request_id"],
                    message,
                )
        for node in (leg["source_node_id"], leg["target_node_id"]):
            if self.net.nodes[node].role != "ground":
                share = (
                    leg["payload_bytes"]
                    if leg.get("link_kind", leg["kind"]) == "sgl"
                    else leg["payload_bytes"] / 2
                )
                self.debit(job, node, size=share, stage="communication")
        job["path"].append(leg)

    def on_hop_arrival(self, job, message):
        self.messages.pop(message['leg']['transfer_id'], None)
        node = message["leg"]["target_node_id"]
        try:
            self.send(job, node, message['target'], message['size'], message['kinds'],
                      message['stage'], message['completion'], visited=message['visited'],
                      context=message['context'])
        except NetworkUnavailable:
            if self.net.spec.failure_policy != 'retry_next_sample' or not self.net.spec.max_transfer_retries:
                raise
            self.log.append(dict(time_s=self.now, kind='route_wait', request_id=message['request_id'],
                                 source_node_id=node, target_node_id=message['target']))
            self.waiting_retries[message['leg']['transfer_id']] = {
                **message, 'resume_node': node, 'attempt': 1,
                'context': {**message['context'], 'attempt': 1}}

    def on_access_done(self, job, unused):
        self.send(
            job,
            job["access"].satellite_id,
            job["node"],
            job["request"]["input_bytes"],
            ISL_KINDS,
            "input",
            "input_done",
        )

    def on_input_done(self, job, unused):
        from .simulator import _preview_kv_copy

        request, node = job["request"], job["node"]
        required = min(request["prompt_tokens"], request["context_tokens_before_turn"])
        local = local_kv_reuse_tokens(
            self.state,
            satellite_id=node,
            session_id=request.get("session_id"),
            required_context_tokens=required,
        )
        plan = _preview_kv_copy(
            state=self.state,
            session_id=request.get("session_id"),
            target_satellite_id=node,
            required_context_tokens=required,
            prompt_length=request["prompt_tokens"],
            local_hit_tokens=local,
            copy_enabled=self.spec.strategy.kv_cache_copy_enabled,
            kv_bytes_per_token=float(self.serving.kv_cache_bytes_per_token or 0),
            environment_state=self.env,
            start_s=self.now,
            isl_bandwidth_bps=0.0,
            prefill_rate_table=self.serving.prefill_rate_table,
            concurrency=self.state.satellite_for(
                node
            ).compute_queue.comparison_concurrency_at(self.now),
            network=self.net.fork(),
            routing_policy=self.spec.strategy.routing_policy,
        )
        job.update(
            local_hit_tokens=local,
            effective_reuse_tokens=local,
            copied_from=plan["source_satellite_id"],
            copied_tokens=plan["copied_tokens"],
        )
        if plan["copied_tokens"]:
            self.send(
                job,
                plan["source_satellite_id"],
                node,
                plan["copied_tokens"] * float(self.serving.kv_cache_bytes_per_token),
                ISL_KINDS,
                "kv_copy",
                "copy_done",
            )
        else:
            self.on_copy_done(job, None)

    def on_copy_done(self, job, unused):
        job["effective_reuse_tokens"] += job["copied_tokens"]
        job["recompute_tokens"] = max(
            0, job["request"]["prompt_tokens"] - job["effective_reuse_tokens"]
        )
        job["compute_ready_s"] = self.now
        touch_kv_entry(
            self.state,
            satellite_id=job["copied_from"],
            session_id=job["request"].get("session_id"),
            now_s=self.now,
        )
        self.waiting[job["node"]].append(job["request"]["request_id"])
        self.push(self.now, 70, "admit", None, job["node"])

    def members(self, node_id):
        from .simulator import _compute_members

        return _compute_members(self.net, node_id)

    def cell_available(self, node_id):
        members = set(self.members(node_id))
        if any(f['target_type'] == 'node' and f['target_id'] in members for f in self.net.active_faults.values()):
            return False
        if len(members) < 2:
            return True
        adjacency = {node: [] for node in members}
        for edge in self.net.edges:
            if edge["kind"] != "cell_isl" or edge["endpoint_a"] not in members:
                continue
            row = self.net.link_at(edge["link_id"], self.now)
            if row["state"] == "UP" and row["bandwidth_bps"] > 0:
                adjacency[edge["endpoint_a"]].append(edge["endpoint_b"])
                adjacency[edge["endpoint_b"]].append(edge["endpoint_a"])
        reached = {node_id}
        pending = [node_id]
        while pending:
            for neighbor in adjacency[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        return reached == members

    def on_admit(self, unused, node_id):
        node = self.state.satellite_for(node_id)
        if self.waiting[node_id] and not self.cell_available(node_id):
            members = set(self.members(node_id))
            changes = [
                t
                for edge in self.net.edges
                if edge["kind"] == "cell_isl" and edge["endpoint_a"] in members
                for t in self.net.link_times[edge["link_id"]]
                if t > self.now
            ]
            if changes:
                self.push(min(changes), 70, "admit", None, node_id)
                self.log.append(
                    {
                        "time_s": self.now,
                        "kind": "compute_wait_for_cell",
                        "node_id": node_id,
                    }
                )
            else:
                while self.waiting[node_id]:
                    self.fail(
                        self.jobs[self.waiting[node_id].popleft()],
                        "compute_cell_disconnected",
                    )
            return
        while (
            self.waiting[node_id]
            and node.compute_queue.active_count_at(self.now)
            < self.serving.max_concurrency
        ):
            job = self.jobs[self.waiting[node_id].popleft()]
            if job["done"]:
                continue
            try:
                self.admit_job(job, node)
            except (ValueError, EnergyWindowError) as error:
                self.fail(job, str(error))

    def admit_job(self, job, node):
        node_id = job["node"]
        request = job["request"]
        window = energy_window_at(self.energy, node_id, self.now)
        dvfs = (
            self.spec.strategy.shadow_dvfs_scale
            if self.spec.strategy.shadow_dvfs_enabled and not window.get("sunlit", True)
            else 1.0
        )
        concurrency = node.compute_queue.active_count_at(self.now) + 1
        prefill = (
            prefill_suffix_time_s(
                self.serving.prefill_rate_table,
                prompt_length=request["prompt_tokens"],
                cached_prefix_tokens=job["effective_reuse_tokens"],
                concurrency=concurrency,
            )
            / dvfs
        )
        decode = (
            decode_time_s(
                self.serving.decode_rate_table,
                context_length=request["prompt_tokens"],
                output_tokens=request["output_tokens"],
                concurrency=concurrency,
            )
            / dvfs
        )
        reservation = node.compute_queue.preview(
            request_id=request["request_id"],
            ready_s=self.now,
            duration_for=lambda c, t: prefill + decode,
        )
        if reservation.finish_s > self.end:
            self.fail(job, "simulation_horizon_exceeded")
            return
        node.compute_queue.reserve(reservation)
        job.update(
            reservation=reservation,
            prefill_start_s=self.now,
            prefill_done_s=self.now + prefill,
            decode_done_s=reservation.finish_s,
            prefill_service_s=prefill,
            decode_service_s=decode,
            concurrency=concurrency,
            dvfs=dvfs,
            window=window,
        )
        self.push(self.now, 80, "compute_start", request["request_id"])

    def on_compute_start(self, job, unused):
        self.active_compute[job["request"]["request_id"]] = job
        request = job["request"]
        members = self.members(job["node"])
        job["physical_members"] = members
        updates = {
            member: self.debit(
                job,
                member,
                prompt=job["recompute_tokens"] / len(members),
                output=request["output_tokens"] / len(members),
                stage="compute",
            )
            for member in members
        }
        job["soc_before"] = updates[job["node"]]["soc_before"]
        first_duration = (
            decode_time_s(
                self.serving.decode_rate_table,
                context_length=request["prompt_tokens"],
                output_tokens=1,
                concurrency=job["concurrency"],
            )
            / job["dvfs"]
        )
        self.push(job["prefill_done_s"], 20, "prefill_complete", request["request_id"])
        self.push(
            job["prefill_done_s"] + first_duration,
            20,
            "first_token_ready",
            request["request_id"],
        )
        self.push(job["decode_done_s"], 20, "compute_complete", request["request_id"])

    def on_prefill_complete(self, job, unused):
        pass

    def on_first_token_ready(self, job, unused):
        from .simulator import _first_token_bytes

        self.send(
            job,
            job["node"],
            job["request"]["ingress_id"],
            _first_token_bytes(job["request"]),
            None,
            "first_token",
            "first_delivered",
        )

    def on_compute_complete(self, job, unused):
        self.active_compute.pop(job["request"]["request_id"], None)
        from .simulator import _first_token_bytes

        request = job["request"]
        job["kv_event"] = commit_session_kv(
            self.state,
            session_id=request.get("session_id"),
            satellite_id=job["node"],
            context_tokens=request["prompt_tokens"] + request["output_tokens"],
            now_s=self.now,
            copied_from=job["copied_from"],
        )
        self.log.append(
            {
                "time_s": self.now,
                "kind": "kv_publish",
                "request_id": request["request_id"],
                "node_id": job["node"],
                "context_tokens": request["prompt_tokens"] + request["output_tokens"],
            }
        )
        self.push(self.now, 70, "admit", None, job["node"])
        size = max(0, request["output_bytes"] - _first_token_bytes(request))
        if size:
            self.send(
                job,
                job["node"],
                request["ingress_id"],
                size,
                None,
                "output",
                "output_delivered",
            )
        else:
            job["output_done_s"] = self.now
            self.finish(job)

    def on_first_delivered(self, job, unused):
        job["first_done_s"] = self.now
        self.finish(job)

    def on_output_delivered(self, job, unused):
        job["output_done_s"] = self.now
        self.finish(job)

    def finish(self, job):
        if (
            job["first_done_s"] is None
            or job["output_done_s"] is None
            or "kv_event" not in job
        ):
            return
        request, kv, paths = job["request"], job["kv_event"], tuple(job["path"])
        node = self.state.satellite_for(job["node"])
        advance_satellite_energy(node, self.energy, self.now)
        job["record"] = RuntimeRequestRecord(
            **{
                k: request.get(k)
                for k in (
                    "request_id",
                    "arrival_time_s",
                    "prompt_tokens",
                    "output_tokens",
                    "session_id",
                    "region_id",
                )
            },
            completed=True,
            response_streaming_model="first_token_then_remaining_output",
            action_type="compute",
            access_type=job["access"].access_type,
            access_satellite_id=job["access"].satellite_id,
            access_compute_node_id=(
                job["access"].satellite_id
                if job["access"].access_type == "compute"
                else None
            ),
            serving_compute_node_id=job["node"],
            serving_satellite_id=job["node"],
            compute_owner_compute_node_id=job["node"],
            compute_owner_compute_node_ids=(job["node"],),
            compute_owner_satellite_id=job["node"],
            compute_owner_satellite_ids=tuple(
                job.get("physical_members", [job["node"]])
            ),
            dvfs_scale=job["dvfs"],
            prefill_queue_wait_s=job["prefill_start_s"] - job["compute_ready_s"],
            decode_queue_wait_s=0.0,
            prefill_service_s=job["prefill_service_s"],
            decode_service_s=job["decode_service_s"],
            ttft_s=job["first_done_s"] - request["arrival_time_s"],
            e2e_s=max(job["first_done_s"], job["output_done_s"])
            - request["arrival_time_s"],
            link_queue_wait_s=sum(leg["queue_wait_s"] for leg in paths),
            link_transmission_s=sum(leg["transmission_s"] for leg in paths),
            selected_path=paths,
            energy_ledger=tuple(job["ledger"]),
            energy_violation=False,
            soc_before=job["soc_before"],
            soc_after=node.soc,
            compute_sunlit=job["window"].get("sunlit", True),
            kv_capacity_admission=kv.get("admission"),
            kv_serve_action=kv.get("action"),
            kv_hit_tokens=job["effective_reuse_tokens"],
            kv_copied_tokens=job["copied_tokens"],
            kv_miss_tokens=job["recompute_tokens"],
            context_recompute_tokens=job["recompute_tokens"],
            kv_evicted_tokens=kv.get("evicted_tokens", 0),
            kv_evicted_bytes=kv.get("evicted_tokens", 0)
            * float(self.serving.kv_cache_bytes_per_token or 0),
            kv_evicted_entry_count=len(kv.get("evicted_session_ids", ())),
            kv_capacity_tokens=kv.get("capacity_tokens"),
            kv_occupancy_tokens_before=kv.get("occupancy_tokens_before"),
            kv_occupancy_tokens_after=kv.get("occupancy_tokens_after"),
            kv_copy_source_satellite_id=job["copied_from"],
            kv_copy_source_compute_node_id=job["copied_from"],
            bottleneck_reason="none",
            provenance={
                "scope": "physical_network_discrete_event_runtime",
                "network_transport": self.transport_provenance,
                "trace_model": self.env.trace_model,
                "serving_concurrency_at_admission": job["concurrency"],
                "serving_rate_model": "admission_snapshot_fixed_rate",
                "trace_id": self.env.trace_id,
                "score": job["selection"].score,
                "compute_ready_s": job["compute_ready_s"],
                "compute_start_s": job["prefill_start_s"],
                "prefill_complete_s": job["prefill_done_s"],
                "compute_complete_s": job["decode_done_s"],
            },
        )
        job["done"] = True
        for rid in self.followers.get(request["request_id"], ()):
            follower = self.jobs[rid]["request"]
            follower["source_arrival_time_s"] = follower["arrival_time_s"]
            follower["arrival_time_s"] = self.now + float(follower["inter_turn_gap_s"])
            self.push(follower["arrival_time_s"], 30, "request_arrival", rid)
        self.log.append(
            {
                "time_s": self.now,
                "kind": "request_complete",
                "request_id": request["request_id"],
            }
        )
