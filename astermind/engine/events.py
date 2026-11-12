"""Shared transport events on the Native simulator clock."""

from __future__ import annotations

import heapq
from collections import defaultdict

from astermind.engine.clock import Time, ClockLog
from astermind.engine.state.network import NetworkUnavailable
from astermind.engine.setup.radio_capacity import hop_pools


class _EventEngine:
    def __init__(self, spec, environment):
        from astermind.engine.control import current_controller
        self.controller = current_controller()
        self.spec, self.env = spec, environment
        self.net = environment.network
        self.end = Time(environment.time['end_s'])
        if spec.time_horizon_s is not None:
            self.end = min(self.end, Time(spec.time_horizon_s))
        self.jobs, self.heap, self.log = {}, [], ClockLog()
        self.service_jobs = {}
        self.sequence = 0
        self.now = Time()
        self.transport = None
        self.global_scheduler = self.net.spec.transport_model == 'simgrid'
        self.shared_messages = {}
        self.shared_capacities = {}
        self.messages = {}
        self.waiting_retries = {}
        self.transport_provenance = {
            "backend": "queue",
            "sharing_scope": "exclusive_directed_hop",
        }
        transport_options = self.net.transport_options() if hasattr(self.net, 'transport_options') else {}
        if self.net.spec.transport_model == "simgrid":
            from astermind.engine.simgrid_transport import SimGridTransport

            self.transport = SimGridTransport(
                self.net.nodes, self.net.edges, self.net.spec.link_profiles,
                global_scheduler=self.global_scheduler, **transport_options
            )
            self.transport_provenance = self.transport.provenance
        elif self.net.spec.transport_model in {'fluid', 'ns3', 'flowsim'}:
            if self.net.spec.transport_model == 'fluid':
                from astermind.engine.fluid_transport import FluidTransport as Transport
            elif self.net.spec.transport_model == 'ns3':
                from astermind.engine.ns3_transport import Ns3Transport as Transport
            else:
                from astermind.engine.flowsim_transport import FlowSimTransport as Transport
            options = {'record_packets': getattr(self, 'record_packets', False)} if self.net.spec.transport_model == 'ns3' else {}
            self.transport = Transport(self.net.nodes, self.net.edges, self.net.spec.link_profiles, **options, **transport_options)
            self.transport_provenance = self.transport.provenance
        if self.global_scheduler:
            if self.net.spec.transport_model != 'simgrid':
                raise ValueError('SimGrid global scheduling requires native SimGrid transport')
            from astermind.engine.simgrid_events import SimGridEventQueue, native_time, simgrid_time
            self.heap = SimGridEventQueue(self.transport)
            self.end = simgrid_time(native_time(self.end))

    def push(self, time, priority, kind, request_id=None, payload=None):
        # Preserve native SimGrid doubles carried by Time; record projections
        # must never round a same-time callback into a different native instant.
        time = time if isinstance(time, Time) else Time(time)
        if time < self.now:
            raise RuntimeError("event scheduled in the past")
        self.sequence += 1
        if self.global_scheduler:
            return self.heap.append((time, priority, self.sequence, kind, request_id, payload))
        heapq.heappush(
            self.heap, (time, priority, self.sequence, kind, request_id, payload)
        )

    def message_owner(self, request_id):
        return self.jobs.get(request_id) or self.service_jobs[request_id]

    def hop_pools(self, source, target):
        if hasattr(self.net, 'pools_for_hop'):
            return self.net.pools_for_hop(source, target)
        return hop_pools(self.net.nodes, source, target)

    def fail(self, job, reason):
        if job.get('external_failure_callback') is not None:
            if job['done']:
                return
            job['done'] = True
            if self.transport:
                self.cancel_shared(job['request']['request_id'], reason)
            for collection in (self.messages, self.waiting_retries):
                for identity, message in tuple(collection.items()):
                    if message['request_id'] == job['request']['request_id']:
                        message['cancelled'] = True
                        if not self.transport and collection is self.messages:
                            leg = message['leg']
                            interval = leg['start_s'], leg['finish_s']
                            for key in self.net.reservation_keys(leg):
                                intervals = self.net.calendars.get(key, [])
                                if interval in intervals:
                                    intervals.remove(interval)
                                    if leg['start_s'] < self.now:
                                        intervals.append((leg['start_s'], min(leg['finish_s'], self.now)))
                        collection.pop(identity, None)
            job['external_failure_callback'](reason)
            return
        raise RuntimeError(f"external communication failed: {reason}")

    def _record_event(self, time, kind, request_id, payload, job):
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
            details.update({k: v for k, v in payload.get('context', {}).items() if k not in {'origin', 'callback'}})
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

    def run(self):
        # The orchestration boundary owns cleanup, including pre-run failures.
        self._run()
        if self.controller:
            self.controller.finished(self)

    def close(self):
        if self.transport:
            self.transport.close()

    def _run(self):
        while self.heap or (self.global_scheduler and self.transport.active):
            if self.controller and hasattr(self.controller, 'poll'):
                self.controller.poll(self)
            if self.global_scheduler:
                event = self.heap.pop_due()
                if event is None:
                    if self.now >= self.end:
                        break
                    self.advance_shared(None)
                    continue
                time, priority, sequence, kind, request_id, payload = event
            elif self.transport:
                target = min(self.heap[0][0], self.end)
                if target > self.transport.now:
                    self.advance_shared(target)
                    continue
            if not self.global_scheduler:
                time, priority, sequence, kind, request_id, payload = heapq.heappop(self.heap)
            if time > self.end:
                break
            self.now = time
            if kind in {'link_state', 'network_update'} and (payload or {}).get('revision', 0) != self.net.revision:
                continue
            job = self.jobs.get(request_id) or self.service_jobs.get(request_id)
            if job and job["done"]:
                continue
            if isinstance(payload, dict) and payload.get('cancelled') and kind != 'retry_transfer':
                continue
            self._record_event(time, kind, request_id, payload, job)
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
                            self.interrupt_transfer(self.message_owner(message['request_id']), message,
                                                    'network_contact_lost_during_transmission')
                continue
            try:
                getattr(self, "on_" + kind)(job, payload)
            except (NetworkUnavailable, ValueError) as error:
                if job is None or job['done']:
                    raise
                self.fail(job, str(error))
        if self.global_scheduler:
            while self.transport.now < self.end:
                self.advance_shared(None)
            self.now = self.transport.now
        else:
            self.now = self.end
        for job in tuple(self.service_jobs.values()):
            if not job['done'] and job.get('external_failure_callback') is not None:
                self.fail(job, 'simulation_horizon_exceeded')
        for job in self.jobs.values():
            if not job["done"]:
                self.fail(job, "simulation_horizon_exceeded")

    def on_control_boundary(self, unused, payload):
        self.controller.checkpoint(self)

    def on_external_callback(self, unused, callback):
        callback()

    def on_external_flow_done(self, unused, callback):
        callback['callback']()

    def wake_retries(self):
        """Retry against a newly observed graph, including mid-sample repairs."""
        pending, self.waiting_retries = self.waiting_retries, {}
        for message in pending.values():
            if not self.message_owner(message['request_id'])['done']:
                self.push(self.now, 25, 'retry_transfer', message['request_id'], message)

    def interrupt_unavailable(self, reason):
        for message in tuple(self.messages.values()):
            leg = message['leg']
            if self.net.link_at(leg['link_id'], self.now)['state'] == 'DOWN':
                self.interrupt_transfer(self.message_owner(message['request_id']), message, reason)

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
            self.spec.routing_policy,
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
        leg['shared_pools'] = list(self.hop_pools(source, leg['target_node_id']))
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
            for pool in self.hop_pools(key[1], key[2]):
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
                self.interrupt_transfer(self.message_owner(self.shared_messages[identity]['request_id']), self.shared_messages[identity], "network_contact_lost_during_transmission")
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
        from astermind.engine.setup.faults import fault_step_details
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
                self.interrupt_transfer(self.message_owner(message['request_id']), message, 'injected_' + kind + '_failure')
        for identity, message in tuple(self.waiting_retries.items()):
            if message['resume_node'] in failed_nodes:
                self.waiting_retries.pop(identity, None)
                self.fail(self.message_owner(message['request_id']), 'forwarding_buffer_lost_on_node_failure')
        self.wake_retries()

    def schedule_reallocation(self):
        # Replaced graph versions must not accumulate future events or wake
        # route waiters after they have become obsolete.
        def keep(e):
            return e[3] not in {'link_state', 'network_update'} or (e[5] or {}).get('revision', 0) == self.net.revision
        if self.global_scheduler:
            self.heap.retain(keep)
        else:
            self.heap[:] = [e for e in self.heap if keep(e)]
            heapq.heapify(self.heap)
        for event in self.net.events:
            if event['time_s'] >= self.now:
                self.push(event['time_s'], 0, 'link_state', None,
                          {**event, 'revision': self.net.revision})
        for frame in self.net.frames:
            if frame['time_s'] > self.now:
                self.push(frame['time_s'], 0, 'network_update', None, {'revision': self.net.revision})

    def advance_shared(self, target):
        response = self.transport.advance_next(self.end) if self.global_scheduler else self.transport.advance_until(target)
        self.now = response["time_s"]
        start = response["interval_start_s"]
        duration = self.now - start
        native_clock = {name: response[name] for name in ('interval_start_ticks', 'interval_end_ticks') if name in response}
        if duration > 0 and 'resources' in response:
            self.log.append(dict(time_s=self.now, kind='resource_service', interval_start_s=start,
                                 interval_end_s=self.now, **native_clock, resources=response['resources']))
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
                    "bandwidth_bps": response.get('allocated_rates_bps', {}).get(identity, sent * 8 / duration),
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
                            **native_clock,
                            **({'service_clock':response['service_clock'],
                                'native_interval_start_s':response['native_interval_start_s'],'native_interval_end_s':response['native_interval_end_s'],
                                'native_service_error_bytes':response['service_errors_bytes'][identity]}
                               if 'service_clock' in response else {}),
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


    def on_transfer_queued(self, job, message):
        # The next hop is now eligible; its reserved service may start later.
        pass

    def on_transmit_complete(self, job, message):
        # Link serialization ends here; propagation continues until hop_arrival.
        if self.transport:
            self.shared_messages.pop(message["leg"]["transfer_id"], None)

    def on_transmit_start(self, job, message):
        leg = message["leg"]
        if self.global_scheduler:
            leg['start_s'] = self.now
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
