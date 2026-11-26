"""Single-transfer timing for Fast Mode, without a flow allocator."""

from time import perf_counter
from types import SimpleNamespace

from astermind.engine.clock import Time
from astermind.engine.timeline import nanoseconds
from .event_runtime import CommunicationFailure, RuntimeCommunication
from .frame_surrogate import NoRouteError
from .intra_hierarchy import classify_participants
from .intra_topology import p2p_ns
from .path_budget import path_budget


class StaticCommunication(RuntimeCommunication):
    static_timing = True

    def __init__(self, network, loop, policy):
        # Only the provenance field is available. An accidental allocator call fails loudly.
        super().__init__(network, loop, policy, allocator=SimpleNamespace(provenance='static_quote'))
        self.waiting = {}
        self.request_policy = policy
        self.request_routes = None
        self.stats['static_transfers'] = 0

    def submit(self, operator, on_complete, on_failure=None, *, task_id=None, deadline_s=None):
        from astermind.engine.compute.fast_mode.quote import BatchCommunication
        from .operator import OperatorEstimate

        self.serial += 1
        identity = task_id or f'communication-{self.serial}'
        self._register(identity, on_failure, deadline_s)

        def attempt():
            try:
                duration = BatchCommunication(self)(operator)
            except NoRouteError:
                self.waiting[identity] = attempt
                return
            except Exception as error:
                self._fail(identity, error)
                return
            self.waiting.pop(identity, None)

            def complete():
                if self.tasks.pop(identity, None) is not None:
                    on_complete(OperatorEstimate(nanoseconds(duration), 0, 0, 0, 0))

            self.loop.schedule(self.loop.now_s + duration, 1, 'comm:static_operator_complete', complete)

        attempt()
        return identity

    def submit_transfer(self, identity, source, target, size, on_complete, on_failure=None, *,
                        deadline_s=None, first_hop=None, stage=None, correlation_id=None):
        if type(size) is not int or size < 0:
            raise ValueError('payload must be nonnegative integer bytes')
        if first_hop == source:
            raise ValueError('first hop must differ from source')
        self._register(identity, on_failure, deadline_s)

        def attempt():
            legs = []
            try:
                if not size or source == target:
                    end, path = self.loop.now_s, (source, target) if source != target else (source,)
                else:
                    end, path = self._quote_transfer(source, target, size, first_hop, stage, legs)
            except NoRouteError:
                self.waiting[identity] = attempt
                return
            except Exception as error:
                self._fail(identity, error)
                return

            self.waiting.pop(identity, None)
            self.stats['static_transfers'] += 1
            start = self.loop.now_s
            self.trace.append(dict(kind='static_transfer', id=identity, task=identity,
                                   path=path, payload_bytes=size, duration_ns=nanoseconds(end - start)))
            context = dict(cell_transfer_id=identity, correlation_id=correlation_id, stage=stage)

            def observe(event):
                if identity in self.tasks:
                    self.trace.append(event)

            for index, leg in enumerate(legs):
                fields = dict(**context, transfer_id=f'{identity}/hop/{index}', link_id=leg['link_id'],
                              source_node_id=leg['source_node_id'], target_node_id=leg['target_node_id'],
                              payload_bytes=size, time_semantics='static_per_hop')
                for kind, field in (('transmit_start', 'start_s'), ('transmit_complete', 'finish_s'),
                                    ('hop_arrival', 'arrival_s')):
                    at = max(start, Time(leg[field]))
                    self.loop.schedule(at, 0, 'comm:observe_hop',
                                       lambda row=dict(kind=kind, **fields): observe(row))

            def finish():
                if self.tasks.pop(identity, None) is None:
                    return
                if size and source != target:
                    self.delivery_paths[identity] = tuple(path)
                if stage:
                    self.trace.append(dict(kind='cell_transfer_terminal', **context,
                                           terminal_kind='completed'))
                on_complete()

            self.loop.schedule(end, 1, 'comm:static_arrival', finish)

        attempt()
        return identity

    def _quote_transfer(self, source, target, size, first_hop, stage, legs):
        started = perf_counter()
        try:
            if stage and self.request_routes is not None:
                arrival, path, selected = self.request_routes.quote(source, target, size, legs=legs)
                if first_hop is not None and path[1:2] != (first_hop,):
                    raise ValueError("first_hop would change the retained orbital route")
                self.stats['max_path_links'] = max(self.stats['max_path_links'], len(selected.hops))
                return arrival, path
            routes = self._refresh()
            route = routes.route(source, target, size, first_hop=first_hop)
            self.stats['max_path_links'] = max(self.stats['max_path_links'], len(route['resources']))
        finally:
            self.stats['route_wall_s'] += perf_counter() - started

        if all(edge['source_owner'] == edge['target_owner'] for edge in routes.route_resources(route)):
            if all(routes.topology.devices[node].kind == 'gpu' for node in (source, target)):
                _, hierarchy, _, _ = classify_participants(routes.frame, (source, target))
                duration = p2p_ns(routes, route, size, hierarchy)
            else:
                duration = 8 * size / route['bandwidth_gbps'] + route['latency_ns']
            return self.loop.now_s + Time.from_ns(duration), route['path']

        policy = self.request_policy if stage else self.policy
        budget = path_budget(routes, route, size, policy)
        rate = min(self.capacities[key] / demand for key, demand in budget['demand'].items() if demand > 0)
        duration = (Time.from_ns(budget['startup']) + Time.from_ns(size / rate) + Time.from_ns(budget['tail']))
        return self.loop.now_s + duration, route['path']

    def network_changed(self):
        self.stats['network_notifications'] += 1
        state = self.loop.now_s.ticks, self._network_version()
        if state == self._observed_network_state:
            return
        self._observed_network_state = state
        for attempt in tuple(self.waiting.values()):
            attempt()

    def _fail(self, identity, reason):
        self.waiting.pop(identity, None)
        task = self.tasks.pop(identity, None)
        if task is None:
            return
        failure = CommunicationFailure(identity, str(reason), nanoseconds(self.loop.now_s))
        self.trace.append(failure.model_dump())
        if task['failure']:
            task['failure'](failure)
        else:
            raise RuntimeError(f'communication {identity} failed: {reason}')

    def cancel(self, identity, reason='cancelled'):
        self._fail(identity, reason)

    def close(self):
        self.waiting.clear()
        super().close()
