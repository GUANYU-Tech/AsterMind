"""Bind eligible Fast communication paths before pricing the current batch."""

from collections import Counter
from itertools import permutations
from time import perf_counter

from astermind.engine.clock import Time
from astermind.engine.comm import registered_profiles as profiles
from astermind.engine.comm._vendor.inter_mean.mean import mean_frame_budget
from astermind.engine.comm.frame_surrogate import NoRouteError
from astermind.engine.comm.intra_hierarchy import classify_participants
from astermind.engine.comm.path_budget import _capacity_contract, path_budget
from astermind.engine.comm.planner import plan_communication
from .quote import BatchCommunication


class UnsupportedFormation(ValueError):
    pass


class LocalLane:
    """Bind one homogeneous SU lane, including its byte-dependent protocol."""

    def __init__(self, route, policy, card, port):
        self.bandwidth_gbps = route['bandwidth_gbps']
        self.latency_ns = route['latency_ns']
        self.bandwidth = route['bandwidth_gbps'] * 125
        self.latency = route['latency_ns'] / 1000
        self.policy = policy
        self.cards = (card, card)
        self.port = port
        self.select = profiles.registered(card, 'send_recv', 'send_recv', profiles.LINKS[card]).protocol_policy.select
        self.protocols = {}

    def _entry(self, size):
        protocol, _ = self.select(size)
        if protocol not in self.protocols:
            timing, = profiles.p2p_timing(size, 'single_su', self.cards, self.policy)
            bandwidth = min(self.bandwidth, self.port, timing['rate'] or self.bandwidth)
            parameters = dict(timing['parameters'], ring_bw_bytes_per_us=bandwidth,
                              packet_bw_bytes_per_us=bandwidth,
                              link_latency_us=max(self.latency, timing['phase_fixed_us']),
                              wire_efficiency=timing['wire_efficiency'])
            model = profiles.upstream_theory().TheoryDerivedSurrogate(**parameters)
            self.protocols[protocol] = model, timing['credits'], timing['startup_us']
        return self.protocols[protocol]

    def startup(self, size):
        return profiles.elapsed_ns(self._entry(size)[2]) * 10**9

    def ticks(self, size, include_startup):
        model, credits, startup = self._entry(size)
        duration = model.predict(size, 2, 'ring', credits,
                                 startup_us=startup if include_startup else 0., collective='sendrecv')
        return profiles.elapsed_ns(duration) * 10**9


class InterLane:
    """Freeze directed path constants while applying COMM_Backend framing to each payload."""

    def __init__(self, quote, route):
        routes, policy = quote.routes, quote.runtime.policy
        self.settings = policy.inter_parameters
        links = []
        for edge in routes.route_resources(route):
            if edge['source_owner'] == edge['target_owner']:
                continue
            if edge['shared_pools'] or {edge['source_owner'], edge['target_owner']} & routes.ground_owners:
                raise UnsupportedFormation('shared or ground path needs the general communication estimator')
            key = edge['resource_key']
            attrs = {**edge['attributes'], **policy.resource_profiles.get(key[0], {})}
            _, applied = _capacity_contract(attrs, policy.runtime_capacity_basis)
            links.append((quote.capacities[key], applied))
        if not links:
            raise UnsupportedFormation('missing directed ISL route')
        budget = path_budget(routes, route, 1, policy)
        self.fixed_ticks = Time.from_ns(budget['startup']).ticks + Time.from_ns(budget['tail']).ticks
        by_contract = {}
        for capacity, applied in links:
            by_contract[applied] = min(capacity, by_contract.get(applied, capacity))
        self.links = tuple(sorted((capacity, applied) for applied, capacity in by_contract.items()))
        self.signature = self.fixed_ticks, self.links

    def ticks(self, size, frames):
        settings = self.settings
        wire_bytes = frames.get(size)
        if wire_bytes is None:
            wire_bytes = mean_frame_budget(size, settings).ethernet_time_bytes
            frames[size] = wire_bytes
        rates = []
        for capacity, applied in self.links:
            demand = 1.
            if 'framing' not in applied:
                demand *= wire_bytes / size
            if 'mapping' not in applied:
                demand /= settings.mapping_efficiency
            if settings.fec.enabled and 'fec' not in applied:
                demand *= settings.fec.n / settings.fec.k
            rates.append(capacity / demand)
        return self.fixed_ticks + Time.from_ns(max(size / min(rates), 1e-6)).ticks


class BoundCommunicationPlan:
    """Reuse geometry and coefficients; one call prices only the current batch."""

    def __init__(self, pricing, *, direct_factory=None):
        self.pricing = pricing
        self.direct_factory = direct_factory
        self.runtime = pricing.runtime
        self.policy = self.runtime.policy.model_dump_json()
        self.templates = getattr(self.runtime, 'fast_lane_templates', {})
        self.stats = Counter()
        self.local = {}
        self.inter = {}
        self.pairs = {}
        self.programs = {}
        self.groups = {}
        self.local_classes = {}
        self.routes = None
        self.capacities = None
        self.prepare_ms = 0.

    def _template(self, key, create):
        # Scope immutable formulas to the simulation. Do not retain prior
        # batch durations or routes, and bound independent timing profiles.
        key = self.policy, key
        template = self.templates.get(key)
        if template is None:
            template = create()
            if len(self.templates) < 512:
                self.templates[key] = template
        return template

    def _initialize(self, quote):
        runtime = self.runtime
        from astermind.engine.comm.target_network import TargetRoutes
        from astermind.engine.state.device_network import DeviceNetworkState

        net = getattr(getattr(runtime.network, 'host', None), 'net', None)
        target_routes = (type(quote.routes) is TargetRoutes and type(net) is DeviceNetworkState
                         and quote.routes.routing_policy in ('min_hops', 'min_propagation'))
        fixed_network = (target_routes
                and net.physical is None and len(net.environment_times) == 2
                and not (net.calendars or net.disabled_device_links or net.failed_device_links or net.active_faults))
        frame_bound = ((target_routes or getattr(quote.routes, "frame_bound_paths", False)) and
                       quote.routes.frame.valid_until_s is not None and
                       quote.time < quote.routes.frame.valid_until_s)
        if not (fixed_network or frame_bound):
            raise UnsupportedFormation('timing reuse requires a fixed network or frame-bound paths')
        policy = runtime.policy
        expected = type(policy)(ep_strategy=policy.ep_strategy, default_hardware_id=policy.default_hardware_id,
                                cell_policies=policy.cell_policies, gpu_hardware=policy.gpu_hardware,
                                gpu_port_gbps=policy.gpu_port_gbps, link_bandwidth_factors=policy.link_bandwidth_factors)
        # ISL processing is a scenario coefficient in each bound directed lane.
        expected = expected.model_copy(update={'inter_parameters': expected.inter_parameters.model_copy(
            update={'hop_processing_us': policy.inter_parameters.hop_processing_us})})
        if policy != expected or policy.runtime_protocol != 'inter_reference':
            raise UnsupportedFormation('communication policy requires the general estimator')
        cell_policy = policy.cell_policies.get(self.pricing.cell_id)
        if cell_policy and cell_policy.algorithms.get('inter_satellite'):
            raise UnsupportedFormation('custom inter-satellite algorithms require the general estimator')
        self.default_card = ((cell_policy.hardware_id if cell_policy else None)
                             or policy.default_hardware_id)
        self.static_direct_network = fixed_network or getattr(quote.routes, 'frame_bound_paths', False)
        self.routes = quote.routes
        # Runtime replaces the entire mapping when it replaces the Frame.
        self.capacities = quote.capacities
        self.revision = quote.routes.frame.revision
        self.deadline = quote.routes.frame.valid_until_s
        self.hierarchical_owners = frozenset(
            node.satellite_id for node in quote.routes.frame.node_states.values()
            if node.attributes.get('intra_hierarchy'))
        self.cell_owners = {quote.routes.topology.devices[rank].satellite for rank in self.pricing.devices}
        self.local.clear()
        self.inter.clear()
        self.pairs.clear()
        self.programs.clear()
        self.groups.clear()
        self.local_classes.clear()

    def lanes(self, quote, source, target):
        pair = source, target
        if pair in self.pairs:
            return self.pairs[pair]
        route = quote.routes.route(source, target, 1)
        devices = quote.routes.topology.devices
        path = route['path']
        if len({devices[node].satellite for node in path}) == 1:
            policy = quote.runtime.policy
            cards = {policy.gpu_hardware.get(rank, self.default_card) for rank in (source, target)}
            if len(cards) != 1:
                raise UnsupportedFormation('heterogeneous GPU links require the general estimator')
            card, = cards
            edges = tuple(quote.routes.route_resources(route))
            if (len(path) not in (2, 3)
                    or (len(path) == 3 and devices[path[1]].kind != 'switch')
                    or any(devices[node].kind != 'gpu' for node in (source, target))
                    or card not in profiles.LINKS
                    or any(edge['shared_pools'] or
                           edge['attributes'].get('comm_technology') not in (None, profiles.LINKS[card])
                           for edge in edges)
                    or (devices[source].satellite in self.hierarchical_owners
                        and classify_participants(quote.routes.frame, (source, target))[0] != 'single_su')):
                raise UnsupportedFormation('requires one homogeneous scale-up fabric without shared pools')
            port = min(profiles.gpu_port_bandwidth(rank, card, policy) for rank in (source, target))
            key = card, route['bandwidth_gbps'], route['latency_ns'], port
            if key not in self.local:
                self.local[key] = self._template(('local', *key),
                    lambda: LocalLane(route, policy, card, port))
            price = self.local[key], False
        else:
            if any(devices[node].satellite not in self.cell_owners for node in path):
                raise UnsupportedFormation('path leaves the deployed Cell')
            lane = InterLane(quote, route)
            lane = self._template(('inter', lane.signature), lambda: lane)
            price = self.inter.setdefault(lane.signature, lane), True
        self.pairs[pair] = price
        return price

    def _local_group(self, quote, ranks):
        if ranks not in self.groups:
            lanes = {self.lanes(quote, a, b) for a, b in permutations(ranks, 2)}
            if len(lanes) != 1 or next(iter(lanes))[1]:
                raise UnsupportedFormation('TP group is not one homogeneous direct fabric')
            self.groups[ranks] = next(iter(lanes))[0]
        return self.groups[ranks]

    def _operation(self, quote, transfer):
        op = transfer.operation
        key = op, transfer.ranks
        if key in self.programs:
            return self.programs[key]
        ranks = tuple(self.pricing.devices[rank] for rank in transfer.ranks)
        size = len(ranks)
        if op == 'send_recv' and size == 2:
            lane, cross = self.lanes(quote, *ranks)
            program = lambda n, frames: lane.ticks(n, frames if cross else True)
        elif op in ('broadcast', 'all_reduce', 'all_gather', 'reduce_scatterv') and size in (2, 4, 8):
            lane = self._local_group(quote, ranks)
            descriptor = op, size, lane
            if descriptor not in self.local_classes:
                registered_ring = op in ('all_reduce', 'all_gather')
                # A collective can move from explicit waves to the registered
                # profile as its payload grows. Only bind the registered rule;
                # byte sizes it does not accept keep the ordinary quote.
                probe_bytes = 65536 if registered_ring else size
                prototype = self.pricing._operator(op, transfer.ranks, probe_bytes)
                plan = plan_communication(self.runtime.network, self.runtime.policy, prototype, quote.time, quote.routes)
                if (registered_ring and not plan.waves and
                        plan.estimate.mode == 'intra_ring' and plan.selection['scope'] == 'single_su'
                        and plan.selection['algorithm'] == 'ring'):
                    program = self._template(('whole_ring', *descriptor),
                        lambda: profiles.bind_registered_ring(prototype, plan.policy, lane.cards[0],
                                                              lane.bandwidth_gbps, lane.latency_ns, lane.port))
                    self.local_classes[descriptor] = self.programs[key] = program
                    return program
                if registered_ring:
                    raise UnsupportedFormation('collective uses payload-dependent planning')
                divisor = size if op in ('all_reduce', 'reduce_scatterv') else 1
                expected = {'broadcast': 1, 'all_reduce': 2 * (size - 1),
                            'all_gather': size - 1, 'reduce_scatterv': size - 1}[op]
                if (not plan.local_runtime or len(plan.waves) != expected
                        or plan.selection['scope'] != 'single_su'
                        or plan.selection['algorithm'] != ('direct' if op == 'broadcast' else 'ring')
                        or any(not wave or any(amount != size // divisor for _, amount in wave) for wave in plan.waves)):
                    raise UnsupportedFormation('unrecognized local collective waves')

                def uniform(n, frames):
                    if n % divisor:
                        return None
                    return lane.startup(n) + expected * lane.ticks(n // divisor, False)

                self.local_classes[descriptor] = uniform
            program = self.local_classes[descriptor]
        else:
            raise UnsupportedFormation(f'{op} at width {size} needs the general communication estimator')
        self.programs[key] = program
        return program

    def prepare(self):
        stamp = perf_counter()
        quote = BatchCommunication(self.runtime)
        self._initialize(quote)

        def bind(transfer):
            try:
                return transfer, self._operation(quote, transfer)
            except (UnsupportedFormation, NoRouteError):
                return transfer, None

        self.inputs = tuple(map(bind, self.pricing.program.input_transfers))
        self.stages = tuple(tuple((bind(t), count) for t, count in stage) for stage in self.pricing.stages)
        self.boundaries = tuple(tuple(tuple(map(bind, phase)) for phase in boundary)
                                for boundary in self.pricing.program.boundaries)
        self.direct = None
        policy = self.runtime.policy
        if (self.direct_factory is not None and self.static_direct_network and self.default_card == 'nvidia_h200'
                and not (policy.cell_policies or policy.gpu_hardware or policy.gpu_port_gbps
                         or self.cell_owners & self.hierarchical_owners)):
            try:
                self.direct = self.direct_factory(self, quote)
            except (UnsupportedFormation, NoRouteError):
                self.stats['direct_fallback'] += 1
        self.prepare_ms = (perf_counter() - stamp) * 1000

    def price(self, batch, observe=None):
        runtime = self.runtime
        routes = runtime._refresh()
        if (routes is not self.routes or routes.frame.revision != self.revision or runtime.active or runtime.tasks
                or runtime.policy.model_dump_json() != self.policy or runtime.capacities is not self.capacities):
            raise UnsupportedFormation('prepared communication paths require the admitted frame and private resources')
        reference = None
        prices, frames = {}, {}
        program = self.pricing.program

        def duration(bound):
            nonlocal reference
            transfer, evaluate = bound
            payload = transfer.payload(batch, program.tp)
            if not payload:
                return 0.
            key = evaluate or (transfer.operation, transfer.ranks), payload
            if key not in prices:
                ticks = evaluate(payload, frames) if evaluate is not None else None
                if ticks is None:
                    if reference is None:
                        reference = BatchCommunication(runtime)
                    ticks = reference(self.pricing._operator(transfer.operation, transfer.ranks, payload)).ticks
                    self.stats['ordinary_fallback'] += 1
                prices[key] = ticks
            if observe is not None:
                observe(transfer, payload, prices[key])
            return prices[key] / 10**18

        total = max(map(duration, self.inputs), default=0.)
        for stage, (ordinary, boundary, groups) in enumerate(zip(
                self.stages, self.boundaries, program.experts, strict=True)):
            per_replica = [0.] * program.dp
            for bound, count in ordinary:
                per_replica[bound[0].replica] += count * duration(bound)
            total += max(per_replica)
            if groups:
                if self.direct is not None:
                    total += self.direct.price(stage, batch, prices, frames)
                else:
                    if reference is None:
                        reference = BatchCommunication(runtime)
                    total += max(self.pricing._expert_seconds(reference, group, batch) for group in groups)
            total += sum(max(map(duration, phase), default=0.) for phase in boundary)
        return total, self.deadline
