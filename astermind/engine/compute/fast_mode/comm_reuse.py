"""Parameterize COMM_Backend quotes on the fixed, uncontended H200 test formation."""

from collections import Counter
from itertools import permutations
from time import perf_counter

from astermind.engine.clock import Time
from astermind.engine.comm import registered_profiles as profiles
from astermind.engine.comm._vendor.inter_mean.mean import mean_frame_budget
from astermind.engine.comm.frame_surrogate import NoRouteError
from astermind.engine.comm.path_budget import _capacity_contract, path_budget
from astermind.engine.comm.planner import plan_communication
from .quote import BatchCommunication


class UnsupportedFormation(ValueError):
    pass


class LocalLane:
    """Bind the native H200 P2P rule, including its byte-dependent protocol."""

    def __init__(self, route, policy):
        self.bandwidth = route['bandwidth_gbps'] * 125
        self.latency = route['latency_ns'] / 1000
        self.policy = policy
        self.cards = ('nvidia_h200',) * 2
        self.port = profiles.nominal_gpu_bandwidth(self.cards[0])
        self.select = profiles.registered(self.cards[0], 'send_recv', 'send_recv', 'nvlink').protocol_policy.select
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

    def _initialize(self, quote):
        runtime = self.runtime
        from astermind.engine.comm.target_network import TargetRoutes
        from astermind.engine.state.device_network import DeviceNetworkState

        net = getattr(getattr(runtime.network, 'host', None), 'net', None)
        fixed_network = (type(quote.routes) is TargetRoutes and type(net) is DeviceNetworkState
                and quote.routes.routing_policy in ('min_hops', 'min_propagation')
                and net.physical is None and len(net.environment_times) == 2
                and not (net.calendars or net.disabled_device_links or net.failed_device_links or net.active_faults))
        frame_bound = (getattr(quote.routes, "frame_bound_paths", False) and
                       quote.routes.frame.valid_until_s is not None and
                       quote.time < quote.routes.frame.valid_until_s)
        if not (fixed_network or frame_bound):
            raise UnsupportedFormation('timing reuse requires a fixed network or frame-bound paths')
        policy = runtime.policy
        expected = type(policy)(ep_strategy=policy.ep_strategy, default_hardware_id='nvidia_h200')
        # ISL processing is a scenario coefficient in each bound directed lane.
        expected = expected.model_copy(update={'inter_parameters': expected.inter_parameters.model_copy(
            update={'hop_processing_us': policy.inter_parameters.hop_processing_us})})
        if policy != expected or policy.runtime_protocol != 'inter_reference':
            raise UnsupportedFormation('ineligible fixed H200 communication timing environment')
        self.routes = quote.routes
        # Runtime replaces the entire mapping when it replaces the Frame.
        self.capacities = quote.capacities
        self.revision = quote.routes.frame.revision
        self.deadline = quote.routes.frame.valid_until_s
        self.hierarchical_owners = frozenset(
            node.satellite_id for node in quote.routes.frame.node_states.values()
            if node.attributes.get('intra_hierarchy'))
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
            if (len(path) not in (2, 3)
                    or (len(path) == 3 and devices[path[1]].kind != 'switch')
                    or any(devices[node].kind != 'gpu' for node in (source, target))
                    or devices[source].satellite in self.hierarchical_owners
                    or any(edge['shared_pools'] for edge in quote.routes.route_resources(route))):
                raise UnsupportedFormation('requires one homogeneous H200 scale-up fabric without shared pools')
            key = route['bandwidth_gbps'], route['latency_ns']
            if key not in self.local:
                self.local[key] = LocalLane(route, quote.runtime.policy)
            price = self.local[key], False
        else:
            lane = InterLane(quote, route)
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
        elif op in ('broadcast', 'all_reduce', 'all_gather', 'reduce_scatterv') and size in (4, 8):
            lane = self._local_group(quote, ranks)
            descriptor = op, size, lane
            if descriptor not in self.local_classes:
                prototype = self.pricing._operator(op, transfer.ranks, size)
                plan = plan_communication(self.runtime.network, self.runtime.policy, prototype, quote.time, quote.routes)
                divisor = size if op in ('all_reduce', 'reduce_scatterv') else 1
                expected = {'broadcast': 1, 'all_reduce': 2 * (size - 1),
                            'all_gather': size - 1, 'reduce_scatterv': size - 1}[op]
                if (not plan.local_runtime or len(plan.waves) != expected
                        or plan.selection['scope'] != 'single_su'
                        or any(not wave or any(amount != size // divisor for _, amount in wave) for wave in plan.waves)):
                    raise UnsupportedFormation('unrecognized H200 native collective waves')

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
        if self.direct_factory is not None:
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
