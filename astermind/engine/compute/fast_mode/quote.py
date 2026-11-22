"""An isolated COMM_Backend quote from a single current Network Frame."""

from dataclasses import replace

from astermind.engine.clock import Time
from astermind.engine.comm.frame_surrogate import NoRouteError
from astermind.engine.comm.intra_hierarchy import classify_participants
from astermind.engine.comm.intra_topology import p2p_ns
from astermind.engine.comm.operator import CommunicationOperator
from astermind.engine.comm.path_budget import _capacity_contract, path_budget, resource_capacities
from astermind.engine.comm.planner import plan_communication
from astermind.engine.comm.policy import resolve_task_policy
from astermind.engine.comm.registered_profiles import LINKS, p2p_duration, route_scope
from astermind.engine.comm.task import CommunicationTask


def _mixed_local_ns(routes, route, size, policy, cell_id):
    """Quote a mixed operator's intra-SU leg using the physical GPU profile."""
    if policy.intra_mode != "surrogate":
        return None
    source, target = route["path"][0], route["path"][-1]
    scope = route_scope(routes, route, policy.intra_hierarchy)
    if scope != "single_su":
        return None
    probe = CommunicationTask("send_recv", (source, target), size, "payload", cell_id=cell_id)
    _, _, info = resolve_task_policy(policy, probe, routes)
    cards = info.get("hardware_by_gpu", {})
    card = cards.get(source)
    if card != cards.get(target) or card == "nvidia_h200" or card not in LINKS:
        return None
    fabric = LINKS[card]
    if any(edge["attributes"].get("comm_technology") not in (None, fabric)
           for edge in routes.route_resources(route)):
        raise ValueError(f"Fast Mode mixed local path does not match the {fabric} profile for {card}")
    duration, _ = p2p_duration(routes, route, size, scope, (card, card), policy, include_startup=True)
    return duration


def mixed_runtime_waves(runtime, operator):
    """Keep V1's mixed-task waves and submit their P2Ps on the same runtime."""
    if runtime.policy.task_timeout_s is not None:
        return None  # Keep V1's single deadline for the whole operator.
    routes = runtime._refresh()
    owners = {routes.frame.node_states[rank].satellite_id for rank in operator.participants}
    if not 1 < len(owners) < len(operator.participants):
        return None
    try:
        plan = plan_communication(runtime.network, runtime.policy, operator, runtime.loop.now_s, routes)
    except NoRouteError:
        return None  # The normal runtime owns the original missing-route behavior.
    if plan.local_runtime or not any(
            _mixed_local_ns(routes, route, size, plan.policy, operator.cell_id) is not None
            for wave in plan.waves for route, size in wave if size):
        return None
    return tuple(tuple(
        CommunicationOperator("send_recv", None, (route["path"][0], route["path"][-1]),
                              size, "payload", cell_id=operator.cell_id)
        for route, size in wave if size and route["path"][0] != route["path"][-1]
    ) for wave in plan.waves)


class BatchCommunication:
    """Use COMM_Backend protocol and wave rules; retain durations only for this batch."""

    def __init__(self, runtime, *, routes=None, time=None, collect_paths=False):
        self.runtime = runtime
        self.routes = runtime._refresh() if routes is None else routes
        self.time = runtime.loop.now_s if time is None else time
        self.capacities = (runtime.capacities if routes is None
                           else resource_capacities(routes, runtime.policy))
        self.signature = [] if collect_paths else None
        self.complete_paths = True
        self.values = {}

    def __call__(self, operator):
        key = replace(operator, task_id=None, traffic_matrix_bytes=())
        # The direct binder returns the same immutable matrix object for equal
        # layers. Keep the original matrix in the entry to pin its identity.
        key = key, id(operator.traffic_matrix_bytes)
        if key not in self.values:
            self.values[key] = operator.traffic_matrix_bytes, self._price(operator)
        return self.values[key][1]

    def _price(self, operator):
        runtime = self.runtime
        plan = plan_communication(runtime.network, runtime.policy, operator, self.time, self.routes)
        paths = [] if self.signature is not None else None
        if not plan.waves:
            total = Time.from_ns(plan.estimate.duration_ns)
            if paths is not None:
                for route in plan.estimate.paths:
                    paths.append(self._path_signature(route, operator.bytes))
                if total and not paths:
                    self.complete_paths = False
        else:
            total = Time.from_ns(plan.startup_ns)
            for wave in plan.waves:
                limit = Time()
                wave_paths = [] if paths is not None else None
                for route, size in wave:
                    source, target = route["path"][0], route["path"][-1]
                    if size and source != target:
                        route = self.routes.route(source, target, size)
                        limit = max(limit, self._p2p(plan, route, size, operator.operation))
                        if wave_paths is not None:
                            wave_paths.append(self._path_signature(route, size))
                if paths is not None:
                    paths.append(tuple(wave_paths))
                total += limit
        if paths is not None:
            self.signature.append((operator.operation, operator.participants, operator.bytes,
                plan.estimate.mode, plan.policy.model_dump_json(), plan.selection,
                plan.estimate.model_inputs, tuple(paths), total.ticks))
        return total

    def _path_signature(self, route, size):
        policy = self.runtime.policy
        hops = []
        for edge in self.routes.route_resources(route):
            resource = edge["resource_key"]
            attrs = {**edge["attributes"], **policy.resource_profiles.get(resource[0], {})}
            pools = tuple((pool, self.capacities.get(("pool", pool)),
                           edge.get("shared_pool_coefficients", {}).get(pool))
                          for pool in edge["shared_pools"])
            hops.append((resource, edge["bandwidth_gbps"], edge["latency_ns"],
                         self.capacities[resource], pools, _capacity_contract(attrs, policy.runtime_capacity_basis),
                         attrs.get("processing_ns"), attrs.get("comm_technology")))
        nodes = tuple((node, self.routes.frame.node_states[node].satellite_id,
                       self.routes.frame.node_states[node].attributes)
                      for node in route["path"])
        return (size, tuple(route["resources"]), nodes, tuple(hops),
                route["bandwidth_gbps"], route["latency_ns"])

    def _p2p(self, plan, route, size, operation):
        source, target = route["path"][0], route["path"][-1]
        devices = self.routes.topology.devices
        if len({devices[node].satellite for node in route["path"]}) == 1:
            if all(devices[node].kind == "gpu" for node in (source, target)):
                if plan.local_runtime:
                    scope = route_scope(self.routes, route, plan.policy.intra_hierarchy)
                    hardware = tuple(plan.selection["hardware_by_gpu"][node] for node in (source, target))
                    duration, _ = p2p_duration(self.routes, route, size, scope, hardware, plan.policy,
                                               include_startup=operation == "send_recv")
                else:
                    duration = _mixed_local_ns(
                        self.routes, route, size, plan.policy, plan.selection["cell_id"])
                    if duration is None:
                        _, hierarchy, _, _ = classify_participants(self.routes.frame, (source, target))
                        duration = p2p_ns(self.routes, route, size, hierarchy)
            else:
                duration = 8 * size / route["bandwidth_gbps"] + route["latency_ns"]
            return Time.from_ns(duration)
        budget = path_budget(self.routes, route, size, self.runtime.policy)
        rate = min(self.capacities[key] / weight for key, weight in budget["demand"].items() if weight > 0)
        return (Time.from_ns(budget["startup"]) + Time.from_ns(max(size / rate, 1e-6))
                + Time.from_ns(budget["tail"]))
