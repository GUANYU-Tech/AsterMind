"""One collective expansion for the runtime communication path.

Reuse FormationEstimator's validated rules instead of maintaining a second
operator switch in Execution. Whole-operation intra predictions remain intact.
"""
from dataclasses import asdict, dataclass, replace
from .estimator import Estimate
from .formation import FormationEstimator
from .frame_surrogate import NoRouteError
from .task import CommunicationTask
from .policy import resolve_task_policy
from .registered_profiles import collective_startup, route_scope
from .topology_collective import layered_plan


@dataclass(frozen=True)
class PlannedCommunication:
    estimate: object
    waves: tuple
    model: object
    policy: object = None
    selection: object = None
    startup_ns: int = 0
    local_runtime: bool = False


class _Recorder(FormationEstimator):
    def __init__(self, routes, settings, hierarchy=None, policy=None):
        super().__init__(routes,settings,routes.ground_owners,share_isl=False,intra_hierarchy=hierarchy,policy=policy)
        self.waves=[]

    def _wave_ns(self, transfers):
        self.waves.append(tuple(transfers))
        return 0

    def _p2p(self, route, size):
        self.waves.append(((route,size),))
        return 0

    def _route(self,a,b,size):
        try:return self.network.route(a,b,size)
        except NoRouteError:
            if self._sat(a)==self._sat(b):raise
            return dict(path=[a,b],resources=(),unavailable=True)


def _from_layers(task, policy, selection, routes, network):
    members = set(task.participants)
    hierarchy = policy.intra_hierarchy
    groups = (task.participants,)
    if hierarchy:
        groups = tuple(tuple(gpu for gpu in group if gpu in members) for group in hierarchy.su_groups)
        groups = tuple(group for group in groups if group)
    waves, stages, schedules = layered_plan(task, groups, routes, hierarchy, selection['layer_choices'])
    selection['stages'] = stages
    if len(groups) == 1:
        _, layout = schedules[groups[0], 'single_su']
        if task.algorithm == 'ring':
            task = replace(task, ring_order=layout)
        # Tree stages use the actual spanning tree; the registered tree
        # formula assumes a pre-defined heap tree and cannot price it.
        if (task.algorithm in ('ring', 'fullmesh', 'fc2', 'mesh')
                and task.operation in ('all_gather', 'all_reduce')):
            from .model_selection import select_whole
            _, result = select_whole(task, policy, selection, routes, network)
            if result is not None:
                return PlannedCommunication(result, (), None, policy, selection)
    else:
        selection['algorithm'] = 'hierarchical'
        if any(stage['algorithm_source'] == 'cell_policy' for stage in stages):
            selection['algorithm_source'] = 'cell_policy'
    paths = tuple(route for wave in waves for route, _ in wave)
    estimate = Estimate(0, 'explicit_layered', len(waves), len(paths), 0, 0, paths)
    startup, by_scope = collective_startup(task, waves, routes, policy, selection)
    selection.update(collective_startup_ns=startup, startup_us_by_scope=by_scope)
    return PlannedCommunication(estimate, waves, None, policy, selection, startup, local_runtime=True)


def plan_communication(network, policy, operator, time_s, routes=None):
    routes=routes or network.routes(network.frame_at(time_s))
    task=CommunicationTask(**asdict(operator))
    if (len(task.participants) < 2 or len(set(task.participants)) != len(task.participants)
            or type(task.bytes) is not int or task.bytes < 0):
        raise ValueError('expected distinct participants and nonnegative integer bytes')
    task,effective,selection=resolve_task_policy(policy,task,routes)
    if 'layer_choices' in selection and effective.intra_mode == 'surrogate':
        return _from_layers(task, effective, selection, routes, network)
    model=_Recorder(routes,effective.inter_parameters.model_dump(),effective.intra_hierarchy,effective)
    from .model_selection import select_whole
    cross_satellite = selection['scope'] == 'inter_satellite'
    result = None
    if not cross_satellite:
        task,result=select_whole(task,effective,selection,routes,network)
    if result is None:
        result=model.estimate(task)
    waves=tuple(model.waves)
    if result.mode=="explicit_ring":
        if task.operation == 'all_reduce':
            # Keep all RS waves before all AG waves; alternating them changes
            # the dependency structure. _Recorder collected one of each.
            assert len(waves)==2
            steps=len(task.participants)-1
            waves=(waves[0],)*steps+(waves[1],)*steps
        else:
            waves=waves*result.rounds
    if cross_satellite or any(
        route_scope(routes, route, effective.intra_hierarchy) == 'inter_satellite'
        for wave in waves for route, _ in wave
    ):
        return PlannedCommunication(result, waves, model, effective, selection)
    startup_ns, startup_by_scope = collective_startup(task, waves, routes, effective, selection)
    if waves:
        selection.update(collective_startup_ns=startup_ns, startup_us_by_scope=startup_by_scope)
    return PlannedCommunication(result, waves, model, effective, selection, startup_ns, local_runtime=True)
