"""One collective expansion for the runtime communication path.

Reuse FormationEstimator's validated rules instead of maintaining a second
operator switch in Execution. Whole-operation intra predictions remain intact.
"""
from dataclasses import asdict, dataclass
from .formation import FormationEstimator
from .frame_surrogate import NoRouteError
from .task import CommunicationTask
from .policy import resolve_task_policy


@dataclass(frozen=True)
class PlannedCommunication:
    estimate: object
    waves: tuple
    model: object
    policy: object = None
    selection: object = None


class _Recorder(FormationEstimator):
    def __init__(self, routes, settings, hierarchy=None):
        super().__init__(routes,settings,routes.ground_owners,share_isl=False,intra_hierarchy=hierarchy)
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


def plan_communication(network, policy, operator, time_s, routes=None):
    routes=routes or network.routes(network.frame_at(time_s))
    task=CommunicationTask(**asdict(operator))
    task,effective,selection=resolve_task_policy(policy,task,routes)
    model=_Recorder(routes,effective.inter_parameters.model_dump(),effective.intra_hierarchy)
    from .model_selection import select_whole
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
    return PlannedCommunication(result,waves,model,effective,selection)
