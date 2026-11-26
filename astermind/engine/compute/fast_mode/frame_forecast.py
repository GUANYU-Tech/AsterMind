"""Read a future offline Frame using V1's normal projection on a fork."""

from copy import copy
from types import SimpleNamespace

from astermind.engine.comm.target_network import TargetNetworkFrames


def forecast_routes(frames, at):
    if type(frames) is not TargetNetworkFrames:
        # Explicit external Frame providers may supply their own projection.
        forecast = getattr(frames, "forecast_routes", None)
        return forecast(at) if forecast is not None else None

    host = frames.host
    state = host.net
    physical = state.physical
    if (not host.now < at < frames.horizon
            or host.controller is not None and not getattr(host.controller, 'read_only_topology', False)
            or state.active_faults or physical is not None and physical.spec.faults
            or host.spec.routing_policy not in ("min_hops", "min_propagation")):
        return None

    # The Frame and route selector see the same projected clock and state.
    # The device state shares immutable topology. Fork the path cache used
    # by its read-only Frame projection and route lookups.
    scratch = copy(state)
    cache = getattr(state, "_path_route_cache", None)
    scratch._path_route_cache = cache.fork() if cache is not None else None
    network = frames.network
    view = SimpleNamespace(
        host=SimpleNamespace(now=at, net=scratch, spec=host.spec, end=host.end,
                             comm_policy=getattr(host, 'comm_policy', None)),
        topology=network.topology,
        resolve_pair=network.resolve_pair,
    )
    projection = TargetNetworkFrames(view)
    return projection.routes(projection.frame_at(at))
