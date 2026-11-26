"""Reuse a complete isolated quote while its exact native routes still match."""

from collections import OrderedDict
from weakref import ref

from astermind.engine.comm.target_network import TargetRoutes


_MISSING = object()


class WitnessRoutes(TargetRoutes):
    """Record the route-cache objects that Comm actually consulted."""

    def __init__(self, routes, cache):
        # Share the already built physical frame. Only route lookups are observed.
        self.__dict__.update(vars(routes))
        self.cache = cache
        self.witnesses = OrderedDict()

    def _select_route(self, source, target):
        first, last = self.network.resolve_pair(source, target)
        key = first, last, self.routing_policy, None, frozenset()
        try:
            return super()._select_route(source, target)
        finally:
            self.witnesses[key] = self.cache.entries.get(key, _MISSING)
            self.witnesses.move_to_end(key)


class FrameQuoteMemo:
    """Keep a bounded table per rank program and live Comm frame."""

    def __init__(self):
        self.routes = None
        self.contract = None
        self.values = OrderedDict()
        self.hits = 0

    def price(self, quote, key, build):
        routes = quote.routes
        runtime = quote.runtime
        if type(routes) is not TargetRoutes or routes.routing_policy not in ("min_hops", "min_propagation"):
            return build(quote)
        cache = getattr(routes.network.host.net, "_path_route_cache", None)
        if cache is None or cache.deadline is None or not cache.start <= quote.time < cache.deadline:
            return build(quote)
        contract = (id(cache), id(cache.edges), runtime.policy.model_dump_json(),
                    id(runtime.policy.intra_hierarchy), id(quote.capacities))
        if self.routes is None or self.routes() is not routes or self.contract != contract:
            self.routes = ref(routes)
            self.contract = contract
            self.values.clear()
        saved = self.values.get(key)
        if saved is not None:
            duration, witnesses = saved
            if all(cache.entries.get(pair, _MISSING) is selected for pair, selected in witnesses):
                # Keep the native LRU order equivalent to making those lookups.
                for pair, _ in witnesses:
                    cache.entries.move_to_end(pair)
                self.values.move_to_end(key)
                self.hits += 1
                return duration
            del self.values[key]

        recorded = WitnessRoutes(routes, cache)
        quote.routes = recorded
        duration = build(quote)
        witnesses = tuple(recorded.witnesses.items())
        if len(witnesses) <= 256 and all(
            selected is not _MISSING and cache.entries.get(pair, _MISSING) is selected
            for pair, selected in witnesses
        ):
            self.values[key] = duration, witnesses
            if len(self.values) > 128:
                self.values.popitem(last=False)
        return duration
