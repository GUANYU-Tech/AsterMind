"""Synchronous estimates against the Network's active physical routes."""
from collections import defaultdict
from dataclasses import dataclass
from .task import CommunicationTask, SatelliteTaskAdapter
from functools import lru_cache
from itertools import permutations, product
from math import ceil, isfinite


@dataclass(frozen=True)
class Estimate:
    duration_ns: int
    mode: str
    rounds: int | None = None
    flow_count: int | None = None
    cross_satellite_flows: int | None = None
    cross_satellite_bytes: int | None = None
    paths: tuple = ()
    model_inputs: dict | None = None


@lru_cache(maxsize=256)
def profile(rate, latency, inter):
    from surrogate.theory.whitebox_surrogate_v2 import make_surrogate_from_wire
    # Synthetic ideal wire profile. Network supplies Gbit/s and nanoseconds.
    return make_surrogate_from_wire(rate * 125, 1, "ring", eta_ring=1,
        link_latency_us=latency / 1000, startup_us=0,
        **({"credit_bdp_ring_packets": float("inf")} if inter else {}))


class CommunicationEstimator:
    def __init__(self, network, *, expand_inter=False):
        self.network = network
        self.topology = network.topology
        self.expand_inter = expand_inter

    def _route(self, a, b, size):
        return self.network.route(a, b, size)

    def _sat(self, endpoint):
        return self.topology.devices[endpoint].satellite

    def _p2p(self, route, size):
        model = profile(route["bandwidth_gbps"], route["latency_ns"],
                        self._sat(route["path"][0]) != self._sat(route["path"][-1]))
        us = model.predict(size, 2, "ring", 128, collective="sendrecv")
        if not isfinite(us) or us < 0:
            raise ValueError("invalid P2P surrogate duration")
        return ceil(1000 * us) if size else 0

    def estimate_satellite(self, source, target, size):
        return self.estimate(SatelliteTaskAdapter.from_transfer(source, target, size))

    def _wave_ns(self, transfers):
        return max((self._p2p(route,size) for route,size in transfers),default=0)

    def _satellite_p2p(self, source, target, size):
        if type(size) is not int or size < 0 or source == target:
            raise ValueError("expected two distinct satellites and nonnegative bytes")
        def terminals(sat):
            return sorted(d.id for d in self.topology.devices.values()
                          if d.kind == "terminal" and d.satellite == sat)
        choices = []
        for a, b in product(terminals(source), terminals(target)):
            try:
                route = self._route(a, b, size)
            except ValueError as error:
                if str(error).startswith("no route from "):
                    continue
                raise
            choices.append((self._p2p(route, size), a, b, route))
        if not choices:
            raise ValueError(f"no active terminal route from {source} to {target}")
        ns, _, _, route = min(choices)
        return Estimate(ns, "satellite_p2p", 1, 1, 1, size, (route,))

    def estimate(self, op: CommunicationTask) -> Estimate:
        if not isinstance(op, CommunicationTask):
            raise TypeError("estimate requires a CommunicationTask")
        p, n, size = op.participants, len(op.participants), op.bytes
        if n < 2 or len(set(p)) != n or type(size) is not int or size < 0:
            raise ValueError("expected distinct participants and nonnegative integer bytes")
        if op.endpoint_kind == "satellite":
            if (n != 2 or op.operation != "send_recv" or op.bytes_semantics != "payload"
                    or op.algorithm not in (None, "send_recv", "direct") or op.reduction is not None
                    or op.partition_bytes or op.traffic_matrix_bytes or op.ring_order):
                raise ValueError("satellite transfer requires an ordered source, target and payload")
            return self._satellite_p2p(*p, size)
        if op.endpoint_kind != "gpu":
            raise ValueError(f"unknown endpoint kind: {op.endpoint_kind}")
        if any(x not in self.topology.devices or self.topology.devices[x].kind != "gpu" for x in p):
            raise ValueError("participants must be physical GPU IDs")
        groups = defaultdict(list)
        for x in p:
            groups[self._sat(x)].append(x)
        if op.operation in ("all_gatherv", "reduce_scatterv", "broadcast"):
            if op.algorithm not in ("direct", "pairwise") or op.ring_order or op.traffic_matrix_bytes:
                raise NotImplementedError("direct transfers require an explicit direct/pairwise layout")
            if op.operation == "broadcast":
                if op.bytes_semantics != "payload" or op.partition_bytes or op.reduction is not None:
                    raise ValueError("broadcast requires payload bytes and the first participant as root")
                edges = [(p[0], target, size) for target in p[1:]]
            else:
                gathering = op.operation == "all_gatherv"
                if (op.bytes_semantics != ("aggregate_input" if gathering else "per_rank_input")
                        or op.reduction != (None if gathering else "sum")
                        or len(op.partition_bytes) != n
                        or any(type(s) is not int or s < 0 for s in op.partition_bytes)
                        or sum(op.partition_bytes) != size):
                    raise ValueError("invalid variable collective partitions, reduction or byte semantics")
                edges = [(owner,peer,s) if gathering else (peer,owner,s)
                         for owner,s in zip(p,op.partition_bytes) if s for peer in p if peer != owner]
            routes = [self._route(a,b,s) for a,b,s in edges]
            cross = [s for a,b,s in edges if self._sat(a) != self._sat(b)]
            ns = self._wave_ns((r,e[2]) for r,e in zip(routes,edges))
            return Estimate(ns,"direct",int(bool(edges)),len(edges),len(cross),sum(cross),tuple(routes))
        if op.partition_bytes:
            raise NotImplementedError("variable partitions require a variable collective")
        if op.operation == "send_recv":
            if n != 2 or op.bytes_semantics != "payload" or op.algorithm not in (None, "send_recv","direct"):
                raise ValueError("send_recv requires two ordered GPUs and payload bytes")
            route = self._route(*p, size)
            cross = int(len(groups) > 1)
            return Estimate(self._p2p(route, size), "gpu_p2p", 1, 1, cross, size*cross, (route,))
        if op.operation not in ("all_reduce","all_gather","all_to_all"):
            raise NotImplementedError(f"unsupported collective: {op.operation}")
        if op.operation == "all_reduce" and op.reduction != "sum":
            raise NotImplementedError("all_reduce supports sum")
        if op.operation != "all_reduce" and op.reduction is not None:
            raise ValueError("unexpected reduction")
        if op.operation == "all_to_all" and (op.traffic_matrix_bytes or op.algorithm=="pairwise" or
                                             (self.expand_inter and len(groups)>1)):
            if op.algorithm not in (None,"hierarchical","pairwise") or op.ring_order:
                raise NotImplementedError("explicit AllToAll supports hierarchical or pairwise")
            matrix = op.traffic_matrix_bytes
            if matrix:
                if (op.bytes_semantics!="aggregate_input" or len(matrix)!=n or
                        any(len(row)!=n for row in matrix) or
                        any(type(v) is not int or v<0 for row in matrix for v in row) or
                        any(matrix[i][i] for i in range(n)) or sum(map(sum,matrix))!=size):
                    raise ValueError("invalid AllToAll traffic matrix or byte semantics")
            elif size % n or op.bytes_semantics!="per_rank_input":
                raise ValueError("equal AllToAll requires divisible per-rank bytes")
            phases = defaultdict(list)
            for i,a in enumerate(p):
                for j,b in enumerate(p):
                    amount = matrix[i][j] if matrix else size//n
                    if i==j or not amount:
                        continue
                    cross = self._sat(a)!=self._sat(b)
                    phase = (j-i)%n if op.algorithm=="pairwise" else int(cross)
                    phases[phase].append((self._route(a,b,amount), amount, cross))
            flows = [edge for wave in phases.values() for edge in wave]
            ns = sum(self._wave_ns((r,s) for r,s,_ in wave) for wave in phases.values())
            return Estimate(ns,"expanded_all_to_all",len(phases),len(flows),
                            sum(c for _,_,c in flows),sum(s for _,s,c in flows if c),
                            tuple(r for r,_,_ in flows))
        if op.traffic_matrix_bytes or op.bytes_semantics != "per_rank_input":
            raise NotImplementedError("only equal per_rank_input collectives are supported")
        algorithm = op.algorithm or ("hierarchical" if len(groups)>1 else "ring")
        if op.ring_order and (algorithm != "ring" or len(op.ring_order)!=n or set(op.ring_order)!=set(p)):
            raise ValueError("ring_order requires a ring permutation")
        if algorithm == "ring" and op.operation in ("all_reduce","all_gather"):
            if len(groups)>1 or op.ring_order:
                if op.operation == "all_reduce" and size % n:
                    raise ValueError("ring AllReduce requires divisible per-rank input")
                order = op.ring_order or p
                rounds = (n-1)*(2 if op.operation=="all_reduce" else 1)
                chunk = size//n if op.operation=="all_reduce" else size
                routes = tuple(self._route(x, order[(i+1)%n], chunk) for i,x in enumerate(order))
                # ProtoBridge RingAllReduce::GetSendTarget: RS follows next,
                # then AG follows prev. Reusing only forward edges changes
                # the workload on directional/asymmetric physical networks.
                forward_ns = self._wave_ns((r,chunk) for r in routes)
                if op.operation == "all_reduce":
                    reverse = tuple(self._route(x, order[(i-1)%n], chunk) for i,x in enumerate(order))
                    reverse_ns = self._wave_ns((r,chunk) for r in reverse)
                    duration = (n-1)*(forward_ns+reverse_ns)
                    paths = routes+reverse
                    cross = (n-1)*sum(self._sat(r["path"][0]) != self._sat(r["path"][-1]) for r in paths)
                else:
                    duration = (n-1)*forward_ns
                    paths = routes
                    cross = (n-1)*sum(self._sat(r["path"][0]) != self._sat(r["path"][-1]) for r in paths)
                return Estimate(duration, "explicit_ring", rounds, rounds*n,
                                cross, cross*chunk, paths)
        elif algorithm != "hierarchical" and not (len(groups)==1 and algorithm=="ring"):
            raise NotImplementedError(f"unsupported algorithm: {op.operation}/{algorithm}")
        from surrogate.theory.whitebox_surrogate_v2 import HierarchicalSurrogate
        g = len(next(iter(groups.values())))
        if any(len(m)!=g for m in groups.values()) or g < 2:
            raise NotImplementedError("baseline requires uniform domains with at least two participants")
        if op.operation in ("all_reduce","all_to_all") and size % n:
            raise NotImplementedError("hierarchical baseline requires equal integral partitions")
        D = size*n if op.operation=="all_gather" else size
        local = [self._route(a,b,size) for m in groups.values() for a,b in permutations(m,2)]
        def fabric(routes, inter):
            return profile(min(r["bandwidth_gbps"] for r in routes),
                           max(r["latency_ns"] for r in routes), inter)
        su = fabric(local, False)
        operation = op.operation.replace("_","")
        if len(groups)==1:
            us = su.predict(D, g, "ring", 128, collective=operation)
            mode = "intra_surrogate"
        else:
            if len(groups)!=2:
                raise NotImplementedError("baseline supports two satellites")
            # One shared ISL per direction. Divide its aggregate wire rate across
            # concurrent rank lanes before evaluating the per-lane So phase.
            remote = []
            common_direct = True
            for left,right in permutations(groups.values(),2):
                pairs = product(left,right) if operation=="alltoall" else zip(left,right)
                routes = [self._route(a,b,size) for a,b in pairs]
                def crossings(r):
                    return tuple((a,b) for a,b in zip(r["path"],r["path"][1:]) if self._sat(a)!=self._sat(b))
                isl = crossings(routes[0])
                if len(isl)!=1 or any(crossings(r)!=isl for r in routes):
                    common_direct = False
                remote.extend(routes)
            if not common_direct or self.expand_inter:
                if operation not in ("allreduce","allgather"):
                    raise NotImplementedError("irregular ISL AllToAll requires expanded mode")
                local_data = D if operation=="allreduce" else D//2
                passes = ("rs","ag") if operation=="allreduce" else ("ag",)
                local_us = sum(su.predict(local_data,g,"ring",128,collective=operation,phase=phase)
                               for phase in passes)
                chunk = D//(2*g) if operation=="allreduce" else D//2
                cross_ns = len(passes)*max(self._p2p(r,chunk) for r in remote)
                count = len(remote)*len(passes)
                return Estimate(ceil(local_us*1000)+cross_ns if size else 0,"hierarchical_phases",
                                None,None,count,count*chunk,tuple(remote))
            so = profile(min(r["bandwidth_gbps"] for r in remote)/g,
                         max(r["latency_ns"] for r in remote), True)
            us = HierarchicalSurrogate(su, so, g, 2).predict(D,"ring",128,collective=operation)
            mode = "hierarchical_surrogate"
        if not isfinite(us) or us<0:
            raise ValueError("invalid collective surrogate duration")
        return Estimate(ceil(us*1000) if size else 0, mode)
