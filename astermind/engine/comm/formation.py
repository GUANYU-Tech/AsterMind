"""Collective rules with topology-adapted intra and isolated whole-path Inter."""
from dataclasses import replace

from .estimator import CommunicationEstimator, Estimate
from .task import CommunicationTask


class FormationEstimator(CommunicationEstimator):
    def __init__(self, network, settings=None, ground_owners=(), *, share_isl=False, intra_hierarchy=None, policy=None):
        super().__init__(network)
        if share_isl:
            raise ValueError('static within-wave sharing is retired; use runtime for resource competition')
        from .frame_surrogate import CommPolicy
        self.policy = policy or CommPolicy(inter_parameters=settings or {})
        self.settings = self.policy.inter_parameters
        self.ground = set(ground_owners)
        self.intra_hierarchy=intra_hierarchy

    def for_hardware(self, hardware, method):
        if hardware.hardware_id != "nvidia_h200":
            raise NotImplementedError("formation H200 profile requires nvidia_h200")
        return self

    def estimate_call(self, task, spec):
        if spec.transfer.implementation or spec.transfer.destination_hardware_id:
            raise NotImplementedError("specialized hardware transfer requires its own model")
        nics=spec.transfer.participant_nic_ids
        if len(set(nics))!=len(nics):
            raise NotImplementedError("shared source NIC capacity requires a bound hardware model")
        return self.estimate(task)

    def _route(self, a, b, size):
        try:
            return super()._route(a,b,size)
        except ValueError as error:
            raise ValueError(f"route {a} -> {b} ({size} B): {error}") from error

    def _p2p(self, route, size):
        if len({self._sat(x) for x in route["path"]})==1:
            from .intra_topology import p2p_ns
            return p2p_ns(self.network,route,size,self.intra_hierarchy)
        from .path_budget import isolated_path_ns
        return isolated_path_ns(self.network,route,size,self.policy)

    def estimate(self, task):
        if not isinstance(task, CommunicationTask) or task.endpoint_kind != "gpu":
            raise ValueError("formation mode expects a physical GPU CommunicationTask")
        p, op, size = task.participants,task.operation,task.bytes
        if (len(p)<2 or len(set(p))!=len(p) or type(size) is not int or size<0
                or any(g not in self.topology.devices or self.topology.devices[g].kind!="gpu" for g in p)):
            raise ValueError("expected distinct physical GPUs and nonnegative bytes")
        if op in ("all_gatherv","reduce_scatterv","broadcast") or (op=="all_to_all" and task.traffic_matrix_bytes):
            return CommunicationEstimator.estimate(self,replace(task,algorithm=task.algorithm or "pairwise"))
        if task.partition_bytes or task.traffic_matrix_bytes:
            raise ValueError("unexpected variable traffic for a uniform operation")
        if op=="send_recv":
            if task.ring_order or task.reduction:
                raise ValueError("unexpected ring/reduction on send_recv")
            result=CommunicationEstimator.estimate(self,task)
            mode="intra_sendrecv" if len({self._sat(x) for x in result.paths[0]["path"]})==1 else "inter_surrogate_sendrecv"
            return replace(result,mode=mode)
        if op not in ("all_reduce","all_gather","all_to_all"):
            raise NotImplementedError(f"unsupported formation operation: {op}")
        if task.bytes_semantics!="per_rank_input" or task.reduction!=("sum" if op=="all_reduce" else None):
            raise ValueError("uniform collective requires its per-rank input and reduction contract")
        requested={"mesh":"fullmesh","fc2":"fullmesh"}.get(task.algorithm,task.algorithm)
        if requested in ("fullmesh","tree") and op!="all_to_all":
            from .collective_plan import collective_waves
            if task.ring_order:
                raise ValueError("tree/fullmesh do not accept ring_order")
            if requested=='tree' and op!='all_reduce':
                raise NotImplementedError('tree is only supported for all_reduce')
            from .intra_topology import whole_estimate
            whole=whole_estimate(replace(task,algorithm=requested),self.network,self.intra_hierarchy)
            if whole is not None:return whole
            waves=collective_waves(op,requested,p,size)
            routed=[[(self._route(a,b,n),n) for a,b,n in wave] for wave in waves]
            flows=[item for wave in routed for item in wave]
            paths=tuple(r for r,_ in flows)
            cross=[n for r,n in flows if self._sat(r["path"][0])!=self._sat(r["path"][-1])]
            return Estimate(sum(self._wave_ns(wave) for wave in routed),f"explicit_{requested}",
                            len(waves),len(flows),len(cross),sum(cross),paths)
        algo = "pairwise" if op=="all_to_all" else "ring"
        if task.algorithm not in (None,algo):
            raise NotImplementedError(f"unsupported formation algorithm: {op}/{task.algorithm}")
        if op in ("all_reduce","all_to_all") and size%len(p):
            raise ValueError("equal partitions require divisible per-rank bytes")
        if task.ring_order and (algo!="ring" or len(task.ring_order)!=len(p) or set(task.ring_order)!=set(p)):
            raise ValueError("invalid ring order")
        order=task.ring_order or p
        from .intra_topology import whole_estimate
        whole=whole_estimate(replace(task,algorithm=algo),self.network,self.intra_hierarchy)
        if whole is not None:return whole
        return CommunicationEstimator.estimate(self,replace(task,algorithm=algo,ring_order=order if algo=="ring" else ()))
