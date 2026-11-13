"""Compute interface to the surrogate using only the public Network State Frame."""
from types import SimpleNamespace
from astermind.builder.config.communication import CommPolicy


class NoRouteError(ValueError):
    pass


class CommunicationRouting:
    """Route and candidate queries delegated to the retained target network."""
    def route(self, source, target, size_bytes=0, *, first_hop=None):
        routes = self._refresh()
        result = routes.route(source, target, size_bytes, first_hop=first_hop)
        return {**result, 'frame_revision': routes.frame.revision,
                'frame_time_s': routes.frame.start_time_s}

    def candidate_path(self, source, target):
        return self.network.candidate_path(source, target)


class FrameRoutes:

    def __init__(self,frame):
        self.frame=frame
        self.routing_policy="earliest_arrival"
        self.ground_owners={v.satellite_id for v in frame.node_states.values() if v.role=="ground"}
        self.topology=SimpleNamespace(devices={n:SimpleNamespace(kind=v.attributes["device_kind"],
            satellite=v.satellite_id) for n,v in frame.node_states.items()})
        self.adj={n:[] for n in frame.node_states}
        self.edges={}
        for link in frame.established_topology_graph:
            a,b=link.source_node_id,link.target_node_id
            if any(k in link.attributes for k in ("reverse_bandwidth_bps","reverse_bandwidth_gbps","reverse_latency_ns","reverse_propagation_delay_s")):
                raise ValueError(f"{link.link_id}: asymmetric parameters require a directional Frame contract")
            pairs=[(a,b),(b,a)] if link.attributes.get("bidirectional",True) else [(a,b)]
            for src,dst in pairs:
                key=(link.link_id,src,dst)
                self.edges[key]=dict(index=len(self.edges),source=src,target=dst,
                    source_owner=frame.node_states[src].satellite_id,target_owner=frame.node_states[dst].satellite_id,
                    bandwidth_gbps=link.bandwidth_bps/1e9,latency_ns=link.propagation_delay_s*1e9)
                self.edges[key]["resource_key"]=key
                directional=link.attributes.get('forward_shared_capacity_pools' if src==a else 'reverse_shared_capacity_pools',())
                self.edges[key]["shared_pools"]=tuple(dict.fromkeys((*getattr(link,"shared_capacity_pools",()),*directional)))
                self.edges[key]["attributes"]=dict(link.attributes)
                direction=link.attributes.get("duplex", "full_duplex")
                if direction not in ("full_duplex", "shared"):
                    raise ValueError(f"unknown duplex mode: {direction}")
                if direction=="shared":self.edges[key]["resource_key"]=(link.link_id,"shared")
                self.adj[src].append(key)

    def _result(self,path,keys,size):
        bandwidth=min((self.edges[k]["bandwidth_gbps"] for k in keys),default=float("inf"))
        # With intra contention disabled this is a per-transfer ceiling, not
        # a claim that concurrent transfers share the switch aggregate.
        if len({self.topology.devices[n].satellite for n in path}) == 1:
            switch_caps = [self.frame.node_states[n].attributes.get('capacity_gbps') for n in path
                           if self.topology.devices[n].kind == 'switch']
            bandwidth = min([bandwidth] + [c for c in switch_caps if c is not None])
        latency=sum(self.edges[k]["latency_ns"] for k in keys)
        return dict(path=list(path),resources=keys,latency_ns=latency,bandwidth_gbps=bandwidth,
                    isolated_estimate_ns=latency+(0. if bandwidth==float("inf") else 8*size/bandwidth),
                    routing_policy=self.routing_policy)





    def route_resources(self,route):
        return [dict(self.edges[key],reverse=self.edges.get((key[0],key[2],key[1])))
                for key in route["resources"]]
