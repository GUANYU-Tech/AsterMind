"""Fixed test-only network; path selection is an explicit unique-path oracle."""
from types import SimpleNamespace as NS
from collections import deque

class Frames:
    horizon=None
    def __init__(self,change=None,down=False,route_type=None):
        self.change,self.down,self.route_type=change,down,route_type
    def frame_at(self,t):
        nodes={name:NS(satellite_id=owner,role='ground' if owner=='earth' else 'compute',attributes=dict(device_kind=kind))
            for name,owner,kind in [('g0','A','gpu'),('g2','A','gpu'),('a','A','terminal'),
                                   ('g1','B','gpu'),('g3','B','gpu'),('b','B','terminal'),('ground','earth','terminal')]}
        links=[]
        for source,target,bw,delay in [('g0','a',1600,10),('g2','a',1600,10),('b','g1',1600,10),
                                     ('b','g3',1600,10),('a','b',400,100),('ground','a',20,1000)]:
            if source=='a' and target=='b' and self.change is not None and t>=self.change:
                if self.down:continue
                bw=200
            links.append(NS(link_id=source+target,source_node_id=source,target_node_id=target,
                bandwidth_bps=bw*1e9,propagation_delay_s=delay/1e9,attributes=dict(bidirectional=True),shared_capacity_pools=()))
        return NS(node_states=nodes,established_topology_graph=links,revision=0,start_time_s=t,
            valid_until_s=self.change if self.change is not None and t<self.change else None)
    def routes(self,frame):
        if self.route_type is None:
            from astermind.engine.comm.frame_surrogate import FrameRoutes,NoRouteError
            self.route_type=fixed_routes(FrameRoutes,NoRouteError)
        return self.route_type(frame)


def fixed_routes(base,no_route):
    class FixedRoutes(base):
        def __init__(self,frame,*args,**kwargs):super().__init__(frame)
        def route(self,source,target,size,*,first_hop=None):
            queue=deque([(source,(source,),())]);seen={source}
            while queue:
                node,path,keys=queue.popleft()
                if node==target:return self._result(path,keys,size)
                for key in self.adj[node]:
                    nxt=key[2]
                    if nxt not in seen:
                        seen.add(nxt);queue.append((nxt,path+(nxt,),keys+(key,)))
            raise no_route('fixed fixture path unavailable')
    return FixedRoutes
