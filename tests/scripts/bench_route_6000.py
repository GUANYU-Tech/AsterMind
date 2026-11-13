"""Read-only benchmark of production routing; no search limits or algorithm edits."""
import json, time, heapq, statistics
from types import SimpleNamespace
from pathlib import Path
import argparse
from tests.unit.runtime.test_dynamic_network import network
from astermind.engine.policies import routing
from astermind.engine.state.route_cache import PersistentRouteCache

parser=argparse.ArgumentParser()
parser.add_argument('--output', default='tests/fixtures/routing-benchmark/after.json')
args=parser.parse_args()
edges=[]; positions={}
for r in range(60):
    for c in range(100):
        n=f'{r}:{c}'
        positions[n]=((7000.,r*500.,c*500.),)*2
        for dr,dc in ((1,0),(0,1)):
            if r+dr<60 and c+dc<100:
                edges.append((n,f'{r+dr}:{c+dc}',0.,20.))
t=time.perf_counter()
net=network(tuple(edges),bandwidth=10e9,terminals=4,positions=positions)
build=time.perf_counter()-t
print(json.dumps(dict(build_s=build,nodes=len(net.nodes),edges=len(net.edges))),flush=True)
rows=[]
original_heap=routing.heapq
original_edge=net._edge_transfer
original_adj=net._routing_adjacency
class CountAdj:
    def __init__(self, base, count): self.base,self.count=base,count
    def get(self,key,default):
        self.count['expanded']+=1
        return self.base.get(key,default)
for load in ('empty','uneven'):
    net.shared_load.clear()
    if load=='uneven':
        for i,e in enumerate(net.edges):
            for a,b in ((e['endpoint_a'],e['endpoint_b']),(e['endpoint_b'],e['endpoint_a'])):
                net.shared_load[(e['link_id'],a,b)]=i%8
    for target in ('0:1','20:20','59:99'):
        for policy in ('min_hops','min_propagation','earliest_arrival'):
            count=dict(popped=0,pushed=0,expanded=0,edge_predictions=0,queue_peak=1)
            def pop(q): count['popped']+=1; return heapq.heappop(q)
            def push(q,item):
                heapq.heappush(q,item);count['pushed']+=1
                count['queue_peak']=max(count['queue_peak'],len(q))
            def edge(*a,**kw): count['edge_predictions']+=1;return original_edge(*a,**kw)
            routing.heapq=SimpleNamespace(heappop=pop,heappush=push)
            net._edge_transfer=edge
            net._routing_adjacency=lambda kinds:CountAdj(original_adj(kinds),count)
            net._route_cache=PersistentRouteCache()
            _,path=net.transfer('0:0',target,0.,1048576,policy=policy)
            before=count.copy()
            net.transfer('0:0',target,.1,1048576,policy=policy)
            cache_counts={k:count[k]-before[k] for k in ('popped','expanded','edge_predictions')}
            count=before
            routing.heapq=original_heap;net._edge_transfer=original_edge;net._routing_adjacency=original_adj
            cold=[];warm=[]
            for repeat in range(3):
                net._route_cache=PersistentRouteCache()
                t=time.perf_counter();net.transfer('0:0',target,0.,1048576,policy=policy);cold.append(time.perf_counter()-t)
                t=time.perf_counter();net.transfer('0:0',target,.1,1048576,policy=policy);warm.append(time.perf_counter()-t)
            row=dict(load=load,target=target,policy=policy,hops=len(path),**count,
                cold_ms=statistics.median(cold)*1000,cache_ms=statistics.median(warm)*1000,
                cache_info=net.route_cache_info(),cache_counts=cache_counts)
            rows.append(row);print(json.dumps(row),flush=True)
            Path(args.output).write_text(json.dumps(dict(nodes=6000,edges=len(edges),build_s=build,rows=rows),indent=2))
