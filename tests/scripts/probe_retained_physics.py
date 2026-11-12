"""Identical route/cache query history and physical samples before/after port."""
import argparse
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.repo))
    from tests.unit.runtime.test_dynamic_network import network
    from dataclasses import replace
    from astermind.engine import NetworkState, NetworkUnavailable
    result=[]
    for topology in ((('a','b',0.,20.),('b','c',0.,20.),('a','c',0.,20.)),
                     (('a','b',0.,5.),('b','c',6.,20.),('a','d',2.,20.),('d','c',2.,20.))):
        base=network(edges=topology,bandwidth=8e9,schedule=((4.,.5),(8.,1.)),setup=.01)
        faults=(dict(target_type='link',target_id=base.edges[0]['link_id'],start_s=3.,end_s=7.),)
        for policy in ('min_hops','min_propagation','earliest_arrival'):
            state=NetworkState(base.trace,replace(base.spec,faults=faults))
            snapshots=[];paths=[]
            for at in (0.,.1,2.5,3.,4.5,7.,8.5,15.):
                if at==3.:state.observe_fault(0,faults[0],at)
                if at==7.:state.observe_fault(0,faults[0],at,restored=True)
                snapshots.append(state.snapshot(at))
                for size in (0,1000,1_000_000):
                    try:
                        arrival,legs=state.transfer('a','c',at,size,policy=policy)
                        paths.append(dict(at=at,size=size,arrival=arrival,legs=legs))
                    except NetworkUnavailable as error:paths.append(dict(at=at,size=size,error=type(error).__name__))
            result.append(dict(policy=policy,snapshots=snapshots,paths=paths))
    assert sum('arrival' in path for row in result for path in row['paths']) > 0
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2))
    print(dict(scenarios=len(result),queries=sum(len(row['paths']) for row in result)))
if __name__=='__main__':main()
