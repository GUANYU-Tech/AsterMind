"""Source/port differential on identical frozen resources and route inputs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--source',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from tests.comm_fixture import Frames,fixed_routes
    sys.path.insert(0,str(args.repo))
    if args.source:
        from simulator.bootstrap import activate
        activate()
        from network_sim.comm import event_runtime as runtime
        from network_sim.comm.frame_surrogate import FrameRoutes,NoRouteError,CommPolicy
        from network_sim.comm.operator import CommunicationOperator
        from orbitserve.runtime.timeline import GlobalTimeline
    else:
        from astermind.common.bootstrap import activate
        activate()
        from astermind.engine.comm import event_runtime as runtime
        from astermind.engine.comm.frame_surrogate import FrameRoutes,NoRouteError,CommPolicy
        from astermind.engine.comm.operator import CommunicationOperator
        from astermind.engine.timeline import GlobalTimeline
    routes=fixed_routes(FrameRoutes,NoRouteError)
    # This freezes route choice for both executors; no upstream routing is
    # imported into the production adapter or compared as if it were retained.
    runtime.FrameRoutes=routes
    results=[]
    participants=('g0','g2','g1','g3')
    operators=[CommunicationOperator('send_recv','send_recv',('g0','g1'),1_000_000,'payload'),
        CommunicationOperator('all_reduce','ring',participants,1024,'per_rank_input',reduction='sum'),
        CommunicationOperator('all_gather','ring',participants,1024,'per_rank_input'),
        CommunicationOperator('all_gatherv','pairwise',participants,24,'aggregate_input',partition_bytes=(8,0,4,12)),
        CommunicationOperator('reduce_scatterv','pairwise',participants,24,'per_rank_input',reduction='sum',partition_bytes=(8,0,4,12)),
        CommunicationOperator('all_to_all','pairwise',participants,30,'aggregate_input',traffic_matrix_bytes=((0,0,10,0),(0,0,0,20),(0,0,0,0),(0,0,0,0)))]
    for protocol in ('payload_only','inter_reference'):
        for index,op in enumerate(operators):
            for copies in (1,2):
                loop=GlobalTimeline();network=Frames(change=10e-6,route_type=routes)
                policy=CommPolicy(runtime_protocol=protocol,inter_parameters=dict(startup_us=0,hop_processing_us=0,routing_processing_us=0,fec=dict(enabled=False)))
                comm=runtime.RuntimeCommunication(network,loop,policy);done=[];failures=[]
                for n in range(copies):comm.submit(op,lambda e,n=n:done.append((n,asdict(e))),failures.append,task_id=f'cell{n}/comm')
                loop.run()
                assert not failures and len(done)==copies
                assert not (comm.active or comm.tasks or comm.flows)
                results.append(dict(protocol=protocol,operation=op.operation,copies=copies,completions=done,
                    serviced_bytes=sum(r['application_bytes'] for r in comm.trace if r['kind']=='service'),
                    peak_active_flows=comm.stats['peak_active_flows']))
                comm.close()
    args.output.write_text(json.dumps(results,sort_keys=True,indent=2))
    print(json.dumps(dict(cases=len(results))))
if __name__=='__main__':main()
