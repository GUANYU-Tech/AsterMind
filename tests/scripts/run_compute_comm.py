"""Run real pinned schedulers, compute predictions and Comm; no dependency skips."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from astermind.__main__ import json_default
from astermind.builder.config import SimulationConfig
from astermind.engine.simulation import Simulation
from astermind.engine.audit import audit
from examples.native.compute_comm_matrix import scene


def run(case, mode, intra, destination):
    if case=='DS02':
        from cross_cell_competition import run as competition
        solo=competition(mode,intra,concurrent=False)
        result=competition(mode,intra,concurrent=True)
        assert int(result['completion_ticks']['cell0'])>=int(solo['completion_ticks']['cell0'])
        result['solo_completion_ticks']=solo['completion_ticks']
        (destination/f'{case}-{mode}-{intra}.report.json').write_text(json.dumps(result,default=json_default))
        print(json.dumps(dict(case=case,compute_mode=mode,intra_mode=intra,audit=result['audit'])),flush=True)
        return
    raw=scene(case,compute_mode=mode,intra_mode=intra)
    config=SimulationConfig.model_validate(raw)
    (destination/f'{case}-{mode}-{intra}.input.json').write_text(json.dumps(raw,indent=2))
    with Simulation(config,cell_init_workers=1) as simulation:
        result=simulation.run()
        assert not simulation.network.communication_policy.tasks, 'leaked tasks'
        assert not simulation.network.communication_policy.flows, 'leaked flows'
    if case=='DS08-kv-failure':
        assert result['status']=='completed_with_failures',result['requests']
        handoffs=result['cells']['cellA']['pd_handoffs']
        assert handoffs and all(h['failures'] for h in handoffs.values()),result['requests']
        assert not result['cells']['cellA']['pools']['decode']['steps']
    elif case.endswith('idle'):
        assert result['status']=='completed_with_failures',result['requests']
        assert all(state['status']=='failed' and 'scheduled token ownership is unbound' in state['failure']['reason']
                   for state in result['requests'].values()),result['requests']
        assert not result['produced_outputs']
    else:
        assert result['status']=='completed',result['requests']
    result['audit']=audit(result)
    path=destination/f'{case}-{mode}-{intra}.report.json'
    path.write_text(json.dumps(result,default=json_default))
    print(json.dumps(dict(case=case,compute_mode=mode,intra_mode=intra,audit=result['audit'])),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',nargs='+',default=['DS01','DS03','DS04','DS06-copy','DS06-move'])
    parser.add_argument('--modes',nargs='+',default=['midlayer','roofline'])
    parser.add_argument('--intra',nargs='+',default=['surrogate'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for case in args.cases:
        for mode in args.modes:
            for intra in args.intra:run(case,mode,intra,args.output)


if __name__=='__main__':main()
