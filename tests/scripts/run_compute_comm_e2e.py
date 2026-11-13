"""Full native E2E with real physics, migration, faults and deterministic replay."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from astermind.builder.config import SimulationConfig
from astermind.engine.simulation import Simulation
from astermind.engine.audit import audit
from astermind.__main__ import json_default
from examples.native.compute_comm_e2e import scene

def run(case,mode,destination,suffix=''):
    raw=scene(mode,case);name=f'{case}-{mode}{suffix}'
    (destination/(name+'.input.json')).write_text(json.dumps(raw,indent=2))
    with Simulation(SimulationConfig.model_validate(raw),cell_init_workers=1,
                    physical_output_dir=destination/(name+'-physics')) as simulation:
        result=simulation.run()
        assert not (simulation.network.communication_policy.tasks or simulation.network.communication_policy.active or simulation.network.communication_policy.flows)
    assert result['status']==('completed_with_failures' if case=='timeout' else 'completed'),result['requests']
    if case=='timeout':assert all(r['status']=='failed' for r in result['requests'].values())
    else:
        assert result['migrations']['kv']['state']=='completed'
        assert int(result['requests']['child']['cell_ready_ticks'])>=int(result['requests']['parent']['compute_complete_ticks'])
    if case=='fault':
        kinds={row['kind'] for row in result['network']['event_log']}
        assert {'fault_injected','fault_restored'}<=kinds
    result['audit']=audit(result)
    (destination/(name+'.report.json')).write_text(json.dumps(result,default=json_default))
    print(json.dumps(dict(case=case,mode=mode,audit=result['audit'])),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['midlayer','roofline']);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for mode in args.modes:
        normal=run('normal',mode,args.output)
        repeated=run('normal',mode,args.output,'-repeat')
        for key in ('requests','migrations','decisions','produced_outputs','transfers','batches'):
            assert normal[key]==repeated[key],('nondeterminism',key)
        run('fault',mode,args.output);run('timeout',mode,args.output)
if __name__=='__main__':main()
