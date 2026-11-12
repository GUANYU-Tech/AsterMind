"""Emit a deterministic source/port oracle from real unpriced native plans.

Run in separate interpreters: this never mixes two native snapshots in sys.path.
"""
import argparse
from collections.abc import Mapping
from dataclasses import fields,is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import sys

def canonical(value):
    if isinstance(value,Enum):return value.value
    if is_dataclass(value):return {f.name:canonical(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,Mapping):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    if isinstance(value,(set,frozenset)):return sorted(canonical(v) for v in value)
    if isinstance(value,Path):return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(type(value))

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--source',action='store_true');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.repo))
    if args.source:
        from simulator.bootstrap import activate
        root=activate()
        from astermind.ArchPrism.midlayer.runtime import MidlayerCellConfig,create_midlayer_runtime
    else:
        from astermind.common.bootstrap import activate
        root=activate()
        from astermind.engine.compute.runtime import MidlayerCellConfig,create_midlayer_runtime
    from src.serving.schedule.schedule_step import ScheduleStep,ScheduledRequest
    results=[]
    for case,options in [('TP2',dict()),('TP2-PP2',dict(pipeline_parallel_size=2,scheduler_adapter='vllm_0_30')),
                         ('EP2',dict(model_file='configs/model/qwen3-30b-a3b.json',enable_expert_parallel=True,expert_routing_seed=42))]:
        for mode in ('midlayer','roofline'):
            _,execution,_=create_midlayer_runtime(MidlayerCellConfig(root,
                model_file=options.get('model_file','configs/model/qwen3-8b.json'),
                hardware_file='configs/hardware/nvidia_h200.yaml',hardware_id='nvidia_h200',
                tensor_parallel_size=2,compute_mode=mode,**{k:v for k,v in options.items() if k!='model_file'}),with_scheduler=False)
            for phase,computed in (('prefill',0),('decode',2)):
                schedule=ScheduleStep('oracle-'+phase,'main',(ScheduledRequest('r',phase,2 if phase=='prefill' else 0,
                    0 if phase=='prefill' else 1,computed_tokens=computed),))
                plan=execution.prepare_step(schedule,embedding_token_ids=(42,)*schedule.token_count)
                predictions=execution.predict_local_plan(plan)
                calls=canonical(tuple(plan.calls))
                digest=hashlib.sha256(json.dumps(calls,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                row=dict(case=case,mode=mode,phase=phase,logical_calls=plan.logical_call_count,
                    plan_sha256=digest,predictions={k:v.t_total_s for k,v in predictions.items()})
                results.append(row)
    args.output.write_text(json.dumps(results,sort_keys=True,indent=2))
    print(json.dumps(dict(cases=len(results),predictions=sum(len(r['predictions']) for r in results))))
if __name__=='__main__':main()
