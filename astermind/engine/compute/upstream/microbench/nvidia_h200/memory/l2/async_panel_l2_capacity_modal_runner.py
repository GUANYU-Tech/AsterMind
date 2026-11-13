"""Pure async-byte-copy L2 capacity and counter attribution, not a GEMM."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
TASK=Path('/tmp/midlayer-async-panel-l2-capacity-20261107')
SOURCE=HERE/'async_panel_l2_capacity.cu' if HERE.name=='l2' else TASK/'async_panel_l2_capacity.cu'
IMAGE=(modal.Image.from_registry('nvidia/cuda:12.6.3-devel-ubuntu22.04',add_python='3.11')
    .apt_install('cuda-nsight-compute-12-6').add_local_file(SOURCE,remote_path=str(TASK/SOURCE.name)))
APP=modal.App('midlayer-async-panel-l2-capacity-20261107')
def run(cmd,limit):
    t=time.monotonic();p=subprocess.run(cmd,cwd=TASK,capture_output=True,text=True,timeout=limit)
    return dict(command=cmd,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.monotonic()-t)
@APP.function(gpu='H200',image=IMAGE,timeout=240,retries=0,max_containers=1,single_use_containers=True)
def capture():
    t=time.monotonic();result={'boundary':'pure_async_copy_no_matrix_routing_or_production_duration','cells':[],
        'selection':'per-cell smaller of two process medians, then largest cell; counter DRAM/logical read <=1%, within-cell IQR/median <=3%, cross-rotation relative median difference <=3%; no production-error selection'}
    try:
        for name,cmd,limit in (
            ('gpu',['nvidia-smi','--query-gpu=name,uuid,clocks.sm,clocks.mem,driver_version','--format=csv,noheader'],20),
            ('nvcc',['nvcc','--version'],20),('ncu',['ncu','--version'],20),
            ('build',['nvcc','-O3','-std=c++17','-arch=sm_90','-Xptxas=-v',SOURCE.name,'-o','probe'],80),
            ('resources',['cuobjdump','--dump-resource-usage','probe'],20),
            ('sass',['cuobjdump','--dump-sass','probe'],30)):
            result[name]=run(cmd,limit)
            if result[name]['exit_code']:result['status']=name+'_failed';return result
        sass=result['sass']['stdout']
        if 'LDGSTS.E.BYPASS.128' not in sass or 'HGMMA.' in sass or 'HMMA.' in sass:
            result['status']='sass_boundary_failed';return result
        for rotation in range(2):
            for backing in (8,32):
                for grid in (132,264,528,1056):
                    for rounds in (256,1024):
                        cell=run([str(TASK/'probe'),str(grid),str(backing),str(rounds),'5'],12)
                        result['cells'].append(dict(rotation=rotation,measurement=cell))
                        if cell['exit_code']:result['status']='measurement_failed';return result
        result['counter']=run(['ncu','--csv','--page','raw','--cache-control','none','--clock-control','none',
            '--kernel-name','regex:async_panel_l2','--launch-skip','5','--launch-count','1','--metrics',
            'lts__t_sectors_op_read.sum,dram__bytes_read.sum',str(TASK/'probe'),'264','8','1024','3'],75)
        result['binary_sha256']=hashlib.sha256((TASK/'probe').read_bytes()).hexdigest()
        result['status']='complete' if result['counter']['exit_code']==0 else 'counter_failed'
        return result
    finally:
        result['gpu_function_wall_s']=time.monotonic()-t
        shutil.rmtree(TASK)
@APP.local_entrypoint()
def main(output_dir:str):
    root=HERE.parents[3];target=Path(output_dir).resolve()
    if target.parent!=(root/'profiler_output/microbench/nvidia_h200/async_panel_l2_capacity_20261107').resolve() or target.name!='attempt_1' or target.exists():
        raise ValueError('output must be a new registered attempt_1')
    t=time.monotonic();result=capture.remote();result['client_wall_s']=time.monotonic()-t;result['app_id']=APP.app_id
    result['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    target.mkdir(parents=True);(target/'raw.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'app_id':result['app_id'],'gpu_function_wall_s':result['gpu_function_wall_s']}))
