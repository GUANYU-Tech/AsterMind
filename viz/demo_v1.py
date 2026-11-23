"""Build a fresh interactive example using V1 physical and scene builders."""
import json
import time
from paths import VIZ


def configuration(root,payload):
    import console
    from studio import save,physical_data
    from native_scene import generate
    document=json.loads((VIZ/'templates/interactive_demo.json').read_text())
    physical=document['physical_network']
    # The old example used 400 bit/s to force a visible slow transfer. Native
    # device links require >= 1000 bit/s; this V1 example declares 400 Mbit/s.
    physical['network']['link_profiles']['sgl']['bandwidth_bps']=400_000_000
    model=save(root,'physical',dict(name='V1 在线示例 400 Mbps 星地接入',physical_network=physical))
    job=console.prepare_physical(model['id'])
    deadline=time.monotonic()+120
    while True:
        state=console.run_status(job['id'])
        if state['state']=='completed':break
        if state['state'] not in ('running','preparing') or time.monotonic()>deadline:
            raise ValueError(state.get('error','物理准备尚未完成，请在运行记录中查看进度'))
        time.sleep(.2)
    config=generate(root,dict(physical_model_id=model['id'],deployment=dict(
        model='configs/model/qwen3-8b.json',tp=2,pp=2,
        communication=dict(intra_mode='IntraSatelliteRoofline'))))
    config['requests']=[dict(id='demo-request',arrival_ns=6_000_000_000,
        user_device=config['ground_owners'][0]+'/terminal',cell=config['cells'][0]['id'],
        input_tokens=16,output_tokens=2,input_bytes=64,output_bytes_per_token=4)]
    if payload.get('random_faults'):
        config['runtime_network']={'fault_timing':'continuous','random_faults':payload['random_faults']}
    return config
