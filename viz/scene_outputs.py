"""Export a V1 scene's saved evidence without rerunning its workload."""
import json
from types import SimpleNamespace


def export(output,result,config,physics,*,progress=None,on_ready=None):
    from result_analysis import native_analysis
    from native_live import physical_export
    from viz.export import export_reports
    from scene_recording import build_recording
    identity=output.name
    if progress:progress('生成请求结果与分析摘要')
    (output/'analysis.json').write_text(json.dumps(native_analysis(result,identity),ensure_ascii=False))
    if physics is not None:
        if progress:progress('整理场景几何快照')
        physical_export(output,SimpleNamespace(net=physics),dict(
            trace=dict(trace_manifest=dict(time=physics.trace.time)),
            sun_positions=[dict(time_s=t,position_itrs_km=p) for t,p in physics.sun_positions]))
    summary=dict(id=identity,communication_mode='runtime_comm',communication_label='V1.0 Runtime Comm',
        timeline_mode='orbitserve',requests=result['requests'],audit=result.get('audit'),measurement=result['measurement'],
        gpu_count=sum(d['kind']=='gpu' for d in config.physical['devices']),
        flows=sum(e['kind']=='admit' for e in result.get('communication_events',[])),surrogate_operations=None)
    (output/'native_summary.json').write_text(json.dumps(summary,ensure_ascii=False))
    if on_ready:on_ready()
    if progress:progress('整理算子与通信回放')
    export_reports([output/'native_result.json'],output/'native-replay',loaded_results={str(output/'native_result.json'):result})
    from replay_pages import LIMIT, build_index
    replay = output/'native-replay/native_result.json'
    if replay.stat().st_size > LIMIT:
        if progress:progress('建立算子回放分页索引')
        build_index(replay)
    if (output/'runtime_events.jsonl.gz').is_file():
        if progress:progress('建立场景回放分页与传输索引')
        build_recording(output,force=True,configuration=result['configuration'])
    return summary
