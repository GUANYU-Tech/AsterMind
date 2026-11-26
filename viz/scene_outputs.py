"""Export a V1 scene's saved evidence without rerunning its workload."""
import json
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace


def reuse_physical_frames(output, physics, horizon_s):
    root = Path(physics.dataset_root).resolve()
    source = root / 'network_result.json'
    if not source.is_file():
        return False
    metadata = json.loads(source.read_bytes())
    if metadata.get('trace_id') != physics.trace.trace_id:
        return False
    descriptors = []
    sources = []
    for frame in metadata.get('artifacts', {}).get('frames', []):
        path = (root / frame['artifact']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return False
        name = f'frames/frame-{len(descriptors):06d}.json.gz'
        descriptors.append({**frame, 'artifact': name})
        sources.append(path)
        if frame['time_s'] >= horizon_s:
            break
    if len(descriptors) < 2 or descriptors[-1]['time_s'] < horizon_s:
        return False
    (output / 'frames').mkdir(exist_ok=True)
    for path, frame in zip(sources, descriptors, strict=True):
        copyfile(path, output / frame['artifact'])
    metadata.update(artifacts={'frames': descriptors}, events=[], routes=[], snapshots=[],
                    boundary_sample_times_s=[frame['time_s'] for frame in descriptors])
    (output / 'network_result.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
    return True


def export(output,result,config,physics,*,progress=None,on_ready=None):
    from result_analysis import native_analysis
    from native_live import physical_export
    from viz.export import export_reports
    from scene_recording import build_recording
    identity=output.name
    if progress:progress('生成请求结果与分析摘要')
    analysis = native_analysis(result, identity)
    fast = config.compute_mode == "fast_mode"
    if fast:
        analysis["operator_replay"] = False
    (output/'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False))
    if physics is not None:
        if progress:progress('整理场景几何快照')
        horizon = config.horizon_ns / 1e9
        if config.fast_orbital_route_frames is None or not reuse_physical_frames(output, physics, horizon):
            physical_export(output,SimpleNamespace(net=physics),dict(
                trace=dict(trace_manifest=dict(time=physics.trace.time)),
                sun_positions=[dict(time_s=t,position_itrs_km=p) for t,p in physics.sun_positions]),
                horizon_s=horizon if fast else None)
    summary=dict(id=identity,communication_mode='runtime_comm',communication_label='V1.0 Runtime Comm',
        timeline_mode='orbitserve',requests=result['requests'],audit=result.get('audit'),measurement=result['measurement'],
        gpu_count=sum(d['kind']=='gpu' for d in config.physical['devices']),
        flows=sum(e['kind']=='admit' for e in result.get('communication_events',[])),surrogate_operations=None)
    (output/'native_summary.json').write_text(json.dumps(summary,ensure_ascii=False))
    if on_ready:on_ready()
    if not fast:
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
