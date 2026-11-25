"""Reconcile local assets against the executable physics contract.

Obsolete files leave all active collections. A manifest records their reason and
original location; historical execution evidence remains recoverable locally.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import threading
from zoneinfo import ZoneInfo

LOCK = threading.RLock()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dataset_folder(folder):
    folder = Path(folder)
    return folder/'network' if (folder/'network/physical_dataset.json').is_file() else folder


def dataset_islsim_problem(reference):
    from astermind.engine.setup.islsim import ISLSimUnavailable, checkout, source_identity

    folder = Path(reference)
    if folder.is_file():
        folder = folder.parent
    try:
        profiles = read(folder/'materialized_input.json')['network']['link_profiles']
        optical = [profile for profile in profiles.values() if profile.get('capacity_model') == 'islsim']
        if optical:
            current = source_identity(checkout())
            if any(profile.get('islsim_source_sha256') != current for profile in optical):
                return '物理数据使用旧版 ISLSim，请打开物理模型并重新生成数据'
    except (OSError, ValueError, KeyError, TypeError, ISLSimUnavailable) as error:
        return '无法确认物理数据的 ISLSim 版本：'+str(error)
    return None


def dataset_problem(folder, thorough=False):
    from astermind.builder.physical.physical_dataset import SCHEMA, LEGACY_SCHEMA, PHYSICS_VERSION
    from astermind.builder.physical.network import TERMINAL_ALLOCATION_MODEL
    folder = dataset_folder(folder)
    try:
        data = read(folder/'physical_dataset.json')
        if data.get('schema_version') not in {SCHEMA, LEGACY_SCHEMA} or data.get('physics_version') != PHYSICS_VERSION:
            return '物理数据版本已过期'
        required = {'physical_state.sqlite', 'materialized_input.json', 'orbit_trace/trace_manifest.json'}
        required.update({'core_result.json','state/index.json','state/timeline.json'} if data['schema_version']==SCHEMA
                        else {'network_result.json','node_positions.jsonl.gz','link_attributes.jsonl.gz','link_events.jsonl.gz','astermind_network.jsonl.gz'})
        if not required <= data.get('files', {}).keys():
            return '物理数据索引不完整'
        for name in data['files']:
            file = (folder/name).resolve()
            if not file.is_relative_to(folder.resolve()) or not file.is_file():
                return '物理数据文件缺失或路径不合法：'+name
        result = read(folder/('core_result.json' if data['schema_version']==SCHEMA else 'network_result.json'))
        provenance = result.get('provenance', {})
        if provenance.get('terminal_allocation_model') != TERMINAL_ALLOCATION_MODEL or provenance.get('between_samples') != 'sample_hold':
            return '物理模型语义已过期'
        if thorough:
            from astermind.builder.physical.physical_dataset import load_dataset
            load_dataset(folder, verify_exports=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return '物理数据不可用：'+str(error)
    return None


def native_run_problem(folder, thorough=False):
    """Validate the explicit native-preset artifact type independently of datasets."""
    import hashlib
    folder = Path(folder).resolve()
    try:
        manifest = read(folder/'native_run_manifest.json')
        if manifest.get('schema_version') != 'orbitserve.native_preset_run.v1' or manifest.get('preset') != 'qwen3-4b-tp2pp2':
            return '原生预设结果索引不受支持'
        deployment = manifest['deployment']
        if (deployment.get('model'),deployment.get('transport_model'),deployment.get('tp'),deployment.get('pp')) != ('qwen3-4b','flowsim',2,2):
            return '原生预设模型或后端不匹配'
        files = manifest['files']
        if not {'network_result.json','experiment_result.json','native_serving.json','runtime_playback/index.json'} <= files.keys():
            return '原生预设结果不完整'
        for name, digest in files.items():
            path = (folder/name).resolve()
            if not path.is_relative_to(folder) or not path.is_file():
                return '原生预设文件缺失或路径不合法：'+name
            if thorough and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return '原生预设文件校验失败：'+name
        native = read(folder/'native_serving.json')
        if native['backend'] != 'flowsim' or native['parallel'] != 'tp_pp':
            return '原生执行记录与预设不一致'
    except (OSError,ValueError,KeyError,TypeError) as error:
        return '原生预设结果不可用：'+str(error)
    return None


def scene_problem(root, document, thorough=False):
    root = Path(root)
    try:
        experiment = document['experiment']
        reference = experiment.get('physical_dataset')
        if document.get('physical_model_id'):
            identity = document['physical_model_id']
            if not (root/'local/inputs/physical'/(identity+'.json')).is_file():
                return '依赖的物理模型不存在'
            binding = read(root/'local/inputs/physical-data'/(identity+'.json'))
            reference = root/'local/results/network'/binding['run_id']
            if (reference/'status.json').is_file() and read(reference/'status.json').get('state') != 'completed':
                return '依赖的物理预计算未成功完成'
        if not reference:
            return '场景没有完整物理数据'
        problem = dataset_problem(root/reference)
        if problem:
            return problem
        if experiment.get('schema_version') in (1, 'orbitserve.native_cells.v1'):
            if thorough:
                from native_scene import validate_document
                validate_document(root, document)
            return None
        if not any(experiment.get(k) for k in ('local_smoke_workload','local_contention_workload','local_fault_workload')):
            if not (root/experiment['workload']['bundle_dir']/'workload_manifest.json').is_file():
                return '场景依赖的 Workload 不存在'
        if thorough:
            from studio import scene_input, validate_model
            from console import prepare_input
            value = scene_input(root, document['id'])
            validate_model(value['physical_network'], root)
            prepare_input(value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return '场景不可用：'+str(error)
    return None


def owned(root, path):
    path = Path(path)
    root = Path(root).resolve()
    if not path.resolve().is_relative_to(root/'local'):
        raise ValueError('清理目标必须位于本项目 local 内：'+str(path))
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or (hasattr(cursor, 'is_junction') and cursor.is_junction()):
            raise ValueError('清理目标包含链接：'+str(cursor))
        cursor = cursor.parent
    return path


def reconcile(root, *, apply=False, thorough=False):
    root = Path(root).resolve()
    with LOCK:
        from tasks import processes
        from live import SESSIONS
        if any(s.payload['state'] in {'preparing','paused','advancing','finalizing','stopping'} for s in list(SESSIONS.values())):
            raise ValueError('有在线仿真正在使用数据，清理延后到实验结束')
        outputs = list((root/'local/results/network').glob('*'))
        if any(p.is_dir() and processes(root,p.name) for p in outputs):
            raise ValueError('有仿真进程正在使用数据，清理延后到实验结束')
        plan = {}
        def add(path, reason, kind):
            path = owned(root,path)
            if path.exists():
                plan[str(path)] = dict(path=path.relative_to(root).as_posix(), reason=reason, kind=kind)
        for folder in outputs:
            if not folder.is_dir():
                continue
            problem = dataset_problem(folder,thorough)
            try:
                status = read(folder/'status.json') if (folder/'status.json').is_file() else {}
                if status.get('native_preset'):
                    problem = native_run_problem(folder,thorough)
                if status.get('state') in {'failed','cancelled','interrupted','stopping'}:
                    problem = '运行未成功完成：'+status['state']
                manifest = root/'local/inputs/network'/folder.name/'manifest.json'
                if manifest.is_file():
                    value = read(manifest)
                    if value.get('physical_dataset'):
                        problem = problem or dataset_problem(root/value['physical_dataset'])
            except (ValueError,OSError,TypeError) as error:
                problem = '运行记录损坏：'+str(error)
            if problem:
                add(folder,problem,'runs')
                add(root/'local/inputs/network'/folder.name,problem,'run_inputs')
        for path in (root/'local/inputs/scenes').glob('*.json'):
            try:
                problem = scene_problem(root,read(path),thorough)
            except (ValueError,OSError,TypeError) as error:
                problem = '场景文件损坏：'+str(error)
            if problem:
                add(path,problem,'scenes')
        stale_models = set()
        for path in (root/'local/inputs/physical-data').glob('*.json'):
            try:
                source = root/'local/results/network'/read(path)['run_id']
                problem = dataset_problem(source) or (plan.get(str(source)) or {}).get('reason')
            except (ValueError,OSError,KeyError,TypeError) as error:
                problem = '物理数据绑定失效：'+str(error)
            if problem:
                add(path,problem,'bindings')
                stale_models.add(path.stem)
        for identity in stale_models:
            add(root/'local/inputs/physical'/(identity+'.json'),'绑定的物理数据已失效','physical_models')
        # Broken and byte-equivalent draft revisions do not belong in the UI.
        seen = set()
        protected_models = {p.stem for p in (root/'local/inputs/physical-data').glob('*.json') if str(p) not in plan}
        for path in (root/'local/inputs/scenes').glob('*.json'):
            if str(path) not in plan:
                try:
                    protected_models.add(read(path).get('physical_model_id'))
                except (OSError,ValueError,TypeError):
                    pass
        for path in sorted((root/'local/inputs/physical').glob('*.json'),key=lambda p:(p.stem in protected_models,p.stat().st_mtime),reverse=True):
            if str(path) in plan:
                continue
            try:
                document = read(path)
                from studio import validate_model
                from astermind.engine import physical_network_settings
                from copy import deepcopy
                physical = deepcopy(document['physical_network'])
                physical['network'] = physical_network_settings(physical['network'])
                if thorough:
                    validate_model(physical,root)
                fingerprint = json.dumps(physical,sort_keys=True)
                if fingerprint in seen and path.stem not in protected_models:
                    add(path,'重复的物理模型旧修订','physical_models')
                else:
                    seen.add(fingerprint)
            except (ValueError,OSError,KeyError,TypeError) as error:
                add(path,'物理模型不可用：'+str(error),'physical_models')
        # A healthy run may share a workload kept under an older run's input
        # directory. Retain those live dependencies without retaining its stale
        # manifest or result. Never rewrite immutable executed inputs.
        dependencies = {}
        candidates = list((root/'local/inputs/network').glob('*/manifest.json'))
        candidates.extend((root/'local/inputs/scenes').glob('*.json'))
        def strings(value):
            if isinstance(value,dict):
                for child in value.values():
                    yield from strings(child)
            elif isinstance(value,list):
                for child in value:
                    yield from strings(child)
            elif isinstance(value,str) and ('/' in value or '\\' in value):
                yield value
        for path in candidates:
            if any(path.is_relative_to(root/row['path']) for row in plan.values()):
                continue
            try:
                for value in strings(read(path)):
                    for base in (path.parent,root):
                        reference = (base/value).resolve()
                        if reference.exists() and reference.is_relative_to(root/'local'):
                            dependencies.setdefault(reference,set()).add(path.relative_to(root).as_posix())
            except (OSError,ValueError,TypeError):
                continue
        retained = []
        for source,row in tuple(plan.items()):
            source = Path(source)
            needed = {p:owners for p,owners in dependencies.items() if p.is_relative_to(source)}
            if not needed:
                continue
            plan.pop(str(source))
            retained.extend(dict(path=p.relative_to(root).as_posix(),required_by=sorted(owners)) for p,owners in needed.items())
            def split(directory):
                for child in directory.iterdir():
                    if any(child.is_relative_to(p) for p in needed):
                        continue
                    if any(p.is_relative_to(child) for p in needed):
                        if child.is_dir():
                            split(child)
                    else:
                        add(child,row['reason'],row['kind'])
            if source.is_dir():
                split(source)
        report = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                      counts=dict(Counter(row['kind'] for row in plan.values())),
                      retained_dependencies=retained,entries=list(plan.values()))
        if not apply or not plan:
            return report
        archive = root/'local/archive'/('obsolete-'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f'))
        archive.mkdir(parents=True,exist_ok=False)
        report['archive'] = str(archive)
        report['moved'] = []
        audit = archive/'cleanup-manifest.json'
        audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        # Children first, then run trees. Original absolute paths are recorded.
        for row in plan.values():
            source = owned(root,root/row['path'])
            target = owned(root,archive/row['path'])
            target.parent.mkdir(parents=True,exist_ok=True)
            source.rename(target)
            report['moved'].append(row['path'])
            audit.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--thorough',action='store_true')
    args = parser.parse_args()
    if args.apply:
        from server import Server
        if Server().find() is not None:
            parser.error('请先执行 server end；在线服务运行期间不执行离线清理')
    report = reconcile(__import__('paths').WORKSPACE,apply=args.apply,thorough=args.thorough)
    print(json.dumps(report,ensure_ascii=False,indent=2))
