"""Exact structural regression comparison with named host-only exceptions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

WALL_PHASES = {'workload_construction', 'timeline_construction', 'kernel_lowering', 'kernel_prediction'}


def scheduler_runtime(path):
    tail = path[2:] if len(path)>2 and path[0]=='cells' and isinstance(path[1], str) else ()
    while len(tail)>2 and tail[0] in ('pools', 'replicas'):
        tail = tail[2:]
    return tail == ('scheduler_runtime',) or (
        len(tail)==3 and tail[:2]==('scheduler_runtime', 'replicas') and type(tail[2]) is int)


def config_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def startup_progress(rows, cell_ids):
    """Validate host FIRST_COMPLETED progress; retain initial/final boundaries."""
    indices = [i for i,row in enumerate(rows) if isinstance(row,dict) and row.get('phase')=='cells']
    if not indices:
        return rows
    assert cell_ids is not None and len(indices)>=2
    assert indices == list(range(indices[0], indices[-1]+1)), 'Interleaved startup phases'
    selected = [rows[i] for i in indices]
    total, workers = selected[0]['total'], selected[0]['workers']
    assert type(total) is int and total==len(cell_ids)
    assert type(workers) is int and 1<=workers<=total
    previous = -1
    for row in selected:
        assert set(row)=={'phase','completed','total','workers','active','wall_s'}
        assert type(row['total']) is int and type(row['workers']) is int
        assert row['total']==total and row['workers']==workers
        count, active, wall = row['completed'], row['active'], row['wall_s']
        assert type(count) is int and previous<count<=total
        assert isinstance(active,list) and all(isinstance(x,str) for x in active)
        assert len(active)==len(set(active))==min(workers,total-count)
        assert set(active)<=cell_ids
        assert type(wall) in (int,float) and math.isfinite(wall) and wall>=0
        previous = count
    assert selected[0]['completed']==0 and selected[-1]['completed']==total
    return rows[:indices[0]]+[selected[0],selected[-1]]+rows[indices[-1]+1:]


def differences(a, b, path=()):
    if type(a) is not type(b):
        yield path, a, b
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            yield (*path, '<keys>'), sorted(a), sorted(b)
        else:
            for key in a:
                yield from differences(a[key], b[key], (*path, key))
    elif isinstance(a, list):
        if len(a) != len(b):
            yield (*path, '<length>'), len(a), len(b)
        else:
            for index, (left, right) in enumerate(zip(a, b)):
                yield from differences(left, right, (*path, index))
    elif a != b:
        yield path, a, b


def host_observation(path, a, b):
    tail = path[2:] if len(path)>2 and path[0]=='cells' and isinstance(path[1], str) else ()
    while len(tail)>2 and tail[0] in ('pools', 'replicas'):
        tail = tail[2:]
    pid = bool(path) and path[-1]=='worker_pid' and scheduler_runtime(path[:-1])
    performance = len(tail)==3 and tail[:2]==('execution_performance', 'phase_wall_time_s') and tail[2] in WALL_PHASES
    timing = path in (
        ('communication_runtime_stats', 'allocation_wall_s'), ('communication_runtime_stats', 'route_wall_s'),
        ('measurement', 'runtime_startup_wall_s'), ('measurement', 'event_loop_wall_s'),
        ('measurement', 'cell_startup_policy', 'memory_slots'),
        ('measurement', 'cell_startup_policy', 'available_memory_bytes'))
    stage = len(path)==4 and path[:2]==('measurement', 'startup_stages') and type(path[2]) is int and path[3]=='wall_s'
    if not (pid or performance or timing or stage):
        return False
    if pid:
        assert all(type(x) is int and x>0 for x in (a,b)), (path,a,b)
    else:
        assert all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in (a,b)), (path,a,b)
    return True


def normalize(value, output, repo, path=(), cell_ids=None):
    if isinstance(value, dict):
        if not path and isinstance(value.get('cells'),dict):
            cell_ids = set(value['cells'])
        derived = scheduler_runtime(path) and 'source_scheduler_config_sha256' in value
        if derived:
            assert value['source_scheduler_config_sha256']==config_digest(value['source_scheduler_config']), 'Invalid source Scheduler config receipt'
        result = {k:normalize(v, output, repo, (*path,k), cell_ids) for k,v in value.items()}
        if derived:
            # Verify the original receipt first, then compare the full normalized
            # configuration and its recomputed hash. Never ignore config hashes.
            result['source_scheduler_config_sha256'] = config_digest(result['source_scheduler_config'])
        return result
    if isinstance(value, list):
        if path==('measurement','startup_stages'):
            value = startup_progress(value, cell_ids)
        return [normalize(v, output, repo, (*path,i), cell_ids) for i,v in enumerate(value)]
    if isinstance(value, str):
        for prefix, marker in ((output/'distribution/installed', '$INSTALLED'), (output, '$OUTPUT'), (repo, '$REPO')):
            if value == str(prefix) or value.startswith(str(prefix)+'/'):
                return marker+value[len(str(prefix)):]
        # Only Python traceback File entries may embed artifact-root prefixes.
        # Error messages, relative file names, line numbers and code remain exact.
        if path and path[-1]=='cleanup_error' and 'Traceback (most recent call last):\n' in value:
            for prefix,marker in ((output/'distribution/installed','$INSTALLED'),(output,'$OUTPUT'),(repo,'$REPO')):
                value = re.sub(r'(?m)^([ \t]*File ")'+re.escape(str(prefix))+r'/(?=[^"\n]+", line [0-9]+, in [^\n]+$)',
                               lambda match: match.group(1)+marker+'/', value)
    return value


def compare_file(job):
    name, roots, repos = job
    values = []
    for root,repo in zip(roots,repos):
        with (gzip.open(root/name, 'rt') if name.endswith('.gz') else (root/name).open()) as source:
            values.append(normalize(json.load(source), root, repo))
    diffs = list(differences(*values))
    bad = [(p,a,b) for p,a,b in diffs if not host_observation(p,a,b)]
    result = dict(equal_simulation_values=not bad, host_measurement_differences=len(diffs)-len(bad))
    return name, result, [dict(path=list(p), baseline=a, candidate=b) for p,a,b in bad[:30]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, help='Independent report comparisons; output remains sorted')
    args = parser.parse_args()
    assert args.workers>0
    roots = [args.baseline.resolve(), args.candidate.resolve()]
    summaries = [json.loads((root/'summary.json').read_text()) for root in roots]
    assert all(s['complete'] for s in summaries), 'Both full scoped gates must finish before comparison'
    assert summaries[0]['required_phases']==summaries[1]['required_phases']
    assert summaries[0]['scope_exceptions']==summaries[1]['scope_exceptions']
    assert json.loads((roots[0]/'dependencies.json').read_text()) == json.loads((roots[1]/'dependencies.json').read_text()), 'External dependencies differ'
    repos = [Path(s['repo']) for s in summaries]
    names = {'physics.json', 'compute-oracle.json', 'comm-oracle.json', 'contracts.json', 'policies.json',
             'communication.json', 'supported-compute/summary.json', 'browser/browser-result.json',
             'distribution/audit.json', 'distribution/native_result.json',
             'distribution/native030.report.json', 'distribution/native030.audit.json'}
    if 'islsim-core' in summaries[0]['required_phases']:
        names.add('islsim-core.json')
    for folder in ('matrix', 'e2e'):
        left = {str(p.relative_to(roots[0])) for p in (roots[0]/folder).rglob('*.report.json.gz')}
        right = {str(p.relative_to(roots[1])) for p in (roots[1]/folder).rglob('*.report.json.gz')}
        assert left == right and left, (folder, left ^ right)
        assert len(left)==(30 if folder=='matrix' else 8), (folder, len(left))
        names.update(left)
    results, unexpected = {}, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(name,roots,repos) for name in sorted(names)]
        for name,result,bad in pool.map(compare_file,jobs):
            results[name] = result
            if bad:
                unexpected[name] = bad
            print(json.dumps(dict(file=name, equal_simulation_values=not bad)),flush=True)
    for folder in ('distribution/replay', 'browser-fixture/replay', 'browser'):
        suffixes = {'.json', '.html', '.js', '.css'} if folder.endswith('replay') else {'.png'}
        left = {p.relative_to(roots[0]/folder) for p in (roots[0]/folder).rglob('*') if p.suffix in suffixes}
        right = {p.relative_to(roots[1]/folder) for p in (roots[1]/folder).rglob('*') if p.suffix in suffixes}
        assert left==right and left
        for name in sorted(left):
            a,b = [(root/folder/name).read_bytes() for root in roots]
            if a != b:
                unexpected[folder+'/'+str(name)] = 'Export/browser evidence differs byte-for-byte'
    result = dict(complete=not unexpected, compared_files=results, unexpected=unexpected,
                  comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope_exceptions=summaries[0]['scope_exceptions'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(files=len(results), unexpected=list(unexpected))))
    assert not unexpected, str(args.output)


if __name__ == '__main__':
    main()
