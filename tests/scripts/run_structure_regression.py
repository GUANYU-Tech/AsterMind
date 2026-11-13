"""Run the real branch suites; each batch has its own complete evidence directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from structure_dependencies import hash_tree, python_identity


def fingerprint(repo):
    """Include all non-generated tracked inputs, plus new untracked test/code files."""
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=repo,
    ).decode().split('\0')
    digest = hashlib.sha256()
    for name in sorted(set(filter(None, names))):
        path = repo / name
        if not path.is_file():
            continue
        digest.update(name.encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python031', type=Path, default=Path(sys.executable))
    parser.add_argument('--python011', type=Path, required=True)
    parser.add_argument('--vllm030-source', type=Path, required=True)
    parser.add_argument('--islsim-root', type=Path)
    parser.add_argument('--islsim-config', type=Path)
    parser.add_argument('--islsim-core-only', action='store_true',
                        help='Explicit accepted scope exception: core tested, external adapter unverified')
    parser.add_argument('--node-path', type=Path)
    parser.add_argument('--browser-path', type=Path)
    parser.add_argument('--browser-libraries', type=Path)
    parser.add_argument('--node', type=Path, default=Path('/usr/bin/node'))
    parser.add_argument('--browser-script', type=Path)
    parser.add_argument('--only', nargs='+', help='Development subset; never labels incomplete evidence as full acceptance')
    parser.add_argument('--resume', action='store_true', help='Reuse only identical successful phase inputs in this directory')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.json').write_text(json.dumps(dict(complete=False, status='preparing'))+'\n')
    harness = Path(__file__).resolve().parent
    # Resolving a venv's Python symlink bypasses its site-packages.
    python = str(args.python031.absolute())
    pd_python = str(args.python011.absolute())
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', PYTHONPATH=str(repo),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        ORBITServe_VLLM030_PYTHON=python, ORBITServe_VLLM030_SOURCE=str(args.vllm030_source.resolve()))
    for name in ('ORBITServe_MIDLAYER_ROOT', 'ORBITSERVE_MIDLAYER_ROOT'):
        env.pop(name, None)
    scope_exceptions = ([dict(suite='external_islsim_adapter', status='unverified_missing_dependency',
        reason='Original validator.py absent; user accepted core-only test boundary')]
        if args.islsim_core_only else [])
    code_identity = fingerprint(repo)
    steps = []

    def add(label, arguments, interpreter=python):
        steps.append((label, [interpreter, '-B', *map(str, arguments)]))

    def existing(name, *arguments):
        return [repo/'tests/scripts'/name, *arguments]

    def probe(label, name):
        add(label, existing(name, '--repo', repo, '--output', output/(label+'.json')))

    add('build', ['-m', 'astermind.common.build_native'])
    add('source', existing('check_port_sources.py'))
    add('retained', existing('run_retained_tests.py'))
    add('request-import', ['-m', 'unittest', '-v', 'tests.external.native_cells.test_request_batch'])
    for label, script in (('physics', 'probe_retained_physics.py'), ('compute-oracle', 'probe_compute_source.py'),
                          ('comm-oracle', 'probe_comm_source.py')):
        probe(label, script)
    for label, script in (('contracts', 'probe_structure_contracts.py'), ('policies', 'run_policy_matrix.py'),
                          ('communication', 'run_comm_matrix.py')):
        add(label, [harness/script, '--repo', repo, '--output', output/(label+'.json')])
    add('supported-compute', [harness/'run_supported_compute.py', '--repo', repo,
                             '--output', output/'supported-compute', '--python011', pd_python,
                             '--expectations', harness/'compute_support_rejections.json'])
    for label, cases, intra, interpreter in (
        ('native', ['DS01','DS02','DS03','DS04','DS05','DS06-copy','DS06-move'], 'surrogate', python),
        ('native-intra-roofline', ['DS01','DS02'], 'IntraSatelliteRoofline', python),
        ('wide-ep', ['DS07-pp1','DS07-pp2','DS07-idle'], 'surrogate', python),
        ('pd', ['DS08-one-token','DS08','DS08-kv-failure'], 'surrogate', pd_python),
    ):
        add(label, [harness/'run_structure_native.py', '--repo', repo, '--cases', *cases,
                    '--intra', intra, '--output', output/'matrix'], interpreter)
    add('e2e', [harness/'run_structure_native.py', '--repo', repo, '--e2e', '--output', output/'e2e'])
    if args.islsim_core_only:
        if not args.islsim_root:
            parser.error('--islsim-core-only requires --islsim-root')
        add('islsim-core', [harness/'probe_islsim_core.py', '--root', args.islsim_root,
                            '--output', output/'islsim-core.json'])
    else:
        add('islsim', ['-m','unittest','discover','-s','tests/external/islsim','-v'])
    add('distribution', existing('check_distribution.py', '--work-dir', output/'distribution'))
    add('browser', [harness/'run_structure_browser.py', '--repo', repo,
                    '--replay', output/'distribution/replay', '--output', output/'browser', '--node', args.node,
                    '--script', args.browser_script or repo/'tests/browser/native_replay.cjs'])
    known = {label for label, _ in steps}
    if args.only and set(args.only)-known:
        parser.error('unknown phases: '+str(set(args.only)-known))
    environment = {key: env.get(key) for key in (
        'ORBITServe_VLLM030_PYTHON','ORBITServe_VLLM030_SOURCE','ORBITSERVE_ISLSIM_ROOT',
        'ORBITSERVE_ISLSIM_CONFIG','NODE_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    phase_envs = {label: dict(env) for label in known}
    if 'islsim' in known:
        if args.islsim_root: phase_envs['islsim']['ORBITSERVE_ISLSIM_ROOT'] = str(args.islsim_root.resolve())
        if args.islsim_config: phase_envs['islsim']['ORBITSERVE_ISLSIM_CONFIG'] = str(args.islsim_config.resolve())
    if args.node_path: phase_envs['browser']['NODE_PATH'] = str(args.node_path.resolve())
    if args.browser_path: phase_envs['browser']['PLAYWRIGHT_BROWSERS_PATH'] = str(args.browser_path.resolve())
    if args.browser_libraries: phase_envs['browser']['LD_LIBRARY_PATH'] = str(args.browser_libraries.resolve())
    dependencies = dict(python031=python_identity(python), python011=python_identity(pd_python),
        vllm030_source_sha256=hash_tree(args.vllm030_source),
        islsim_sha256=hash_tree(args.islsim_root) if args.islsim_root else None,
        islsim_config_sha256=hashlib.sha256(args.islsim_config.read_bytes()).hexdigest() if args.islsim_config else None,
        node_sha256=hashlib.sha256(args.node.read_bytes()).hexdigest(),
        browser_script_sha256=hashlib.sha256((args.browser_script or repo/'tests/browser/native_replay.cjs').read_bytes()).hexdigest(),
        browser_libraries_sha256=hash_tree(args.browser_libraries, suffixes=('.0', '.1', '.2', '.so')) if args.browser_libraries else None,
        playwright_lock_sha256=hashlib.sha256((args.node_path.parent/'package-lock.json').read_bytes()).hexdigest() if args.node_path else None)
    (output/'dependencies.json').write_text(json.dumps(dependencies, sort_keys=True, indent=2)+'\n')
    result_file = output/'steps.json'
    results = json.loads(result_file.read_text()) if args.resume and result_file.exists() else {}
    helpers = {
        'contracts': ('structure_test_support.py',), 'policies': ('structure_test_support.py',),
        'communication': ('structure_test_support.py',),
        'supported-compute': ('compute_support_rejections.json',),
    }
    def phase_signature(label, command):
        script_hash = hashlib.sha256(Path(command[2]).read_bytes()).hexdigest() if Path(command[2]).is_file() else None
        phase_environment = {key: phase_envs[label].get(key) for key in (
            *environment, 'PLAYWRIGHT_BROWSERS_PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONHASHSEED')}
        helper_identity = {name: hashlib.sha256((harness/name).read_bytes()).hexdigest()
                           for name in helpers.get(label, ())}
        return hashlib.sha256(json.dumps(dict(command=command, source=code_identity,
            script=script_hash, helpers=helper_identity, environment=phase_environment,
            dependencies=dependencies), sort_keys=True).encode()).hexdigest()
    signatures = {label: phase_signature(label, command) for label,command in steps}
    def summarize(status):
        passed = sorted(label for label in known if results.get(label, {}).get('returncode') == 0
            and results[label].get('signature') == signatures[label])
        summary = dict(repo=str(repo), source_sha256=code_identity, complete=status=='finished' and len(passed)==len(known),
            status=status, required_phases=sorted(known), passed_phases=passed,
            scope_exceptions=scope_exceptions, environment=environment)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
    summarize('running')
    for label, command in steps:
        if args.only and label not in args.only:
            continue
        prior = results.get(label, {})
        if args.resume and prior.get('returncode') == 0 and prior.get('signature') == signatures[label]:
            print(json.dumps(dict(phase=label, reused_identical_success=True)), flush=True)
            continue
        with (output/(label+'.log')).open('w') as log:
            completed = subprocess.run(command, cwd=repo, env=phase_envs[label], stdout=log, stderr=subprocess.STDOUT)
        results[label] = dict(command=command, returncode=completed.returncode, signature=signatures[label])
        result_file.write_text(json.dumps(results, indent=2)+'\n')
        print(json.dumps(dict(phase=label, returncode=completed.returncode)), flush=True)
        if completed.returncode:
            summarize('failed:'+label)
            print((output/(label+'.log')).read_text()[-6000:], flush=True)
            raise SystemExit(completed.returncode)
    assert fingerprint(repo) == code_identity, 'Source changed during test run'
    # A changed harness invalidates only phases that actually consume it; never
    # certify evidence against a script that was changed while the phase ran.
    signatures = {label: phase_signature(label, command) for label,command in steps}
    summary = summarize('finished')
    print(json.dumps(summary), flush=True)
    if not args.only and not summary['complete']:
        raise SystemExit('Incomplete full acceptance')


if __name__ == '__main__':
    main()
