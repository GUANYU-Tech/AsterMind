"""Run independent existing native cases in isolated Python processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--intra', default='surrogate')
    parser.add_argument('--e2e', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assert args.workers > 0
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.e2e:
        for mode in ('midlayer', 'roofline'):
            jobs.append((mode, [args.repo/'tests/scripts/run_compute_comm_e2e.py', '--modes', mode]))
    else:
        assert args.cases
        for case in args.cases:
            for mode in ('midlayer', 'roofline'):
                jobs.append((f'{case}-{mode}-{args.intra}', [args.repo/'tests/scripts/run_compute_comm.py',
                    '--cases', case, '--modes', mode, '--intra', args.intra]))
    def run(job):
        name, arguments = job
        command = [sys.executable, '-B', *map(str, arguments), '--output', str(args.output)]
        path = args.output/(name+'.execution.log')
        with path.open('w') as log:
            result = subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            pattern = f'*-{name}*.report.json' if args.e2e else name+'.report.json'
            for report in args.output.glob(pattern):
                # WideEP evidence is large. Compression changes no report bytes.
                target = report.with_suffix(report.suffix+'.gz')
                before = hashlib.file_digest(report.open('rb'), 'sha256').hexdigest()
                with report.open('rb') as source, gzip.open(target, 'wb', compresslevel=1) as archive:
                    shutil.copyfileobj(source, archive)
                with gzip.open(target, 'rb') as source:
                    assert hashlib.file_digest(source, 'sha256').hexdigest() == before
                report.unlink()
        print(json.dumps(dict(case=name, returncode=result.returncode)), flush=True)
        if result.returncode:
            print(path.read_text()[-4000:], flush=True)
        return result.returncode
    with ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(run, jobs))
    if any(results):
        raise SystemExit('One or more original native cases failed')


if __name__ == '__main__':
    main()
