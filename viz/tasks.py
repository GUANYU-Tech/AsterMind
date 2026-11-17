"""Inspect and control only CLI processes belonging to this local console."""
import json
import os
from pathlib import Path
import re
import signal
import time


def processes(root, run_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('invalid run ID')
    expected = Path(root) / 'local/results/network' / run_id
    matches = []
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args = entry.joinpath('cmdline').read_bytes().decode().strip('\0').split('\0')
            if not args or Path(args[0]).name not in {'python', 'python3', 'python3.12'}:
                continue
            option = '--artifact-dir' if '--artifact-dir' in args else '--out'
            if option not in args:
                continue
            output = Path(args[args.index(option)+1]).resolve()
            if output != expected.resolve():
                continue
            if not any(a in {str(Path(__file__).with_name('worker.py'))} for a in args):
                continue
            stat = entry.joinpath('stat').read_text().rsplit(')',1)[1].split()
            if stat[0] != 'Z':
                matches.append((int(entry.name), stat[19]))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return matches


def same_process(pid, stamp):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19] == stamp and fields[0] != 'Z'
    except OSError:
        return False


def stop(root, run_id):
    roots = processes(root, run_id)
    if not roots:
        raise ValueError('此任务当前没有可停止的计算进程，请刷新状态')
    targets = dict(roots)
    # Capture descendants before terminating their parent; never signal the console.
    tree = {}
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            fields = entry.joinpath('stat').read_text().rsplit(')',1)[1].split()
            tree[int(entry.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    while True:
        more = {pid: stamp for pid,(parent,stamp) in tree.items() if parent in targets and pid not in targets}
        if not more:
            break
        targets.update(more)
    folder = Path(root)/'local/results/network'/run_id
    (folder/'stop_requested.json').write_text(json.dumps({'requested_at':time.time(),'pids':list(targets)}))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid,stamp in targets.items():
            if same_process(pid,stamp):
                try:
                    os.kill(pid,sig)
                except ProcessLookupError:
                    pass
        deadline=time.monotonic()+2
        while any(same_process(pid,stamp) for pid,stamp in targets.items()) and time.monotonic()<deadline:
            time.sleep(.05)
    if any(same_process(pid,stamp) for pid,stamp in targets.items()):
        raise RuntimeError('停止请求已发出，进程尚未退出，请刷新状态')
    return {'id':run_id,'state':'cancelled','stopped_pids':list(targets)}


if __name__ == '__main__':
    import sys
    print(json.dumps(stop(__import__('paths').WORKSPACE,sys.argv[1])))
