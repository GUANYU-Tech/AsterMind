"""Serve the actual exported replay and run the repository's browser assertions."""
import argparse
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import subprocess
import sys
import threading


def interaction_fixture(repo, output):
    """UI contract fixture, explicitly separate from real simulation evidence.

    Extend the existing recorded-contract fixture with a Scheduler-shaped step
    and one dependent tail call. Slower display times let real playback/pause
    and shorter-scene switching be exercised on the unchanged frontend.
    """
    sys.path.insert(0, str(repo))
    from tests.unit.runtime.test_native_replay import report
    from viz.export import export_reports, audit_checks
    from astermind.common.bootstrap import activate
    activate()
    from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
    output.mkdir(parents=True, exist_ok=True)
    original = report()
    def scale(value, key=''):
        if isinstance(value, dict):
            return {k:scale(v,k) for k,v in value.items()}
        if isinstance(value, list):
            return [scale(v) for v in value]
        if type(value) in (int,float) and key.endswith(('_ns','_s')):
            return value*1_000_000
        return value
    extended = {k:(deepcopy(v) if k=='configuration' else scale(v,k)) for k,v in original.items()}
    # Preserve the fixture's sub-nanosecond notification rounding, rather than
    # stretching that fixed clock allowance with the illustrative durations.
    extended['network']['flows'][0]['completion_time_ns'] = extended['transfers'][0]['completion_time_ns'] - .8
    batch = extended['batches'][0]
    tail = deepcopy(batch['calls'][0])
    tail.update(call_id='tail', dependencies=[['pp', -1]], start_time_ns=10_000_000,
                end_time_ns=15_000_000, native_prediction_s=.005)
    batch['calls'].append(tail)
    batch['end_time_ns'] = 15_000_000
    extended['requests']['r0']['compute_complete_ns'] = 15_000_000
    extended['produced_outputs'][0]['produced_time_s'] = .015
    tokens = extended['configuration']['requests'][0]['input_tokens']
    schedule = ScheduleStep('main:0', 'main', (ScheduledRequest('r0','prefill',tokens,0),))
    extended['cells']['cellA']['steps'] = [dict(start_time_s=.002, end_time_s=.015,
        step=dict(schedule=asdict(schedule), kv_cache_events=[], generated_tokens={'r0':1}))]
    checks = audit_checks(extended)
    assert all(row['state'] in ('pass','warn') for row in checks), [r for r in checks if r['state']=='fail']
    reports = []
    for name, value in (('historical', original), ('interaction-contract', extended)):
        path = output/(name+'.report.json')
        path.write_text(json.dumps(value))
        reports.append(path)
    replay = output/'replay'
    export_reports(reports, replay)
    (output/'scope.json').write_text(json.dumps(dict(scope='historical_ui_contract',
        real_simulation_evidence=False, source='tests.unit.runtime.test_native_replay.report',
        additions=['scaled display time', 'Scheduler-shaped step', 'dependent tail call']))+'\n')
    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--node', default='node')
    parser.add_argument('--script', type=Path, help='Same reviewed browser harness for both source revisions')
    args = parser.parse_args()
    assert (args.replay/'catalog.json').is_file()
    replay = interaction_fixture(args.repo, args.output.parent/'browser-fixture')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(replay)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        subprocess.run([args.node, str(args.script or args.repo/'tests/browser/native_replay.cjs'),
            f'http://127.0.0.1:{server.server_port}/', str(args.output)], check=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
