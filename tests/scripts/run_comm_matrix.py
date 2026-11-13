"""Run original Comm algorithms on real retained device routes and timeline."""
import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys
from structure_test_support import run_suites, write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from astermind.engine.comm.policy import ALGORITHMS
    from astermind.engine.comm.operator import CommunicationOperator
    from astermind.engine.comm.planner import plan_communication
    from tests.unit.runtime.test_comm_port import CommPortTests
    from examples.native.compute_comm_matrix import scene
    participants = {
        'single_su': ('satA-gpu-0', 'satA-gpu-1'),
        'intra_cross_su': tuple(f'satA-gpu-{i}' for i in range(8)),
        'inter_satellite': ('satA-gpu-0', 'satB-gpu-0'),
    }
    rows = []
    for scope, protocol, basis in product(participants, ('payload_only', 'inter_reference'),
            ('raw_line', 'net_after_existing_fec', 'application_effective')):
        for operation, algorithms in ALGORITHMS.items():
            for algorithm in sorted(algorithms):
                if algorithm == 'hierarchical' and scope != 'intra_cross_su':
                    continue  # explicitly rejected by CellCommPolicy
                fixture = CommPortTests()
                try:
                    raw = scene('DS05')
                    raw['communication'].update(runtime_protocol=protocol, runtime_capacity_basis=basis,
                        cell_policies={'cellA': {'algorithms': {scope: {operation: algorithm}}}})
                    host, network, comm = fixture.runtime(raw)
                    ranks = participants[scope]
                    if operation == 'send_recv' and scope == 'intra_cross_su':
                        ranks = ('satA-gpu-0', 'satA-gpu-4')
                    options = {}
                    semantics = 'per_rank_input'
                    if operation in ('send_recv', 'broadcast'):
                        semantics = 'payload'
                    if operation in ('all_gatherv', 'reduce_scatterv'):
                        options['partition_bytes'] = (4096//len(ranks),)*len(ranks)
                        if operation == 'all_gatherv':
                            semantics = 'aggregate_input'
                    if operation in ('all_reduce', 'reduce_scatterv'):
                        options['reduction'] = 'sum'
                    task = CommunicationOperator(operation, algorithm, ranks, 4096, semantics,
                        cell_id='cellA', task_id='matrix-task', **options)
                    plan = plan_communication(comm.network, comm.policy, task, host.now)
                    assert plan.selection['scope'] == scope
                    assert plan.selection['algorithm'] == algorithm
                    done, failures = [], []
                    comm.submit(task, lambda event: done.append(asdict(event)), failures.append)
                    host.run()
                    assert not failures and len(done) == 1, (scope, operation, algorithm, failures)
                    assert not (comm.tasks or comm.active or comm.flows)
                    rows.append(dict(scope=scope, protocol=protocol, basis=basis, operation=operation,
                        algorithm=algorithm, selection=plan.selection, completions=done,
                        service=[row for row in comm.trace if row['kind']=='service']))
                finally:
                    fixture.doCleanups()
    suites = run_suites(['tests.unit.runtime.test_comm_port', 'tests.unit.runtime.test_comm_semantics',
                         'tests.unit.runtime.test_comm_failure_isolation',
                         'tests.unit.runtime.test_service_precision', 'tests.unit.runtime.test_unified_clock'])
    write_result(args.output, dict(cases=rows, suites=suites))


if __name__ == '__main__':
    main()
