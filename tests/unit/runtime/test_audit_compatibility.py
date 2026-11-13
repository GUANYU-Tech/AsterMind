"""Mainline migration auditing stays independent of historical replay checks."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

from astermind.engine.audit import migration_barrier_valid
from astermind.engine.clock_audit import migration_barrier_valid as historical_barrier_valid
from tests.unit.runtime.test_cell_time_records import result


class AuditCompatibilityTests(unittest.TestCase):
    def check_barrier(self, migration, terminals, expected):
        for checker in (migration_barrier_valid, historical_barrier_valid):
            with self.subTest(migration=migration, terminals=terminals, checker=checker):
                self.assertEqual(checker(migration, terminals), expected)

    def test_exact_ticks_reject_early_commit_and_late_or_negative_delivery(self):
        migration = dict(network_complete_time_ticks='5', completed_time_ticks='6')
        terminal = dict(kind='completed', time_ticks='5')
        self.check_barrier(migration, [terminal], True)
        self.check_barrier(dict(migration, completed_time_ticks='4'), [terminal], False)
        self.check_barrier(migration, [dict(terminal, time_ticks='6')], False)
        self.check_barrier(migration, [dict(terminal, time_ticks='-1')], False)
        self.check_barrier(dict(migration, network_complete_time_ticks='-1'), [terminal], False)

    def test_legacy_nanoseconds_round_up_without_accepting_early_commit(self):
        migration = dict(network_complete_time_s=0.5e-9, completed_time_s=0.6e-9)
        terminal = dict(kind='completed', completion_time_ns=1)
        self.check_barrier(migration, [terminal], True)
        self.check_barrier(migration, [dict(terminal, completion_time_ns=2)], False)
        self.check_barrier(dict(migration, completed_time_s=0.4e-9), [terminal], False)

    def test_absent_or_failed_partition_never_satisfies_migration(self):
        migration = dict(network_complete_time_ticks='5', completed_time_ticks='6')
        completed = dict(kind='completed', time_ticks='5')
        for terminals in ([], [None], [dict(kind='failed', time_ticks='5')], [completed, None]):
            self.check_barrier(migration, terminals, False)

    def test_malformed_clock_evidence_returns_false(self):
        migration = dict(network_complete_time_ticks='5', completed_time_ticks='6')
        terminal = dict(kind='completed', time_ticks='5')
        for bad in ({}, dict(migration, completed_time_ticks=None),
                    dict(migration, network_complete_time_ticks='invalid'),
                    dict(migration, network_complete_time_ticks=float('inf'))):
            self.check_barrier(bad, [terminal], False)
        self.check_barrier(migration, [dict(kind='completed')], False)
        self.check_barrier(dict(network_complete_time_s=float('nan'), completed_time_s=1),
                           [dict(kind='completed', completion_time_ns=1)], False)

    def test_mainline_audits_completed_migration_without_historical_module(self):
        report = result()
        report['migrations']['m'] = dict(state='completed', network_complete_time_ticks='4',
            completed_time_ticks='4', bytes=1, partitions=[dict(transfer_id='t')])
        report['transfers'] = [dict(id='t', kind='completed', time_ticks='4', size_bytes=1)]
        script = '''
import importlib.abc
import json
import sys
class RejectHistorical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'astermind.engine.clock_audit':
            raise AssertionError('mainline audit imported historical checker')
sys.meta_path.insert(0, RejectHistorical())
from astermind.engine.audit import audit
report = json.load(sys.stdin)
assert any(m['state'] == 'completed' for m in report['migrations'].values())
assert audit(report)['passed']
assert 'astermind.engine.clock_audit' not in sys.modules
'''
        completed = subprocess.run([sys.executable, '-B', '-c', script],
            cwd=Path(__file__).resolve().parents[3], input=json.dumps(report),
            text=True, capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
