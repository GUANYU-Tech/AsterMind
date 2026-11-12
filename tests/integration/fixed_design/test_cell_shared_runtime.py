"""Retained KV barriers and rollback on real Comm and the host integer clock."""
import unittest
from types import SimpleNamespace
from tests.unit.runtime import test_comm_port as fixtures
scenario=fixtures.scenario
from astermind.engine.cell_bridge import CellTransferExecutor
from astermind.engine.cell_migration import NetworkKVMigration
from astermind.engine.clock import Time
from astermind.engine.device_controls import apply_device_command

class StorageCell:
    storage_signature=('same-model','same-shards')
    _inflight=None
    def __init__(self,cell_id):self.cell_id,self.paused,self.calls=cell_id,False,[]
    def pause_for_migration(self,identifier):self.paused=True
    def resume_after_migration(self,identifier):self.paused=False
    def migration_memory_time(self,size,stage):return .1
    def storage_rpc(self,operation,*args):
        self.calls.append(operation)
        return {'accepted':True,'bytes':1_000_000}

class CellSharedRuntimeTests(unittest.TestCase):
    def fixture(self,horizon_ns=1_000_000_000,retry_limit=8):
        raw=scenario(horizon_ns=horizon_ns);raw['communication']['retry_limit']=retry_limit
        f=fixtures.CommPortTests();self.addCleanup(f.doCleanups)
        host,network,comm=f.runtime(raw)
        executor=CellTransferExecutor(host.timeline,network)
        migration=NetworkKVMigration(host.timeline,executor,{'A':{0:'ag',1:'ag'},'B':{0:'bg',1:'bg'}})
        self.addCleanup(migration.close)
        return host,network,comm,executor,migration

    def test_per_rank_migration_waits_for_all_shards_and_target_write(self):
        host,network,comm,executor,migration=self.fixture()
        source,target=StorageCell('A'),StorageCell('B')
        record=migration.start(SimpleNamespace(id='kv',mode='move'),source,target,{})
        host.run()
        self.assertEqual(record['state'],'completed')
        self.assertEqual(record['bytes'],2_000_000)
        self.assertEqual(len(record['partitions']),2)
        self.assertAlmostEqual(record['completed_time_s']-record['network_complete_time_s'],.1)
        self.assertIn('commit_import',target.calls)
        self.assertFalse(source.paused or target.paused)
        self.assertFalse(comm.tasks or comm.active or comm.flows)

    def test_fault_rolls_back_reservations_and_releases_both_cells(self):
        host,network,comm,executor,migration=self.fixture(retry_limit=0)
        source,target=StorageCell('A'),StorageCell('B')
        record=migration.start(SimpleNamespace(id='kv',mode='copy'),source,target,{})
        host.timeline.schedule(Time('.1001'),-3,'fault',lambda:apply_device_command(host,
            dict(action='device_link_fail',target_id='alocal')))
        host.run()
        self.assertEqual(record['state'],'failed')
        self.assertIn('rollback_import',target.calls)
        self.assertNotIn('commit_import',target.calls)
        self.assertFalse(source.paused or target.paused)

    def test_horizon_never_publishes_incomplete_target_kv(self):
        host,network,comm,executor,migration=self.fixture(horizon_ns=100_500_000)
        source,target=StorageCell('A'),StorageCell('B')
        record=migration.start(SimpleNamespace(id='kv',mode='copy'),source,target,{})
        host.run();migration.close()
        self.assertEqual(record['state'],'failed')
        self.assertNotIn('commit_import',target.calls)
        self.assertFalse(source.paused or target.paused)
        self.assertFalse(comm.tasks or comm.active or comm.flows)

    def test_terminal_callback_error_is_not_swallowed(self):
        host,network,comm,executor,migration=self.fixture()
        def failed_callback():raise RuntimeError('caller callback failure')
        executor.submit_transport('r','ag','bg',1000,'input','request',failed_callback)
        with self.assertRaisesRegex(RuntimeError,'caller callback failure'):host.run()
