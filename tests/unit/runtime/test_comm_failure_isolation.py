"""Failure seam checks with real KernelPlan, executor, Comm and Timeline.

Compute durations and Scheduler control-plane doubles are explicit fixtures;
these tests are not model inference or hardware calibration.
"""
import unittest
from types import SimpleNamespace as NS

from astermind.common.bootstrap import activate
activate()
from astermind.engine.compute.batch_executor import RuntimeBatchExecutor
from astermind.engine.compute.executor_base import NetworkBatchExecutor
from astermind.engine.compute.placement import RankPlacement
from astermind.engine.compute.serving import ServingScheduler
from astermind.engine.cell_events import CellRequest
from astermind.engine.timeline import GlobalTimeline
from astermind.engine.comm.event_runtime import RuntimeCommunication
from astermind.engine.comm.frame_surrogate import CommPolicy
from astermind.engine.comm.operator import CommunicationOperator
from tests.comm_fixture import Frames
from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig
from src.serving.kernel_lowering.kernel_plan import KernelCall, KernelPlan


def communication(name, source, target, deps=()):
    transfer = PhysicalTransferConfig(operation='send_recv', algorithm='send_recv',
        bytes=1_000_000, bytes_semantics='payload', direction='point_to_point',
        link='nvlink', topology='point_to_point', participants=(source, target),
        source_rank=source, destination_rank=target, resource_id=name,
        backend='analytical_roofline_v1')
    return KernelCall(name, KernelSpec(KernelKind.COMMUNICATION, 'nvidia_h200', transfer=transfer),
        'main', source, name, (name,), dependencies=deps)


def executor_fixture(loop, comm, cell='A'):
    placements = tuple(RankPlacement(cell, 'main', i, 'sa' if i % 2 == 0 else 'sb', f'g{i}')
                       for i in range(4))
    executor = RuntimeBatchExecutor(cell, loop, comm, placements)
    pool = NS(hardware_id='nvidia_h200',
        parallel_plan=NS(config=NS(data_parallel_size=1)),
        physical_ranks_for_pipeline_stage=lambda stage: (0,))
    execution = NS(host_runtime_prepare_profiles=None, host_launch_profiles=None,
        deployment=NS(worker_pools={'main': pool}), predict_local_plan=lambda plan: {})
    return executor, execution


class FailureIsolationTests(unittest.TestCase):
    def test_fast_independent_rates_keep_paths_and_ignore_overlapping_transfers(self):
        def replay(sizes, shared):
            loop = GlobalTimeline()
            comm = RuntimeCommunication(Frames(), loop, CommPolicy(execution='runtime'),
                                        shared_bandwidth=shared)
            completed = {}
            for index, size in enumerate(sizes):
                identity = str(index)
                comm.submit_transfer(identity, 'g0', 'g1', size,
                                     lambda key=identity: completed.update({key: loop.now_s.ticks}))
            loop.run()
            return completed, comm

        alone, _ = replay((1_000_000,), False)
        overlap, fast = replay((1_000_000, 2_000_000), False)
        shared, detailed = replay((1_000_000, 2_000_000), True)
        self.assertEqual(alone['0'], overlap['0'])
        self.assertGreater(shared['0'], overlap['0'])
        self.assertEqual(fast.delivery_paths, detailed.delivery_paths)
        self.assertEqual(fast.stats['rate_allocations'], 0)
        self.assertGreater(detailed.stats['rate_allocations'], 0)
        self.assertTrue(any(row['kind'] == 'admit' for row in fast.trace))

    def test_failed_batch_cancels_siblings_and_never_releases_successor(self):
        loop = GlobalTimeline()
        comm = RuntimeCommunication(Frames(), loop, CommPolicy(execution='runtime',
            inter_parameters=dict(startup_us=0,hop_processing_us=0,routing_processing_us=0,
                                  fec=dict(enabled=False))))
        executor, execution = executor_fixture(loop, comm)
        # Explicit mathematical compute fixture; duration comes from the test
        # predictor below, not a claimed calibrated native GEMM launch.
        compute = KernelCall('compute_after',KernelSpec(KernelKind.GEMM,'nvidia_h200',
            shape=dict(b=1,m=1,n=1,k=1),dtype='bf16',variant='gemm',
            execution_support='unsupported',unsupported_reason='launch_selector_unavailable'),
            'main',0,'compute',('source',),('first',))
        plan = KernelPlan((communication('first', 0, 1), communication('sibling', 2, 3),
                           communication('after', 0, 1, ('first',)),compute))
        execution.predict_local_plan=lambda plan:dict(compute_after=NS(t_total_s=1e-6))
        failures, complete, other = [], [], []
        executor.start_batch('A', 's0', plan, execution, complete.append,
            token_count=1, on_failure=failures.append)
        comm.submit(CommunicationOperator('send_recv', 'send_recv', ('g2', 'g3'), 2_000_000, 'payload'),
                    lambda result: other.append(loop.now_s), task_id='B/independent')
        def fault():
            self.assertEqual(len(comm.active),3)
            identity=next(key for key in comm.tasks if '/first/' in key)
            comm.cancel(identity)
        loop.schedule(1e-7, 0, 'fault', fault)
        loop.run()
        self.assertEqual(len(failures), 1)
        self.assertFalse(complete)
        self.assertEqual(len(other), 1)
        self.assertFalse(comm.tasks)
        self.assertFalse(comm.flows)
        self.assertFalse(executor._active)
        self.assertFalse(any('/after/' in row.get('id','') for row in comm.trace))
        self.assertEqual(executor.records[-1]['status'], 'failed')
        row=next(row for row in executor.records[-1]['calls'] if row['call_id']=='compute_after')
        self.assertIsNone(row['start_time_ns'])
        self.assertGreater(other[0],1e-7)

    def test_input_broadcast_failure_does_not_start_plan(self):
        loop=GlobalTimeline()
        comm=RuntimeCommunication(Frames(),loop,CommPolicy(execution='runtime'))
        executor,execution=executor_fixture(loop,comm)
        execution.deployment.worker_pools['main'].physical_ranks_for_pipeline_stage=lambda stage:(0,1)
        failed=[]
        executor.start_batch('A','input',KernelPlan((communication('never',0,1),)),execution,
            lambda at:self.fail('failed input completed a batch'),token_count=1,on_failure=failed.append)
        loop.schedule(1e-9,0,'fault',lambda:comm.cancel('A/main/input/input/0'))
        loop.run()
        self.assertEqual(len(failed),1)
        self.assertFalse(comm.tasks)
        self.assertFalse(any('/never/' in row.get('id','') for row in comm.trace))

    def test_shared_task_adapter_has_no_static_execution_entry(self):
        self.assertFalse(hasattr(NetworkBatchExecutor,'start_batch'))
        self.assertFalse(hasattr(NetworkBatchExecutor,'_time_plan'))

    def test_runtime_error_without_failure_consumer_is_not_silenced(self):
        loop=GlobalTimeline()
        executor,execution=executor_fixture(loop,NS(network=None))
        execution.predict_local_plan=lambda plan: (_ for _ in ()).throw(ValueError('predictor failed'))
        with self.assertRaisesRegex(RuntimeError,'predictor failed'):
            executor.start_batch('A','s',KernelPlan((communication('send',0,1),)),execution,
                                 lambda at:None,token_count=1)


    def test_scheduler_failure_stops_only_its_pool_and_reports_waiting_requests(self):
        loop = GlobalTimeline()
        closed, failures = [], []
        backend = NS(config=NS(scheduler_pipeline_depth=1, pool_name='main'), close=lambda: closed.append(True))
        scheduler = ServingScheduler('A', loop, backend, NS(), lambda result: None,
            max_model_len=128, on_failure=failures.append, batch_executor=NS(loop=loop))
        scheduler._record_load = lambda: None
        for rid in ('inflight', 'waiting'):
            scheduler._requests[rid] = CellRequest(rid, 0, 4, 2)
        scheduler._waiting.add('waiting')
        scheduler._inflight = (NS(schedule=NS(step_id='s0')), 0)
        scheduler._fail_step(dict(reason='communication failed', task_id='A/main/s0/first/0'))
        loop.run()
        self.assertEqual(len(closed), 1)
        self.assertEqual({f['request_id'] for f in failures}, {'inflight', 'waiting'})
        self.assertTrue(all(f['failure_scope'] == 'pool_fail_stop' for f in failures))
        self.assertIsNone(scheduler._inflight)
        self.assertFalse(scheduler._waiting)
        scheduler.submit_ready_request(CellRequest('later', loop.now_s, 4, 2))
        loop.run()
        self.assertEqual(failures[-1]['request_id'], 'later')


if __name__ == '__main__':
    unittest.main()
