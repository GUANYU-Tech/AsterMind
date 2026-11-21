"""Exact report causality rejects sub-nanosecond errors and audits failures."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from astermind.engine.audit import audit
from astermind.engine.clock import Time
from astermind.engine.timeline import GlobalTimeline
from astermind.common.bootstrap import activate


def result():
    first=dict(call_id='compute',occurrence=0,start_time_ticks='5',end_time_ticks='7',
        start_time_ns=1,end_time_ns=1,host_ready_time_ticks='5',launch_gap_ticks='0',
        communication=False,native_prediction_s=2e-18,dependencies=[])
    second=dict(call_id='comm',occurrence=0,start_time_ticks='8',end_time_ticks='9',
        start_time_ns=1,end_time_ns=1,host_ready_time_ticks='8',launch_gap_ticks='1',
        communication=True,dependencies=[['compute',0]],communication_estimate=dict(
            duration_ns=1,rounds=1,flow_count=1,cross_satellite_flows=1,cross_satellite_bytes=1))
    return dict(configuration={'requests':[dict(id='r',output_tokens=1)]},
        requests={'r':dict(status='completed',received_tokens=1,arrival_ns=0,cell_ready_ns=1,
            input_ready_ticks='4',cell_ready_ticks='5',compute_complete_ticks='9',completed_ticks='10',
            first_token_received_ns=1)},transfers=[],migrations={},cells={},
        request_state_events=[dict(request_id='r',previous='pending',status='completed',event_sequence=0,time_ticks='10')],
        event_log=[dict(sequence=0,time_ticks='10')],
        input_distributions=[dict(cell_id='c',pool_name='main',step_id='s',completed_ticks='5')],
        batches=[dict(cell_id='c',pool_name='main',step_id='s',start_time_ticks='5',end_time_ticks='9',
            native_communication_duration_charged_s=0,calls=[first,second])])

class CellTimeRecordTests(unittest.TestCase):
    def test_distribution_batch_and_communication_share_exact_tick_authority(self):
        self.assertTrue(audit(result())['passed'])

    def test_fast_mode_audits_compute_and_communication_without_kernel_records(self):
        data = result()
        data['configuration']['compute_mode'] = 'fast_mode'
        batch = data['batches'][0]
        batch.update(mode='fast_mode', calls=[], compute_s=3e-18, communication_s=1e-18,
                     communication_pricing='isolated')
        self.assertEqual(audit(data)['fast_mode_batches'], 1)
        legacy = deepcopy(data)
        legacy['configuration']['compute_mode'] = 'batch_roofline'
        legacy['batches'][0].update(mode='batch_roofline', communication_pricing='isolated_jt')
        self.assertEqual(audit(legacy)['fast_mode_batches'], 1)
        for update in (dict(end_time_ticks='10'), dict(communication_pricing='unknown'),
                       dict(compute_s=-1)):
            with self.subTest(update=update), self.assertRaises(AssertionError):
                audit({**data, 'batches': [{**batch, **update}]})

    def test_audit_rejects_one_tick_distribution_dependency_and_clock_errors(self):
        mutations=[lambda r:r['input_distributions'][0].update(completed_ticks='6'),
                   lambda r:r['batches'][0]['calls'][1].update(start_time_ticks='7'),
                   lambda r:r['batches'][0]['calls'][0].update(end_time_ticks='8'),
                   lambda r:r['request_state_events'][0].update(time_ticks='9'),
                   lambda r:r['requests']['r'].update(input_ready_ticks='6'),
                   lambda r:r['requests']['r'].pop('input_ready_ticks'),
                   lambda r:r['batches'][0].update(native_communication_duration_charged_s=1)]
        for mutate in mutations:
            data=result();mutate(data)
            with self.subTest(mutation=mutate),self.assertRaises(AssertionError):audit(data)

    def test_audit_requires_configured_parent_and_committed_kv_evidence(self):
        data=result()
        data['configuration']['requests'][0].update(session_id='s',context_id='child',parent_context_id='parent')
        data['configuration']['requests'].append(dict(id='p',session_id='s',context_id='parent',output_tokens=1))
        data['requests']['p']=deepcopy(data['requests']['r'])
        data['requests']['p']['compute_complete_ticks']='3'
        data['request_state_events'].append(dict(request_id='p',previous='pending',status='completed',event_sequence=0,time_ticks='10'))
        with self.assertRaises(AssertionError):audit(data)
        data['requests']['r']['parent_ready_ticks']='3'
        self.assertTrue(audit(data)['passed'])
        data['configuration']['migrations']=[dict(id='m',before_request='r')]
        data['migrations']['m']=dict(state='completed',completed_time_ticks='4',network_complete_time_ticks='4',bytes=1,
            partitions=[dict(transfer_id='t')])
        data['transfers']=[dict(id='t',kind='completed',time_ticks='4',size_bytes=1)]
        with self.assertRaises(AssertionError):audit(data)
        data['requests']['r']['kv_ready_ticks']='4'
        self.assertTrue(audit(data)['passed'])
        data['migrations']['m']['completed_time_ticks']='6'
        with self.assertRaises(AssertionError):audit(data)

    def test_failure_before_plan_has_complete_auditable_identity(self):
        activate()
        from astermind.engine.compute.batch_executor import RuntimeBatchExecutor
        loop=GlobalTimeline(start_s=Time.from_ticks(123));failures=[]
        executor=RuntimeBatchExecutor('c',loop,SimpleNamespace(network=None,tasks={}),())
        executor.start_batch('c','broken',object(),None,self.fail,token_count=1,on_failure=failures.append)
        record=executor.records[0]
        self.assertEqual((record['pool_name'],record['start_time_ticks'],record['end_time_ticks']),('main','123','123'))
        data=result();data['batches']=[record];data['input_distributions']=[]
        data['requests']['r']['status']='failed';data['request_state_events'][0]['status']='failed'
        self.assertEqual(audit(data)['failed_batches'],1)
