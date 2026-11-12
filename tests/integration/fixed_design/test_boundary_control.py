"""Boundary interventions execute on the real engine across transport backends."""

import importlib.util
import unittest
from unittest.mock import patch
from dataclasses import replace
from types import SimpleNamespace

from astermind.engine import NetworkState
from astermind.engine.events import _EventEngine
from astermind.engine.setup.faults import realize_runtime_faults, validate_random_faults
from astermind.engine.state.interventions import apply_topology_command
from tests.unit.runtime.test_dynamic_network import network


def scenario(mode='queue', faults=(), edges=None, setup=0, terminals=4):
    base = network(edges=edges or (('a','b',0.,240.),('b','c',0.,240.),('a','d',0.,240.),('d','c',0.,240.)),
                   bandwidth=8000., setup=setup, terminals=terminals)
    trace = base.trace
    trace.time = dict(start_s=0., end_s=240., step_s=60.)
    trace.ephemeris_samples = tuple(dict(row, time_s=t) for t in (0.,60.,120.,180.,240.)
                                   for row in trace.ephemeris_samples if row['time_s']==0.)
    nodes = tuple(replace(n, role='communication', shell_id='s') for n in base.spec.nodes)
    net = NetworkState(trace, replace(base.spec, nodes=nodes, faults=faults, fault_timing='boundary',
                                     failure_policy='retry_next_sample', transport_model=mode))
    spec = SimpleNamespace(serving_profile=SimpleNamespace(internal_comm_accounting='embedded'),
                           time_horizon_s=240., routing_policy='earliest_arrival')
    engine = _EventEngine(spec, SimpleNamespace(network=net, time=trace.time))
    engine.debit = lambda *args, **kwargs: None
    engine.on_delivered = lambda job, unused: job.update(done=True)
    engine.fail = lambda job, reason: job.update(done=True, failure=reason)
    return engine


def job(engine, identity):
    value = dict(request=dict(request_id=identity), path=[], ledger=[], done=False)
    engine.jobs[identity] = value
    return value


class BoundaryControlTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('simgrid'), 'requires native SimGrid')
    def test_native_extension_keeps_zero_capacity_work_suspended(self):
        engine=scenario('simgrid')
        transport=engine.transport
        leg=dict(link_id='communication_isl:a:b',source_node_id='a',target_node_id='b')
        key=transport.resource(leg)
        try:
            transport.configure({key:8000.})
            transport.start('suspended',leg,8000.)
            self.assertAlmostEqual(transport.advance_until(1.)['bytes_sent']['suspended'],1000.,places=5)
            transport.configure({key:0.})
            self.assertEqual(transport.advance_until(2.)['bytes_sent']['suspended'],0.)
            transport.register_edge(dict(link_id='communication_isl:a:c',endpoint_a='a',endpoint_b='c',kind='communication_isl'))
            self.assertEqual(transport.advance_until(3.)['bytes_sent']['suspended'],0.)
            transport.configure({key:8000.})
            self.assertAlmostEqual(transport.advance_until(4.)['bytes_sent']['suspended'],1000.,places=5)
        finally:
            transport.close()

    def test_malformed_commands_do_not_mutate_runtime(self):
        engine=scenario()
        for command in ({'action':[]},{'action':'node_fail','target_id':[]},
                        {'action':'connect','endpoint_a':[],'endpoint_b':'b'}):
            with self.subTest(command=command),self.assertRaises(ValueError):
                apply_topology_command(engine,command)
        self.assertEqual(engine.net.revision,0)
        self.assertFalse(engine.net.active_faults)

    def test_boundary_fault_retries_at_same_boundary_for_all_available_backends(self):
        for mode, module in [('queue',None),('fluid',None),('simgrid','simgrid'),('ns3','ns')]:
            if module and not importlib.util.find_spec(module):
                continue
            with self.subTest(mode=mode):
                fault = dict(target_type='link', target_id='communication_isl:a:b', start_s=60., end_s=180.)
                engine = scenario(mode, (fault,))
                request = job(engine, 'r')
                engine.send(request,'a','c',80000.,None,'input','delivered')
                engine.push(60.,-2,'fault_start',None,(0,fault))
                engine.push(180.,-2,'fault_end',None,(0,fault))
                engine.run()
                recovery = next(e for e in engine.log if e['kind']=='route_recovery')
                self.assertEqual(recovery['time_s'],60.)
                self.assertNotIn('failure',request)
                self.assertTrue(any(p['target_node_id']=='d' for p in request['path']))

    def test_nonboundary_fault_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '时间片边界'):
            scenario(faults=(dict(target_type='node', target_id='b', start_s=30., end_s=120.),))

    def test_disconnect_persists_and_connect_really_restores_backend_graph(self):
        engine = scenario()
        apply_topology_command(engine,dict(action='disconnect',target_id='communication_isl:a:b'))
        for t in (0.,60.,120.):
            self.assertEqual(engine.net.link_at('communication_isl:a:b',t)['state'],'DOWN')
        result = apply_topology_command(engine,dict(action='connect',endpoint_a='a',endpoint_b='b'))
        self.assertEqual(result['message'],'连接成功')
        self.assertEqual(engine.net.link_at(result['target_id'],0.)['state'],'UP')
        self.assertFalse(engine.net.manual_disabled_links)

    def test_recovery_clears_only_manual_cause_and_failed_connect_is_atomic(self):
        engine = scenario()
        natural=dict(target_type='node',target_id='a',start_s=0.,end_s=120.,source='random')
        engine.on_fault_start(None,(0,natural))
        apply_topology_command(engine,dict(action='node_fail',target_id='a'))
        apply_topology_command(engine,dict(action='node_restore',target_id='a'))
        self.assertEqual(engine.net.snapshot(0.)['nodes'][0]['state'],'DOWN')
        revision = engine.net.revision
        with self.assertRaises(ValueError):
            apply_topology_command(engine,dict(action='connect',endpoint_a='a',endpoint_b='c'))
        self.assertEqual(engine.net.revision,revision)
        self.assertNotIn('communication_isl:a:c',engine.net.edge_by_id)

    def test_new_link_rejects_earth_occlusion_and_terminal_exhaustion(self):
        engine=scenario(edges=(('a','b',0.,240.),('b','c',0.,240.)),terminals=1)
        with self.assertRaisesRegex(ValueError,'终端'):
            apply_topology_command(engine,dict(action='connect',endpoint_a='a',endpoint_b='c'))
        self.assertNotIn('communication_isl:a:c',engine.net.edge_by_id)
        engine=scenario()
        engine.net.samples['c']=[(t,(-7000.,0.,0.)) for t,_ in engine.net.samples['c']]
        with self.assertRaisesRegex(ValueError,'不可见'):
            apply_topology_command(engine,dict(action='connect',endpoint_a='a',endpoint_b='c'))

    def test_new_native_link_preserves_active_transfer_progress(self):
        for mode,module in [('queue',None),('fluid',None),('simgrid','simgrid'),('ns3','ns')]:
            if module and not importlib.util.find_spec(module):
                continue
            with self.subTest(mode=mode):
                engine=scenario(mode)
                first=job(engine,'existing')
                engine.send(first,'a','b',100000.,None,'input','delivered')
                result=[]
                def connect(unused,payload):
                    result.append(apply_topology_command(engine,dict(action='connect',endpoint_a='a',endpoint_b='c')))
                    second=job(engine,'new')
                    engine.send(second,'a','c',1000.,None,'input','delivered')
                engine.on_test_connect=connect
                engine.push(60.,5,'test_connect')
                engine.run()
                self.assertEqual(result[0]['message'],'连接成功')
                self.assertNotIn('failure',first)
                self.assertNotIn('failure',engine.jobs['new'])
                sent=sum(s['bytes'] for p in first['path'] for s in p['segments'])
                self.assertAlmostEqual(sent,100000.,places=3)

    def test_runtime_random_models_reproducible_and_boundary_aligned(self):
        engine=scenario()
        for model in ('fixed_ratio','per_step'):
            settings=dict(enabled=True,model=model,target_type='node',seed=4,ratio=.5,
                          failure_probability=.5,duration_steps=2,start_step=0,end_step=3)
            spec=replace(engine.net.spec,random_faults=settings)
            a=realize_runtime_faults(spec,engine.net.environment_times,engine.net.edges)
            b=realize_runtime_faults(spec,engine.net.environment_times,engine.net.edges)
            self.assertEqual(a.faults,b.faults)
            self.assertTrue(a.faults)
            self.assertTrue(all(f['source']=='random' and f['start_s'] in engine.net.environment_times
                                and f['end_s'] in engine.net.environment_times for f in a.faults))

    def test_random_duration_repairs_after_injection_window_without_truncation(self):
        engine=scenario()
        settings=dict(enabled=True,model='per_step',target_type='node',seed=4,
                      failure_probability=1.,duration_steps=2,start_step=1,end_step=2)
        realized=realize_runtime_faults(replace(engine.net.spec,random_faults=settings),
                                        engine.net.environment_times,engine.net.edges)
        self.assertEqual(len(realized.faults),4)
        for fault in realized.faults:
            self.assertEqual((fault['start_s'],fault['end_s']),(60.,180.))
        engine.now=60.
        engine.on_fault_batch(None,tuple((i,f,False) for i,f in enumerate(realized.faults)))
        event=next(e for e in engine.log if e['kind']=='fault_injected')
        self.assertEqual([event[k] for k in ('start_step','end_step','duration_steps','duration_s')],
                         [1,3,2,120.])

    def test_random_zero_probability_and_recovery_boundary_exclusion(self):
        engine=scenario()
        settings=dict(enabled=True,model='per_step',target_type='node',seed=4,
                      failure_probability=0.,duration_steps=1,start_step=0,end_step=4)
        def realize(probability):
            return realize_runtime_faults(replace(engine.net.spec,random_faults={**settings,'failure_probability':probability}),
                                          engine.net.environment_times,engine.net.edges).faults
        self.assertFalse(realize(0.))
        intervals={}
        for fault in realize(1.):
            intervals.setdefault(fault['target_id'],[]).append((fault['start_s'],fault['end_s']))
        self.assertEqual(len(intervals),4)
        self.assertTrue(all(rows==[(0.,60.),(120.,180.)] for rows in intervals.values()))

    def test_random_step_ranges_reject_fractional_mixed_and_out_of_horizon_inputs(self):
        engine=scenario()
        settings=dict(enabled=True,model='per_step',target_type='link',seed=4,
                      failure_probability=.5,duration_steps=2,start_step=0,end_step=3)
        for change in ({'start_step':-.1},{'start_step':True},{'end_step':3.5},
                       {'duration_steps':0},{'duration_steps':1.5},{'duration_steps':True},
                       {'start_s':0.},{'end_step':5},{'end_step':4}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                spec=replace(engine.net.spec,random_faults={**settings,**change})
                realize_runtime_faults(spec,engine.net.environment_times,engine.net.edges)
        with self.assertRaisesRegex(ValueError,'duration_steps'):
            validate_random_faults({k:v for k,v in settings.items() if k!='duration_steps'})

    def test_fixed_ratio_step_interval_and_legacy_seconds_are_equivalent(self):
        engine=scenario()
        settings=dict(enabled=True,model='fixed_ratio',target_type='node',seed=4,ratio=.5)
        def realize(window):
            return realize_runtime_faults(replace(engine.net.spec,random_faults={**settings,**window}),
                                          engine.net.environment_times,engine.net.edges).faults
        faults=realize(dict(start_step=1,end_step=3))
        self.assertEqual(len(faults),2)
        self.assertTrue(all((f['start_s'],f['end_s'])==(60.,180.) for f in faults))
        self.assertEqual(faults,realize(dict(start_s=60.,end_s=180.)))

    def test_simultaneous_faults_apply_as_one_allocation_revision(self):
        engine=scenario()
        a=dict(target_type='node',target_id='b',start_s=60.,end_s=120.)
        b=dict(target_type='link',target_id='communication_isl:a:d',start_s=60.,end_s=120.)
        engine.now=60.
        with patch.object(engine.net,'_compile',wraps=engine.net._compile) as compile_timeline:
            engine.on_fault_batch(None,((0,a,False),(1,b,False)))
            self.assertEqual(compile_timeline.call_count,1)
        self.assertEqual(engine.net.revision,1)
        for identity in ('communication_isl:a:b','communication_isl:b:c','communication_isl:a:d'):
            self.assertEqual(engine.net.link_at(identity,60.)['state'],'DOWN')
