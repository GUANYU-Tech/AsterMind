"""Whole-path availability, capacity bases and whole-intra selection boundaries."""
import unittest
from astermind.engine.clock import Time
from astermind.engine.comm.frame_surrogate import CommPolicy
from astermind.engine.comm.operator import CommunicationOperator
from astermind.engine.comm.path_budget import path_budget,resource_capacities
from astermind.engine.comm.planner import plan_communication
from astermind.engine.comm._vendor.inter_mean.mean import mean_frame_budget
from tests.comm_fixture import Frames
from tests.unit.runtime import test_comm_port as fixtures
from tests.unit.runtime.test_dynamic_network import network as physical_network
from tests.network_case import device_runtime
from examples.native.compute_comm_matrix import scene

class CommSemanticsTests(unittest.TestCase):
    def test_temporal_path_without_simultaneous_edges_cannot_store_forward(self):
        physical=physical_network(edges=(('a','b',0.,5.),('b','c',6.,20.)),bandwidth=8e9)
        host,network=device_runtime(physical);self.addCleanup(host.close);self.addCleanup(network.close)
        a,b=network.resolve_pair('a:hub','c:hub')
        _,legs=host.net.transfer(a,b,host.now,1000)
        self.assertTrue(any(leg['start_s']>=6 for leg in legs))
        done=[];failed=[]
        network.communication_policy.submit_transfer('r','a:hub','c:hub',1000,done.append,failed.append)
        host.run()
        self.assertFalse(done)
        self.assertEqual(len(failed),1)
        self.assertFalse(any(row['kind']=='service' for row in network.communication_policy.trace))
        self.assertFalse(network.communication_policy.tasks or network.communication_policy.flows)

    def test_waiting_path_starts_when_all_edges_become_available(self):
        physical=physical_network(edges=(('a','b',0.,20.),('b','c',6.,20.)),bandwidth=8e9)
        host,network=device_runtime(physical);self.addCleanup(host.close);self.addCleanup(network.close)
        done=[];comm=network.communication_policy
        comm.submit_transfer('r','a:hub','c:hub',1000,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(len(done),1)
        self.assertGreater(done[0],Time(6))
        service=[row for row in comm.trace if row['kind']=='service']
        self.assertTrue(service)
        self.assertTrue(all(int(row['start_ticks'])>=Time(6).ticks for row in service))
        self.assertAlmostEqual(sum(row['application_bytes'] for row in service),1000)

    def test_framing_mapping_and_fec_are_applied_once_at_the_declared_basis(self):
        routes=Frames().routes(Frames().frame_at(0));size=2000
        route=routes.route('g0','g1',size)
        budgets={}
        for basis in ('raw_line','net_after_existing_fec','application_effective'):
            policy=CommPolicy(runtime_capacity_basis=basis,inter_parameters=dict(hop_processing_us=0))
            budget=path_budget(routes,route,size,policy)
            coefficient=next(v for k,v in budget['demand'].items() if len(k)==3 and k[0]=='ab')
            budgets[basis]=coefficient
        settings=CommPolicy().inter_parameters;frame=mean_frame_budget(size,settings)
        expected=frame.ethernet_time_bytes/size/settings.mapping_efficiency
        self.assertAlmostEqual(budgets['net_after_existing_fec'],expected)
        self.assertAlmostEqual(budgets['raw_line'],expected*settings.fec.n/settings.fec.k)
        self.assertEqual(budgets['application_effective'],1)

    def test_shared_physical_pool_still_counts_each_traversal(self):
        routes=Frames().routes(Frames().frame_at(0))
        edge=routes.edges[('ab','a','b')]
        edge['shared_pools']=('rf',)
        routes.frame.shared_capacity_pools_bps={'rf':8e9}
        policy=CommPolicy(runtime_protocol='payload_only',inter_parameters=dict(startup_us=0))
        route=routes.route('g0','g1',1000)
        self.assertEqual(resource_capacities(routes,policy)[('pool','rf')],1)
        self.assertEqual(path_budget(routes,route,1000,policy)['demand'][('pool','rf')],1)

    def test_whole_hierarchical_and_ragged_decomposition_keep_source_boundaries(self):
        f=fixtures.CommPortTests();self.addCleanup(f.doCleanups)
        host,network,comm=f.runtime(scene('DS05'))
        participants=tuple(f'satA-gpu-{i}' for i in range(8))
        task=CommunicationOperator('all_reduce','ring',participants,4096,'per_rank_input',reduction='sum',cell_id='cellA')
        plan=plan_communication(comm.network,comm.policy,task,host.now)
        self.assertFalse(plan.waves)
        self.assertIn('hierarchical',plan.estimate.mode)
        selected=plan.selection
        self.assertEqual(selected['scope'],'intra_cross_su')
        variable=CommunicationOperator('all_gatherv','pairwise',participants,24,'aggregate_input',
            partition_bytes=(8,0,4,12,0,0,0,0),cell_id='cellA')
        plan=plan_communication(comm.network,comm.policy,variable,host.now)
        self.assertTrue(plan.waves)
        self.assertEqual(plan.selection['algorithm'],'pairwise')

    def test_whole_intra_operations_do_not_compete_globally_or_reprice(self):
        f=fixtures.CommPortTests();self.addCleanup(f.doCleanups)
        host,network,comm=f.runtime(scene('DS01'));done=[]
        task=CommunicationOperator('all_reduce','ring',('satA-gpu-0','satA-gpu-1'),4096,
            'per_rank_input',reduction='sum',cell_id='cellA')
        plan=plan_communication(comm.network,comm.policy,task,host.now)
        for i in range(2):comm.submit(task,lambda e:done.append(host.now),self.fail,task_id=f'whole{i}')
        self.assertFalse(comm.active)
        # Upstream whole-intra tasks are timed predictions, not active flows.
        host.timeline.schedule(Time.from_ns(1),0,'refresh',comm.network_changed)
        host.run()
        self.assertEqual(done,[Time.from_ns(plan.estimate.duration_ns)]*2)

    def test_roofline_does_not_fall_back_when_native_context_is_absent(self):
        f=fixtures.CommPortTests();self.addCleanup(f.doCleanups)
        host,network,comm=f.runtime(scene('DS01',intra_mode='IntraSatelliteRoofline'));failed=[];done=[]
        task=CommunicationOperator('all_reduce','ring',('satA-gpu-0','satA-gpu-1'),4096,
            'per_rank_input',reduction='sum',cell_id='cellA',task_id='missing-context')
        comm.submit(task,done.append,failed.append);host.run()
        self.assertFalse(done);self.assertEqual(len(failed),1)
        self.assertIn('missing original',failed[0].reason)
