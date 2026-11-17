"""Real Comm allocator on the retained route/device graph, with analytic oracles."""
import unittest
from unittest.mock import patch

from astermind.builder.config import SimulationConfig
from astermind.engine.network_host import DeviceRuntimeHost, DeviceCommunicationRuntime
from astermind.engine.clock import Time


def scenario(*, horizon_ns=100_000, capacity_schedule_ns=()):
    physical=dict(cells=[],satellites=[],devices=[],links=[])
    cells=[]
    for owner in ('a','b'):
        physical['cells'].append(dict(id=owner,satellites=[owner]))
        physical['satellites'].append(dict(id=owner))
        physical['devices'] += [dict(id=f'{owner}g',satellite=owner,kind='gpu'),
            dict(id=f'{owner}t',satellite=owner,kind='terminal',external_access=True,capacity_gbps=8)]
        physical['links'].append(dict(id=f'{owner}local',source=f'{owner}g',target=f'{owner}t',bandwidth_gbps=100))
        cells.append(dict(id=owner,placements=[dict(rank=0,device=f'{owner}g')],
            ingress_device=f'{owner}g',output_device=f'{owner}g',external_terminals=[f'{owner}t']))
    physical['links'].append(dict(id='isl',source='at',target='bt',bandwidth_gbps=8,
        latency_ns=10,capacity_schedule_ns=capacity_schedule_ns))
    return dict(physical=physical,cells=cells,requests=[],horizon_ns=horizon_ns,
        communication=dict(runtime_protocol='payload_only',inter_parameters=dict(startup_us=0)))


class CommPortTests(unittest.TestCase):
    def runtime(self, raw=None, **options):
        config=SimulationConfig.model_validate(raw or scenario(**options))
        topology=config.topology();host=DeviceRuntimeHost(topology,config)
        network=DeviceCommunicationRuntime(host,topology)
        self.addCleanup(host.close);self.addCleanup(network.close)
        return host,network,network.communication_policy

    def test_single_external_flow_matches_serialization_and_propagation(self):
        host,network,comm=self.runtime();done=[]
        comm.submit_transfer('r','ag','bg',1000,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(done,[Time.from_ns(1010)])
        self.assertAlmostEqual(sum(r['application_bytes'] for r in comm.trace if r['kind']=='service'),1000)

    def test_two_flows_share_the_same_resource_once(self):
        host,network,comm=self.runtime();done=[]
        for i in range(2):
            comm.submit_transfer(str(i),'ag','bg',1000,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(done,[Time.from_ns(2010)]*2)
        self.assertEqual(comm.stats['peak_active_flows'],2)
        self.assertFalse(comm.tasks or comm.flows or comm.active)

    def test_capacity_change_settles_old_service_before_reallocation(self):
        host,network,comm=self.runtime(capacity_schedule_ns=((500,0.5),));done=[]
        comm.submit_transfer('r','ag','bg',1000,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(done,[Time.from_ns(1510)])
        self.assertAlmostEqual(sum(r['application_bytes'] for r in comm.trace if r['kind']=='service'),1000)

    def test_exact_tick_progress_at_large_absolute_time(self):
        origin=Time(10**12)
        host,network,comm=self.runtime(horizon_ns=10**21+100_000)
        host.heap.clear();host.now=origin;comm.last_time=origin;comm.last_ns=comm.now_ns
        done=[]
        comm.submit_transfer('r','ag','bg',1,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(done[0].ticks-origin.ticks,11*10**9)

    def test_cancel_frees_resources_and_notifies_once(self):
        host,network,comm=self.runtime();failed=[];done=[]
        comm.submit_transfer('r','ag','bg',1000,lambda:done.append(host.now),failed.append)
        host.timeline.schedule(Time.from_ns(100),0,'cancel',lambda:comm.cancel('r'))
        host.run()
        self.assertFalse(done or comm.active or comm.flows or comm.tasks)
        self.assertEqual([e.reason for e in failed],['cancelled'])

    def test_unavailable_path_waits_then_times_out(self):
        host,network,comm=self.runtime();failed=[];done=[]
        host.net.failed_device_links.add('isl')
        comm.submit_transfer('r','ag','bg',1000,lambda:done.append(host.now),failed.append)
        host.run()
        self.assertFalse(done)
        self.assertEqual(len(failed),1)
        self.assertIn('deadline',failed[0].reason)
        self.assertFalse(comm.active or comm.tasks or comm.flows)

    def test_route_delegates_exact_ordered_original_legs(self):
        host,network,comm=self.runtime()
        a,b=network.resolve_pair('ag','bg')
        _,original=host.net.transfer(a,b,host.now,1000)
        route=comm.route('ag','bg',1000)
        self.assertEqual(route['target_route_legs'],tuple((r['link_id'],r['source_node_id'],r['target_node_id']) for r in original))
        self.assertEqual(route['path'],network.route('ag','bg',1000)['path'])
        with self.assertRaisesRegex(ValueError,'retained routing'):
            comm.route('ag','bg',1000,first_hop='bg')

    def test_horizon_is_inclusive_for_arrival_but_explicit_deadline_is_strict(self):
        host,network,comm=self.runtime();done=[]
        comm.network.horizon=Time.from_ns(1010)
        comm.submit_transfer('last','ag','bg',1000,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(done,[Time.from_ns(1010)])
        host,network,comm=self.runtime(horizon_ns=2000);failed=[]
        comm.submit_transfer('last','ag','bg',1000,self.fail,failed.append,deadline_s=Time.from_ns(1010))
        host.run()
        self.assertEqual(len(failed),1)

    def test_completion_at_actual_host_horizon(self):
        # A later contender moves an already admitted flow to the host end.
        raw=scenario(horizon_ns=1510)
        for device in raw['physical']['devices']:
            if device['kind']=='terminal':device['capacity_gbps']=1000
        for link in raw['physical']['links']:
            if link['id']!='isl':link['bandwidth_gbps']=1000
        host,network,comm=self.runtime(raw);done=[]
        comm.submit_transfer('last','ag','bg',1000,lambda:done.append(host.now),self.fail)
        host.timeline.schedule(Time.from_ns(100),0,'join',
            lambda:comm.submit_transfer('other','ag','bg',500,lambda:None,self.fail))
        host.run()
        self.assertEqual(done,[host.end])
        self.assertFalse(comm.tasks or comm.flows or comm.active)

    def test_relay_terminal_is_one_payload_gate_with_mixed_link_contracts(self):
        from astermind.engine.comm.path_budget import path_budget,resource_capacities
        raw=scenario()
        raw['physical']['cells'].append(dict(id='relay',satellites=['relay']))
        raw['physical']['satellites'].append(dict(id='relay'))
        raw['physical']['devices'].append(dict(id='rt',satellite='relay',kind='terminal',
            external_access=True,capacity_gbps=8,max_external_links=2))
        raw['physical']['links'][-1:]=[
            dict(id='ar',source='at',target='rt',bandwidth_gbps=100,latency_ns=5),
            dict(id='rb',source='rt',target='bt',bandwidth_gbps=100,latency_ns=5)]
        for device in raw['physical']['devices']:
            if device['id'] in ('at','bt'):device['capacity_gbps']=100
        host,network,comm=self.runtime(raw);done=[]
        routes=comm._refresh();route=routes.route('ag','bg',1000)
        external=[key for key in route['resources'] if routes.edges[key]['source_owner']!=routes.edges[key]['target_owner']]
        routes.edges[external[0]]['attributes']['capacity_basis']='raw_line'
        routes.edges[external[1]]['attributes']['capacity_basis']='application_effective'
        resource_capacities(routes,comm.policy)
        self.assertEqual(path_budget(routes,route,1000,comm.policy)['demand'][('pool','device:rt')],1)
        comm.submit_transfer('relay','ag','bg',1000,lambda:done.append(host.now),self.fail)
        host.run()
        self.assertEqual(done,[Time.from_ns(1010)])

    def test_route_does_not_forecast_current_external_contenders(self):
        host,network,comm=self.runtime()
        comm.submit_transfer('r','ag','bg',1000,lambda:None,self.fail)
        observed=[]
        def check():
            with patch.object(host.net,'_edge_transfer',side_effect=AssertionError('route forecast')):
                route=network.route('ag','bg',1)
                self.assertNotIn('isolated_estimate_ns',route)
                observed.append(route['path'])
        host.timeline.schedule(Time.from_ns(50),2,'check',check)
        host.timeline.schedule(Time.from_ns(2000),2,'check',check)
        host.run()
        self.assertEqual(observed[0],observed[1])
        self.assertEqual(comm.stats['cell_estimate_calls'],0)
        self.assertEqual(host.net.path_route_cache_info()['searches'],1)

    def test_cell_estimate_is_explicit_and_accounted_separately(self):
        host,network,comm=self.runtime()
        for _ in range(10):
            network.route('ag','bg',1000)
        self.assertEqual(comm.stats['cell_estimate_calls'],0)
        with patch('astermind.engine.comm.path_budget.resource_capacities',side_effect=AssertionError('global rescan')):
            estimate=network.estimate_transfer('ag','bg',1000)
        self.assertEqual(estimate['estimated_transfer_ns'],1010)
        self.assertEqual(comm.stats['cell_estimate_calls'],1)
        self.assertGreater(comm.stats['cell_estimate_wall_s'],0)
        self.assertEqual(comm.stats['route_wall_s'],0)
        self.assertEqual(host.net.path_route_cache_info()['searches'],1)

    def test_routing_configuration_is_extensible_but_refresh_is_internal(self):
        raw=scenario()
        raw['routing']={'policy':'my-policy'}
        config=SimulationConfig.model_validate(raw)
        self.assertEqual(config.routing.policy,'my-policy')
        with self.assertRaisesRegex(ValueError,'Unregistered routing policy'):
            DeviceRuntimeHost(config.topology(),config)
        raw['routing']['update_interval_s']=5
        with self.assertRaises(ValueError):
            SimulationConfig.model_validate(raw)

    def test_optional_estimate_observes_competition_without_changing_execution(self):
        from copy import deepcopy
        host, network, comm = self.runtime()
        done = []
        comm.submit_transfer('background', 'ag', 'bg', 1000, lambda: done.append(host.now), self.fail)
        observed = []
        def check():
            active, rates, generation = deepcopy(comm.active), dict(comm.rates), comm.generation
            isolated = network.estimate_transfer('ag', 'bg', 1000)
            context = network.estimation_context()
            competing = network.estimate_transfer('ag', 'bg', 1000, context=context)
            self.assertEqual(isolated['estimated_transfer_ns'], 1010)
            self.assertEqual(competing['estimated_transfer_ns'], 2010)
            self.assertEqual(comm.active, active)
            self.assertEqual(comm.rates, rates)
            self.assertEqual(comm.generation, generation)
            observed.append(context)
        host.timeline.schedule(Time.from_ns(50), 2, 'estimate', check)
        def stale_context():
            with self.assertRaisesRegex(ValueError, 'no longer current'):
                network.estimate_transfer('ag', 'bg', 1000, context=observed[0])
        host.timeline.schedule(Time.from_ns(60), 2, 'stale-estimate', stale_context)
        host.run()
        self.assertEqual(done, [Time.from_ns(1010)])
        self.assertEqual(comm.stats['cell_estimate_calls'], 3)
        self.assertEqual(comm.stats['cell_estimate_context_calls'], 1)

    def test_internal_link_fault_retries_then_recovers_with_one_terminal_event(self):
        from astermind.engine.device_controls import apply_device_command
        host,network,comm=self.runtime();done=[]
        comm.submit_transfer('r','ag','bg',1000,lambda:done.append(host.now),self.fail)
        for at,action in ((100,'device_link_fail'),(200,'device_link_restore')):
            host.timeline.schedule(Time.from_ns(at),-3,action,
                lambda action=action:apply_device_command(host,dict(action=action,target_id='alocal')))
        host.run()
        self.assertEqual(done,[Time.from_ns(1210)])
        self.assertEqual(comm.stats['retries'],1)
        self.assertAlmostEqual(comm.stats['discarded_application_bytes'],100)

    def test_staggered_flows_conserve_bytes_and_cancelled_generation_stays_stale(self):
        host,network,comm=self.runtime();done=[]
        comm.submit_transfer('a','ag','bg',1000,lambda:done.append(('a',host.now)),self.fail)
        host.timeline.schedule(Time.from_ns(500),0,'join',
            lambda:comm.submit_transfer('b','ag','bg',1000,lambda:done.append(('b',host.now)),self.fail))
        host.run()
        self.assertEqual(done,[('a',Time.from_ns(1510)),('b',Time.from_ns(2010))])
        self.assertAlmostEqual(sum(row['application_bytes'] for row in comm.trace if row['kind']=='service'),2000)
        self.assertFalse(comm.tasks or comm.active or comm.flows)

    def test_collective_variable_partitions_and_traffic_matrix(self):
        from astermind.engine.comm.operator import CommunicationOperator
        from examples.native.compute_comm_matrix import scene
        raw=scene('DS01');raw['requests']=[]
        raw['communication']=dict(runtime_protocol='payload_only',inter_parameters=dict(startup_us=0))
        p=('satA-gpu-0','satA-gpu-1','satB-gpu-0','satB-gpu-1')
        ops=[CommunicationOperator('all_gatherv','pairwise',p,24,'aggregate_input',partition_bytes=(8,0,4,12)),
             CommunicationOperator('reduce_scatterv','pairwise',p,24,'per_rank_input',reduction='sum',partition_bytes=(8,0,4,12)),
             CommunicationOperator('all_to_all','pairwise',p,30,'aggregate_input',traffic_matrix_bytes=((0,0,10,0),(0,0,0,20),(0,0,0,0),(0,0,0,0))),
             CommunicationOperator('all_reduce','ring',p,1024,'per_rank_input',reduction='sum'),
             CommunicationOperator('all_gather','ring',p,1024,'per_rank_input'),
             CommunicationOperator('send_recv','send_recv',p[:2],0,'payload')]
        for op in ops:
            with self.subTest(operation=op.operation):
                host,network,comm=self.runtime(raw);done=[]
                comm.submit(op,done.append,self.fail)
                host.run()
                self.assertEqual(len(done),1)
                if op.operation in ('all_gatherv','reduce_scatterv'):
                    self.assertEqual((done[0].flow_count,done[0].cross_satellite_bytes),(9,48))
                if op.operation=='all_to_all':
                    self.assertEqual((done[0].flow_count,done[0].cross_satellite_bytes),(2,30))
                self.assertFalse(comm.tasks or comm.active or comm.flows)

    def test_zero_capacity_candidate_does_not_break_other_reachable_routes(self):
        from examples.native.compute_comm_matrix import scene
        raw=scene('DS01');raw['requests']=[]
        raw['physical']['links'][-1]['capacity_schedule_ns']=[(1,0)]
        host,network,comm=self.runtime(raw)
        route=network.route('user','satA-gpu-0',1000)
        self.assertEqual(route['path'][0],'user')
        self.assertEqual(route['path'][-1],'satA-gpu-0')

    def test_old_modes_and_upstream_routing_selector_are_rejected(self):
        for field in ('backend','timeline_mode','communication_mode','protobridge_startup_us'):
            raw=scenario();raw[field]='old'
            with self.subTest(field=field),self.assertRaises(ValueError):SimulationConfig.model_validate(raw)
        raw=scenario();raw['communication']['routing_policy']='unknown'
        with self.assertRaises(ValueError):SimulationConfig.model_validate(raw)


if __name__=='__main__':unittest.main()
