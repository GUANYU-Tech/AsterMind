"""Opt-in integration test against a checked-out Midlayer AsterMind."""

import importlib.util
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('_flowsim') and
                     importlib.util.find_spec('astermind') and
                     importlib.util.find_spec('midlayer'),
                     'requires AsterMind and shared-runtime FlowSim')
class AsterMindFlowSimRuntimeTests(unittest.TestCase):
    def test_two_cells_share_network_and_preserve_cross_boundary_progress(self):
        self.run_scenario('flowsim')

    @unittest.skipUnless(importlib.util.find_spec('ns'), 'requires native ns-3')
    def test_two_cells_share_native_packet_network(self):
        self.run_scenario('ns3')

    def run_scenario(self, transport):
        from astermind.config import load_config
        from orbitserve.runtime import (NetworkSpec, NetworkNodeSpec, LinkProfile,
                                        RuntimeStrategySpec, run_simulation)
        from orbitserve.runtime.engine import decision
        from orbitserve.runtime.engine.astermind_bridge import AsterMindCell, AsterMindRuntime
        from tests.unit.runtime.test_runtime_engine import _scenario, _request, _trace

        repository = Path(os.environ.get('ASTERMIND_REPOSITORY',
                          Path(__file__).resolve().parents[3] / 'third_party/astermind'))
        trace = _trace()
        satellites = {'relay': [7000., 0., 0.], 'a0': [7100., 0., 0.],
                      'a1': [7050., 0., 0.], 'b0': [7200., 0., 0.]}
        times = tuple(float(t) for t in range(0, 601, 60))
        pairs = (('relay', 'a1'), ('relay', 'b0'), ('a0', 'a1'))
        trace = replace(trace,
            trace_manifest=replace(trace.trace_manifest, time={**trace.time, 'end_s': 600., 'duration_s': 600., 'step_s': 60.}),
            satellite_catalog=tuple({'satellite_id': n, 'compute_capable': n != 'relay'} for n in satellites),
            ephemeris_samples=tuple({'time_s': t, 'satellite_id': n, 'position_itrs_km': p} for t in times for n, p in satellites.items()),
            sgl_windows=({'ingress_id': 'gateway-0', 'satellite_id': 'relay', 'start_s': 0., 'end_s': 600.},),
            isl_windows=tuple({'endpoint_a': a, 'endpoint_b': b, 'start_s': 0., 'end_s': 600.} for a, b in pairs),
            sunlight_windows=tuple({'satellite_id': n, 'start_s': 0., 'end_s': 600., 'sunlit': True} for n in satellites))
        requests = (
            {**_request('r1', 50., session_id='s1'), 'input_bytes': 100., 'output_bytes': 200., 'prompt_tokens': 16, 'output_tokens': 1},
            {**_request('r2', 51., session_id='s2'), 'input_bytes': 500000., 'output_bytes': 200., 'prompt_tokens': 16, 'output_tokens': 1},
            {**_request('r3', 51.2, session_id='s3'), 'input_bytes': 100., 'output_bytes': 200., 'prompt_tokens': 16, 'output_tokens': 3},
        )
        strategy = RuntimeStrategySpec(latency_weight=1., kv_reuse_weight=0., energy_weight=0., compute_load_weight=0.)
        base = _scenario(strategy, requests=requests, trace=trace)
        nodes = (NetworkNodeSpec('relay', 'communication', 2, 1),
                 NetworkNodeSpec('a0', 'compute', 1, 0, cell_id='A', cell_ingress=False),
                 NetworkNodeSpec('a1', 'compute', 2, 0, cell_id='A', cell_ingress=True, compute_target=False),
                 NetworkNodeSpec('b0', 'compute', 1, 0, cell_id='B', cell_ingress=True),
                 NetworkNodeSpec('gateway-0', 'ground', sgl_terminals=1))
        network = NetworkSpec(nodes, {'sgl': LinkProfile(8e6, 0.),
            'cross_layer': LinkProfile(8e6, 0.),
            'cell_isl': LinkProfile(1e6, 0., capacity_schedule=((60., .5),))},
            communication_topology='explicit', ground_roles=('communication',),
            transport_model=transport, cell_egress_policies={'A': 'single', 'B': 'single'})
        spec = replace(base, network=network, time_horizon_s=600.)
        cells = {node: AsterMindCell(load_config(repository / 'examples' / filename), mapping, repository)
                 for node, filename, mapping in (
                     ('a0', 'midlayer-qwen3-4b-tp2.json', {0: 'a0', 1: 'a1'}),
                     ('b0', 'midlayer-qwen3-4b.json', {0: 'b0'}))}
        bridge = AsterMindRuntime(cells)
        original = decision.CandidateSelector.select_action
        targets = {'r1': 'a0', 'r2': 'a0', 'r3': 'b0'}

        def select_fixed(selector, state, request=None):
            selected = targets[request['request_id']]
            candidates = [c for c in state['candidates'] if c['compute_satellite_id'] == selected]
            return original(selector, {'candidates': candidates}, request)

        with patch.object(decision.CandidateSelector, 'select_action', select_fixed):
            result = run_simulation(spec, compute_runtime=bridge)
        self.assertEqual(result.request_summary['completed_request_count'], 3)
        events = result.state_summary['event_log']
        reversals = [(a, b) for a, b in zip(events, events[1:]) if a['time_s'] > b['time_s']]
        self.assertFalse(reversals, reversals[:2])
        shared = [e for e in events if e['kind'] == 'bandwidth_update' and e.get('link_id') == 'cell_isl:a0:a1']
        inputs = [e for e in shared if e.get('stage') == 'input' and e.get('request_id') == 'r2']
        collectives = [e for e in shared if e.get('stage') == 'astermind_collective']
        self.assertTrue(any(e['kind'] == 'request_arrival' and e['request_id'] == 'r2'
                            and e['time_s'] == 51 for e in events))
        self.assertTrue(any(e['interval_start_s'] <= 51 < e['interval_end_s']
                            and 'all-reduce' in e.get('astermind_flow_id', '')
                            for e in collectives))
        overlap = [(a, b) for a in inputs for b in collectives
                   if (a['source_node_id'], a['target_node_id']) == (b['source_node_id'], b['target_node_id'])
                   and min(a['interval_end_s'], b['interval_end_s']) > max(a['interval_start_s'], b['interval_start_s']) + 1e-8]
        self.assertTrue(overlap)
        if transport == 'flowsim':
            self.assertTrue(any(abs(a['allocated_bandwidth_bps'] - 500000) < .01
                                and abs(b['allocated_bandwidth_bps'] - 500000) < .01 for a, b in overlap))
            allocations = {}
            for event in shared:
                key = (event['interval_start_s'], event['interval_end_s'],
                       event['source_node_id'], event['target_node_id'])
                total, capacity = allocations.get(key, (0., event['link_capacity_bps']))
                allocations[key] = (total + event['bytes_sent'], capacity)
            for (start, end, _, _), (sent, capacity) in allocations.items():
                self.assertLessEqual(sent, capacity * (end - start) / 8 + 1e-3)
        self.assertEqual(result.provenance['network_transport']['backend'], transport)
        self.assertAlmostEqual(sum(e['bytes_sent'] for e in inputs), 500000., places=3)
        before = [e for e in collectives if abs(e['interval_end_s'] - 60.) < 1e-8]
        after = [e for e in collectives if abs(e['interval_start_s'] - 60.) < 1e-8]
        self.assertTrue(any(a['transfer_id'] == b['transfer_id'] and a['link_capacity_bps'] == 1e6
                            and b['link_capacity_bps'] == .5e6 for a in before for b in after))
