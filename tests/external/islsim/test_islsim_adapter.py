"""Check OrbitServe's adapter against the actual, unmodified ISLSim API."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from orbitserve.runtime import LinkProfile, NetworkSpec, NetworkState
from orbitserve.runtime.setup import islsim
from tests.unit.runtime.test_dynamic_network import network


class ISLSimAdapterTests(unittest.TestCase):
    def test_config_files_and_partial_overrides_reach_native_model(self):
        original, identity = islsim.resolve_config()
        first = deepcopy(original)
        first['rx']['terminal_cap_gbps'] = None
        second = deepcopy(first)
        second['tx']['tx_power_w'] = first['tx']['tx_power_w'] / 5
        with tempfile.TemporaryDirectory() as directory:
            files = [Path(directory) / name for name in ('first.json', 'second.json')]
            for path, value in zip(files, (first, second)):
                path.write_text(json.dumps(value))
            with patch.dict('os.environ', ORBITSERVE_ISLSIM_CONFIG=str(files[0])):
                parameters, identity = islsim.resolve_config()
                a = LinkProfile(1e12, 0, capacity_model='islsim', islsim_config=parameters,
                                islsim_source_sha256=identity)
            with patch.dict('os.environ', ORBITSERVE_ISLSIM_CONFIG=str(files[1])):
                parameters, identity = islsim.resolve_config()
                b = LinkProfile(1e12, 0, capacity_model='islsim', islsim_config=parameters,
                                islsim_source_sha256=identity)
                explicit_parameters = deepcopy(parameters)
                explicit_parameters['tx']['tx_power_w'] = first['tx']['tx_power_w']
                explicit = replace(b, islsim_config=explicit_parameters)
            aa, bb = a.attributes(1000, 0), b.attributes(1000, 0)
            self.assertGreater(aa['received_power_w'], bb['received_power_w'])
            self.assertGreater(aa['snr_db'], bb['snr_db'])
            self.assertGreater(aa['bandwidth_bps'], bb['bandwidth_bps'])
            self.assertEqual(explicit.attributes(1000, 0), aa)
            # A frozen input must not depend on a later default-file selection.
            with patch.dict('os.environ', ORBITSERVE_ISLSIM_CONFIG=str(Path(directory) / 'missing.json')):
                self.assertEqual(replace(a).attributes(1000, 0), aa)
            self.assertEqual(aa['islsim_source_sha256'], identity)
            self.assertEqual(json.loads(files[1].read_text()), second)

    def test_fixed_and_automatic_modulation_use_allowed_native_candidates(self):
        automatic = LinkProfile(1e12, 0, capacity_model='islsim',
                                islsim_config={'rx': {'terminal_cap_gbps': None}})
        fixed = replace(automatic, islsim_config={**deepcopy(automatic.islsim_config),
            'tx': {**automatic.islsim_config['tx'], 'modulation_pool': [[4, 'DP-QPSK']]}})
        auto_result, fixed_result = automatic.attributes(1000, 0), fixed.attributes(1000, 0)
        self.assertEqual(auto_result['modulation_order'], 16)
        self.assertEqual(fixed_result['modulation_order'], 4)
        self.assertEqual(auto_result['bandwidth_bps'], 2 * fixed_result['bandwidth_bps'])
        self.assertFalse(fixed.attributes(5000, 0)['link_feasible'])
        self.assertEqual(fixed.attributes(5000, 0)['bandwidth_bps'], 0)

    def test_real_upstream_results_units_ceiling_and_module_isolation(self):
        before = {n: sys.modules.get(n) for n in ('config', 'link_model', 'validator')}
        p = LinkProfile(1e12, 0, capacity_model='islsim')
        _, _, (config, model, _) = islsim._api()
        c = p.islsim_config
        cfg = config.FSOConfig(tx=config.TXConfig(**c['tx']), rx=config.RXConfig(**c['rx']),
                              channel=config.ChannelConfig(**c['channel']))
        for distance in (100., 1000., 2000., 3000., 4000., 5000.):
            raw, actual = model.compute_link(cfg, distance), p.attributes(distance, 0)
            self.assertEqual(actual['budget_bandwidth_bps'], min(1e12, raw.total_bw_gbps * 1e9))
            self.assertEqual(actual['link_feasible'], raw.feasible)
            self.assertEqual(actual['snr_db'], raw.snr_db)
            self.assertEqual(actual['modulation_order'], raw.modulation_order)
            self.assertEqual(actual['islsim_latency_ms'], raw.latency_ms)
            self.assertEqual(actual['receiver_model'], 'external_islsim_unmodified')
        self.assertEqual(before, {n: sys.modules.get(n) for n in before})
        self.assertEqual(p.attributes(1000, 0)['bandwidth_bps'], 400e9)
        self.assertEqual(replace(p, bandwidth_bps=250e9).attributes(1000, 0)['bandwidth_bps'], 250e9)

    def test_raw_network_defaults_to_islsim_and_explicit_models_opt_out(self):
        base = network()
        source = base.spec.to_dict()
        source['link_profiles'] = {'communication_isl': {'bandwidth_bps': 1e12, 'setup_time_s': 0},
                                   'sgl': {'bandwidth_bps': 1234, 'setup_time_s': 0}}
        spec = NetworkSpec.from_value(source)
        self.assertEqual(spec.link_profiles['communication_isl'].capacity_model, 'islsim')
        self.assertEqual(spec.link_profiles['sgl'].capacity_model, 'fixed')
        source['link_profiles']['communication_isl']['capacity_model'] = 'fixed'
        with patch.object(islsim, 'checkout', side_effect=AssertionError('must not load ISLSim')):
            explicit = NetworkSpec.from_value(source)
            self.assertEqual(explicit.link_profiles['communication_isl'].attributes(4000, 0)['bandwidth_bps'], 1e12)

    def test_missing_or_different_upstream_fails_without_fallback(self):
        with patch.dict('os.environ', ORBITSERVE_ISLSIM_ROOT='/nonexistent/orbitserve-islsim'):
            with self.assertRaises(islsim.ISLSimUnavailable):
                LinkProfile(1e12, 0, capacity_model='islsim').attributes(1000, 0)
            self.assertEqual(LinkProfile(8000, 0).attributes(1000, 0)['bandwidth_bps'], 8000)
        p = LinkProfile(1e12, 0, capacity_model='islsim')
        p.attributes(1000, 0)
        with patch.dict('os.environ', ORBITSERVE_ISLSIM_ROOT='/nonexistent/changed-after-binding'):
            self.assertEqual(p.attributes(1000, 0)['capacity_model'], 'islsim')
        with self.assertRaises(islsim.ISLSimUnavailable):
            replace(p, islsim_source_sha256={'config.py': 'incorrect'}).attributes(1000, 0)

    def test_native_parameters_roundtrip_and_capacity_schedule(self):
        p = LinkProfile(250e9, 0, capacity_model='islsim', capacity_schedule=((1.25, .5),))
        spec = replace(network().spec, link_profiles={'communication_isl': p})
        restored = NetworkSpec.from_value(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(restored.to_dict(), spec.to_dict())
        self.assertEqual(p.attributes(1000, 1.25)['bandwidth_bps'], p.attributes(1000, 0)['bandwidth_bps'] / 2)
        c = deepcopy(p.islsim_config)
        c['tx']['tx_power_w'] = float('nan')
        with self.assertRaises(ValueError):
            replace(p, islsim_config=c)

    def test_infeasible_link_is_down_and_saved_physics_preserves_native_result(self):
        base = network(positions={'a': ((7000., 0., 0.), (7000., 0., 0.)),
                                  'b': ((7000., 1000., 0.), (7000., 5000., 0.))})
        base.trace.trace_id = 'islsim-feasibility-test'
        base.trace.ephemeris_samples = tuple(dict(satellite_id=n, time_s=t,
            position_itrs_km=(7000., 0. if n == 'a' else d, 0.))
            for t, d in ((0., 1000.), (5., 5000.), (10., 1000.), (20., 1000.)) for n in ('a', 'b'))
        p = LinkProfile(250e9, 1., capacity_model='islsim')
        state = NetworkState(base.trace, replace(base.spec, link_profiles={'communication_isl': p}))
        self.assertEqual(state.snapshot(5.)['links'][0]['state'], 'DOWN')
        self.assertEqual(state.snapshot(5.)['links'][0]['reason'], 'link_budget_infeasible')
        _, path = state.transfer('a', 'b', 5., 1000)
        self.assertGreaterEqual(path[0]['start_s'], 11.)
        self.assertEqual(state.snapshot(10.)['links'][0]['state'], 'SETTING_UP')
        self.assertEqual(state.snapshot(11.)['links'][0]['state'], 'UP')
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/physical.sqlite'
            state.save_physical_state(path)
            saved = NetworkState(base.trace, state.spec, physical_state=path)
            for t in (0., 1., 5., 10., 11.):
                self.assertEqual(state.snapshot(t), saved.snapshot(t))

    def test_parallel_precompute_matches_serial_with_real_islsim(self):
        positions = {n: ((7000., i * 100., 0.), (7000., i * 100., 0.))
                     for i, n in enumerate(['a', *(f'b{i}' for i in range(9))])}
        base = network(edges=tuple(('a', f'b{i}', 0., 20.) for i in range(9)),
                       positions=positions, terminals=10)
        base.trace.trace_id = 'islsim-parallel-test'
        spec = replace(base.spec, link_profiles={
            'communication_isl': LinkProfile(250e9, 0, capacity_model='islsim')})
        with tempfile.TemporaryDirectory() as directory:
            serial = NetworkState(base.trace, spec, _defer_compile=True)
            parallel = NetworkState(base.trace, spec, _defer_compile=True)
            serial.save_physical_state(directory + '/serial.sqlite', workers=1)
            parallel.save_physical_state(directory + '/parallel.sqlite', workers=2)
            for t in (0., 5., 19.):
                self.assertEqual(serial.snapshot(t), parallel.snapshot(t))


if __name__ == '__main__':
    unittest.main()
