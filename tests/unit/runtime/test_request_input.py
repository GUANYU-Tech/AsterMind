"""Malformed inputs and long causal chains fail before backend allocation."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from orbitserve.runtime import run_simulation
from orbitserve.runtime.engine.request_input import normalize_runtime_request, prepare_runtime_requests
from tests.unit.runtime.test_runtime_engine import _request, _scenario
from tests.unit.runtime.test_runtime_event_order import strategy


def chain(count):
    result = [_request(f'r{i}') for i in range(count)]
    for i, row in enumerate(result[1:], 1):
        row.update(arrival_time_mode='after_dependency_completion',
                   depends_on_request_id=f'r{i-1}', inter_turn_gap_s=0.)
    return result


class RequestInputTests(unittest.TestCase):
    def test_nonfinite_negative_and_fractional_inputs_are_rejected(self):
        for invalid in (None, [], 'not an object'):
            with self.subTest(row=invalid), self.assertRaisesRegex(ValueError, 'must be an object'):
                normalize_runtime_request(invalid)
        for field in ('arrival_time_s', 'input_bytes', 'output_bytes'):
            for invalid in (float('nan'), float('inf'), -1., True):
                with self.subTest(field=field, value=invalid), self.assertRaisesRegex(ValueError, field):
                    normalize_runtime_request({**_request(), field: invalid})
        for field in ('prompt_tokens', 'output_tokens', 'context_tokens_before_turn'):
            for invalid in (1.5, True, -1, None):
                with self.subTest(field=field, value=invalid), self.assertRaisesRegex(ValueError, field):
                    normalize_runtime_request({**_request(), field: invalid})
        with self.assertRaisesRegex(ValueError, 'arrival_time_mode'):
            normalize_runtime_request({**_request(), 'arrival_time_mode': 'after_completin'})
        with self.assertRaisesRegex(ValueError, 'unknown request predecessor'):
            prepare_runtime_requests([{**_request(), 'arrival_time_mode': 'after_dependency_completion',
                                       'depends_on_request_id': [], 'inter_turn_gap_s': 0.}])

    def test_long_reverse_ordered_chain_and_cycle_detection(self):
        rows = list(reversed(chain(5000)))
        self.assertEqual(len(prepare_runtime_requests(rows)), 5000)
        rows[-1].update(arrival_time_mode='after_dependency_completion',
                        depends_on_request_id='r4999', inter_turn_gap_s=0.)
        with self.assertRaisesRegex(ValueError, 'dependency cycle'):
            prepare_runtime_requests(rows)

    def test_invalid_dependency_never_launches_native_worker(self):
        row = {**_request(), 'arrival_time_mode': 'after_dependency_completion',
               'depends_on_request_id': 'missing', 'inter_turn_gap_s': 0.}
        spec = _scenario(strategy(), requests=(row,))
        spec = replace(spec, network=replace(spec.network, transport_model='ns3'))
        with patch('orbitserve.runtime.engine.ns3_transport.Ns3Transport') as backend:
            with self.assertRaisesRegex(ValueError, 'unknown request predecessor'):
                run_simulation(spec)
        backend.assert_not_called()

    def test_failure_propagates_through_long_chain_without_recursion(self):
        rows = chain(1500)
        rows[0]['ingress_id'] = 'missing-ingress'
        result = run_simulation(_scenario(strategy(), requests=tuple(reversed(rows))))
        self.assertEqual(len(result.request_records), 1500)
        self.assertTrue(all(not row['outcome']['completed'] for row in result.request_records))
        self.assertEqual(sum(row['outcome']['uncompleted_reason'] == 'dependency_uncompleted'
                             for row in result.request_records), 1499)

    def test_setup_failure_closes_allocated_transport(self):
        spec = _scenario(strategy())
        spec = replace(spec, network=replace(spec.network, transport_model='fluid'))
        with patch('orbitserve.runtime.engine.fluid_transport.FluidTransport') as backend:
            backend.return_value.provenance = {'backend': 'fluid'}
            with patch('orbitserve.runtime.engine.events._execute_requests', side_effect=ValueError('setup failed')):
                with self.assertRaisesRegex(ValueError, 'setup failed'):
                    run_simulation(spec)
        backend.return_value.close.assert_called_once()
