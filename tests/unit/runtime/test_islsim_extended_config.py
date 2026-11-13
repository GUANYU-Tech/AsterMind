"""Portable validation of the newly imported upstream configuration fields."""
from copy import deepcopy
import unittest

from astermind.engine.setup.islsim_config import prepare_config


class ISLSimExtendedConfigTests(unittest.TestCase):
    def test_existing_defaults_and_explicit_candidate_search(self):
        self.assertEqual(prepare_config()['tx']['baud_rate_ghz'], 60.)
        values = {'tx': {'baud_rate_ghz': None, 'baud_rate_options_ghz': [20, 40],
                         'beam_waist_m': .03},
                  'channel': {'wdm_loss_db': 4., 'pointing_error_rad': .000001}}
        original = deepcopy(values)
        result = prepare_config(values)
        self.assertIsNone(result['tx']['baud_rate_ghz'])
        self.assertEqual(result['tx']['baud_rate_options_ghz'], [20, 40])
        self.assertEqual(result['tx']['beam_waist_m'], .03)
        self.assertEqual(result['channel']['wdm_loss_db'], 4.)
        self.assertEqual(result['channel']['pointing_error_rad'], .000001)
        self.assertEqual(prepare_config(result), result)
        result['tx']['baud_rate_options_ghz'].append(60)
        self.assertEqual(values, original)

    def test_invalid_new_fields(self):
        cases = [
            {'tx': {'baud_rate_options_ghz': []}},
            {'tx': {'baud_rate_options_ghz': '20'}},
            {'tx': {'baud_rate_options_ghz': [True]}},
            {'tx': {'baud_rate_options_ghz': [float('nan')]}},
            {'tx': {'baud_rate_options_ghz': [float('inf')]}},
            {'tx': {'baud_rate_options_ghz': [0]}},
            {'tx': {'baud_rate_ghz': 0}},
            {'tx': {'baud_rate_ghz': True}},
            {'tx': {'beam_waist_m': 0}},
            {'channel': {'wdm_loss_db': -1}},
            {'channel': {'pointing_error_rad': -1}},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_config(value)


if __name__ == '__main__':
    unittest.main()
