"""Check expanded portable defaults preserve existing native calculations."""
from copy import deepcopy
import unittest

from astermind.engine.setup import islsim
from astermind.engine.setup.islsim_config import prepare_config


class ISLSimMigrationDefaultTests(unittest.TestCase):
    def test_explicit_new_fields_preserve_implicit_upstream_defaults(self):
        _, _, (config, model, validator) = islsim._api()
        expanded = prepare_config()
        legacy = deepcopy(expanded)
        for key in ('beam_waist_m', 'baud_rate_options_ghz'):
            del legacy['tx'][key]
        for key in ('wdm_loss_db', 'pointing_error_rad'):
            del legacy['channel'][key]
        def native(value):
            result = config.FSOConfig(tx=config.TXConfig(**value['tx']),
                                     rx=config.RXConfig(**value['rx']),
                                     channel=config.ChannelConfig(**value['channel']))
            validator.validate_config(result)
            return result
        before, after = native(legacy), native(expanded)
        self.assertEqual(config.config_to_dict(before), config.config_to_dict(after))
        for distance in (100., 1000., 3000., 5000.):
            with self.subTest(distance=distance):
                left = model.compute_link(before, distance)
                right = model.compute_link(after, distance)
                for field in ('total_bw_gbps', 'feasible', 'snr_db',
                              'modulation_order', 'latency_ms'):
                    self.assertEqual(getattr(left, field), getattr(right, field))
