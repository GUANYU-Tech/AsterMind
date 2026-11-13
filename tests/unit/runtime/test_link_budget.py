import math
import unittest
from dataclasses import replace

from astermind.engine import LinkProfile


def optical(**values):
    return LinkProfile(200e9, 0, capacity_model='optical_photon_budget',
                       wavelength_m=1550e-9, transmit_power_w=.1,
                       transmit_aperture_m=.05, receive_aperture_m=.05,
                       required_photons_per_bit=20, **values)


class LinkBudgetTests(unittest.TestCase):
    def test_optical_received_energy_and_distance_are_auditable(self):
        profile = optical()
        near, far = (profile.attributes(d, 0) for d in (1000, 2000))
        self.assertAlmostEqual(near['received_power_w'] / far['received_power_w'], 4, places=4)
        self.assertAlmostEqual(near['bandwidth_bps'] / far['bandwidth_bps'], 4, places=4)
        self.assertAlmostEqual(near['uncapped_bandwidth_bps'], near['received_power_w'] / (6.62607015e-34 * 299792458 / 1550e-9) / 20)
        self.assertEqual(profile.attributes(.5, 0)['bandwidth_bps'], 200e9)

    def test_power_receiver_requirement_losses_and_terminal_ceiling(self):
        p = optical()
        base = p.attributes(1000, 0)['bandwidth_bps']
        self.assertAlmostEqual(replace(p, transmit_power_w=.2).attributes(1000, 0)['bandwidth_bps'], 2*base)
        self.assertAlmostEqual(replace(p, required_photons_per_bit=40).attributes(1000, 0)['bandwidth_bps'], base/2)
        self.assertAlmostEqual(replace(p, pointing_loss_db=10).attributes(1000, 0)['bandwidth_bps'], base/10)

    def test_rf_aperture_gain_and_shannon_capacity(self):
        p = LinkProfile(1e12, 0, capacity_model='free_space_shannon', frequency_hz=12e9,
                        channel_bandwidth_hz=250e6, transmit_power_w=10,
                        transmit_aperture_m=.3, receive_aperture_m=.5)
        a = p.attributes(550, 0)
        expected = 10*math.log10(.6*(math.pi*.3/(299792458/12e9))**2)
        self.assertAlmostEqual(a['transmit_gain_db'], expected)
        self.assertLess(p.attributes(1100, 0)['bandwidth_bps'], a['bandwidth_bps'])

    def test_bad_optical_parameters_do_not_create_plausible_numbers(self):
        for changes in ({'wavelength_m': 0}, {'required_photons_per_bit': None}, {'transmit_aperture_m': -1}, {'pointing_loss_db': math.nan}):
            with self.assertRaises(ValueError):
                replace(optical(), **changes)
        with self.assertRaises(ValueError):
            optical().attributes(float('nan'), 0)
