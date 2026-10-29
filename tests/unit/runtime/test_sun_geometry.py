"""Solar incidence must be evaluated at an unocculted receiving terminal."""

import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from orbitserve.runtime import LinkProfile, NetworkState
from orbitserve.runtime.setup.sun_geometry import receiver_sun_separation, sun_visible
from tests.unit.runtime.test_dynamic_network import network


class SunGeometryTests(unittest.TestCase):
    def make_state(self, sun):
        base = network(positions={'a': ((7000., 0., 0.),) * 2,
                                  'b': ((7001., 0., 0.),) * 2})
        base.trace.trace_id = 'sun-visibility-test'
        spec = replace(base.spec, link_profiles={
            'communication_isl': LinkProfile(8., 0., sun_exclusion_angle_deg=3.)})
        rows = tuple(dict(time_s=t, position_itrs_km=sun) for t in (0., 20.))
        return NetworkState(base.trace, spec, sun_positions=rows), rows

    def test_earth_occulted_sun_cannot_disable_an_aligned_night_side_link(self):
        state, _ = self.make_state((-1.5e8, 0., 0.))
        edge = state.edges[0]
        self.assertEqual(state.sun_separation(edge, 0.), 180.)
        self.assertEqual(state.snapshot(0.)['links'][0]['state'], 'UP')
        self.assertEqual(state.transfer('a', 'b', 0., 1.)[1][0]['link_id'], edge['link_id'])

    def test_direct_visible_sun_is_checked_at_each_receiving_direction(self):
        state, _ = self.make_state((1.5e8, 0., 0.))
        edge = state.edges[0]
        self.assertEqual(state.sun_separation(edge, 0., receiver_id='a'), 0.)
        self.assertEqual(state.sun_separation(edge, 0., receiver_id='b'), 180.)
        self.assertEqual(state.snapshot(0.)['links'][0]['reason'], 'sun_exclusion')
        with self.assertRaisesRegex(ValueError, 'endpoint'):
            state.sun_separation(edge, 0., receiver_id='not-an-endpoint')

    def test_surface_horizon_and_ellipsoidal_polar_limb(self):
        self.assertTrue(sun_visible((6378.137, 0., 0.), (1.5e8, 0., 0.)))
        self.assertFalse(sun_visible((6378.137, 0., 0.), (-1.5e8, 0., 0.)))
        self.assertTrue(sun_visible((6378.137, 0., 0.), (6378.137, 1.5e8, 0.)))
        self.assertTrue(sun_visible((0., 0., 6360.), (1.5e8, 0., 6360.)))
        self.assertIsNone(receiver_sun_separation((7000., 0., 0.), (6999., 0., 0.), (-1.5e8, 0., 0.)))
        self.assertEqual(receiver_sun_separation((6378.137, 0., 0.), (7000., 0., 0.), (1.5e8, 0., 0.)), 0.)
        self.assertIsNone(receiver_sun_separation((-7000., 0., 0.), (-6378.137, 0., 0.), (1.5e8, 0., 0.)))

    def test_precomputed_visibility_matches_direct_and_rejects_legacy_cache(self):
        state, rows = self.make_state((-1.5e8, 0., 0.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'physical.sqlite'
            state.save_physical_state(path)
            cached = NetworkState(state.trace, state.spec, sun_positions=rows, physical_state=path)
            self.assertEqual(cached.snapshot(0.)['links'][0]['state'], 'UP')
            with sqlite3.connect(path) as db:
                value = json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])
                value.pop('sun_geometry_version')
                db.execute('UPDATE metadata SET value=?', (json.dumps(value),))
            with self.assertRaisesRegex(ValueError, 'obsolete Sun visibility'):
                NetworkState(state.trace, state.spec, physical_state=path)
