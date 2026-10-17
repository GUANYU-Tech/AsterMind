import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

from simulator.config import SimulationConfig
from simulator.examples.make_examples import example


class SGP4SourceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("astropy") and importlib.util.find_spec("sgp4"), "pinned orbit dependencies required")
    def test_native_propagation_and_coordinate_transform(self):
        elements = dict(schema_version="orbitserve.sgp4_element_bundle.v1", element_semantics="sgp4_mean_elements",
            gravity_model="wgs72", ops_mode="improved", time_scale="utc",
            provenance=dict(source_kind="synthetic_sgp4_mean_elements", source_name="adapter-test",
                            source_version="v1", observationally_fitted=False), satellites=[])
        for index, satellite in enumerate(("satA", "satB")):
            elements["satellites"].append(dict(satellite_id=satellite, satnum=10001+index, classification="U",
                epoch_utc="2026-07-20T00:00:00.000000Z", inclination_deg=53., raan_deg=0., eccentricity=.001,
                argument_of_perigee_deg=0., mean_anomaly_deg=index * .1, mean_motion_rev_per_day=15.25,
                mean_motion_dot_rev_per_day2=0., mean_motion_ddot_rev_per_day3=0., bstar_inv_earth_radii=.00001,
                ephemeris_type=0, element_set_number=1, revolution_number_at_epoch=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"elements.json"
            path.write_text(json.dumps(elements), encoding="utf-8")
            data = example()
            data["physical"]["satellites"][-1]["ecef_position_km"] = [6378.137, 0, 0]
            data["orbit"] = dict(elements_file=str(path), simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                                  sample_step_ns=500000000)
            topology = SimulationConfig.model_validate(data).topology()
            self.assertEqual(len(topology.orbit_grid), 3)
            self.assertNotEqual(topology.orbit_grid[0][1]["satA"], topology.orbit_grid[1][1]["satA"])
            self.assertEqual(set(topology.orbit_grid[1][1]), {"satA", "satB"})


if __name__ == "__main__":
    unittest.main()
