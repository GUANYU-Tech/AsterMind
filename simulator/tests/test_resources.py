import unittest

from network_sim import PhysicalTopology, Scenario
from simulator.config import SimulationConfig
from simulator.examples.make_examples import example
from simulator.resources import expand_presets


class ResourceTests(unittest.TestCase):
    def test_mesh_is_four_neighbors_and_not_eight(self):
        data = dict(cells=[dict(id="cell", satellites=["sat"])], satellites=[dict(id="sat")],
            devices=[dict(id=f"gpu{i}", satellite="sat", kind="gpu") for i in range(9)],
            templates=[dict(id="mesh", kind="mesh", rows=3, columns=3,
                            devices=[f"gpu{i}" for i in range(9)], bandwidth_gbps=10)])
        topology = PhysicalTopology(Scenario.model_validate(expand_presets(data)))
        self.assertEqual(len(topology.links), 12)
        center = [l for l in topology.links if "gpu4" in (l.source, l.target)]
        self.assertEqual(len(center), 4)

    def test_independent_topology_and_placement_checks(self):
        data = example()
        data["cells"][0]["placements"][0]["device"] = "gpu1"
        with self.assertRaisesRegex(ValueError, "outside physical Cell"):
            SimulationConfig.model_validate(data).topology()

    def test_gateway_count_is_checked(self):
        data = example()
        data["cells"][0]["egress_mode"] = "multi_gateway"
        with self.assertRaisesRegex(ValueError, "at least two"):
            SimulationConfig.model_validate(data).topology()

    def test_communication_only_satellite_and_compute_gateway(self):
        data = example()
        # One serving Cell spans a compute star and a communication-only star.
        data["physical"]["cells"] = [dict(id="cellA", satellites=["satA", "satB"]),
                                         dict(id="ground-resources", satellites=["ground"])]
        data["physical"]["devices"] = [d for d in data["physical"]["devices"] if d["id"] != "gpu1"]
        data["physical"]["links"] = [l for l in data["physical"]["links"] if l["id"] not in ("local1", "ground1")]
        data["cells"] = data["cells"][:1]
        data["satellite_roles"] = dict(satA="compute", satB="communication")
        topology = SimulationConfig.model_validate(data).topology()
        self.assertIn("exit0", topology.devices)  # gateway may be on a compute satellite

    def test_islsim_generates_actual_budget(self):
        data = example()
        data["link_parameters"] = {"isl": dict(kind="islsim", distance_km=1000)}
        topology = SimulationConfig.model_validate(data).topology()
        report = topology.environment_report["links"]["isl"]
        self.assertEqual(report["source"], "ISLSim")
        self.assertGreater(report["budgets"][0]["latency_ms"], 3)

    def test_both_orbit_modes_reuse_positions_and_generate_windows(self):
        for mode in ("per_member", "reference_offsets"):
            data = example()
            data["physical"]["satellites"][-1]["ecef_position_km"] = [6378.137, 0, 0]
            samples = []
            for time in (0, 0.5, 1):
                samples.append(dict(satellite_id="satA", time_s=time, position_itrs_km=[7000, time, 0]))
                if mode == "per_member":
                    samples.append(dict(satellite_id="satB", time_s=time, position_itrs_km=[7000, time+10, 0]))
            data["orbit"] = dict(mode=mode, samples=samples)
            if mode == "reference_offsets":
                data["orbit"].update(reference_satellite="satA", offsets_itrs_km=dict(satB=[0,10,0]))
            topology = SimulationConfig.model_validate(data).topology()
            self.assertEqual(topology.orbit_grid[1][1]["satB"], (7000, 10.5, 0))
            self.assertEqual(next(l for l in topology.links if l.id == "isl").availability_windows[0].end_ns, 1000000000)


if __name__ == "__main__":
    unittest.main()
