"""Actual GPU/rank graphs, Cell schedulers and external traffic on both engines."""

import json
import unittest

from tests.external.native_cells import test_native_cells as fixtures


class NativeTimelineModesTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeCellIntegrationTests.setUpClass.__func__)
    config = fixtures.NativeCellIntegrationTests.config

    def run_scene(self, config):
        from simulator.engine import Simulation
        from simulator.audit import audit
        from simulator.__main__ import json_default

        with Simulation(config, midlayer_root=self.midlayer) as simulation:
            result = simulation.run()
        target = (
            self.reports
            / f"{config.timeline_mode}-{config.communication_mode}-{self._testMethodName}.json"
        )
        target.write_text(json.dumps(result, default=json_default), encoding="utf8")
        result["audit"] = audit(result)
        target.write_text(json.dumps(result, default=json_default), encoding="utf8")
        for cell in result["cells"].values():
            self.assertEqual(cell["scheduler_runtime"]["version"], "0.31.0")
        return result

    def run_matrix(self, timelines, scopes):
        from simulator.__main__ import json_default

        for timeline in timelines:
            for scope in scopes:
                with self.subTest(timeline=timeline, scope=scope):
                    result = self.run_scene(
                        self.config(
                            "dynamic_tp2_pp2.json",
                            timeline_mode=timeline,
                            backend="simgrid" if timeline == "simgrid" else "flowsim",
                            communication_mode=scope,
                        )
                    )
                    self.assertEqual(result["status"], "completed")
                    self.assertTrue(result["audit"]["passed"])
                    self.assertGreater(result["audit"]["kernels"], 0)
                    self.assertGreater(result["audit"]["communications"], 0)
                    self.assertEqual(result["provenance"]["timeline_mode"], timeline)
                    if scope == "flowsim":
                        self.assertTrue(
                            any(
                                e["kind"] == "route_wait"
                                for e in result["network"]["event_log"]
                            )
                        )
                    if timeline == "simgrid":
                        self.assertEqual(
                            result["network"]["timeline_authority"], "simgrid_engine"
                        )
                        self.assertGreater(
                            result["network"]["scheduler"]["dispatched"], 0
                        )
                    (self.reports / f"{timeline}-{scope}.json").write_text(
                        json.dumps(result, default=json_default), encoding="utf8"
                    )

    def test_orbitserve_three_communication_scopes(self):
        self.run_matrix(
            ("orbitserve",), ("flowsim", "protobridge_satellite", "protobridge_cell")
        )

    def test_simgrid_all_communication(self):
        self.run_matrix(("simgrid",), ("flowsim",))

    def test_simgrid_satellite_surrogate(self):
        self.run_matrix(("simgrid",), ("protobridge_satellite",))

    def test_simgrid_cell_surrogate(self):
        self.run_matrix(("simgrid",), ("protobridge_cell",))

    def test_simgrid_kv_migration_and_prefix_reuse(self):
        result = self.run_scene(
            self.config(
                "dynamic_migration_tp2.json", backend="simgrid", timeline_mode="simgrid"
            )
        )
        self.assertTrue(result["audit"]["passed"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["migrations"]["copy0"]["state"], "completed")
        migration = result["migrations"]["copy0"]
        self.assertGreaterEqual(
            int(migration["completed_time_ticks"]),
            int(migration["network_complete_time_ticks"]),
        )
        from simulator.replay import audit_checks

        self.assertTrue(all(row["state"] == "pass" for row in audit_checks(result)))
        self.assertEqual(
            result["cells"]["cellB"]["final_load"]["storage"][
                "migration_reserved_blocks"
            ],
            0,
        )
