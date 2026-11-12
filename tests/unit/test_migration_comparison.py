"""The migration comparator must fail on simulation changes, not hide them."""

import copy
import unittest
from tests.scripts.compare_migration import normalize


class MigrationComparisonTests(unittest.TestCase):
    def test_host_measurements_do_not_hide_semantic_measurement_fields(self):
        before = {
            "measurement": {"event_loop_wall_s": 1, "kernels": 2},
            "cells": {
                "a": {
                    "execution_performance": {
                        "phase_wall_time_s": {"kernel_prediction": 1},
                        "cache_hits": 3,
                    }
                }
            },
        }
        after = copy.deepcopy(before)
        after["measurement"]["event_loop_wall_s"] = 7
        after["cells"]["a"]["execution_performance"]["phase_wall_time_s"][
            "kernel_prediction"
        ] = 4
        self.assertEqual(normalize(before), normalize(after))
        after["measurement"]["kernels"] = 3
        self.assertNotEqual(normalize(before), normalize(after))

    def test_only_independent_same_tick_observations_may_reorder(self):
        a = {"time_ticks": "10", "time_s": 1e-17, "transfer_id": "a", "kind": "start"}
        b = {**a, "transfer_id": "b"}
        wrap = lambda rows: normalize({"network": {"event_log": rows}})
        self.assertEqual(wrap([a, b]), wrap([b, a]))
        later = {**b, "time_ticks": "11", "time_s": 1.1e-17}
        self.assertNotEqual(wrap([a, later]), wrap([later, a]))
        end = {**a, "kind": "complete"}
        self.assertNotEqual(wrap([a, end]), wrap([end, a]))
        control = {"time_ticks": "10", "time_s": 1e-17, "kind": "fault"}
        self.assertNotEqual(wrap([a, control, b]), wrap([b, control, a]))

    def test_changed_counts_state_and_timing_are_not_normalized(self):
        before = {
            "requests": {"r": {"completed_time_ns": 100}},
            "network": {"bytes_sent": 10},
            "migrations": {"m": {"state": "completed"}},
            "provenance": {"writer": "old"},
        }
        for path, changed in [
            (("requests", "r", "completed_time_ns"), 101),
            (("network", "bytes_sent"), 11),
            (("migrations", "m", "state"), "failed"),
            (("provenance", "writer"), "new"),
        ]:
            after = copy.deepcopy(before)
            node = after
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = changed
            self.assertNotEqual(normalize(before), normalize(after))

    def test_paths_require_explicit_content_verified_identity(self):
        a = {"configuration": {"physical_dataset": "/tmp/a"}}
        b = {"configuration": {"physical_dataset": "/tmp/b"}}
        self.assertNotEqual(normalize(a), normalize(b))
        self.assertEqual(
            normalize(a, {"/tmp/a": "hash-1"}), normalize(b, {"/tmp/b": "hash-1"})
        )
        self.assertNotEqual(
            normalize(a, {"/tmp/a": "hash-1"}), normalize(b, {"/tmp/b": "hash-2"})
        )

    def test_nonfinite_host_measurements_are_not_accepted(self):
        for value in (float("nan"), float("inf"), -1, "1"):
            with self.assertRaises(ValueError):
                normalize({"measurement": {"event_loop_wall_s": value}})


class StartupObservationTests(unittest.TestCase):
    def test_coalesced_host_completion_preserves_validated_capacity(self):
        rows = [
            {
                "phase": "cells",
                "completed": n,
                "total": 2,
                "workers": 2,
                "active": ["a", "b"][n:],
                "wall_s": 1,
            }
            for n in range(3)
        ]
        wrap = lambda value: normalize(
            {"cells": {"a": {}, "b": {}}, "measurement": {"startup_stages": value}}
        )
        self.assertEqual(wrap(rows), wrap([rows[0], rows[2]]))
        invalid = copy.deepcopy(rows)
        invalid[1]["completed"] = 2
        with self.assertRaises(ValueError):
            wrap(invalid)
        invalid = copy.deepcopy(rows)
        invalid[1]["active"] = ["unknown"]
        with self.assertRaises(ValueError):
            wrap(invalid)
