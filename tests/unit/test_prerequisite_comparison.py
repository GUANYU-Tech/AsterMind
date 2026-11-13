import copy
import unittest

from tests.scripts.compare_migration import differences
from tests.scripts.compare_prerequisites import legacy_projection


class PrerequisiteComparisonTests(unittest.TestCase):
    def baseline(self):
        return dict(requests={"r": dict(status="completed", cell_ready_ticks="12")},
                    cells={"A": dict(steps=[dict(duration=4)])},
                    configuration=dict(migrations=[dict(id="m")] ),
                    transfers=[dict(completion_time_ticks="10")])

    def test_named_additions_preserve_all_legacy_values(self):
        old = self.baseline()
        new = copy.deepcopy(old)
        new["request_state_events"] = []
        new["requests"]["r"].update(metrics={}, input_ready_ticks="11")
        new["cells"]["A"]["load_history"] = []
        new["configuration"]["migrations"][0]["before_request"] = None
        self.assertEqual(legacy_projection(new), old)
        self.assertIn("request_state_events", new, "comparison cannot mutate evidence")

    def test_simulated_values_and_unknown_fields_are_not_hidden(self):
        for mutate in (
            lambda r: r["requests"]["r"].update(cell_ready_ticks="13"),
            lambda r: r["cells"]["A"]["steps"][0].update(duration=5),
            lambda r: r["transfers"][0].update(completion_time_ticks="11"),
            lambda r: r["requests"]["r"].update(unknown_field=1),
        ):
            old, new = self.baseline(), self.baseline()
            mutate(new)
            self.assertTrue(list(differences(old, legacy_projection(new))))

    def test_new_dependency_scenarios_cannot_claim_legacy_equivalence(self):
        report = self.baseline()
        report["configuration"]["migrations"][0]["before_request"] = "r"
        with self.assertRaisesRegex(ValueError, "not legacy baselines"):
            legacy_projection(report)
