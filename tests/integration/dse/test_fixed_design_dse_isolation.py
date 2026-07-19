import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.dse.search import DSESpec, run_dse
from orbitserve.dse.search import _simulation_config_for_dse_design
from orbitserve.experiments.design_evaluation import (
    _simulation_config_with_default_trace_dir,
    evaluate_materialized_design,
    safe_trace_path_component,
)
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.experiments.runner import _runtime_result_summary


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_runtime_matrix.json"


def _skeleton_spec(*, dse: DSESpec):
    base = load_experiment_spec(MANIFEST)
    return replace(
        base,
        simulation=replace(
            base.simulation,
            trace_model="skeleton",
            trace_output_dir=None,
        ),
        dse=dse,
    )



def _without_warnings(value):
    if isinstance(value, dict):
        return {
            key: _without_warnings(item)
            for key, item in value.items()
            if key != "warnings"
        }
    if isinstance(value, list):
        return [_without_warnings(item) for item in value]
    return value


class FixedDesignAndDSEIsolationTests(unittest.TestCase):

    def test_empty_sweep_baseline_matches_fixed_design_core_evaluation(self):
        fixed_spec = _skeleton_spec(dse=DSESpec(enabled=False))
        fixed = run_experiment(fixed_spec).to_dict()

        study = run_dse(_skeleton_spec(dse=DSESpec(enabled=True, sweep={}))).to_dict()
        self.assertEqual(fixed["result_kind"], "fixed_design_simulation")
        self.assertEqual(study["result_kind"], "dse_study")
        self.assertEqual(study["design_count"], 1)

        baseline = study["rows"][0]
        self.assertEqual(baseline["result_kind"], "dse_candidate")
        self.assertEqual(baseline["assignments"], {})
        self.assertEqual(baseline["base_design_id"], fixed_spec.constellation.name)
        self.assertEqual(baseline["design_id"], fixed_spec.constellation.name)
        self.assertEqual(len(baseline["digest"]), 64)

        # Architecture and cost are complete common-evaluator outputs, so every
        # field must agree rather than only the historical 600-Gbps total.
        self.assertEqual(baseline["constellation_architecture"], fixed["constellation_architecture"])
        self.assertEqual(baseline["cost"], fixed["cost"])

        # DSE adds a static-SLO projection while fixed simulation may add a
        # scheduler optimization.  Compare their shared serving and current-
        # runtime evaluation before those caller-specific post-processing steps.
        for field in (
            "prefill_tokens_per_s",
            "decode_tokens_per_s",
            "request_throughput_per_s",
            "token_throughput_per_s",
        ):
            self.assertEqual(baseline["performance"][field], fixed["performance"][field])
        baseline_runtime = json.loads(
            json.dumps(_runtime_result_summary(baseline["performance"]["details"]["runtime"]))
        )
        fixed_runtime = fixed["performance"]["details"]["runtime_initial"]
        self.assertEqual(_without_warnings(baseline_runtime), _without_warnings(fixed_runtime))
        self.assertEqual(
            _without_warnings(baseline["performance"]["details"]["runtime_current_strategy"]),
            _without_warnings(fixed["performance"]["details"]["runtime_current_strategy"]),
        )


    def test_duplicate_candidate_names_get_unique_rows_traces_and_runtime_matrix_ids(self):
        base = load_experiment_spec(MANIFEST)
        first_payload = base.constellation.to_dict()
        first_payload["name"] = "same_candidate_name"
        second_payload = json.loads(json.dumps(first_payload))
        second_payload["orbital_sets"][0]["altitude_km"] = 600

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                base,
                simulation=replace(base.simulation, trace_output_dir=tmpdir),
                dse=DSESpec(
                    enabled=True,
                    design_candidates=(
                        {"candidate_id": "candidate_a", "constellation": first_payload},
                        {"candidate_id": "candidate_b", "constellation": second_payload},
                    ),
                    sweep={},
                    runtime_matrix=base.dse.runtime_matrix,
                ),
            )
            study = run_dse(spec).to_dict()
            materialized_trace_dirs = {
                path.name
                for path in Path(tmpdir).iterdir()
                if path.is_dir() and (path / "link_trace.json").exists()
            }

        design_ids = [row["design_id"] for row in study["rows"]]
        self.assertEqual(len(design_ids), 2)
        self.assertEqual(len(set(design_ids)), 2)
        self.assertEqual(len({row["digest"] for row in study["rows"]}), 2)
        self.assertEqual(
            {
                row["design"]["dse_candidate"]["candidate_id"]
                for row in study["rows"]
            },
            {"candidate_a", "candidate_b"},
        )
        self.assertEqual(materialized_trace_dirs, set(design_ids))
        matrix_design_ids = {
            row["design_id"] for row in study["runtime_matrix"]["matrix_rows"]
        }
        self.assertEqual(matrix_design_ids, set(design_ids))
        for design_id in design_ids:
            self.assertTrue(
                any(
                    row["design_id"] == design_id
                    for row in study["runtime_matrix"]["matrix_rows"]
                )
            )

    def test_profile_catalog_identifier_is_not_a_searchable_physical_parameter(self):
        spec = _skeleton_spec(
            dse=DSESpec(
                enabled=True,
                sweep={"same_plane_profile_id": ("old_catalog_entry",)},
            )
        )
        with self.assertRaisesRegex(ValueError, r"unsupported DSE sweep keys.*same_plane_profile_id"):
            run_dse(spec)

    def test_unscoped_sweeps_reject_multi_orbital_set_designs(self):
        spec = _skeleton_spec(dse=DSESpec(enabled=False))
        first = spec.constellation.orbital_sets[0]
        multi_set_design = replace(
            spec.constellation,
            orbital_sets=(first, replace(first, name=f"{first.name}_second")),
        )
        sweeps = {
            "altitude_km": (600,),
            "satellites_per_cluster": (4,),
        }
        for key, values in sweeps.items():
            with self.subTest(key=key):
                ambiguous = replace(
                    spec,
                    constellation=multi_set_design,
                    dse=DSESpec(enabled=True, sweep={key: values}),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    rf"2 orbital sets.*unscoped orbital/cluster sweep keys.*{key}",
                ):
                    run_dse(ambiguous)

    def test_default_and_dse_trace_directories_are_safe_and_isolated(self):
        base = load_experiment_spec(MANIFEST)
        simulation = replace(base.simulation, trace_output_dir=None)
        dangerous_experiment_id = "../unsafe/study"
        dangerous_design_id = "../../candidate\\escape"

        fixed = _simulation_config_with_default_trace_dir(
            simulation,
            evaluation_id=dangerous_experiment_id,
        )
        dse = _simulation_config_for_dse_design(
            simulation,
            design_id=dangerous_design_id,
            experiment_id=dangerous_experiment_id,
        )
        fixed_path = Path(fixed.trace_output_dir)
        dse_path = Path(dse.trace_output_dir)

        self.assertNotEqual(fixed_path, dse_path)
        self.assertEqual(dse_path.parent.name, "dse")
        self.assertNotIn("..", fixed_path.parts)
        self.assertNotIn("..", dse_path.parts)
        self.assertNotIn("/", dse_path.name)
        self.assertNotIn("\\", dse_path.name)
        self.assertIn("hash-", fixed_path.name)
        self.assertIn("hash-", dse_path.name)
        multibyte_component = safe_trace_path_component("候选" * 100)
        self.assertLessEqual(len(multibyte_component.encode("utf-8")), 120)


if __name__ == "__main__":
    unittest.main()
