import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments.constellation_cost import estimate_constellation_cost
from orbitserve.runtime.state.energy_state import REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.architecture.hardware import SatelliteHardwareProfile
from orbitserve.architecture.hardware import resolve_experiment_hardware
from orbitserve.runtime import ServingDeploymentView
from orbitserve.runtime.scenario.runtime_energy_profile import derive_runtime_energy_profile
from orbitserve.runtime import build_runtime_scenario
from orbitserve.runtime.engine.trace_backed import run_trace_backed_runtime
from orbitserve.runtime.contracts import RuntimeStrategySpec
from orbitserve.orbit.simulation_trace_builder import build_surrogate_orbit_trace
from orbitserve.orbit.simulation_traces import RuntimeTraceManifest, load_surrogate_trace_bundle


ROOT = Path(__file__).resolve().parents[3]
SURROGATE_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json"
CIRCULAR_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_trace.json"
MATRIX_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_runtime_matrix.json"


class RuntimeEnergyProfileTests(unittest.TestCase):
    def _cost_and_hardware(self, manifest_path=SURROGATE_MANIFEST):
        spec = load_experiment_spec(manifest_path)
        base_dir = Path(spec.base_dir) if spec.base_dir else None
        resolved = resolve_experiment_hardware(
            spec.hardware,
            base_dir=base_dir,
        )
        cost = estimate_constellation_cost(
            spec.constellation,
            spec.costing,
            base_dir=base_dir,
            cost_limits=resolved.cost_limits,
            cost_config=resolved.cost_config,
            single_satellite_case_path=resolved.single_satellite_case_path,
        )
        return spec, resolved, cost

    def test_adapter_derives_runtime_energy_fields_from_cost_and_hardware(self):
        spec, resolved, cost = self._cost_and_hardware()
        profile = derive_runtime_energy_profile(
            cost,
            resolved_hardware=resolved,
            base_trace_options={**spec.simulation.trace_options, "battery_capacity_j": 1.0},
        )

        single = cost.single_satellite or {}
        self.assertTrue(profile.evaluated)
        self.assertAlmostEqual(
            profile.trace_options["battery_capacity_j"],
            single["E_battery_req_wh"] * 3600.0 * resolved.cost_config.power.battery_eol_fraction,
        )
        self.assertAlmostEqual(
            profile.trace_options["charge_rate_w"],
            single["P_solar_req_w"]
            * resolved.cost_config.power.eta_pmad
            * resolved.cost_config.power.pv_eol_fraction,
        )
        expected_baseline_w = (
            resolved.cost_config.platform.power_w
            + resolved.cost_config.power.compute_standby_power_w
            + resolved.cost_config.power.communication_standby_power_w
        )
        self.assertAlmostEqual(profile.trace_options["idle_power_w_sunlit"], expected_baseline_w)
        self.assertAlmostEqual(profile.trace_options["idle_power_w_shadow"], expected_baseline_w)
        self.assertEqual(profile.trace_options["energy_accounting_mode"], "event_incremental_v1")
        self.assertAlmostEqual(profile.trace_options["soc_min"], 1.0 - resolved.cost_config.power.dod_max)
        self.assertAlmostEqual(
            profile.trace_options["initial_soc"],
            spec.simulation.trace_options.get("initial_soc", 0.8),
        )
        self.assertAlmostEqual(
            profile.trace_options["shadow_fraction"],
            1.0 - cost.single_satellite["provenance"]["orbit_energy_condition"]["sunlit_fraction"],
        )
        self.assertAlmostEqual(
            profile.trace_options["compute_energy_j_per_token"],
            resolved.cost_config.power.compute_energy_j_per_token,
        )
        self.assertAlmostEqual(
            profile.trace_options["link_energy_j_per_byte"],
            resolved.cost_config.power.link_energy_j_per_byte,
        )
        joined_warnings = " ".join(profile.warnings)
        self.assertIn("overridden by cost/hardware runtime energy profile", joined_warnings)
        self.assertIn("baseline-exclusive and forced from cost/hardware", joined_warnings)

    def test_adapter_rejects_nonfinite_cost_and_hardware_contract_values(self):
        spec, resolved, cost = self._cost_and_hardware()
        payload = cost.to_dict()
        payload["single_satellite"]["E_battery_req_wh"] = float("nan")
        payload["single_satellite"]["E_battery_req"] = float("nan")
        self.assertFalse(
            derive_runtime_energy_profile(payload, resolved_hardware=resolved).evaluated
        )

        bad_power = replace(resolved.cost_config.power, eta_pmad=float("nan"))
        bad_cost_config = replace(resolved.cost_config, power=bad_power)
        bad_resolved = replace(resolved, cost_config=bad_cost_config)
        profile = derive_runtime_energy_profile(cost, resolved_hardware=bad_resolved)
        self.assertFalse(profile.evaluated)
        self.assertIn("eta_pmad", profile.provenance["invalid_contract_fields"])

        profile = derive_runtime_energy_profile(
            cost,
            resolved_hardware=resolved,
            base_trace_options={**spec.simulation.trace_options, "orbit_period_s": float("nan")},
        )
        self.assertFalse(profile.evaluated)
        self.assertIn("orbit_period_s", profile.provenance["invalid_contract_fields"])


    def test_runtime_scenario_owns_static_energy_profile(self):
        spec, resolved, cost = self._cost_and_hardware()
        scenario = build_runtime_scenario(
            scenario_id="energy-profile",
            simulation_config=spec.simulation,
            cost_result=cost,
            resolved_hardware=resolved,
        )

        profile = scenario.energy_profile
        runtime_fields = profile["provenance"]["runtime_fields"]
        self.assertTrue(profile["evaluated"])
        self.assertAlmostEqual(
            profile["trace_options"]["battery_capacity_j"],
            runtime_fields["battery_effective_eol_j"],
        )
        self.assertAlmostEqual(
            profile["trace_options"]["charge_rate_w"],
            runtime_fields["solar_bus_available_eol_w"],
        )
        self.assertEqual(
            profile["trace_options"]["energy_accounting_mode"],
            "event_incremental_v1",
        )

    def test_runtime_rejects_energy_window_missing_required_fields(self):
        spec = load_experiment_spec(SURROGATE_MANIFEST)
        workload = {
            "requests": (
                {
                    "request_id": "req-missing-energy",
                    "arrival_time_s": 0.0,
                    "ingress_id": "ingress",
                    "prompt_tokens": 1,
                    "output_tokens": 1,
                    "input_bytes": 1.0,
                    "output_bytes": 1.0,
                },
            ),
            "gateways": [{"gateway_id": "ingress"}],
            "regions": (),
            "manifest": {"schema_version": "orbitserve.workload"},
            "summary": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(spec.simulation, trace_output_dir=tmpdir)
            result = build_surrogate_orbit_trace(
                config=config,
                constellation=spec.constellation,
                workload_generation=workload,
                base_dir=ROOT,
                experiment_id="missing_energy_field",
            )
            bundle = load_surrogate_trace_bundle(RuntimeTraceManifest.from_value(result.to_trace_manifest()))
        object.__setattr__(
            bundle,
            "energy_windows",
            tuple(
                {key: value for key, value in window.items() if key != "battery_capacity_j"}
                for window in bundle.energy_windows
            ),
        )

        runtime = run_trace_backed_runtime(
            requests=workload["requests"],
            serving_profile=ServingDeploymentView(
                deployment_id="deployment_0",
                prefill_tokens_per_s=1_000_000.0,
                decode_tokens_per_s=1_000_000.0,
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
            trace_bundle=bundle,
        )

        self.assertTrue(runtime.evaluated)
        self.assertEqual(runtime.records[0].uncompleted_reason, "energy_unavailable")
        self.assertIn("missing:battery_capacity_j", runtime.records[0].provenance["warnings"])
        self.assertIn("battery_capacity_j", REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS)

if __name__ == "__main__":
    unittest.main()
