import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from orbitserve.experiments.constellation_cost import estimate_constellation_cost
from orbitserve.experiments.design_evaluation import _runtime_sunlight_summary
from orbitserve.runtime.energy_state import REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.architecture.hardware import SatelliteHardwareProfile
from orbitserve.architecture.hardware import resolve_experiment_hardware
from orbitserve.runtime.deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog
from orbitserve.runtime.runtime_energy_envelope import derive_runtime_energy_envelope
from orbitserve.runtime.engines.trace_backed import run_trace_backed_runtime
from orbitserve.runtime.contracts import RuntimeStrategySpec
from orbitserve.orbit.simulation_trace_builder import build_surrogate_orbit_trace
from orbitserve.orbit.simulation_traces import OrbitTraceManifest, load_surrogate_trace_bundle


ROOT = Path(__file__).resolve().parents[3]
SURROGATE_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json"
CIRCULAR_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_trace.json"
MATRIX_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_runtime_matrix.json"


class RuntimeEnergyEnvelopeTests(unittest.TestCase):
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
        envelope = derive_runtime_energy_envelope(
            cost,
            resolved_hardware=resolved,
            base_trace_options={**spec.simulation.trace_options, "battery_capacity_j": 1.0},
        )

        single = cost.single_satellite or {}
        self.assertTrue(envelope.evaluated)
        self.assertAlmostEqual(
            envelope.trace_options["battery_capacity_j"],
            single["E_battery_req_wh"] * 3600.0 * resolved.cost_config.power.battery_eol_fraction,
        )
        self.assertAlmostEqual(
            envelope.trace_options["charge_rate_w"],
            single["P_solar_req_w"]
            * resolved.cost_config.power.eta_pmad
            * resolved.cost_config.power.pv_eol_fraction,
        )
        expected_baseline_w = (
            resolved.cost_config.platform.power_w
            + resolved.cost_config.power.compute_standby_power_w
            + resolved.cost_config.power.communication_standby_power_w
        )
        self.assertAlmostEqual(envelope.trace_options["idle_power_w_sunlit"], expected_baseline_w)
        self.assertAlmostEqual(envelope.trace_options["idle_power_w_shadow"], expected_baseline_w)
        self.assertEqual(envelope.trace_options["energy_accounting_mode"], "event_incremental_v1")
        self.assertAlmostEqual(envelope.trace_options["soc_min"], 1.0 - resolved.cost_config.power.dod_max)
        self.assertAlmostEqual(
            envelope.trace_options["initial_soc"],
            spec.simulation.trace_options.get("initial_soc", 0.8),
        )
        self.assertAlmostEqual(
            envelope.trace_options["shadow_fraction"],
            1.0 - cost.single_satellite["provenance"]["orbit_energy_condition"]["sunlit_fraction"],
        )
        self.assertAlmostEqual(
            envelope.trace_options["compute_energy_j_per_token"],
            resolved.cost_config.power.compute_energy_j_per_token,
        )
        self.assertAlmostEqual(
            envelope.trace_options["link_energy_j_per_byte"],
            resolved.cost_config.power.link_energy_j_per_byte,
        )
        joined_warnings = " ".join(envelope.warnings)
        self.assertIn("overridden by cost/hardware runtime energy envelope", joined_warnings)
        self.assertIn("baseline-exclusive and forced from cost/hardware", joined_warnings)

    def test_periodic_sunlight_summary_uses_materialized_trace_period(self):
        trace_period_s = 7622.0
        trace_bundle = SimpleNamespace(
            energy_windows=(
                {
                    "satellite_id": "sat-a",
                    "start_s": 0.0,
                    "end_s": trace_period_s,
                    "sunlit": True,
                },
            ),
            energy_trace={"provenance": {"orbit_period_s": trace_period_s}},
        )
        simulation = SimpleNamespace(
            trace_model="surrogate_orbit_trace",
            trace_evaluated=True,
            trace_manifest={"trace_id": "period-authority-test"},
        )
        envelope = {
            "trace_options": {"shadow_fraction": 0.36},
            "provenance": {"runtime_fields": {"orbit_period_s": 5700.0}},
        }

        with patch(
            "orbitserve.experiments.design_evaluation.load_surrogate_trace_bundle",
            return_value=trace_bundle,
        ):
            summary, warnings = _runtime_sunlight_summary(
                simulation,
                runtime_energy_envelope=envelope,
                base_dir=None,
            )

        self.assertEqual(warnings, ())
        self.assertAlmostEqual(summary["periodic_orbit_period_s"], trace_period_s)
        self.assertAlmostEqual(
            summary["maximum_shadow_window_s"],
            0.36 * trace_period_s,
        )
        self.assertEqual(
            summary["measurement_basis"],
            "materialized_periodic_shadow_contract",
        )

    def test_adapter_rejects_nonfinite_cost_and_hardware_contract_values(self):
        spec, resolved, cost = self._cost_and_hardware()
        payload = cost.to_dict()
        payload["single_satellite"]["E_battery_req_wh"] = float("nan")
        payload["single_satellite"]["E_battery_req"] = float("nan")
        self.assertFalse(
            derive_runtime_energy_envelope(payload, resolved_hardware=resolved).evaluated
        )

        bad_power = replace(resolved.cost_config.power, eta_pmad=float("nan"))
        bad_cost_config = replace(resolved.cost_config, power=bad_power)
        bad_resolved = replace(resolved, cost_config=bad_cost_config)
        envelope = derive_runtime_energy_envelope(cost, resolved_hardware=bad_resolved)
        self.assertFalse(envelope.evaluated)
        self.assertIn("eta_pmad", envelope.provenance["invalid_contract_fields"])

        envelope = derive_runtime_energy_envelope(
            cost,
            resolved_hardware=resolved,
            base_trace_options={**spec.simulation.trace_options, "orbit_period_s": float("nan")},
        )
        self.assertFalse(envelope.evaluated)
        self.assertIn("orbit_period_s", envelope.provenance["invalid_contract_fields"])


    def test_runner_surrogate_trace_uses_cost_envelope_over_poisoned_trace_options(self):
        spec = load_experiment_spec(SURROGATE_MANIFEST)
        poison = {
            **spec.simulation.trace_options,
            "battery_capacity_j": 1.0,
            "charge_rate_w": 2.0,
            "idle_power_w": 3.0,
            "idle_power_w_sunlit": 4.0,
            "idle_power_w_shadow": 5.0,
            "compute_energy_j_per_token": 999.0,
            "link_energy_j_per_byte": 999.0,
            "soc_min": 0.99,
            "shadow_fraction": 0.01,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            # This regression checks runner-side envelope authority, not the
            # multi-policy optimizer.  Once the poisoned 999 J coefficients
            # are correctly overwritten, the full optimizer has many more
            # feasible request candidates and makes this focused test
            # needlessly expensive.
            spec = replace(
                spec,
                scheduler=replace(spec.scheduler, runtime_optimization_enabled=False),
                simulation=replace(
                    spec.simulation,
                    trace_output_dir=tmpdir,
                    trace_options=poison,
                ),
            )
            payload = run_experiment(spec).to_dict()

            manifest = payload["performance"]["details"]["runtime"]["provenance"]["trace_manifest"]
            trace_dir = Path(tmpdir)
            energy_path = trace_dir / manifest["energy_trace_path"]
            energy = json.loads(energy_path.read_text(encoding="utf-8"))

        single = payload["cost"]["single_satellite"]
        envelope = energy["soc_inputs"]["runtime_energy_envelope"]
        runtime_fields = envelope["provenance"]["runtime_fields"]
        expected_capacity_j = runtime_fields["battery_effective_eol_j"]
        expected_charge_w = runtime_fields["solar_bus_available_eol_w"]
        expected_baseline_w = runtime_fields["always_on_baseline_power_w"]
        sunlit = next(window for window in energy["energy_windows"] if window["sunlit"])
        shadow = next(window for window in energy["energy_windows"] if not window["sunlit"])

        self.assertTrue(envelope["evaluated"])
        self.assertEqual(envelope["provenance"]["source"], "cost_hardware_orbit_runtime")
        self.assertAlmostEqual(energy["soc_inputs"]["battery_capacity_j"], expected_capacity_j)
        self.assertAlmostEqual(energy["soc_inputs"]["charge_rate_w"], expected_charge_w)
        self.assertAlmostEqual(energy["soc_inputs"]["idle_power_w_sunlit"], expected_baseline_w)
        self.assertAlmostEqual(energy["soc_inputs"]["idle_power_w_shadow"], expected_baseline_w)
        self.assertNotEqual(energy["soc_inputs"]["battery_capacity_j"], 1.0)
        self.assertNotEqual(energy["soc_inputs"]["charge_rate_w"], 2.0)
        self.assertNotEqual(energy["soc_inputs"]["soc_min"], 0.99)
        self.assertNotEqual(energy["soc_inputs"]["idle_power_w_sunlit"], 4.0)
        self.assertNotEqual(energy["soc_inputs"]["idle_power_w_shadow"], 5.0)
        self.assertAlmostEqual(
            energy["soc_inputs"]["compute_energy_j_per_token"],
            runtime_fields["compute_energy_j_per_token"],
        )
        self.assertAlmostEqual(
            energy["soc_inputs"]["link_energy_j_per_byte"],
            runtime_fields["link_energy_j_per_byte"],
        )
        self.assertNotEqual(energy["soc_inputs"]["compute_energy_j_per_token"], 999.0)
        self.assertNotEqual(energy["soc_inputs"]["link_energy_j_per_byte"], 999.0)
        self.assertAlmostEqual(
            energy["provenance"]["trace_options"]["shadow_fraction"],
            1.0 - runtime_fields["sunlit_fraction"],
        )
        self.assertNotEqual(energy["provenance"]["trace_options"]["shadow_fraction"], 0.01)
        self.assertAlmostEqual(sunlit["battery_capacity_j"], expected_capacity_j)
        self.assertAlmostEqual(sunlit["idle_power_w"], expected_baseline_w)
        self.assertAlmostEqual(shadow["idle_power_w"], expected_baseline_w)
        self.assertEqual(sunlit["runtime_energy_envelope"]["provenance"]["source"], "cost_hardware_orbit_runtime")
        self.assertIn("simulation.trace_options.battery_capacity_j overridden", " ".join(payload["warnings"]))
        self.assertIn("simulation.trace_options.charge_rate_w overridden", " ".join(payload["warnings"]))
        self.assertIn("simulation.trace_options.idle_power_w_sunlit overridden", " ".join(payload["warnings"]))
        self.assertIn("simulation.trace_options.shadow_fraction overridden", " ".join(payload["warnings"]))
        self.assertIn("simulation.trace_options.compute_energy_j_per_token overridden", " ".join(payload["warnings"]))
        self.assertIn("simulation.trace_options.link_energy_j_per_byte overridden", " ".join(payload["warnings"]))

    def test_runner_circular_trace_energy_windows_use_cost_envelope(self):
        spec = load_experiment_spec(CIRCULAR_MANIFEST)
        poison = {
            **spec.simulation.trace_options,
            "battery_capacity_j": 1.0,
            "charge_rate_w": 2.0,
            "idle_power_w": 3.0,
            "idle_power_w_sunlit": 4.0,
            "idle_power_w_shadow": 5.0,
            "soc_min": 0.99,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(spec, simulation=replace(spec.simulation, trace_output_dir=tmpdir, trace_options=poison))
            payload = run_experiment(spec).to_dict()

            manifest = payload["performance"]["details"]["runtime"]["provenance"]["trace_manifest"]
            energy = json.loads((Path(tmpdir) / manifest["energy_trace_path"]).read_text(encoding="utf-8"))

        single = payload["cost"]["single_satellite"]
        windows = energy["energy_windows"]
        sunlit = next(window for window in windows if window["sunlit"])
        shadow = next(window for window in windows if not window["sunlit"])

        self.assertEqual(energy["soc_inputs"]["runtime_energy_envelope"]["provenance"]["source"], "cost_hardware_orbit_runtime")
        runtime_fields = energy["soc_inputs"]["runtime_energy_envelope"]["provenance"]["runtime_fields"]
        self.assertAlmostEqual(sunlit["battery_capacity_j"], runtime_fields["battery_effective_eol_j"])
        self.assertAlmostEqual(sunlit["charge_rate_w"], runtime_fields["solar_bus_available_eol_w"])
        self.assertAlmostEqual(sunlit["idle_power_w"], runtime_fields["always_on_baseline_power_w"])
        self.assertAlmostEqual(shadow["charge_rate_w"], 0.0)
        self.assertAlmostEqual(shadow["idle_power_w"], runtime_fields["always_on_baseline_power_w"])
        self.assertNotEqual(sunlit["battery_capacity_j"], 1.0)
        self.assertNotEqual(sunlit["charge_rate_w"], 2.0)
        self.assertNotEqual(sunlit["idle_power_w"], 4.0)
        self.assertNotEqual(shadow["idle_power_w"], 5.0)


    def test_dse_circular_trace_rows_use_cost_envelope_over_poisoned_trace_options(self):
        spec = load_experiment_spec(MATRIX_MANIFEST)
        poison = {
            **spec.simulation.trace_options,
            "battery_capacity_j": 1.0,
            "charge_rate_w": 2.0,
            "idle_power_w": 3.0,
            "idle_power_w_sunlit": 4.0,
            "idle_power_w_shadow": 5.0,
            "soc_min": 0.99,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                spec,
                simulation=replace(spec.simulation, trace_output_dir=tmpdir, trace_options=poison),
                dse=replace(spec.dse, enabled=True, sweep={"altitude_km": (550,)}),
            )
            payload = run_experiment(spec).to_dict()
            row = payload["dse"]["rows"][0]
            row_runtime = row["performance"]["details"]["runtime"]
            manifest = row_runtime["provenance"]["trace_manifest"]
            energy = json.loads(
                (Path(tmpdir) / row["design_id"] / manifest["energy_trace_path"]).read_text(encoding="utf-8")
            )

        single = row["cost"]["single_satellite"]
        sunlit = next(window for window in energy["energy_windows"] if window["sunlit"])
        shadow = next(window for window in energy["energy_windows"] if not window["sunlit"])

        self.assertTrue(row_runtime["evaluated"])
        self.assertEqual(energy["soc_inputs"]["runtime_energy_envelope"]["provenance"]["source"], "cost_hardware_orbit_runtime")
        runtime_fields = energy["soc_inputs"]["runtime_energy_envelope"]["provenance"]["runtime_fields"]
        self.assertAlmostEqual(sunlit["battery_capacity_j"], runtime_fields["battery_effective_eol_j"])
        self.assertAlmostEqual(sunlit["charge_rate_w"], runtime_fields["solar_bus_available_eol_w"])
        self.assertAlmostEqual(sunlit["idle_power_w"], runtime_fields["always_on_baseline_power_w"])
        self.assertAlmostEqual(shadow["idle_power_w"], runtime_fields["always_on_baseline_power_w"])
        self.assertNotEqual(sunlit["battery_capacity_j"], 1.0)
        self.assertNotEqual(sunlit["charge_rate_w"], 2.0)
        self.assertTrue(payload["dse"]["runtime_matrix"]["evaluated"])

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
            bundle = load_surrogate_trace_bundle(OrbitTraceManifest.from_value(result.to_trace_manifest()))
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
            deployment_catalog=RuntimeDeploymentCatalog(
                deployments=(
                    RuntimeDeployment(
                        deployment_id="deployment_0",
                        prefill_tokens_per_s=1_000_000.0,
                        decode_tokens_per_s=1_000_000.0,
                    ),
                ),
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
            trace_bundle=bundle,
        )

        self.assertTrue(runtime.evaluated)
        self.assertEqual(runtime.records[0].reject_reason, "energy_window_missing_field")
        self.assertIn("missing:battery_capacity_j", runtime.records[0].provenance["warnings"])
        self.assertIn("battery_capacity_j", REQUIRED_RUNTIME_ENERGY_WINDOW_FIELDS)

if __name__ == "__main__":
    unittest.main()
