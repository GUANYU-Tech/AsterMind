import sys
import unittest
from pathlib import Path


from orbitserve.runtime.state.energy_state import EnergyWindowError, advance_satellite_energy, apply_request_energy, energy_window_at
from orbitserve.runtime.state.runtime_state import SatelliteState


def _window(start_s, end_s, **overrides):
    row = {
        "satellite_id": "sat-a",
        "start_s": float(start_s),
        "end_s": float(end_s),
        "sunlit": True,
        "dvfs_scale": 1.0,
        "charge_rate_w": 0.0,
        "idle_power_w": 0.0,
        "compute_energy_j_per_token": 1.0,
        "link_energy_j_per_byte": 1.0,
        "battery_capacity_j": 100.0,
        "soc_min": 0.2,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "energy_accounting_mode": "event_incremental_v1",
    }
    row.update(overrides)
    return row


class _Bundle:
    def __init__(self, windows):
        self.energy_windows = tuple(windows)


class EnergyContractCornerCaseTests(unittest.TestCase):
    def test_charge_and_discharge_efficiencies_apply_once(self):
        bundle = _Bundle(
            (
                _window(
                    0,
                    10,
                    charge_rate_w=2.0,
                    idle_power_w=1.0,
                    battery_charge_efficiency=0.5,
                ),
                _window(
                    10,
                    20,
                    sunlit=False,
                    charge_rate_w=0.0,
                    idle_power_w=1.0,
                    battery_discharge_efficiency=0.5,
                ),
            )
        )
        satellite = SatelliteState("sat-a", soc=0.5, initial_soc=0.5)

        advance_satellite_energy(satellite, bundle, 10.0)
        self.assertAlmostEqual(satellite.soc, 0.55)
        advance_satellite_energy(satellite, bundle, 20.0)
        self.assertAlmostEqual(satellite.soc, 0.35)

    def test_soc_saturation_preserves_curtailed_and_unserved_energy(self):
        full_bundle = _Bundle((_window(0, 10, charge_rate_w=10.0),))
        full = SatelliteState("sat-a", soc=0.95, initial_soc=0.95)
        advance_satellite_energy(full, full_bundle, 10.0)
        self.assertEqual(full.soc, 1.0)
        self.assertAlmostEqual(full.energy_ledger[-1]["curtailed_energy_j"], 95.0)

        empty_bundle = _Bundle((_window(0, 10, idle_power_w=10.0),))
        empty = SatelliteState("sat-a", soc=0.05, initial_soc=0.05)
        advance_satellite_energy(empty, empty_bundle, 10.0)
        self.assertEqual(empty.soc, 0.0)
        self.assertAlmostEqual(empty.energy_ledger[-1]["unserved_energy_j"], 95.0)

    def test_charge_efficiency_only_applies_to_energy_accepted_by_battery(self):
        bundle = _Bundle(
            (_window(0, 10, charge_rate_w=10.0, battery_charge_efficiency=0.5),)
        )
        full = SatelliteState("sat-a", soc=1.0, initial_soc=1.0)
        advance_satellite_energy(full, bundle, 10.0)
        row = full.energy_ledger[-1]
        self.assertEqual(row["battery_charge_input_bus_j"], 0.0)
        self.assertEqual(row["conversion_loss_j"], 0.0)
        self.assertEqual(row["curtailed_energy_j"], 100.0)

        nearly_full = SatelliteState("sat-a", soc=0.9, initial_soc=0.9)
        advance_satellite_energy(nearly_full, bundle, 10.0)
        row = nearly_full.energy_ledger[-1]
        self.assertAlmostEqual(nearly_full.soc, 1.0)
        self.assertAlmostEqual(row["battery_charge_input_bus_j"], 20.0)
        self.assertAlmostEqual(row["conversion_loss_j"], 10.0)
        self.assertAlmostEqual(row["curtailed_energy_j"], 80.0)

    def test_out_of_order_event_insertion_replays_chronologically(self):
        bundle = _Bundle((_window(0, 200),))
        satellite = SatelliteState("sat-a", soc=0.8, initial_soc=0.8)

        advance_satellite_energy(satellite, bundle, 100.0)
        apply_request_energy(
            satellite,
            prompt_tokens=10,
            output_tokens=0,
            link_bytes=0.0,
            energy_window=energy_window_at(bundle, "sat-a", 100.0),
            event_time_s=100.0,
        )
        self.assertAlmostEqual(satellite.soc, 0.7)

        advance_satellite_energy(satellite, bundle, 10.0)
        self.assertAlmostEqual(satellite.soc, 0.8)
        apply_request_energy(
            satellite,
            prompt_tokens=20,
            output_tokens=0,
            link_bytes=0.0,
            energy_window=energy_window_at(bundle, "sat-a", 10.0),
            event_time_s=10.0,
        )
        advance_satellite_energy(satellite, bundle, 100.0)
        self.assertAlmostEqual(satellite.soc, 0.5)

    def test_trace_gap_failure_is_atomic(self):
        bundle = _Bundle((_window(0, 5, idle_power_w=1.0), _window(6, 10, idle_power_w=1.0)))
        satellite = SatelliteState("sat-a", soc=0.8, initial_soc=0.8)

        with self.assertRaises(EnergyWindowError):
            advance_satellite_energy(satellite, bundle, 10.0)

        self.assertEqual(satellite.soc, 0.8)
        self.assertEqual(satellite.last_energy_update_s, 0.0)
        self.assertEqual(satellite.energy_ledger, ())

    def test_invalid_energy_values_fail_closed(self):
        for field, value in (
            ("battery_capacity_j", 0.0),
            ("charge_rate_w", -1.0),
            ("idle_power_w", float("nan")),
            ("compute_energy_j_per_token", float("inf")),
            ("soc_min", 1.1),
            ("dvfs_scale", 0.0),
            ("dvfs_scale", 1.1),
            ("battery_charge_efficiency", 0.0),
            ("battery_discharge_efficiency", 1.1),
            ("energy_accounting_mode", "bogus"),
        ):
            with self.subTest(field=field):
                satellite = SatelliteState("sat-a", soc=0.8, initial_soc=0.8)
                with self.assertRaises(EnergyWindowError):
                    advance_satellite_energy(
                        satellite,
                        _Bundle((_window(0, 10, **{field: value}),)),
                        10.0,
                    )
                self.assertEqual(satellite.soc, 0.8)
                self.assertEqual(satellite.last_energy_update_s, 0.0)

    def test_invalid_event_time_cannot_mutate_or_disappear_on_replay(self):
        bundle = _Bundle((_window(0, 10),))
        satellite = SatelliteState("sat-a", soc=0.5, initial_soc=0.5)
        advance_satellite_energy(satellite, bundle, 0.0)

        with self.assertRaises(EnergyWindowError):
            apply_request_energy(
                satellite,
                prompt_tokens=100,
                output_tokens=0,
                link_bytes=0.0,
                energy_window=energy_window_at(bundle, "sat-a", 0.0),
                event_time_s=float("nan"),
            )

        self.assertEqual(satellite.soc, 0.5)
        self.assertEqual(satellite.energy_events, ())
        advance_satellite_energy(satellite, bundle, 1.0)
        self.assertEqual(satellite.soc, 0.5)

    def test_invalid_request_token_and_byte_values_fail_before_mutation(self):
        bundle = _Bundle((_window(0, 10),))
        cases = (
            ("prompt_tokens", -1.0),
            ("prompt_tokens", float("nan")),
            ("output_tokens", -1.0),
            ("output_tokens", float("inf")),
            ("link_bytes", -1.0),
            ("link_bytes", float("nan")),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                satellite = SatelliteState("sat-a", soc=0.5, initial_soc=0.5)
                arguments = {
                    "prompt_tokens": 1.0,
                    "output_tokens": 1.0,
                    "link_bytes": 1.0,
                }
                arguments[field] = value
                with self.assertRaises(EnergyWindowError) as raised:
                    apply_request_energy(
                        satellite,
                        **arguments,
                        energy_window=energy_window_at(bundle, "sat-a", 0.0),
                        event_time_s=0.0,
                    )

                self.assertEqual(raised.exception.reason, "energy_window_invalid_field")
                self.assertIn(f"invalid:{field}", raised.exception.warnings)
                self.assertEqual(satellite.soc, 0.5)
                self.assertEqual(satellite.energy_events, ())
                self.assertEqual(satellite.energy_ledger, ())

    def test_dynamic_energy_overflow_fails_before_event_is_appended(self):
        bundle = _Bundle((_window(0, 10, compute_energy_j_per_token=1e308),))
        satellite = SatelliteState("sat-a", soc=0.5, initial_soc=0.5)

        with self.assertRaises(EnergyWindowError):
            apply_request_energy(
                satellite,
                prompt_tokens=10,
                output_tokens=0,
                link_bytes=0.0,
                energy_window=energy_window_at(bundle, "sat-a", 0.0),
                event_time_s=0.0,
            )

        self.assertEqual(satellite.soc, 0.5)
        self.assertEqual(satellite.energy_events, ())
        self.assertEqual(satellite.energy_ledger, ())

    def test_direct_request_reports_unserved_as_energy_violation(self):
        bundle = _Bundle((_window(0, 10, soc_min=0.0),))
        satellite = SatelliteState("sat-a", soc=0.5, initial_soc=0.5)
        advance_satellite_energy(satellite, bundle, 0.0)
        update = apply_request_energy(
            satellite,
            prompt_tokens=100,
            output_tokens=0,
            link_bytes=0.0,
            energy_window=energy_window_at(bundle, "sat-a", 0.0),
            event_time_s=0.0,
        )
        self.assertTrue(update["energy_violation"])
        self.assertAlmostEqual(update["unserved_energy_j"], 50.0)

    def test_efficiency_ledger_exposes_conversion_loss_in_bus_units(self):
        bundle = _Bundle(
            (_window(0, 10, idle_power_w=1.0, battery_discharge_efficiency=0.5),)
        )
        satellite = SatelliteState("sat-a", soc=1.0, initial_soc=1.0)
        advance_satellite_energy(satellite, bundle, 10.0)
        row = satellite.energy_ledger[-1]
        self.assertAlmostEqual(row["baseline_bus_j"], 10.0)
        self.assertAlmostEqual(row["battery_discharge_output_bus_j"], 10.0)
        self.assertAlmostEqual(row["conversion_loss_j"], 10.0)
        self.assertAlmostEqual(row["battery_delta_j"], -20.0)


if __name__ == "__main__":
    unittest.main()
