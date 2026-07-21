import sys
import unittest
from pathlib import Path


from orbitserve.runtime.state.energy_state import apply_request_energy
from orbitserve.runtime.state.energy_accounting import EnergyIntensity, compute_dynamic_energy_j
from orbitserve.runtime.metrics.energy_alignment import (
    attach_energy_alignment,
    evaluate_energy_alignment,
    summarize_runtime_sunlight,
)
from orbitserve.runtime.state.runtime_state import SatelliteState
from orbitserve.runtime.engine.trace_backed import _energy_window_for_runtime_action


def _window(**overrides):
    row = {
        "satellite_id": "sat-a",
        "start_s": 0.0,
        "end_s": 10.0,
        "sunlit": False,
        "dvfs_scale": 1.0,
        "charge_rate_w": 0.0,
        "idle_power_w": 0.0,
        "compute_energy_j_per_token": 2.0,
        "link_energy_j_per_byte": 0.25,
        "battery_capacity_j": 1_000.0,
        "soc_min": 0.0,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "energy_accounting_mode": "event_incremental_v1",
    }
    row.update(overrides)
    return row


class EnergyAccountingAlignmentTests(unittest.TestCase):
    def test_shadow_dvfs_scales_capacity_without_double_scaling_token_energy(self):
        strategy = type("Strategy", (), {"hard_gate_overrides": {}, "dvfs_modes_enabled": ("shadow_dvfs",)})()

        adjusted = _energy_window_for_runtime_action(
            _window(),
            runtime_strategy=strategy,
            action_type="shadow_dvfs",
        )

        self.assertEqual(adjusted["dvfs_scale"], 0.5)
        self.assertEqual(adjusted["compute_energy_j_per_token"], 2.0)
        self.assertEqual(adjusted["runtime_compute_energy_scale"], 1.0)

    def test_zero_request_has_zero_dynamic_energy(self):
        intensity = EnergyIntensity(
            compute_energy_j_per_token=2.0,
            link_energy_j_per_byte=0.25,
        )
        breakdown = compute_dynamic_energy_j(
            prompt_tokens=0,
            output_tokens=0,
            link_bytes=0,
            intensity=intensity,
        )

        self.assertEqual(breakdown.compute_j, 0.0)
        self.assertEqual(breakdown.link_j, 0.0)
        self.assertEqual(breakdown.total_j, 0.0)

    def test_zero_workload_alignment_is_a_valid_zero_energy_case(self):
        planned = {
            "offered_request_count": 0,
            "target_served_ratio": 1.0,
            "total_compute_energy_j": 0.0,
            "total_link_energy_j": 0.0,
            "total_dynamic_energy_j": 0.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
        }
        runtime = {
            "evaluated": True,
            "runtime_metrics": {"completion_ratio": None},
            "request_records": (),
        }
        envelope = {
            "trace_options": {
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 1e-6,
            }
        }

        result = evaluate_energy_alignment(
            planned,
            runtime,
            runtime_energy_envelope=envelope,
        )

        self.assertTrue(result.evaluated)
        self.assertTrue(result.consistent)
        self.assertTrue(result.target_served_ratio_met)
        self.assertEqual(result.difference_fraction, 0.0)

    def test_zero_target_with_rejected_offered_requests_is_valid_zero_plan(self):
        planned = {
            "offered_request_count": 2,
            "planned_request_count": 0.0,
            "target_served_ratio": 0.0,
            "total_compute_energy_j": 0.0,
            "total_link_energy_j": 0.0,
            "total_dynamic_energy_j": 0.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
        }
        runtime = {
            "evaluated": True,
            "runtime_metrics": {"completion_ratio": 0.0},
            "request_records": (
                { "completed": False, "energy_ledger": ()},
                { "completed": False, "energy_ledger": ()},
            ),
        }
        envelope = {
            "trace_options": {
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 1e-6,
            }
        }

        result = evaluate_energy_alignment(
            planned,
            runtime,
            runtime_energy_envelope=envelope,
        )

        self.assertTrue(result.evaluated)
        self.assertTrue(result.consistent)
        self.assertFalse(result.requires_resizing)
        self.assertEqual(result.difference_fraction, 0.0)

    def test_token_and_byte_scaling_are_independent(self):
        intensity = EnergyIntensity(
            compute_energy_j_per_token=2.0,
            link_energy_j_per_byte=0.25,
        )
        base = compute_dynamic_energy_j(
            prompt_tokens=3,
            output_tokens=2,
            link_bytes=8,
            intensity=intensity,
        )
        doubled_tokens = compute_dynamic_energy_j(
            prompt_tokens=6,
            output_tokens=4,
            link_bytes=8,
            intensity=intensity,
        )
        doubled_bytes = compute_dynamic_energy_j(
            prompt_tokens=3,
            output_tokens=2,
            link_bytes=16,
            intensity=intensity,
        )

        self.assertEqual(doubled_tokens.compute_j, 2.0 * base.compute_j)
        self.assertEqual(doubled_tokens.link_j, base.link_j)
        self.assertEqual(doubled_bytes.compute_j, base.compute_j)
        self.assertEqual(doubled_bytes.link_j, 2.0 * base.link_j)

    def test_runtime_debit_uses_shared_formula_and_exports_auditable_inputs(self):
        satellite = SatelliteState("sat-a", soc=0.8, initial_soc=0.8)
        window = _window()
        expected = compute_dynamic_energy_j(
            prompt_tokens=3,
            output_tokens=2,
            link_bytes=8,
            intensity=EnergyIntensity(
                compute_energy_j_per_token=window["compute_energy_j_per_token"],
                link_energy_j_per_byte=window["link_energy_j_per_byte"],
            ),
        )

        update = apply_request_energy(
            satellite,
            prompt_tokens=3,
            output_tokens=2,
            link_bytes=8,
            energy_window=window,
            event_time_s=0.0,
            metadata={
                "request_id": "req-alignment",
                # Metadata must not be able to falsify authoritative ledger fields.
                "compute_tokens": 999,
                "compute_bus_j": 999,
            },
        )
        ledger = update["ledger_entry"]

        self.assertEqual(update["compute_j"], expected.compute_j)
        self.assertEqual(update["link_j"], expected.link_j)
        self.assertEqual(update["dynamic_energy_j"], expected.total_j)
        self.assertEqual(ledger["compute_bus_j"], expected.compute_j)
        self.assertEqual(ledger["link_bus_j"], expected.link_j)
        self.assertEqual(ledger["dynamic_bus_j"], expected.total_j)
        self.assertEqual(ledger["compute_tokens"], 5.0)
        self.assertEqual(ledger["link_bytes_accounted"], 8.0)
        self.assertEqual(ledger["compute_energy_j_per_token"], 2.0)
        self.assertEqual(ledger["link_energy_j_per_byte"], 0.25)
        self.assertEqual(ledger["base_compute_energy_j_per_token"], 2.0)
        self.assertEqual(ledger["base_link_energy_j_per_byte"], 0.25)
        self.assertEqual(ledger["runtime_compute_energy_scale"], 1.0)
        self.assertEqual(ledger["runtime_link_energy_scale"], 1.0)

    def test_alignment_audits_runtime_policy_energy_scale(self):
        planned = {
            "offered_request_count": 1,
            "target_served_ratio": 1.0,
            "total_compute_energy_j": 5.0,
            "total_link_energy_j": 0.0,
            "total_dynamic_energy_j": 5.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
        }
        ledger = {
            "energy_kind": "request_event",
            "satellite_id": "sat-a",
            "event_id": "energy-event-0",
            "compute_tokens": 20.0,
            "link_bytes_accounted": 0.0,
            "compute_bus_j": 5.0,
            "link_bus_j": 0.0,
            "compute_energy_j_per_token": 0.25,
            "link_energy_j_per_byte": 1e-6,
            "base_compute_energy_j_per_token": 0.5,
            "base_link_energy_j_per_byte": 1e-6,
            "runtime_compute_energy_scale": 0.5,
            "runtime_link_energy_scale": 1.0,
        }
        runtime = {
            "evaluated": True,
            "runtime_metrics": {"completion_ratio": 1.0},
            "request_records": (
                { "completed": True, "energy_ledger": (ledger,)},
            ),
        }
        envelope = {
            "trace_options": {
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 1e-6,
            }
        }

        valid = evaluate_energy_alignment(
            planned,
            runtime,
            runtime_energy_envelope=envelope,
        )
        self.assertTrue(valid.coefficients_match)
        self.assertTrue(valid.consistent)
        self.assertEqual(
            valid.runtime_energy["coefficient_audit"]["formula_mismatch_count"],
            0,
        )

        tampered_runtime = {
            **runtime,
            "request_records": (
                {

                    "completed": True,
                    "energy_ledger": ({**ledger, "runtime_compute_energy_scale": 1.0},),
                },
            ),
        }
        tampered = evaluate_energy_alignment(
            planned,
            tampered_runtime,
            runtime_energy_envelope=envelope,
        )
        self.assertFalse(tampered.coefficients_match)
        self.assertFalse(tampered.consistent)
        self.assertEqual(
            tampered.runtime_energy["coefficient_audit"]["formula_mismatch_count"],
            1,
        )

    def test_alignment_reads_ledger_from_record_provenance_for_both_audits(self):
        ledger = {
            "energy_kind": "request_event",
            "satellite_id": "sat-a",
            "event_id": "energy-event-0",
            "compute_tokens": 10.0,
            "link_bytes_accounted": 2.0,
            "compute_bus_j": 5.0,
            "link_bus_j": 0.5,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 0.25,
            "base_compute_energy_j_per_token": 0.5,
            "base_link_energy_j_per_byte": 0.25,
            "runtime_compute_energy_scale": 1.0,
            "runtime_link_energy_scale": 1.0,
        }
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 5.0,
                "total_link_energy_j": 0.5,
                "total_dynamic_energy_j": 5.5,
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 0.25,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    {

                        "completed": True,
                        "provenance": {"energy_ledger": (ledger,)},
                    },
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 0.5,
                    "link_energy_j_per_byte": 0.25,
                }
            },
        )

        self.assertTrue(result.evaluated)
        self.assertTrue(result.coefficients_match)
        self.assertTrue(result.consistent)
        self.assertFalse(result.requires_resizing)
        self.assertEqual(
            result.runtime_energy["coefficient_audit"]["checked_request_event_count"],
            1,
        )

    def test_less_runtime_sunlight_than_cost_plan_requires_resizing(self):
        ledger = {
            "energy_kind": "request_event",
            "satellite_id": "sat-a",
            "event_id": "energy-event-0",
            "compute_tokens": 10.0,
            "link_bytes_accounted": 0.0,
            "compute_bus_j": 5.0,
            "link_bus_j": 0.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 0.0,
            "base_compute_energy_j_per_token": 0.5,
            "base_link_energy_j_per_byte": 0.0,
            "runtime_compute_energy_scale": 1.0,
            "runtime_link_energy_scale": 1.0,
        }
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 5.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 5.0,
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    { "completed": True, "energy_ledger": (ledger,)},
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 0.5,
                    "link_energy_j_per_byte": 0.0,
                    "shadow_fraction": 0.1,
                },
                "provenance": {"hardware_power": {"sunlit_fraction": 0.9}},
            },
            runtime_sunlight_summary=summarize_runtime_sunlight(
                (
                    {"satellite_id": "sat-a", "start_s": 0.0, "end_s": 5.0, "sunlit": True},
                    {"satellite_id": "sat-a", "start_s": 5.0, "end_s": 10.0, "sunlit": False},
                )
            ),
            tolerance_fraction=0.05,
        )

        self.assertFalse(result.sunlight_fraction_match)
        self.assertTrue(result.requires_resizing)
        self.assertFalse(result.consistent)

    def test_longer_runtime_shadow_than_battery_plan_requires_resizing(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 0,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 0.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 0.0,
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": None},
                "request_records": (),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 0.5,
                    "link_energy_j_per_byte": 0.0,
                    "shadow_fraction": 0.5,
                },
                "provenance": {
                    "hardware_power": {
                        "sunlit_fraction": 0.5,
                        "eclipse_duration_min": 0.0,
                    }
                },
            },
            runtime_sunlight_summary={
                "evaluated": True,
                "actual_sunlit_fraction": 0.5,
                "maximum_shadow_window_s": 50.0,
            },
        )

        self.assertTrue(result.sunlight_fraction_match)
        self.assertFalse(result.eclipse_duration_sufficient)
        self.assertTrue(result.requires_resizing)
        self.assertFalse(result.consistent)

    def test_invalid_runtime_ledger_requires_resizing(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    {

                        "completed": True,
                        "energy_ledger": (
                            {
                                "energy_kind": "request_event",
                                "satellite_id": "sat-a",
                                "event_id": "bad-event",
                                "compute_bus_j": float("nan"),
                                "link_bus_j": 0.0,
                            },
                        ),
                    },
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                    "shadow_fraction": 0.36,
                },
                "provenance": {
                    "hardware_power": {
                        "sunlit_fraction": 0.64,
                        "eclipse_duration_min": 35.0,
                    }
                },
            },
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(result.requires_resizing)
        self.assertEqual(result.runtime_energy["invalid_ledger_entry_count"], 1)

    def test_request_event_requires_satellite_and_event_identifiers(self):
        ledger = {
            "energy_kind": "request_event",
            "compute_tokens": 1.0,
            "link_bytes_accounted": 0.0,
            "compute_bus_j": 1.0,
            "link_bus_j": 0.0,
            "compute_energy_j_per_token": 1.0,
            "link_energy_j_per_byte": 0.0,
            "base_compute_energy_j_per_token": 1.0,
            "base_link_energy_j_per_byte": 0.0,
            "runtime_compute_energy_scale": 1.0,
            "runtime_link_energy_scale": 1.0,
        }
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    { "completed": True, "energy_ledger": (ledger,)},
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(result.requires_resizing)
        self.assertEqual(result.runtime_energy["invalid_ledger_entry_count"], 1)
        self.assertEqual(result.runtime_energy["per_satellite_dynamic_energy_j"], {})

    def test_positive_completion_ratio_without_request_records_requires_resizing(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "scope": "trace_backed_runtime_sanity",
                "trace_model": "surrogate_orbit_trace",
                "runtime_metrics": {"completion_ratio": 1.0},
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(result.requires_resizing)
        self.assertEqual(result.runtime_energy["missing_runtime_request_record_count"], 1)

    def test_completion_ratio_must_match_request_records(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "scope": "trace_backed_runtime_sanity",
                "trace_model": "surrogate_orbit_trace",
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    { "completed": False, "energy_ledger": ()},
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(result.requires_resizing)
        self.assertTrue(result.runtime_energy["runtime_request_record_mismatch"])
        self.assertEqual(result.runtime_energy["missing_runtime_request_record_count"], 0)

    def test_invalid_or_missing_required_sunlight_evidence_requires_resizing(self):
        planned = {
            "offered_request_count": 0,
            "target_served_ratio": 1.0,
            "total_compute_energy_j": 0.0,
            "total_link_energy_j": 0.0,
            "total_dynamic_energy_j": 0.0,
            "compute_energy_j_per_token": 1.0,
            "link_energy_j_per_byte": 0.0,
        }
        runtime = {
            "evaluated": True,
            "scope": "trace_backed_runtime_sanity",
            "trace_model": "surrogate_orbit_trace",
            "runtime_metrics": {"completion_ratio": None},
            "request_records": (),
        }
        envelope = {
            "trace_options": {
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
                "shadow_fraction": 0.36,
            },
            "provenance": {
                "hardware_power": {
                    "sunlit_fraction": 0.64,
                    "eclipse_duration_min": 35.0,
                }
            },
        }
        invalid_summaries = (
            {},
            {
                "evaluated": False,
                "actual_sunlit_fraction": float("nan"),
                "maximum_shadow_window_s": float("nan"),
                "invalid_window_count": 1,
            },
            {
                "evaluated": True,
                "actual_sunlit_fraction": 0.64,
            },
        )

        for summary in invalid_summaries:
            with self.subTest(summary=summary):
                result = evaluate_energy_alignment(
                    planned,
                    runtime,
                    runtime_energy_envelope=envelope,
                    runtime_sunlight_summary=summary,
                )
                self.assertFalse(result.evaluated)
                self.assertFalse(result.consistent)
                self.assertTrue(result.requires_resizing)
                self.assertTrue(result.sunlight_contract["audit_required"])
                self.assertFalse(result.sunlight_contract["audit_valid"])

    def test_served_request_without_energy_ledger_requires_resizing(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    { "completed": True, "energy_ledger": ()},
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertTrue(result.requires_resizing)
        self.assertEqual(
            result.runtime_energy["missing_required_ledger_record_count"], 1
        )

    def test_fully_rejected_workload_without_ledger_is_not_a_sizing_failure(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {"completion_ratio": 0.0},
                "request_records": (
                    { "completed": False, "energy_ledger": ()},
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertFalse(result.requires_resizing)
        self.assertFalse(result.target_served_ratio_met)
        self.assertEqual(
            result.runtime_energy["missing_required_ledger_record_count"], 0
        )

    def test_accepted_timeout_before_compute_does_not_require_energy_ledger(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "scope": "trace_backed_runtime_sanity",
                "trace_model": "surrogate_orbit_trace",
                "runtime_metrics": {"completion_ratio": 0.0},
                "request_records": (
                    {

                        "completed": False,
                        "timeout": True,
                        "energy_violation": False,
                        "energy_ledger": (),
                    },
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertFalse(result.requires_resizing)
        self.assertFalse(result.target_served_ratio_met)
        self.assertEqual(
            result.runtime_energy["missing_required_ledger_record_count"], 0
        )

    def test_idealized_compute_only_runtime_does_not_require_energy_ledger(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "scope": "idealized_compute_only",
                "trace_model": "ideal_full_visibility",
                "runtime_metrics": {
                    "completion_ratio": 1.0,
                    "scope": "idealized_compute_only",
                    "trace_model": "ideal_full_visibility",
                },
                "request_records": (
                    { "completed": True, "energy_ledger": ()},
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                    "shadow_fraction": 0.36,
                },
                "provenance": {
                    "hardware_power": {
                        "sunlit_fraction": 0.64,
                        "eclipse_duration_min": 35.0,
                    }
                },
            },
        )

        self.assertFalse(result.evaluated)
        self.assertFalse(result.requires_resizing)
        self.assertFalse(result.consistent)
        self.assertEqual(
            result.runtime_energy["missing_required_ledger_record_count"], 0
        )
        self.assertFalse(result.provenance["runtime_energy_ledger_expected"])
        self.assertFalse(result.sunlight_contract["audit_required"])
        self.assertTrue(result.sunlight_contract["audit_valid"])

    def test_ground_fallback_does_not_require_satellite_energy_ledger(self):
        result = evaluate_energy_alignment(
            {
                "offered_request_count": 1,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 1.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 1.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "scope": "trace_backed_runtime_sanity",
                "trace_model": "surrogate_orbit_trace",
                "runtime_metrics": {"completion_ratio": 1.0},
                "request_records": (
                    {

                        "completed": True,
                        "fallback_used": True,
                        "action_type": "ground_fallback",
                        "energy_ledger": (),
                    },
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertFalse(result.evaluated)
        self.assertFalse(result.requires_resizing)
        self.assertEqual(
            result.runtime_energy["missing_required_ledger_record_count"], 0
        )

    def test_per_satellite_hotspot_requires_resizing_even_when_total_matches(self):
        def ledger(event_id, *, unserved_energy_j=0.0):
            return {
                "energy_kind": "request_event",
                "satellite_id": "sat-a",
                "event_id": event_id,
                "compute_tokens": 10_000.0,
                "link_bytes_accounted": 0.0,
                "compute_bus_j": 10_000.0,
                "link_bus_j": 0.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
                "base_compute_energy_j_per_token": 1.0,
                "base_link_energy_j_per_byte": 0.0,
                "runtime_compute_energy_scale": 1.0,
                "runtime_link_energy_scale": 1.0,
                "unserved_energy_j": unserved_energy_j,
            }

        result = evaluate_energy_alignment(
            {
                "offered_request_count": 2,
                "target_served_ratio": 1.0,
                "total_compute_energy_j": 20_000.0,
                "total_link_energy_j": 0.0,
                "total_dynamic_energy_j": 20_000.0,
                "per_satellite_dynamic_energy_j": 10_000.0,
                "maximum_request_dynamic_energy_j": 10_000.0,
                "per_satellite_dynamic_buffer_energy_j": 10_000.0,
                "compute_energy_j_per_token": 1.0,
                "link_energy_j_per_byte": 0.0,
            },
            {
                "evaluated": True,
                "runtime_metrics": {
                    "completion_ratio": 0.5,
                    "energy_violation_count": 1,
                },
                "request_records": (
                    { "completed": True, "energy_ledger": (ledger("e0"),)},
                    {

                        "completed": False,
                        "energy_ledger": (
                            ledger("e1", unserved_energy_j=10_000.0),
                        ),
                    },
                ),
            },
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                }
            },
        )

        self.assertEqual(result.difference_fraction, 0.0)
        self.assertTrue(result.coefficients_match)
        self.assertTrue(result.requires_resizing)
        self.assertFalse(result.consistent)
        self.assertEqual(
            result.runtime_energy["per_satellite_dynamic_energy_j"]["sat-a"],
            20_000.0,
        )
        self.assertEqual(result.runtime_energy["unserved_request_energy_j"], 10_000.0)
        self.assertEqual(
            result.runtime_energy["request_energy_violation_event_count"], 1
        )
        self.assertIn("sat-a", result.runtime_energy["per_satellite_overruns"])

    def test_request_energy_is_bus_side_and_discharge_efficiency_applies_once(self):
        satellite = SatelliteState("sat-a", soc=0.8, initial_soc=0.8)
        update = apply_request_energy(
            satellite,
            prompt_tokens=5,
            output_tokens=0,
            link_bytes=0,
            energy_window=_window(
                compute_energy_j_per_token=2.0,
                battery_discharge_efficiency=0.5,
            ),
            event_time_s=0.0,
        )
        ledger = update["ledger_entry"]

        self.assertEqual(update["dynamic_energy_j"], 10.0)
        self.assertEqual(ledger["battery_discharge_output_bus_j"], 10.0)
        self.assertEqual(ledger["battery_discharge_j"], 20.0)
        self.assertEqual(ledger["conversion_loss_j"], 10.0)
        self.assertAlmostEqual(satellite.soc, 0.78)

    def test_attaching_alignment_to_dict_preserves_existing_runtime_details(self):
        runtime = {
            "evaluated": True,
            "runtime_metrics": {"completion_ratio": 1.0},
            "request_records": (),
            "details": {"existing_contract": "keep-me"},
        }
        planned = {
            "offered_request_count": 0,
            "target_served_ratio": 1.0,
            "total_dynamic_energy_j": 0.0,
            "total_compute_energy_j": 0.0,
            "total_link_energy_j": 0.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
        }
        attached, alignment = attach_energy_alignment(
            runtime,
            planned,
            runtime_energy_envelope={
                "trace_options": {
                    "compute_energy_j_per_token": 0.5,
                    "link_energy_j_per_byte": 1e-6,
                }
            },
        )

        self.assertTrue(alignment.consistent)
        self.assertEqual(attached["details"]["existing_contract"], "keep-me")
        self.assertIn("energy_alignment", attached["details"])


if __name__ == "__main__":
    unittest.main()
