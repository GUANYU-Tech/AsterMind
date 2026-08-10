import unittest

from orbitserve.orbit import (
    propagate_sgp4_teme,
    transform_sgp4_teme_to_itrs,
)
from orbitserve.orbit.validation import (
    STK_11_BASELINE_BUILD,
    STK_11_BASELINE_VERSION,
    STK_STATE_CORPUS_CASE_IDS,
    StkDirectStateCapture,
    StkEnvironmentError,
    StkStateSample,
    load_stk_state_corpus,
    run_stk_state_corpus,
)


def _matching_capture(
    elements,
    *,
    simulation_epoch_utc,
    sample_times_s,
    host,
    port,
    installed_build,
    configure_application_defaults,
    connect_timeout_s,
    read_timeout_s,
):
    propagation = propagate_sgp4_teme(
        elements,
        simulation_epoch_utc=simulation_epoch_utc,
        sample_times_s=sample_times_s,
    )
    earth_fixed = transform_sgp4_teme_to_itrs(propagation)
    return StkDirectStateCapture(
        element_set_id=elements.element_set_id,
        simulation_epoch_utc=simulation_epoch_utc,
        sample_times_s=tuple(float(value) for value in sample_times_s),
        stk_version=STK_11_BASELINE_VERSION,
        installed_build=installed_build,
        propagator_defaults_configured=configure_application_defaults,
        teme_samples=tuple(
            StkStateSample(
                satellite_id=sample.satellite_id,
                time_s=sample.time_s,
                position_km=sample.position_teme_km,
                velocity_km_s=sample.velocity_teme_km_s,
            )
            for sample in propagation.samples
        ),
        fixed_samples=tuple(
            StkStateSample(
                satellite_id=sample.satellite_id,
                time_s=sample.time_s,
                position_km=sample.position_itrs_km,
                velocity_km_s=sample.velocity_itrs_km_s,
            )
            for sample in earth_fixed.samples
        ),
        warnings=(),
    )


class StkStateCorpusRunnerTests(unittest.TestCase):
    def test_runs_frozen_cases_in_order_and_builds_canonical_evidence(self):
        corpus = load_stk_state_corpus()

        first = run_stk_state_corpus(
            corpus,
            installed_build=STK_11_BASELINE_BUILD,
            configure_application_defaults=True,
            capture_function=_matching_capture,
        )
        second = run_stk_state_corpus(
            corpus,
            installed_build=STK_11_BASELINE_BUILD,
            configure_application_defaults=True,
            capture_function=_matching_capture,
        )

        self.assertEqual(first.case_ids, STK_STATE_CORPUS_CASE_IDS)
        self.assertEqual(first.corpus_source_sha256, corpus.source_sha256)
        self.assertEqual(first.evidence_scope, "state_vectors_only")
        self.assertFalse(first.formal_thresholds_frozen)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertTrue(first.canonical_bytes().endswith(b"\n"))
        self.assertNotIn(b"\r\n", first.canonical_bytes())
        for case in first.cases:
            self.assertEqual(case.comparison.teme.metrics[0].maximum, 0.0)
            self.assertEqual(case.comparison.fixed.metrics[0].maximum, 0.0)

    def test_forwards_environment_controls_to_each_capture(self):
        calls = []

        def capture(*args, **kwargs):
            calls.append(kwargs.copy())
            return _matching_capture(*args, **kwargs)

        run_stk_state_corpus(
            installed_build=STK_11_BASELINE_BUILD,
            host="127.0.0.2",
            port=6001,
            configure_application_defaults=True,
            capture_function=capture,
        )

        self.assertEqual(len(calls), len(STK_STATE_CORPUS_CASE_IDS))
        for call in calls:
            self.assertEqual(call["host"], "127.0.0.2")
            self.assertEqual(call["port"], 6001)
            self.assertEqual(call["installed_build"], STK_11_BASELINE_BUILD)
            self.assertTrue(call["configure_application_defaults"])
            self.assertEqual(call["connect_timeout_s"], 10.0)
            self.assertEqual(call["read_timeout_s"], 120.0)

    def test_stops_at_first_capture_failure(self):
        calls = 0

        def capture(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("synthetic STK failure")
            return _matching_capture(*args, **kwargs)

        with self.assertRaisesRegex(RuntimeError, "synthetic STK failure"):
            run_stk_state_corpus(
                installed_build=STK_11_BASELINE_BUILD,
                capture_function=capture,
            )

        self.assertEqual(calls, 2)

    def test_rejects_invalid_corpus_and_capture_function(self):
        with self.assertRaisesRegex(TypeError, "corpus"):
            run_stk_state_corpus(  # type: ignore[arg-type]
                object(),
                installed_build=STK_11_BASELINE_BUILD,
            )
        with self.assertRaisesRegex(TypeError, "capture_function"):
            run_stk_state_corpus(
                installed_build=STK_11_BASELINE_BUILD,
                capture_function=None,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(StkEnvironmentError, "does not match"):
            run_stk_state_corpus(
                installed_build="11.6.0.999",
                capture_function=_matching_capture,
            )
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            run_stk_state_corpus(
                installed_build=STK_11_BASELINE_BUILD,
                configure_application_defaults="yes",  # type: ignore[arg-type]
                capture_function=_matching_capture,
            )


if __name__ == "__main__":
    unittest.main()
