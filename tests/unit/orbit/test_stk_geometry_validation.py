import unittest

from astermind.builder.orbit import evaluate_sgp4_geometry, propagate_sgp4_teme, transform_sgp4_teme_to_itrs
from astermind.builder.orbit.validation import STK_11_BASELINE_BUILD, STK_11_BASELINE_VERSION, STK_GEOMETRY_CORPUS_CASE_IDS, STK_GEOMETRY_CORPUS_ID, STK_GEOMETRY_CORPUS_REQUIRED_COVERAGE, STK_GEOMETRY_CORPUS_SHA256, StkConnectResponse, StkGeometryCapture, StkGeometryCorpus, StkGeometryCorpusCase, StkIslWindow, StkLightingWindow, StkSglWindow, capture_stk_geometry, compare_stk_geometry_capture, load_stk_geometry_corpus, parse_stk_access_intervals, parse_stk_aer_report, parse_stk_lighting_report, run_stk_geometry_corpus


class _FakeGeometryClient:
    def __init__(self):
        self.commands = []
        self.scenario_open = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True

    def close(self):
        self.closed = True

    def execute(self, command, *, response_kind="none", allow_nack=False):
        self.commands.append((command, response_kind, allow_nack))
        if command == "GetSTKVersion /":
            return StkConnectResponse(ack="ACK", payload=STK_11_BASELINE_VERSION)
        if command == "CheckScenario /":
            return StkConnectResponse(
                ack="ACK", payload="1" if self.scenario_open else "0"
            )
        if command.startswith("New / Scenario "):
            self.scenario_open = True
        elif command == "Unload / *":
            self.scenario_open = False
        elif command.startswith("Access "):
            return StkConnectResponse(ack="ACK", payload="NoAccesses")
        elif command.startswith("Report_RM ") and 'Style "AER"' in command:
            return StkConnectResponse(
                ack="ACK",
                rows=(
                    '"Time (UTCG)","Azimuth (deg)","Elevation (deg)","Range (km)"',
                    "",
                ),
            )
        elif command.startswith("Report_RM ") and 'Style "Lighting Times"' in command:
            return StkConnectResponse(
                ack="ACK",
                rows=("Sunlight Times", "Penumbra Times", "Umbra Times"),
            )
        return StkConnectResponse(ack="ACK")


class StkGeometryCorpusTests(unittest.TestCase):
    def test_frozen_corpus_has_exact_identity_grid_and_coverage(self):
        corpus = load_stk_geometry_corpus()
        case = corpus.cases[0]

        self.assertEqual(corpus.corpus_id, STK_GEOMETRY_CORPUS_ID)
        self.assertEqual(corpus.case_ids, STK_GEOMETRY_CORPUS_CASE_IDS)
        self.assertEqual(corpus.source_sha256, STK_GEOMETRY_CORPUS_SHA256)
        self.assertEqual(len(case.sample_times_s), 721)
        self.assertEqual(case.sample_times_s[:2], (0.0, 30.0))
        self.assertEqual(case.sample_times_s[-1], 21600.0)
        self.assertEqual(
            set(case.coverage), set(STK_GEOMETRY_CORPUS_REQUIRED_COVERAGE)
        )
        self.assertEqual(len(case.ground_ingresses), 2)
        self.assertEqual(case.isl_pairs, (("geo-leo-a", "geo-leo-b"),))

    def test_mapping_loader_rejects_missing_boundary_coverage(self):
        corpus = load_stk_geometry_corpus()
        payload = {
            "schema_version": corpus.schema_version,
            "corpus_id": corpus.corpus_id,
            "cases": [],
        }
        with self.assertRaisesRegex(ValueError, "non-empty"):
            load_stk_geometry_corpus(payload)


class StkGeometryParserTests(unittest.TestCase):
    def test_access_parser_normalizes_and_clips_half_open_intervals(self):
        payload = (
            "/Application/STK/Scenario/X/Facility/F "
            "/Application/STK/Scenario/X/Satellite/S 2  "
            "20 Jul 2026 00:00:10.000 20 Jul 2026 00:00:20.000 "
            "20 Jul 2026 00:00:50.000 20 Jul 2026 00:01:10.000\n"
        )

        intervals = parse_stk_access_intervals(
            payload,
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            interval_start_s=0,
            interval_end_s=60,
        )

        self.assertEqual(intervals, ((10.0, 20.0), (50.0, 60.0)))
        self.assertEqual(
            parse_stk_access_intervals(
                "/Application/STK/Scenario/X/Facility/F "
                "/Application/STK/Scenario/X/Satellite/S NoAccesses",
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                interval_start_s=0,
                interval_end_s=60,
            ),
            (),
        )

    def test_lighting_parser_keeps_penumbra_separate(self):
        rows = (
            "Sunlight Times",
            "  20 Jul 2026 00:00:00.000  20 Jul 2026 00:00:20.000 20.000",
            "Penumbra Times",
            "  20 Jul 2026 00:00:20.000  20 Jul 2026 00:00:21.000 1.000",
            "Umbra Times",
            "  20 Jul 2026 00:00:21.000  20 Jul 2026 00:00:40.000 19.000",
        )

        windows = parse_stk_lighting_report(
            rows,
            satellite_id="sat-a",
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            interval_start_s=0,
            interval_end_s=60,
        )

        self.assertEqual(
            [(window.lighting, window.start_s, window.end_s) for window in windows],
            [
                ("sunlight", 0.0, 20.0),
                ("penumbra", 20.0, 21.0),
                ("umbra", 21.0, 40.0),
            ],
        )

    def test_aer_parser_uses_facility_side_values(self):
        samples = parse_stk_aer_report(
            (
                '"Time (UTCG)","Azimuth (deg)","Elevation (deg)","Range (km)"',
                "20 Jul 2026 00:00:30.000,151.739,10.000,1666.257847",
            ),
            ingress_id="gateway",
            satellite_id="sat-a",
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            interval_start_s=0,
            interval_end_s=60,
        )

        self.assertEqual(samples[0].time_s, 30.0)
        self.assertEqual(samples[0].elevation_deg, 10.0)
        self.assertEqual(samples[0].range_km, 1666.257847)
        self.assertEqual(
            parse_stk_aer_report(
                ("",),
                ingress_id="gateway",
                satellite_id="sat-a",
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                interval_start_s=0,
                interval_end_s=60,
            ),
            (),
        )


class StkGeometryWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_stk_geometry_corpus().cases[0]
        teme = propagate_sgp4_teme(
            cls.case.elements,
            simulation_epoch_utc=cls.case.simulation_epoch_utc,
            sample_times_s=cls.case.sample_times_s,
        )
        fixed = transform_sgp4_teme_to_itrs(teme)
        cls.orbit = evaluate_sgp4_geometry(
            fixed,
            ground_ingresses=cls.case.ground_ingresses,
            isl_pairs=cls.case.isl_pairs,
            isl_clearance_km=cls.case.isl_clearance_km,
            shadow_atmosphere_margin_km=(
                cls.case.shadow_atmosphere_margin_km
            ),
        )

    def test_corpus_exercises_every_required_orbit_boundary(self):
        self.assertGreaterEqual(len(self.orbit.sgl_windows), 1)
        self.assertGreaterEqual(len(self.orbit.isl_windows), 1)
        self.assertGreaterEqual(
            sum(not window.sunlit for window in self.orbit.sunlight_windows), 1
        )

    def test_identical_reference_windows_have_perfect_metrics(self):
        lighting = tuple(
            StkLightingWindow(
                window.satellite_id,
                "sunlight" if window.sunlit else "umbra",
                window.start_s,
                window.end_s,
            )
            for window in self.orbit.sunlight_windows
        )
        capture = StkGeometryCapture(
            element_set_id=self.orbit.element_set_id,
            simulation_epoch_utc=self.orbit.simulation_epoch_utc,
            sample_times_s=self.case.sample_times_s,
            stk_version=STK_11_BASELINE_VERSION,
            installed_build=STK_11_BASELINE_BUILD,
            propagator_defaults_configured=True,
            access_defaults_configured=True,
            sgl_windows=tuple(
                StkSglWindow(
                    window.ingress_id,
                    window.satellite_id,
                    window.start_s,
                    window.end_s,
                )
                for window in self.orbit.sgl_windows
            ),
            isl_windows=tuple(
                StkIslWindow(
                    window.endpoint_a,
                    window.endpoint_b,
                    window.start_s,
                    window.end_s,
                )
                for window in self.orbit.isl_windows
            ),
            lighting_windows=lighting,
            aer_samples=(),
            warnings=(),
        )

        report = compare_stk_geometry_capture(capture, self.orbit)

        self.assertTrue(
            all(metric.sample_agreement_ratio == 1.0 for metric in report.window_metrics)
        )
        self.assertTrue(
            all(metric.interval_iou == 1.0 for metric in report.window_metrics)
        )
        self.assertEqual(report.canonical_bytes(), report.canonical_bytes())

    def test_capture_configures_geometry_defaults_and_cleans_up(self):
        client = _FakeGeometryClient()

        capture = capture_stk_geometry(
            self.case.elements,
            simulation_epoch_utc=self.case.simulation_epoch_utc,
            sample_times_s=(0, 30),
            ground_ingresses=self.case.ground_ingresses,
            isl_pairs=self.case.isl_pairs,
            installed_build=STK_11_BASELINE_BUILD,
            configure_application_defaults=True,
            client_factory=lambda **kwargs: client,
        )

        commands = [item[0] for item in client.commands]
        self.assertIn("AccessConfig / UseLightTimeDelay Off", commands)
        self.assertIn("AccessConfig / AberrationType None", commands)
        self.assertEqual(commands[-1], "Unload / *")
        self.assertFalse(client.scenario_open)
        self.assertTrue(client.closed)
        self.assertTrue(capture.access_defaults_configured)
        self.assertEqual(capture.sgl_windows, ())

    def test_runner_forwards_corpus_inputs_and_builds_canonical_evidence(self):
        case = StkGeometryCorpusCase(
            case_id="unit-geometry",
            description="small runner unit case",
            coverage=("sgl_horizon_boundary",),
            simulation_epoch_utc=self.case.simulation_epoch_utc,
            sample_times_s=(0.0, 30.0),
            elements=self.case.elements,
            ground_ingresses=self.case.ground_ingresses[:1],
            isl_pairs=self.case.isl_pairs,
            isl_clearance_km=0.0,
            shadow_atmosphere_margin_km=0.0,
            minimum_expected_event_counts=(),
        )
        corpus = StkGeometryCorpus("unit-corpus", (case,), "a" * 64)
        calls = []

        def capture_function(elements, **kwargs):
            calls.append((elements, kwargs))
            return StkGeometryCapture(
                element_set_id=elements.element_set_id,
                simulation_epoch_utc=kwargs["simulation_epoch_utc"],
                sample_times_s=tuple(kwargs["sample_times_s"]),
                stk_version=STK_11_BASELINE_VERSION,
                installed_build=kwargs["installed_build"],
                propagator_defaults_configured=True,
                access_defaults_configured=True,
                sgl_windows=(),
                isl_windows=(),
                lighting_windows=(),
                aer_samples=(),
                warnings=(),
            )

        result = run_stk_geometry_corpus(
            corpus,
            installed_build=STK_11_BASELINE_BUILD,
            configure_application_defaults=True,
            capture_function=capture_function,
        )

        self.assertEqual(result.case_ids, ("unit-geometry",))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["sample_times_s"], (0.0, 30.0))
        self.assertEqual(result.canonical_bytes(), result.canonical_bytes())


if __name__ == "__main__":
    unittest.main()
