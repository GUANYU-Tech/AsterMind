import unittest
from copy import deepcopy

from orbitserve.orbit import SGP4_ELEMENT_BUNDLE_SCHEMA, Sgp4ElementBundle
from orbitserve.orbit.validation import (
    STK_11_BASELINE_BUILD,
    STK_11_BASELINE_VERSION,
    StkCommandError,
    StkConnectClient,
    StkConnectError,
    StkConnectResponse,
    StkEnvironmentError,
    StkMappingError,
    StkValidationError,
    build_stk_set_state_command,
    capture_stk_direct_states,
    format_stk_utcg,
    parse_stk_state_report,
    probe_stk_11_connect,
)


def _satellite(satellite_id="sat-00001", satnum=10001):
    return {
        "satellite_id": satellite_id,
        "satnum": satnum,
        "classification": "U",
        "object_id": "2026-001A",
        "epoch_utc": "2026-07-20T00:00:00.000000Z",
        "inclination_deg": 53.0,
        "raan_deg": 20.0,
        "eccentricity": 0.001,
        "argument_of_perigee_deg": 30.0,
        "mean_anomaly_deg": 40.0,
        "mean_motion_rev_per_day": 15.25,
        "mean_motion_dot_rev_per_day2": 0.000002,
        "mean_motion_ddot_rev_per_day3": -0.000000003,
        "bstar_inv_earth_radii": 0.00001,
        "ephemeris_type": 0,
        "element_set_number": 1,
        "revolution_number_at_epoch": 7,
    }


def _bundle(two_satellites=False):
    satellites = [_satellite()]
    if two_satellites:
        second = _satellite("sat-00002", 10002)
        second["object_id"] = "2026-001B"
        satellites.append(second)
    return {
        "schema_version": SGP4_ELEMENT_BUNDLE_SCHEMA,
        "element_semantics": "sgp4_mean_elements",
        "gravity_model": "wgs72",
        "ops_mode": "improved",
        "time_scale": "utc",
        "satellites": satellites,
        "provenance": {
            "source_kind": "synthetic_sgp4_mean_elements",
            "source_name": "stk-validation-unit-test",
            "source_version": "unit-v1",
            "observationally_fitted": False,
        },
    }


def _header(return_type, payload_length):
    return f"{return_type} {payload_length}".ljust(40).encode("ascii")


class _ScriptedSocket:
    def __init__(self, incoming):
        self.incoming = bytearray(incoming)
        self.sent = []
        self.timeout = None
        self.closed = False

    def recv(self, count):
        if not self.incoming:
            return b""
        chunk = bytes(self.incoming[:count])
        del self.incoming[:count]
        return chunk

    def sendall(self, data):
        self.sent.append(data)

    def settimeout(self, value):
        self.timeout = value

    def close(self):
        self.closed = True


class _FakeValidationClient:
    def __init__(
        self,
        *,
        scenario_open=False,
        fail_on_report=False,
        fail_new_transport=False,
        shared_state=None,
        version=STK_11_BASELINE_VERSION,
    ):
        self.state = (
            {"scenario_open": scenario_open}
            if shared_state is None
            else shared_state
        )
        self.fail_on_report = fail_on_report
        self.fail_new_transport = fail_new_transport
        self.version = version
        self.commands = []
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
            return StkConnectResponse(ack="ACK", payload=self.version)
        if command == "CheckScenario /":
            return StkConnectResponse(
                ack="ACK",
                payload="1" if self.state["scenario_open"] else "0",
            )
        if command.startswith("New / Scenario "):
            self.state["scenario_open"] = True
            if self.fail_new_transport:
                raise StkConnectError("synthetic lost scenario ACK")
            return StkConnectResponse(ack="ACK")
        if command == "Unload / *":
            self.state["scenario_open"] = False
            return StkConnectResponse(ack="ACK")
        if command.startswith("Report_RM "):
            if self.fail_on_report:
                raise StkCommandError(command)
            return StkConnectResponse(
                ack="ACK",
                rows=(
                    "Time (UTCG),x (km),y (km),z (km),vx (km/sec),vy (km/sec),vz (km/sec)",
                    "20 Jul 2026 00:00:00.000000,1,2,3,4,5,6",
                    "20 Jul 2026 00:01:00.000000,7,8,9,10,11,12",
                ),
            )
        return StkConnectResponse(ack="ACK")


class StkConnectProtocolTests(unittest.TestCase):
    def test_reads_single_and_multiple_responses(self):
        version = STK_11_BASELINE_VERSION.encode("ascii")
        rows = (b"first", b"second")
        incoming = (
            b"ACKACK"
            + b"ACK"
            + _header("GetSTKVersion", len(version))
            + version
            + b"ACK"
            + _header("Report_RM", len(str(len(rows))))
            + str(len(rows)).encode("ascii")
            + b"".join(_header("Report_RM", len(row)) + row for row in rows)
        )
        connection = _ScriptedSocket(incoming)
        client = StkConnectClient(
            connection_factory=lambda address, timeout: connection
        )

        with client:
            single = client.execute("GetSTKVersion /", response_kind="single")
            multiple = client.execute("Report_RM *", response_kind="multiple")

        self.assertEqual(single.payload, STK_11_BASELINE_VERSION)
        self.assertEqual(multiple.rows, ("first", "second"))
        self.assertEqual(
            connection.sent,
            [
                b"ConControl / AckOn\n",
                b"ConControl / AsyncOff\n",
                b"GetSTKVersion /\n",
                b"Report_RM *\n",
            ],
        )
        self.assertTrue(connection.closed)

    def test_nack_is_strict_unless_explicitly_allowed(self):
        connection = _ScriptedSocket(b"ACKACKNACKNACK")
        client = StkConnectClient(
            connection_factory=lambda address, timeout: connection
        )

        with client:
            with self.assertRaises(StkCommandError):
                client.execute("Bad /")
            response = client.execute("Probe /", allow_nack=True)

        self.assertEqual(response.ack, "NACK")


class StkMappingAndReportTests(unittest.TestCase):
    def test_direct_mapping_uses_rev_per_day_and_full_derivatives(self):
        element = Sgp4ElementBundle.from_value(_bundle()).satellites[0]

        command = build_stk_set_state_command(
            element,
            stk_object_name="OrbitSat000001",
            step_s=60,
        )

        self.assertIn('OrbitEpoch "20 Jul 2026 00:00:00.000000"', command)
        self.assertIn("ElementSet 15.25 0.001 53 30 20 40", command)
        self.assertIn("IntlDesignator 26001A", command)
        self.assertIn("MeanMotionDot 1.9999999999999999e-06", command)
        self.assertIn("MeanMotionDotDot -3e-09", command)
        self.assertNotIn("0.066540", command)

    def test_stk_11_rejects_extended_satellite_number(self):
        payload = _bundle()
        payload["satellites"][0]["satnum"] = 100001
        element = Sgp4ElementBundle.from_value(payload).satellites[0]

        with self.assertRaisesRegex(StkMappingError, "above 99999"):
            build_stk_set_state_command(
                element,
                stk_object_name="OrbitSat000001",
                step_s=60,
            )

    def test_utcg_format_and_report_alignment_are_deterministic(self):
        self.assertEqual(
            format_stk_utcg("2026-07-20T00:00:00.000000Z"),
            "20 Jul 2026 00:00:00.000000",
        )
        samples = parse_stk_state_report(
            (
                "Time (UTCG),x (km),y (km),z (km),vx (km/sec),vy (km/sec),vz (km/sec)",
                "20 Jul 2026 00:00:00.000000,1,2,3,4,5,6",
                "20 Jul 2026 00:01:00.000000,7,8,9,10,11,12",
            ),
            satellite_id="sat-00001",
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            expected_sample_times_s=(0, 60),
        )

        self.assertEqual(samples[1].position_km, (7.0, 8.0, 9.0))
        self.assertEqual(samples[1].velocity_km_s, (10.0, 11.0, 12.0))

    def test_report_rejects_wrong_grid_and_nonfinite_values(self):
        header = (
            "Time (UTCG),x (km),y (km),z (km),"
            "vx (km/sec),vy (km/sec),vz (km/sec)"
        )
        with self.assertRaisesRegex(StkValidationError, "sample count"):
            parse_stk_state_report(
                (header, "20 Jul 2026 00:00:00.000000,1,2,3,4,5,6"),
                satellite_id="sat-00001",
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                expected_sample_times_s=(0, 60),
            )
        with self.assertRaisesRegex(StkValidationError, "finite"):
            parse_stk_state_report(
                (
                    header,
                    "20 Jul 2026 00:00:00.000000,nan,2,3,4,5,6",
                    "20 Jul 2026 00:01:00.000000,7,8,9,10,11,12",
                ),
                satellite_id="sat-00001",
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                expected_sample_times_s=(0, 60),
            )


class StkValidationWorkflowTests(unittest.TestCase):
    def test_probe_is_read_only_and_reports_baseline(self):
        client = _FakeValidationClient()

        result = probe_stk_11_connect(
            installed_build=STK_11_BASELINE_BUILD,
            client_factory=lambda **kwargs: client,
        )

        self.assertTrue(result.baseline_matches)
        self.assertFalse(result.scenario_open)
        self.assertEqual(
            [item[0] for item in client.commands],
            ["GetSTKVersion /", "CheckScenario /"],
        )

    def test_capture_orders_evidence_and_unloads_before_disconnect(self):
        client = _FakeValidationClient()

        capture = capture_stk_direct_states(
            _bundle(two_satellites=True),
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            sample_times_s=(0, 60),
            installed_build=STK_11_BASELINE_BUILD,
            client_factory=lambda **kwargs: client,
        )

        command_names = [item[0] for item in client.commands]
        self.assertEqual(command_names[-1], "Unload / *")
        self.assertTrue(client.closed)
        self.assertEqual(
            [(sample.time_s, sample.satellite_id) for sample in capture.teme_samples],
            [
                (0.0, "sat-00001"),
                (0.0, "sat-00002"),
                (60.0, "sat-00001"),
                (60.0, "sat-00002"),
            ],
        )
        self.assertEqual(len(capture.fixed_samples), 4)
        self.assertTrue(capture.canonical_bytes().endswith(b"\n"))
        self.assertNotIn(b"\r\n", capture.canonical_bytes())
        self.assertEqual(capture.canonical_bytes(), capture.canonical_bytes())
        self.assertEqual(capture.propagator_module, "SGP4")
        self.assertEqual(capture.update_mode, "None")
        self.assertEqual(capture.teme_report_style, "TEMEofDate Position Velocity")
        self.assertEqual(capture.fixed_report_style, "Fixed Position Velocity")
        self.assertEqual(capture.time_scale, "UTC")
        self.assertFalse(capture.formal_thresholds_frozen)

    def test_capture_cleans_up_when_report_fails(self):
        client = _FakeValidationClient(fail_on_report=True)

        with self.assertRaises(StkCommandError):
            capture_stk_direct_states(
                _bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0, 60),
                installed_build=STK_11_BASELINE_BUILD,
                client_factory=lambda **kwargs: client,
            )

        self.assertEqual(client.commands[-1][0], "Unload / *")
        self.assertTrue(client.closed)

    def test_capture_refuses_existing_scenario_without_mutation(self):
        client = _FakeValidationClient(scenario_open=True)

        with self.assertRaisesRegex(StkEnvironmentError, "already open"):
            capture_stk_direct_states(
                _bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0, 60),
                installed_build=STK_11_BASELINE_BUILD,
                client_factory=lambda **kwargs: client,
            )

        self.assertNotIn("Unload / *", [item[0] for item in client.commands])
        self.assertFalse(
            any(item[0].startswith("New /") for item in client.commands)
        )

    def test_capture_rejects_build_or_nonuniform_grid_before_connecting(self):
        called = False

        def factory(**kwargs):
            nonlocal called
            called = True
            return _FakeValidationClient()

        with self.assertRaisesRegex(StkEnvironmentError, "does not match"):
            capture_stk_direct_states(
                _bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0, 60),
                installed_build="11.6.0.999",
                client_factory=factory,
            )
        with self.assertRaisesRegex(ValueError, "uniform"):
            capture_stk_direct_states(
                deepcopy(_bundle()),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0, 60, 121),
                installed_build=STK_11_BASELINE_BUILD,
                client_factory=factory,
            )

        self.assertFalse(called)

    def test_capture_recovers_scenario_when_creation_ack_is_lost(self):
        shared_state = {"scenario_open": False}
        clients = []

        def factory(**kwargs):
            client = _FakeValidationClient(
                shared_state=shared_state,
                fail_new_transport=not clients,
            )
            clients.append(client)
            return client

        with self.assertRaisesRegex(StkConnectError, "lost scenario ACK"):
            capture_stk_direct_states(
                _bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0, 60),
                installed_build=STK_11_BASELINE_BUILD,
                client_factory=factory,
            )

        self.assertEqual(len(clients), 2)
        self.assertFalse(shared_state["scenario_open"])
        self.assertEqual(
            [item[0] for item in clients[1].commands],
            ["CheckScenario /", "Unload / *"],
        )

    def test_capture_rejects_non_boolean_default_control_before_connecting(self):
        called = False

        def factory(**kwargs):
            nonlocal called
            called = True
            return _FakeValidationClient()

        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            capture_stk_direct_states(
                _bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0, 60),
                installed_build=STK_11_BASELINE_BUILD,
                configure_application_defaults="yes",  # type: ignore[arg-type]
                client_factory=factory,
            )

        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
