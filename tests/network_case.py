"""Physical network fixtures independent of request computation."""

from types import SimpleNamespace
from dataclasses import dataclass, field
from orbitserve.orbit import ORBIT_TRACE_SCHEMA, OrbitTraceBundle, OrbitTraceManifest
from orbitserve.runtime import NetworkSpec, NetworkNodeSpec, LinkProfile, NetworkState
from orbitserve.runtime.engine.events import _EventEngine


def _trace():
    manifest = OrbitTraceManifest(
        schema_version=ORBIT_TRACE_SCHEMA,
        trace_model="unit_geometry",
        evaluated=True,
        trace_id="unit-trace",
        time={
            "epoch": "2026-01-01T00:00:00Z",
            "start_s": 0.0,
            "end_s": 100.0,
            "duration_s": 100.0,
            "step_s": 1.0,
        },
        files={},
        sha256={},
        valid_for_paper_main_results=False,
        scope="orbit_geometry_only",
        warnings=(),
        provenance={},
    )
    return OrbitTraceBundle(
        trace_manifest=manifest,
        satellite_catalog=(
            {"satellite_id": "sat-a", "compute_capable": True},
            {"satellite_id": "sat-b", "compute_capable": True},
        ),
        ephemeris_samples=tuple(
            {**row, "time_s": t}
            for t in (0.0, 100.0)
            for row in (
                {
                    "time_s": 0.0,
                    "satellite_id": "sat-a",
                    "position_itrs_km": [7000.0, 0.0, 0.0],
                },
                {
                    "time_s": 0.0,
                    "satellite_id": "sat-b",
                    "position_itrs_km": [0.0, 7000.0, 0.0],
                },
            )
        ),
        sgl_windows=(
            {
                "ingress_id": "gateway-0",
                "satellite_id": "sat-a",
                "start_s": 0.0,
                "end_s": 100.0,
            },
            {
                "ingress_id": "gateway-1",
                "satellite_id": "sat-b",
                "start_s": 0.0,
                "end_s": 100.0,
            },
        ),
        isl_windows=(
            {
                "link_id": "isl:sat-a:sat-b",
                "endpoint_a": "sat-a",
                "endpoint_b": "sat-b",
                "start_s": 0.0,
                "end_s": 100.0,
                "distance_km": 9000.0,
                "propagation_delay_s": 0.03,
            },
        ),
        sunlight_windows=(
            {"satellite_id": "sat-a", "start_s": 0.0, "end_s": 100.0, "sunlit": False},
            {"satellite_id": "sat-b", "start_s": 0.0, "end_s": 100.0, "sunlit": True},
        ),
    )


def _network():
    return NetworkSpec(
        (
            NetworkNodeSpec("sat-a", "compute", 2, 1),
            NetworkNodeSpec("sat-b", "compute", 2, 1),
            NetworkNodeSpec("gateway-0", "ground", sgl_terminals=1),
            NetworkNodeSpec("gateway-1", "ground", sgl_terminals=1),
        ),
        {"communication_isl": LinkProfile(1e9, 0.0), "sgl": LinkProfile(1e9, 0.0)},
        communication_topology="explicit",
        ground_roles=(),
        fault_timing="continuous",
        transport_model="queue",
        explicit_pairs=(
            ("sat-a", "sat-b"),
            ("gateway-0", "sat-a"),
            ("gateway-1", "sat-b"),
        ),
    )


class NetworkHost(_EventEngine):
    def debit(self, job, node_id, *, prompt=0, output=0, size=0, stage):
        return None


def run_network(spec, probe):
    network = NetworkState(spec.environment_state, spec.network, spec.gateways)
    environment = SimpleNamespace(network=network, time=spec.environment_state.time)
    host = NetworkHost(spec, environment)
    for row in network.frames:
        host.push(row["time_s"], 0, "network_update")
    for event in network.events:
        host.push(event["time_s"], 0, "link_state", payload=event)
    for at in network.environment_times:
        host.push(at, 0, "topology_sample")
    for at, index, restored, fault in network.fault_observations():
        host.push(at, -2, "fault_batch", payload=((index, fault, restored),))
    from orbitserve.runtime.timeline import GlobalTimeline

    host.timeline = GlobalTimeline(host=host)
    try:
        probe.initialize(host, ())
        host.run()
        return host
    finally:
        try:
            probe.close()
        finally:
            host.close()


@dataclass
class NetworkCase:
    network: object = field(default_factory=_network)
    environment_state: object = field(default_factory=_trace)
    gateways: tuple = (
        {"ingress_id": "gateway-0", "lat_deg": 0.0, "lon_deg": 0.0},
        {"ingress_id": "gateway-1", "lat_deg": 0.0, "lon_deg": 90.0},
    )
    traffic: tuple = ()
    time_horizon_s: float = 100.0
    routing_policy: str = "earliest_arrival"
    target: str = "sat-a"


def traffic(identifier="r", arrival=0.0, size=1000.0):
    return dict(
        request_id=identifier,
        arrival_time_s=arrival,
        ingress_id="gateway-0",
        input_bytes=size,
        output_bytes=200.0,
    )


def run_traffic(spec):
    """Round-trip byte transfers only: no LLM timing, token state or energy model."""

    class Probe:
        def initialize(self, host, unused):
            for row in spec.traffic:
                identity = row["request_id"]
                job = dict(request={"request_id": identity}, done=False, path=[])
                job["external_failure_callback"] = lambda reason, job=job: job.update(
                    failure=reason
                )
                host.service_jobs[identity] = job

                def start(row=row, job=job):
                    def finish():
                        job.update(done=True, completed=True, completed_at=host.now)

                    def inbound():
                        host.send(
                            job,
                            spec.target,
                            row["ingress_id"],
                            row.get("output_bytes", 0),
                            None,
                            "output",
                            "external_flow_done",
                            context={"callback": finish},
                        )

                    try:
                        host.send(
                            job,
                            row["ingress_id"],
                            spec.target,
                            row["input_bytes"],
                            None,
                            "input",
                            "external_flow_done",
                            context={"callback": inbound},
                        )
                    except (ValueError, KeyError) as error:
                        host.fail(job, str(error))

                host.timeline.schedule(row["arrival_time_s"], 30, identity, start)

        def close(self):
            pass

    # The same host/event engine used by production Native Cells drives every hop.
    host = run_network(spec, Probe())
    host.transfer_jobs = list(host.service_jobs.values())
    host.completed = sum(job.get("completed", False) for job in host.transfer_jobs)
    return host
