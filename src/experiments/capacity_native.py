"""Execute native AsterMind operators on OrbitServe's shared network clock.

External packages are imported only inside the worker. No external source,
Scheduler policy, model dimension or kernel duration is patched.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .capacity_metrics import trial_metrics
from .capacity_study import StudySpec, arrivals, digest, placements, replica_resources, source_identity, write_json


def activate(repository):
    import importlib
    repository = Path(repository).resolve()
    vendor = repository / "third_party/midlayer"
    for path in (repository / "src", vendor):
        if not path.is_dir():
            raise FileNotFoundError(f"missing external AsterMind dependency: {path}")
        sys.path.insert(0, str(path))
    # Require the prepared extension and runtime; do not install or substitute them.
    import _flowsim
    importlib.import_module("astermind.embedded")
    return repository, vendor, _flowsim


@lru_cache(maxsize=32)
def _load_model(path):
    from midlayer.serving.workload.model_config import load_model_config
    return load_model_config(path)


@lru_cache(maxsize=8)
def _load_hardware(path):
    from midlayer.kernel.hardware_spec import HWSpec
    return HWSpec(path)


def build_topology(case, spec):
    from network_sim import PhysicalTopology, Scenario

    owners = [f"sat{s}" for s in range(4)]
    devices = [dict(id="user", satellite="ground", kind="terminal", external_access=True,
                    max_external_links=4)]
    links = []
    for s, satellite in enumerate(owners):
        switch = f"switch{s}"
        devices.append(dict(id=switch, satellite=satellite, kind="switch",
                            capacity_gbps=spec.local_switch_bandwidth_gbps))
        for slot in range(case["cards_per_satellite"]):
            gpu = f"gpu{s}-{slot}"
            devices.append(dict(id=gpu, satellite=satellite, kind="gpu"))
            links.append(dict(id=f"local-{gpu}", source=gpu, target=switch,
                              bandwidth_gbps=spec.local_gpu_bandwidth_gbps,
                              latency_ns=spec.local_latency_ns))
        for side in ("left", "right", "ground"):
            terminal = f"{side}{s}"
            devices.append(dict(id=terminal, satellite=satellite, kind="terminal", external_access=True))
            capacity = spec.ground_bandwidth_gbps if side == "ground" else case["per_direction_gbps"]
            links.append(dict(id=f"port-{terminal}", source=switch, target=terminal,
                              bandwidth_gbps=capacity))
        links.append(dict(id=f"ground{s}", source="user", target=f"ground{s}",
                          bandwidth_gbps=spec.ground_bandwidth_gbps,
                          latency_ns=round(spec.ground_latency_s * 1e9)))
    for s in range(4):
        links.append(dict(id=f"isl{s}", source=f"right{s}", target=f"left{(s + 1) % 4}",
                          bandwidth_gbps=case["per_direction_gbps"],
                          latency_ns=math.ceil(spec.link_distance_km / 299792.458 * 1e9)))
    return PhysicalTopology(Scenario.model_validate(dict(
        cells=[dict(id="formation", satellites=owners), dict(id="ground", satellites=["ground"])],
        satellites=[dict(id=s) for s in [*owners, "ground"]], devices=devices, links=links,
    )))


def native_config(case, spec, model, group, tp, pp, expert, blocks, seed, model_file):
    from astermind.config import AsterMindConfig
    from midlayer.serving.workload.runtime_inputs import ExpertRoutingInput

    collectives = [dict(operation=operation, algorithm="ring", link="rdma", topology="ring",
                        resource_id="physical-network", backend="flowsim")
                   for operation in ("all_reduce", "all_gather")]
    if expert:
        collectives.append(dict(operation="all_to_all", algorithm="pairwise", link="rdma",
                                topology="fully_connected", resource_id="expert-network", backend="flowsim"))
    pool = dict(name="main", role="colocated", hardware_id=case["hardware"],
                parallelism=dict(tensor_parallel_size=tp, pipeline_parallel_size=pp,
                                 enable_expert_parallel=expert), collectives=collectives,
                pipeline_p2p=[dict(source_stage=s, destination_stage=s + 1, algorithm="send_recv",
                                   link="infiniband", topology="point_to_point", resource_id="pipeline-network",
                                   backend="flowsim") for s in range(pp - 1)])
    routing = (ExpertRoutingInput.seeded_assumption("capacity-study-seeded-assumption", root_seed=seed)
               if model.num_experts else ExpertRoutingInput.not_applicable())
    workload = case["workload"]
    model_len = workload["input_tokens"] + workload["output_tokens"]
    resources, local_indices = replica_resources(group)
    return AsterMindConfig.model_validate(dict(
        model=str(model_file),
        requests=dict(source="synthetic", synthetic=dict(num_requests=1, request_rate=1,
                      input_lengths=[workload["input_tokens"]], output_lengths=[workload["output_tokens"]],
                      arrival_distribution="constant", seed=seed)),
        deployment=dict(worker_pools=[pool], placements=[
            dict(worker_pool="main", parallel_coordinate=dict(dp=0, pp=rank // tp, cp=0, tp=rank % tp),
                 satellite_id=f"sat{s}", local_index=local_indices[rank]) for rank, (s, slot) in enumerate(group)
        ]),
        hardware=dict(**resources, default_hardware_id=case["hardware"]),
        scheduler=dict(main=dict(framework="vllm", max_batched_tokens=spec.max_batched_tokens,
                       max_requests=spec.max_concurrency_per_replica, num_kv_blocks=blocks,
                       kv_block_size_tokens=spec.kv_block_size_tokens, scheduler_pipeline_depth=1,
                       vllm=dict(model=str(model_file), dtype={"bf16": "bfloat16", "fp16": "float16"}[model.compute_dtype],
                                 max_model_len=model_len, gpu_memory_utilization=spec.memory_utilization,
                                 enable_prefix_caching=False))),
        runtime_inputs=dict(expert_routing=asdict(routing),
                            attention_kv_layout="mla" if model.attention_layout in ("mla", "dsa_mla") else "paged_varlen",
                            attention_kv_distribution="uniform"),
        flowsim=dict(acknowledgement=dict(enabled=spec.acknowledgements_enabled,
                     packet_payload_bytes=spec.ack_packet_payload_bytes,
                     ack_interval_packets=spec.ack_every_packets, ack_size_bytes=spec.ack_size_bytes)),
        topology=dict(source="file", file="host-owned-topology.json"),
        runtime=dict(timing_mode="device_only", timeout_seconds=spec.timeout_s),
    ))


def prepare(payload):
    repository, vendor, native = activate(payload["astermind_root"])
    from astermind.engine import resolve_model_deployment
    from midlayer.kernel.hardware_ids import CANONICAL_HARDWARE_IDS
    from midlayer.serving.kv_cache.model_residency import model_rank_residencies
    from midlayer.serving.parallelism.parallel_config import ParallelConfig

    spec, case = StudySpec.model_validate(payload["spec"]), payload["case"]
    expected = payload.get("source_identity_sha256")
    if expected is not None and digest(source_identity(repository, spec)) != expected:
        raise ValueError("study source/dependency identity changed before worker execution")
    if payload["model"] is None:
        raise FileNotFoundError("model has no catalog entry")
    model_file = Path(payload["model"]["file"])
    if hashlib.sha256(model_file.read_bytes()).hexdigest() != payload["model"]["sha256"]:
        raise ValueError("model changed since study planning")
    if case["hardware"] not in CANONICAL_HARDWARE_IDS:
        raise NotImplementedError(f"external runtime has no registered hardware: {case['hardware']}")
    try:
        model = _load_model(model_file)
    except ValueError as error:
        if "unsupported model_type" in str(error):
            raise NotImplementedError(f"external runtime model support: {error}") from error
        raise
    tp, pp, expert, groups = placements(case, model.model_type)
    config = native_config(case, spec, model, groups[0], tp, pp, expert, 1, payload["seed"], model_file)
    resolve_model_deployment(config, repository)  # Native support gate, before any Scheduler startup.
    hardware_file = vendor / "configs/hardware" / (case["hardware"] + ".yaml")
    hardware = _load_hardware(hardware_file)
    residents = model_rank_residencies(model, ParallelConfig(
        tensor_parallel_size=tp, pipeline_parallel_size=pp, enable_expert_parallel=expert), worker_role="colocated")
    hbm = hardware.resource_by_role("dram").size_bytes
    reserve = math.ceil(hbm * spec.non_kv_reserve_fraction)
    usable = math.floor(hbm * spec.memory_utilization)
    ranks = []
    for rank in residents:
        block_bytes = spec.kv_block_size_tokens * model.kv_bytes_per_token_for_layer_range(
            tp, rank.layer_start, rank.layer_end)
        free = usable - reserve - rank.weight_resident_bytes
        ranks.append(dict(rank=rank.rank, weight_bytes=rank.weight_resident_bytes,
                          kv_block_bytes=block_bytes, available_kv_bytes=free,
                          physical_block_limit=max(0, free // block_bytes)))
    physical_limit = min(r["physical_block_limit"] for r in ranks)
    per_request = math.ceil((case["workload"]["input_tokens"] + case["workload"]["output_tokens"]) / spec.kv_block_size_tokens)
    if physical_limit < per_request:
        return dict(status="memory_infeasible", reason="at least one rank cannot hold weights, reserve and one full request",
                    memory=dict(hbm_per_rank_bytes=hbm, reserve_bytes=reserve, ranks=ranks))
    # Limit the CPU allocator to the declared concurrency, without exceeding any physical rank.
    blocks = min(physical_limit, spec.max_concurrency_per_replica * per_request)
    configs = [native_config(case, spec, model, group, tp, pp, expert, blocks,
                             payload["seed"], model_file) for group in groups]
    for config in configs:
        resolve_model_deployment(config, repository)
    memory = dict(mode="analytical_hbm_with_assumed_non_kv_reserve", calibrated=False,
                  hbm_per_rank_bytes=hbm, utilization=spec.memory_utilization,
                  reserve_fraction=spec.non_kv_reserve_fraction, reserve_bytes=reserve,
                  num_kv_blocks=blocks, physical_block_limit=physical_limit, ranks=ranks,
                  allocation_scope="one logical block pool bounded by the tightest physical rank")
    provenance = dict(
        model_config_sha256=payload["model"]["sha256"], hardware_sha256=hashlib.sha256(hardware_file.read_bytes()).hexdigest(),
        native_module=str(native.__file__), native_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        model_type=model.model_type, compute_dtype=model.compute_dtype, weight_dtype=model.weight_dtype,
        attention_layout=model.attention_layout,
        acknowledgement=dict(enabled=spec.acknowledgements_enabled,
                             packet_payload_bytes=spec.ack_packet_payload_bytes,
                             ack_interval_packets=spec.ack_every_packets, ack_size_bytes=spec.ack_size_bytes),
        tp=tp, pp=pp, expert_parallel=expert, replicas=len(groups),
        gpu_count=4 * case["cards_per_satellite"], energy="not_evaluated", timing_mode="device_only",
        physical_scope="declared static four-satellite ring with distance/c propagation; not an orbital availability result",
        accuracy="native prediction; model/hardware-specific held-out accuracy is not established by this study",
    )
    return dict(status="ready", configs=configs, groups=groups, memory=memory,
                provenance=provenance, model=model, repository=repository)


class NetworkLog(deque):
    def __init__(self, stream, topology, spec):
        super().__init__(maxlen=1000)
        self.stream, self.topology, self.spec = stream, topology, spec
        self.links = defaultdict(lambda: dict(bytes=0.0, capacity_bps=0.0, samples=0))

    def append(self, row):
        if row["kind"] == "bandwidth_update":
            start, end = row["interval_start_s"], row["interval_end_s"]
            lo, hi = self.spec.warmup_s, self.spec.warmup_s + self.spec.measurement_s
            overlap = max(0.0, min(end, hi) - max(start, lo))
            if end > start and overlap:
                a = self.topology.node_device[row["source_node_id"]]
                b = self.topology.node_device[row["target_node_id"]]
                owners = self.topology.devices[a].satellite, self.topology.devices[b].satellite
                category = "ground" if "ground" in owners else "isl" if owners[0] != owners[1] else "local"
                key = f"{category}:{row['link_id']}:{row['source_node_id']}->{row['target_node_id']}"
                item = self.links[key]
                item["bytes"] += row["bytes_sent"] * overlap / (end - start)
                item["capacity_bps"] = max(item["capacity_bps"], row["link_capacity_bps"])
                item["samples"] += 1
            self.stream.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
        super().append(row)


class Host:
    def __init__(self, engine, executor):
        self.engine, self.executor = engine, executor

    @property
    def now_ns(self):
        return self.engine.now * 1e9

    def schedule(self, delay_ns, callback, priority=0):
        self.engine.push(self.engine.now + delay_ns / 1e9, 30 + priority, "external_callback", payload=callback)

    def start_flow(self, flow_id, source, target, size_bytes, callback, is_control=False):
        self.executor.submit_transport(flow_id, str(source), str(target), size_bytes,
                                       "native_collective", flow_id, callback, lambda event: None)


def _union_length(intervals):
    end, total = -1.0, 0.0
    for a, b in sorted(intervals):
        total += max(0, b - max(end, a))
        end = max(end, b)
    return total


def close_resources(resources, errors, *, allow_pending=False):
    """Reap every owned resource without hiding the original simulation error."""
    active_error = sys.exc_info()[0] is not None
    first = None
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            expected_pending = allow_pending and "cannot close vLLM Scheduler with pending steps" in str(error)
            if not expected_pending and first is None:
                first = error
    if first is not None and not active_error:
        raise first


def execute(payload, directory):
    started = time.perf_counter()
    try:
        prepared = prepare(payload)
    except NotImplementedError as error:
        return dict(status="unsupported", reason=f"{type(error).__name__}: {error}")
    except FileNotFoundError as error:
        return dict(status="missing_dependency", reason=str(error))
    if prepared["status"] != "ready":
        return prepared
    if payload.get("preflight_only"):
        return dict(status="ready", memory=prepared["memory"], provenance=prepared["provenance"])
    from astermind.embedded import EmbeddedSession
    from midlayer.serving.backend import vllm_scheduler_process
    from midlayer.serving.request_gen.request_stream import RequestSpec, RequestStream
    from simulator.network import CellTransferExecutor, DeviceCommunicationRuntime, DeviceRuntimeHost

    spec, case = StudySpec.model_validate(payload["spec"]), payload["case"]
    # Process startup is wall time, excluded from every simulated latency. Slow
    # host filesystems may need a longer budget; source pins/device guards remain.
    vllm_scheduler_process._STARTUP_TIMEOUT_S = spec.scheduler_startup_timeout_s
    import astropy
    import astropy.coordinates
    import astropy.units
    import numpy
    if (numpy.__version__, astropy.__version__) != ("2.2.6", "6.1.3"):
        return dict(status="environment_error", reason="execution requires repository NumPy 2.2.6 / Astropy 6.1.3 pins")
    prepared["provenance"]["numerical_environment"] = dict(numpy=numpy.__version__, astropy=astropy.__version__)
    prepared["provenance"]["python_version"] = sys.version
    source = arrivals(spec, payload["rate"], payload["seed"], case["workload"])
    if not source:
        return dict(status="insufficient_population", reason="no arrivals in the finite trial")
    topology = build_topology(case, spec)
    write_json(directory / "topology.json", topology.scenario.model_dump(mode="json"))
    write_json(directory / "memory.json", prepared["memory"])
    write_json(directory / "provenance.json", prepared["provenance"])
    horizon = spec.warmup_s + spec.measurement_s + spec.drain_s
    engine = DeviceRuntimeHost(topology, SimpleNamespace(
        horizon_ns=math.ceil(horizon * 1e9), backend="flowsim", physical_link_bindings={},
        runtime_network={}, ground_owners=("ground",), record_packets=False))
    network = DeviceCommunicationRuntime(engine, topology)
    executor = CellTransferExecutor(engine.timeline, network)
    # Terminal histories are diagnostic only; active protocol state remains native.
    executor.transfer_events = deque(maxlen=1000)
    network.events = deque(maxlen=1000)
    records = {row["request_id"]: {**row, "delivered_tokens": 0, "produced_tokens": 0} for row in source}
    kernel_intervals, events, counts = defaultdict(list), {}, Counter()
    sessions, streams = [], []
    result = None
    failures, cleanup_errors = [], []
    network.subscribe(lambda event: failures.append(event.model_dump()) if event.kind == "failed" else None)
    stream = gzip.open(directory / "network.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
    kernel_stream = gzip.open(directory / "operators.jsonl.gz", "wt", encoding="utf-8", compresslevel=1)
    engine.log = log = NetworkLog(stream, topology, spec)

    def emission(replica, kind, row):
        counts[kind] += 1
        if kind == "execution_timeline":
            selected = {k: row[k] for k in ("call_id", "rank", "kind", "start_time_s", "end_time_s", "dependencies")}
            kernel_stream.write(json.dumps(dict(replica=replica, **selected), default=str, separators=(",", ":")) + "\n")
            if str(row["kind"]).upper() not in ("COMMUNICATION", "KERNELKIND.COMMUNICATION"):
                a, b = max(spec.warmup_s, row["start_time_s"]), min(spec.warmup_s + spec.measurement_s, row["end_time_s"])
                if b > a:
                    kernel_intervals[replica, row["rank"]].append((a, b))
        elif kind == "scheduler_steps":
            events[replica] = events.get(replica, 0) + 1

    def finish(rid):
        record = records[rid]
        if record["delivered_tokens"] == record["output_tokens"] and "compute_completed_s" in record:
            record["completed_s"] = max(record["last_received_s"], record["compute_completed_s"])

    def tokens(replica, rid, count, now):
        record = records[rid]
        offset = record["produced_tokens"]
        record["produced_tokens"] += count
        group = prepared["groups"][replica]
        tp, pp = prepared["provenance"]["tp"], prepared["provenance"]["pp"]
        s, slot = group[tp * (pp - 1)]

        def received():
            record["delivered_tokens"] += count
            if offset == 0:
                record["first_received_s"] = engine.now
            record["last_received_s"] = max(engine.now, record.get("last_received_s", 0))
            finish(rid)

        executor.submit_transport(f"out-{rid}-{offset}", f"gpu{s}-{slot}", "user", count * 4,
                                   "request_output", rid, received, lambda event: None)

    def completed(rid, now):
        records[rid]["compute_completed_s"] = engine.now
        finish(rid)

    # Arrival order and round-robin assignment are held constant across compared configurations.
    assignment = {row["request_id"]: i % len(prepared["groups"]) for i, row in enumerate(source)}
    host = Host(engine, executor)
    try:
        for replica, (config, group) in enumerate(zip(prepared["configs"], prepared["groups"])):
            requests = [RequestSpec(r["arrival_s"], r["request_id"], r["input_tokens"], r["output_tokens"])
                        for r in source if assignment[r["request_id"]] == replica]
            if not requests:
                sessions.append(None)
                continue
            requests = RequestStream(requests)
            streams.append(requests)
            write_json(directory / f"replica-{replica}.json", config.model_dump(mode="json"))
            print(f"starting native Scheduler {replica + 1}/{len(prepared['groups'])}", flush=True)
            sessions.append(EmbeddedSession(
                config, requests, simulator=host,
                rank_mapping={r: f"gpu{s}-{slot}" for r, (s, slot) in enumerate(group)},
                namespace=f"replica-{replica}", emit=lambda kind, row, replica=replica: emission(replica, kind, row),
                on_tokens=lambda rid, count, now, replica=replica: tokens(replica, rid, count, now),
                on_complete=completed, repository=prepared["repository"],
            ))
        startup_s = time.perf_counter() - started
        for row in source:
            rid, replica = row["request_id"], assignment[row["request_id"]]
            s, slot = prepared["groups"][replica][0]

            def arrive(rid=rid, replica=replica, s=s, slot=slot, row=row):
                def ready():
                    records[rid]["cell_ready_s"] = engine.now
                    sessions[replica].submit(rid)
                executor.submit_transport(f"in-{rid}", "user", f"gpu{s}-{slot}", row["input_tokens"] * 4,
                                           "request_input", rid, ready, lambda event: None)
            engine.push(row["arrival_s"], 33, "external_callback", payload=arrive)
        print(f"native run: {len(source)} requests, {len(sessions)} replicas", flush=True)
        engine.run()
        for record in records.values():
            if record["delivered_tokens"] > record["output_tokens"]:
                raise RuntimeError("duplicate or excess token delivery")
        values = list(records.values())
        write_json(directory / "requests.json", values)
        metrics = trial_metrics(values, spec, payload["rate"])
        links = []
        for name, item in log.links.items():
            links.append(dict(link=name, **item,
                              utilization=item["bytes"] * 8 / (spec.measurement_s * item["capacity_bps"])
                              if item["capacity_bps"] else None))
        gpu_active = sum(_union_length(v) for v in kernel_intervals.values()) / (spec.measurement_s * case["total_cards"])
        runtime = []
        for session in sessions:
            if session:
                for backend in session.backends.values():
                    value = getattr(backend, "runtime_info", None)
                    runtime.append(asdict(value) if value is not None else None)
        result = dict(status="evaluated", metrics=metrics, memory=prepared["memory"],
                      provenance={**prepared["provenance"], "scheduler_runtime": runtime,
                                  "arrival_stream_sha256": digest(source), "routing": "fixed_round_robin_by_arrival_order",
                                  "payload_bytes_per_text_token": 4, "payload_scope": "explicit text-only traffic assumption"},
                      diagnostics=dict(links=links, gpu_kernel_active_fraction=gpu_active,
                                       emit_counts=dict(counts), scheduler_steps=events,
                                       cleanup_errors=cleanup_errors,
                                       failed_transfers=failures[:20], failed_transfer_count=len(failures)),
                      measurement=dict(startup_wall_s=startup_s, total_wall_s=time.perf_counter() - started,
                                       horizon_s=horizon, stopped_at_s=engine.now))
        return result
    finally:
        close_resources([*sessions, network, engine, stream, kernel_stream], cleanup_errors,
                        allow_pending=any("completed_s" not in r for r in records.values()))
        if result is not None:
            write_json(directory / "diagnostics.json", result["diagnostics"])
