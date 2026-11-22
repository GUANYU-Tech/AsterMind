"""Lower a user-facing scenario to the existing native configuration offline."""

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import Field

from astermind.common.contracts import Contract, Identifier, Nanoseconds

from .deployment import Positive, ServingCell
from .requests import Migration

SCHEMA = "astermind.scenario.v1"
NETWORK_SCHEMA = "orbitserve.network_experiment.v1"
Seconds = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class ScenarioTime(Contract):
    epoch_utc: datetime
    duration_s: Seconds


class Region(Contract):
    name: str
    lat_deg: Annotated[float, Field(ge=-90, le=90)]
    lon_deg: Annotated[float, Field(ge=-180, le=180)]


class ServiceLevel(Contract):
    ttft_s: Seconds | None = None
    e2e_s: Seconds | None = None


class WorkloadManifest(Contract):
    schema_version: Literal["astermind.workload.v1"]
    workload_id: Identifier
    requests: str
    input_bytes_per_token: Nanoseconds
    output_bytes_per_token: Nanoseconds
    regions: dict[Identifier, Region] = Field(default_factory=dict)
    gateway_regions: dict[Identifier, tuple[Identifier, ...]] = Field(default_factory=dict)
    slo: dict[Identifier, ServiceLevel] = Field(default_factory=dict)


class WorkloadRow(Contract):
    id: Identifier
    arrival_s: Seconds
    gateway_id: Identifier
    region_id: Identifier | None = None
    session_id: Identifier | None = None
    turn: Positive | None = None
    new_input_tokens: Positive
    history_tokens: Nanoseconds = 0
    output_tokens: Positive
    slo_class: Identifier | None = None


class RequestDecision(Contract):
    cell: Identifier
    history_upload_tokens: Nanoseconds | None = None


class SchedulePlan(Contract):
    requests: dict[Identifier, RequestDecision] = Field(default_factory=dict)
    migrations: tuple[Migration, ...] = ()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _ns(seconds):
    value = Decimal(str(seconds)) * 1_000_000_000
    if value != value.to_integral_value():
        raise ValueError("scenario times must have an exact nanosecond representation")
    return int(value)


def _explicit_ground_metadata(network):
    if "constellations" in network:
        return network
    settings = network["network"]
    owners = {node["node_id"]: node for node in settings["nodes"] if node["role"] == "ground"}
    stations = []
    for station in network["ground_ingresses"]:
        station = dict(station)
        for source, field in (("terminal_count", "sgl_terminals"), ("ground_kind", "ground_kind")):
            if source not in station:
                continue
            identity = station["ingress_id"]
            if identity not in owners:
                raise ValueError(f"unknown ground station {identity}")
            node = owners[identity]
            value = station.pop(source)
            if node.get(field, value) != value:
                raise ValueError(f"ground station {identity} has conflicting {field}")
            owners[identity] = {**node, field: value}
        stations.append(station)
    nodes = [
        owners[node["node_id"]] if node["role"] == "ground" else node
        for node in settings["nodes"]
    ]
    return {**network, "network": {**settings, "nodes": nodes}, "ground_ingresses": stations}



def resolve_network_input(value, base, time):
    from astermind.builder.physical.network_inputs import snapshot_network_input
    from astermind.builder.physical.physical_dataset import LEGACY_SCHEMA, SCHEMA as DATASET_SCHEMA

    path = (base / value).resolve() if isinstance(value, str) else None
    network = _read_json(path) if path is not None else value
    if not isinstance(network, dict):
        raise ValueError("physical_network must be a Network configuration or dataset manifest")
    epoch = time.epoch_utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if network.get("schema_version") in {LEGACY_SCHEMA, DATASET_SCHEMA}:
        if path is None or path.name != "physical_dataset.json":
            raise ValueError("set physical_network to the physical_dataset.json file")
        prepared = _read_json(path.parent / "materialized_input.json")
        recorded_time = prepared["time"]
        if datetime.fromisoformat(recorded_time["epoch_utc"]) != time.epoch_utc:
            raise ValueError("scene epoch does not match the precomputed physical dataset")
        if _ns(recorded_time["duration_s"]) < _ns(time.duration_s):
            raise ValueError("scene duration exceeds the precomputed physical dataset")
        return "physical_dataset", str(path), prepared["ground_ingresses"]

    if network.get("schema_version") != NETWORK_SCHEMA:
        raise ValueError("unrecognized physical_network file schema")
    settings = network.get("time", {})
    if "epoch_utc" in settings and datetime.fromisoformat(settings["epoch_utc"]) != time.epoch_utc:
        raise ValueError("Network epoch does not match scene time")
    if "duration_s" in settings and _ns(settings["duration_s"]) != _ns(time.duration_s):
        raise ValueError("Network duration does not match scene time")
    network = {
        **network,
        "time": {**settings, "epoch_utc": epoch, "duration_s": float(time.duration_s)},
    }
    prepared = snapshot_network_input(_explicit_ground_metadata(network), path.parent if path else base)
    return "physical_network", prepared, prepared["ground_ingresses"]


def _add_default_local_links(physical):
    """Use the scene builder's default local fabric for an unwired satellite."""
    from .deployment import SceneDeployment

    devices = {device["id"]: device for device in physical["devices"]}
    links = physical.setdefault("links", [])
    wired = {
        devices[link["source"]]["satellite"]
        for link in links
        if devices[link["source"]]["satellite"] == devices[link["target"]]["satellite"]
    }
    wired.update(
        devices[identifier]["satellite"]
        for template in physical.get("templates", ())
        for identifier in template["devices"]
    )
    by_owner = {}
    for device in devices.values():
        by_owner.setdefault(device["satellite"], []).append(device)

    defaults = SceneDeployment()
    for owner, members in by_owner.items():
        if owner in wired:
            continue
        gpus = [device["id"] for device in members if device["kind"] == "gpu"]
        terminals = [
            device["id"] for device in members
            if device["kind"] == "terminal" and device.get("external_access")
        ]
        if not gpus or len(terminals) != 1:
            continue
        fabric = f"{owner}/fabric"
        if fabric not in devices:
            physical["devices"].append({
                "id": fabric, "satellite": owner, "kind": "switch",
                "capacity_gbps": len(gpus) * defaults.gpu_link_bandwidth_gbps,
            })
        elif devices[fabric]["satellite"] != owner or devices[fabric]["kind"] != "switch":
            raise ValueError(f"default local fabric ID conflicts with device {fabric}")
        links.append({
            "id": f"{owner}/egress", "source": fabric, "target": terminals[0],
            "bandwidth_gbps": defaults.gpu_link_bandwidth_gbps,
        })
        for gpu in gpus:
            links.append({
                "id": f"{gpu}/port", "source": gpu, "target": fabric,
                "bandwidth_gbps": defaults.gpu_link_bandwidth_gbps,
                "latency_ns": defaults.gpu_link_latency_ns,
            })


def _add_dataset_links(data, manifest):
    """Project prepared Network edges when the user declared only local wiring."""
    from astermind.builder.config.network import NetworkSpec
    from astermind.builder.physical.physical_dataset import load_dataset
    from astermind.engine.state.physical_dataset import PhysicalAttributes
    from astermind.engine.topology import candidate_links

    physical = data["physical"]
    devices = {device["id"]: device for device in physical["devices"]}
    links = physical.setdefault("links", [])
    if data.get("physical_link_bindings") or any(
        devices[link["source"]]["satellite"] != devices[link["target"]]["satellite"]
        for link in links
    ):
        return

    root, _, _ = load_dataset(manifest)
    spec = NetworkSpec.from_value(_read_json(root / "materialized_input.json")["network"])
    attributes = PhysicalAttributes(root / "physical_state.sqlite")
    prepared_ids = attributes.metadata["link_frames"]
    terminals = {}
    for device in physical["devices"]:
        if device["kind"] == "terminal" and device.get("external_access"):
            terminals.setdefault(device["satellite"], []).append(device["id"])

    bindings = {}
    terminal_degrees = {}
    for edge in candidate_links(spec):
        physical_id = edge["link_id"]
        if physical_id not in prepared_ids:
            continue
        endpoints = (edge["endpoint_a"], edge["endpoint_b"])
        if any(len(terminals.get(owner, ())) != 1 for owner in endpoints):
            raise ValueError(f"Network link {physical_id} needs explicit device terminal endpoints")
        source, target = (terminals[owner][0] for owner in endpoints)
        for terminal in (source, target):
            terminal_degrees[terminal] = terminal_degrees.get(terminal, 0) + 1
        link_id = f"network:{physical_id}"
        links.append({
            "id": link_id,
            "source": source,
            "target": target,
            "bandwidth_gbps": spec.link_profiles[edge["profile_id"]].bandwidth_bps / 1e9,
        })
        bindings[link_id] = physical_id
    for terminal, count in terminal_degrees.items():
        devices[terminal].setdefault("max_external_links", count)
    data["physical_link_bindings"] = bindings


def _cell_inputs(data, base):
    from astermind.common.bootstrap import DEFAULT_ML

    physical = data["physical"]
    devices = {device["id"]: device for device in physical["devices"]}
    communication = data.setdefault("communication", {})
    policies = communication.setdefault("cell_policies", {})
    gpu_hardware = communication.setdefault("gpu_hardware", {})
    owners = {}
    for cell in data["cells"]:
        identity = cell["id"]
        hardware_id = cell.get("hardware_id", "nvidia_a800")
        profile = f"configs/hardware/{hardware_id}.yaml"
        if "hardware" not in cell:
            if not (DEFAULT_ML / profile).is_file():
                raise ValueError(f"Cell {identity} needs a hardware file for {hardware_id}")
            cell["hardware"] = profile
        for key in ("model", "hardware", "host_launch_profile", "memory_policy_file"):
            reference = cell.get(key)
            if reference and (base / reference).is_file():
                cell[key] = str((base / reference).resolve())

        policy = policies.setdefault(identity, {})
        if policy.get("hardware_id", hardware_id) != hardware_id:
            raise ValueError(f"Cell {identity} must use the same GPU model for compute and Comm")
        policy["hardware_id"] = hardware_id

        member_ids = [placement["device"] for placement in cell["placements"]]
        for identifier in member_ids:
            override = gpu_hardware.setdefault(identifier, hardware_id)
            if override != hardware_id:
                raise ValueError(f"GPU {identifier} has conflicting compute and Comm models")
        for identifier in member_ids + cell["external_terminals"]:
            if identifier not in devices:
                raise ValueError(f"Cell {identity} references unknown device {identifier}")
            satellite = devices[identifier]["satellite"]
            if owners.setdefault(satellite, identity) != identity:
                raise ValueError(f"satellite {satellite} cannot host multiple Cells in the native runtime")

    if "cells" not in physical:
        members = {cell["id"]: [] for cell in data["cells"]}
        for satellite, cell_id in owners.items():
            members[cell_id].append(satellite)
        others = [
            node["id"] for node in physical["satellites"] if node["id"] not in owners
        ]
        if others:
            if "scene-network-resources" in members:
                raise ValueError("scene-network-resources is reserved for unassigned physical nodes")
            members["scene-network-resources"] = others
        physical["cells"] = [
            {"id": identity, "satellites": satellites}
            for identity, satellites in members.items()
        ]

    if communication.get("intra_profile_file"):
        communication["intra_profile_file"] = str(
            (base / communication["intra_profile_file"]).resolve()
        )


def _validate_reused_prefix(request, reused, previous, decisions, cells, deferred):
    decision = decisions[request.id]
    if previous is None or previous[1] not in decisions:
        raise ValueError("planned KV reuse needs explicit Cell assignments for both session turns")
    _, parent, _, history_length = previous
    target = cells.get(decision.cell)
    if target is None:
        raise ValueError(f"unknown target Cell {decision.cell}")
    if not target.prefix_caching:
        raise ValueError("planned KV reuse requires prefix caching on the target Cell")
    if reused > history_length or reused % target.kv_block_size_tokens:
        raise ValueError("planned KV reuse must match complete blocks in the previous context")
    source = decisions[parent].cell
    if source != decision.cell and (source, decision.cell, request.session_id, parent) not in deferred.get(request.id, ()):
        raise ValueError("cross-Cell KV reuse requires a migration dependency for this request")


def _workload_rows(path, data, decisions):
    from astermind.builder.request_batch import ground_devices, parse_rows, resolve_ground

    manifest = WorkloadManifest.model_validate(_read_json(path))
    request_path = path.parent / manifest.requests
    rows = parse_rows(request_path.read_text(encoding="utf-8"), "jsonl")
    gateways = ground_devices(SimpleNamespace(
        ground_owners=data.get("ground_owners", ()), physical=data["physical"]
    ))
    cells = {
        cell["id"]: ServingCell.model_validate(cell)
        for cell in data["cells"]
    }
    deferred = {}
    for item in data.get("migrations", ()):
        migration = Migration.model_validate(item)
        if migration.before_request is not None:
            deferred.setdefault(migration.before_request, []).append((
                migration.source_cell, migration.target_cell, migration.session_id, migration.context_id,
            ))
    for gateway, regions in manifest.gateway_regions.items():
        resolve_ground(gateway, gateways)
        if unknown := set(regions) - manifest.regions.keys():
            raise ValueError(f"gateway {gateway} references unknown regions: {sorted(unknown)}")

    session_tips = {}
    used_decisions = set()
    requests = []
    for line, raw in rows:
        try:
            if "wait_after_response_s" in raw:
                raise ValueError("closed-loop arrivals require runtime support; give an offline arrival_s")
            request = WorkloadRow.model_validate(raw)
            if request.region_id is not None and request.region_id not in manifest.regions:
                raise ValueError(f"unknown region {request.region_id}")
            served_regions = manifest.gateway_regions.get(request.gateway_id)
            if request.region_id is not None and served_regions is not None and request.region_id not in served_regions:
                raise ValueError(f"gateway {request.gateway_id} does not serve region {request.region_id}")
            if request.slo_class is not None and request.slo_class not in manifest.slo:
                raise ValueError(f"unknown SLO class {request.slo_class}")
            decision = decisions.get(request.id)
            if decision is not None:
                used_decisions.add(request.id)
            arrival = _ns(request.arrival_s)
            parent = None
            previous = None
            if request.session_id is not None:
                previous = session_tips.get(request.session_id)
                expected_turn = 1 if previous is None else previous[0] + 1
                if request.turn != expected_turn or (previous is not None and arrival <= previous[2]):
                    raise ValueError("session turns must be consecutive with increasing arrivals")
                parent = previous[1] if previous is not None else None
                session_tips[request.session_id] = (
                    request.turn, request.id, arrival,
                    request.history_tokens + request.new_input_tokens + request.output_tokens - 1,
                )
            elif request.turn is not None:
                raise ValueError("a turn requires a session ID")

            upload = decision.history_upload_tokens if decision is not None else None
            if upload is None:
                upload = request.history_tokens
            if upload > request.history_tokens:
                raise ValueError("history upload exceeds the request history")
            reused = request.history_tokens - upload
            if reused:
                _validate_reused_prefix(request, reused, previous, decisions, cells, deferred)
            requests.append({
                "id": request.id,
                "arrival_ns": arrival,
                "user_device": resolve_ground(request.gateway_id, gateways),
                "input_tokens": request.new_input_tokens + request.history_tokens,
                "output_tokens": request.output_tokens,
                "input_bytes": (request.new_input_tokens + upload) * manifest.input_bytes_per_token,
                "output_bytes_per_token": manifest.output_bytes_per_token,
                "cell": decision.cell if decision is not None else None,
                "session_id": request.session_id,
                "context_id": request.id if request.session_id is not None else None,
                "parent_context_id": parent,
                "prefix_hit_tokens": reused,
            })
        except (TypeError, ValueError) as error:
            raise ValueError(f"{request_path}:{line}: {error}") from error
    if unused := decisions.keys() - used_decisions:
        raise ValueError(f"schedule references unknown requests: {sorted(unused)}")
    return requests


def parse_time(value):
    time = ScenarioTime.model_validate(value)
    if time.epoch_utc.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("scenario epoch must be UTC")
    if not time.duration_s:
        raise ValueError("scenario duration must be positive")
    _ns(time.duration_s)
    return time


def lower_scenario(data, base):
    """Resolve input files and emit only fields understood by the native engine."""
    time = parse_time(data.pop("time"))
    if set(data) & {"horizon_ns", "physical_dataset"}:
        raise ValueError("use time and the single physical_network input in this schema")
    data["schema_version"] = 1
    data["horizon_ns"] = _ns(time.duration_s)
    network = data.pop("physical_network", None)
    if network is not None:
        key, source, stations = resolve_network_input(network, base, time)
        data[key] = source
        ground = [station["ingress_id"] for station in stations]
        if "ground_owners" in data and set(data["ground_owners"]) != set(ground):
            raise ValueError("scene ground owners must match the Network station IDs")
        data["ground_owners"] = ground
        if key == "physical_dataset":
            _add_dataset_links(data, source)
    _add_default_local_links(data["physical"])
    _cell_inputs(data, base)

    workload = data.pop("workload", None)
    schedule = data.pop("schedule", None)
    plan = SchedulePlan()
    if schedule is not None:
        if workload is None:
            raise ValueError("an offline schedule requires a workload")
        plan = SchedulePlan.model_validate(_read_json((base / schedule).resolve()))
        if plan.migrations:
            if "migrations" in data:
                raise ValueError("declare migrations in the scene or the offline schedule")
            data["migrations"] = plan.migrations
    if workload is not None:
        if "requests" in data:
            raise ValueError("supply workload or inline requests")
        data["requests"] = _workload_rows((base / workload).resolve(), data, plan.requests)
    return data
