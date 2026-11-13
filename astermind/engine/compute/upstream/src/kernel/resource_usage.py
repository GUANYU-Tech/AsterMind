"""Read-only analytical demand evidence for optional execution recording.

Demand and compulsory bytes come from the predictor owner. No timing is
modified and unavailable source contracts remain explicitly unavailable.
"""
import json
import math
from pathlib import Path

from src.kernel.kernel_base import create_predictor


def report_memory_bandwidth_reference(hardware_id):
    """Independent peak evidence for display only; never changes prediction."""
    if hardware_id != "nvidia_h200":
        return None
    path = Path(__file__).resolve().parents[2] / "microbench/nvidia_h200/references/l2_peak_metadata_sm90.json"
    reference = json.loads(path.read_text())
    if reference["version"] != 1 or reference["hardware_id"] != hardware_id:
        raise ValueError("invalid report memory bandwidth reference identity")
    clock = reference["clock_hz"]
    l2 = reference["l2"]
    if not all(math.isfinite(value) and value > 0 for value in (
            clock, l2["read_bytes_per_cycle"], l2["write_bytes_per_cycle"])):
        raise ValueError("invalid report memory bandwidth reference peak")
    source = l2["source"]
    if (l2["read_bytes_per_cycle"] != source["read_sectors_per_cycle"] * source["sector_bytes"]
            or l2["write_bytes_per_cycle"] != source["write_sectors_per_cycle"] * source["sector_bytes"]
            or l2["read_bytes_per_cycle"] != source["xbar_peak"] * source["sector_bytes"]):
        raise ValueError("report memory bandwidth peak disagrees with NCU metadata")
    hbm = reference["hbm"]
    if not math.isfinite(hbm["shared_bytes_per_second"]) or hbm["shared_bytes_per_second"] <= 0:
        raise ValueError("invalid HBM shared bandwidth peak")
    return {"l2": {"read_bytes_per_second": l2["read_bytes_per_cycle"] * clock,
        "write_bytes_per_second": l2["write_bytes_per_cycle"] * clock,
        "combined": "max_directional", "basis": l2["basis"], "source": l2["source"]},
        "hbm": {"read_bytes_per_second": hbm["shared_bytes_per_second"],
        "write_bytes_per_second": hbm["shared_bytes_per_second"],
        "combined": "shared_sum", "basis": hbm["basis"], "source": hbm["source"]}}


def kernel_resource_usage(kernel, hardware) -> dict:
    """Export modeled work/traffic and configured HWSpec bandwidths, with units."""
    if kernel.launch is None:
        return {"availability": "unavailable", "reason": "resource-input export requires native launch metadata; this operation has only mathematical execution support"}
    predictor = create_predictor(kernel.kind)
    try:
        inputs, compulsory_bytes = predictor.resource_input(kernel, hardware)
    except NotImplementedError as error:
        return {"availability": "unavailable", "reason": str(error)}
    return _project_resource_usage(kernel, hardware, inputs, compulsory_bytes)


def _project_resource_usage(kernel, hardware, inputs, compulsory_bytes) -> dict:
    """Project an already constructed public input using the same evidence rules."""
    compute = []
    for role, amount in inputs.demand_per_resource.items():
        if role.startswith("compute_") and amount > 0:
            compute.append({"role": role, "operations": float(amount),
                "capacity_operations_per_s": hardware.resource_by_role(role).capacity})
    # These aggregate nodes are produced by the predictor's memory accounting,
    # including its cache/LRU replay where that path uses replay. Never use
    # compulsory logical bytes or resource durations as replacement traffic.
    memory_traffic = {}
    directional_pair = {path.access_kind for path in inputs.memory_paths} == {"read", "write"}
    for name, role in (("l2", "l2_cache"), ("hbm", "dram")):
        directional = {}
        for direction in ("read", "write"):
            nodes = [node for path in inputs.memory_paths if path.access_kind == direction
                for node in path.nodes if node.resource_role == role]
            # MemoryNodeDemand requires strictly positive bytes. The owner
            # omits a downstream node when the exported read/write ledger has
            # zero traffic there (e.g. an LRU hit or retained dirty output).
            # This is a proven zero only with both directional paths present.
            directional[direction + "_bytes"] = sum(node.bytes for node in nodes) if nodes or directional_pair else None
        complete = all(value is not None for value in directional.values())
        memory_traffic[name] = {**directional,
            "availability": "available" if complete else "unavailable",
            "capacity_bytes_per_s": hardware.resource_by_role(role).capacity if complete else None,
            "basis": "predictor aggregate memory-path bytes; same HWSpec configured bandwidth denominator for read, write and total",
            "source": "PredictionInput.memory_paths; MemoryNodeDemand.bytes; role=" + role,
            "limitations": ([] if complete else ["read_or_write_path_not_exported; unknown is not zero"])}
    read = memory_traffic["l2"]["read_bytes"]
    l2_traffic = ({"availability": "available", "read_bytes": read,
        "write_bytes": memory_traffic["l2"]["write_bytes"],
        "read_capacity_bytes_per_s": hardware.resource_by_role("l2_cache").capacity,
        "basis": "modeled aggregate read-path bytes / configured l2_cache bandwidth",
        "source": "PredictionInput.memory_paths; MemoryNodeDemand.bytes; role=l2_cache"}
        if read is not None else {"availability": "unavailable",
            "reason": "predictor did not export an aggregate L2 read path"})
    return {"availability": "available", "compute": compute, "l2_traffic": l2_traffic,
        "memory_traffic": memory_traffic,
        "dtype": kernel.dtype, "logical_bytes": float(compulsory_bytes),
        "hbm_capacity_bytes_per_s": hardware.resource_by_role("dram").capacity,
        "traffic_basis": "compulsory logical work reference only; use memory_traffic for modeled L2/HBM bandwidth",
        "source": "KernelPredictor.resource_input; HWSpec configured resource capacity",
        "limitations": list(inputs.limitations)}


def kernel_link_usage(kernel, hardware) -> dict:
    """Expose algorithmic per-endpoint wire demand from transfer semantics.

    Ring all-reduce sends and receives 2*(P-1)/P payload bytes per rank;
    all-gather's payload is the input shard. These are algorithmic bytes,
    excluding packet headers, retransmits and routing overhead, not observed
    network transactions. No duration enters this accounting.
    """
    from src.kernel.kernel_spec import KernelKind
    if kernel.kind is not KernelKind.COMMUNICATION:
        return {"availability": "not_applicable"}
    transfer = kernel.transfer
    payload = transfer.bytes
    ranks = len(transfer.participants)
    if transfer.operation == "send_recv":
        wire_bytes = float(payload)
    elif transfer.operation == "all_reduce" and transfer.algorithm == "ring":
        wire_bytes = 2.0 * (ranks - 1) / ranks * payload
    elif transfer.operation == "all_gather" and transfer.algorithm == "ring":
        wire_bytes = float((ranks - 1) * payload)
    elif transfer.operation == "all_to_all":
        wire_bytes = (ranks - 1) / ranks * payload
    else:
        return {"availability": "unavailable", "link": transfer.link,
            "reason": "per-endpoint wire accounting unavailable for this collective algorithm"}
    return {"availability": "available", "link": transfer.link,
        "resource_id": transfer.resource_id,
        "participants": list(transfer.participants),
        "payload_bytes": payload, "wire_bytes_per_endpoint": wire_bytes,
        "capacity_bytes_per_s": hardware.analytical_link_bandwidth_bytes_s(transfer.link),
        "basis": "algorithmic bytes per participating endpoint; excludes packet overhead; configured analytical link bandwidth"}
