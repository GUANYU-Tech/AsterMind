"""Strict Native migration comparison with narrow, baseline-observed exceptions.

Only host measurements, content-verified input paths and independent transfer
observations within one clock tick are canonicalized. Simulation values remain
exact, including counters, configuration, provenance, audit and record order.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

WALL_PHASES = {
    "workload_construction",
    "timeline_construction",
    "kernel_lowering",
    "kernel_prediction",
}
PHYSICAL_TIMINGS = {
    "allocation_cache_read_s",
    "allocation_s",
    "attribute_cache_setup_s",
    "attributes_compute_s",
    "attributes_encode_write_s",
    "candidate_geometry_s",
    "formation_preflight_s",
    "network_setup_s",
    "orbit_assembly_s",
    "orbit_export_s",
    "position_s",
    "snapshot_store_s",
    "validation_s",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def measured(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"invalid host measurement: {value!r}")
    return "<host measurement>"


def normalize_observations(events):
    result = []
    # Never sort across time, across a global control event, or within a flow.
    for _, group in itertools.groupby(
        events, lambda e: (e.get("time_ticks"), e.get("time_s"))
    ):
        pending = []
        for event in group:
            if event.get("transfer_id"):
                pending.append(event)
            else:
                result.extend(sorted(pending, key=lambda e: e["transfer_id"]))
                pending.clear()
                result.append(event)
        result.extend(sorted(pending, key=lambda e: e["transfer_id"]))
    return result


def normalize_startup_progress(stages, cell_ids):
    """FIRST_COMPLETED may observe multiple host workers finishing together."""
    indices = [i for i, row in enumerate(stages) if row.get("phase") == "cells"]
    if not indices:
        return stages
    if indices != list(range(indices[0], len(stages))):
        raise ValueError("Cell startup progress must be the final startup phase")
    rows = [stages[i] for i in indices]
    total, workers = rows[0]["total"], rows[0]["workers"]
    previous = -1
    for row in rows:
        if set(row) != {"phase", "completed", "total", "workers", "active", "wall_s"}:
            raise ValueError("unreviewed Cell startup fields")
        completed = row["completed"]
        if (
            row["total"] != total
            or row["workers"] != workers
            or not previous < completed <= total
            or len(row["active"]) != min(workers, total - completed)
            or len(set(row["active"])) != len(row["active"])
            or cell_ids is not None
            and not set(row["active"]) <= cell_ids
        ):
            raise ValueError("invalid Cell startup progress")
        measured(row["wall_s"])
        previous = completed
    if rows[0]["completed"] != 0 or previous != total:
        raise ValueError("incomplete Cell startup")
    # Retain initial/final capacity and completion; host poll granularity is not
    # a simulated event. All intermediate records have been validated above.
    return stages[: indices[0]] + [rows[0], rows[-1]]


def normalize(value, references=None, path=(), cell_ids=None):
    references = references or {}
    if not path and isinstance(value, dict) and "cells" in value:
        cell_ids = set(value["cells"])
    measurement = path[1:] if path[:1] == ("audit",) else path
    if measurement[:1] == ("measurement",):
        rest = measurement[1:]
        if rest == ("startup_stages",):
            value = normalize_startup_progress(value, cell_ids)
        if rest in (
            ("runtime_startup_wall_s",),
            ("event_loop_wall_s",),
            ("cell_startup_policy", "available_memory_bytes"),
            ("cell_startup_policy", "memory_slots"),
        ):
            return measured(value)
        if len(rest) == 3 and rest[0] == "startup_stages" and rest[2] == "wall_s":
            return measured(value)
        if len(rest) == 3 and rest[0] == "startup_stages" and rest[2] == "active":
            if (
                not isinstance(value, list)
                or any(not isinstance(x, str) for x in value)
                or len(set(value)) != len(value)
                or cell_ids is not None
                and not set(value) <= cell_ids
            ):
                raise ValueError("invalid host startup occupancy")
            # Parallel host initialization can finish either Cell first. Preserve
            # occupancy count; completed/total/workers remain exact separately.
            return ["<host worker>"] * len(value)
    if (
        len(path) == 5
        and path[0] == "cells"
        and path[2:4] == ("execution_performance", "phase_wall_time_s")
        and path[4] in WALL_PHASES
    ):
        return measured(value)
    if isinstance(value, str) and value in references:
        return references[value]
    if isinstance(value, dict):
        return {
            key: normalize(item, references, path + (key,), cell_ids)
            for key, item in value.items()
        }
    if isinstance(value, list):
        if path == ("network", "event_log"):
            value = normalize_observations(value)
        return [normalize(item, references, path + ("*",), cell_ids) for item in value]
    return value


def differences(a, b, path="$"):
    if type(a) is not type(b):
        yield path + ": type"
        return
    if isinstance(a, dict):
        if a.keys() != b.keys():
            yield path + ": keys"
        for key in a.keys() & b.keys():
            yield from differences(a[key], b[key], path + "/" + key)
    elif isinstance(a, list):
        if len(a) != len(b):
            yield path + ": length"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from differences(x, y, path + "/" + str(i))
    elif a != b:
        yield path


def input_tree(directory):
    """Check every stored file hash before normalizing measured build durations."""
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        raw = path.read_bytes()
        if relative in ("core_result.json", "network_result.json"):
            value = json.loads(raw)
            for key in value.get("timing_s", {}):
                if key not in PHYSICAL_TIMINGS:
                    raise ValueError(f"unreviewed physical timing: {key}")
                value["timing_s"][key] = measured(value["timing_s"][key])
            raw = canonical_bytes(value)
        elif relative == "physical_dataset.json":
            value = json.loads(raw)
            for filename, expected in value["files"].items():
                actual = digest((directory / filename).read_bytes())
                if actual != expected:
                    raise ValueError(f"corrupt physical artifact: {filename}")
            # This is the only manifest hash covering allowed wall measurements.
            core = json.loads((directory / "core_result.json").read_text())
            for key in core["timing_s"]:
                if key not in PHYSICAL_TIMINGS:
                    raise ValueError(key)
                core["timing_s"][key] = measured(core["timing_s"][key])
            value["files"]["core_result.json"] = digest(canonical_bytes(core))
            raw = canonical_bytes(value)
        result[relative] = digest(raw)
    return result


def verified_references(folder):
    references, inputs = {}, {}
    for manifest in sorted((folder / "inputs").glob("*/references.json")):
        case = manifest.parent.name
        for original, relative in json.loads(manifest.read_text()).items():
            source = manifest.parent / relative
            content = (
                input_tree(source)
                if source.is_dir()
                else {source.name: digest(source.read_bytes())}
            )
            inputs[case + "/" + relative] = content
            references[original] = (
                "<verified input:" + digest(canonical_bytes(content)) + ">"
            )
    return references, inputs


def compare_folders(before, after):
    before, after = Path(before), Path(after)
    left_refs, left_inputs = verified_references(before)
    right_refs, right_inputs = verified_references(after)
    problems = list(differences(left_inputs, right_inputs, "inputs"))
    left_files = {p.name for p in before.glob("*.json")}
    right_files = {p.name for p in after.glob("*.json")}
    if left_files != right_files:
        problems.append("report file set differs")
    for name in sorted(left_files & right_files):
        left = normalize(json.loads((before / name).read_text()), left_refs)
        right = normalize(json.loads((after / name).read_text()), right_refs)
        found = list(differences(left, right, name))
        problems.extend(found)
        print(name, "PASS" if not found else f"FAIL ({len(found)} differences)")
    return {
        "reports": len(left_files),
        "input_sets": len(left_inputs),
        "differences": problems,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    result = compare_folders(args.before, args.after)
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result["differences"]))
