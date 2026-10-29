"""Versioned numeric ephemeris shards, preserving canonical Orbit Trace values."""

import hashlib
import json
from pathlib import Path

import numpy as np

from orbitserve.common import atomic_output

STORAGE = "orbitserve.ephemeris_columns.v1"
VECTORS = (
    "position_teme_km",
    "velocity_teme_km_s",
    "position_itrs_km",
    "velocity_itrs_km_s",
)
CONSTANTS = ("source_reference_frame", "earth_fixed_reference_frame")
DTYPE = np.dtype(
    [
        ("time_s", "<f8"),
        ("satellite_index", "<u4"),
        *((name, "<f8", (3,)) for name in VECTORS),
    ]
)


def supported(rows):
    keys = {"time_s", "satellite_id", *VECTORS, *CONSTANTS}
    return bool(rows) and all(
        set(row) == keys and all(row[k] == rows[0][k] for k in CONSTANTS)
        for row in rows
    )


def write_shards(root, rows, trace_id, hashes):
    identifiers = sorted({row["satellite_id"] for row in rows})
    indices = {name: i for i, name in enumerate(identifiers)}
    size = min(65536, 32 * len(identifiers))
    shards = []
    for number, offset in enumerate(range(0, len(rows), size)):
        batch = rows[offset : offset + size]
        values = np.empty(len(batch), dtype=DTYPE)
        values["time_s"] = [row["time_s"] for row in batch]
        values["satellite_index"] = [indices[row["satellite_id"]] for row in batch]
        for name in VECTORS:
            values[name] = [row[name] for row in batch]
        name = f"ephemeris/part-{number:06d}.npy"
        with atomic_output(Path(root) / name) as output:
            np.save(output, values, allow_pickle=False)
        hashes[name] = output.sha256
        shards.append(
            {
                "path": name,
                "sha256": output.sha256,
                "rows": len(batch),
                "start_s": batch[0]["time_s"],
                "end_s": batch[-1]["time_s"],
            }
        )
    return {
        "storage": STORAGE,
        "trace_id": trace_id,
        "satellite_ids": identifiers,
        "constants": {k: rows[0][k] for k in CONSTANTS},
        "record_count": len(rows),
        "shards": shards,
    }


def read_shards(root, payload):
    from .trace import _file_sha256, _resolve_relative_trace_path

    if payload.get("storage") != STORAGE:
        raise ValueError("unsupported ephemeris shard storage")
    identifiers = payload["satellite_ids"]
    if not identifiers or identifiers != sorted(set(identifiers)):
        raise ValueError("invalid ephemeris satellite index")
    constants = payload["constants"]
    if set(constants) != set(CONSTANTS):
        raise ValueError("invalid ephemeris constants")
    rows, paths = [], set()
    for shard in payload["shards"]:
        name = shard["path"]
        if name in paths:
            raise ValueError("duplicate ephemeris shard")
        paths.add(name)
        path = _resolve_relative_trace_path(Path(root), name)
        if _file_sha256(path) != shard["sha256"]:
            raise ValueError("ephemeris shard SHA-256 mismatch")
        values = np.load(path, allow_pickle=False)
        if values.dtype != DTYPE or values.ndim != 1 or len(values) != shard["rows"]:
            raise ValueError("invalid ephemeris shard shape")
        if (
            not len(values)
            or np.any(values["satellite_index"] >= len(identifiers))
            or not np.isfinite(values["time_s"]).all()
            or any(not np.isfinite(values[k]).all() for k in VECTORS)
        ):
            raise ValueError("invalid ephemeris shard values")
        if (
            values["time_s"][0] != shard["start_s"]
            or values["time_s"][-1] != shard["end_s"]
        ):
            raise ValueError("invalid ephemeris shard time range")
        rows.extend(
            {
                "time_s": float(v["time_s"]),
                "satellite_id": identifiers[int(v["satellite_index"])],
                **{k: v[k].tolist() for k in VECTORS},
                **constants,
            }
            for v in values
        )
    if len(rows) != payload["record_count"]:
        raise ValueError("incomplete ephemeris shards")
    return rows


def prepared_bytes(value):
    """Inputs were normalized by the writer; avoid copying/rounding them again."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def identity(value):
    return hashlib.sha256(prepared_bytes(value)).hexdigest()
