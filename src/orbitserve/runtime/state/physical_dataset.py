"""Portable, read-only physical attributes and initial causal network timeline."""

from __future__ import annotations

import bisect
import json
import sqlite3
import zlib
from contextlib import closing
from functools import lru_cache
from pathlib import Path

SCHEMA = "orbitserve.physical_state.v1"


def physical_spec(spec):
    value = spec.to_dict()
    value.pop("transport_model", None)
    return value


def _write_block(db, link_id, block):
    db.execute(
        "INSERT INTO attributes VALUES (?,?,?)",
        (
            link_id,
            block[0][0],
            zlib.compress(
                json.dumps(block, allow_nan=False, separators=(",", ":")).encode()
            ),
        ),
    )


def write_physical_state(network, path):
    """Stream sampled budgets to disk; never hold a dense constellation in RAM."""
    metadata = {
        "schema_version": SCHEMA,
        "trace_id": network.trace.trace_id,
        "network": physical_spec(network.spec),
        "ground_positions": network.ground_positions,
        "sun_positions": network.sun_positions,
        "frames": network.frames,
        "events": network.events,
        "link_frames": network.link_frames,
        "allocation_checks": network.allocation_checks,
    }
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE metadata (value TEXT NOT NULL)")
        db.execute(
            "INSERT INTO metadata VALUES (?)", (json.dumps(metadata, allow_nan=False),)
        )
        db.execute(
            "CREATE TABLE attributes (link TEXT, time REAL, value BLOB NOT NULL, PRIMARY KEY(link,time)) WITHOUT ROWID"
        )
        for edge in network.edges:
            times = set(network.environment_times)
            for window in network.windows.get(
                (edge["endpoint_a"], edge["endpoint_b"]), ()
            ):
                times.update((float(window["start_s"]), float(window["end_s"])))
            from ..setup.radio_capacity import directional_profile

            profiles = (
                network.spec.link_profiles[edge.get("profile_id", edge["kind"])],
                *(
                    directional_profile(
                        network.nodes, network.spec.link_profiles, edge, source
                    )
                    for source in (edge["endpoint_a"], edge["endpoint_b"])
                ),
            )
            times.update(
                t for profile in profiles for t, _ in profile.capacity_schedule
            )
            block = []
            for t in sorted(t for t in times if network.start_s <= t <= network.end_s):
                value = {
                    "geometry": network._geometry_attributes(edge, t),
                    "directions": network.directions(edge, t),
                }
                block.append((t, value))
                if len(block) == 256:
                    _write_block(db, edge["link_id"], block)
                    block = []
            if block:
                _write_block(db, edge["link_id"], block)


class PhysicalAttributes:
    """Bounded cache over an immutable, indexed dataset shared by network forks."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        # Connections are short-lived so forks and concurrent preview requests
        # never share thread-affine handles or leak descriptors.
        with closing(self.connect()) as db:
            self.metadata = json.loads(
                db.execute("SELECT value FROM metadata").fetchone()[0]
            )
            self.block_times = {}
            for link, time_s in db.execute(
                "SELECT link,time FROM attributes ORDER BY link,time"
            ):
                self.block_times.setdefault(link, []).append(time_s)
        if self.metadata.get("schema_version") != SCHEMA:
            raise ValueError("unsupported physical state; regenerate in step one")
        self._block = lru_cache(maxsize=128)(self._read_block)
        self.at = lru_cache(maxsize=4096)(self._at)

    def connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)

    def _at(self, link, time_s):
        starts = self.block_times.get(link, ())
        index = bisect.bisect_right(starts, time_s) - 1
        if index < 0:
            raise ValueError("physical dataset does not cover link/time")
        times, values = self._block(link, starts[index])
        return values[bisect.bisect_right(times, time_s) - 1]

    def _read_block(self, link, start_s):
        with closing(self.connect()) as db:
            value = db.execute(
                "SELECT value FROM attributes WHERE link=? AND time=?", (link, start_s)
            ).fetchone()[0]
        rows = json.loads(zlib.decompress(value))
        return [r[0] for r in rows], [r[1] for r in rows]
