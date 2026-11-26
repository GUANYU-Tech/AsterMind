"""Static metadata once, time-sharded position arrays and immutable link columns."""

import bisect
import json
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import numpy as np

from astermind.common import atomic_output, write_json
from astermind.engine import NetworkSpec, NetworkState

SCHEMA = "orbitserve.snapshot_store.v1"


def write_store(network, root, regular_times, *, chunk_samples=32):
    root = Path(root)
    state = root / "state"
    hashes = {}
    network._snapshot_nodes(network.start_s)
    ids = [row["node_id"] for row in network._snapshot_catalog]
    times = network.environment_times
    shards = []
    for offset in range(0, len(times), chunk_samples):
        batch = times[offset : offset + chunk_samples]
        values = np.array(
            [[network.position(n, t) for n in ids] for t in batch], dtype="<f8"
        )
        name = f"positions/part-{offset // chunk_samples:06d}.npy"
        with atomic_output(state / name) as output:
            np.save(output, values, allow_pickle=False)
        hashes[str((state / name).resolve())] = output.sha256
        shards.append({"path": name, "offset": offset, "count": len(batch)})
    static = {
        "schema_version": SCHEMA,
        "trace_id": network.physical_attributes.metadata["trace_id"],
        "network": network.spec.to_dict(),
        "nodes": network._snapshot_catalog,
        "edges": network.edges,
        "pool_capacities": network.pool_capacities,
        "environment_times": times,
        "regular_times": regular_times,
        "position_shards": shards,
        "attribute_boundaries": network.attribute_boundaries,
    }
    timeline = {
        name: getattr(network, name)
        for name in ("frames", "link_frames", "events", "observed_faults")
    }
    for name, value in (("index.json", static), ("timeline.json", timeline)):
        path = state / name
        hashes[str(path.resolve())] = write_json(path, value)
    return hashes


class SnapshotSource(NetworkState):
    """Each worker opens shared read-only arrays; no full trace/model is pickled."""

    def __init__(self, root, *, _share_static=False, initial_spec=None):
        self.root = Path(root)
        self._share_snapshot_metadata = _share_static
        self.manual_disabled_links = set()
        self.added_edge_ids = set()
        value = json.loads((self.root / "state/index.json").read_text(encoding="utf-8"))
        if value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported snapshot store")
        self.physical_attributes = NetworkState.open_physical_attributes(
            self.root / "physical_state.sqlite"
        )
        if value["trace_id"] != self.physical_attributes.metadata["trace_id"]:
            raise ValueError("snapshot store trace mismatch")
        self.spec = NetworkSpec.from_value(value["network"])
        self.nodes = {n.node_id: n for n in self.spec.nodes}
        self._snapshot_catalog = value["nodes"]
        self.node_indices = {
            n["node_id"]: i for i, n in enumerate(self._snapshot_catalog)
        }
        self.edges = value["edges"]
        self.pool_capacities = value["pool_capacities"]
        self.environment_times = value["environment_times"]
        self.start_s, self.end_s = self.environment_times[0], self.environment_times[-1]
        self.regular_times = value["regular_times"]
        self.attribute_boundaries = value["attribute_boundaries"]
        self.position_shards = value["position_shards"]
        self.shard_offsets = [s["offset"] for s in self.position_shards]
        self._positions = OrderedDict()
        self._position_lock = RLock()
        if initial_spec is None:
            timeline = json.loads(
                (self.root / "state/timeline.json").read_text(encoding="utf-8")
            )
        else:
            from astermind.engine.state.physical_dataset import validate_stored_link_budgets
            from astermind.engine.setup.faults import physical_network_settings

            metadata = self.physical_attributes.metadata
            validate_stored_link_budgets(metadata["network"], self.edges)
            stored_spec = NetworkSpec.from_value(metadata["network"])
            if physical_network_settings(initial_spec.to_dict()) != physical_network_settings(stored_spec.to_dict()):
                self.close()
                raise ValueError("offline frame data does not match runtime network configuration")
            self.spec = initial_spec
            trace_manifest = json.loads(
                (self.root / "orbit_trace/trace_manifest.json").read_text(encoding="utf-8")
            )
            if trace_manifest["trace_id"] != metadata["trace_id"]:
                self.close()
                raise ValueError("offline position and link-state traces differ")
            self.trace = SimpleNamespace(time=trace_manifest["time"], trace_id=metadata["trace_id"])
            timeline = {key: metadata[key] for key in ("frames", "events")}
            timeline["observed_faults"] = {}
            self.allocation_checks = metadata["allocation_checks"]
        self.edge_by_id = {edge["link_id"]: edge for edge in self.edges}
        self.candidate_edge_ids = set(self.edge_by_id)
        self.geometry_sample_times = tuple(self.environment_times)
        self.windows = {}
        self.ground_positions = self.physical_attributes.metadata["ground_positions"]
        self.sun_positions = self.physical_attributes.metadata["sun_positions"]
        self.sun_times = tuple(row[0] for row in self.sun_positions)
        self.calendars, self.shared_load, self.shared_pool_load = {}, {}, {}
        self.active_faults, self.fault_boundaries, self.revision = {}, set(), 0
        for name, rows in timeline.items():
            setattr(self, name, rows)
        if "link_frames" in timeline:
            self.link_times = {
                key: [v["time_s"] for v in rows] for key, rows in self.link_frames.items()
            }
        else:
            store = self.physical_attributes.link_store
            if store is None:
                raise ValueError("snapshot store is missing the offline link-state index")
            self.link_frames, self.link_times = store.frames, store.times
        self.frame_times = [v["time_s"] for v in self.frames]

    def __enter__(self):
        self.physical_attributes.open_read_session()
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        self.physical_attributes.close()
        self._positions.clear()

    def position(self, node_id, time_s):
        with self._position_lock:
            return self._position_at(node_id, time_s)

    def _position_at(self, node_id, time_s):
        if not self.start_s <= time_s <= self.end_s:
            raise ValueError("position outside snapshot store")
        index = bisect.bisect_right(self.environment_times, time_s) - 1
        number = bisect.bisect_right(self.shard_offsets, index) - 1
        if number not in self._positions:
            shard = self.position_shards[number]
            relative = Path(shard["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid snapshot position path")
            values = np.load(
                self.root / "state" / relative, mmap_mode="r", allow_pickle=False
            )
            if values.dtype != np.dtype("<f8") or values.shape != (
                shard["count"],
                len(self.nodes),
                3,
            ):
                raise ValueError("invalid snapshot position shape")
            self._positions[number] = values
            while len(self._positions) > 2:
                self._positions.popitem(last=False)
        self._positions.move_to_end(number)
        return tuple(
            self._positions[number][
                index - self.shard_offsets[number], self.node_indices[node_id]
            ].tolist()
        )
