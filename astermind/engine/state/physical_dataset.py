"""Portable, read-only physical attributes and initial causal network timeline."""

from __future__ import annotations

import bisect
import json
import math
import multiprocessing
import sqlite3
import zlib
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from copy import deepcopy
from functools import lru_cache
from itertools import islice
from pathlib import Path
from time import perf_counter
from threading import RLock
from types import SimpleNamespace

from astermind.engine.state import attribute_workers
from astermind.engine.state.attribute_cache import AttributeBlock
from astermind.engine.setup.sun_geometry import SUN_GEOMETRY_VERSION

SCHEMA = "orbitserve.physical_state.v2"
LEGACY_SCHEMA = "orbitserve.physical_state.v1"
COMPRESSED_READ_CACHE_BYTES = 64 * 1024 * 1024
DECODED_READ_CACHE_BYTES = 512 * 1024 * 1024
MMAP_READ_BYTES = 256 * 1024 * 1024
MAX_BLOCK_BYTES = 64 * 1024 * 1024


def _decode_block(value):
    """Bound decompression before parsing externally supplied physical state."""
    from astermind.engine.state.attribute_codec import MAGIC, decode
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(value, MAX_BLOCK_BYTES + 1)
    except zlib.error as error:
        raise ValueError('invalid compressed physical block') from error
    if len(raw) > MAX_BLOCK_BYTES:
        raise ValueError('physical block exceeds decompression size limit')
    if not decoder.eof or decoder.unused_data:
        raise ValueError('truncated or trailing compressed physical block')
    try:
        return decode(raw) if raw.startswith(MAGIC) else AttributeBlock(json.loads(raw))
    except (KeyError, IndexError, TypeError, OverflowError) as error:
        raise ValueError('invalid physical block structure') from error


def physical_spec(spec):
    from astermind.engine.setup.faults import physical_network_settings
    return physical_network_settings(spec.to_dict())


def _edge_blocks(network, edge, *, encoded=True):
    from astermind.engine.setup.radio_capacity import directional_profile

    times = set(network.environment_times)
    for window in network.windows.get((edge['endpoint_a'], edge['endpoint_b']), ()):
        times.update((float(window['start_s']), float(window['end_s'])))
    profiles = (network.spec.link_profiles[edge.get('profile_id', edge['kind'])],
                *(directional_profile(network.nodes, network.spec.link_profiles, edge, source)
                  for source in (edge['endpoint_a'], edge['endpoint_b'])))
    times.update(t for profile in profiles for t, _ in profile.capacity_schedule)
    block = []
    for t in sorted(t for t in times if network.start_s <= t <= network.end_s):
        geometry = network._geometry_attributes(edge, t)
        block.append((t, {'geometry': geometry,
                          'directions': network.directions(edge, t, geometry=geometry)}))
        if len(block) == 256:
            yield edge['link_id'], block[0][0], (_encode_block(block) if encoded else block)
            block = []
    if block:
        yield edge['link_id'], block[0][0], (_encode_block(block) if encoded else block)


def _calculate_chunk(payload):
    from astermind.engine.state.network import NetworkState

    context, edges = payload
    network = NetworkState.__new__(NetworkState)
    network.__dict__.update(context)
    network.spec = SimpleNamespace(link_profiles=context['profiles'])
    return [row for edge in edges for row in _edge_blocks(network, edge, encoded=False)]


def _chunks(network):
    from astermind.engine.setup.radio_capacity import hop_pools

    # Send only positions/windows used by this small edge chunk, never pickle
    # the whole trace or acquisition timeline into each spawned process.
    for offset in range(0, len(network.edges), 8):
        edges = network.edges[offset:offset+8]
        ids = {edge[key] for edge in edges for key in ('endpoint_a', 'endpoint_b')}
        pools = {key for edge in edges for source,target in
                 ((edge['endpoint_a'],edge['endpoint_b']), (edge['endpoint_b'],edge['endpoint_a']))
                 for key in hop_pools(network.nodes, source, target)}
        context = {name: getattr(network, name) for name in
                   ('start_s', 'end_s', 'environment_times', 'sun_positions', 'sun_times')}
        context.update(
            nodes={key:network.nodes[key] for key in ids},
            samples={key:network.samples[key] for key in ids if key in network.samples},
            sample_times={key:network.sample_times[key] for key in ids if key in network.sample_times},
            ground_positions={key:network.ground_positions[key] for key in ids if key in network.ground_positions},
            windows={(edge['endpoint_a'],edge['endpoint_b']):network.windows.get((edge['endpoint_a'],edge['endpoint_b']), ()) for edge in edges},
            profiles=network.spec.link_profiles,
            pool_capacities={key:network.pool_capacities[key] for key in pools},
            observed_faults={}, physical_attributes=None, manual_disabled_links=set(), added_edge_ids=set(),
        )
        yield context, edges


def _encode_block(block):
    from astermind.engine.state.attribute_codec import encode
    compact = block if isinstance(block, AttributeBlock) else AttributeBlock(block)
    raw = encode(compact)
    if len(raw) > MAX_BLOCK_BYTES:
        raise ValueError('physical block exceeds decompression size limit')
    return zlib.compress(raw, level=1)


def _encode_chunk(rows):
    return [(link, time, _encode_block(block)) for link, time, block in rows]


def _encode_prepared(rows):
    return [(link, time, _encode_block(block)) for link, time, block, _ in rows]


def write_physical_state(network, path, *, workers=1, progress=None):
    """Calculate bounded batches once; serialize separately, then allocate."""
    if type(workers) is not int or workers < 1:
        raise ValueError('physical attribute worker count must be a positive integer')
    report = progress or (lambda stage: None)
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if path.exists():
        raise FileExistsError('physical state already exists; choose a new output directory')
    if temporary.exists():
        raise FileExistsError('incomplete physical state already exists; choose a new output directory')
    timings = {'attributes_compute_s': 0., 'attributes_encode_write_s': 0.,
               'allocation_s': 0., 'allocation_cache_read_s': 0.}
    count = len(network.edges)
    actual = min(workers, max(1, (count + 7)//8))
    report(f'Physical attributes: 0/{count} links; {actual} workers')
    chunks = iter(_chunks(network))
    context = multiprocessing.get_context('spawn')
    queue = context.Queue() if actual > 1 else None
    gate = context.Event() if actual > 1 else None
    pool = (ProcessPoolExecutor(max_workers=actual, mp_context=context,
                               initializer=attribute_workers.initialize,
                               initargs=(queue, gate))
            if actual > 1 else None)
    done = 0
    prepared, prepared_bytes = OrderedDict(), 0
    record_count = block_count = 0
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.execute('CREATE TABLE metadata (value TEXT NOT NULL)')
            db.execute('CREATE TABLE attributes (link TEXT, time REAL, value BLOB NOT NULL, PRIMARY KEY(link,time)) WITHOUT ROWID')
            while batch := list(islice(chunks, actual)):
                start = perf_counter()
                if pool:
                    futures = [pool.submit(attribute_workers.calculate_then_encode, (i, payload))
                               for i, payload in enumerate(batch)]
                    staged = attribute_workers.receive_prepared(queue, futures)
                else:
                    raw = [_calculate_chunk(batch[0])]
                    staged = [[(link, time_s, AttributeBlock(values), len(values))
                               for link, time_s, values in raw[0]]]
                for rows in staged:
                    for link, time_s, block, records in rows:
                        record_count += records
                        block_count += 1
                        while prepared and prepared_bytes + block.nbytes > DECODED_READ_CACHE_BYTES:
                            _, old = prepared.popitem(last=False)
                            prepared_bytes -= old.nbytes
                        if block.nbytes <= DECODED_READ_CACHE_BYTES:
                            prepared[link, time_s] = block
                            prepared_bytes += block.nbytes
                timings['attributes_compute_s'] += perf_counter() - start
                start = perf_counter()
                if pool:
                    gate.set()
                    encoded = [future.result() for future in futures]
                    gate.clear()
                else:
                    encoded = [_encode_prepared(staged[0])]
                for rows in encoded:
                    db.executemany('INSERT INTO attributes VALUES (?,?,?)', rows)
                timings['attributes_encode_write_s'] += perf_counter() - start
                done += sum(len(payload[1]) for payload in batch)
                report(f'Physical attributes: {done}/{count} links; {actual} workers')
        metadata = {
            'schema_version': SCHEMA, 'trace_id': network.trace.trace_id,
            'sun_geometry_version': SUN_GEOMETRY_VERSION,
            'network': physical_spec(network.spec),
            'ground_positions': network.ground_positions, 'sun_positions': network.sun_positions,
        }
        start = perf_counter()
        reader = PhysicalAttributes(temporary, _metadata=metadata)
        reader._blocks = prepared
        reader.decoded_bytes = reader.peak_decoded_bytes = prepared_bytes
        timings['attribute_cache_setup_s'] = perf_counter() - start
        previous = network.physical_attributes
        network.physical_attributes = reader
        try:
            if network._compile_pending:
                report('Dynamic link acquisition and terminal allocation')
                start = perf_counter()
                before_io = reader.read_decode_s
                network._set_attribute_boundaries()
                network._compile()
                timings['allocation_cache_read_s'] = reader.read_decode_s - before_io
                timings['allocation_s'] = perf_counter() - start - timings['allocation_cache_read_s']
                network._compile_pending = False
                network.frame_times = [f['time_s'] for f in network.frames]
            else:
                network._set_attribute_boundaries()
            metadata.update(frames=network.frames, events=network.events,
                            link_frames=network.link_frames, allocation_checks=network.allocation_checks)
            with closing(sqlite3.connect(temporary)) as db, db:
                db.execute('INSERT INTO metadata VALUES (?)', (json.dumps(metadata, allow_nan=False),))
            temporary.replace(path)
        except BaseException:
            network.physical_attributes = previous
            raise
        reader.path = path.resolve()
        reader.metadata = metadata
        network.precompute_timings = timings
        network.physical_counts = {'records': record_count, 'blocks': block_count,
                                   'peak_decoded_cache_bytes': reader.peak_decoded_bytes}
    finally:
        if pool:
            start = perf_counter()
            gate.set()
            pool.shutdown(wait=True, cancel_futures=True)
            queue.close()
            queue.join_thread()
            timings['attribute_worker_shutdown_s'] = perf_counter() - start


class PhysicalAttributes:
    """Bounded cache over an immutable, indexed dataset shared by network forks."""

    def __init__(self, path, *, _metadata=None):
        self.path = Path(path).resolve()
        self._read_connection = None
        # Connections are short-lived so forks and concurrent preview requests
        # never share thread-affine handles or leak descriptors.
        with closing(self.connect()) as db:
            self.metadata = (_metadata if _metadata is not None else json.loads(
                db.execute("SELECT value FROM metadata").fetchone()[0]))
            self.block_times = {}
            # Scan small immutable datasets once as compressed bytes. Avoid
            # repeated filesystem connection/lock roundtrips in route searches.
            packed_bytes = db.execute('SELECT COALESCE(SUM(length(value)),0) FROM attributes').fetchone()[0]
            self._packed = {} if packed_bytes <= COMPRESSED_READ_CACHE_BYTES else None
            columns = 'link,time,value' if self._packed is not None else 'link,time'
            for row in db.execute(f'SELECT {columns} FROM attributes ORDER BY link,time'):
                link, time_s = row[:2]
                self.block_times.setdefault(link, []).append(time_s)
                if self._packed is not None:
                    self._packed[link,time_s] = row[2]
        if self.metadata.get("schema_version") not in {SCHEMA, LEGACY_SCHEMA}:
            raise ValueError("unsupported physical state; regenerate in step one")
        self._blocks = OrderedDict()
        self._cache_lock = RLock()
        self.decoded_bytes = self.peak_decoded_bytes = 0
        self.read_decode_s = 0.
        self.block_loads = 0
        self._cached_at = lru_cache(maxsize=4096)(self._at)

        def read(link, time_s):
            return deepcopy(self._cached_at(link, time_s))

        read.cache_info = self._cached_at.cache_info
        read.cache_clear = self._cached_at.cache_clear
        self.at = read

    def connect(self, *, immutable=False):
        options = '?mode=ro&immutable=1' if immutable else '?mode=ro'
        db = sqlite3.connect(self.path.as_uri() + options, uri=True)
        try:
            # Share file pages through the OS instead of issuing a mounted-drive
            # read RPC for every small SQLite page. Unsupported builds use zero.
            db.execute(f'PRAGMA mmap_size={MMAP_READ_BYTES}')
        except BaseException:
            db.close()
            raise
        return db

    def open_read_session(self):
        """Own one connection in a single export thread over a completed dataset.

        Immutable mode avoids per-query file locks and change checks on mounted
        filesystems. Callers must close the session before passing the reader to
        another thread or process; ordinary readers keep short-lived connections.
        """
        if self._read_connection is None:
            self._read_connection = self.connect(immutable=True)

    def close_read_session(self):
        if self._read_connection is not None:
            self._read_connection.close()
            self._read_connection = None

    def _locate(self, link, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError('physical query time must be finite and nonnegative')
        starts = self.block_times.get(link, ())
        index = bisect.bisect_right(starts, time_s) - 1
        if index < 0:
            raise ValueError('physical dataset does not cover link/time')
        block = self._block(link, starts[index])
        row = bisect.bisect_right(block.times, time_s) - 1
        if row < 0:
            raise ValueError('physical block does not cover query time')
        return block, row

    def _at(self, link, time_s):
        block, index = self._locate(link, time_s)
        return block.at(index)

    def geometry_at(self, link, time_s):
        block, index = self._locate(link, time_s)
        return block.geometry(index)

    def direction_at(self, link, time_s, source, target):
        block, index = self._locate(link, time_s)
        return block.direction(index, source, target)

    def _block(self, link, start_s):
        # Forked network previews share this immutable reader across threads.
        # Keep eviction, cache population and byte accounting one transaction.
        with self._cache_lock:
            return self._cached_block(link, start_s)

    def _cached_block(self, link, start_s):
        key = link, start_s
        if key in self._blocks:
            self._blocks.move_to_end(key)
            return self._blocks[key]
        start = perf_counter()
        block = self._read_block(link, start_s)
        self.read_decode_s += perf_counter() - start
        self.block_loads += 1
        while self._blocks and self.decoded_bytes + block.nbytes > DECODED_READ_CACHE_BYTES:
            _, old = self._blocks.popitem(last=False)
            self.decoded_bytes -= old.nbytes
        if block.nbytes <= DECODED_READ_CACHE_BYTES:
            self._blocks[key] = block
            self.decoded_bytes += block.nbytes
            self.peak_decoded_bytes = max(self.peak_decoded_bytes, self.decoded_bytes)
        return block

    def _read_block(self, link, start_s):
        if self._packed is not None:
            value = self._packed[link, start_s]
        elif self._read_connection is not None:
            value = self._read_connection.execute(
                'SELECT value FROM attributes WHERE link=? AND time=?', (link, start_s)
            ).fetchone()[0]
        else:
            with closing(self.connect()) as db:
                value = db.execute('SELECT value FROM attributes WHERE link=? AND time=?', (link, start_s)).fetchone()[0]
        return _decode_block(value)
