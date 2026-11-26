"""Disk-backed link intervals with a bounded cache of decoded time windows."""

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import json
import mmap
from pathlib import Path
import struct
import tempfile
from threading import RLock
from weakref import proxy


RECORD = struct.Struct("<dQ")
SCHEMA = 1
INDEX_DIRECTORY = Path("state/link-state-index")


def source_identity(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_index(source, target, trace_id, histories):
    """Publish an index of the original link intervals alongside offline data."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = source_identity(source)
    with tempfile.TemporaryDirectory(prefix="link-state-index-", dir=target.parent) as temporary:
        stage = Path(temporary) / "index"
        stage.mkdir()
        ranges = {}
        boundaries = set()
        count = 0
        with (stage / "index.bin").open("xb") as index, (stage / "rows.bin").open("xb") as records:
            for link, history in histories.items():
                if not history:
                    continue
                ranges[link] = [count, len(history)]
                previous = float("-inf")
                for row in history:
                    when = float(row["time_s"])
                    if when <= previous:
                        raise ValueError(f"unsorted link history: {link}")
                    previous = when
                    boundaries.add(when)
                    index.write(RECORD.pack(when, records.tell()))
                    records.write(json.dumps(row, allow_nan=False, separators=(",", ":")).encode())
                    count += 1
            index.write(RECORD.pack(0, records.tell()))
        if source_identity(source) != identity:
            raise ValueError("physical state changed while indexing link intervals")
        manifest = dict(schema=SCHEMA, source=identity, trace_id=trace_id,
                        rows=count, links=ranges, boundaries=sorted(boundaries))
        (stage / "index.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        if target.is_dir():
            for name in ("index.bin", "rows.bin", "index.json"):
                (stage / name).replace(target / name)
        else:
            stage.rename(target)


class LinkSequence(Sequence):
    def __init__(self, store, begin, end, rows):
        self.store, self.begin, self.end, self.rows = store, begin, end, rows

    def __len__(self):
        return self.end - self.begin

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = self.begin + index
        if self.rows:
            return self.store.row(position)
        return self.store.record(position)[0]


class LinkMap(Mapping):
    def __init__(self, store, rows):
        self.store, self.rows = proxy(store), rows

    def __getitem__(self, link):
        begin, end = self.store.bounds(link)
        return LinkSequence(self.store, begin, end, self.rows)

    def __iter__(self):
        for link in self.store.links:
            if self.store.bounds_or_empty(link):
                yield link

    def __len__(self):
        return sum(1 for _ in self)


class WindowedLinkStates:
    """Retain small indexes and at most two decoded 60-second working sets."""

    def __init__(self, root, source, trace_id, start, end, *, rows_per_window=8192):
        root = Path(root)
        meta = json.loads((root / "index.json").read_text(encoding="utf-8"))
        if (meta["schema"] != SCHEMA or meta["source"]["size"] != source_identity(source)["size"]
                or meta["trace_id"] != trace_id):
            raise ValueError("link state index does not match the verified offline source")
        self.links = meta["links"]
        self.boundaries = meta.get("boundaries")
        self.start, self.end = start, end
        index_path, rows_path = root / "index.bin", root / "rows.bin"
        rows_size = rows_path.stat().st_size
        if index_path.stat().st_size + rows_size <= 8 * 1024**2:
            self.index, self.rows_bytes, self.rows_file = index_path.read_bytes(), rows_path.read_bytes(), None
        else:
            with index_path.open("rb") as stream:
                self.index = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self.rows_file, self.rows_bytes = rows_path.open("rb"), None
        if (len(self.index) != (meta["rows"] + 1) * RECORD.size
                or self.record(meta["rows"])[1] != rows_size):
            self.close()
            raise ValueError("incomplete link state index")
        self.full_horizon = start == 0 and all(
            self.record(first + count - 1)[0] <= end for first, count in self.links.values())
        self._lock = RLock()
        self.rows_per_window = rows_per_window
        self.windows = OrderedDict()
        self.current = None
        self.hits = self.misses = self.window_evictions = self.row_evictions = 0
        self.frames, self.times = LinkMap(self, True), LinkMap(self, False)
        self.at_time(start)

    def record(self, index):
        return RECORD.unpack_from(self.index, index * RECORD.size)

    def bounds_or_empty(self, link):
        begin, count = self.links[link]
        end = begin + count
        if self.full_horizon:
            return begin, end
        all_times = LinkSequence(self, begin, end, False)
        low = max(0, bisect_right(all_times, self.start) - 1)
        high = bisect_right(all_times, self.end)
        if high > low:
            return begin + low, begin + high
        return None

    def bounds(self, link):
        span = self.bounds_or_empty(link)
        if span is None:
            raise KeyError(link)
        return span

    def at_time(self, time_s):
        # Cache buckets are storage policy only. All physical lookup times and
        # sub-frame link changes are still handled by the original transport.
        key = int(time_s // 60)
        with self._lock:
            self._select_window(key)

    def _select_window(self, key):
        if key == self.current:
            return
        if key not in self.windows:
            self.windows[key] = OrderedDict()
            if len(self.windows) > 2:
                self.windows.popitem(last=False)
                self.window_evictions += 1
        self.windows.move_to_end(key)
        self.current = key

    def row(self, index):
        with self._lock:
            return self._read_row(index)

    def _read_row(self, index):
        cache = self.windows[self.current]
        if index in cache:
            self.hits += 1
            cache.move_to_end(index)
            return cache[index]
        self.misses += 1
        start = self.record(index)[1]
        end = self.record(index + 1)[1]
        if self.rows_bytes is None:
            self.rows_file.seek(start)
            data = self.rows_file.read(end - start)
        else:
            data = self.rows_bytes[start:end]
        row = cache[index] = json.loads(data)
        if len(cache) > self.rows_per_window:
            cache.popitem(last=False)
            self.row_evictions += 1
        return row

    def close(self):
        if self.rows_file is not None:
            self.index.close()
            self.rows_file.close()
            self.rows_file = None
        self.rows_bytes = None
        if hasattr(self, "windows"):
            self.windows.clear()
