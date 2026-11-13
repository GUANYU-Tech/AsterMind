"""Lossless chunked records: retain execution evidence without millions of dicts.

Chunks are created and consumed inside this process only. Pickle is never read
from an uploaded file. Mutable records stay in the tail until explicitly sealed.
"""

import bisect
import pickle
import zlib
from collections.abc import Sequence
from itertools import zip_longest


class RecordView(Sequence):
    def __init__(self, source, indices, project=None):
        self.source, self.indices, self.project = source, indices, project

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RecordView(self.source, self.indices[key], self.project)
        row = self.source[self.indices[key]]
        return self.project(row) if self.project else row


class CompressedRecords(list):
    """Append-only JSON list with bounded decoded storage and stable indices.

The list subclass allows the standard JSON encoder to stream the iterator.
Use append/extend and read operations only; sealed values are immutable.
"""

    def __init__(self, *, chunk_size=4096, sealed=None):
        super().__init__()
        self.chunk_size = chunk_size
        self.sealed = sealed
        self.chunks, self.starts, self.tail = [], [], []
        self.saved = 0
        self.cached_index, self.cached_rows = -1, None

    def __len__(self):
        return self.saved + len(self.tail)

    def append(self, row):
        from astermind.engine.clock import stamp
        stamp(row)
        self.tail.append(row)
        while len(self.tail) > self.chunk_size:
            rows = self.tail[:self.chunk_size]
            if self.sealed is not None and not all(self.sealed(r) for r in rows):
                break
            self.starts.append(self.saved)
            self.chunks.append(zlib.compress(pickle.dumps(rows, protocol=5), level=1))
            self.saved += len(rows)
            del self.tail[:self.chunk_size]

    def extend(self, rows):
        for row in rows:
            self.append(row)

    def _chunk(self, index):
        if self.cached_index != index:
            self.cached_rows = pickle.loads(zlib.decompress(self.chunks[index]))
            self.cached_index = index
        return self.cached_rows

    def __getitem__(self, key):
        if isinstance(key, slice):
            return RecordView(self, range(len(self))[key])
        if key < 0:
            key += len(self)
        if key < 0 or key >= len(self):
            raise IndexError(key)
        if key >= self.saved:
            return self.tail[key-self.saved]
        chunk = bisect.bisect_right(self.starts, key)-1
        return self._chunk(chunk)[key-self.starts[chunk]]

    def __iter__(self):
        for index in range(len(self.chunks)):
            yield from self._chunk(index)
        yield from self.tail

    def __bool__(self):
        return bool(len(self))

    def __eq__(self, other):
        if not isinstance(other, (list, Sequence)):
            return NotImplemented
        marker = object()
        return all(a == b for a, b in zip_longest(self, other, fillvalue=marker))

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __contains__(self, value):
        return any(row == value for row in self)
