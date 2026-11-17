"""Lossless live evidence without retaining a decoded global table per event."""
from astermind.engine.record_store import CompressedRecords
from pathlib import Path
import threading


class DiskChunks:
    """Internal immutable pickle/zlib chunks; decoded data stays one chunk wide."""
    def __init__(self, folder):
        from paths import owned
        self.folder = owned(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.stored_bytes = 0
        self.offsets = []
        self.lock = threading.RLock()
        # A single append-only file avoids thousands of open/rename operations
        # on WSL-mounted workspaces. These are private, rebuildable chunks;
        # public evidence is published separately by LiveSession._persist_events.
        self.stream = (self.folder / 'chunks.bin').open('x+b')

    def __len__(self):
        return len(self.offsets)

    def append(self, data):
        with self.lock:
            self.stream.seek(self.stored_bytes)
            self.stream.write(data)
            self.offsets.append((self.stored_bytes, len(data)))
            self.stored_bytes += len(data)

    def __getitem__(self, index):
        with self.lock:
            start, size = self.offsets[index]
            self.stream.seek(start)
            data = self.stream.read(size)
            if len(data) != size:
                raise ValueError('Incomplete internal live record chunk')
            return data


class LiveRecords(CompressedRecords):
    def __init__(self, loop=None, *, chunk_size=4096, output=None, name='events'):
        super().__init__(chunk_size=chunk_size)
        if output is not None:
            self.chunks = DiskChunks(Path(output) / '.live-records' / name)
        self.loop = loop
        self._capacities = None
        self._capacity_source = None

    def append(self, event):
        row = dict(event)
        if self.loop is not None:
            row['time_ticks'] = str(self.loop.now_s.ticks)
        capacities = row.get('capacities')
        if row.get('kind') in {'allocation', 'v1_allocation'} and capacities is not None:
            # Allocations change flow rates much more often than physical
            # supply. Preserve the table while sharing identical immutable
            # snapshots, including inside each compressed chunk's pickle memo.
            if capacities is self._capacity_source or (self._capacities is not None and capacities == self._capacities):
                row['capacities'] = self._capacities
            else:
                self._capacities = capacities
            self._capacity_source = capacities
        super().append(row)


def compact_communication_trace(host, output=None):
    policy = host.communication_policy
    if isinstance(policy.trace, LiveRecords):
        return
    records = LiveRecords(host.timeline, output=output, name='native-communication')
    # Preserve timestamps if initialization already emitted native evidence.
    for event in policy.trace:
        loop, records.loop = records.loop, None
        records.append(event)
        records.loop = loop
    policy.trace = records
