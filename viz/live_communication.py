"""Incremental, lossless communication pages for an active session.

Reads the append-only gzip evidence; it never advances or modifies the solver.
The existing artifact endpoint serves the immutable pages and atomic index.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
import zlib
import uuid

from event_catalog import annotate
from live_replay import LiveReplayWindow
from transmission_recording import is_transmission_event


class CommunicationPublisher:
    def __init__(self, output):
        self.output = Path(output)
        self.folder = self.output / 'native-replay'
        self.folder.mkdir(exist_ok=True)
        self.offset = 0
        self.decoder = zlib.decompressobj(31)
        self.pending = b''
        self.order = 0
        self.pages = []
        self.generation = uuid.uuid4().hex[:12]
        self.state = LiveReplayWindow()
        self.open = {}
        self.links = {}
        self.intervals = []
        self.end = 0
        self.end_ticks = None
        self.error = None

    def context(self):
        return deepcopy(dict(active_transfers=list(self.state.flows.values()), faults=list(self.state.faults.values()), link_states=list(self.links.values())))

    def write(self, name, value):
        target = self.folder / name
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf8')
        temporary.replace(target)

    def chunks(self, path, max_bytes):
        """Read only complete atomic members when the producer provides them."""
        parts = sorted((self.output/'runtime-chunks').glob('*.jsonl.gz'))
        sources = parts or [path]
        base = 0
        remaining = max_bytes
        for source in sources:
            size = source.stat().st_size
            if self.offset < base + size and remaining > 0:
                with source.open('rb') as stream:
                    stream.seek(max(0, self.offset-base))
                    while remaining > 0 and (chunk := stream.read(min(256*1024, remaining))):
                        self.offset += len(chunk)
                        remaining -= len(chunk)
                        yield chunk
            base += size
            if remaining <= 0:
                break

    def poll(self, max_bytes=512 * 1024):
        if self.error:
            return
        try:
            self._poll(max_bytes)
        except (zlib.error, ValueError) as error:
            self.error = '通信记录读取失败，已保留原始记录'
            index = self.folder/'live-communication.json'
            value = json.loads(index.read_text()) if index.exists() else dict(
                schema='orbitserve.live_communication.v1',through_order=0,
                end_s=0,pages=[],busy_intervals=[])
            value.update(error=self.error, error_detail=str(error))
            self.write(index.name,value)
            raise

    def _poll(self, max_bytes):
        path = self.output / 'runtime_events.jsonl.gz'
        if not path.is_file() or path.stat().st_size == self.offset:
            return
        events = []
        initial = self.context()

        def flush():
            nonlocal initial
            if not events:
                return
            name = f'live-communication-{self.generation}-{len(self.pages):06d}.json'
            self.write(name, dict(initial=initial, events=events))
            first, last = events[0], events[-1]
            self.pages.append(dict(file=name, start_s=first['time_s'], end_s=last['time_s'], start_ticks=first.get('time_ticks'), end_ticks=last.get('time_ticks'), count=sum(is_transmission_event(row) for row in events)))
            events.clear()
            initial = self.context()

        def accept(line):
            row = json.loads(line)
            order = self.order
            self.order += 1
            self.end, self.end_ticks = row['time_s'], row.get('time_ticks')
            if not is_transmission_event(row) and row['kind'] != 'link_state':
                return
            row = annotate(dict(row, event_order=order))
            events.append(row)
            self.state.observe(row)
            if row['kind'] == 'link_state':
                self.links[row['link_id']] = row
            if row['kind'] == 'transmit_start':
                self.open[row['transfer_id']] = row['time_s']
            if row['kind'] in {'hop_arrival', 'transfer_cancelled'}:
                start = self.open.pop(row.get('transfer_id'), None)
                if start is not None:
                    self.intervals.append([start, row['time_s']])
            if len(events) >= 2048:
                flush()

        for chunk in self.chunks(path,max_bytes):
            while chunk:
                self.pending += self.decoder.decompress(chunk)
                lines = self.pending.split(b'\n')
                self.pending = lines.pop()
                for line in lines:
                    accept(line)
                chunk = self.decoder.unused_data
                if self.decoder.eof:
                    self.decoder = zlib.decompressobj(31)
                else:
                    break
        flush()
        def merge(intervals):
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            return merged
        self.intervals = merge(self.intervals)
        merged = merge(self.intervals + [[start, self.end] for start in self.open.values()])
        self.write('live-communication.json', dict(schema='orbitserve.live_communication.v1', through_order=self.order, end_s=self.end, end_ticks=self.end_ticks, pages=self.pages, busy_intervals=merged))


def watch(root):
    root = Path(root)
    import fcntl
    lock = (root / 'local/communication-publisher.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    publishers = {}
    discover_at = 0
    while True:
        if time.monotonic() >= discover_at:
            discover_at = time.monotonic() + 2
            for path in (root / 'local/results/network').glob('*/runtime_events.jsonl.gz'):
                if path.parent.name not in publishers:
                    status = path.parent / 'status.json'
                    # Status files can be replaced/truncated by another worker
                    # during discovery; one incomplete write must not kill the
                    # only communication publisher for every live experiment.
                    try:
                        active = json.loads(status.read_text()).get('state') in {'running', 'stopping'}
                    except (OSError, ValueError, AttributeError):
                        continue
                    if not active:
                        continue
                    publishers[path.parent.name] = CommunicationPublisher(path.parent)
        for name, publisher in list(publishers.items()):
            if not publisher.output.exists():
                del publishers[name]
                continue
            try:
                publisher.poll()
            except (OSError, EOFError, zlib.error, ValueError):
                pass  # A concurrently appended gzip member may not yet be complete.
        time.sleep(.1)


if __name__ == '__main__':
    watch(__import__('paths').WORKSPACE)
