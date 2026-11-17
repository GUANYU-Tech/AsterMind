"""Lossless native reports with shared capacity tables and streaming hashes.

The solver/auditor still see ordinary allocation events. Only the on-disk
representation uses references; load_result restores the original schema.
Legacy reports without storage metadata remain supported.
"""
import hashlib
import json
from pathlib import Path

from astermind.common.artifact_io import atomic_output, write_json

SCHEMA = 'orbitserve.native_result_storage.v1'


class CapacityEvents(list):
    def __init__(self, events, tables):
        self.events, self.tables = events, tables
        self.source, self.reference = None, None

    def __len__(self):
        return len(self.events)

    def __bool__(self):
        return bool(self.events)

    def __iter__(self):
        for event in self.events:
            if event.get('kind') != 'allocation' or 'capacities' not in event:
                yield event
                continue
            capacities = event['capacities']
            if capacities is not self.source:
                raw = json.dumps(capacities, sort_keys=True, ensure_ascii=False,
                                 separators=(',', ':'), allow_nan=False).encode('utf-8')
                self.reference = hashlib.sha256(raw).hexdigest()
                self.tables.setdefault(self.reference, capacities)
                self.source = capacities
            row = {key: value for key, value in event.items() if key != 'capacities'}
            row['capacity_snapshot_id'] = self.reference
            yield row


def write_result(path, result, *, default=None):
    """Publish a self-contained report; never change live solver records."""
    path = Path(path)
    document = dict(result)
    tables = {}
    if 'communication_events' in document:
        document['communication_events'] = CapacityEvents(document['communication_events'], tables)
        # Written after communication_events, once its iterator collected tables.
        document['_storage'] = dict(schema=SCHEMA, capacity_snapshots=tables)
    encoder = json.JSONEncoder(default=default, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':'))
    with atomic_output(path) as output:
        parts, size = [], 0
        for part in encoder.iterencode(document):
            parts.append(part)
            size += len(part)
            if size >= 256 * 1024:
                output.write(''.join(parts).encode('utf-8'))
                parts, size = [], 0
        if parts:
            output.write(''.join(parts).encode('utf-8'))
    stat = path.stat()
    digest = dict(sha256=output.sha256, size=stat.st_size, stamp=stat.st_mtime_ns)
    write_json(path.with_suffix(path.suffix + '.digest.json'), digest)
    return digest


def restore_result(document):
    if not isinstance(document, dict):
        return document
    storage = document.get('_storage')
    if storage is None:
        return document
    if storage.get('schema') != SCHEMA:
        raise ValueError('Unsupported native result storage schema')
    tables = storage['capacity_snapshots']
    for identity, capacities in tables.items():
        raw = json.dumps(capacities, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
        if hashlib.sha256(raw).hexdigest() != identity:
            raise ValueError('Native result capacity snapshot checksum mismatch')
    for event in document.get('communication_events', ()):
        identity = event.pop('capacity_snapshot_id', None)
        if identity is not None:
            if identity not in tables or 'capacities' in event:
                raise ValueError('Invalid native result capacity snapshot reference')
            event['capacities'] = tables[identity]
    del document['_storage']
    return document


def load_result(path):
    with Path(path).open(encoding='utf-8') as stream:
        return restore_result(json.load(stream))


def result_digest(path):
    path = Path(path)
    try:
        value = json.loads(path.with_suffix(path.suffix + '.digest.json').read_text())
        stat = path.stat()
        if (value['size'], value['stamp']) == (stat.st_size, stat.st_mtime_ns):
            return value['sha256']
    except (OSError, ValueError, KeyError, TypeError):
        pass
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
