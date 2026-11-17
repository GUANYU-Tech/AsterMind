"""Lossless shared capacity snapshots for append-only live evidence.

Only repeated diagnostic tables use references. Solver traces and events keep
their original contents; expand_event reconstructs an archived event exactly.
"""
import gzip
import hashlib
import json
from pathlib import Path


class CapacitySnapshots:
    def __init__(self, output):
        self.output = Path(output)
        self.capacities = None
        self.reference = None
        self.source = None

    def encode(self, event):
        if event.get('kind') != 'v1_allocation' or 'capacities' not in event:
            return event
        capacities = event['capacities']
        if self.reference is None or (capacities is not self.source and capacities != self.capacities):
            raw = json.dumps(capacities, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
            digest = hashlib.sha256(raw).hexdigest()
            artifact = 'runtime-capacities/' + digest + '.json.gz'
            target = self.output / artifact
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix('.tmp')
                temporary.write_bytes(gzip.compress(raw, compresslevel=1))
                temporary.replace(target)
            self.capacities = capacities
            self.reference = dict(schema='orbitserve.capacity_snapshot.v1', artifact=artifact,
                                  sha256=digest, pool_count=len(capacities))
        self.source = capacities
        row = dict(event)
        del row['capacities']
        row['capacity_snapshot'] = self.reference
        return row


def expand_event(event, output):
    reference = event.get('capacity_snapshot')
    if not reference:
        return event
    folder = (Path(output) / 'runtime-capacities').resolve()
    path = (Path(output) / reference['artifact']).resolve()
    if path.parent != folder or reference.get('schema') != 'orbitserve.capacity_snapshot.v1':
        raise ValueError('Invalid capacity snapshot reference')
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('Capacity snapshot checksum mismatch')
    row = dict(event)
    del row['capacity_snapshot']
    row['capacities'] = json.loads(raw)
    return row
