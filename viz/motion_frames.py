"""Small, on-demand ITRS position/velocity frames for visual interpolation."""

from functools import lru_cache
import gzip
import json
from pathlib import Path

import numpy as np


@lru_cache(maxsize=8)
def motion_bytes(folder, number):
    from physical_frames import LOCK, source
    folder = Path(folder)
    with LOCK:
        model = source(str(folder.resolve()))
        if not 0 <= number < len(model.environment_times):
            raise ValueError('运动采样编号超出时域')
        time = model.environment_times[number]
        velocities = {}
        index_path = folder / 'orbit_trace/ephemeris_trace.json'
        # Numeric shards keep a day-long trace out of the web server heap.
        if index_path.is_file() and index_path.stat().st_size < 1_000_000:
            index = json.loads(index_path.read_text(encoding='utf-8'))
            if index.get('storage') == 'orbitserve.ephemeris_columns.v1':
                for shard in index['shards']:
                    if shard['start_s'] <= time <= shard['end_s']:
                        path = Path(shard['path'])
                        if path.is_absolute() or '..' in path.parts:
                            raise ValueError('非法运动数据路径')
                        values = np.load(index_path.parent / path, mmap_mode='r', allow_pickle=False)
                        for row in values[values['time_s'] == time]:
                            velocities[index['satellite_ids'][int(row['satellite_index'])]] = row['velocity_itrs_km_s'].tolist()
        with model:
            before = model.environment_times[max(0, number-1)]
            after = model.environment_times[min(len(model.environment_times)-1, number+1)]
            rows = []
            for node in model.nodes.values():
                velocity = velocities.get(node.node_id)
                if node.role == 'ground':
                    velocity = [0., 0., 0.]
                if velocity is None:
                    # Legacy physical stores lack velocities; visibly declare
                    # this approximation instead of claiming recorded vectors.
                    a, b = model.position(node.node_id, before), model.position(node.node_id, after)
                    velocity = [(y-x)/(after-before) if after > before else 0. for x, y in zip(a, b)]
                rows.append(dict(node_id=node.node_id, position_itrs_km=model.position(node.node_id, time),
                                 velocity_itrs_km_s=velocity))
        value = dict(time_s=time, nodes=rows, velocity_source='ephemeris' if velocities else 'estimated')
        return gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), compresslevel=1, mtime=0)


def motion_index(folder):
    index = json.loads((Path(folder)/'state/index.json').read_text(encoding='utf-8'))
    return [dict(time_s=time, artifact=f'motion/frame-{number:06d}.json.gz')
            for number, time in enumerate(index['environment_times'])]
