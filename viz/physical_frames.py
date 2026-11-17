"""Serve a physical boundary from the immutable store without orbital work."""
from functools import lru_cache
import gzip
import json
from pathlib import Path
import threading

LOCK = threading.Lock()


def environment_index(folder):
    folder = Path(folder)
    path = folder / 'network_result.json'
    if not path.is_file():
        path = folder / 'network_replay.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    from astermind.engine.setup.sun_geometry import SUN_GEOMETRY_VERSION
    if (any(p.get('sun_exclusion_angle_deg') is not None for p in value.get('network', {}).get('link_profiles', {}).values())
            and value.get('provenance', {}).get('sun_geometry_version') != SUN_GEOMETRY_VERSION):
        value['physics_warning'] = '历史结果使用旧版太阳避让判定，可能含地球遮挡误判；请重新生成物理场景并运行。此处保留原始执行记录。'
    if not value.get('artifacts', {}).get('frames') and (folder / 'state/index.json').is_file():
        static = json.loads((folder / 'state/index.json').read_text(encoding='utf-8'))
        value.setdefault('artifacts', {})['frames'] = [
            dict(time_s=time, artifact=f'frames/frame-{index:06d}.json.gz')
            for index, time in enumerate(static['regular_times'])]
    if (folder / 'state/index.json').is_file():
        from motion_frames import motion_index
        value['motion_frames'] = motion_index(folder)
        # State transitions travel with the requested physical interval. A day
        # of link history must not inflate the initial browser payload.
        value['events'] = []
    return value


@lru_cache(maxsize=1)
def source(folder):
    from astermind.builder.physical.snapshot_store import SnapshotSource
    return SnapshotSource(folder)


@lru_cache(maxsize=8)
def _frame(folder, number):
    model = source(folder)
    if not 0 <= number < len(model.regular_times):
        raise ValueError('物理快照编号超出时域')
    # SQLite read sessions belong to the request thread and are always closed.
    with model:
        time = model.regular_times[number]
        snapshot = model.snapshot(time)
        end = model.regular_times[number+1] if number+1 < len(model.regular_times) else model.end_s
        snapshot['state_events'] = [event for event in model.events if time <= event['time_s'] < end]
    return gzip.compress(json.dumps(snapshot, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode(), compresslevel=1, mtime=0)


def frame_bytes(folder, number):
    with LOCK:
        return _frame(str(Path(folder).resolve()), number)
