"""Independent physical route queries; no Workload or Serving execution."""
from functools import lru_cache
import json
import math
from pathlib import Path
import threading

from astermind.builder.orbit import load_orbit_trace
from astermind.engine import NetworkSpec, NetworkState, NetworkUnavailable

POLICIES = ('min_hops', 'min_propagation', 'earliest_arrival')
LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_state(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'materialized_input.json').read_text(encoding='utf-8'))
    trace = load_orbit_trace(folder / 'orbit_trace/trace_manifest.json')
    return NetworkState(trace, NetworkSpec.from_value(manifest['network']), manifest['ground_ingresses'],
                        physical_state=folder / 'physical_state.sqlite' if (folder / 'physical_state.sqlite').is_file() else None)


def probe_routes(folder, payload):
    time_s, size = float(payload['time_s']), float(payload.get('payload_bytes', 0))
    if not all(math.isfinite(v) and v >= 0 for v in (time_s, size)) or size > 1e12:
        raise ValueError('时间和探测字节数必须有限且非负；探测上限 1 TB')
    allow_wait = payload.get('allow_wait', False)
    if type(allow_wait) is not bool:
        raise ValueError('allow_wait must be boolean')
    source, target = payload['source'], payload['target']
    # Bound the cache to one constellation and serialize expensive queries.
    # Each request starts from a fresh fork: previous probes reserve no resources.
    with LOCK:
        network = load_state(str(Path(folder).resolve())).fork()
        if source not in network.nodes or target not in network.nodes:
            raise ValueError('起点或终点不属于此物理模型')
        if not network.start_s <= time_s <= network.end_s:
            raise ValueError('查询时间超出此物理模型')
        for observed_at, index, restored, fault in network.fault_observations():
            if observed_at > time_s:
                break
            network.observe_fault(index, fault, observed_at, restored=restored)
        rows = []
        for policy in POLICIES:
            row = dict(policy=policy, source=source, target=target, time_s=time_s, payload_bytes=size)
            try:
                arrival, path = network.fork().transfer(source, target, time_s, size, policy, allow_wait=allow_wait)
                row.update(completed=True, arrival_s=arrival, path=path,
                           latency_s=arrival-time_s, hops=len(path),
                           propagation_s=sum(p['propagation_delay_s'] for p in path),
                           wait_s=sum(p['queue_wait_s'] for p in path),
                           serialization_s=sum(p['finish_s']-p['start_s'] for p in path))
            except NetworkUnavailable as error:
                row.update(completed=False, reason=str(error), path=[])
            rows.append(row)
        return dict(time_s=time_s, source=source, target=target, routes=rows,
                    semantics='independent_non_reserving_dynamic_route_prediction',
                    allow_wait=allow_wait, future_faults_known=False)
