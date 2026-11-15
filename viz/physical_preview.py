"""Independent physical route queries; no Workload or Serving execution."""
from functools import lru_cache
import json
import math
from pathlib import Path
import threading

from astermind.builder.orbit import load_orbit_trace
from astermind.engine import NetworkSpec, NetworkState, NetworkUnavailable

POLICIES = ('min_hops', 'min_propagation')
LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_state(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'materialized_input.json').read_text(encoding='utf-8'))
    trace = load_orbit_trace(folder / 'orbit_trace/trace_manifest.json')
    return NetworkState(trace, NetworkSpec.from_value(manifest['network']), manifest['ground_ingresses'],
                        physical_state=folder / 'physical_state.sqlite' if (folder / 'physical_state.sqlite').is_file() else None)


@lru_cache(maxsize=8)
def query_state(folder, time_s):
    network = load_state(folder).fork()
    for observed_at, index, restored, fault in network.fault_observations():
        if observed_at > time_s:
            break
        network.observe_fault(index, fault, observed_at, restored=restored)
    return network


def probe_routes(folder, payload):
    time_s = float(payload['time_s'])
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError('查询时间必须有限且非负')
    source, target = payload['source'], payload['target']
    # A time-local state makes backwards playback causal while repeated queries
    # share a path table. Probes neither reserve capacity nor forecast messages.
    with LOCK:
        network = query_state(str(Path(folder).resolve()), time_s)
        if source not in network.nodes or target not in network.nodes:
            raise ValueError('起点或终点不属于此物理模型')
        if not network.start_s <= time_s <= network.end_s:
            raise ValueError('查询时间超出此物理模型')
        rows = []
        for policy in POLICIES:
            row = dict(policy=policy, source=source, target=target, time_s=time_s)
            try:
                selected = network.select_route(source, target, time_s, policy)
                path = [dict(link_id=identity, source_node_id=a, target_node_id=b)
                        for identity, a, b in selected.hops]
                row.update(completed=True, path=path, hops=len(path),
                           distance_km=selected.distance_km, routing_cost=selected.cost)
            except NetworkUnavailable as error:
                row.update(completed=False, reason=str(error), path=[])
            rows.append(row)
        return dict(time_s=time_s, source=source, target=target, routes=rows,
                    semantics='current_topology_path_query_no_transfer_prediction',
                    route_cache=network.path_route_cache_info(), future_faults_known=False)
