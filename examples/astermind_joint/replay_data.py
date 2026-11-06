"""Turn executed Runtime observations into a deterministic 3D replay trace."""

import math


def build_replay(log, requests, link_profiles, backend):
    hops = {}
    milestones = []
    link_states = []
    for event in log:
        kind = event['kind']
        if kind in {'request_arrival', 'input_done', 'request_complete'}:
            milestones.append({k: event.get(k) for k in ('time_s', 'kind', 'request_id', 'compute_node_id')})
        elif kind == 'link_state':
            link_states.append({k: event.get(k) for k in ('time_s', 'link_id', 'state', 'reason')})
        elif kind == 'transmit_start':
            identity = event['transfer_id']
            if identity in hops:
                raise ValueError(f'duplicate transmission start: {identity}')
            hops[identity] = {
                'transfer_id': identity, 'request_id': event['request_id'],
                'link_id': event['link_id'], 'source': event['source_node_id'],
                'target': event['target_node_id'], 'stage': event['stage'],
                'flow_id': event.get('astermind_flow_id'),
                'bytes': event['payload_bytes'], 'start_s': event['time_s'],
                'finish_s': None, 'arrival_s': None, 'segments': [],
            }
        elif kind == 'bandwidth_update':
            hop = hops[event['transfer_id']]
            hop['segments'].append({k: event[k] for k in (
                'interval_start_s', 'interval_end_s', 'allocated_bandwidth_bps',
                'link_capacity_bps', 'bytes_sent')})
        elif kind in {'transmit_complete', 'hop_arrival'}:
            hop = hops[event['transfer_id']]
            hop['finish_s' if kind == 'transmit_complete' else 'arrival_s'] = event['time_s']
    for hop in hops.values():
        if hop['finish_s'] is None or hop['arrival_s'] is None:
            raise ValueError(f"unfinished hop in completed replay: {hop['transfer_id']}")
        if not hop['start_s'] <= hop['finish_s'] <= hop['arrival_s']:
            raise ValueError(f"out-of-order hop times: {hop['transfer_id']}")
        sent = sum(segment['bytes_sent'] for segment in hop['segments'])
        if not math.isclose(sent, hop['bytes'], rel_tol=1e-9, abs_tol=1e-4):
            raise ValueError(f"transmitted bytes do not match hop payload: {hop['transfer_id']}")
    return {
        'backend': backend,
        'layout': 'schematic_not_to_scale',
        'trace': 'synthetic_static_positions',
        'duration_s': 600,
        'requests': requests,
        'hops': list(hops.values()),
        'milestones': milestones,
        'link_states': link_states,
        'links': [
            {'id': 'sgl:gateway-0:relay', 'source': 'gateway-0', 'target': 'relay', 'profile': 'sgl'},
            {'id': 'cross_layer:a1:relay', 'source': 'relay', 'target': 'a1', 'profile': 'cross_layer'},
            {'id': 'cross_layer:b0:relay', 'source': 'relay', 'target': 'b0', 'profile': 'cross_layer'},
            {'id': 'cell_isl:a0:a1', 'source': 'a1', 'target': 'a0', 'profile': 'cell_isl'},
        ],
        'profiles': {name: {'bandwidth_bps': profile.bandwidth_bps,
                            'capacity_schedule': list(profile.capacity_schedule)}
                     for name, profile in link_profiles.items()},
    }
