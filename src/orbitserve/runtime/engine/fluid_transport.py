"""Deterministic equal-share fluid service on directed physical hops."""
from __future__ import annotations

import math
from ..setup.radio_capacity import capacity_pools, hop_pools, max_min_rates


class FluidTransport:
    def __init__(self, nodes, edges, profiles, *, shared_pools=None, link_pools=None):
        self.now = 0.0
        self.active = {}
        self.capacities = {}
        self.nodes = nodes
        self.pools = capacity_pools(nodes) if shared_pools is None else dict(shared_pools)
        self.link_pools = link_pools
        self.provenance = {
            'backend': 'fluid', 'sharing_scope': 'each_directed_physical_hop',
            'completion_source': 'orbitserve_equal_share_fluid',
            'propagation_accounting': 'orbitserve_after_serialization',
            'packetization': False, 'aggregate_sharing': 'max_min_directed_link_and_satellite_rf_pools',
        }

    def resource(self, leg):
        return (leg['link_id'], leg['source_node_id'], leg['target_node_id'])

    def configure(self, capacities):
        if any(not math.isfinite(v) or v < 0 for v in capacities.values()):
            raise ValueError('fluid capacities must be finite nonnegative bits/s')
        self.capacities.update(capacities)

    def start(self, identity, leg, payload_bytes):
        if identity in self.active or isinstance(payload_bytes, bool) or not math.isfinite(payload_bytes) or payload_bytes <= 0:
            raise ValueError('fluid requires a unique transfer and positive finite bytes')
        self.active[identity] = {'key': self.resource(leg), 'leg': leg, 'remaining': float(payload_bytes)}

    def advance_until(self, time_s):
        if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < self.now:
            raise ValueError('fluid time must be finite and monotonic')
        requirements = {i: (r['key'], *(self.link_pools.get(r['key'], ()) if self.link_pools is not None else
                                      hop_pools(self.nodes, r['leg']['source_node_id'], r['leg']['target_node_id'])))
                        if self.nodes else (r['key'],) for i, r in self.active.items()}
        rates = {i: v / 8 for i, v in max_min_rates(requirements, {**self.capacities, **self.pools}).items()}
        dt = min([time_s - self.now] + [r['remaining'] / rates[i] for i, r in self.active.items() if rates[i] > 0])
        start = self.now
        self.now += dt
        sent = {i: min(r['remaining'], dt * rates[i]) for i, r in self.active.items()}
        done = []
        for i, r in self.active.items():
            # Round only the residual of actual service, relative to that flow.
            # An absolute epsilon completes tiny zero-capacity transfers for free.
            if rates[i] > 0 and math.isclose(sent[i], r['remaining'], rel_tol=1e-12, abs_tol=0.):
                sent[i] = r['remaining']
            r['remaining'] -= sent[i]
            if r['remaining'] <= 0.:
                done.append(i)
        for i in done:
            del self.active[i]
        return {'time_s': self.now, 'interval_start_s': start, 'bytes_sent': sent, 'completed': sorted(done)}

    def cancel(self, identities):
        for identity in identities:
            self.active.pop(identity, None)

    def close(self, force=False):
        self.active.clear()
