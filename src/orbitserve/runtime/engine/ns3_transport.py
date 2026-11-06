"""Optional native ns-3 packet transport; no silent analytical fallback."""
import importlib.util
from .simgrid_transport import SimGridTransport, SimGridUnavailable


class Ns3Unavailable(SimGridUnavailable):
    """The explicitly selected native ns-3 worker could not run."""


def require_ns3():
    if importlib.util.find_spec('ns') is None:
        raise Ns3Unavailable('ns-3 mode requires the optional ns3 package; install OrbitServe[ns3] on a supported platform')


class Ns3Transport(SimGridTransport):
    worker_module = 'orbitserve.runtime.engine.ns3_worker'
    worker_name = 'ns-3'
    error_type = Ns3Unavailable
    check_available = staticmethod(require_ns3)

    def __init__(self, nodes, edges, profiles, *, start_s=0.0, record_packets=False):
        super().__init__(nodes, edges, profiles, start_s=start_s)
        self.packet_trace = []
        if record_packets:
            self._request({'op':'trace','enabled':True})
        self.provenance = {
            'backend': 'ns3', 'ns3_version': self.provenance['simgrid_version'],
            'packet_trace_enabled': record_packets,
            'completion_source': 'native_ns3_point_to_point_phy_tx_end',
            'sharing_scope': 'each_directed_physical_hop',
            'packet_payload_bytes': 1400, 'wire_overhead_bytes': 2,
            'arbitration': 'round_robin_packets', 'tcp': False,
            'capacity_change': 'next_packet_boundary',
            'aggregate_sharing': 'round_robin_packets_exclusive_satellite_rf_pools',
            'payload_quantization_bytes': 1e-6,
            'wire_payload_quantization_bytes': 1,
            'fractional_payload': 'logical_bytes_preserved_wire_tail_padded_to_integer',
            'time_resolution_s': 1e-12,
            'propagation_accounting': 'orbitserve_after_serialization',
        }

    def register_edge(self, edge):
        from ..setup.radio_capacity import hop_pools
        rows = []
        for direction, (a, b) in enumerate(((edge['endpoint_a'], edge['endpoint_b']), (edge['endpoint_b'], edge['endpoint_a']))):
            key = f'link-{len(self.edges)}-{direction}'
            rows.append(dict(key=key, pools=list(hop_pools(self.nodes, a, b))))
        self._request({'op': 'register', 'links': rows})
        for row, (a, b) in zip(rows, ((edge['endpoint_a'], edge['endpoint_b']), (edge['endpoint_b'], edge['endpoint_a']))):
            self.resources[edge['link_id'], a, b] = row['key']
        self.edges = (*self.edges, edge)

    def advance_until(self, time_s):
        response = super().advance_until(time_s)
        self.provenance['native_counters'] = response['native_counters']
        self.packet_trace.extend(response['packets'])
        return response
