"""Optional native ns-3 packet transport; no silent analytical fallback."""
import importlib.util
from .simgrid_transport import SimGridTransport


def require_ns3():
    if importlib.util.find_spec('ns') is None:
        raise RuntimeError('ns-3 mode requires the optional ns3 package; install OrbitServe[ns3] on a supported platform')


class Ns3Transport(SimGridTransport):
    worker_module = 'orbitserve.runtime.engine.ns3_worker'
    check_available = staticmethod(require_ns3)

    def __init__(self, nodes, edges, profiles):
        super().__init__(nodes, edges, profiles)
        self.provenance = {
            'backend': 'ns3', 'ns3_version': self.provenance['simgrid_version'],
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
