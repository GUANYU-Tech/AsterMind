import unittest
from orbitserve.runtime import export_cell_network


def snapshot():
    nodes = [
        dict(node_id=n, cell_id="cell", role="compute", position_itrs_km=[7000, i, 0])
        for i, n in enumerate("abc")
    ]
    nodes.append(dict(node_id="relay", role="communication"))
    links = [
        dict(
            link_id=a + b,
            endpoint_a=a,
            endpoint_b=b,
            kind="cell_isl",
            state="UP",
            bandwidth_bps=rate,
            propagation_delay_s=delay,
        )
        for a, b, rate, delay in [
            ("a", "b", 200e9, 1e-6),
            ("b", "c", 180e9, 2e-6),
            ("a", "c", 190e9, 3e-6),
        ]
    ]
    links.append(
        dict(
            link_id="ar",
            endpoint_a="a",
            endpoint_b="relay",
            kind="cross_layer",
            state="UP",
            bandwidth_bps=50e9,
            propagation_delay_s=0.002,
        )
    )
    links.append(
        dict(
            link_id="ag",
            endpoint_a="a",
            endpoint_b="ground",
            kind="sgl",
            state="UP",
            bandwidth_bps=1e9,
            propagation_delay_s=0.003,
        )
    )
    return dict(time_s=7, valid_until_s=8, nodes=nodes, links=links)


class CellNetworkExchangeTests(unittest.TestCase):
    def test_asymmetric_gsl_is_not_exported_as_a_symmetric_rate(self):
        frame = snapshot()
        frame["links"][-1]["directions"] = [
            dict(
                source_node_id="ground",
                target_node_id="a",
                bandwidth_bps=1e8,
                shared_pools=["ul"],
            ),
            dict(
                source_node_id="a",
                target_node_id="ground",
                bandwidth_bps=4e8,
                shared_pools=["dl"],
            ),
        ]
        frame["shared_capacity_pools_bps"] = {"ul": 1e8, "dl": 4e8}
        result = export_cell_network(frame)
        link = result["cells"][0]["links"][-1]
        self.assertIsNone(link["bandwidth_gbps_per_direction"])
        self.assertEqual(
            [d["bandwidth_gbps"] for d in link["bandwidth_gbps_by_direction"]],
            [0.1, 0.4],
        )
        self.assertEqual(
            result["shared_capacity_pools_bps"], frame["shared_capacity_pools_bps"]
        )

    def test_preserves_exact_edges_and_exports_conservative_static_fragments(self):
        cell = export_cell_network(snapshot())["cells"][0]
        self.assertEqual(cell["deployment"]["rank_mapping"], {"0": 0, "1": 1, "2": 2})
        self.assertEqual(len(cell["links"]), 5)
        self.assertEqual(cell["links"][3]["bandwidth_gbps_per_direction"], 50)
        self.assertEqual(cell["links"][3]["latency_ns"], 2e6)
        static = cell["astermind_static_fragment"]["network"]
        self.assertEqual(
            static, dict(topology="FullyConnected", bandwidth_gbps=180, latency_ns=3000)
        )
        serving = cell["serving_formation_fragment"]
        self.assertEqual(serving["topology"], "fully_connected")
        self.assertAlmostEqual(serving["link_propagation_delay_s"], 3e-6)
        self.assertFalse(cell["astermind_compatibility"]["exact_static_import"])

    def test_down_or_zero_capacity_never_exports_usable_uniform_network(self):
        for key, value in [("state", "DOWN"), ("bandwidth_bps", 0)]:
            frame = snapshot()
            frame["links"][0][key] = value
            cell = export_cell_network(frame)["cells"][0]
            self.assertIsNone(cell["astermind_static_fragment"])
            self.assertIsNone(cell["serving_formation_fragment"])
            self.assertEqual(cell["links"][0][key], value)
