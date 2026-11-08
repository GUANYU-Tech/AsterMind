"""Native link reports must account for idle time, cancellation and capacity changes."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('_flowsim'), 'requires native FlowSim')
class FlowSimStatisticsTests(unittest.TestCase):
    def test_capacity_integral_includes_idle_links_and_reports_are_read_only(self):
        import _flowsim

        graph = _flowsim.Topology()
        for name in ('a', 'b', 'c', 'd'):
            graph.add_node(name)
        graph.add_link('a', 'b', 8., 0, False)
        graph.add_link('c', 'd', 4., 0, False)
        sim = _flowsim.EventSimulator(graph)

        def advance(target):
            for _ in range(20):
                if sim.now_ns >= target:
                    return
                sim.advance_until(target)
            self.fail('native clock did not reach the requested barrier')

        sim.start_hop('cancel', 'a', 'b', 100, lambda: None)
        advance(50)
        sim.cancel('cancel')
        advance(75)
        done = []
        sim.start_hop('complete', 'a', 'b', 200, lambda: done.append(sim.now_ns))
        advance(200)
        sim.set_bandwidth('c', 'd', 8.)
        advance(275)
        self.assertEqual(done, [275.])
        first = sim.report()
        self.assertEqual(sim.report().links[0].busy_time_ns, first.links[0].busy_time_ns)
        advance(300)
        sim.set_bandwidth('a', 'b', 0.)
        advance(400)
        sim.set_bandwidth('a', 'b', 4.)
        advance(500)
        links = {(x.source, x.target): x for x in sim.report().links}
        ab, cd = links['a', 'b'], links['c', 'd']
        self.assertEqual(ab.bytes_transferred, 200)
        self.assertEqual(ab.busy_time_ns, 250)
        self.assertEqual(ab.peak_active_flows, 1)
        self.assertAlmostEqual(ab.utilization, 200 / 350)
        self.assertEqual((cd.bytes_transferred, cd.busy_time_ns, cd.peak_active_flows), (0, 0, 0))

    def test_cancel_before_first_advance_does_not_create_busy_interval(self):
        import _flowsim

        graph = _flowsim.Topology()
        graph.add_node('a')
        graph.add_node('b')
        graph.add_link('a', 'b', 8., 0, False)
        sim = _flowsim.EventSimulator(graph)
        sim.start_hop('cancel', 'a', 'b', 1000, lambda: self.fail('cancelled callback'))
        sim.cancel('cancel')
        sim.advance_until(1000)
        row = sim.report().links[0]
        self.assertEqual((row.bytes_transferred, row.busy_time_ns, row.peak_active_flows), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
