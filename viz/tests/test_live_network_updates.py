from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_network_updates import coalesce_network_updates


class SupplyUpdateTests(unittest.TestCase):
    def test_one_supply_build_per_exact_tick_and_revision(self):
        calls = []
        host = SimpleNamespace(now=SimpleNamespace(ticks=0), net=SimpleNamespace(revision=0))
        host.communication_policy = SimpleNamespace(network_changed=lambda: calls.append((host.now.ticks, host.net.revision)))
        coalesce_network_updates(host)
        for _ in range(4381):
            host.communication_policy.network_changed()
        self.assertEqual(calls, [(0, 0)])
        # Faults and user interventions at the same time increment revision.
        host.net.revision += 1
        host.communication_policy.network_changed()
        host.now.ticks += 1
        host.communication_policy.network_changed()
        self.assertEqual(calls, [(0, 0), (0, 1), (1, 1)])

    def test_failed_builds_retry_and_installation_is_idempotent(self):
        calls = []
        def rebuild():
            calls.append(1)
            if len(calls) == 1:
                raise ValueError('frame failed')
        host = SimpleNamespace(now=SimpleNamespace(ticks=0), net=SimpleNamespace(revision=0),
                               communication_policy=SimpleNamespace(network_changed=rebuild))
        coalesce_network_updates(host)
        wrapper = host.communication_policy.network_changed
        coalesce_network_updates(host)
        self.assertIs(host.communication_policy.network_changed, wrapper)
        with self.assertRaisesRegex(ValueError, 'frame failed'):
            wrapper()
        wrapper()
        wrapper()
        self.assertEqual(len(calls), 2)

    def test_float_only_clocks_are_never_coalesced(self):
        calls = []
        host = SimpleNamespace(now=1.0, net=SimpleNamespace(revision=0),
                               communication_policy=SimpleNamespace(network_changed=lambda: calls.append(1)))
        coalesce_network_updates(host)
        for _ in range(3):
            host.communication_policy.network_changed()
        self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
