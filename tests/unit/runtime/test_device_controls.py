"""Manual fabric controls reach the sole communication runtime."""
import unittest
from tests.unit.runtime import test_comm_port as fixtures
from astermind.engine.device_controls import apply_device_command

class DeviceControlsTests(unittest.TestCase):
    def test_invalid_command_keeps_graph_unchanged(self):
        fixture=fixtures.CommPortTests();self.addCleanup(fixture.doCleanups)
        host,network,comm=fixture.runtime()
        before=host.net.revision
        for command in ({'action':'device_link_fail','target_id':'isl'},
                        {'action':'unsupported','target_id':'alocal'}):
            with self.assertRaises(ValueError):apply_device_command(host,command)
        self.assertEqual(host.net.revision,before)
        self.assertFalse(host.net.failed_device_links)
