"""Native crashes retain their root diagnostic and identify the right backend."""
from io import StringIO
from queue import Queue
from types import SimpleNamespace
import unittest

from orbitserve.runtime.engine.ns3_transport import Ns3Transport, Ns3Unavailable
from orbitserve.runtime.engine.simgrid_transport import SimGridTransport, SimGridUnavailable


class NativeWorkerErrorTests(unittest.TestCase):
    def failed_worker(self, kind, diagnostic):
        transport = kind.__new__(kind)
        transport.process = SimpleNamespace(stdin=StringIO(), poll=lambda: -11)
        transport.errors = StringIO(diagnostic)
        transport.responses = Queue()
        transport.responses.put(None)
        return transport

    def test_crash_reports_backend_and_exit_status(self):
        for kind, error, name in ((SimGridTransport, SimGridUnavailable, 'SimGrid'),
                                   (Ns3Transport, Ns3Unavailable, 'ns-3')):
            with self.subTest(backend=name):
                transport = self.failed_worker(kind, 'native root cause')
                with self.assertRaisesRegex(error, name + r' worker exited \(returncode -11\)') as caught:
                    transport._request({'op': 'start'})
                self.assertIn('native root cause', str(caught.exception))

    def test_large_stack_trace_keeps_root_cause_and_tail(self):
        diagnostic = 'ROOT: invalid native pointer\n' + 'stack frame\n' * 3000 + 'FINAL STACK FRAME'
        transport = self.failed_worker(Ns3Transport, diagnostic)
        with self.assertRaises(Ns3Unavailable) as caught:
            transport._request({'op': 'start'})
        message = str(caught.exception)
        self.assertIn('ROOT: invalid native pointer', message)
        self.assertIn('FINAL STACK FRAME', message)
        self.assertIn('stderr truncated', message)
        self.assertLess(len(message), 12500)
