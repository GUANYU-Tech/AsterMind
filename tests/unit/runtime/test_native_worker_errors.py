"""Native crashes retain their root diagnostic and identify the right backend."""
from io import StringIO
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from orbitserve.runtime.engine.ns3_transport import Ns3Transport, Ns3Unavailable
from orbitserve.runtime.engine.simgrid_transport import SimGridTransport, SimGridUnavailable


class NativeWorkerErrorTests(unittest.TestCase):
    def failed_worker(self, kind, diagnostic):
        transport = kind.__new__(kind)
        transport.closed, transport.now, transport.reader_error = False, 0., None
        transport.close = Mock()
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

    def test_malformed_responses_and_timeouts_retire_worker(self):
        for reply in ('[]', '{}', '{"time_s":NaN}', '{"time_s":-1}', '{"time_s":true}', 'invalid JSON', 'timeout'):
            with self.subTest(reply=reply):
                transport = self.failed_worker(Ns3Transport, '')
                transport.responses.get()
                if reply == 'timeout':
                    from queue import Empty
                    transport.responses = Mock()
                    transport.responses.get.side_effect = Empty('timed out')
                else:
                    transport.responses.put(reply)
                with self.assertRaisesRegex(Ns3Unavailable, 'protocol failed'):
                    transport._request({'op': 'advance'})
                transport.close.assert_called_once_with(force=True)
                self.assertEqual(transport.now, 0.)

    def test_native_operation_error_keeps_worker_usable(self):
        transport = self.failed_worker(Ns3Transport, '')
        transport.responses.get()
        transport.responses.put('{"time_s":0,"error":"invalid operation"}')
        transport.responses.put('{"time_s":1,"completed":[]}')
        with self.assertRaisesRegex(Ns3Unavailable, 'invalid operation'):
            transport._request({'op': 'invalid'})
        self.assertEqual(transport._request({'op': 'advance'})['time_s'], 1.)
        transport.close.assert_not_called()

    def test_invalid_profiles_fail_before_subprocess_creation(self):
        from unittest.mock import patch
        edge = dict(endpoint_a='a', endpoint_b='b', link_id='ab', kind='isl')
        for kind in (SimGridTransport, Ns3Transport):
            with self.subTest(backend=kind.__name__), patch.object(kind, 'check_available'), \
                    patch('orbitserve.runtime.engine.simgrid_transport.subprocess.Popen') as start:
                with self.assertRaises(KeyError):
                    kind(('a', 'b'), [edge], {})
                start.assert_not_called()
