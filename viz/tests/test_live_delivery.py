"""Polling must remain usable while evidence is being saved."""
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

VIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIZ))
from live import LiveSession, terminal_snapshot, get, SESSIONS


class SnapshotDeliveryTests(unittest.TestCase):
    def test_duplicate_terminal_evidence_does_not_double_business_totals(self):
        from types import SimpleNamespace
        session = LiveSession('counts')
        engine = SimpleNamespace(log=[dict(kind='request_complete',request_id='a')]*2)
        session._count_requests(engine)
        self.assertEqual((session.completed_requests,session.failed_requests), (1,0))
        engine.log.extend([dict(kind='request_failed',request_id='b')]*2)
        session._count_requests(engine)
        self.assertEqual((session.completed_requests,session.failed_requests), (1,1))

    def test_readers_do_not_wait_for_capture_and_cannot_mutate_published_state(self):
        session = LiveSession('test')
        session.payload = dict(id='test', state='paused', version=4, nodes=[{'id':'a'}])
        result = []
        with session.condition:
            thread = threading.Thread(target=lambda: result.append(session.state()))
            thread.start()
            thread.join(1)
            self.assertFalse(thread.is_alive(), 'status read blocked on evidence capture')
        result[0]['nodes'][0]['id'] = 'changed'
        self.assertEqual(session.payload['nodes'][0]['id'], 'a')
        self.assertEqual(session.state(4), dict(unchanged=True, state='paused', version=4))

    def test_terminal_restore_never_opens_large_result_and_running_is_not_resurrected(self):
        temporary = VIZ / '.runtime/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            root = Path(folder)
            self.assertTrue(root.resolve().is_relative_to(temporary.resolve()))
            output = root / 'local/results/network/delivery-test'
            output.mkdir(parents=True)
            (output/'native_result.json').write_text('must not be parsed')
            (output/'status.json').write_text(json.dumps(dict(state='completed',native_scene=True)))
            (output/'native_summary.json').write_text(json.dumps(dict(
                requests={'a':{'status':'completed'},'b':{'status':'failed'}},
                measurement={'simulated_time_ns':2_000_000_001})))
            with patch('paths.WORKSPACE', root):
                try:
                    state = get('delivery-test').state()
                    self.assertEqual((state['completed'],state['failed']), (1,1))
                    self.assertEqual(state['time_ticks'], '2000000001000000000')
                    self.assertFalse(state['snapshot_available'])
                    self.assertIn('/result',state['result_url'])
                finally:
                    SESSIONS.pop('delivery-test', None)
                for saved in ('{incomplete',json.dumps(dict(id='delivery-test',state='completed',version=3,completed=4,failed=0))):
                    (output/'live_state.json').write_text(saved)
                    try:
                        restored=get('delivery-test').state()
                        self.assertEqual((restored['completed'],restored['failed']), (1,1))
                    finally:
                        SESSIONS.pop('delivery-test',None)
            (output/'status.json').write_text('{"state":"running"}')
            self.assertIsNone(terminal_snapshot(output,'delivery-test'))


class JsonDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from runtime_environment import activate
        activate()
        from console import Handler
        cls.respond = staticmethod(Handler.respond)

    def handler(self, accept='gzip', stream=None):
        from types import SimpleNamespace
        headers = {}
        return SimpleNamespace(headers={'Accept-Encoding':accept}, wfile=stream or io.BytesIO(),
            path='/', send_response=lambda code: None,
            send_header=lambda key,value: headers.update({key:value}), end_headers=lambda:None,
            sent_headers=headers)

    def test_negotiated_large_json_roundtrips_and_small_or_forbidden_gzip_stays_plain(self):
        payload = {'data':'卫星' * 50000}
        handler = self.handler()
        self.respond(handler,payload)
        body=handler.wfile.getvalue()
        self.assertEqual(json.loads(gzip.decompress(body)), payload)
        self.assertEqual(handler.sent_headers['Content-Encoding'], 'gzip')
        self.assertEqual(int(handler.sent_headers['Content-Length']),len(body))
        for accept in ('gzip;q=0','br','gzip;q=bad'):
            handler=self.handler(accept)
            self.respond(handler,payload)
            self.assertNotIn('Content-Encoding',handler.sent_headers)
            self.assertEqual(json.loads(handler.wfile.getvalue()),payload)
        handler=self.handler()
        self.respond(handler,{'a':1})
        self.assertNotIn('Content-Encoding',handler.sent_headers)

    def test_cancelled_poll_does_not_raise_and_explicit_gzip_is_not_recompressed(self):
        class ClosedSocket:
            def write(self,content):
                raise BrokenPipeError('client left')
        self.respond(self.handler(stream=ClosedSocket()),{'ok':True})
        handler=self.handler()
        body=gzip.compress(b'hello')
        self.respond(handler,body,encoding='gzip')
        self.assertEqual(handler.wfile.getvalue(),body)


if __name__ == '__main__':
    unittest.main()
