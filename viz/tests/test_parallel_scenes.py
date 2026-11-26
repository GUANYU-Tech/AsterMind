"""Full-scene concurrency regression, using the Linux simulation environment."""
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

VIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIZ))


@unittest.skipUnless(sys.platform == 'linux', 'the simulation runtime requires Linux')
class ParallelScenes(unittest.TestCase):
    def test_new_physical_models_get_separate_workers(self):
        import console
        from paths import RUNTIME
        from studio import save

        physical = json.loads((VIZ.parent / 'examples/native/dynamic_physics.json').read_text())
        with tempfile.TemporaryDirectory(dir=RUNTIME / 'tmp') as directory:
            root = Path(directory)
            with patch.object(console, 'ROOT', root), patch.object(console, 'JOBS', {}):
                models = [save(root, 'physical', {'name': name, 'physical_network': physical})['id']
                          for name in ('A', 'B', 'C')]
                def prepare(model):
                    return console.prepare_physical(model, independent=True)['id']

                with ThreadPoolExecutor(max_workers=2) as pool:
                    runs = list(pool.map(prepare, models[:2]))
                self.assertNotEqual(*runs)
                for identity in runs:
                    self.wait_physical(console, identity)
                # New task C uses the same parameters as completed tasks A and B.
                # The frontend's independent launch must still build C's topology.
                last = prepare(models[2])
                self.assertNotIn(last, runs)
                self.wait_physical(console, last)

    def wait_physical(self, console, identity):
        deadline = time.monotonic() + 120
        while (status := console.run_status(identity))['state'] == 'running' and time.monotonic() < deadline:
            time.sleep(.25)
        self.assertEqual(status['state'], 'completed', status)
        self.assertTrue((Path(status['output']) / 'physical_dataset.json').is_file())

    def test_independent_runs_and_targeted_stop(self):
        import console
        import live
        import paths

        examples = VIZ.parent / 'examples/native'
        def scene(name):
            value = json.loads((examples / 'dynamic_tp2_pp2.json').read_text())
            value['physical_network'] = json.loads((examples / 'dynamic_physics.json').read_text())
            value['requests'][0]['id'] = name
            return value

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(console, 'ROOT', root), patch.object(paths, 'WORKSPACE', root):
                class Handler(console.Handler):
                    def log_message(self, *args):
                        pass

                server = self.enterContext(ThreadingHTTPServer(('127.0.0.1', 0), Handler))
                threading.Thread(target=server.serve_forever, daemon=True).start()
                self.addCleanup(server.shutdown)
                def request(route, body=None):
                    data = None if body is None else json.dumps(body).encode()
                    req = Request(f'http://127.0.0.1:{server.server_port}' + route, data=data,
                                  headers={'Content-Type': 'application/json'})
                    with urlopen(req, timeout=60) as response:
                        return json.load(response)

                def launch(name, batch=False):
                    saved = request('/save-scene', {'name': name, 'experiment': scene(name)})
                    if batch:
                        return request('/launch-scene', {'id': saved['id']})
                    return request('/live/start', {'scene_id': saved['id']})

                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = list(pool.map(launch, ('first', 'second')))
                first, second = [live.get(job['id']) for job in jobs]
                try:
                    self.assertNotEqual(first.process.pid, second.process.pid)
                    catalog = request('/task-catalog')
                    self.assertEqual({row['state'] for row in catalog}, {'running'})
                    self.assertEqual({row['title'] for row in catalog}, {'first', 'second'})
                    for worker in (first, second):
                        deadline = time.monotonic() + 180
                        url = worker.job['runtime_url']
                        while (state := request(url + '/state'))['state'] == 'preparing' and time.monotonic() < deadline:
                            time.sleep(0.2)
                        self.assertEqual(state['state'], 'paused', console.run_status(worker.job['id']))
                        self.assertTrue((worker.output / 'physical' / 'physical_dataset.json').is_file())
                    request('/runs/' + first.job['id'] + '/stop', {})
                    self.assertTrue(first.finished.wait(20))
                    self.assertEqual(console.run_status(first.job['id'])['state'], 'cancelled')
                    self.assertEqual(second.state()['state'], 'paused')
                    request(second.job['runtime_url'] + '/advance', {'until_s': second.state()['end_s'], 'mode': 'time'})
                    self.assertTrue(second.finished.wait(180))
                    self.assertEqual(console.run_status(second.job['id'])['replay_state'], 'ready')

                    # The default Run button starts a batch. Submit two together
                    # and require both to compute and export their own results.
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        batches = list(pool.map(lambda name: launch(name, batch=True), ('third', 'fourth')))
                    workers = [live.get(batch['id']) for batch in batches]
                    try:
                        running = {row['id'] for row in request('/task-catalog') if row['state'] == 'running'}
                        self.assertEqual(running, {batch['id'] for batch in batches})
                        for worker, name in zip(workers, ('third', 'fourth')):
                            self.assertTrue(worker.finished.wait(180))
                            status = console.run_status(worker.job['id'])
                            self.assertEqual(status['state'], 'completed', status)
                            self.assertEqual(status['replay_state'], 'ready', status)
                            self.assertTrue((worker.output / 'physical' / 'physical_dataset.json').is_file())
                            summary = json.loads((worker.output / 'native_summary.json').read_text())
                            self.assertEqual(set(summary['requests']), {name})
                    finally:
                        for worker in workers:
                            worker.stop()
                            worker.finished.wait(10)
                finally:
                    for worker in (first, second):
                        worker.stop()
                        worker.finished.wait(10)


if __name__ == '__main__':
    unittest.main()
