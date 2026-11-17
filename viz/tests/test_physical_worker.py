import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

VIZ = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('physical_frontend_worker', VIZ / 'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
TEST_TMP = VIZ / '.runtime/tmp'
TEST_TMP.mkdir(parents=True, exist_ok=True)


class PhysicalWorkerTests(unittest.TestCase):
    def test_backend_selects_parallelism_and_new_logs_are_unquoted_unicode(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            root = Path(folder)
            self.assertTrue(root.resolve().is_relative_to(TEST_TMP.resolve()))
            manifest = root / 'input.json'
            manifest.write_text(json.dumps({'name': '中文场景'}, ensure_ascii=False), encoding='utf-8')
            calls = []

            def runner(value, output, **options):
                calls.append((value, output, options))
                options['progress']('Orbit batch 1/10: 0–61 s')
                return {'checks': {'passed': True, 'communication_candidate_degrees': {'sat-a': 4}}}

            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                result = worker.prepare_physical(manifest, root / 'output', runner)
            self.assertTrue(result['checks']['passed'])
            self.assertEqual(result['checks']['communication_candidate_degrees'], {'sat-a': 4})
            self.assertEqual(calls[0][0]['name'], '中文场景')
            self.assertNotIn('_workers', calls[0][2])
            self.assertNotIn('export_workers', calls[0][2])
            self.assertEqual(log.getvalue().splitlines()[0], 'Orbit batch 1/10: 0–61 s')
            self.assertNotIn('\\u2013', log.getvalue())
            self.assertNotIn('communication_candidate_degrees', json.loads(log.getvalue().splitlines()[-1])['checks'])


if __name__ == '__main__':
    unittest.main()
