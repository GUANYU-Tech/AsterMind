import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime

from viz.tasks import record_progress, write_status


class TaskProgressTests(unittest.TestCase):
    def test_record_only_observed_transitions_including_cell_counts(self):
        job = dict(status='running', phase='cells', progress=20)
        for count in (0, 0, 1):
            job['startup_stage'] = dict(phase='cells', completed=count, total=3)
            record_progress(job)
        history = job['progress_history']
        self.assertEqual([row['event']['completed'] for row in history], [0, 1])
        self.assertTrue(all(datetime.fromisoformat(row['at']).utcoffset().total_seconds() == 0
                            for row in history))
        self.assertEqual(history[-1]['event']['total'], 3)

    def test_write_and_reload_bounded_history(self):
        with tempfile.TemporaryDirectory() as folder:
            job = dict(output=folder)
            for count in range(102):
                write_status(job, 'running', phase=str(count))
            saved = json.loads((Path(folder) / 'status.json').read_text(encoding='utf-8'))
            self.assertTrue(saved['progress_history_truncated'])
            self.assertEqual(len(saved['progress_history']), 100)
            self.assertEqual(saved['progress_history'][0]['event']['phase'], '2')
            self.assertEqual(saved['progress_history'][-1]['event']['phase'], saved['phase'])
