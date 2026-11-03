"""Artifact publication stays atomic with overlapping producers and failures."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from orbitserve.common import atomic_output, write_json


class ArtifactPublicationTests(unittest.TestCase):
    def test_overlapping_writers_publish_complete_independent_bytes(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'data.bin'
            with atomic_output(path) as first:
                first.write(b'first-')
                self.assertFalse(path.exists())
                with atomic_output(path) as second:
                    second.write(b'second')
                self.assertEqual(path.read_bytes(), b'second')
                first.write(b'complete')
            self.assertEqual(path.read_bytes(), b'first-complete')
            self.assertEqual(first.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(second.sha256, hashlib.sha256(b'second').hexdigest())
            self.assertEqual(list(Path(root).iterdir()), [path])

    def test_failed_writer_preserves_published_peer_and_original_exception(self):
        with TemporaryDirectory() as root:
            path = Path(root) / 'value.json'
            write_json(path, {'old': True})
            with self.assertRaisesRegex(RuntimeError, 'producer failed'):
                with atomic_output(path) as failed:
                    failed.write(b'partial')
                    write_json(path, {'new': True})
                    raise RuntimeError('producer failed')
            self.assertEqual(path.read_bytes(), b'{"new":true}\n')
            self.assertEqual(list(Path(root).iterdir()), [path])
