"""Explicit real native matrix; choose scheduler environment as in tests/README.md."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/scripts'))
from run_compute_comm import run

class NativeComputeCommTests(unittest.TestCase):
    def test_selected_real_matrix(self):
        cases=os.environ.get('ASTERMIND_NATIVE_CASES','DS01 DS03 DS04 DS06-copy DS06-move').split()
        with tempfile.TemporaryDirectory(prefix='astermind-native-') as output:
            for case in cases:
                for mode in ('midlayer','roofline'):
                    with self.subTest(case=case,mode=mode):run(case,mode,'surrogate',Path(output))
