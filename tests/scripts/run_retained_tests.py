"""Run all current tests: missing dependencies and skips are failures."""
from pathlib import Path
import os
import sys
import unittest
ROOT=Path(__file__).resolve().parents[2]
def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    loader=unittest.TestLoader();suite=loader.discover('tests',top_level_dir='.')
    assert not loader.errors,loader.errors
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:print('UNEXPECTED SKIPS',result.skipped)
    return 0 if result.wasSuccessful() and not result.skipped else 1
if __name__=='__main__':raise SystemExit(main())
