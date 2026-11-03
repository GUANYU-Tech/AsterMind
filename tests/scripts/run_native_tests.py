"""Run every suite with native backends and reject skipped validation."""

from pathlib import Path
import sys
import unittest


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    suite = unittest.defaultTestLoader.discover(
        str(root / "tests"), top_level_dir=str(root)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print("Full native validation must not skip tests:", result.skipped)
    return int(not result.wasSuccessful() or bool(result.skipped))


if __name__ == "__main__":
    sys.exit(main())
