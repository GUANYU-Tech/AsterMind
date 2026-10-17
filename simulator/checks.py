"""Run local integration contracts and save a machine-readable test summary."""
import json
from pathlib import Path
import sys
import unittest


if __name__ == "__main__":
    root = Path(__file__).parent
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/"tests")))
    output = root/"runs/unit_tests.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(passed=result.wasSuccessful(), tests=result.testsRun,
        skipped=[(str(test), reason) for test, reason in result.skipped],
        failures=[(str(test), trace) for test, trace in result.failures],
        errors=[(str(test), trace) for test, trace in result.errors]), indent=2), encoding="utf-8")
    sys.exit(0 if result.wasSuccessful() else 1)
