"""Save repeatable unit-test evidence without invoking the compute runtimes."""

from pathlib import Path
import io
import json
import time
import unittest
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/verification")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    identifiers = []
    def collect(items):
        for item in items:
            if isinstance(item, unittest.TestSuite): collect(item)
            else: identifiers.append(item.id())
    collect(suite)
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    original = json.loads((ROOT / "build/build_record.json").read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "tests.log").write_text(stream.getvalue(), encoding="utf-8")
    evidence = {"tests_run": result.testsRun, "failures": len(result.failures),
                "errors": len(result.errors), "successful": result.wasSuccessful(),
                "wall_seconds": elapsed, "tests": identifiers,
                "upstream_regression": original["upstream_tests"],
                "binary_sha256": original["binary_sha256"],
                "scope": "static paths, acquisition, sticky/OS selection, flow interruption, fluid sharing, event contract, packaging; no hardware calibration"}
    (output / "checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(stream.getvalue(), end="")
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == "__main__":
    main()
