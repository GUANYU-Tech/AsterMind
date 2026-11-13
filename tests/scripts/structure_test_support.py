"""Small shared helpers for structural regression observations."""
import json
import unittest


def run_suites(names):
    """Run existing assertions and retain the exercised test/subtest identities."""
    class Result(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.observations = []

        def addSuccess(self, test):
            super().addSuccess(test)
            self.observations.append(dict(test=test.id(), status='passed'))

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            self.observations.append(dict(test=subtest.id(), status='passed' if err is None else 'failed'))

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromNames(names)
    assert not loader.errors, loader.errors
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(suite)
    if not result.wasSuccessful() or result.skipped or result.expectedFailures:
        raise AssertionError('Regression suite failed or did not execute every assertion')
    return dict(tests=result.testsRun, assertions=result.observations)


def write_result(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
