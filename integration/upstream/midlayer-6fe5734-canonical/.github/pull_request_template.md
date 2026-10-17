## Contract

- [ ] This change follows `AGENTS.md`, including owner, dependency, naming,
      fail-loud, and protected-data rules.
- [ ] Generated output, logs, caches, profiler databases, and local reports are
      not committed.

## Verification

- [ ] `python -m compileall -q` on the changed production/test owners
- [ ] `python scripts/run_tests.py` (all maintained tests, 48 workers)
- [ ] `git diff --check`
- [ ] If this changes Serving Scheduler, simulation orchestration, or Production Trace: affected fast tests and `pytest -q -n 7 --dist=worksteal -m scheduler_smoke`
