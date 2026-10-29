# Local ISLSim integration

These opt-in tests invoke the real external ISLSim checkout. They are separate
from the four repository CI suites, which validate configuration generation
without accessing the ISLSim repository. This directory intentionally has no
`__init__.py`, so the default `unittest discover -s tests -t .` does not collect it.

Run from the repository root with ISLSim explicitly available:

```bash
ORBITSERVE_ISLSIM_ROOT=/path/to/ISLSim python -m unittest discover -s tests/external/islsim -v
```

The real-model tests have no analytical substitute. Missing ISLSim fails the
explicit local command. SimGrid/ns3 rows require their native optional packages.
