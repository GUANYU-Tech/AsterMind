# Regression tests

The fixed-design entry is SimulationConfig -> Simulation. DSE, Analytical
performance tables, energy/SOC/DVFS execution and their exclusive tests are removed.
Independent physical geometry and workload tests remain.

```bash
python -m unittest discover -s tests -t . -v
ORBITSERVE_MIDLAYER_ROOT=/path/to/midlayer python -m unittest discover -s tests/external/native_cells -v
```

Install the actual native dependencies before claiming backend validation. A skip
is not a successful native test. FlowSim needs the compatible shared-runtime
extension, not an upstream extension with a different API. Real ISLSim tests are
explicitly run from `tests/external/islsim` with its external source configured.

| Behavior | Coverage |
| --- | --- |
| Native CLI, schema defaults, rejected removed fields | `smoke/test_native_entry.py` |
| Shared host clock, byte conservation, directional RF, fault/retry/acquisition | `integration/fixed_design/test_network_acceptance.py`, `test_network_recovery.py`, `test_boundary_control.py`, `test_time_varying_transport.py`, `test_radio_sharing.py` |
| Native Cell callbacks, input before collective, migration rollback/horizon | `integration/fixed_design/test_cell_shared_runtime.py`, `unit/runtime/test_cell_migration.py` |
| Actual Mid-Layer/vLLM TP/PP, KV move/prefix reuse, communication scopes | `external/native_cells/` |
| Physical dataset hashes, reuse, fault causality, serial/parallel exports | `integration/fixed_design/test_physical_dataset.py`, `test_physical_precompute.py`, `test_parallel_physical_exports.py` |
| Independent workload behavior | `unit/workload/`, `smoke/test_main_composite_chat_bundle.py` |
| Installed package and packaged resources | `scripts/check_distribution.py` |

`tests/network_case.py` is a transfer-only fixture for the shared event engine.
It executes physical request/reply bytes, without synthesizing LLM computation,
energy or KV state. Real LLM scheduling and KV are validated by the Native suite.
Legacy test documents are not accepted as SimulationConfig inputs.

The V1 cleanup, follow-up removal and explicit test exclusions are recorded in [V1 validation](V1_VALIDATION.md).

## Repeating the layout regression

With the pinned native dependencies available:

```bash
python tests/scripts/run_retained_tests.py
ORBITSERVE_NATIVE_REPORT_DIR=/absolute/path/to/reports python tests/scripts/run_migration_native.py
python tests/scripts/compare_migration.py /absolute/path/to/baseline /absolute/path/to/reports
python tests/scripts/check_distribution.py --work-dir /absolute/path/to/new-package-check
python -m viz.export /absolute/path/to/reports/*.json --out /absolute/path/to/replay
```

The Native matrix saves the exact physical and memory-policy inputs before test
cleanup. `ASTER_TEST_ROOT` can point to a frozen baseline checkout; select its
installed package with `PYTHONPATH` when testing the old layout. Use the same
dependencies and report filenames on both sides.

Serve an exported replay over HTTP and run the browser check with Playwright
available in the Node module search path:

```bash
node tests/browser/native_replay.cjs http://127.0.0.1:8767/replay/ /absolute/path/to/browser-evidence
```

The optional final argument `baseline` expects the documented old scene-switch
bug; omit it for the migrated frontend. A multi-case export tests scene switching.
The installation check also produces a real ns3 report and a `replay/` folder
for the same browser test. It requires the Native dependencies; successful imports
alone do not count as installed execution validation.
