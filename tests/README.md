# Regression tests

The fixed-design entry is SimulationConfig -> Simulation. DSE, Analytical
performance tables, energy/SOC/DVFS execution and their exclusive tests are removed.
Independent physical geometry, cost, workload and terrestrial baseline tests remain.

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
| Independent cost/workload/ground behavior | `unit/architecture/`, `unit/workload/`, `integration/ground_baseline/` |
| Installed package and packaged resources | `scripts/check_distribution.py` |

`tests/network_case.py` is a transfer-only fixture for the shared event engine.
It executes physical request/reply bytes, without synthesizing LLM computation,
energy or KV state. Real LLM scheduling and KV are validated by the Native suite.
Legacy test documents are not accepted as SimulationConfig inputs.

The V1 cleanup run and its explicit exclusions are recorded in [V1 validation](V1_VALIDATION.md).
