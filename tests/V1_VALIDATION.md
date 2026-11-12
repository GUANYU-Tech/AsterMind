# AsterMind V1.0 cleanup validation

Base: `feature/runtime-dynamic-links` at `cf96341e52ee2bdbb249ce54ffd3aef97aa9250e`.
Working branch: `version/AsterMind-V1.0`. Validation host: SeeSpace1.

## Scope

- The experiment CLI invokes the existing Native wrapper directly. SimulationConfig
  is the run configuration; numeric-v1 and native_cells.v1 communication defaults
  are unchanged. Removed orbit/link_parameters fields fail validation even when empty.
- Deleted DSE, old Analytical performance generation/interpolation, request admission,
  energy/SOC/DVFS execution, and their exclusive configuration/metrics/test modules.
- Deleted the old request API and its CellRuntime/CellDeployment/AsterMindRuntime
  adapters after explicit scope confirmation. Shared clock, transport, topology,
  fault/retry, Native Cell execution and KV migration remain in their original places.
- Independent architecture sizing/cost, workload and ground-baseline tools remain.
  The historical JSON output utility remains independent of the Native CLI; its
  DSE-specific output branches were removed. No dependency versions were changed.

## Environment

Python 3.12.16; pinned CPU Scheduler lock; official vLLM 0.31.0 Python source identity
verified by the existing Native audit (2771 files, manifest SHA-256
`1ddbe0ebcca48d6ed773a0ab92d55bbaaeffb883ddad2e97ac08d6fec5d446d5`).
Mid-Layer commit `6fe573479c37a931744a6442e7a4b3503eec02e6`;
ns3 3.44.post0; SimGrid 4.1 built with the existing pybind11 2.13.6 constraint.
SimGrid required explicit Python 3.12 CMake include/library paths and a local
LD_LIBRARY_PATH. Build settings, external source and binaries were not committed.
This CPU environment is not a full GPU/HTTP vLLM dependency validation.

## Results

| Check | Result |
| --- | --- |
| Pre-change real ns-3 Native baseline | 2 requests complete, 2717 kernels, audit passes |
| Batch 1 entry/config smoke | 8 tests; Native summary CLI matches baseline requests |
| Batch 2 DSE removal smoke | 20 tests; Native full CLI matches baseline requests |
| Batch 3 shared-event trim smoke | 15 tests; Native ns-3 requests still match baseline |
| Retained non-FlowSim unit/integration/smoke suite | 551 tests pass, no skips in selected set |
| Real Native integrations | 9 scenarios pass, described below |
| Review follow-up | 10 tests pass, including real ns-3/SimGrid transfers with physical regeneration forbidden |
| Installed sdist/wheel | Build succeeds; 108 packaged modules import; resource hashes and portable CLI pass |
| Static checks | `ruff check --select F src simulator integration network_sim tests` and `git diff --check` pass |

The 551-test run initially used a stdin launcher: 544 passed and 7 multiprocessing
spawn cases could not reopen `<stdin>`. All 7 were rerun with `python -m unittest`
and passed. This was a launcher issue, not an implementation correction.
The full broad Ruff style command still reports repository findings; this cleanup
did not bulk-format unrelated code. The Python F-rule check is clean.

The 9 actual Native scenarios cover: SimGrid direct/satellite/cell communication
scopes; SimGrid KV migration and prefix reuse; ns-3 TP2 KV move and prefix reuse;
SimGrid cross-satellite TP2/PP2 recovery, allocator rollback, heterogeneous Cells/HBM,
and MoE TP2/EP2. The final four reuse the existing Native assertions with explicit
SimGrid/direct-communication settings. These execute real Mid-Layer/vLLM, not mocked
compute or storage. The ns-3 migration test now explicitly selects direct communication,
so it does not accidentally inherit the string-schema ProtoBridge default.

At the user's request, 20 FlowSim-dependent tests were excluded. Neither skips nor
an incompatible upstream FlowSim extension were counted as validation. The external
ISLSim suite was not run. FlowSim production implementation remains unchanged.

## Coverage retained during test migration

`tests/network_case.py` drives actual byte transfers through the shared event engine;
it does not synthesize LLM computation, KV or energy. Directional RF, capacity changes,
byte conservation, route recovery and acquisition timing assertions remain.
Cell migration placement/cleanup tests now exercise Simulation callbacks directly.
Physical dataset tests retain identity/hash rejection, non-sample-time equivalence,
fork isolation, causal random faults and execution without physical recomputation.

Local test environment and logs are under `/data1/guolidong/.orbitserve-tools/`:
`test-env.sh`, `full-retained.log`, `spawn-rerun.log`, `native-final.log`,
`native-ns3-final.log`, `native-simgrid-extra.log`, `review-followup-tests.log`,
`distribution-check.log`. Initial unsuccessful launches remain in the logs; the
successful reruns above provide the final results.

Static review found no blocking production regression. The obsolete README entry
and the weakened dataset execution assertion identified during review were fixed.
