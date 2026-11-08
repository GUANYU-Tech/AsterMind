# Regression coverage

Run all four suites before integration, including `tests/slow/`:

```bash
python -m ruff check src tests
python -m unittest discover -s tests -t . -v
```

Install `.[simgrid,ns3]` on a supported platform for the native backend checks.
Skipped native cases do not validate those backends. On Ubuntu 24.04 with Python
3.12, install the C++ compiler, CMake, Boost and Python/pybind11 development headers
before installing both native backends. Use pip 26.2.2 and
`tests/requirements/native-build.txt` to select pybind11 2.13.6 in the isolated
SimGrid build; see the root README for the installation commands.
The package pins NumPy 2.2.6 with Astropy 6.1.3. Before lint and regression tests,
check dependency consistency with `python -m pip check` and import Astropy's
coordinate and unit modules.
The causal workload contract checks use `unittest.TestCase` so they are collected
by the same full-suite command and do not depend on pytest.
Run `python tests/scripts/run_native_tests.py` for full native validation. Its
guarded file entry point supports the orbit pipeline's `spawn` workers and
rejects skipped tests.
The ns-3 worker compiles its adapter with C++20 against the wheel's libraries;
local runs require a compiler on `PATH` or selected through `CXX`.

Run `python tests/scripts/check_distribution.py` with `build==1.6.1`: it builds
from current Git inputs in a temporary directory, then builds the wheel from the
sdist. It installs into a separate target, imports every packaged module and
checks reference hashes and the portable configuration CLI with ISLSim absent.
Build isolation requires setuptools 84.1.0 or newer; native build constraints
retain pybind11 2.13.6. The caller's Python environment is not overwritten.

Reliability regressions cover overlapping artifact writers, 5000-node dependency
graphs, 1500-request failure propagation, native protocol retirement and startup
cleanup, invalid model counts, tiny zero-capacity transfers, bounded compressed
blocks and isolation of cached physical metadata.

The feature branch follows develop's directory layout and does not provide a
GitHub Actions workflow. These commands remain available for local validation
and for the repository's configured CI; local results do not replace required
remote checks or maintainer review.

ISLSim is a separate local integration boundary. The default suites generate and
validate native JSON parameters without importing ISLSim, and their simulation
fixtures explicitly choose self-contained capacity models. The suites do not clone the
ISLSim private repository or need its credentials. `tests/external/islsim/` keeps
the real-model integration checks outside default discovery; run them locally
with `ORBITSERVE_ISLSIM_ROOT=/path/to/ISLSim python -m unittest discover -s tests/external/islsim -v`.
Those tests exercise the real model, including native SimGrid/ns3 Runtime paths.

## Preserved behavior during the physical-network migration

| Behavior | Current checks |
| --- | --- |
| KV source selection, same-session isolation, copy versus recompute, source retention | `unit/runtime/test_kv_copy_selection.py`, `unit/runtime/test_runtime_engine.py`, `integration/fixed_design/test_kv_backends.py` |
| Capacity-based LRU, local eviction, failed admission, runtime recomputation after eviction | `unit/runtime/test_kv_capacity_lru.py`, `unit/runtime/test_candidate_safety.py` |
| Concurrent requests, queue limits, no future KV visibility, dependent turns | `unit/runtime/test_runtime_event_order.py`, `integration/fixed_design/test_serving_runtime_concurrency.py`, `slow/test_serving_concurrency_burst.py` |
| Hardware qualification, healthy relay use, low-energy relays and non-owner cell members, isolated candidate previews | `unit/runtime/test_candidate_safety.py` |
| SOC, charge/discharge efficiency, energy history and invalid inputs | `unit/runtime/test_energy_contract_corner_cases.py` |
| Horizon and missing ingress, actual per-hop paths, chronological event records | `integration/fixed_design/test_network_request_events.py`, `slow/test_physical_runtime_burst.py` |
| Geometry, trace hashes, boundary coverage, reusable physical datasets | `unit/orbit/`, `integration/fixed_design/test_physical_trace_contract.py`, `test_physical_trace_io.py`, `test_physical_dataset.py` |
| DSE/fixed-design isolation and design-by-strategy execution | `integration/dse/`, `slow/test_physical_runtime_matrix.py` |
| Terrestrial cache/version readiness, routing and serving/cost evaluation | `integration/ground_baseline/` |

## Deliberate interface changes

Runtime routing now persists selected paths across transfers. The routing policy
selects the initial path; traffic changes update timing without reselecting it.
At physical time-slice, acquisition, or fault boundaries, only reused paths whose
directed-link attributes changed are invalidated. Unrelated paths remain cached,
and newly available alternatives do not preempt an unchanged path. A cached path
that cannot carry a payload falls back to a feasibility search. Cache entries
are bounded and isolated between preview forks. `NetworkState.transfer(...,
recompute=True)` performs the original fresh search for comparison;
`route_cache_info()` reports searches, hits, and invalidations. The regression
checks are in `unit/runtime/test_persistent_routing.py`.

The physical SGP4 path replaces the fixed-shell Arc and surrogate/circular trace
runtime inputs. Legacy fixtures using `constellation`, `scheduler`, `simulation`,
`communication_ingress`, and the old `RuntimeScenario` are not compatibility
fixtures for the current LEO API. Current manifests use `physical_network` and
`RuntimeSimulationSpec`; terrestrial scenarios have their own baseline contracts.

Physical DSE now evaluates current runtime strategies directly. Geometry screening
reports an upper bound on access, without synthesizing SLO or energy results.
The former static-SLO proxy, closed-loop search controller, ground-config matrix
mixing, custom objective, optional-cost and preliminary-metric switches are not
retained in this path. Unsupported controller options are rejected explicitly.
Static-screening configuration produces diagnostics; it does not perform the old
top-k runtime pruning. These are interface/behavior changes, not evidence that
every original develop test passes unchanged.

The pre-push audit also ran the original develop source and test snapshot in
isolation. That snapshot has inconsistent legacy imports and fixtures, so its
existing failures must be distinguished from regressions introduced by this
migration. The checks above preserve specific business invariants on current
inputs; test counts alone do not prove compatibility.
