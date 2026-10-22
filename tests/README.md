# Regression coverage

Run all four suites before integration, including `tests/slow/`:

```bash
python -m ruff check src tests
python -m unittest discover -s tests -t . -v
```

Install `.[simgrid,ns3]` on a supported platform for the native backend checks.
Skipped native cases do not validate those backends. The GitHub Actions workflow
installs both native dependencies, runs the full suite on Python 3.12, and fails
if any case is skipped.

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
