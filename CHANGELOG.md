# Changelog

All notable changes to OrbitServe are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a first-order L2 Serving formation model with member count, ring/switch/
  fully-connected topology, link bandwidth, and one-hop propagation delay.
- Added canonical logical compute-node views for Orbit Trace and Runtime state.
- Added frozen Serving profiles for Llama 3.3 70B, DeepSeek V3, and Kimi K3,
  with batch-one prompt-length tables for prefill and decode.
- Added cumulative-table lookup in Runtime and same-session KV copy decisions
  that compare transfer-plus-prefill time with local prefill time.
- Added request-level TPOT and aggregate TTFT/TPOT/E2E distributions with
  average, p50, p95, and p99 statistics.

- Added a deterministic SGP4 pipeline from validated mean elements through
  WGS-72 TEME propagation, pinned-data ITRS conversion, WGS-84 geometry, and
  canonical Orbit Trace generation via `generate_orbit_trace`.
- Added an isolated STK 11.6 validation lane with frozen state-vector and
  geometry-event corpora, direct-element SGP4 mapping, deterministic evidence,
  and separate TEME, Fixed, SGL, ISL, lighting, and AER comparisons.
- Added `RuntimeSimulationSpec` as the immutable aggregate input for one
  deterministic Runtime simulation.
- Added per-satellite capacity-driven LRU management for KV cache state.

### Changed

- Made `OrbitTraceManifest` and `OrbitTraceBundle` the canonical Orbit trace
  class names, with geometry-only records and metadata as their complete public
  output contract.
- Grouped SGP4 implementation details by responsibility under
  `orbitserve.orbit.sgp4` while keeping cross-module use on the public
  `orbitserve.orbit` entry point.
- Removed `v2` from the canonical Orbit Trace schema and public API names;
  `orbitserve.orbit_trace`, `write_orbit_trace`, and `load_orbit_trace` now
  define the single current output format.
- Declared the pinned SGP4, Astropy, Earth-orientation dependencies and vendored
  IERS package data in `pyproject.toml`.
- Extended the STK validation contract with Facility-side AER sampling,
  half-open event-window normalization, explicit Umbra/Penumbra handling, and
  fail-closed temporary-scenario cleanup.
- Reorganized Runtime into `setup`, `engine`, `state`, and `metrics`, with
  public contracts and results kept at the package root.
- Made `RuntimeStrategySpec` part of `RuntimeSimulationSpec` and consolidated
  execution behind `run_runtime_simulation(simulation_spec)`.
- Moved deterministic candidate selection into the Runtime engine while making
  DSE solely responsible for strategy search.
- Applied one colocated prefill/decode `ServingCapacityProfile` uniformly to
  all satellites.
- Replaced workload-aggregated Serving rate scalars with one-dimensional
  prompt-length tables; Runtime now derives each request's service time from
  those tables without a running-request throughput multiplier.
- Standardized DVFS as a strategy-owned capacity scale in the range `(0, 1]`.
- Simplified request outcomes to completed and uncompleted, and limited
  TTFT/TPOT/E2E statistics to completed requests.
- Replaced Runtime `workload_metrics` with separate `inference_metrics` and
  `request_summary` outputs; inference efficiency now contains only TTFT,
  TPOT, and E2E latency distributions.
- Made Runtime consume the static satellite energy profile provided by
  Architecture directly.
- Made Runtime derive per-satellite KV capacity from Architecture storage and
  Serving's model-specific KV bytes per token.
- Renamed the Runtime orbit input to `environment_state`; KV transfer is now a
  boolean `kv_cache_copy_enabled` strategy.
- Defined KV movement as source-retaining, same-session copy from the source
  with minimum transfer-plus-prefill time, only when it beats local prefill;
  reuse is capped by `context_tokens_before_turn`.
- Aggregated cost as logical compute-node count times L2 formation member count
  times Architecture's closed single-satellite cost, without a second laser-
  terminal or topology surcharge.

### Removed

- Removed the legacy circular-orbit constellation configuration, simulator,
  surrogate trace implementation, trace builders, and SGL compatibility
  helpers from Orbit.
- Removed the `RuntimeTraceManifest`, `RuntimeTraceBundle`, and
  `SurrogateTraceBundle` aliases and the remaining legacy circular-orbit public
  interfaces.
- Removed compute-only execution, disaggregated prefill/decode deployment,
  deployment catalogs, and deployment-group placement logic.
- Removed planned-versus-observed energy alignment, duplicate queue state,
  redundant decision contracts, and the standalone Runtime strategy package.
- Removed SLO classification and SLO pass/fail accounting from Runtime results.
- Removed the remaining request-level `slo` section and top-level
  `slo_metrics` placeholder from Runtime output.
- Removed Runtime HBM/page-accounting configuration, duplicated session KV
  locations, and unused context-version state.
- Removed Serving batch-size, prefill/decode placement, workload-distribution,
  and duplicated parallel satellite-count inputs.

### Fixed

- Fixed trace-backed route reservation and isolated mutable KV cache state by
  satellite.

### Verification

- Passed all 77 Orbit unit tests.
- Reproduced the complete STK geometry corpus byte-for-byte across two runs;
  SGL and ISL sample-state agreement was 100%, while the two satellite Umbra
  agreement ratios were 99.72% and 99.86%.
- Kept STK results diagnostic-only; formal acceptance thresholds remain
  intentionally unfrozen pending review.
- Confirmed that remaining module-boundary, fixed-design integration, and smoke
  failures are limited to Runtime, Experiments, and DSE consumers that still
  import the removed Orbit interfaces and require owner-led migration.
- Slow tests were not run because the complete Runtime and DSE SGP4 paths have
  not yet been migrated.

## [0.0.1] - 2026-07-19

### Added

- Established the `architecture`, `orbit`, `workload`, `serving`, `runtime`, and `common` module boundaries under `src/orbitserve/`.
- Added package-level public interfaces for cross-module contracts and services.
- Added canonical seven-file workload bundle loading, validation, and materialization.
- Added fixed-design evaluation, DSE, terrestrial baselines, WildChat conversion tools, and public examples.
- Added unit, integration, smoke, slow, and module-boundary test suites.

### Changed

- Standardized the project name as OrbitServe and the import package as `orbitserve`.
- Separated the core simulation engine from experiments, DSE, baselines, and dataset-specific tools.
- Renamed the former trace domain to `orbit` and limited it to constellation geometry, visibility, sunlight, and trace generation.
- Consolidated static per-satellite hardware configuration in `architecture/hardware.py`.
- Standardized workload input on one unversioned canonical bundle contract.
- Reorganized runtime code around contracts, scenarios, deployments, schedulers, engines, records, metrics, and energy state.
- Reorganized tests into `unit`, `integration`, `smoke`, and `slow` suites.
- Synchronized documentation and examples with the current source layout and public interfaces.

### Removed

- Removed compatibility facades and the obsolete workload v1 contract.
- Removed ISL inventory modeling from Orbit.
- Removed workload-driven hardware sizing and obsolete runtime contention configuration.
- Removed confirmed unused scripts, duplicate tests, generated caches, and checked-in experiment outputs.

### Fixed

- Aligned the runtime energy-window mode with its consumer contract.
- Corrected KV migration routing to use a satellite identifier instead of a deployment identifier.

### Verification

- Passed 398 tests, including all 48 slow tests.
- Verified fixed-design, trace-backed runtime, runtime optimization, terrestrial baseline, and DSE paths.
- Kept `results/` and `scripts/` empty in the source repository.
