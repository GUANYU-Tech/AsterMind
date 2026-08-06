# Changelog

All notable changes to OrbitServe are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added a deterministic SGP4 pipeline from validated mean elements through
  WGS-72 TEME propagation, pinned-data ITRS conversion, WGS-84 geometry, and
  canonical Orbit Trace generation via `generate_orbit_trace`.
- Added `RuntimeSimulationSpec` as the immutable aggregate input for one
  deterministic Runtime simulation.
- Added per-satellite capacity-driven LRU management for KV cache state.

### Changed

- Made `OrbitTraceManifest` and `OrbitTraceBundle` the canonical Orbit trace
  class names, with geometry-only records and metadata as their complete public
  output contract.
- Kept the SGP4 implementation as flat `sgp4_*` modules under
  `orbitserve.orbit` while keeping cross-module use on the public
  `orbitserve.orbit` entry point.
- Removed `v2` from the canonical Orbit Trace schema and public API names;
  `orbitserve.orbit_trace`, `write_orbit_trace`, and `load_orbit_trace` now
  define the single current output format.
- Declared the pinned SGP4, Astropy, Earth-orientation dependencies and vendored
  IERS package data in `pyproject.toml`.
- Reorganized Runtime into `setup`, `engine`, `state`, and `metrics`, with
  public contracts and results kept at the package root.
- Made `RuntimeStrategySpec` part of `RuntimeSimulationSpec` and consolidated
  execution behind `run_runtime_simulation(simulation_spec)`.
- Moved deterministic candidate selection into the Runtime engine while making
  DSE solely responsible for strategy search.
- Applied one colocated prefill/decode `ServingCapacityProfile` uniformly to
  all satellites.
- Standardized DVFS as a strategy-owned capacity scale in the range `(0, 1]`.
- Simplified request outcomes to completed and uncompleted, and limited
  TTFT/E2E statistics to completed requests.
- Made Runtime consume the static satellite energy profile provided by
  Architecture directly.
- Made Runtime derive per-satellite KV capacity from Architecture storage and
  Serving's model-specific KV bytes per token.
- Renamed the Runtime orbit input to `environment_state` and the KV strategy
  limit to `kv_migration_fraction`.
- Defined KV movement as source-retaining copy with deterministic single-source
  selection and token-level LRU capacity accounting.

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
- Removed Runtime HBM/page-accounting configuration, duplicated session KV
  locations, and unused context-version state.

### Fixed

- Fixed trace-backed route reservation and isolated mutable KV cache state by
  satellite.

### Verification

- Passed all 39 Orbit unit tests.
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
