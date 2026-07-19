# Changelog

All notable changes to OrbitServe are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-07-19

### Added

- Established the `architecture`, `orbit`, `workload`, `serving`, `runtime`, and `common` module boundaries under `src/orbitserve/`.
- Added package-level public interfaces for cross-module contracts and services.
- Added frozen cross-module contracts: `SatelliteArchitectureProfile`, `SatelliteEnergyProfile`, `OrbitTraceManifest`, `OrbitTraceBundle`, `MaterializedWorkload`, and `ServingCapacityProfile`.
- Added canonical seven-file workload bundle loading, validation, and materialization.
- Added fixed-design evaluation, DSE, terrestrial baselines, WildChat conversion tools, and public examples.
- Added unit, integration, smoke, slow, and module-boundary test suites.

### Changed

- Standardized the project name as OrbitServe and the import package as `orbitserve`.
- Separated the core simulation engine from experiments, DSE, baselines, and dataset-specific tools.
- Renamed the former trace domain to `orbit` and limited it to constellation geometry, visibility, sunlight, and trace generation.
- Consolidated static per-satellite hardware configuration in `architecture/hardware.py`.
- Made battery capacity and solar-array power explicit static Architecture inputs while retaining cost-model feasibility checks.
- Standardized workload input on one unversioned canonical bundle contract.
- Renamed trace, workload, and serving outputs to domain-owned contract names and removed the former Runtime-prefixed and versioned names.
- Decoupled Serving from Workload token distributions and constellation-wide satellite budgets; Serving now reports one deployment template and Runtime instantiates deployment groups.
- Restricted the public Orbit bundle to geometry, visibility, propagation, and illumination information.
- Required Runtime to consume canonical workload request fields without legacy aliases.
- Reorganized runtime code around contracts, scenarios, deployments, schedulers, engines, records, metrics, and energy state.
- Reorganized tests into `unit`, `integration`, `smoke`, and `slow` suites.
- Synchronized documentation and examples with the current source layout and public interfaces.

### Removed

- Removed compatibility facades and the obsolete workload v1 contract.
- Removed `ResolvedHardware`, `RuntimeTraceManifest`, `RuntimeTraceBundle`, `WorkloadGenerationResult`, `ServingThroughputEstimate`, and `estimate_serving_throughput` from the public API.
- Removed ISL inventory modeling from Orbit.
- Removed workload-driven hardware sizing and obsolete runtime contention configuration.
- Removed confirmed unused scripts, duplicate tests, generated caches, and checked-in experiment outputs.

### Fixed

- Aligned the runtime energy-window mode with its consumer contract.
- Corrected KV migration routing to use a satellite identifier instead of a deployment identifier.

### Verification

- Passed 330 non-slow tests: 179 unit, 129 integration, and 22 smoke tests.
- Verified module contracts, fixed-design evaluation, trace-backed runtime, terrestrial baselines, and DSE without running the long-duration suite.
- Kept `results/` and `scripts/` empty in the source repository.
