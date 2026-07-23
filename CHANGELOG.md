# Changelog

All notable changes to OrbitServe are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-07-22

### Added

- Established the `architecture`, `orbit`, `workload`, `serving`, `runtime`,
  and `common` module boundaries under `src/orbitserve/`.
- Added package-level public interfaces for cross-module contracts and
  services.
- Added canonical seven-file workload bundle loading, validation, and
  materialization.
- Added fixed-design evaluation, DSE, terrestrial baselines, WildChat
  conversion tools, and public examples.
- Added unit, integration, smoke, slow, and module-boundary test suites.
- Added a versioned structured SGP4 mean-element input contract with strict
  validation, deterministic normalization, provenance, and stable element-set
  identities.
- Added opt-in WGS-72 propagation through `python-sgp4==2.27`, producing
  deterministically ordered TEME position and velocity samples.
- Added pinned-data TEME-to-ITRS conversion and Sun ephemerides through
  `astropy==6.1.3`, with network access disabled and vendored IERS and
  leap-second data verified by fixed hashes and coverage checks.
- Added deterministic WGS-84 ground-access and inter-satellite visibility
  evaluation, conical-umbra sunlight classification, and half-open access,
  ISL, and sunlight windows.
- Added an opt-in SGP4 adapter that writes geometry-only Orbit Trace v2
  artifacts with explicit TEME and ITRS ephemerides, full model provenance,
  content hashes, and byte-for-byte reproducible serialization.
- Added frozen Orbit Trace v2 and SGP4 input contracts, an Orbit module handoff
  guide, and focused unit coverage for the complete SGP4 pipeline.

### Changed

- Standardized the project name as OrbitServe and the import package as
  `orbitserve`.
- Separated the core simulation engine from experiments, DSE, baselines, and
  dataset-specific tools.
- Renamed the former trace domain to `orbit` and limited it to constellation
  geometry, visibility, sunlight, and trace generation.
- Consolidated static per-satellite hardware configuration in
  `architecture/hardware.py`.
- Standardized workload input on one unversioned canonical bundle contract.
- Reorganized runtime code around contracts, scenarios, deployments,
  schedulers, engines, records, metrics, and energy state.
- Reorganized tests into `unit`, `integration`, `smoke`, and `slow` suites.
- Synchronized documentation and examples with the current source layout and
  public interfaces.
- Made `OrbitTraceManifest` and `OrbitTraceBundle` the canonical Orbit trace
  class names and migrated Orbit-internal references to them.
- Clarified the Orbit ownership boundary: new Orbit traces contain physical
  geometry and environmental facts, while `satellite_initial_soc`,
  `terminal_catalog`, and `active_graph_snapshots` are derived by Runtime from
  Orbit, Architecture, and runtime configuration.
- Strengthened stable-output requirements from semantic equivalence to
  byte-for-byte reproducibility for identical canonical inputs.

### Compatibility

- Retained `RuntimeTraceManifest`, `RuntimeTraceBundle`, and
  `SurrogateTraceBundle` as object-identical aliases.
- Kept the existing circular two-body model and legacy trace writer as the
  default path; the new SGP4 pipeline remains opt-in and does not yet change
  Runtime or DSE behavior.

### Removed

- Removed compatibility facades and the obsolete workload v1 contract.
- Removed ISL inventory modeling from Orbit.
- Removed workload-driven hardware sizing and obsolete runtime contention
  configuration.
- Removed confirmed unused scripts, duplicate tests, generated caches, and
  checked-in experiment outputs.

### Fixed

- Aligned the runtime energy-window mode with its consumer contract.
- Corrected KV migration routing to use a satellite identifier instead of a
  deployment identifier.

### Verification

- The initial project baseline passed 398 tests, including all 48 slow tests,
  covering fixed-design, trace-backed runtime, runtime optimization,
  terrestrial baseline, and DSE paths.
- Passed 49 of 49 Orbit unit tests, 6 of 6 module-boundary tests, 45 of 45
  fixed-design integration tests, and 22 of 22 smoke tests.
- Did not run slow tests for the opt-in SGP4 path because it is not yet consumed
  by a complete Runtime or DSE execution path.
