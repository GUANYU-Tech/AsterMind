# Changelog

## AsterMind v1.0 · Windows preview 9

- Simultaneous arrivals in Compute Formation simulation submit every synthetic request at simulation time zero.
- The Windows app can launch a Compute Formation task from the workspace selector.
- Full Scenario simulation waits for vLLM schedulers to finish starting, including when multiple Cells take longer than 60 seconds to initialize.
- [Download the Windows 11 x64 installer](https://github.com/GUANYU-Tech/AsterMind/releases/download/v1.0.0-preview.9/AsterMind-v1.0-Windows-x64-Setup.exe). The installer includes the WSL 2 simulation engine.
- SHA-256: `fd5754123bfbd480c185b2cb307b8280621ecd01e71ec39d7ce6416a6c66921d`.

## AsterMind v1.0 · Windows preview 7

- Windows 11 x64 desktop installer for the AsterMind simulation platform. The simulation engine uses WSL 2.
- Package version `1.0.0-preview.7`, built from `8f5d366f4cc4820f28e535225f27fbe81d523cb1`. This package predates subsequent changes on `main`.
- Release asset: `AsterMind-v1.0-Windows-x64-Setup.exe` (1,906,600,651 bytes).
- SHA-256: `4271241e89dc3c6734ab9e45531cf97abbb9db26b535d5b3efb936173eca072a`.
- [Release and download](https://github.com/GUANYU-Tech/AsterMind/releases/tag/v1.0.0-preview.7)

- Prefilled versioned Starlink single-carrier RF and optical budgets with separate
  user/feeder uplink/downlink profiles, clearly labelled inference and assumptions,
  and separate V3 160-Gbps specification / >200-Gbps report editions. Added tests
  that check actual equal-byte transfer times and satellite capacity sharing.

All notable changes to OrbitServe are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Reorganize the package as root `astermind/builder` and `astermind/engine`;
  merge Runtime into Engine and place source workload adapters, examples and
  replay UI in root `workloads/`, `examples/` and `viz/`.
  Preserve Native simulation behavior and serialized identities. Add strict
  baseline comparisons, real browser checks and installed Native execution.
- Fix replay scene switching when the previously selected call index exceeds
  the new scene's call count.


### Added

- Default new satellite-link JSON profiles and new optical hardware presets to
  the external, unmodified ISLSim model; retain explicit legacy model selection.
- Preserve native configuration, source hashes, SNR, modulation and feasibility
  in physical inputs and snapshots. Infeasible links become unavailable to routing.
- Add model selection to the local physical editor and require regenerated
  physical datasets when switching models. Use the existing local ISLSim checkout.
- Layer scenario ISLSim parameters over generation and native JSON defaults;
  pass Starlink optical scenario ceilings into the native terminal configuration.
  Offer automatic, fixed and custom modulation pools in the local editor.

- Independent node/link random fault settings with exponential UP/DOWN and
  Weibull healthy-period models, selectable outage duration distributions and
  explicit right-censoring of repairs beyond the simulation horizon.
- Boundary-only fault inputs with same-boundary recovery, reproducible fixed-ratio
  and per-step random fault models, and Runtime fault controls independent of
  reusable physical dataset identity. Explicit continuous mode preserves legacy
  experiment timing.
- An opt-in boundary controller and authoritative topology commands for persistent
  disconnections, layered manual faults and physically validated new satellite
  links. Native transport extension preserves active transfer progress.
- Regression coverage for boundary timing, multi-cause outages, atomic connection
  rejection, dataset reuse and in-flight byte conservation across four backends.

### Fixed

- Restore develop's three research notes under `docs/research/`; place native
  build requirements and validation scripts under `tests/` and remove the
  feature branch's root `.github/` and `constraints/` directories.
- Isolate overlapping artifact writers and preserve the published file on a
  producer failure. Use the same publication primitive for dataset manifests.
- Validate Runtime requests before allocating backends; use iterative dependency
  checks and failure propagation for long causal chains. Reject fractional model
  counts and nonfinite Serving coefficients instead of silently changing inputs.
- Preserve tiny fluid transfers until actual service completes them. Reject
  duplicate native transfers, invalid clock changes and oversized payloads;
  retire workers after malformed RPC/timeout and clean up startup failures.
- Bound physical-block decompression, validate binary indices and times, isolate
  returned metadata and serialize concurrent cache population/eviction.
- Remove the obsolete ground evaluator and unused sequential implementation;
  check every source module and installed distribution module for import errors.
- Require a patched setuptools build backend and verify sdist-to-wheel packaging,
  reference hashes and the portable CLI outside the source checkout.

- Generate and validate portable ISLSim JSON inputs without loading its external
  implementation. Bind source hashes only for actual local physical calculation;
  keep real-model integration in an explicit local test suite.
- Compile the ns3 adapter against the installed native libraries with a C++20
  compiler to avoid cppyy initialization and JIT traps on clean runners.
  Preserve native worker diagnostic heads,
  tails, exit status and backend-specific exception names.
- Prevented cross-track rows of synthetic compute formations from converging
  at orbit-plane intersections by aligning relative eccentricity/inclination
  vectors. Added an early sampled SGP4 clearance check before link budgets,
  with cell, member pair and closest-approach time in failure diagnostics.
- Replaced per-step random repair draws with explicit outage duration in time
  steps. Random injection windows and fixed-ratio outage intervals accept step
  indices, with complete repair-window validation and duration audit fields.
- Avoid repeated SQLite opens and block decoding during Runtime route forecasts
  by caching bounded compressed data and normalizing reads to physical boundaries.
- Screen impossible adaptive access pairs across the complete horizon and skip
  blocked physical budgets. Compute and compress attribute chunks in bounded
  worker processes with deterministic, atomic SQLite publication and progress.
  Regenerate datasets made with the previous physics version.
- Preserve explicit hardware compute exclusions when combining Orbit catalog
  metadata with network roles.
- Check predicted energy for every physical hop endpoint and all compute-cell
  members before dispatch, without mutating live candidate state.
- Remove accidentally tracked Python caches and local research documents from
  distributed files; retain local copies outside the tracked tree.

### Tests

- Preserve KV copy/recompute, session isolation, bounded context reuse and causal
  KV visibility across queue, fluid, SimGrid and ns-3 in the regular suite.
- Add physical candidate qualification, relay/cell energy and Runtime LRU
  regressions, a legacy migration coverage map, and full native-backend CI.

### Changed

- Add physical `handover_policy` choices for retaining valid dynamic contacts or
  reranking them by quality, with explicit cross-layer ranking and setup delays.
- Split physical preparation from Runtime with reusable, versioned and hashed
  datasets containing complete position/link data and compressed directional
  budgets. Runtime loads saved physics across transport backends; the local
  console requires completed physical data before entering execution.

- Enforced generation-specific optical port counts and explicit endpoint rate
  ceilings; distinguished RF antennas, beams and scenario peer concurrency.
- Corrected path audit distances to their serialization-completion sample and
  added independent per-sample physics and real-backend update regressions.

- Released faulted terminals causally and reacquired replacement links at the next
  Orbit sample; preserved unaffected contacts and discarded only observed bytes.
- Included observed competition in fluid/ns-3 routing forecasts, with native ns-3
  packet overhead and logical payload preservation under wire padding.
- Added versioned Starlink V2 Mini/V3 radio editions, directional RF link budgets
  and satellite aggregate capacity pools across queue/fluid/SimGrid/ns-3.
- Added a canonical-Workload three-route/four-backend networking acceptance matrix
  and cross-GSL aggregate/full-duplex conservation regressions.

- Moved terrestrial baseline metric and cost evaluation into the ground baseline
  package; experiment runtime dispatch now only selects the appropriate evaluator.

### Added

- Added explicit compute-only, communication-only and mixed network scenes,
  dawn-dusk initial orbit designs and observed per-member TLE cell membership.
- Added sparse Grid/Honeycomb baseline and Ronchetti-style intra-orbit hole
  patching, retaining physical visibility, atmosphere and terminal constraints.
- Added per-service ground association quotas, two-sided preference matching,
  and single/multi/free compute-cell egress policies.

- Added auditable optical aperture/photon link budgets, RF aperture gains and
  per-direction capacity diagnostics; exported exact cell link snapshots plus
  labeled static Serving/AsterMind parameter fragments.
- Added unexpected link/relay outages, in-flight message discard and bounded
  next-sample recovery over alternate routes or ground access links.
- Added optional native SimGrid 4.1 transport alongside the default directed
  queue model, with causal completion callbacks, dynamic shared capacities,
  byte-progress audit intervals, zero-capacity suspension and contact-loss cleanup.
- Added analytical native-backend and end-to-end model-comparison regressions.

- Added physical multi-shell communication experiments with Walker/TLE inputs,
  independent grid/honeycomb shells, separate compute formations, dynamic acquisition,
  terminal limits, RF link budgets, Sun exclusion and switchable routing.
- Added automatic parallel orbit/geometry precomputation and complete compressed
  per-boundary network audit artifacts with a small result index.
- Added network unit, integration and experiment-CLI tests, including serial/parallel
  determinism and complete node/link export checks.

- Added a first-order L2 Serving formation model with member count, ring/switch/
  fully-connected topology, link bandwidth, and one-hop propagation delay.
- Added canonical logical compute-node views for Orbit Trace and Runtime state.
- Added frozen Serving profiles for Llama 3.3 70B, DeepSeek V3, and Kimi K3,
  with length-and-concurrency rate tables for prefill and decode.
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

- Isolated personal launchers, templates, browser replay exports and their tests
  under ignored local directories; shared experiment commands remain portable.
- Removed cross-communication-shell connectivity and reject explicit bypasses.
- Kept six-file Workload loading unchanged; corrected the older README description.
- Documented directed link reservations separately from SimGrid bandwidth sharing.

- Personal large communication scenarios use 600 seconds with 1-second physical
  sampling. Production positions, Sun directions and visibility windows use
  sample-and-hold semantics; network simulation no longer interpolates positions.
- Streamed large frame exports, retained sparse link-state timelines and batched
  Orbit precomputation to bound memory at finer sampling intervals.
- Added vectorized, same-epoch satellite coordinate transforms with parity tests
  against the scalar transformation.
- Exported actual serialization-completion events for scientific audit; personal
  paginated Runtime replay is derived separately by the ignored local tools.


- Integrated physical networking into the shared Architecture/Serving/Workload/
  Runtime/cost evaluator and physical DSE; migrated executable experiment examples.
- Replaced future request-order state publication with chronological request,
  per-hop transmission, compute admission/completion and client delivery events.
  Session KV becomes visible at actual compute completion.
- Enforced physical cell connectivity for Serving admission, per-member compute
  energy and cancellation on internal link disconnection or zero capacity.
- Kept terrestrial baseline contracts and execution separate from satellite
  Runtime DTOs, restoring baseline integration coverage.
- Added complete-system, event-causality, dense/MoE coupling and import regressions.
  Rejected mismatched hardware, physical cells and Workload models.


- Removed fixed-shell Arc communication and its configuration; Runtime now
  consumes physical dynamic network state and directed transmission calendars.
- Enforced LF for pinned reference data so checkout line endings preserve
  provenance hashes, and excluded personal inputs, results and viewer files.

- Replaced the per-node Prefill/Decode FIFOs with colocated request-concurrency
  reservations and fixed admission-rate lookup.
- Serving now emits per-request Prefill and Decode rates indexed by length and
  homogeneous static-batch concurrency.

- Made `OrbitTraceManifest` and `OrbitTraceBundle` the canonical Orbit trace
  class names, with geometry-only records and metadata as their complete public
  output contract.
- Grouped SGP4 implementation details by responsibility under
  `astermind.builder.orbit.sgp4` while keeping cross-module use on the public
  `astermind.builder.orbit` entry point.
- Removed `v2` from the canonical Orbit Trace schema and public API names;
  `astermind.builder.orbit_trace`, `write_orbit_trace`, and `load_orbit_trace` now
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

- Established the `architecture`, `orbit`, `workload`, `serving`, `runtime`, and `common` module boundaries under `astermind/`.
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
