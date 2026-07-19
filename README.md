# OrbitServe

OrbitServe is a simulation platform for LEO compute constellations serving global terrestrial AI inference demand. It studies two-timescale cost-SLO co-design across construction-time design-space exploration and runtime scheduling.

## Repository Layout

```text
src/orbitserve/              Core simulation engine
  architecture/              Identical per-satellite hardware, static energy sizing, and cost parameters
  orbit/                     Constellation geometry, orbital motion, visibility, sunlight, and traces
  workload/                  Canonical workload contracts, validation, loading, and materialization
  serving/                   Workload-independent model deployment and per-group capacity
  runtime/                   Routing, queues, scheduling, KV state, SOC, and request-level execution
  common/                    Policy-free metric DTOs and serialization
src/experiments/             Manifests, cross-module orchestration, fixed-design evaluation, and output
src/dse/                     Design enumeration, runtime matrix, static SLO proxy, and frontier analysis
src/baselines/ground/        Terrestrial serving baselines
src/workloads/wildchat/      WildChat conversion tools with optional dependencies
examples/                    Versioned, user-facing input examples
tests/                       Unit, integration, smoke, and slow regression suites
```

See `CHANGELOG.md` for release history.

## Execution Path

```text
orbitserve.experiments.cli
  -> load ExperimentSpec
  -> load and materialize a canonical workload bundle
  -> evaluate_materialized_design
  -> Architecture + Orbit + Serving
  -> build RuntimeScenario
  -> evaluate Runtime
  -> energy alignment
  -> optional runtime optimization / DSE
  -> structured output
```

Fixed-design evaluation and DSE candidates share `evaluate_materialized_design()`. Architecture does not infer hardware from workload demand. Orbit produces geometrically visible candidates. Runtime selects the active graph under Architecture terminal limits.

## Stable Module Contracts

- Architecture exports `SatelliteArchitectureProfile`, `SatelliteHardwareProfile`, and `SatelliteEnergyProfile`. These contain static, identical per-satellite hardware and energy configuration; cost-only inputs remain private to Architecture.
- Orbit exports `OrbitTraceManifest` and `OrbitTraceBundle`. Their public representation contains geometry, visibility, propagation, and illumination data, but no link capacity, DVFS decision, or SOC state.
- Workload exports `MaterializedWorkload` and the canonical seven-file bundle. Runtime accepts canonical request fields only and does not translate legacy aliases.
- Serving exports `ServingCapacityProfile`. It describes one colocated deployment or one prefill/decode group from model, deployment, and per-satellite hardware configuration. It does not read Workload or choose constellation-wide group counts.
- Runtime combines the four contracts, instantiates serving groups from the available constellation, and owns routing, queueing, request latency, DVFS, SOC, and request-level metrics.

Core modules import only through public package interfaces. The boundary regression test is `tests/unit/test_module_boundaries.py`.

## Installation

```bash
python3 -m pip install -e .
```

WildChat dependencies such as `pyarrow` and `transformers` are optional and are not required by the core package.

## Workload Contract

The core engine accepts one canonical seven-file workload bundle:

```text
workload_manifest.json
regions.json
gateways.json
requests.jsonl
sessions.jsonl
timebin_demand.csv
workload_summary.json
```

Dataset-specific inputs must first be converted under `src/workloads/<dataset>/`. Experiment manifests then reference the generated bundle through `workload.bundle_dir`. The core package does not retain legacy workload readers or Runtime request-field aliases.

## Running Examples

Fixed design:

```bash
python3 -m orbitserve.experiments.cli \
  examples/experiments/fixed_design_geometry.json \
  --out results/experiments/fixed_design_geometry.json
```

DSE:

```bash
python3 -m orbitserve.experiments.cli \
  examples/experiments/orbit_geometry_dse.json \
  --out results/experiments/orbit_geometry_dse.json
```

Terrestrial baseline:

```bash
python3 -m orbitserve.experiments.cli \
  examples/experiments/skywalker_ground_reproduction.json \
  --out results/experiments/skywalker_ground_reproduction.json
```

The CLI writes a summary and separate audit artifacts by default. Use `--output-mode full` for a monolithic debugging output. Examples contain inputs only and do not include pre-generated experiment results.

## Single-Satellite Cost Tool

```bash
python3 -m orbitserve.architecture.single_satellite_cost.cli \
  examples/single_satellite/h200_gpu_derived.json \
  --out results/single_satellite/h200_gpu_derived.json
```

## Tests

```bash
python3 -m unittest discover -s tests/unit -t .
python3 -m unittest discover -s tests/integration -t .
python3 -m unittest discover -s tests/smoke -t .
python3 -m unittest discover -s tests/slow -t .
python3 -m unittest discover -s tests -t .
```

`tests/slow/` contains long-running trace-backed runtime, prefill/decode, and runtime-matrix tests. Full validation must include this suite; routine development may run the first three suites separately.

## Directory Rules

- `results/` contains final experiment outputs and is not tracked by default.
- `scripts/` is reserved for paper experiment orchestration after the methodology is frozen.
- `examples/` contains user-facing inputs, not test-generated artifacts.
