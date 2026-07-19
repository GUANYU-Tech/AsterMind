# OrbitServe

OrbitServe is a simulation platform for LEO compute constellations serving global terrestrial AI inference demand. It studies two-timescale cost-SLO co-design across construction-time design-space exploration and runtime scheduling.

## Repository Layout

```text
src/orbitserve/              Core simulation engine
  architecture/              Identical per-satellite hardware, static energy sizing, and cost parameters
  orbit/                     Constellation geometry, orbital motion, visibility, sunlight, and traces
  workload/                  Canonical workload contracts, validation, loading, and materialization
  serving/                   Model deployment, prefill/decode, KV capacity, and throughput estimation
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

Dataset-specific inputs must first be converted under `src/workloads/<dataset>/`. Experiment manifests then reference the generated bundle through `workload.bundle_dir`. The core package does not retain legacy workload readers.

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
