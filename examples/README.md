# Examples

This directory contains readable, reusable, version-controlled input examples. It does not contain test-generated artifacts or pre-generated experiment results.

## Layout

- `experiments/` contains complete manifests for fixed-design, DSE, and terrestrial-baseline runs.
- `single_satellite/` contains static per-satellite hardware, energy-sizing, and cost inputs.
- `workloads/descriptions/` contains descriptions used to build canonical workload bundles.
- `workloads/trace_examples/` contains canonical seven-file bundles that experiments can load directly.
- `workloads/source_traces/` contains small raw inputs for trace-replay demonstrations.

`fixed_design_geometry.json` demonstrates a complete Orbit-geometry-to-Runtime path. `orbit_geometry_dse.json` enumerates orbital geometry candidates through the same evaluation path. Both use the same workload and per-satellite hardware input.

See the repository root `README.md` for commands. Write outputs to `results/`, never back into `examples/`.
