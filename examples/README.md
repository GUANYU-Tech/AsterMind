# Examples

Native fixed-design scenarios live in `simulator/examples/` and are loaded as
SimulationConfig. See [Native setup and execution](../simulator/README.md).

- `capacity/` retains Native capacity-study inputs.
- `experiments/` retains documents for the independent terrestrial baseline API.
- `single_satellite/` retains static hardware, energy sizing and cost inputs.
- `workloads/` retains independent workload descriptions, bundles and source traces.
- `astermind_joint/` retains standalone configuration/replay helpers; its scripts
  using the retired request adapters have been removed.

The Native CLI does not dispatch terrestrial baseline documents.
Write generated outputs under `results/` or `local/`, not `examples/`.
中文说明见 [使用说明](使用说明.md)。
