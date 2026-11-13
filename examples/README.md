# Examples

Native fixed-design scenarios live in `examples/native/` and are loaded as
SimulationConfig. See [Native setup and execution](../astermind/README.md).

- `workloads/` retains independent workload descriptions, bundles and source traces.
- `astermind_joint/replay_data.py` reads historical recorded traces only; it is not
  an executable scene for the new compute/Comm pipeline.
- `native/compute_comm_matrix.py` and `native/compute_comm_e2e.py` generate the
  current acceptance scenarios.

Write generated outputs under `results/` or `local/`, not `examples/`.
中文说明见 [使用说明](使用说明.md)。
