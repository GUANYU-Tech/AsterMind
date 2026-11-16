# Native compute and communication

`SimulationConfig → Simulation` evaluates one fixed configuration. Builder owns
scene/physical preparation; Engine owns execution. This change imports source
compute and Comm from `Communication_backend` `9c08c9cad81df5a4fee5e606114753df8dab6c2d`,
keeping this branch's routing, physical Builder and automatic Cell assignment.
`engine/compute/SOURCE.json` records per-file origins and adaptation boundaries.

## Configuration and runtime ownership

`builder/config/` groups configuration by domain: `scenario.py` composes the
SimulationConfig; `resources.py` declares devices and links; `deployment.py`
declares Cell placement and parallelism; `requests.py` declares requests and KV
migrations; `network.py` describes physical network policies/resources;
`communication.py` declares Comm options. Existing import paths remain aliases to
the same classes, with their existing serialization identities.

`engine/policies/` separates request scheduling (`scheduling.py`), readiness gates
(`admission.py`), path selection (`routing.py`), candidate/connection priority
(`connection.py`) and ground association (`association.py`). These rules operate
on existing Engine owners. They do not introduce another queue, route cache,
reservation table or network state. Native batch scheduling and communication
collective selection remain in their original compute and Comm modules.

`builder/physical/loading.py` prepares the dataset, validated network spec and
orbit trace. `engine/setup/physical_runtime.py` realizes runtime faults, constructs
the live network and applies explicit Cell link edits. The public `load_physics`
entry composes these steps and retains temporary-resource ownership and cleanup.

## Setup

Install the core package, prepare the required official Scheduler environment,
then build the two small CPU helpers:

```bash
python -m pip install -e .
python -m astermind.common.build_native
python -m astermind --scenario examples/native/two_cells.json --output results/native.json
python -m astermind.engine.audit results/native.json --output results/audit.json
```

The complete native source/config/calibration snapshot is bundled in
`engine/compute/upstream/`. An explicit external Midlayer root must have the same
snapshot tree identity; old external snapshots are rejected. The source export
has no native Git commit (`null`); its source/config tree SHA and snapshot manifest
SHA identify it honestly. No missing predictor or Scheduler is replaced by a stub.

The upstream Scheduler selectors are `auto`, `vllm_0_11`, `vllm_0_30`, `vllm_0_31`.
The two shipped CPU lockfiles and vLLM identities are in
`engine/compute/scheduler_runtime/`; they are Scheduler test environments, not a
complete GPU/HTTP serving dependency stack. Native 0.30 requires the exact source
checkout and `ORBITServe_VLLM030_SOURCE` / `ORBITServe_VLLM030_PYTHON`. Version 0.11
and 0.31 workers inherit their launching Python environment. Use separate runs
where those package versions differ, as upstream does.

## Configuration

- Global `compute_mode`: `midlayer` or `roofline`. Both use the source operator
  graph; only local kernel prediction changes. There is no per-Cell override.
- Cell model/hardware paths and `hardware_id` must agree with source profiles.
  Old A800 profiles may lack new service parameters; unsupported predictions fail
  explicitly instead of substituting a formula.
- `parallel` declares TP/PP/DP/EP; SP remains 1. Placement covers TP×PP×DP GPUs,
  with exclusive GPU ownership. EP is 1 or the stage-local TP×DP group.
- `wide_ep_enabled`, `native_expert_parallel`, `expert_routing_seed`,
  `scheduler_adapter`, `execution_mode`, `host_launch_profile`,
  `memory_policy_file` and explicit KV limits retain source meanings/restrictions.
  `execution_mode` controls source graph execution, not Midlayer/Roofline selection.
- DP>1 requires supported WideEP configuration. Input distribution assumes input
  already reaches every DP leader; it charges only intra-replica broadcast.
  The pinned source may reject a WideEP batch with idle replicas because token
  ownership is unbound; `DS07-idle` verifies this failure rather than inventing work.
- `deployment=pd_disagg` uses separate prefill/decode pools and twice the physical
  placements, prefill ranks first. It retains the restricted upstream model/layout
  support; arbitrary DP/PD combinations are not implied. Explicit KV capacity,
  no session prefix caching and no memory-policy/watermark override are required.
  Decode starts only after every KV shard arrives; one-token output bypasses handoff.

`communication` is the source Comm policy with the source route selector removed:

| Configuration | Meaning |
| --- | --- |
| `intra_mode` | `surrogate` or `IntraSatelliteRoofline` |
| `cell_policies` | Upstream per-Cell algorithms by single-SU/cross-SU/inter-star scope and intra model override |
| `inter_parameters` | Source inter model startup/protocol/calibration parameters; runtime rejects ARQ/feedback/reload modes |
| `runtime_capacity_basis` | `raw_line`, `net_after_existing_fec`, or `application_effective` |
| `runtime_protocol` | `inter_reference` or `payload_only` |
| `resource_profiles` | Per-resource protocol/capacity contracts |
| `ground_processing_ns` | Ground endpoint processing |
| `retry_limit`, `task_timeout_s` | Whole-path retry and terminal deadline |
| `trace_service` | Detailed allocation/service evidence |

Physical `intra_hierarchies` identify GPU→satellite→scale-up groups and topology.
They describe resources; algorithm choice stays in Comm policy. Existing dynamic
physical inputs, route defaults and caches remain authoritative. Deprecated
`backend`, `timeline_mode`, `communication_mode`, packet and ProtoBridge startup
fields are rejected. Read-only config labels exist solely for historical exporters.

## Reports and limits

Schema 2 preserves source model/runtime inputs, pool/replica reports, full operator
plans and predictions, communication selections/phases/resources, capacity and KV
information when actually provided. Request metrics and event correspondence,
all-shard migration commits and failures remain present. The authoritative unit is
an integer attosecond tick (10^-18 s); ns fields are display projections. Native
floating model predictions are converted once at the boundary. Energy is not inferred.

Star-local whole predictions do not share global contention or change during an
operation. External flows require simultaneous whole paths. The original route
forecast can be more conservative than whole-path execution near deadlines.
Hardware prediction accuracy is separate from integration correctness.

`viz/` is unchanged. New reports are raw-only; historical report export is tested.
See `../tests/README.md` for the current test commands and validation requirements.
Historical acceptance reports are local-only and are not distributed with source.
