# Reference formation: isolated handoff

Base: `ae6ee46` (version/AsterMind-V1.0). Branch: `feature/reference-formation-20261114`.
No full offline dataset is built. No compute inference is run by this audit.

## Production changes

- `astermind/builder/physical/network_inputs.py`: optional `cluster.motion_model=reference_fixed_lvlh`, required ordered `member_offsets_m` list, reference member 0 at zero offset. Each vector is radial / along-track / orbit-normal in metres. Non-square and radial offsets are supported. Positions do not create links. Existing free-SGP4 remains the default.
- `astermind/builder/orbit/sgp4/elements.py`: optional serialized `fixed_lvlh_offset_km`; offsets participate in element identity.
- `astermind/builder/orbit/sgp4/propagation.py`: propagate reference elements, rotate offsets into TEME, include offset rotation in velocity using a centred 0.1s derivative. Algorithm version marks fixed-LVLH. Per-member duplicated reference elements preserve independent worker partitioning; this is prescribed rigid formation motion, not station-keeping dynamics.

No Network allocator, terminal quotas, external-link-cap, explicit Ring order, compute or Comm implementation changes.

## Configuration and deployment

`scenario.json` is the high-level review configuration. Its cluster offsets are a list ordered by member ID: `[0,0,0], [0,1000,0], [0,1000,1000], [0,0,1000]`. Normal sorted Ring connects 0-1-2-3-0. Only member 0 is eligible for external links.

The existing high-level `isl_terminals` applies to all members of a formation. The audit materializes the configuration and uses existing per-node `isl_terminals` fields to set exit=3, others=2; `materialized-network.json` records the exact resulting configuration. No new member-terminal schema is introduced.

`deployment.json` specifies stage-to-member `[0,1,2,3]`, TP8 per star, PP4, and exit=member0=stage0. It is a deployment specification, not evidence that a real GLM KernelPlan ran.

10 cells (5 Walker 53deg, 5 dawn-dusk), 550km, fixed 1km square, internal 1Tbps/10km maximum, 6000 communication satellites at 500km with 6 shared optical ports, Grid backbone and cross-layer 400Gbps/4000km. No independent 4+2 quota or external cap.

## Reproduce

Run from this worktree with the project Python dependencies (SGP4 2.27, numpy, pytest and existing Orbit dependencies):

```sh
python -m pytest tests/unit/orbit tests/unit/experiments/test_reference_formation.py tests/unit/experiments/test_network_inputs.py -q
python Temp/reference-formation/audit_day.py
```

On Windows checkout, pinned Orbit data must retain repository LF bytes. Initial broad test failures were checksum mismatches caused by CRLF conversion; restoring LF, without changing contents, resolved them. These data files are not feature changes.

## Evidence scope

The day audit samples 0..86400 seconds at 60s intervals. It uses real SGP4 states and the unmodified base Network allocation block, checks all internal Ring edges allocated and 1km long at every sample, checks terminal counts and exit allocations, and reports sampled readiness separately from allocated association.

It is not the full Network event queue or an offline dataset. A 5s handover gap can fall between samples; sampled checks cannot prove continuous-time external availability. Ground links/online requests are excluded. `shared-optical-scan.json` records source hash and wall time; logs are retained in this directory.

Only this feature and its tests/configuration belong on this branch; other workspaces remain separate. Compatibility with a newer Communication_backend must be reviewed at merge time because its package paths differ.
