# Validation

All commands run from the repository root with native dependencies installed.
The retained suite rejects skipped tests; explicit native scripts fail on missing
Schedulers/models. External directories are explicit suites, not implicit unittest
recursive discovery. Old execution-mode tests are replaced by the cases below.

```bash
python -m astermind.common.build_native
python tests/scripts/run_retained_tests.py
python tests/scripts/run_compute_comm.py --cases DS01 DS02 DS03 DS04 DS05 DS06-copy DS06-move --output results/matrix
python tests/scripts/run_compute_comm.py --cases DS07-pp1 DS07-pp2 DS07-idle --output results/matrix
# Run in the separate official vLLM 0.11 Scheduler environment:
python tests/scripts/run_compute_comm.py --cases DS08-one-token DS08 DS08-kv-failure --output results/matrix
python tests/scripts/run_compute_comm_e2e.py --output results/e2e
python tests/scripts/check_distribution.py --work-dir /absolute/path/to/new-check
```

The matrix defaults to both global compute modes. Add `--intra surrogate
IntraSatelliteRoofline` for both intra models where supported. DS03/05/07 need the
pinned vLLM 0.30 source and worker Python environment variables documented in
`astermind/README.md`. No test supplies fake successful compute as native evidence.

| Case | Acceptance |
| --- | --- |
| DS01 | Two stars, TP2 per star, real prefill/decode, both intra models |
| DS02 | Two independent cross-star TP2 schedulers compete for one shared Comm; solo vs concurrent |
| DS03 | Cross-star TP2/PP2, actual pipeline sends and dependencies |
| DS04 | Cross-star MoE EP2, seeded routing and exchanges |
| DS05 | Two scale-up groups per star, source hierarchical collective |
| DS06-copy/move | Real native KV partitions, import/commit, prefix reuse, source release |
| DS07-pp1/pp2 | Original WideEP DP4/TP8/EP32 with four active replicas |
| DS07-idle | Source unbound token ownership rejection with idle DP replicas; no fabricated output |
| DS08-one-token/full | Original Kimi TP8/EP8 PD, bypass, all-shard decode handoff, and failed KV prevents decode |
| E2E | 8 moving satellites, 2 grounds, 4 TP2 Cells, 6 requests, gated migration, capacity changes, fault, timeout, repeat determinism |

DS02 is deliberately a lower-level integration: the unchanged physical Builder
assigns a satellite to one Cell, so two Cells sharing both satellites cannot be
claimed as an accepted SimulationConfig deployment.

Unit/module tests cover original route/cache behavior, physical data, controls,
clock precision, prerequisites/KV rollback, collector decompositions, phase barriers,
protocol/FEC capacity bases, directional/shared RF resources, whole-path wait/fail,
capacity updates, cancellation, failure isolation, and no double compute/Comm charge.
`test_cell_time_records` corrupts evidence by one tick or removes gate evidence to
ensure the audit detects it. `test_comm_semantics` tests source intra limitations.

Source differential scripts:

- `probe_compute_source.py`: identical native RuntimeInputs/operator plans and
  predictions in frozen source vs port (not the obsolete compute baseline).
- `probe_comm_source.py`: identical unique-path resource oracle, protocol, traffic
  and model inputs in frozen source vs port; integer ns projection can differ ≤1 ns.
- `probe_retained_physics.py`: old target baseline vs new target, same ordered route
  queries/cache history/faults/physical snapshots; require exact JSON equality.

Build validation creates an sdist, builds a wheel from that sdist with isolated
build dependencies, installs outside source, verifies resources/imports, builds
native helpers, runs a real native CLI case and audits it. It also exports a
historical report using unchanged viz. New modes have complete raw schema-2 reports;
new-mode frontend rendering is deliberately outside scope. Historical browser checks
use `tests/browser/native_replay.cjs` with a suitable recorded fixture.

Real ISLSim remains explicit: `ORBITSERVE_ISLSIM_ROOT=/path/to/ISLSim python -m
unittest discover -s tests/external/islsim -v`. Historical validation records in
`V1_VALIDATION.md` describe earlier commits and must not be read as current features.

WideEP fixtures disable per-allocation service dumps (`trace_service=false`) to
avoid quadratic trace volume; operator/dependency/failure records and aggregate
Comm statistics remain. Resource-conservation tests retain full allocation evidence.

## Structural equivalence gate

See [STRUCTURE_VALIDATION.md](STRUCTURE_VALIDATION.md) for the complete regression
runner, exact baseline comparator, configuration and policy matrices, compute
support boundaries, native WideEP/PD coverage, browser checks and the explicitly
accepted ISLSim external-adapter limitation. The full gate invokes the existing
branch test scripts; partial/discovery runs cannot certify acceptance.
