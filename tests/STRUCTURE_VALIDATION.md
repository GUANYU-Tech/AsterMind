# Structural refactor validation

This iteration reorganizes configuration and runtime policy ownership without
changing simulation behavior. The comparison baseline is
`6930042ce3222d5d1a323fd9fcc509c2b52b3b68` on `version/AsterMind-V1.0`.

## Changes and compatibility

1. `astermind/builder/config/` separates scenario, resources, deployment, requests,
   network and communication configuration. Existing public imports, class
   identities, pickle identities, schemas and type-hint reflection remain valid.
2. `astermind/engine/policies/` contains request scheduling, admission, routing,
   connection and ground association rules. Existing Engine objects still own
   queues, session bindings, route caches, reservations and mutable network state.
   Native vLLM batch scheduling remains in `engine/compute/`; collective selection
   remains in `engine/comm/`.
3. Builder physical loading prepares datasets and configuration. Engine physical
   setup constructs `NetworkState`, realizes runtime faults and applies Cell link
   edits. Existing physical loading APIs preserve return values, ordering, temporary
   directory ownership and cleanup on errors.

No route algorithm, topology rule, compute predictor, communication model, request
gate, event clock, default configuration or frontend implementation is changed.

## Reproducible full gate

The final agreed sequence completes all three structural moves before unified
acceptance. Earlier completed baseline/Builder evidence is retained; the
interrupted intermediate policy run is not a completed acceptance result.

Run from the repository root. Use the same Python environments, pinned vLLM 0.30
source, original ISLSim core and browser runtime for baseline and candidate. The
0.11 environment is required for the original PD tests. Use a new output directory
for the final integrated candidate; `--resume` only reuses successful phases with identical
source, script, helper, command, environment and dependency identities.
`dependencies.json` covers Python/distribution identities, compute source, ISLSim
source, Node, the browser test script, extracted shared libraries and the npm lock.
The installed Playwright packages and Chromium files also need a frozen content
manifest when verifying the browser environment; their directory paths alone do
not establish byte identity.

```bash
python -B tests/scripts/run_structure_regression.py \
  --repo "$PWD" \
  --output /absolute/path/to/batch-results \
  --python011 /absolute/path/to/vllm011/bin/python \
  --vllm030-source /absolute/path/to/pinned-vllm030 \
  --islsim-root /absolute/path/to/original-islsim-core \
  --islsim-core-only \
  --node /absolute/path/to/node \
  --node-path /absolute/path/to/node_modules \
  --browser-path /absolute/path/to/playwright-browsers \
  --browser-libraries /absolute/path/to/optional-shared-libraries

python -B tests/scripts/compare_structure_results.py \
  --baseline /absolute/path/to/baseline-results \
  --candidate /absolute/path/to/batch-results \
  --output /absolute/path/to/comparison.json
```

The optional library-path argument can be omitted when Chromium dependencies are
already available. A shared reviewed browser script can be supplied with
`--browser-script`; its bytes must match between baseline and candidate. For an
unchanged historical checkout, launch the same refactor harness from outside both
repositories. It imports and executes the repository selected by `--repo`.

`--only`, compute `--discover` and compute `--cases` are development tools. Their
partial or discovery evidence cannot satisfy full acceptance. Missing dependencies,
unexpected failures and skipped module tests are not reported as passes.

## Coverage and scope

| Gate | What is executed and checked |
| --- | --- |
| Native build and source identity | Both CPU helper libraries; pinned source mapping and adaptation hashes |
| Retained and request import | Existing module tests, explicit request-batch external tests, added compatibility/comparator regressions |
| Contracts | Schema and normalized config equality, invalid-input errors, pickle round trips, type hints, old and new import orders |
| Physical oracle | 6 scenarios, 144 ordered route queries; 3 routing policies, geometry/topology/fault/cache behavior |
| Policy matrix | 32 ground association combinations, 8 candidate topology variants, existing scheduling/admission/control/routing suites |
| Source oracles | 12 compute comparisons with 9,576 predictions; 24 communication comparisons |
| Communication matrix | Supported operation/algorithm/scope combinations, protocol and capacity bases, selection and completion traces, resource cleanup |
| Compute support | 146 mode-cases: 98 successful prefill/decode plan and prediction checks, 48 frozen negative entry contracts |
| Native integration | 30 original DS reports in both global compute modes, TP/PP/MoE EP, shared contention, intra surrogate and roofline, KV copy/move, WideEP DP and PD |
| End to end | 8 original whole-platform reports: moving physical network, requests, gated migration, capacity changes, failure, timeout and repeat determinism |
| ISLSim core | 8 checks using the original `config.py` and `link_model.py`; capacity, latency, received power and modulation |
| Distribution | Build sdist and wheel, install away from source, resource/import checks, native CLI execution and audit |
| Historical frontend | Real Chromium: replay controls, actual pause, filters, dependencies, Scheduler detail, switching, desktop/mobile captures |

The compute matrix distinguishes the host compute entry from direct imported
source APIs. Twenty-two successful mode-cases exercise original vendor-fabric or
GLM APIs with their required runtime declarations. They do not establish that the
current host wrapper or full Simulation supports those same configurations.
Negative host contracts include missing profile bindings or declarations and
specialized entries such as Kimi K3/DSpark. They are not assertions that those
models are unsupported by every upstream API. Full Scheduler integration is
verified separately by the native DS and end-to-end cases.

DS02 deliberately uses the existing lower-level integration; it is not a legal
full Builder deployment with shared physical GPU ownership. DS07-idle verifies
the original explicit unbound-token failure. DS08 verifies real prefill/decode
handoff, one-token bypass and failed KV preventing decode. None uses successful
mock compute as evidence.

## Exact comparison

Both full scoped gates must complete, with identical external dependency
identities and scope exceptions. Compare all probes, contracts, matrix summaries,
30 native and 8 end-to-end reports, installed-package output and audits. Simulation
values, types, array order, event order and integer ticks must match exactly.
The comparator only permits enumerated host observations (worker PID, measured
wall times and host-memory startup measurements) and verified artifact-root path
prefixes. Differing host measurements are checked for valid finite values before
their differences are permitted.
Parallel Cell startup can report completion groups such as 0→1→2 or 0→2.
Only this host progress sequence is reduced to its initial/final boundaries after
validating complete, monotonic integer counts, active Cell membership and worker
limits. Runtime events are not reduced. Absolute paths in `cleanup_error` Python
traceback frame lines are normalized; ordinary messages remain exact. Source
Scheduler config receipts are checked against their original raw configuration
hash before recomputing a hash of the full path-normalized configuration. Neither
configuration contents nor receipt integrity are ignored.
Unknown differences fail. Frontend exports and screenshots compare byte for byte.
Large reports are losslessly compressed after verifying their decompressed SHA;
the original evidence is not reduced.

## Explicit limits accepted for this iteration

- Full external ISLSim adapter testing remains **unverified**: the available
  original snapshot lacks `validator.py`, which the unchanged adapter requires.
  The user accepted core-only verification. No validator or default file is
  fabricated, and the adapter is not changed.
- `viz/` remains unchanged. New schema-2 mode reports are raw reports; their
  historical exporter incompatibility already exists on the baseline. Browser
  fixtures are explicitly historical UI contracts, including an extended
  interaction fixture; they are not native Simulation results. New-mode numerical
  correctness is established by the native and end-to-end gates.
- Passing tests establish structural equivalence and integration contracts; they
  do not establish predictor accuracy against measurements on physical hardware.

## Execution evidence

On SeeSpace1, evidence for this iteration is stored outside the source repository
in `/data1/guolidong/.astermind-structure-20261113/`. The frozen original baseline
is `results/baseline-full`; the final integrated candidate is `results/final-full`.
`results/final-comparison.json` records the final exact comparison and comparator
SHA. Each full gate records its source fingerprint, dependency manifest, phase
commands/signatures/exit codes, scoped completion summary and individual logs.
The separate `browser-runtime-files.json` freezes actual Chromium/Playwright files.

Final acceptance requires both complete scoped summaries, all 19 required phases
passing, a successful final comparison with no unexpected differences, unchanged
browser runtime bytes, and a reviewed candidate-to-branch transfer manifest.
The original first-pass Builder comparison is retained for traceability; its
reviewed replacement is `results/batch1-comparison-reviewed.json`. The interrupted
`results/batch2-full` must not be presented as a completed full gate.
