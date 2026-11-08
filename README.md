# OrbitServe

OrbitServe is a simulation platform for LEO compute constellations serving global terrestrial AI inference demand. It studies two-timescale cost-SLO co-design across construction-time design-space exploration and runtime scheduling.

## Repository Layout

```text
src/orbitserve/              Core simulation engine
  architecture/              Per-role physical hardware, static energy sizing, and cost parameters
  orbit/                     Constellation geometry, orbital motion, visibility, sunlight, and traces
  workload/                  Canonical workload contracts, validation, loading, and materialization
  serving/                   Model deployment, prefill/decode, KV capacity, and throughput estimation
  runtime/                   Routing, queues, scheduling, KV state, SOC, and request-level execution
  common/                    Policy-free metric DTOs and serialization
src/experiments/             Manifests, cross-module orchestration, fixed-design evaluation, and output
src/dse/                     Physical design enumeration, runtime strategies, geometry screening, and frontier analysis
src/baselines/ground/        Terrestrial serving baselines
src/workloads/main_composite_chat/  Synthetic workload adapter and source profiles
examples/                    Versioned, user-facing input examples
tests/                       Unit, integration, smoke, and slow regression suites
docs/research/               Research notes retained from develop
```

See `CHANGELOG.md` for release history.

Development changes are reviewed through a pull request to `develop`. Keep
validation results, affected interfaces and a rollback procedure in the PR;
merging requires the repository's CI and maintainer review.

## Execution Path

```text
orbitserve.experiments.cli
  -> load ExperimentSpec
  -> load and materialize a canonical workload bundle
  -> evaluate_materialized_design
  -> Architecture + Serving deployment validation
  -> independent communication/compute constellations and SGP4 traces
  -> acquisition/terminal allocation/dynamic link attributes
  -> RuntimeSimulationSpec with Workload requests, Serving tables and per-node energy
  -> chronological events: requests, physical hops, compute, KV and delivery
  -> optional DSE runtime-matrix search
  -> structured output
```

Fixed-design evaluation and DSE candidates share `evaluate_materialized_design()`. Architecture does not infer hardware from workload demand. Orbit produces geometrically visible candidates. Runtime selects the active graph under Architecture terminal limits.

## Installation

```bash
python3 -m pip install -e .
```

The physical orbit engine pins Astropy 6.1.3 and NumPy 2.2.6 in the package
dependencies. Keep these versions together; newer NumPy releases remove APIs
used by this Astropy version. Installing the package applies both pins.

WildChat dependencies such as `pyarrow` and `transformers` are optional and are not required by the core package.

## Workload Contract

The core engine accepts one canonical six-file workload bundle:

```text
workload_manifest.json
regions.json
gateways.json
session_requests.jsonl
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

## Physical Network Model

The existing experiment CLI runs fixed designs, DSE and terrestrial baselines in
any supported Python environment. WSL, a particular Conda environment name, a
web server and browser assets are not required.

Satellite manifests provide `physical_network`, compute `hardware`, optional
`communication_hardware`, `serving`, `workload` and `runtime_strategy`.
`physical_network` may be inline or a JSON path relative to the manifest.
Communication shells have independent Walker/TLE inputs and fixed same-shell
grid/honeycomb neighbors. Communication shells do not connect to one another.
Compute cells have explicit physical members and fixed internal neighbors.
Ground access and compute-to-communication access are allocated from currently
eligible candidates under terminal limits and acquisition delays.
`network.handover_policy` selects `retain_valid` (default: keep eligible contacts
and fill free ports) or `quality_first` (rerank dynamic contacts at each boundary).
A retained winner keeps its original acquisition clock; a new target always pays
its configured setup delay. Fixed ISL neighbors do not change with this policy.
Ground matching uses its configured ground/satellite preferences;
`cross_layer_access_policy` selects `nearest_distance` (default),
`highest_capacity` (the effective two-direction bottleneck) or `longest_available`.
These are physical rankings, without Runtime traffic-load prediction. Fixed neighbors do not guarantee physical
availability: visibility, range and Sun constraints still apply.

The production network trace holds positions and environment-derived link
attributes at the latest sample. Its visibility windows change at sampled
boundaries. Acquisition, configured capacity changes, request arrivals and
transmission/computation events keep their exact event times. Environment sample
spacing does not round requests or reset partially transmitted bytes. The direct
geometry/STK benchmark API retains its separate continuous-boundary convention.
Orbit precomputation can split independent work across processes and bounded time
batches without changing these simulated-time semantics.

### Reusable physical datasets

Physical preparation and Runtime can run as separate stages. First call
`orbitserve.experiments.network.run_network_experiment(physical_input, dataset_dir)`.
Its completion marker, `physical_dataset.json`, identifies a complete dataset:
the self-contained materialized input, Orbit/environment trace, every sampled
node position and candidate link, directional budgets, link events and the
initial causal acquisition timeline. `physical_state.sqlite` stores compressed,
indexed attribute blocks; Runtime reads them through bounded caches.
Small datasets (up to 64 MiB of compressed attribute payloads) are read once
into memory. Fractional route-arrival times reuse the same physical-boundary
cache entry, while exact capacity-schedule and contact boundaries stay distinct.

After geometric screening, adaptive ground/cross-layer pairs without any contact
in the full horizon are excluded. Future contacts remain eligible when their
window opens. Outside geometric availability, audit records carry zero bandwidth
and `budget_evaluated: false`; they do not fabricate a physical budget. Fixed
ISLs retain their configured topology. Physical attributes, JSON encoding and
compression run in bounded process chunks using the same worker budget as Orbit,
with one SQLite writer and per-chunk progress. An incomplete database is kept
under a temporary name and is never published as a finished dataset.
Datasets from the previous physics version must be regenerated.

Set the experiment manifest's top-level `physical_dataset` to that directory or
its `physical_dataset.json`, relative to the experiment manifest. Runtime verifies
the physics version, input identity and all artifact SHA-256 hashes before use.
`physical_network` can be omitted to load the baked input directly. If supplied,
it must match the source or materialized input; only `network.transport_model`
and independent `route_probes` are excluded from physical identity. Workload,
Serving and routing choices remain independent experiment inputs.

An explicit dataset reference never falls back to physical regeneration. Missing,
damaged, stale or mismatched datasets fail with an instruction to prepare a new
dataset. Same-filesystem runs share immutable artifacts with hard links, falling
back to file copies across filesystems. Each run retains its own traffic state
and records the source dataset identity. Legacy one-command manifests without a
dataset reference still prepare physics before Runtime.

The local console's first step generates this full dataset before entering the
second step. The second step can also open directly and select a previously
completed dataset. Configuration drafts alone cannot start Runtime. Fault inputs,
random seeds and recovery policy are Runtime choices, excluded from physical
identity. Changing them reuses the same dataset. Runtime observes faults causally
and reallocates terminals using saved physical attributes; it never learns future
faults from exported audit frames. Interactive creation of an unprepared satellite
pair evaluates that pair from saved positions without modifying the base dataset.

### ISLSim configuration and local link calculation

New JSON network inputs default satellite-to-satellite profiles to
`capacity_model: "islsim"`. This includes communication ISLs, compute-cell ISLs
and cross-layer optical links. Ground/RF profiles keep their existing model.
Explicit `"fixed"`, `"optical_photon_budget"` or `"free_space_shannon"` profiles
opt out. Low-level callers constructing `LinkProfile` objects select the model
explicitly; their existing fixed-capacity constructor remains supported.

Configuration generation does not import or invoke ISLSim and does not read a
personal checkout or `local/` settings. Export a complete native JSON input with
OrbitServe's explicit scenario defaults and optional partial parameters:

```bash
python3 -m orbitserve.experiments.islsim_cli --out results/islsim.json
python3 -m orbitserve.experiments.islsim_cli --input profile.json --out results/islsim.json
```

`profile.json` may contain native `tx`/`rx`/`channel` sections or a link profile's
`islsim_config`. `prepare_islsim_config()` provides the same portable validation
and export contract. Repository CI tests this contract without an ISLSim checkout
or cross-repository credentials. The bundled examples explicitly select fixed
link capacities so they run independently of the external optical model.

Local physical calculations import the external ISLSim `compute_link()` without
modifying its formulas. Set `ORBITSERVE_ISLSIM_ROOT` to its checkout directory;
the personal console also reads `local/config/communication-modules.json`.
`ORBITSERVE_ISLSIM_CONFIG` optionally selects a native JSON configuration.
The local `resolve_islsim_config()` helper can read the selected native JSON
(otherwise `configs/google_constellation.json`) and return parameters to place
in a profile before generating its portable input.
A missing or mismatched external model raises an error; it never silently falls
back to another model. Integration uses the locally available checkout; it does
not clone, pull, push or modify the ISLSim repository.

```json
{
  "communication_isl": {
    "capacity_model": "islsim",
    "bandwidth_bps": 200000000000,
    "setup_time_s": 3
  }
}
```

The profile's optional `islsim_config` contains ISLSim's native `tx`, `rx` and
`channel` objects. Partial sections override OrbitServe's portable input defaults;
the frozen physical input always contains the complete resolved configuration.
The Python API receives an `FSOConfig` object plus `distance_km`, so creating or
editing a file in the external ISLSim repository is unnecessary. Native inputs
are separate from the legacy photon-budget fields.

| Section | OrbitServe supplies to ISLSim |
| --- | --- |
| `tx` | `tx_power_w`, `tx_aperture_m`, `wavelength_m`, `baud_rate_ghz`, `wdm_channels`, `fec_overhead`, `dual_polarization`, `wdm_power_mode`, `modulation_pool` |
| `rx` | `rx_aperture_m`, `preamp_gain_db`, `preamp_nf_db`, `ber_threshold`, `terminal_cap_gbps` |
| `channel` | `link_margin_db`, `optical_efficiency` |
| Per sample | Current endpoint separation in km |

The Starlink generation presets explicitly pass their per-direction optical
scenario ceiling to ISLSim's `rx.terminal_cap_gbps` (100 Gbps for V2 Mini and
200 Gbps for the current V3 presets). Unpublished power, aperture and waveform
values use explicit base parameters or OrbitServe's portable defaults and remain editable engineering
assumptions shared across generations, not inferred vendor measurements. Scenario
overrides layer over generation defaults. Changing only a TX field retains the
generation's RX settings. Optical port count and aggregate RF capacities stay in
OrbitServe's topology/resource allocation; they are not single-link ISLSim inputs.

The console offers automatic modulation from the configured candidate pool,
fixed QPSK/8QAM/16QAM/64QAM, and a custom ordered pool. Fixed mode supplies one
candidate; it still fails the link when the native BER condition is unmet.
The current external implementation validates `preamp_gain_db` but does not use
it in its high-gain SNR approximation. Wavelength affects received power, while
the current noise approximation uses a fixed 1550 nm photon energy. These upstream
behaviors are preserved and must not be interpreted as independent calibration.

OrbitServe freezes these parameters and the external source SHA-256 identities
in the materialized physical input. Actual capacity uses ISLSim's terminal-capped
`total_bw_gbps`, converted to bit/s and limited by the configured per-direction
link and endpoint ceilings. ISLSim's BER feasibility result makes an infeasible
link DOWN; reacquisition still pays OrbitServe's configured setup delay.
Snapshots retain native SNR, modulation, bandwidth, limiting reason and latency.
Propagation is charged once. OrbitServe continues to own SGP4 geometry,
visibility, acquisition, scheduling, routing and fault recovery.

The physical dataset version changes with this integration. Rebuild older
datasets before running with ISLSim; selecting another link model also requires
matching physical data. This adapter verifies consistency with upstream results,
not independent calibration of ISLSim's physical assumptions or formulas.

Network transport is selected by `physical_network.network.transport_model`:

| Value | Execution and contention |
| --- | --- |
| `queue` | Whole-message store-and-forward; one transmission occupies a link direction and later messages queue. |
| `fluid` | OrbitServe equal-share fluid service; arrivals, completions and capacity changes update the shares of active flows. |
| `flowsim` (default) | AsterMind FlowSim provides max-min sharing of directed hops and aggregate RF pools; OrbitServe drives network state and the event clock. |
| `simgrid` | Real SimGrid 4.1 CM02 activities share each directed physical hop; native activity completion drives the Runtime event queue. |
| `ns3` | Real ns-3 point-to-point serialization with 1400-byte payload packets, 2-byte PPP overhead and packet round-robin admission. |

The default FlowSim backend requires the shared-runtime `_flowsim` extension
from a separately prepared AsterMind checkout. Install that checkout into the
same Python environment (`python -m pip install -e /path/to/astermind`) for
the extension, standalone CLI and joint serving bridge. On the configured
workstation, `third_party/astermind` holds the modified local checkout; it and
the root workstation-only build adapter are ignored by Git and are not supplied
by an OrbitServe checkout. Joint examples require that local dependency to be
prepared first. An older `_flowsim` fails explicitly.
The native shared `advance_until()` must also guard positive residual-byte
completion times that round back to the current native clock, using the next
representable native time as its standalone `run()` loop already does. The
FlowSim fractional-tail regression checks this requirement. The workstation
source fix remains in the ignored AsterMind checkout and must be carried with
that external dependency when preparing another environment.
OrbitServe supplies candidate links and updates capacity at runtime; FlowSim
retains in-flight bytes during capacity changes and shares each directed hop
and applicable RF pools across all flows. OrbitServe handles per-hop routing,
propagation, network visibility, and link recovery. New candidate edges added
after the simulation begins are not yet supported by this backend. A lost
contact follows OrbitServe's existing interrupted-hop policy: partial bytes on
that hop are discarded on retry, while completed hops are retained.

An experimental joint run can pass `compute_runtime=AsterMindRuntime(cells)`
to `run_simulation`, with `AsterMindRuntime` and `AsterMindCell` imported from
`orbitserve.runtime.engine.astermind_bridge`. The `cells` mapping associates
each selectable compute node with `AsterMindCell(config, rank_mapping,
repository)`, where the rank mapping covers exactly the physical cell members.
OrbitServe owns the clock and network; each cell runs a Midlayer scheduler and
submits its collective flows into that same network. FlowSim is the default;
select `transport_model="ns3"` for the packet-level alternative. The current bridge accepts
independent requests with at least one output token. Cross-request KV reuse,
request cancellation and recovery of a failed active cell are not yet supported;
an active serving failure aborts the run. Compute energy uses OrbitServe's
per-token admission estimate, not Midlayer kernel-level energy. Collective
communication is recorded against the cell, not attributed to individual
requests in their path or energy ledgers. OrbitServe's shadow DVFS policy is
not supported in this mode. The existing queue and legacy backends remain
available pending full integration.

### Native Cells from Merge_ML_Version

The first-party `integration/`, `simulator/` and `network_sim/` adapters now run
real Mid-Layer Cells and the official vLLM 0.31 Scheduler on the dynamic Runtime.
Explicit GPU/switch/terminal graphs support multiple GPUs per satellite, cross-
satellite TP/PP and per-rank KV migration. The existing `_EventEngine` owns all
physical hops, resource contention, acquisition and fault recovery. Native COMM
predictions are not charged again alongside physical communication.

```bash
python -m orbitserve.experiments.cli simulator/examples/dynamic_tp2_pp2.json \
  --midlayer-root /path/to/midlayer --out results/cells/cross.json
python -m orbitserve.experiments.cli simulator/examples/dynamic_migration_tp2.json \
  --midlayer-root /path/to/midlayer --out results/cells/migration.json
```

See [native setup, examples, resource semantics and validation](simulator/README.md)
and [the Runtime adapter contract](src/orbitserve/runtime/CELL_INTEGRATION.md).
Mid-Layer source/calibration and native binaries remain external local dependencies.
Explicit test KV block counts are not hardware-derived HBM limits; bound memory
policies are checked separately. Native kernel energy is not evaluated.

`python -m simulator.replay REPORT.json --out local/results/native-cells-viewer`
exports an offline replay of real Scheduler batches, per-GPU operators, dependency
edges, physical communication and KV transactions. The local console links it at
`/native-cells/`. Its checks expose concrete trace violations and keep hardware
accuracy and native energy explicitly outside the verified scope.

`examples/astermind_joint/full_serving.py` runs a 72-plane, 22-satellite-per-plane
communication shell and two compute replicas on dawn-dusk reference orbits.
Use `--users single|multi`, `--parallel tp_pp|ep` and `--backend flowsim|ns3`:

```bash
python examples/astermind_joint/full_serving.py --users multi --parallel tp_pp \
  --backend flowsim --output local/results/full-serving/multi-tp_pp-flowsim.json
python examples/astermind_joint/full_serving.py --users multi --parallel tp_pp \
  --backend ns3 --output local/results/full-serving/multi-tp_pp-ns3.json
```

The single-user case submits one request; the multi-user case submits eight
independent requests from four users. Inputs and replies traverse actual eligible
ground, communication-shell and compute-access hops. Qwen3-4B uses TP2 and PP2
within each four-satellite replica; Qwen3-30B-A3B uses TP2 and EP2 within each
two-satellite replica. Replica-level request dispatch is separate from native
worker-pool DP, which remains one. Midlayer predicts kernel latencies and lowers
collectives; these runs do not execute model weights or generate text. EP uses
explicit seeded routing assumptions. Formation offsets are prescribed in LVLH
around SGP4 reference states, rather than independently propagated formation orbits.

On the configured workstation, the original local console at port 8766 exposes
`/serving?users=multi&parallel=tp_pp&backend=flowsim` and the corresponding
`backend=ns3` view. It retains the original 3D Earth and satellite renderer.
Both views replay executed events; the page can launch a new simulation.
FlowSim shows continuous flow service with changing allocated bandwidth. The
ns-3 view shows packet serialization intervals recorded at native `PhyTxEnd`
events, with exact logical payload, wire size and message offset. Link zoom,
slow playback and a contention jump expose simultaneous fluid sharing versus
packet interleaving. The 3D packet marker's position encodes its message byte
offset, not a packet's propagation trajectory: propagation still follows the
OrbitServe whole-message per-hop model. Optional packet recording is disabled
for ordinary runs; this demonstration enables it explicitly.

`examples/astermind_joint/collectives.py` runs four-rank training communication
graphs (`--mode ring4` for one formation cell, `--mode multi` for four single-satellite
cells), with FlowSim or ns-3. It uses Midlayer's collective lowering and validates
all 51 primitive completions and dependency ordering while three background
requests arrive. SGP4 propagates an initial LTAN-06:00 dawn-dusk design; the formation
case prescribes a square in the reference satellite's LVLH frame. The workload
models communication and configured compute barriers, not a full training compute
or accuracy model. On the configured workstation, the original local 8766 console
exposes `/collectives` with its existing Earth, satellite, cell and transfer renderer.

Install ns-3 separately with `python3 -m pip install ".[ns3]"` on a supported
platform, with a C++20 compiler available (`sudo apt-get install build-essential`
on Ubuntu, or set `CXX`). Each worker compiles a small temporary adapter against
the installed ns-3 headers and libraries, avoiding cppyy JIT initialization traps.
It uses a 1 ps native clock and lazy directed
point-to-point resources. Capacity changes apply at the next packet boundary.
Logical payloads retain microbyte precision; the native wire packet pads its last
payload to an integer byte and adds the 2-byte PPP header. Padding is not counted
as delivered application data.
OrbitServe still performs hop selection, propagation and fault recovery; this
adapter does not simulate TCP or IP routing. Selecting a missing native backend
fails explicitly. The queue and fluid backends require neither library.

SimGrid 4.1 may build from source. On Ubuntu or Ubuntu under WSL, install its
system build dependencies before installing the optional backend:

```bash
sudo apt-get update
sudo apt-get install --yes --no-install-recommends \
  build-essential cmake libboost-dev pybind11-dev python3-dev
python3 -m pip install --upgrade 'pip==26.2.2'
python3 -m pip install --build-constraint tests/requirements/native-build.txt ".[simgrid]"
```

Run these commands from the repository root. The build constraint selects
pybind11 2.13.6 for SimGrid 4.1's isolated build; newer pybind11 property checks
are incompatible with this SimGrid release. It does not disable build isolation.

SimGrid's supported operating systems are determined by SimGrid, not a Conda
environment name. Queue mode does not import or require SimGrid. Explicitly
selecting an unavailable SimGrid backend raises an error; there is no fallback.

All backends retain per-hop store-and-forward and independent full-duplex
directions. In SimGrid mode, OrbitServe converts bits/s to native bytes/s, uses
CM02 with zero native link latency and disabled reverse cross-traffic, and adds
Orbit-derived propagation after native serialization. Thus propagation is charged
once. Payload work is scaled to microbytes to preserve analytical first-token
sizes; the quantization is declared in result provenance. Shared transmission
completion times are not booked in advance. New contenders, completions and
capacity updates change active transfers. Zero capacity suspends and later
resumes their remaining bytes; contact loss cancels the hop and fails its request.
No packet loss or retransmission is fabricated.

Native engines run in isolated subprocesses, one per Runtime evaluation, so
repeated experiments and DSE do not reuse SimGrid's process-global clock. The two
event engines stop at each native completion or next Runtime/environment event.
`bandwidth_update` audit events report observed byte progress over a stated
interval, including allocated bandwidth, link capacity and contender count.
Route/compute selection in fluid, SimGrid and ns-3 uses currently observed link
and RF-pool contenders; ns-3 forecasts additionally charge packet padding/PPP.
These forecasts assume the observed contenders persist, without predicting their
future completion or exact packet order. Actual completion comes from the selected
transport. There is no end-to-end multi-hop fluid flow or implicit ground NIC pool.
Serving collective latency remains analytical with physical connectivity checks.
Cell operator execution and collective scheduling belong to Serving/AsterMind;
the network layer supplies physical topology and calculated link properties.

Serving retains its analytical model by default. Optional `serving.external_profile`
accepts `orbitserve.astermind_measurements.v1`: matching `model_name`, resolved
`hardware`, `formation`, `parallelism`, scope
`phase_latency_including_cell_communication`, and `samples` covering both phases
at every configured concurrency. Each sample declares `phase`, token/context
`length`, `concurrency` and an embedded successful, versioned AsterMind `result`
with positive `metrics.total_latency_s`; decode samples require `decode_tokens: 1`.
Prefill rates are length/latency, decode rates are 1/latency. Measured phase times
already include internal communication, so no analytical communication term is
added again. A single total-latency result cannot substitute for these tables.
This is an import contract; OrbitServe does not build or execute AsterMind.

Link profiles support fixed capacities, RF free-space/Shannon budgets and an
optical Gaussian-beam/photon-budget model. Optical inputs include wavelength,
transmit power, both apertures and efficiencies, pointing/system losses, receiver
photons per bit and a terminal rate ceiling. RF inputs include channel width,
frequency, power, gains or effective apertures, thermal noise and losses. Rates
are net bit/s **per direction**; propagation is sampled physical distance divided
by light speed. Serialization, queueing and setup are separate delays. Public
[Starlink specifications](https://starlink.com/technology) are reference points,
not disclosure of all terminal parameters. Record undisclosed parameters as
assumptions; these engineering models are not calibrated vendor emulators.

Communication nodes may select a versioned `radio_profile`, independently of
Walker/TLE shell geometry. Public source editions are pinned in
`runtime/setup/radio_capacity.py`:

| Profile | Satellite user downlink | Satellite user uplink | RF feeder aggregate |
| --- | --- | --- | --- |
| `starlink_v2_mini_2024` | 96 Gbps | Unpublished in this edition | Unpublished in this edition |
| `starlink_v3_design_160_20261021` | 1,000 Gbps | 160 Gbps | 1,200 Gbps, conservatively shared across directions |
| `starlink_v3_progress_2025` | >1,000 Gbps (reported lower bound) | >200 Gbps (reported lower bound) | V3 hardware-page reference, separate source |

V2 Mini's downlink classification is corroborated by the
[June 2026 SpaceX prospectus](https://content.spacex.com/cms-assets/FINAL_Documents%20and%20Updates/SpaceX%20-%20EU%20Prospectus%20%28Approved%20by%20Bafin%29%20-%20June%205%2C%202026.pdf).
The V3 profile freezes the [dedicated V3 design page](https://starlink.com/en-ps/updates/starlink-version-3-satellites)
as retrieved on 2026-10-21. The [Network Update](https://starlink.com/networkupdate)
instead reports over 200 Gbps uplink; the prospectus lists 1,024 Gbps downlink.
Those editions are not silently combined. V3 capacities are design figures.
Constellation materialization supplies auditable `simulation_defaults` for each
edition. V2's runnable uplink estimate is 160/24 = 6.667 Gbps, inferred from the
[2024 report, printed p.61](https://starlink.com/public-files/starlinkProgressReport_2024.pdf#page=62),
not a directly published V2 uplink specification. The 2025-report edition uses
1,000/200 Gbps as explicit lower-bound scenario assumptions, not exact ceilings.
Unpublished directional feeder provisioning matches the user-side aggregates;
this is a traffic-sizing assumption, not measured gateway hardware capacity.

Versioned shells materialize four independent RF profiles: user uplink/downlink
and feeder uplink/downlink. Nodes reference them through `sgl_profiles`; explicit
`hardware_link_profiles` override the shell defaults. Representative Ku carriers
use 14.25 GHz / 62.5 MHz uplink and 12 GHz / 240 MHz downlink; E-band feeder
carriers use 83.5 GHz / 2 GHz and 73.5 GHz / 500 MHz. Selected carrier widths,
power, effective apertures, losses and receiver parameters are **engineering
assumptions**, not vendor-disclosed channel specifications. Per-link rates still
come from the distance-dependent RF budget and satellite shared-pool limits.
End-user speed statistics are not used as physical link capacity. Starlink's
2025 report also describes symmetric 10 Gbps Community Gateways; these are
not ordinary user terminals or a disclosure of every satellite feeder link.
Unversioned scenarios and their explicit directional profiles retain their behavior.
All numeric capacities use decimal SI bit/s. Advertised laser counts/rates do not
establish duplex accounting, power or aperture; optical budgets remain explicit.
The hardware references distinguish V2 Mini's three optical terminals from V3's
six. A selected profile rejects allocations requesting more optical terminals
than the referenced hardware. `optical_terminal_bandwidth_bps` optionally bounds
every incident optical link in both directions, including compute-access links;
its per-direction interpretation requires `terminal_assumptions`. RF feeder
antenna counts (three/four), user beams and simultaneous GSL peers are distinct
quantities. `sgl_terminals` remains an explicit scenario concurrency limit, not an
inferred number of beams. Hardware references and terminal assumptions are exported.

Ground nodes use `ground_kind: "gateway"` by default (RF feeder), or `"user"`
for user access. The matching satellite aggregate is shared across **all** its
GSLs: uplink means ground to satellite; downlink means satellite to ground.
Separate UL/DL pools permit simultaneous full duplex. Feeder capacity never
borrows user capacity or optical capacity. If a selected edition does not publish
a required aggregate, supply `radio_capacity_bps` with explicit
`capacity_assumptions`; an unspecified value produces an input error. Overrides
are recorded in snapshots. Unversioned existing engineering scenarios retain
their explicit link inputs.

Physical inputs accept `scenario_type: "compute_only"`, `"communication_only"`,
or `"mixed"`, checked against the configured satellite roles. Compute-only
constellations access ground nodes directly; communication-only inputs support
network diagnostics and route probes. The LLM experiment runner requires compute
cells. Synthetic sources support `walker_delta` and `dawn_dusk`. The latter
derives a circular first-order J2 sun-synchronous inclination and epoch mean-Sun
RAAN from `ltan_hours` (6 or 18); `plane_spread_deg` optionally spreads planes
around that center. It is an initial design without station keeping or an
eclipse-free guarantee.

TLE communication sources specify nominal `planes` and `satellites_per_plane`
for Grid/Honeycomb. Common-epoch angular fitting leaves missing positions empty;
misaligned satellites remain physical nodes without grid ISLs. Optional
`source.grid_positions` maps catalog numbers to `[plane, slot]` explicitly.
This prescribed-grid fit does not reproduce automatic multimodal shell/spacing
recovery. Compute TLE sources instead require `source.cell_members`, mapping
each cell name to ordered catalog numbers; every imported satellite must appear
exactly once. Observed per-member elements are preserved without formation offsets.

Synthetic multi-row compute cells use aligned relative eccentricity/inclination
vectors, so radial separation accompanies cross-track orbit-plane crossings.
The requested grid is an initial layout; each member then propagates independently
with SGP4, without a rigid offset or station-keeping controller. Initialization
provenance identifies this as `free_sgp4_ei_separated_mean_element_offsets_v2`.
Before link-budget preparation, a compute-only TEME pass checks pairwise clearance
on the configured physical time grid, including the final boundary. Unsafe inputs
fail with the cell, closest member pair, time and required separation. Full ITRS
validation remains in place. These sampled checks do not guarantee continuous-time
collision avoidance between boundaries. The separation design follows the
[relative e/i-vector approach](https://www.dlr.de/en/rb/research-operation/research-projects/flight-dynamics-navigation-and-orbital-sustainability/gnss-technology-and-navigation/tandem-x-autonomous-formation-flying-experiment-taff).

Each communication shell selects `topology_repair: "none"` (baseline) or
`"ronchetti_intra_orbit"` (method1). Baseline retains Grid/Honeycomb holes.
Method1 adds candidate links across consecutive same-orbit holes, including
zero-laser satellites, and subjects them to geometry, 100 km atmospheric clearance,
range, setup time and endpoint terminal limits. It adds no synthetic satellites
and does not bridge missing planes. This follows the patched hexGrid rule in
[Ronchetti et al., LEO-NET 2026](https://github.com/nsg-ethz/irregular-constellations);
applying that rule to Grid is an extension. Candidate grid assignments stay fixed
within a run while physical link availability varies over time.

Ground `terminal_count` limits simultaneous satellite peers. Satellite
`sgl_terminals` limits total ground associations; optional `max_user_links` and
`max_gateway_links` add independent service quotas (zero disables that service,
omitted means only the total quota applies). A ground user node represents one
modeled terminal, not every user in a region. These counts are separate from
RF capacity and hardware beam/antenna counts. `network.ground_matching` selects
the default `greedy` admission or ground-proposing `deferred_acceptance` with
satellite rejection and retries. `ground_access_policy` and
`satellite_access_policy` independently select `nearest_distance`,
`longest_available`, `highest_elevation`, or `highest_capacity` (the effective
two-direction bottleneck rate). Preferences are evaluated per allocation boundary;
this does not claim globally optimal throughput or traffic-load-aware association.

Compute formations select `egress_policy: "single"` (one specified member),
`"multi"` (the specified `ingress_members`), or `"free"` (all members).
The default `multi` preserves existing member lists. Materialized inputs store
these rules in `network.cell_egress_policies`; explicit links cannot bypass them.
Free access still consumes each member's actual terminals. Internal cell topology
and inter-cell connectivity remain independently configured.

An SGL profile may reference `uplink_profile` and `downlink_profile`, each naming
another nonrecursive link profile for its own RF power, frequency, apertures,
modem cap and capacity schedule. Geometry/setup stay on the parent profile.
The effective directional rate is bounded by its physical budget and the relevant
satellite pools. Snapshot `directions` contains these rates and pool IDs;
top-level `bandwidth_bps` is the parent budget. Exact AsterMind exports preserve
the directional details and aggregate resource definitions.
Asymmetric exports set the legacy symmetric `bandwidth_gbps_per_direction` to
null; consumers must read `bandwidth_gbps_by_direction` and its pool IDs.

Fluid uses max-min allocation across links and RF pools. SimGrid routes contain
native shared RF resources. ns-3 arbitrates whole native packets across GSLs
using exclusive RF pools, rotating between links; this is conservative when
slow links hold a pool. Queue mode reserves the link direction and all matching
RF pools for the whole message. These are deliberately different service models,
not a promise of identical latency or packet fairness.

`network/astermind_network.jsonl.gz` exports per-sample cell membership, physical
rank mappings, internal links, external satellite links and enabled direct ground
links, including inactive links and budget diagnostics. `export_cell_network`
is also available from `orbitserve.runtime`. Exact heterogeneous snapshots are
preserved. A usable static snapshot additionally supplies conservative uniform
`serving_formation_fragment` and `astermind_static_fragment` inputs (minimum
bandwidth and maximum propagation delay); these approximations are labeled and
never automatically replace Serving's existing analytical communication model.
Current AsterMind's uniform static network schema cannot express the full dynamic
graph; consuming exact dynamic snapshots requires an adapter on that side.

Optional `network.faults` entries contain `target_type` (`link` or `node`),
`target_id`, `start_s` and `end_s`, with optional `source` for audit attribution.
The default `network.fault_timing: "boundary"` requires onset and repair at
physical sample boundaries. Runtime learns outages at their onset;
`failure_policy: "retry_next_sample"` discards the incomplete transmission on
the interrupted hop. Completed upstream hops remain delivered, and the sending
node retains its buffered message. In boundary mode it immediately reroutes at
that same boundary, up to `max_transfer_retries`; unavailable routes wait for a
later boundary. The explicitly selected legacy `fault_timing: "continuous"`
retains arbitrary fault times and next-sample recovery for old experiments.
Link and relay failures can
use alternate paths; computation-node failures terminate affected work because
there is no model checkpoint recovery. The default policy remains `fail`.
Cancellation logs report discarded in-flight bytes and the retained buffer owner;
recovery logs identify the current node and the bytes sent on the replacement
route. Completed upstream hops are not retransmitted. The model uses whole-message
store-and-forward at each hop, without TCP packet acknowledgements. If the node
holding the only forwarding buffer fails, recovery reports that buffer loss.

`network.random_faults.nodes` and `.links` independently configure `enabled`,
integer `seed`, zero-based `start_step`, `end_step`, and `model`. Each group has
its own random stream; changing link parameters does not alter sampled node
failures. Legacy flat settings with `target_type` remain readable.

| Model | Healthy period / onset | Outage duration |
| --- | --- | --- |
| `per_step` | Bernoulli `failure_probability` per healthy target per boundary | Configurable `duration` distribution |
| `exponential` | Exponential with `mean_up_steps` | Exponential with `mean_down_steps` |
| `weibull` | Weibull `up_scale_steps` and `up_shape`; fresh healthy age after repair | Configurable `duration` distribution |
| `fixed_ratio` | Sample `ceil(target_count * ratio)` targets once | Prescribed `[start_step, end_step)` interval |

For stochastic onset, `[start_step, end_step)` is the eligible injection window.
Duration accepts `{"distribution":"fixed","steps":4}`,
`{"distribution":"uniform","min_steps":2,"max_steps":6}`,
`{"distribution":"exponential","mean_steps":4}`, or
`{"distribution":"lognormal","mean_steps":4,"sigma":0.6}`. Lognormal
`mean_steps` is the arithmetic mean before discretization; `sigma` is the
standard deviation in log space. Legacy `duration_steps` means fixed duration.
Continuous samples are rounded up to at least one step, so configured means
refer to the distributions before rounding. A target cannot fail again on its
repair boundary. Exponential/Weibull healthy periods start at the window's first
boundary and restart after each repair; independent components are assumed.
Weibull shape below/at/above 1 means decreasing/constant/increasing healthy-age
hazard, rather than an assumption of cumulative unrepaired hardware aging.

Explicit intervals and bounded-duration configurations must fit within the
horizon. Unbounded repair samples may exceed it: the sampled recovery time is
preserved and actual onset logs mark `right_censored: true`; no repair is
invented at simulation end and no sample is shortened or redrawn. For example,
onset at step 3 with duration 4 repairs at step 7 and lasts `4 * step_s` seconds.
Probabilities are per step, not per second; use `p = 1 - exp(-rate_per_second *
step_s)` when converting a constant hazard. These are component UP/DOWN models;
packet error probabilities and Sun-exclusion geometry are separate mechanisms.
Random targets are realized by Runtime and remain hidden from route forecasting
until observation. Actual outcomes are retained in provenance and execution logs;
actual fault events include start/end step, duration in steps and seconds.

The optional `runtime_control(controller)` context binds an in-process controller
to one experiment thread. It pauses at boundaries after scheduled faults and
physical changes, before business events. Commands execute on the engine thread.
Manual faults and administrative disconnections persist independently; restoring
a manual fault does not clear a simultaneous random or scheduled outage. The
local console uses this hook for its event-timeline workspace. It performs
authoritative physical/terminal checks before publishing connection changes.
When the failed hop is the initial ground upload, recovery reselects an available
access satellite while retaining the chosen compute destination. A ready backup
requires a separate allocated ground/space terminal; acquisition delays still
apply when the alternative link is not yet established. A fault immediately
releases its assigned terminals; the next geometry sample may assign them to a
different visible peer, which must wait its full setup time. Repairs are learned
only when observed. Unaffected contacts retain their acquisition clocks;
recovered contacts reacquire instead of inheriting an obsolete setup timestamp.

Fixed designs and DSE share the same evaluator and canonical Workload. Hardware
terminal counts/rates, cell member counts, request model names and gateway
coordinates must match across modules. DSE scans physical configuration paths and
supported routing strategies. Geometric coverage is a screening upper bound, not
an inferred Runtime latency or SLO result.

Outputs include the canonical `network/orbit_trace/`, a machine-readable
`network/network_result.json`, compressed position/link snapshots and timelines,
`runtime_events.jsonl.gz`, `runtime_requests.jsonl.gz` and
`experiment_result.json`. These are scientific audit data. The shared runner does
not create HTML, browser replay indexes or personal viewer data.

## Single-Satellite Cost Tool

```bash
python3 -m orbitserve.architecture.single_satellite_cost.cli \
  examples/single_satellite/h200_gpu_derived.json \
  --out results/single_satellite/h200_gpu_derived.json
```

## Physical data preparation

Physical preparation can publish a compact, reusable core independently of full
JSON snapshots and exchange exports. Combined experiments use the core directly;
standalone callers retain the full-export default. Parallel exports support
bounded time chunks and selectable compression levels. See
[physical data and optional exports](src/experiments/PHYSICAL_DATA.md) for commands, layout,
integrity checks and compatibility with existing data.

## Tests

See [the regression coverage map](tests/README.md) for preserved behavior and
intentional changes from the legacy Runtime/DSE interfaces.

```bash
python3 -m unittest discover -s tests/unit -t .
python3 -m unittest discover -s tests/integration -t .
python3 -m unittest discover -s tests/smoke -t .
python3 -m unittest discover -s tests/slow -t .
python3 -m unittest discover -s tests -t .
python3 -m ruff check src tests
# With the optional dependency installed, explicitly verify the native backend:
python3 -m unittest tests.unit.runtime.test_simgrid_transport \
  tests.integration.fixed_design.test_simgrid_runtime
python3 -m unittest tests.unit.runtime.test_radio_capacity \
  tests.integration.fixed_design.test_radio_sharing \
  tests.integration.fixed_design.test_network_acceptance
```

`tests/slow/` contains physical-network Runtime, prefill/decode concurrency and strategy-matrix tests. Full validation must include this suite; routine development may run the first three suites separately.

Real ISLSim invocation is an additional local integration suite, separate from
the four CI suites. With a local checkout and the native backends installed, run:

```bash
ORBITSERVE_ISLSIM_ROOT=/path/to/ISLSim python3 -m unittest discover -s tests/external/islsim -v
```

The local suite verifies that generated inputs are accepted by the actual model
and checks its results, source identity and Runtime integration. A missing model
fails this explicit local command; there is no substitute calculation.

The network acceptance case loads one canonical workload bundle and reuses one
real SGP4 trace from a small synthetic constellation across three routing policies and four
backends. It asserts chronological events, delivered/discarded byte conservation,
single-terminal failover, acquisition waiting, mid-transfer capacity changes and
competition. Native rows require the respective optional dependencies; skipped
rows are not native-backend validation. Dedicated RF tests check aggregate sharing
across distinct GSLs and independent uplink/downlink pools.
`test_network_time_consistency` independently reconstructs distance, propagation,
Friis/Shannon RF and optical photon budgets at every sample and selected interior
times. It also integrates a long transfer across geometry and exact capacity
boundaries. `test_time_varying_transport` observes real backend configuration calls
to verify updates during execution, rather than only checking precomputed frames.
Path records timestamp their distance at serialization completion; individual
service intervals retain their own sample time and distance. Positions are held
within samples, while configured capacity schedules retain exact event times.

## Directory Rules

- `results/` contains generated experiment outputs and is ignored.
- `local/` contains personal tools, inputs, results, tests and internal documents; it is ignored. Do not force-add it.
- `scripts/` is reserved for paper experiment orchestration after the methodology is frozen.
- `examples/` contains user-facing inputs, not test-generated artifacts.

## Input Reliability and Distribution Checks

Canonical request token counts, hardware counts, parallel degrees and Serving
length grids use integers. Request times and byte counts must be finite and
nonnegative. Fractional counts, booleans and NaN/Infinity are rejected before
allocating a native backend. Causal requests may arrive in any input order;
duplicate IDs, missing predecessors and cycles are rejected explicitly.

Physical attribute readers bound decompression to 64 MiB per block, validate
time and column indices, and return independent metadata values. Atomic artifact
writes use a separate temporary file per producer; overlapping writers publish
complete files, with the last successful publication taking effect.

The distribution check builds a source distribution and a wheel from it, installs that wheel
outside the source tree, imports all modules and verifies the pinned reference
data and ISLSim configuration CLI. To run this check from a Git checkout with
the core dependencies installed:

```bash
python -m pip install 'build==1.6.1'
python tests/scripts/check_distribution.py
```

The project license is pending maintainer selection. Before public release,
review dependency and bundled-data redistribution terms and the Git history
that will become visible. The optional ns-3 backend is
[GPL-2.0-only](https://pypi.org/project/ns3/3.44.post0/); the bundled IERS data
retains its own license and provenance. Passing software tests does not replace
validation of model assumptions and scientific reference results.
