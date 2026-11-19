# Physical data and optional parallel exports

Physical preparation now publishes a reusable core before optional snapshot and
exchange exports. Combined experiments use this core directly. Physical equations,
sample-hold behavior, directional budgets and causal terminal allocation are unchanged.

```bash
# Physics only; no Runtime benchmark or simulation is started.
python -m astermind.experiments.physical_cli compute network.json \
  --out results/physical --workers 15

# Optional views can be produced later without propagating or selecting links again.
python -m astermind.experiments.physical_cli export results/physical \
  --workers 8 --compression-level 1 --chunk-samples 4
```

`run_network_experiment(..., export_mode="core")` returns after publishing the core.
Its default `export_mode="full"` preserves standalone callers that expect full
snapshots. `export_network_artifacts` provides the second command as a Python API.
Existing complete exports are not overwritten; use a fresh core-only destination
through `reuse_dataset(..., include_exports=False)` for compression/worker comparisons.

## Core layout

* `physical_dataset.json`: version 2 manifest, hashes and input identity; atomic
  publication is the completion marker.
* `core_result.json`: immutable core result index. `network_result.json` is a
  compatible result view, updated when optional exports complete.
* `orbit_trace/`: existing geometry manifest and windows, plus an ephemeris index
  and bounded numeric `.npy` shards. Reconstructed values and semantic `trace_id`
  match canonical JSON traces; the shard encoding has its own version.
* `physical_state.sqlite`: version 2 numeric-column attribute blocks and the initial
  causal allocation timeline. Constants and equal directional columns are shared
  inside blocks. SQLite has one writer; encoding/compression uses workers.
* `state/index.json`: static node/radio metadata, link identities, time index and
  position shard index. `state/timeline.json` contains the complete audit timeline.
* `state/positions/`: lossless float64 arrays split by time. Workers memory-map
  these read-only instead of copying the complete Orbit Trace or NetworkState.

The core contains all physical samples, link attributes and allocation results;
it does not require expanded JSON graphs. `SnapshotSource(root).snapshot(time_s)`
reconstructs a graph from a core already verified with `load_dataset`.

For repeated queries, use `with SnapshotSource(root) as source:`. This owns one
immutable read-only SQLite connection in the current thread, avoiding per-block
open/lock overhead when the attribute database exceeds its compressed cache.
SQLite can map up to 256 MiB of read-only file pages, shared by the operating
system, to avoid repeated small-page RPCs on mounted drives. This is an address
mapping limit, not an extra Python heap copy per worker; SQLite builds that
disable mapping retain the normal read path.
The context closes the connection and releases mapped position shards, including
on failure. Export workers use this scope independently in each process.

## Optional exports

Each worker constructs, encodes, compresses and writes a bounded time chunk. Only
file metadata, hashes and timing counters return to the parent. At most twice the
worker count of jobs is outstanding. Static radio parameters are prepared once.

`frames/` retains one complete legacy-compatible JSON snapshot per regular sample.
`exchange/` and `positions/` contain chunk files. Compatible all-time gzip streams
are assembled by concatenating compressed members without decoding or recompressing.
Hashes are calculated during writes. `physical_exports.json` is published only
after all export tasks succeed and lists all optional files and timings.

`load_dataset` verifies optional exports when present by default. Runtime uses
`verify_exports=False` and reuses only core files, so missing optional views do not
prevent simulation. Core corruption is always rejected. Version 1 physical
datasets, JSON Orbit traces and JSON attribute blocks remain readable under their
original integrity requirements.

Solar exclusion now evaluates the signal/Sun angle at each receiving endpoint
only when the Sun centre is visible past the opaque WGS-84 Earth. The shared
bidirectional link is excluded if either visible receiver violates its configured
angle. This remains an angular avoidance model, without solar noise power,
finite-disc or atmospheric modelling. New physical data records
`sun_geometry_version=receiver_visible_sun_wgs84_v1` and uses physics version
`sample_hold_directional_solar_visibility_v3`; older solar attributes must be
regenerated for new simulations. Historical results remain readable as recorded.
Physical scenes retain Sun ephemerides even when solar exclusion is disabled,
so illumination and direction can be displayed independently of outage policy.

Runtime faults default to continuous event time. Fractional failure/repair times
overlay this immutable sampled baseline; physical sampling does not delay fault
observation, route recovery or acquisition. Explicit `fault_timing=boundary`
preserves boundary-only experiments. Random models with `time_basis=seconds`
sample healthy/outage durations without rounding to physical samples. Historical
step-based inputs retain their original random-process semantics.

Compression levels 0–9 and positive chunk/worker counts are supported. Lower gzip
levels exchange a larger file size for less compression work. Select concurrency
from measured end-to-end time and memory use, rather than assuming linear speedup.
Worker build/encode CPU-stage wall times are reported as worker sums; they overlap
and must not be added to the parent's elapsed time. Core readiness and optional
export elapsed time are separate measurements.

The API defaults to eight workers and 32 samples per job.
