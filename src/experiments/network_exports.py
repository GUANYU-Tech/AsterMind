"""Bounded parallel snapshot construction, encoding and independent file writes."""

import gzip
import json
import multiprocessing
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import perf_counter

from orbitserve.common import atomic_output
from orbitserve.runtime import export_cell_network

from .snapshot_store import SnapshotSource

_source = None


def _initialize(root):
    global _source
    _source = SnapshotSource(root, _share_static=True).__enter__()


def _json_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _chunk(job):
    number, times, level = job
    source = _source
    regular = {t: i for i, t in enumerate(source.regular_times)}
    frames, hashes = [], {}
    build_s = encode_s = 0.0
    exchange = f"exchange/part-{number:06d}.jsonl.gz"
    positions = f"positions/part-{number:06d}.jsonl.gz"
    with (
        atomic_output(source.root / exchange) as raw_exchange,
        atomic_output(source.root / positions) as raw_positions,
        gzip.GzipFile(
            filename="",
            fileobj=raw_exchange,
            mode="wb",
            mtime=0,
            compresslevel=level,
        ) as stream,
        gzip.GzipFile(
            filename="",
            fileobj=raw_positions,
            mode="wb",
            mtime=0,
            compresslevel=level,
        ) as node_stream,
    ):
        for t in times:
            started = perf_counter()
            frame = source.snapshot(t)
            build_s += perf_counter() - started
            started = perf_counter()
            stream.write(_json_bytes(export_cell_network(frame)) + b"\n")
            if t in regular:
                i = regular[t]
                next_t = source.regular_times[min(i + 1, len(regular) - 1)]
                frame["events_until_next_boundary"] = [
                    e for e in source.events if t < e["time_s"] <= next_t
                ]
                name = f"frames/frame-{i:06d}.json.gz"
                with (
                    atomic_output(source.root / name) as raw,
                    gzip.GzipFile(
                        filename="",
                        fileobj=raw,
                        mode="wb",
                        mtime=0,
                        compresslevel=level,
                    ) as destination,
                ):
                    destination.write(_json_bytes(frame))
                hashes[str((source.root / name).resolve())] = raw.sha256
                frames.append(
                    {
                        "time_s": t,
                        "artifact": name,
                        "node_count": len(frame["nodes"]),
                        "link_count": len(frame["links"]),
                    }
                )
                for node in frame["nodes"]:
                    node_stream.write(
                        _json_bytes(
                            {
                                "time_s": t,
                                "node_id": node["node_id"],
                                "role": node["role"],
                                "position_itrs_km": node["position_itrs_km"],
                            }
                        )
                        + b"\n"
                    )
            encode_s += perf_counter() - started
    hashes.update(
        {
            str((source.root / name).resolve()): out.sha256
            for name, out in ((exchange, raw_exchange), (positions, raw_positions))
        }
    )
    return {
        "number": number,
        "frames": frames,
        "hashes": hashes,
        "exchange": exchange,
        "positions": positions,
        "build_worker_s": build_s,
        "encode_worker_s": encode_s,
    }


def export_snapshots(root, *, workers=8, compression_level=1, chunk_samples=32):
    """Only small result manifests cross processes; outstanding jobs are bounded."""
    global _source
    if (
        type(workers) is not int
        or workers < 1
        or type(chunk_samples) is not int
        or chunk_samples < 1
    ):
        raise ValueError("invalid snapshot export worker/chunk count")
    if type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer from 0 to 9")
    root = Path(root).resolve()
    static = json.loads((root / "state/index.json").read_text(encoding="utf-8"))
    spec = static["network"]
    times = static["regular_times"]
    exchange = sorted(
        {
            *times,
            *(
                t
                for f in spec.get("faults", ())
                for t in (f["start_s"], f["end_s"])
                if times[0] <= t <= times[-1]
            ),
        }
    )
    jobs = [
        (i, exchange[offset : offset + chunk_samples], compression_level)
        for i, offset in enumerate(range(0, len(exchange), chunk_samples))
    ]
    actual = min(workers, len(jobs))
    started = perf_counter()
    results = []
    if actual == 1:
        _initialize(root)
        try:
            results = [_chunk(job) for job in jobs]
        finally:
            _source.close()
            _source = None
    else:
        with ProcessPoolExecutor(
            max_workers=actual,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(root,),
        ) as pool:
            iterator = iter(jobs)
            pending = {
                pool.submit(_chunk, job) for _, job in zip(range(actual * 2), iterator)
            }
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    job = next(iterator, None)
                    if job is not None:
                        pending.add(pool.submit(_chunk, job))
    results.sort(key=lambda r: r["number"])
    hashes = {k: v for result in results for k, v in result["hashes"].items()}
    # Concatenated gzip members are portable; this copies compressed bytes only.
    for kind, name in (
        ("exchange", "astermind_network.jsonl.gz"),
        ("positions", "node_positions.jsonl.gz"),
    ):
        with atomic_output(root / name) as output:
            for result in results:
                with (root / result[kind]).open("rb") as part:
                    shutil.copyfileobj(part, output, length=1024 * 1024)
        hashes[str((root / name).resolve())] = output.sha256
    return {
        "frames": [frame for result in results for frame in result["frames"]],
        "hashes": hashes,
        "timing_s": {
            "snapshot_export_wall_s": perf_counter() - started,
            "snapshot_build_worker_sum_s": sum(r["build_worker_s"] for r in results),
            "snapshot_encode_write_worker_sum_s": sum(
                r["encode_worker_s"] for r in results
            ),
        },
        "workers": actual,
        "compression_level": compression_level,
        "chunks": len(jobs),
    }


def export_network_artifacts(
    root, *, workers=8, compression_level=1, chunk_samples=32, _core_result=None
):
    """Generate optional legacy-compatible views from an already complete core."""
    from .physical_dataset import load_dataset, publish_exports

    root = Path(root).resolve()
    if (root / "physical_exports.json").exists():
        raise FileExistsError(
            "physical exports already exist; choose a fresh destination"
        )
    if _core_result is None:
        _, _, _core_result = load_dataset(root, verify_exports=False)
    started = perf_counter()
    result = export_snapshots(
        root,
        workers=workers,
        compression_level=compression_level,
        chunk_samples=chunk_samples,
    )
    hashes = result["hashes"]
    source = SnapshotSource(root)
    for name, rows in (
        ("link_events.jsonl.gz", source.events),
        (
            "link_attributes.jsonl.gz",
            (
                row
                for key in sorted(source.link_frames)
                for row in source.link_frames[key]
            ),
        ),
    ):
        with (
            atomic_output(root / name) as raw,
            gzip.GzipFile(
                filename="",
                fileobj=raw,
                mode="wb",
                mtime=0,
                compresslevel=compression_level,
            ) as stream,
        ):
            for row in rows:
                stream.write(_json_bytes(row) + b"\n")
        hashes[str((root / name).resolve())] = raw.sha256
    artifacts = {
        "astermind_network": "astermind_network.jsonl.gz",
        "node_positions": "node_positions.jsonl.gz",
        "link_attributes": "link_attributes.jsonl.gz",
        "link_events": "link_events.jsonl.gz",
        "frames": result["frames"],
        "exchange_parts": [
            f"exchange/part-{i:06d}.jsonl.gz" for i in range(result["chunks"])
        ],
    }
    timing = {**result["timing_s"], "optional_exports_wall_s": perf_counter() - started}
    from orbitserve.common import write_json

    view = {
        **_core_result,
        "artifacts": {**_core_result["artifacts"], **artifacts},
        "timing_s": {**_core_result["timing_s"], **timing},
        "snapshots": result["frames"][:-1],
        "events": [],
        "candidate_links": [],
    }
    hashes[str((root / "network_result.json").resolve())] = write_json(
        root / "network_result.json", view
    )
    publish_exports(
        root,
        {
            "trace_id": _core_result["trace_id"],
            "artifacts": artifacts,
            "timing_s": timing,
        },
        hashes,
    )
    return {
        "artifacts": artifacts,
        "timing_s": timing,
        "execution": {k: result[k] for k in ("workers", "compression_level", "chunks")},
    }
