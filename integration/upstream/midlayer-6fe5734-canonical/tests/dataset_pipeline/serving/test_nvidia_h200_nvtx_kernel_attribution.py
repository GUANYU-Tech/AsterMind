"""Contract for H200 NVTX-range kernel attribution in the ShareGPT raw adapter."""

from pathlib import Path
import sqlite3

import pytest

from dataset_pipeline.serving_dataset import sharegpt_raw_adapter as adapter


_PID = 7 << 24
_TID = _PID + 11


def _database(path: Path, ranges, launches) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE StringIds(id INTEGER, value TEXT);"
        "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, globalTid INTEGER, "
        "text TEXT, textId INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, globalTid INTEGER, "
        "correlationId INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, "
        "correlationId INTEGER, globalPid INTEGER, demangledName INTEGER, "
        "gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER, "
        "blockZ INTEGER, staticSharedMemory INTEGER, dynamicSharedMemory INTEGER, "
        "registersPerThread INTEGER, streamId INTEGER, deviceId INTEGER);"
        "INSERT INTO StringIds VALUES (1, 'gemm_kernel');"
    )
    connection.executemany(
        "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, NULL)",
        [(start, end, _TID, text) for start, end, text in ranges])
    for correlation, (call_ns, kernel_start) in enumerate(launches, start=1):
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
            (call_ns, _TID, correlation))
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, 1, 4, 1, 1, 128, 1, 1, 0, 0, 32, 7, 0)",
            (kernel_start, kernel_start + 5, correlation, _PID))
    connection.commit()
    connection.close()
    return path


def _scheduler():
    raw = [{"iter_id": 0}, {"iter_id": 1}, {"iter_id": 2}]
    normalized = [
        {"step_id": "step-000000000000", "scheduled_requests": []},
        {"step_id": "step-000000000001",
         "scheduled_requests": [{"phase": "prefill"}]},
        {"step_id": "step-000000000002",
         "scheduled_requests": [{"phase": "decode"}]},
    ]
    return raw, normalized


def test_kernels_follow_their_launching_nvtx_step(tmp_path):
    database = _database(
        tmp_path / "nsys.sqlite",
        [(100, 200, "midlayer-op-step:1"), (200, 250, "midlayer-op-sample:1"),
         (300, 400, "midlayer-op-step:2")],
        # step 1 forward + sampler, an unattributed launch, step 2 forward.
        [(110, 1000), (210, 1010), (260, 1020), (310, 1030)],
    )
    raw, normalized = _scheduler()
    launches, timing = adapter._normalize_h200_profiler(
        database, raw, normalized, (0,))
    assert launches.status == "available"
    assert launches.coverage == "complete"
    steps = [record["step_id"] for record in timing.payload]
    assert steps == ["step-000000000001", "step-000000000001", "step-000000000002"]


def test_overlapping_nvtx_ranges_fail_closed(tmp_path):
    database = _database(
        tmp_path / "nsys.sqlite",
        [(100, 300, "midlayer-op-step:1"), (200, 400, "midlayer-op-step:2")],
        [(110, 1000)],
    )
    raw, normalized = _scheduler()
    with pytest.raises(ValueError, match="overlap"):
        adapter._normalize_h200_profiler(database, raw, normalized, (0,))
