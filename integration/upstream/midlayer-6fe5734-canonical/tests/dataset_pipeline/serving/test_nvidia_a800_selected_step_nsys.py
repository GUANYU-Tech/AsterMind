import json
import sqlite3

import pytest

from dataset_pipeline.serving_dataset.nvidia_a800_selected_step_nsys import (
    analyze_a800_selected_step_nsys,
)


def _write_fixture(path, trace_path, *, graph_rows=2):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER, text TEXT,
            textId INTEGER, globalTid INTEGER, endGlobalTid INTEGER
        );
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER,
            graphId INTEGER, graphExecId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER
        );
        """
    )
    connection.execute(
        "INSERT INTO StringIds VALUES (?, ?)", (1, "cudaGraphLaunch_v10000")
    )
    connection.execute(
        "INSERT INTO StringIds VALUES (?, ?)", (2, "cudaLaunchKernel")
    )
    for rank in range(2):
        global_pid = (100 + rank) << 24
        global_tid = global_pid + 7
        connection.execute("INSERT INTO PROCESSES VALUES (?, ?)", (global_pid, 100 + rank))
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?)",
            (10 + rank, 30 + rank, 59, "midlayer-step:vllm:7", None, global_tid, None),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            (12 + rank, 15 + rank, global_tid, 20 + rank, 1),
        )
        for offset, correlation in enumerate((30, 40, 50), start=16):
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
                (offset, offset + 1, global_tid, correlation + rank, 2),
            )
        if rank < graph_rows:
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?, ?, ?, ?, ?, ?)",
                (40 + rank, 60 + rank, rank, 20 + rank, global_pid, 2, 3),
            )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            (61 + rank, 65 + rank, rank, 30 + rank, global_pid),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
            (66 + rank, 68 + rank, rank, 40 + rank, global_pid),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?, ?, ?, ?)",
            (69 + rank, 70 + rank, rank, 50 + rank, global_pid),
        )
    connection.commit()
    connection.close()
    trace_path.write_text(
        json.dumps({
            "step_id": "vllm:7",
            "schedule_end_ns": 20,
            "update_start_ns": 55,
            "rank_observations": [
                {
                    "rank": rank,
                    "worker_execution_start_ns": rank,
                    "worker_execution_end_ns": 20 + rank,
                }
                for rank in range(2)
            ],
        }) + "\n",
        encoding="utf-8",
    )


def _write_node_fixture(path, trace_path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, eventType INTEGER, text TEXT,
            textId INTEGER, globalTid INTEGER, endGlobalTid INTEGER
        );
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            start INTEGER, end INTEGER, globalTid INTEGER,
            correlationId INTEGER, nameId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER,
            graphNodeId INTEGER, graphId INTEGER, shortName INTEGER,
            gridX INTEGER, gridY INTEGER, gridZ INTEGER,
            blockX INTEGER, blockY INTEGER, blockZ INTEGER,
            staticSharedMemory INTEGER, dynamicSharedMemory INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (
            start INTEGER, end INTEGER, deviceId INTEGER,
            correlationId INTEGER, globalPid INTEGER
        );
        """
    )
    connection.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        ((1, "cudaGraphLaunch_v10000"), (2, "cudaLaunchKernel"), (3, "gemm")),
    )
    for rank in range(2):
        global_pid = (100 + rank) << 24
        global_tid = global_pid + 7
        connection.execute("INSERT INTO PROCESSES VALUES (?, ?)", (global_pid, 100 + rank))
        connection.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?, ?, ?, ?)",
            (10 + rank, 30 + rank, 59, "midlayer-step:vllm:7", None, global_tid, None),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            (12 + rank, 15 + rank, global_tid, 20 + rank, 1),
        )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)",
            (16 + rank, 17 + rank, global_tid, 30 + rank, 2),
        )
        for start, end, node_id in ((40 + rank, 50 + rank, 1), (52 + rank, 60 + rank, 2)):
            connection.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (start, end, rank, 20 + rank, global_pid, node_id, 9, 3,
                 4, 1, 1, 128, 1, 1, 0, 16),
            )
        connection.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (61 + rank, 65 + rank, rank, 30 + rank, global_pid, None, None, 3,
             1, 1, 1, 32, 1, 1, 0, 0),
        )
    connection.commit()
    connection.close()
    trace_path.write_text(
        json.dumps({
            "step_id": "vllm:7",
            "schedule_end_ns": 20,
            "update_start_ns": 55,
            "rank_observations": [
                {
                    "rank": rank,
                    "worker_execution_start_ns": rank,
                    "worker_execution_end_ns": 20 + rank,
                }
                for rank in range(2)
            ],
        }) + "\n",
        encoding="utf-8",
    )


def test_extracts_cross_rank_graph_union(tmp_path):
    sqlite_path = tmp_path / "nsys.sqlite"
    trace_path = tmp_path / "scheduler.jsonl"
    _write_fixture(sqlite_path, trace_path)

    result = analyze_a800_selected_step_nsys(
        sqlite_path, trace_path, ["vllm:7"], 2
    )

    assert result["steps"][0]["cross_device_graph_task_union_ms"] == 0.000021
    assert result["schema_version"] == 3
    assert result["steps"][0]["cross_device_task_union_ms"] == 0.000031
    assert result["steps"][0]["exposed_device_task_union_ms"] == 0.000025
    assert result["steps"][0]["uncovered_serialized_service_ms"] == 0.000010
    assert result["steps"][0]["runtime_inter_task_gap_ms"] is None
    assert result["steps"][0]["ranks"][0]["device_task_counts"] == {
        "graph": 1,
        "kernel": 1,
        "memcpy": 1,
        "memset": 1,
    }


def test_extracts_node_traced_graph_kernel_groups(tmp_path):
    sqlite_path = tmp_path / "nsys.sqlite"
    trace_path = tmp_path / "scheduler.jsonl"
    _write_node_fixture(sqlite_path, trace_path)

    result = analyze_a800_selected_step_nsys(
        sqlite_path, trace_path, ["vllm:7"], 2
    )

    assert result["schema_version"] == 3
    assert result["cuda_graph_trace_mode"] == "node"
    step = result["steps"][0]
    assert step["cross_device_graph_task_union_ms"] == 0.000020
    assert step["cross_device_task_union_ms"] == 0.000025
    assert step["exposed_device_task_union_ms"] == 0.000024
    assert step["uncovered_serialized_service_ms"] == 0.000011
    assert step["ranks"][0]["device_task_counts"] == {
        "graph": 1,
        "graph_kernel_node": 2,
        "kernel": 3,
        "memcpy": 0,
        "memset": 0,
    }
    assert step["ranks"][0]["graph_kernel_groups"] == [{
        "name": "gemm",
        "launch": {
            "grid": [4, 1, 1],
            "block": [128, 1, 1],
            "static_shared_bytes": 0,
            "dynamic_shared_bytes": 16,
        },
        "count": 2,
        "additive_duration_ns": 18,
    }]


def test_fails_closed_when_graph_correlation_is_missing(tmp_path):
    sqlite_path = tmp_path / "nsys.sqlite"
    trace_path = tmp_path / "scheduler.jsonl"
    _write_fixture(sqlite_path, trace_path, graph_rows=1)

    with pytest.raises(ValueError, match="correlation is not unique"):
        analyze_a800_selected_step_nsys(
            sqlite_path, trace_path, ["vllm:7"], 2
        )
