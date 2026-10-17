import json
from pathlib import Path

import pytest

from dataset_pipeline.collective_selection.hccl_info import (
    build_selection_observations,
    write_selection_observations,
)


OPERATION_CONTRACTS = {
    "all_reduce": {
        "in_place": True,
        "operation_name": "AllReduce",
        "reduction": "sum",
        "selection_suffix": ".",
    },
    "all_gather": {
        "in_place": False,
        "operation_name": "AllGather",
        "reduction": None,
        "selection_suffix": ", current mode is [1]",
    },
}


def _metadata(rank: int, operation: str = "all_reduce") -> dict[str, object]:
    contract = OPERATION_CONTRACTS[operation]
    return {
        "cann_version_file_sha256": "1" * 64,
        "communication_library": "HCCL",
        "device": {
            "device_name": "Ascend 910B2",
            "hostname": "ascend-host",
            "physical_device": rank,
        },
        "driver_version": "25.2.3",
        "driver_version_file_sha256": "2" * 64,
        "hardware_id": "ascend_910b",
        "library_sha256": "3" * 64,
        "library_version": "8.5.1",
        "local_rank": rank,
        "log_environment": {
            "ASCEND_GLOBAL_LOG_LEVEL": "1",
            "ASCEND_PROCESS_LOG_PATH": f"/snapshot/{operation}-run/hccl_logs",
            "ASCEND_SLOG_PRINT_TO_STDOUT": "0",
        },
        "observation_method": "HCCL INFO selection log",
        "observations": [
            {
                "dtype": "bfloat16",
                "in_place": contract["in_place"],
                "message_bytes": 1024,
                "operation": operation,
                "participant_count": 2,
                "rank": rank,
                "reduction": contract["reduction"],
                "repeat": repeat,
                "validation": "passed",
            }
            for repeat in range(2)
        ],
        "process_id": 100 + rank,
        "rank": rank,
        "rank_mapping": [0, 1],
        "run_id": f"{operation}-run",
        "runner_sha256": "4" * 64,
        "schema_version": 1,
        "source_commit": "5" * 40,
        "source_identity": {
            "kind": "audited_source_snapshot",
            "manifest_sha256": "6" * 64,
        },
        "topology": "single_node_full_hccs",
        "torch_npu_version": "2.9.0.post1",
        "torch_version": "2.9.0+cpu",
    }


def _selection_line(
    process_id: int, thread_id: int, operation: str, executor: str
) -> str:
    contract = OPERATION_CONTRACTS[operation]
    operation_name = contract["operation_name"]
    suffix = contract["selection_suffix"]
    return (
        f"[INFO] HCCL({process_id},python3):timestamp "
        f"[operator.cc:1] [{thread_id}][SelectAlgfor910B] {operation_name} "
        f"SelectAlgfor910B is algName [{executor}]{suffix}"
    )


def _count_line(process_id: int, thread_id: int, count_bytes: int) -> str:
    return (
        f"[INFO] HCCL({process_id},python3):timestamp "
        f"[communicator.cc:1] [{thread_id}]NSLB-HCCL "
        f"algValue:[2],nslbAlg[1], count:[{count_bytes}]."
    )


def _log_lines(
    rank: int,
    *,
    operation: str = "all_reduce",
    second_executor: str | None = None,
) -> list[str]:
    process_id = 100 + rank
    thread_id = 200 + rank
    if operation == "all_reduce":
        default_executor = "AllReduceMeshSmallCountExecutor"
        barrier = [
            _selection_line(
                process_id,
                thread_id,
                "all_reduce",
                "AllReduceMeshSmallCountExecutor",
            ),
            _count_line(process_id, thread_id, 4),
        ]
        count_bytes = 1024
    else:
        default_executor = "AllGatherMeshOpbaseExecutor"
        barrier = []
        count_bytes = 2048
    if second_executor is None:
        second_executor = default_executor
    return [
        "unrelated log line with timestamp",
        *barrier,
        _selection_line(process_id, thread_id, operation, default_executor),
        _count_line(process_id, thread_id, count_bytes),
        _selection_line(process_id, thread_id, operation, second_executor),
        _count_line(process_id, thread_id, count_bytes),
    ]


def _write_run(
    path: Path,
    *,
    operation: str = "all_reduce",
    second_executor: str | None = None,
) -> None:
    path.mkdir()
    log_dir = path / "hccl_logs" / "debug" / "plog"
    log_dir.mkdir(parents=True)
    for rank in range(2):
        (path / f"rank-{rank}.json").write_text(
            json.dumps(_metadata(rank, operation)), encoding="utf-8"
        )
        (log_dir / f"plog-{100 + rank}_timestamp.log").write_text(
            "\n".join(_log_lines(
                rank,
                operation=operation,
                second_executor=second_executor,
            )) + "\n",
            encoding="utf-8",
        )


def test_build_selection_observations_ignores_timestamps_and_barrier(tmp_path):
    run_dir = tmp_path / "all_reduce-run"
    _write_run(run_dir)

    payload = build_selection_observations(run_dir)

    assert payload["observations"] == [
        {
            "algorithm": "mesh",
            "algorithm_value": 2,
            "channels": None,
            "message_bytes": 1024,
            "mode": None,
            "nslb_algorithm": 1,
            "operation": "all_reduce",
            "protocol": None,
            "repeat": repeat,
            "vendor_executor": "AllReduceMeshSmallCountExecutor",
        }
        for repeat in range(2)
    ]
    assert len(payload["raw_files"]) == 4
    assert "timestamp" not in json.dumps(payload["observations"])


def test_matches_all_gather_output_byte_count(tmp_path):
    run_dir = tmp_path / "all_gather-run"
    _write_run(run_dir, operation="all_gather")

    payload = build_selection_observations(run_dir)

    assert [value["message_bytes"] for value in payload["observations"]] == [
        1024,
        1024,
    ]
    assert [value["mode"] for value in payload["observations"]] == [1, 1]


def test_rejects_cross_rank_selection_difference(tmp_path):
    run_dir = tmp_path / "all_reduce-run"
    _write_run(run_dir)
    log = run_dir / "hccl_logs/debug/plog/plog-101_timestamp.log"
    contents = log.read_text(encoding="utf-8").replace(
        "AllReduceMeshSmallCountExecutor",
        "AllReduceMeshOpbaseLoopExecutor",
    )
    log.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="differs across ranks"):
        build_selection_observations(run_dir)


def test_rejects_repeat_selection_difference(tmp_path):
    run_dir = tmp_path / "all_reduce-run"
    _write_run(run_dir, second_executor="AllReduceMeshOpbaseLoopExecutor")

    with pytest.raises(ValueError, match="differs across repeats"):
        build_selection_observations(run_dir)


def test_rejects_unpaired_selection(tmp_path):
    run_dir = tmp_path / "all_reduce-run"
    _write_run(run_dir)
    log = run_dir / "hccl_logs/debug/plog/plog-101_timestamp.log"
    lines = log.read_text(encoding="utf-8").splitlines()
    log.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lack count records"):
        build_selection_observations(run_dir)


def test_writer_refuses_to_overwrite(tmp_path):
    run_dir = tmp_path / "all_reduce-run"
    _write_run(run_dir)

    output = write_selection_observations(run_dir)

    assert output == run_dir / "selection_observations.json"
    with pytest.raises(FileExistsError):
        write_selection_observations(run_dir)
