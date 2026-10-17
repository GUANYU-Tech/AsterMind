import json
from pathlib import Path

import pytest

from dataset_pipeline.collective_selection.nccl_profiler import (
    build_selection_observations,
    write_selection_observations,
)


OPERATION_CONTRACTS = {
    "all_reduce": {
        "datatype": "ncclBfloat16",
        "event_count": 32768,
        "event_name": "AllReduce",
        "in_place": True,
        "reduction": "sum",
    },
    "all_gather": {
        "datatype": "ncclInt8",
        "event_count": 65536,
        "event_name": "AllGather",
        "in_place": False,
        "reduction": None,
    },
}


def _metadata(rank: int, operation: str = "all_reduce") -> dict[str, object]:
    contract = OPERATION_CONTRACTS[operation]
    return {
        "communication_library": "NCCL",
        "cuda_runtime_version": "12.8",
        "device": {
            "device_name": "NVIDIA A800-SXM4-80GB",
            "device_uuid": f"GPU-{rank}",
            "driver_version": "535.183.06",
            "pci_bus_id": f"00000000:{rank:02X}:00.0",
            "physical_device": rank,
        },
        "hardware_id": "nvidia_a800",
        "library_version": "2.27.3",
        "local_rank": rank,
        "observation_method": "NCCL profiler plugin API v4",
        "observations": [
            {
                "dtype": "bfloat16",
                "in_place": contract["in_place"],
                "message_bytes": 65536,
                "operation": operation,
                "participant_count": 2,
                "rank": rank,
                "reduction": contract["reduction"],
                "repeat": repeat,
                "validation": "passed",
            }
            for repeat in range(2)
        ],
        "profiler_plugin_sha256": "1" * 64,
        "profiler_plugin_source_commit": "2" * 40,
        "rank": rank,
        "rank_mapping": ["GPU-0", "GPU-1"],
        "run_id": "test-run",
        "runner_sha256": "3" * 64,
        "schema_version": 1,
        "source_commit": "4" * 40,
        "topology": "single_node_nvlink",
        "torch_version": "2.8.0+cu128",
    }


def _events(
    rank: int, *, operation: str = "all_reduce", second_protocol: str = "LL"
) -> list[dict[str, object]]:
    result = []
    contract = OPERATION_CONTRACTS[operation]
    target_name = contract["event_name"]
    target_count = contract["event_count"]
    target_datatype = contract["datatype"]
    specifications = [
        ("AllGather", 0, 8, "ncclInt8", "LL", 1),
        ("AllReduce", 0, 1, "ncclFloat32", "LL", 1),
        (target_name, 1, target_count, target_datatype, "LL", 8),
        ("AllReduce", 2, 1, "ncclFloat32", "LL", 1),
        (target_name, 3, target_count, target_datatype, second_protocol, 8),
    ]
    for event_id, (name, sequence, count, datatype, protocol, channels) in enumerate(specifications):
        result.extend([
            {
                "args": {
                    "Algorithm": "RING",
                    "CommHash": 123,
                    "Count": count,
                    "Datatype": datatype,
                    "Protocol": protocol,
                    "Rank": rank,
                    "SeqNum": sequence,
                    "nChannels": channels,
                },
                "cat": "COLL",
                "id": event_id,
                "name": name,
                "ph": "b",
                "pid": 10 + rank,
                "tid": 1,
                "ts": float(event_id),
            },
            {
                "cat": "COLL",
                "id": event_id,
                "name": name,
                "ph": "e",
                "pid": 10 + rank,
                "tid": 1,
                "ts": float(event_id) + 0.5,
            },
        ])
    result.append({})
    return result


def _write_run(
    path: Path, *, operation: str = "all_reduce", second_protocol: str = "LL"
) -> None:
    path.mkdir()
    for rank in range(2):
        (path / f"rank-{rank}.json").write_text(
            json.dumps(_metadata(rank, operation)), encoding="utf-8"
        )
        (path / f"nccl-profile_123_{rank}.json").write_text(
            json.dumps(_events(
                rank, operation=operation, second_protocol=second_protocol
            )),
            encoding="utf-8",
        )


def test_build_selection_observations_excludes_barriers_and_timestamps(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    payload = build_selection_observations(run_dir)

    assert payload["observations"] == [
        {
            "algorithm": "ring",
            "channels": 8,
            "message_bytes": 65536,
            "operation": "all_reduce",
            "protocol": "ll",
            "repeat": repeat,
        }
        for repeat in range(2)
    ]
    assert len(payload["raw_files"]) == 4
    assert "ts" not in json.dumps(payload)


def test_rejects_cross_rank_selection_difference(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    profile = run_dir / "nccl-profile_123_1.json"
    events = json.loads(profile.read_text(encoding="utf-8"))
    events[4]["args"]["Protocol"] = "SIMPLE"
    profile.write_text(json.dumps(events), encoding="utf-8")

    with pytest.raises(ValueError, match="differ across ranks"):
        build_selection_observations(run_dir)


def test_matches_byte_counted_all_gather_after_initialization(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, operation="all_gather")

    payload = build_selection_observations(run_dir)

    assert [observation["operation"] for observation in payload["observations"]] == [
        "all_gather",
        "all_gather",
    ]
    assert [observation["message_bytes"] for observation in payload["observations"]] == [
        65536,
        65536,
    ]


def test_rejects_repeat_selection_difference(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, second_protocol="SIMPLE")

    with pytest.raises(ValueError, match="differs across repeats"):
        build_selection_observations(run_dir)


def test_rejects_profile_event_without_protocol(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    profile = run_dir / "nccl-profile_123_0.json"
    events = json.loads(profile.read_text(encoding="utf-8"))
    del events[4]["args"]["Protocol"]
    profile.write_text(json.dumps(events), encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys.*Protocol"):
        build_selection_observations(run_dir)


@pytest.mark.parametrize(
    "operation,field,value,error",
    (
        ("all_reduce", "reduction", None, "reduction differs"),
        ("all_reduce", "in_place", False, "in_place differs"),
        ("all_gather", "reduction", "sum", "reduction differs"),
        ("all_gather", "in_place", True, "in_place differs"),
    ),
)
def test_rejects_metadata_collective_semantics(
        tmp_path, operation, field, value, error):
    run_dir = tmp_path / "run"
    _write_run(run_dir, operation=operation)
    metadata = run_dir / "rank-0.json"
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    payload["observations"][0][field] = value
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        build_selection_observations(run_dir)


def test_writer_refuses_to_overwrite(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    output = write_selection_observations(run_dir)

    assert output == run_dir / "selection_observations.json"
    with pytest.raises(FileExistsError):
        write_selection_observations(run_dir)
