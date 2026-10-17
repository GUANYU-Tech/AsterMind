import json
from pathlib import Path

import pytest

from dataset_pipeline.collective_selection.mccl_mctracer import (
    build_selection_observations,
    write_selection_observations,
)


OPERATION_CONTRACTS = {
    "all_reduce": {
        "dtype_name": "mccl_bfloat16",
        "in_place": True,
        "operation_name": "AllReduce",
        "reduction": "sum",
    },
    "all_gather": {
        "dtype_name": "int8_t",
        "in_place": False,
        "operation_name": "AllGather",
        "reduction": None,
    },
}


def _metadata(rank: int, operation: str = "all_reduce") -> dict[str, object]:
    contract = OPERATION_CONTRACTS[operation]
    return {
        "communication_library": "MCCL",
        "device": {
            "device_name": "MetaX C500",
            "hostname": "c500-host",
            "physical_device": rank,
        },
        "hardware_id": "metax_c500",
        "library_sha256": "1" * 64,
        "library_version": "2.16.5",
        "local_rank": rank,
        "observation_method": "mcTracer kernel symbol",
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
        "profiler_attach_wait_seconds": 30.0,
        "rank": rank,
        "rank_mapping": [0, 1],
        "run_id": "test-run",
        "runner_sha256": "2" * 64,
        "schema_version": 2,
        "source_commit": "3" * 40,
        "source_identity": {
            "kind": "audited_source_snapshot",
            "manifest_sha256": "4" * 64,
        },
        "topology": "single_node_metaxlink",
        "torch_version": "2.4.0+metax3.7.2.0",
    }


def _kernel(name: str, rank: int, grid_x: int = 2) -> dict[str, object]:
    return {
        "args": {
            "block": {"x": 512, "y": 1, "z": 1},
            "device_id": rank,
            "grid": {"x": grid_x, "y": 1, "z": 1},
            "name": f"_Z_mangled_{name}",
        },
        "dur": 12345,
        "name": name,
        "ph": "X",
        "ts": 67890,
    }


def _events(
    rank: int,
    *,
    operation: str = "all_reduce",
    second_algorithm: str = "RING",
    second_protocol: str | None = "LL",
) -> list[dict[str, object]]:
    setup = (
        "mcclKernel_AllGather_RING_LL_Sum_int8_t"
        "(mcclDevComm*, unsigned long, mcclWork*)"
    )
    barrier = (
        "mcclKernel_AllReduce_RING_LL_Sum_uint8_t"
        "(mcclDevComm*, unsigned long, mcclWork*)"
    )
    contract = OPERATION_CONTRACTS[operation]
    operation_name = contract["operation_name"]
    dtype_name = contract["dtype_name"]

    def target(algorithm: str, protocol: str | None) -> str:
        protocol_component = "" if protocol is None else f"_{protocol}"
        if operation == "all_gather" and algorithm == "FC2":
            return (
                f"mcclkernel_AllGather_FC2{protocol_component}_int8_t(fcinfo)"
            )
        if algorithm == "FC2":
            arguments = "fcinfo"
        elif algorithm in {"RING", "TREE"}:
            arguments = "mcclDevComm*, unsigned long, mcclWork*"
        else:
            raise ValueError(f"unsupported test algorithm: {algorithm!r}")
        return (
            f"mcclKernel_{operation_name}_{algorithm}{protocol_component}_Sum_"
            f"{dtype_name}({arguments})"
        )

    return [
        {"name": "process_name", "ph": "M"},
        _kernel(setup, rank),
        _kernel(setup, rank),
        _kernel(barrier, rank),
        _kernel(target("RING", "LL"), rank),
        _kernel(target(second_algorithm, second_protocol), rank),
    ]


def _write_run(
    path: Path,
    *,
    operation: str = "all_reduce",
    second_algorithm: str = "RING",
    second_protocol: str | None = "LL",
) -> None:
    path.mkdir()
    trace_dir = path / "tracer_out_20260826"
    trace_dir.mkdir()
    for rank in range(2):
        (path / f"rank-{rank}.json").write_text(
            json.dumps(_metadata(rank, operation)), encoding="utf-8"
        )
        (trace_dir / f"tracer_out-{100 + rank}.json").write_text(
            json.dumps({
                "traceEvents": _events(
                    rank,
                    operation=operation,
                    second_algorithm=second_algorithm,
                    second_protocol=second_protocol,
                )
            }),
            encoding="utf-8",
        )


def test_build_selection_observations_excludes_duration_and_setup(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    payload = build_selection_observations(run_dir)

    assert [value["algorithm"] for value in payload["observations"]] == [
        "ring",
        "ring",
    ]
    assert [value["protocol"] for value in payload["observations"]] == [
        "ll",
        "ll",
    ]
    assert [value["channels"] for value in payload["observations"]] == [None, None]
    assert len(payload["raw_files"]) == 4
    serialized = json.dumps(payload)
    assert '"dur"' not in serialized
    assert '"ts"' not in serialized


def test_preserves_unobservable_unsuffixed_fc2_protocol(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, second_algorithm="FC2", second_protocol=None)
    for rank in range(2):
        metadata_path = run_dir / f"rank-{rank}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["observations"][0]["message_bytes"] = 32768
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = build_selection_observations(run_dir)

    assert payload["observations"][1]["algorithm"] == "fc2"
    assert payload["observations"][1]["protocol"] is None


def test_parses_lowercase_all_gather_fc2_symbol(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        operation="all_gather",
        second_algorithm="FC2",
        second_protocol="LL",
    )
    for rank in range(2):
        metadata_path = run_dir / f"rank-{rank}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["observations"][0]["message_bytes"] = 32768
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = build_selection_observations(run_dir)

    assert payload["observations"][1]["algorithm"] == "fc2"
    assert payload["observations"][1]["protocol"] == "ll"


def test_normalizes_schema_v1_missing_attach_wait(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    for rank in range(2):
        metadata_path = run_dir / f"rank-{rank}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["schema_version"] = 1
        del metadata["profiler_attach_wait_seconds"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    payload = build_selection_observations(run_dir)

    assert payload["profiler_attach_wait_seconds"] is None


def test_rejects_incomplete_target_sequence(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    trace = run_dir / "tracer_out_20260826" / "tracer_out-101.json"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["traceEvents"].pop()
    trace.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="target count differs"):
        build_selection_observations(run_dir)


def test_rejects_cross_rank_selection_difference(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    trace = run_dir / "tracer_out_20260826" / "tracer_out-101.json"
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["traceEvents"][-1] = _kernel(
        "mcclKernel_AllReduce_TREE_LL_Sum_mccl_bfloat16"
        "(mcclDevComm*, unsigned long, mcclWork*)",
        1,
    )
    trace.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="differs across ranks"):
        build_selection_observations(run_dir)


def test_rejects_repeat_selection_difference(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir, second_algorithm="TREE")

    with pytest.raises(ValueError, match="differs across repeats"):
        build_selection_observations(run_dir)


def test_writer_refuses_to_overwrite(tmp_path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    output = write_selection_observations(run_dir)

    assert output == run_dir / "selection_observations.json"
    with pytest.raises(FileExistsError):
        write_selection_observations(run_dir)
