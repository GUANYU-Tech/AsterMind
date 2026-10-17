from dataclasses import FrozenInstanceError
import inspect

import pytest

from src.kernel.launch_config.select_launch_config import (
    select_collective_launch,
)


HARDWARE_CONTRACTS = {
    "nvidia_a800": {
        "communication_library": "NCCL",
        "library_version": "2.27.3",
        "topology": "single_node_nvlink",
    },
    "metax_c500": {
        "communication_library": "MCCL",
        "library_version": "2.16.5",
        "topology": "single_node_metaxlink",
    },
    "ascend_910b": {
        "communication_library": "HCCL",
        "library_version": "8.5.1",
        "topology": "single_node_full_hccs",
    },
}

OPERATION_CONTRACTS = {
    "all_reduce": {"reduction": "sum", "in_place": True},
    "all_gather": {"reduction": None, "in_place": False},
}


def _request(
    hardware_id: str,
    operation: str,
    message_bytes: int,
    participant_count: int,
) -> dict[str, object]:
    hardware = HARDWARE_CONTRACTS[hardware_id]
    operation_contract = OPERATION_CONTRACTS[operation]
    return {
        "hardware_id": hardware_id,
        "communication_library": hardware["communication_library"],
        "library_version": hardware["library_version"],
        "operation": operation,
        "message_bytes": message_bytes,
        "participant_count": participant_count,
        "dtype": "bfloat16",
        "reduction": operation_contract["reduction"],
        "in_place": operation_contract["in_place"],
        "topology": hardware["topology"],
    }


def _select(
    hardware_id: str,
    operation: str,
    message_bytes: int,
    participant_count: int,
):
    request = _request(
        hardware_id, operation, message_bytes, participant_count)
    return select_collective_launch(**request)


@pytest.mark.parametrize(
    ("operation", "participants", "message_bytes", "protocol"),
    (
        ("all_reduce", 2, 1_179_648, "ll"),
        ("all_reduce", 2, 1_184_000, "ll"),
        ("all_reduce", 2, 1_184_002, "simple"),
        ("all_reduce", 2, 1_257_472, "simple"),
        ("all_reduce", 2, 1_269_760, "simple"),
        ("all_reduce", 2, 1_310_720, "simple"),
        ("all_reduce", 4, 1_572_864, "ll"),
        ("all_reduce", 4, 1_835_008, "ll128"),
        ("all_reduce", 4, 3_932_160, "ll128"),
        ("all_reduce", 4, 4_194_304, "simple"),
        ("all_reduce", 8, 2_359_296, "ll"),
        ("all_reduce", 8, 2_621_440, "ll128"),
        ("all_reduce", 8, 15_728_640, "ll128"),
        ("all_reduce", 8, 16_777_216, "simple"),
        ("all_gather", 2, 720_896, "ll"),
        ("all_gather", 2, 736_000, "ll"),
        ("all_gather", 2, 736_002, "simple"),
        ("all_gather", 2, 759_680, "simple"),
        ("all_gather", 2, 786_432, "simple"),
        ("all_gather", 4, 524_288, "ll"),
        ("all_gather", 4, 589_824, "simple"),
        ("all_gather", 8, 393_216, "ll"),
        ("all_gather", 8, 425_984, "ll128"),
        ("all_gather", 8, 1_179_648, "ll128"),
        ("all_gather", 8, 1_310_720, "simple"),
    ),
)
def test_a800_rules_bind_protocol_to_operation_cards_and_size(
    operation, participants, message_bytes, protocol,
):
    selection = _select(
        "nvidia_a800", operation, message_bytes, participants)

    assert selection.vendor_algorithm == "ring"
    assert selection.vendor_protocol == protocol
    assert selection.protocol_status == "selected"
    assert selection.evidence_run_ids


def test_a800_collective_source_rule_is_shared_by_launch_policies():
    assert "launch_policy" not in inspect.signature(
        select_collective_launch).parameters

    selections = tuple(
        _select("nvidia_a800", "all_gather", 759_680, 2)
        for _launch_policy in ("analytical_launch", "injected_launch")
    )

    assert selections[0] == selections[1]
    assert selections[0].vendor_protocol == "simple"


@pytest.mark.parametrize(
    ("operation", "participants", "message_bytes"),
    (
        ("all_reduce", 4, 1_600_000),
        ("all_reduce", 4, 4_000_000),
        ("all_reduce", 8, 2_400_000),
        ("all_reduce", 8, 16_000_000),
        ("all_gather", 4, 550_000),
        ("all_gather", 8, 400_000),
        ("all_gather", 8, 1_200_000),
    ),
)
def test_a800_unresolved_protocol_brackets_fail_loud(
    operation, participants, message_bytes,
):
    with pytest.raises(NotImplementedError, match="boundary remains bracketed"):
        _select("nvidia_a800", operation, message_bytes, participants)


@pytest.mark.parametrize(
    ("operation", "message_bytes", "algorithm", "protocol", "status"),
    (
        ("all_reduce", 24, "ring", "ll", "selected"),
        ("all_reduce", 32, "fc2", "ll", "selected"),
        ("all_reduce", 48, "ring", "ll", "selected"),
        ("all_reduce", 131_040, "fc2", "ll", "selected"),
        ("all_reduce", 131_072, "fc2", None, "unobservable"),
        ("all_reduce", 131_088, "ring", "ll", "selected"),
        ("all_reduce", 131_106, "ring", None, "unresolved"),
        ("all_gather", 8, "ring", "ll", "selected"),
        ("all_gather", 16, "fc2", "ll", "selected"),
        ("all_gather", 24, "ring", "ll", "selected"),
        ("all_gather", 131_056, "fc2", "ll", "selected"),
        ("all_gather", 131_072, "fc2", None, "unobservable"),
        ("all_gather", 131_074, "ring", "ll", "selected"),
        ("all_gather", 131_078, "ring", None, "unresolved"),
    ),
)
def test_c500_two_rank_rules_include_alignment(
    operation, message_bytes, algorithm, protocol, status,
):
    selection = _select("metax_c500", operation, message_bytes, 2)

    assert selection.vendor_algorithm == algorithm
    assert selection.vendor_protocol == protocol
    assert selection.protocol_status == status


@pytest.mark.parametrize(
    ("message_bytes", "algorithm", "protocol"),
    (
        (32_766, "tree", "ll"),
        (32_768, "ring", "ll"),
        (524_290, "ring", "ll"),
        (540_672, "ring", "simple"),
        (16_777_214, "ring", "simple"),
        (16_777_216, "tree", "simple"),
        (33_554_430, "tree", "simple"),
        (33_554_432, "ring", "simple"),
        (67_108_862, "ring", "simple"),
        (67_108_864, "tree", "simple"),
    ),
)
def test_c500_four_rank_all_reduce_nonmonotonic_algorithm_rule(
    message_bytes, algorithm, protocol,
):
    selection = _select("metax_c500", "all_reduce", message_bytes, 4)

    assert selection.vendor_algorithm == algorithm
    assert selection.vendor_protocol == protocol


def test_c500_four_rank_protocol_brackets_fail_loud():
    with pytest.raises(NotImplementedError, match="AllReduce LL/SIMPLE"):
        _select("metax_c500", "all_reduce", 524_292, 4)
    with pytest.raises(NotImplementedError, match="AllGather LL/SIMPLE"):
        _select("metax_c500", "all_gather", 155_650, 4)


def test_c500_four_rank_all_gather_protocol_rule():
    ll = _select("metax_c500", "all_gather", 155_648, 4)
    simple = _select("metax_c500", "all_gather", 159_744, 4)

    assert (ll.vendor_algorithm, ll.vendor_protocol) == ("ring", "ll")
    assert (simple.vendor_algorithm, simple.vendor_protocol) == (
        "ring", "simple")


@pytest.mark.parametrize("participants", (2, 4, 8))
def test_910b_all_reduce_has_exact_bfloat16_adjacent_boundary(participants):
    small = _select("ascend_910b", "all_reduce", 262_144, participants)
    large = _select("ascend_910b", "all_reduce", 262_146, participants)

    assert small.vendor_algorithm == "mesh"
    assert small.vendor_executor == "AllReduceMeshSmallCountExecutor"
    assert large.vendor_executor == "AllReduceMeshOpbaseLoopExecutor"
    assert small.vendor_protocol is None
    assert small.protocol_status == "unobservable"


@pytest.mark.parametrize("participants", (2, 4, 8))
def test_910b_all_gather_selects_mesh_opbase(participants):
    selection = _select(
        "ascend_910b", "all_gather", 128 * 1024 * 1024, participants)

    assert selection.vendor_algorithm == "mesh"
    assert selection.vendor_executor == "AllGatherMeshOpbaseExecutor"
    assert selection.vendor_protocol is None


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("library_version", "2.28.0", "supports only NCCL"),
        ("topology", "multi_node", "supports only"),
        ("dtype", "float16", "supports only bfloat16"),
        ("participant_count", 3, "supports only 2, 4, or 8 ranks"),
        ("message_bytes", 3, "divisible by 2"),
        ("message_bytes", 256 * 1024 * 1024, "128 MiB"),
    ),
)
def test_collective_rules_reject_unprofiled_a800_inputs(field, value, match):
    request = _request("nvidia_a800", "all_reduce", 4096, 2)
    request[field] = value

    with pytest.raises(ValueError, match=match):
        select_collective_launch(**request)


def test_collective_rule_rejects_wrong_operation_semantics():
    request = _request("metax_c500", "all_reduce", 4096, 2)
    request["reduction"] = None

    with pytest.raises(ValueError, match="requires sum"):
        select_collective_launch(**request)


def test_collective_selection_is_immutable():
    selection = _select("ascend_910b", "all_gather", 4096, 2)

    with pytest.raises(FrozenInstanceError):
        selection.vendor_algorithm = "ring"
