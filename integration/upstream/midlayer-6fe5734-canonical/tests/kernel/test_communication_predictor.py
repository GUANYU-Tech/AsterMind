from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.predictors.protobridge.adapter import (
    ProtoBridgeProfile,
    ProtoBridgeProtocolPolicy,
    ProtoBridgeTopology,
    development_profile_config_sha256,
    predict_synthetic_for_testing,
    resolve_registered_profile,
)
from src.kernel.predictors.protobridge.whitebox_surrogate_v2 import (
    Topology as UpstreamTopology,
)


def _all_reduce_transfer() -> PhysicalTransferConfig:
    return PhysicalTransferConfig(
        algorithm="ring",
        bytes=4096,
        direction="collective",
        link="nvlink",
        topology="ring",
        participants=(0, 1, 2, 3),
        operation="all_reduce",
        bytes_semantics="per_rank_input",
        ring_order=(0, 1, 2, 3),
        reduction="sum",
        backend="analytical_roofline_v1",
        resource_id="node0-nvlink",
    )


def _spec(
    transfer: PhysicalTransferConfig,
    hardware_id: str = "nvidia_a800",
) -> KernelSpec:
    return KernelSpec(
        KernelKind.COMMUNICATION,
        hardware_id,
        transfer=transfer,
    )


def _synthetic_model_parameters():
    return {
        "packet_bw_bytes_per_us": 1024,
        "ring_bw_bytes_per_us": 1024,
        "tree_bw_per_level_bytes_per_us": 1024,
        "nvls_bw_bytes_per_us": 1024,
        "fullmesh_bw_bytes_per_us": 1024,
        "link_latency_us": 0.5,
        "credit_bdp_ring_packets": 8,
        "credit_bdp_tree_packets": 8,
        "spray_chunk_bytes": 128,
        "bulk_chunk_bytes": 1024,
        "num_lanes": 1,
        "t_min_switch_us": 0.5,
        "fec_n": 2,
        "fec_k": 1,
        "fec_t": 0,
        "fec_symbol_bits": 1,
        "fec_encode_ns": 0,
        "fec_decode_ns": 0,
        "startup_us": 2.0,
        "nvls_startup_us": 2.0,
        "retry_source_bw_bytes_per_us": 1024,
        "retry_source_latency_us": 0.0,
        "wire_efficiency": 1.0,
    }


def _synthetic_profile(
    *,
    operation: str = "all_reduce",
    algorithm: str = "ring",
    credits: int = 16,
    surrogate_topology: ProtoBridgeTopology | None = None,
    hardware_id: str = "nvidia_a800",
    model_parameters=None,
    profile_version: str = "synthetic-interface-v1",
    provenance: str = (
        "unit-test synthetic profile; not hardware calibration"),
) -> ProtoBridgeProfile:
    if model_parameters is None:
        normalized_model_parameters = _synthetic_model_parameters()
    else:
        normalized_model_parameters = model_parameters
    if operation == "all_reduce":
        if algorithm == "fullmesh":
            topology = "fully_connected"
        elif algorithm == "tree":
            topology = "tree"
        elif algorithm in ("ring", "nvls", "pairwise", "send_recv"):
            topology = "ring"
        else:
            raise ValueError(f"unknown synthetic algorithm: {algorithm!r}")
        bytes_semantics = "per_rank_input"
        reduction = "sum"
    elif operation == "all_gather":
        if algorithm == "fullmesh":
            topology = "fully_connected"
        elif algorithm in ("ring", "tree", "nvls", "pairwise", "send_recv"):
            topology = "ring"
        else:
            raise ValueError(f"unknown synthetic algorithm: {algorithm!r}")
        bytes_semantics = "per_rank_input"
        reduction = None
    elif operation == "all_to_all":
        algorithm = "pairwise"
        topology = "fully_connected"
        bytes_semantics = "per_rank_input"
        reduction = None
    elif operation == "send_recv":
        algorithm = "send_recv"
        topology = "point_to_point"
        bytes_semantics = "payload"
        reduction = None
    else:
        raise ValueError(f"unsupported synthetic operation: {operation!r}")
    return ProtoBridgeProfile(
        config_sha256=None,
        profile_version=profile_version,
        provenance=provenance,
        hardware_id=hardware_id,
        source_kind="synthetic_test_only",
        link="nvlink",
        operation=operation,
        algorithm=algorithm,
        topology=topology,
        bytes_semantics=bytes_semantics,
        reduction=reduction,
        credits=credits,
        timeline_contention_contract="execution_timeline_owns_contention",
        model_parameters=normalized_model_parameters,
        protocol_policy=ProtoBridgeProtocolPolicy(
            64 * 1024, 4 * 1024 * 1024,
            2.0, 2.0, 3.0,
            0.5, 0.5, 0.5, ((2, 2048.0),)),
        surrogate_topology=surrogate_topology,
    )


def _protobridge_transfer(
    profile: ProtoBridgeProfile | None = None,
) -> PhysicalTransferConfig:
    if profile is None:
        normalized_profile = _synthetic_profile()
    else:
        normalized_profile = profile
    transfer = _all_reduce_transfer()
    if normalized_profile.algorithm == "ring":
        topology = "ring"
        ring_order = transfer.ring_order
    elif normalized_profile.algorithm in (
            "fc2", "fullmesh", "mesh", "pairwise"):
        topology = "fully_connected"
        ring_order = ()
    elif normalized_profile.algorithm == "tree":
        topology = "tree"
        ring_order = ()
    elif normalized_profile.algorithm == "send_recv":
        topology = "point_to_point"
        ring_order = ()
    else:
        raise AssertionError("validated synthetic algorithm disappeared")
    if normalized_profile.operation == "all_reduce":
        transfer = replace(
            transfer,
            algorithm=normalized_profile.algorithm,
            topology=topology,
            ring_order=ring_order,
        )
    elif normalized_profile.operation == "all_gather":
        transfer = replace(
            transfer,
            algorithm=normalized_profile.algorithm,
            operation="all_gather",
            topology=topology,
            ring_order=ring_order,
            reduction=None,
        )
    elif normalized_profile.operation == "all_to_all":
        transfer = replace(
            transfer,
            algorithm="pairwise",
            operation="all_to_all",
            topology="fully_connected",
            ring_order=(),
            reduction=None,
        )
    elif normalized_profile.operation == "send_recv":
        transfer = replace(
            transfer,
            algorithm="send_recv",
            direction="point_to_point",
            topology="point_to_point",
            participants=(0, 1),
            operation="send_recv",
            bytes_semantics="payload",
            source_rank=0,
            destination_rank=1,
            ring_order=(),
            reduction=None,
        )
    else:
        raise AssertionError(
            f"unhandled synthetic operation: {normalized_profile.operation!r}")
    return replace(
        transfer,
        link=normalized_profile.link,
        backend="protobridge",
        backend_config_sha256=normalized_profile.config_sha256,
    )


def test_analytical_roofline_prices_operation_specific_network_bytes():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    all_reduce = _all_reduce_transfer()
    all_gather = replace(
        all_reduce,
        operation="all_gather",
        reduction=None,
    )
    send_recv = replace(
        all_reduce,
        algorithm="send_recv",
        direction="point_to_point",
        topology="point_to_point",
        participants=(0, 1),
        operation="send_recv",
        bytes_semantics="payload",
        source_rank=0,
        destination_rank=1,
        ring_order=(),
        reduction=None,
    )
    all_to_all = replace(
        all_reduce,
        algorithm="pairwise",
        operation="all_to_all",
        topology="fully_connected",
        ring_order=(),
        reduction=None,
    )
    bandwidth = hardware.interconnect["nvlink_bw_bytes_s"]
    expected_seconds = {
        "all_reduce": 1.5 * all_reduce.bytes / bandwidth,
        "all_gather": 3.0 * all_gather.bytes / bandwidth,
        "send_recv": send_recv.bytes / bandwidth,
        "all_to_all": 0.75 * all_to_all.bytes / bandwidth,
    }

    for transfer in (all_reduce, all_gather, send_recv, all_to_all):
        prediction = predict(_spec(transfer), hardware, "midlayer")
        roofline = predict(_spec(transfer), hardware, "roofline")
        expected = expected_seconds[transfer.operation]
        assert prediction.t_total_s == pytest.approx(expected)
        assert roofline.t_total_s == pytest.approx(expected)
        assert prediction.t_launch_s == 0.0
        assert prediction.bottleneck_group == "communication"


def test_analytical_roofline_scales_linearly_with_payload():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    transfer = _all_reduce_transfer()

    base = predict(_spec(transfer), hardware, "midlayer")
    doubled = predict(
        _spec(replace(transfer, bytes=2 * transfer.bytes)),
        hardware,
        "midlayer",
    )

    assert doubled.t_total_s == pytest.approx(2.0 * base.t_total_s)


def test_a800_analytical_infiniband_send_recv_uses_ndr_line_rate():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    transfer = replace(
        _all_reduce_transfer(),
        algorithm="send_recv",
        direction="point_to_point",
        link="infiniband",
        topology="point_to_point",
        participants=(0, 8),
        operation="send_recv",
        bytes_semantics="payload",
        source_rank=0,
        destination_rank=8,
        ring_order=(),
        reduction=None,
        resource_id="infiniband:main:node-0-to-node-1",
    )

    prediction = predict(_spec(transfer), hardware, "midlayer")

    assert hardware.interconnect["infiniband_bw_bytes_s"] == 50e9
    assert prediction.t_total_s == pytest.approx(transfer.bytes / 50e9)


def test_analytical_all_to_all_is_an_endpoint_injection_lower_bound():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    transfer = replace(
        _all_reduce_transfer(),
        algorithm="pairwise",
        operation="all_to_all",
        topology="fully_connected",
        ring_order=(),
        reduction=None,
    )

    prediction = predict(_spec(transfer), hardware, "midlayer")

    assert prediction.t_total_s == pytest.approx(
        0.75 * transfer.bytes
        / hardware.interconnect["nvlink_bw_bytes_s"])


@pytest.mark.parametrize(
    ("hardware_id", "hardware_config"),
    (
        ("nvidia_a800", "configs/hardware/nvidia_a800.yaml"),
        ("metax_c500", "configs/hardware/metax_c500.yaml"),
        ("ascend_910b", "configs/hardware/ascend_910b.yaml"),
    ),
)
@pytest.mark.parametrize(
    "operation",
    ("all_reduce", "all_gather", "all_to_all", "send_recv"),
)
def test_protobridge_production_backend_fails_for_unregistered_profile(
        hardware_id, hardware_config, operation):
    hardware = HWSpec(hardware_config)
    profile = _synthetic_profile(
        operation=operation,
        hardware_id=hardware_id,
    )
    transfer = _protobridge_transfer(profile)

    for method in ("midlayer", "roofline"):
        with pytest.raises(
                NotImplementedError,
                match="no registered hardware profile"):
            predict(_spec(transfer, hardware_id), hardware, method)


@pytest.mark.parametrize(
    ("hardware_id", "hardware_config", "link", "algorithm", "ring_bw",
     "fullmesh_bw", "small", "simple"),
    (
        ("nvidia_a800", "configs/hardware/nvidia_a800.yaml", "nvlink", "ring",
         80_000.0, 25_000.0, 10.383, 23.859),
        ("metax_c500", "configs/hardware/metax_c500.yaml", "metaxlink", "fc2",
         76_800.0, 64_000.0, 6.392, 10.0),
        ("ascend_910b", "configs/hardware/ascend_910b.yaml", "hccs", "mesh",
         78_400.0, 28_000.0, 60.0, 60.0),
    ),
)
@pytest.mark.parametrize("operation", ("all_reduce", "all_gather"))
def test_registered_development_profiles_bind_hardware_operation_and_link(
    hardware_id, hardware_config, link, algorithm, ring_bw, fullmesh_bw,
    small, simple, operation,
):
    digest = development_profile_config_sha256(hardware_id, operation)
    profile = resolve_registered_profile(digest)
    hardware = HWSpec(hardware_config)

    transfer = _protobridge_transfer(profile)
    if algorithm == "fc2":
        transfer = replace(transfer, participants=(0, 1))
    prediction = predict(_spec(transfer, hardware_id), hardware, "midlayer")

    assert profile.source_kind == "development_only"
    assert profile.hardware_id == hardware_id
    assert profile.operation == operation
    assert profile.link == link
    assert profile.algorithm == algorithm
    assert profile.model_parameters["ring_bw_bytes_per_us"] == ring_bw
    assert profile.model_parameters["fullmesh_bw_bytes_per_us"] == fullmesh_bw
    assert profile.protocol_policy.small_startup_us == small
    assert profile.protocol_policy.simple_startup_us == simple
    assert prediction.t_total_s > 0.0


@pytest.mark.parametrize(
    ("hardware_id", "hardware_config", "link"),
    (
        ("nvidia_a800", "configs/hardware/nvidia_a800.yaml", "nvlink"),
        ("metax_c500", "configs/hardware/metax_c500.yaml", "metaxlink"),
        ("ascend_910b", "configs/hardware/ascend_910b.yaml", "hccs"),
    ),
)
def test_registered_development_send_recv_profiles(
        hardware_id, hardware_config, link):
    digest = development_profile_config_sha256(hardware_id, "send_recv")
    profile = resolve_registered_profile(digest)
    transfer = replace(
        _protobridge_transfer(profile),
        bytes=4096,
        participants=(0, 1),
        source_rank=0,
        destination_rank=1,
    )

    prediction = predict(
        _spec(transfer, hardware_id),
        HWSpec(hardware_config),
        "midlayer",
    )

    assert profile.operation == "send_recv"
    assert profile.algorithm == "send_recv"
    assert profile.topology == "point_to_point"
    assert profile.bytes_semantics == "payload"
    assert profile.link == link
    assert prediction.t_total_s > 0.0


def test_h200_scale_out_send_recv_profile_resolves_by_link():
    digest = development_profile_config_sha256(
        "nvidia_h200", "send_recv", link="infiniband")
    profile = resolve_registered_profile(digest)
    transfer = replace(
        _protobridge_transfer(profile),
        bytes=8 * 1024 * 1024,
        participants=(0, 1),
        source_rank=0,
        destination_rank=1,
    )

    prediction = predict(
        _spec(transfer, "nvidia_h200"),
        HWSpec("configs/hardware/nvidia_h200.yaml"),
        "midlayer",
    )

    assert profile.source_kind == "development_only"
    assert profile.link == "infiniband"
    assert profile.model_parameters["ring_bw_bytes_per_us"] == 160_000.0
    assert profile.model_parameters["wire_efficiency"] == 0.95
    assert profile.model_parameters["t_min_switch_us"] == 1.5
    assert profile.protocol_policy.simple_startup_us == 40.0
    assert prediction.t_total_s > 0.0


def test_h200_scale_out_profile_leaves_scale_up_resolution_unchanged():
    nvlink_profile = resolve_registered_profile(
        development_profile_config_sha256(
            "nvidia_h200", "send_recv", link="nvlink"))
    assert nvlink_profile.link == "nvlink"
    for operation in ("all_reduce", "all_gather"):
        profile = resolve_registered_profile(
            development_profile_config_sha256("nvidia_h200", operation))
        assert profile.link == "nvlink"


def test_h200_send_recv_resolution_requires_explicit_link():
    with pytest.raises(NotImplementedError, match="no unique development"):
        development_profile_config_sha256("nvidia_h200", "send_recv")


def test_h200_scale_up_record_reproduces_two_rank_nccl_anchors():
    reference = json.loads(Path(
        "microbench/nvidia_h200/references/"
        "nccl_ring_protocol_ws2_protobridge_inputs_sm90.json").read_text())
    digest = development_profile_config_sha256("nvidia_h200", "all_reduce")
    profile = resolve_registered_profile(digest)
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    anchors = [
        anchor for anchor in reference["startup_anchors"].values()
        if anchor["anchor_rule"] == "smallest_selected_size_at_n2"
    ]
    assert len(anchors) == 2
    for anchor in anchors:
        transfer = replace(
            _protobridge_transfer(profile),
            bytes=anchor["message_bytes"],
            participants=(0, 1),
            ring_order=(0, 1),
        )
        prediction = predict(_spec(transfer, "nvidia_h200"), hardware, "midlayer")
        assert prediction.t_total_s * 1e6 == pytest.approx(
            anchor["measured_us"], rel=1e-6)


@pytest.mark.parametrize(
    ("message_bytes", "protocol", "startup_us", "phase_fixed_us"),
    (
        (64 * 1024 - 1, "ll", 10.383, 2.590),
        (64 * 1024, "ll128", 25.738, 0.659),
        (4 * 1024 * 1024, "ll128", 25.738, 0.659),
        (4 * 1024 * 1024 + 1, "simple", 23.859, 6.321),
    ),
)
def test_development_profile_protocol_boundaries(
        message_bytes, protocol, startup_us, phase_fixed_us):
    profile = resolve_registered_profile(
        development_profile_config_sha256("nvidia_a800", "all_reduce"))

    assert profile.protocol_policy.select(message_bytes) == (
        protocol, startup_us)
    assert profile.protocol_policy.phase_fixed_for_protocol(protocol) == (
        phase_fixed_us)


@pytest.mark.parametrize(
    ("message_bytes", "startup_us", "phase_fixed_us"),
    (
        (64 * 1024 - 2, 10.383, 2.590),
        (64 * 1024, 10.383, 2.590),
        (3_932_160, 25.738, 0.659),
        (4 * 1024 * 1024, 23.859, 6.321),
    ),
)
def test_prediction_passes_protocol_selected_startup_to_surrogate(
        monkeypatch, message_bytes, startup_us, phase_fixed_us):
    profile = resolve_registered_profile(
        development_profile_config_sha256("nvidia_a800", "all_reduce"))
    transfer = replace(_protobridge_transfer(profile), bytes=message_bytes)
    captured = {}

    def fake_predict(_surrogate, **kwargs):
        captured.update(kwargs)
        captured["phase_fixed_us"] = _surrogate.lat
        return 1.0

    monkeypatch.setattr(
        "src.kernel.predictors.protobridge.adapter."
        "TheoryDerivedSurrogate.predict",
        fake_predict,
    )

    predict(
        _spec(transfer), HWSpec("configs/hardware/nvidia_a800.yaml"),
        "midlayer")

    assert captured["startup_us"] == startup_us
    assert captured["phase_fixed_us"] == phase_fixed_us


@pytest.mark.parametrize(
    ("participants", "message_bytes", "expected_ring_bw"),
    (
        ((0, 1), 2 * 1024 * 1024, 86_000.0),
        ((0, 1, 2, 3), 8 * 1024 * 1024, 103_000.0),
        ((0, 1, 2, 3, 4, 5, 6, 7), 32 * 1024 * 1024, 113_000.0),
    ),
)
def test_a800_simple_allreduce_selects_internal_rank_bandwidth(
        monkeypatch, participants, message_bytes, expected_ring_bw):
    profile = resolve_registered_profile(
        development_profile_config_sha256("nvidia_a800", "all_reduce"))
    transfer = replace(
        _protobridge_transfer(profile),
        bytes=message_bytes,
        participants=participants,
        ring_order=participants,
    )
    captured = {}

    def fake_predict(surrogate, **kwargs):
        captured["ring_bw"] = surrogate.ring_bw
        return 1.0

    monkeypatch.setattr(
        "src.kernel.predictors.protobridge.adapter."
        "TheoryDerivedSurrogate.predict",
        fake_predict,
    )
    predict(
        _spec(transfer), HWSpec("configs/hardware/nvidia_a800.yaml"),
        "midlayer")

    assert captured["ring_bw"] == expected_ring_bw


def test_registered_profile_rejects_wrong_physical_link():
    profile = resolve_registered_profile(
        development_profile_config_sha256("metax_c500", "all_reduce"))
    transfer = replace(_protobridge_transfer(profile), link="hccs")

    with pytest.raises(
            NotImplementedError,
            match="unsupported ProtoBridge hardware/link pairing"):
        predict(
            _spec(transfer, "metax_c500"),
            HWSpec("configs/hardware/metax_c500.yaml"),
            "midlayer",
        )


def test_synthetic_protobridge_profile_maps_bytes_participants_credits_and_us():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    base_profile = _synthetic_profile()
    credit_profile = _synthetic_profile(credits=4)
    transfer = _protobridge_transfer(base_profile)

    base = predict_synthetic_for_testing(
        _spec(transfer), hardware, base_profile)
    two_participants = predict_synthetic_for_testing(
        _spec(replace(
            transfer,
            participants=(0, 1),
            ring_order=(0, 1),
        )),
        hardware,
        base_profile,
    )
    credit_limited = predict_synthetic_for_testing(
        _spec(_protobridge_transfer(credit_profile)),
        hardware,
        credit_profile,
    )

    assert base.t_total_s == pytest.approx(8.0e-6)
    assert two_participants.t_total_s == pytest.approx(5.0e-6)
    assert credit_limited.t_total_s == pytest.approx(11.0e-6)
    assert base.t_launch_s == 0.0
    assert base.t_per_group == {"communication": pytest.approx(8.0e-6)}
    assert base.bottleneck_group == "communication"


def test_synthetic_protobridge_profile_maps_physical_topology():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    topology = ProtoBridgeTopology(
        bisection_bw_bytes_per_us=512,
        hop_count=2.0,
    )
    profile = _synthetic_profile(surrogate_topology=topology)

    prediction = predict_synthetic_for_testing(
        _spec(_protobridge_transfer(profile)),
        hardware,
        profile,
    )

    assert prediction.t_total_s == pytest.approx(14.0e-6)


def test_protobridge_profile_sha_is_content_addressed_and_deterministic():
    base = _synthetic_profile()
    same = _synthetic_profile()
    changed_parameters = _synthetic_model_parameters()
    changed_parameters["ring_bw_bytes_per_us"] = 2048
    topology_a = _synthetic_profile(
        surrogate_topology=ProtoBridgeTopology(512, 2))
    topology_b = _synthetic_profile(
        surrogate_topology=ProtoBridgeTopology(512, 3))

    variants = (
        _synthetic_profile(credits=17),
        _synthetic_profile(model_parameters=changed_parameters),
        _synthetic_profile(operation="all_gather"),
        _synthetic_profile(operation="all_to_all"),
        _synthetic_profile(operation="send_recv"),
        topology_a,
        _synthetic_profile(profile_version="synthetic-interface-v2"),
        _synthetic_profile(provenance="different synthetic provenance"),
    )

    assert base.config_sha256 == same.config_sha256
    assert len(base.config_sha256) == 64
    assert all(
        variant.config_sha256 != base.config_sha256
        for variant in variants)
    assert topology_a.config_sha256 != topology_b.config_sha256


def test_protobridge_profile_rejects_caller_supplied_unrelated_sha():
    with pytest.raises(
            ValueError,
            match="config_sha256 does not match normalized profile content"):
        replace(_synthetic_profile(), config_sha256="a" * 64)


def test_protobridge_profile_prediction_content_is_immutable_after_build():
    topology = ProtoBridgeTopology(512, 2)
    profile = _synthetic_profile(surrogate_topology=topology)

    with pytest.raises(FrozenInstanceError):
        profile.credits = 4
    with pytest.raises(FrozenInstanceError):
        topology.hop_count = 3
    with pytest.raises(TypeError):
        profile.model_parameters["ring_bw_bytes_per_us"] = 2048
    with pytest.raises(
            TypeError,
            match="immutable ProtoBridgeTopology"):
        _synthetic_profile(
            surrogate_topology=UpstreamTopology(512, 2))


@pytest.mark.parametrize(
    ("operation", "upstream_collective", "upstream_algorithm",
     "upstream_data_bytes", "participant_count"),
    (
        ("all_reduce", "allreduce", "ring", 4096, 4),
        ("all_gather", "allgather", "ring", 16384, 4),
        ("all_to_all", "alltoall", "ring", 4096, 4),
        ("send_recv", "sendrecv", "ring", 4096, 2),
    ),
)
def test_synthetic_protobridge_maps_operation_and_byte_semantics(
    monkeypatch,
    operation,
    upstream_collective,
    upstream_algorithm,
    upstream_data_bytes,
    participant_count,
):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    profile = _synthetic_profile(operation=operation)
    transfer = _protobridge_transfer(profile)
    captured = {}

    def fake_predict(_surrogate, **kwargs):
        captured.update(kwargs)
        return 7.25

    monkeypatch.setattr(
        "src.kernel.predictors.protobridge.adapter."
        "TheoryDerivedSurrogate.predict",
        fake_predict,
    )

    prediction = predict_synthetic_for_testing(
        _spec(transfer), hardware, profile)

    assert prediction.t_total_s == pytest.approx(7.25e-6)
    assert captured["collective"] == upstream_collective
    assert captured["algo"] == upstream_algorithm
    assert captured["data_bytes"] == upstream_data_bytes
    assert captured["num_gpus"] == participant_count
    assert captured["credits"] == profile.credits
    assert captured["phase"] == "full"


@pytest.mark.parametrize(
    "operation",
    ("all_reduce", "all_gather", "all_to_all", "send_recv"),
)
def test_synthetic_protobridge_executes_supported_operations(operation):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    profile = _synthetic_profile(operation=operation)

    prediction = predict_synthetic_for_testing(
        _spec(_protobridge_transfer(profile)), hardware, profile)

    assert prediction.t_total_s > 0.0
    assert prediction.t_launch_s == 0.0


@pytest.mark.parametrize(
    ("operation", "expected_seconds"),
    (("all_reduce", 5.0e-6), ("all_gather", 6.5e-6)),
)
def test_synthetic_protobridge_executes_fullmesh_collectives(
        operation, expected_seconds):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    profile = _synthetic_profile(operation=operation, algorithm="fullmesh")

    prediction = predict_synthetic_for_testing(
        _spec(_protobridge_transfer(profile)), hardware, profile)

    assert prediction.t_total_s == pytest.approx(expected_seconds)
    assert prediction.t_launch_s == 0.0


def test_synthetic_protobridge_maps_fullmesh_to_upstream(monkeypatch):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    profile = _synthetic_profile(algorithm="fullmesh")
    captured = {}

    def fake_predict(_surrogate, **kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(
        "src.kernel.predictors.protobridge.adapter."
        "TheoryDerivedSurrogate.predict",
        fake_predict,
    )

    predict_synthetic_for_testing(
        _spec(_protobridge_transfer(profile)), hardware, profile)

    assert captured["collective"] == "allreduce"
    assert captured["algo"] == "fullmesh"


@pytest.mark.parametrize(
    ("hardware_id", "hardware_config", "operation", "algorithm",
     "participants", "message_bytes", "upstream_algorithm",
     "expected_startup_us"),
    (
        ("metax_c500", "configs/hardware/metax_c500.yaml", "all_reduce",
         "fc2", (0, 1), 4096, "fullmesh", 6.392),
        ("metax_c500", "configs/hardware/metax_c500.yaml", "all_gather",
         "fc2", (0, 1), 151_936, "fullmesh", 10.520),
        ("metax_c500", "configs/hardware/metax_c500.yaml", "all_reduce",
         "tree", (0, 1, 2, 3), 16_384, "tree", 6.392),
        ("metax_c500", "configs/hardware/metax_c500.yaml", "all_reduce",
         "tree", (0, 1, 2, 3), 16_777_216, "tree", 10.0),
        ("ascend_910b", "configs/hardware/ascend_910b.yaml", "all_reduce",
         "mesh", (0, 1, 2, 3), 4096, "fullmesh", 60.0),
        ("ascend_910b", "configs/hardware/ascend_910b.yaml", "all_gather",
         "mesh", (0, 1, 2, 3), 4096, "fullmesh", 60.0),
    ),
)
def test_vendor_collective_profiles_map_to_confirmed_upstream_algorithms(
    monkeypatch,
    hardware_id,
    hardware_config,
    operation,
    algorithm,
    participants,
    message_bytes,
    upstream_algorithm,
    expected_startup_us,
):
    profile = resolve_registered_profile(
        development_profile_config_sha256(
            hardware_id, operation, algorithm))
    transfer = replace(
        _protobridge_transfer(profile),
        bytes=message_bytes,
        participants=participants,
    )
    captured = {}

    def fake_predict(_surrogate, **kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(
        "src.kernel.predictors.protobridge.adapter."
        "TheoryDerivedSurrogate.predict",
        fake_predict,
    )

    predict(
        _spec(transfer, hardware_id),
        HWSpec(hardware_config),
        "midlayer",
    )

    assert captured["algo"] == upstream_algorithm
    assert captured["startup_us"] == expected_startup_us


def test_c500_fc2_profile_rejects_a_ring_selected_payload():
    profile = resolve_registered_profile(
        development_profile_config_sha256(
            "metax_c500", "all_reduce", "fc2"))
    transfer = replace(
        _protobridge_transfer(profile),
        bytes=48,
        participants=(0, 1),
    )

    with pytest.raises(
            NotImplementedError,
            match="selected='ring', profile='fc2'"):
        predict(
            _spec(transfer, "metax_c500"),
            HWSpec("configs/hardware/metax_c500.yaml"),
            "midlayer",
        )


def test_c500_tree_profile_executes_selected_four_rank_collective():
    profile = resolve_registered_profile(
        development_profile_config_sha256(
            "metax_c500", "all_reduce", "tree"))
    transfer = replace(
        _protobridge_transfer(profile),
        bytes=16_384,
        participants=(0, 1, 2, 3),
    )

    prediction = predict(
        _spec(transfer, "metax_c500"),
        HWSpec("configs/hardware/metax_c500.yaml"),
        "midlayer",
    )

    assert prediction.t_total_s > 0.0


def test_synthetic_protobridge_rejects_transfer_profile_operation_mismatch():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    profile = _synthetic_profile(operation="all_gather")
    transfer = replace(
        _protobridge_transfer(profile),
        operation="all_reduce",
        reduction="sum",
    )

    with pytest.raises(
            NotImplementedError,
            match="unsupported ProtoBridge operation contract"):
        predict_synthetic_for_testing(_spec(transfer), hardware, profile)


@pytest.mark.parametrize(
    "algorithm",
    ("nvls", "pairwise", "send_recv"),
)
def test_protobridge_profile_rejects_unaligned_algorithms(algorithm):
    with pytest.raises(
            NotImplementedError,
            match="unsupported ProtoBridge operation contract"):
        replace(_synthetic_profile(), algorithm=algorithm)


def test_protobridge_profile_rejects_incomplete_model_parameters():
    parameters = _synthetic_model_parameters()
    del parameters["startup_us"]

    with pytest.raises(
            ValueError,
            match="complete surrogate constructor contract"):
        _synthetic_profile(model_parameters=parameters)


@pytest.mark.parametrize("wire_efficiency", (0.0, 1.01))
def test_protobridge_profile_rejects_invalid_wire_efficiency(wire_efficiency):
    parameters = _synthetic_model_parameters()
    parameters["wire_efficiency"] = wire_efficiency

    with pytest.raises(ValueError, match="wire_efficiency"):
        _synthetic_profile(model_parameters=parameters)


def test_synthetic_protobridge_rejects_hash_and_hardware_mismatch():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    transfer = _protobridge_transfer()

    with pytest.raises(ValueError, match="does not match the resolved profile"):
        predict_synthetic_for_testing(
            _spec(replace(transfer, backend_config_sha256="a" * 64)),
            hardware,
            _synthetic_profile(),
        )
    with pytest.raises(
            NotImplementedError,
            match="unsupported ProtoBridge hardware/profile pairing"):
        metax_profile = _synthetic_profile(hardware_id="metax_c500")
        predict_synthetic_for_testing(
            _spec(_protobridge_transfer(metax_profile)),
            hardware,
            metax_profile,
        )


def _h200_tp8_all_reduce(implementation):
    return PhysicalTransferConfig(
        algorithm="ring",
        bytes=14336,
        direction="collective",
        link="nvlink",
        topology="ring",
        participants=tuple(range(8)),
        operation="all_reduce",
        bytes_semantics="per_rank_input",
        ring_order=tuple(range(8)),
        reduction="sum",
        backend="analytical_roofline_v1",
        resource_id="node0-nvlink",
        implementation=implementation,
    )


def test_sglang_all_reduce_uses_its_own_table_and_fails_closed_without_it():
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    sglang = _spec(_h200_tp8_all_reduce("sglang_custom_all_reduce_v2"),
                   "nvidia_h200")
    vllm = _spec(_h200_tp8_all_reduce(None), "nvidia_h200")
    if hardware.sglang_custom_all_reduce_service is None:
        with pytest.raises(NotImplementedError):
            predict(sglang, hardware, "midlayer")
    hardware.sglang_custom_all_reduce_service = {
        "max_payload_bytes": 16 * 1024 * 1024,
        "tables": {8: ((1024, 7000.0), (1 << 20, 9000.0))},
    }
    assert predict(sglang, hardware, "midlayer").t_total_s == pytest.approx(7e-6)
    assert predict(vllm, hardware, "midlayer").t_total_s == pytest.approx(
        hardware.custom_all_reduce_service_ns(8, 14336) * 1e-9)


def test_sglang_all_reduce_implementation_requires_nvlink_all_reduce():
    with pytest.raises(ValueError):
        replace(_h200_tp8_all_reduce("sglang_custom_all_reduce_v2"),
                operation="all_gather", reduction=None)


def test_sglang_fused_all_reduce_rmsnorm_uses_its_table_and_fails_closed():
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    fused = _spec(
        _h200_tp8_all_reduce("sglang_flashinfer_allreduce_residual_rmsnorm"),
        "nvidia_h200")
    hardware.sglang_flashinfer_allreduce_fusion_service = {
        "max_payload_bytes": 1 << 21,
        "tables": {8: ((1024, 6000.0), (1 << 20, 20000.0))},
    }
    assert predict(fused, hardware, "midlayer").t_total_s == pytest.approx(
        6e-6)
    hardware.sglang_flashinfer_allreduce_fusion_service = {
        "max_payload_bytes": 2048,
        "tables": {8: ((1024, 6000.0),)},
    }
    with pytest.raises(ValueError, match="measured only below"):
        predict(fused, hardware, "midlayer")
    hardware.sglang_flashinfer_allreduce_fusion_service = None
    with pytest.raises(NotImplementedError):
        predict(fused, hardware, "midlayer")
