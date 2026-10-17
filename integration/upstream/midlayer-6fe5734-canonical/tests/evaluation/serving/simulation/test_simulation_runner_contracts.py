from types import SimpleNamespace
import time

import pytest
from evaluation.serving.simulation import simulation_runner

from evaluation.serving.simulation.simulation_runner import (
    _DSV3_PD_MTP_MAX_REQUESTS,
    _GLM_PD_MTP_MAX_REQUESTS,
    _serving_progress_reporter,
    _pd_mtp_allowed_max_requests,
)


def _args(model, hardware, deployment="pd_disagg", enable_expert_parallel=True):
    return SimpleNamespace(
        model=model,
        hardware_config=hardware,
        decode_hardware_config=hardware,
        deployment=deployment,
        prefill_tp_size=8,
        prefill_pp_size=1,
        decode_tp_size=8,
        enable_expert_parallel=enable_expert_parallel,
        additional_colocated_hardware_config=None,
    )


@pytest.mark.parametrize("max_requests", sorted(_GLM_PD_MTP_MAX_REQUESTS))
def test_glm_h200_pd_mtp_scope_accepts_sweep_capacities(max_requests):
    allowed = _pd_mtp_allowed_max_requests(
        _args("configs/model/glm-5.2-fp8.json", "configs/hardware/nvidia_h200.yaml"),
        target_tp8_scope=True,
    )

    assert max_requests in allowed
    assert allowed == _GLM_PD_MTP_MAX_REQUESTS


@pytest.mark.parametrize("max_requests", sorted(_GLM_PD_MTP_MAX_REQUESTS))
def test_glm_h200_pd_tp8_ep1_scope_accepts_sweep_capacities(max_requests):
    allowed = _pd_mtp_allowed_max_requests(
        _args(
            "configs/model/glm-5.2-fp8.json",
            "configs/hardware/nvidia_h200.yaml",
            enable_expert_parallel=False,
        ),
        target_tp8_scope=False,
    )

    assert max_requests in allowed
    assert allowed == _GLM_PD_MTP_MAX_REQUESTS


@pytest.mark.parametrize(
    "args",
    (
        _args("configs/model/kimi-k2.6.json", "configs/hardware/nvidia_h200.yaml"),
        _args("configs/model/glm-5.2-fp8.json", "configs/hardware/nvidia_a800.yaml"),
        _args("configs/model/glm-5.2-fp8.json", "configs/hardware/nvidia_h200.yaml", "colocated"),
    ),
)
def test_pd_mtp_capacity_pin_remains_for_other_scopes(args):
    assert _pd_mtp_allowed_max_requests(args, target_tp8_scope=True) == frozenset((1,))
    assert _pd_mtp_allowed_max_requests(args, target_tp8_scope=False) == frozenset((1,))


def test_kimi_h200_colocated_tp8_ep8_scope_matches_only_kimi_on_h200():
    scope = simulation_runner._is_kimi_h200_colocated_tp8_ep8_target_scope

    assert scope(_args("configs/model/kimi-k2.6.json",
                       "configs/hardware/nvidia_h200.yaml", "colocated"))


@pytest.mark.parametrize(
    "args",
    (
        # GLM stays gated out of the Kimi scope.
        _args("configs/model/glm-5.2-fp8.json",
              "configs/hardware/nvidia_h200.yaml", "colocated"),
        # H200-only: the A800 Kimi PP1/EP8 colocated layout is not validated.
        _args("configs/model/kimi-k2.6.json",
              "configs/hardware/nvidia_a800.yaml", "colocated"),
        # EP must be enabled (the scope is the pinned vLLM EP=TP layout).
        _args("configs/model/kimi-k2.6.json",
              "configs/hardware/nvidia_h200.yaml", "colocated",
              enable_expert_parallel=False),
        # PD keeps its own scope; the colocated predicate must not leak.
        _args("configs/model/kimi-k2.6.json",
              "configs/hardware/nvidia_h200.yaml"),
    ),
)
def test_kimi_h200_colocated_tp8_ep8_scope_rejects_other_layouts(args):
    assert not simulation_runner._is_kimi_h200_colocated_tp8_ep8_target_scope(
        args)


def test_dsv3_h200_aggregate_scope_matches_only_dsv3_colocated_ep1_h200():
    scope = simulation_runner._is_dsv3_h200_aggregate_target_scope

    assert scope(_args("configs/model/deepseek-v3.json",
                       "configs/hardware/nvidia_h200.yaml", "colocated",
                       enable_expert_parallel=False))


@pytest.mark.parametrize(
    "args",
    (
        # GLM and Kimi keep their own scopes; no leakage into DeepSeek-V3.
        _args("configs/model/glm-5.2-fp8.json",
              "configs/hardware/nvidia_h200.yaml", "colocated",
              enable_expert_parallel=False),
        _args("configs/model/kimi-k2.6.json",
              "configs/hardware/nvidia_h200.yaml", "colocated",
              enable_expert_parallel=False),
        # H200-only: no A800 DeepSeek-V3 aggregate layout is validated.
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_a800.yaml", "colocated",
              enable_expert_parallel=False),
        # EP must stay disabled (the scope is the TP8/PP1/EP1 aggregate).
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_h200.yaml", "colocated"),
        # PD keeps its own scope; the colocated predicate must not leak.
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_h200.yaml",
              enable_expert_parallel=False),
    ),
)
def test_dsv3_h200_aggregate_scope_rejects_other_layouts(args):
    assert not simulation_runner._is_dsv3_h200_aggregate_target_scope(args)


def test_dsv3_h200_pd_tp8_ep1_scope_matches_only_dsv3_pd_ep1_h200():
    scope = simulation_runner._is_dsv3_h200_pd_tp8_ep1_target_scope

    assert scope(_args("configs/model/deepseek-v3.json",
                       "configs/hardware/nvidia_h200.yaml",
                       enable_expert_parallel=False))


@pytest.mark.parametrize(
    "args",
    (
        # GLM and Kimi keep their own PD scopes.
        _args("configs/model/glm-5.2-fp8.json",
              "configs/hardware/nvidia_h200.yaml",
              enable_expert_parallel=False),
        _args("configs/model/kimi-k2.6.json",
              "configs/hardware/nvidia_h200.yaml",
              enable_expert_parallel=False),
        # H200-only on both roles.
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_a800.yaml",
              enable_expert_parallel=False),
        # The DeepSeek-V3 PD transfer contract is validated for EP1 only.
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_h200.yaml"),
        # Colocated keeps its own scope; the PD predicate must not leak.
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_h200.yaml", "colocated",
              enable_expert_parallel=False),
    ),
)
def test_dsv3_h200_pd_tp8_ep1_scope_rejects_other_layouts(args):
    assert not simulation_runner._is_dsv3_h200_pd_tp8_ep1_target_scope(args)


@pytest.mark.parametrize("max_requests", sorted(_DSV3_PD_MTP_MAX_REQUESTS))
def test_dsv3_h200_pd_mtp_scope_accepts_sweep_capacities(max_requests):
    allowed = _pd_mtp_allowed_max_requests(
        _args(
            "configs/model/deepseek-v3.json",
            "configs/hardware/nvidia_h200.yaml",
            enable_expert_parallel=False,
        ),
        target_tp8_scope=False,
    )

    assert max_requests in allowed
    assert allowed == _DSV3_PD_MTP_MAX_REQUESTS


@pytest.mark.parametrize(
    "args",
    (
        # The GLM/Kimi PD MTP capacity pins are unchanged by the DSV3 scope.
        _args("configs/model/glm-5.2-fp8.json",
              "configs/hardware/nvidia_h200.yaml", "colocated"),
        _args("configs/model/kimi-k2.6.json",
              "configs/hardware/nvidia_h200.yaml"),
        _args("configs/model/deepseek-v3.json",
              "configs/hardware/nvidia_a800.yaml",
              enable_expert_parallel=False),
    ),
)
def test_dsv3_pd_mtp_capacity_pin_remains_for_other_scopes(args):
    assert _pd_mtp_allowed_max_requests(args, target_tp8_scope=True) == frozenset((1,))
    assert _pd_mtp_allowed_max_requests(args, target_tp8_scope=False) == frozenset((1,))


def test_serving_progress_reporter_has_stoppable_heartbeat(capsys):
    reporter = _serving_progress_reporter(1, heartbeat_seconds=0.01)
    try:
        time.sleep(0.03)
        assert "serving-progress-heartbeat" in capsys.readouterr().err
    finally:
        reporter.stop()


def _marking_args(deployment, collective_backend):
    return SimpleNamespace(
        deployment=deployment,
        collective_backend=collective_backend,
        prefill_tp_size=8,
        prefill_pp_size=1,
        decode_tp_size=8,
        ffn_tp_size=None,
    )


@pytest.mark.parametrize(
    "deployment", ("af_disagg", "pd_disagg", "pd_af_disagg"))
def test_disagg_routes_mark_development_protobridge_with_analytical_collectives(
        deployment):
    # Disaggregated KV/activation routes always consume registered
    # development send_recv profiles, even when tensor collectives are
    # analytical.
    assert simulation_runner._uses_development_protobridge(
        _marking_args(deployment, "analytical_roofline_v1"))


def test_colocated_analytical_run_is_not_marked_development_protobridge():
    assert not simulation_runner._uses_development_protobridge(
        _marking_args("colocated", "analytical_roofline_v1"))


def test_h200_pd_route_profile_follows_requested_transfer_link():
    scale_out_profile, scale_out_digest = (
        simulation_runner._protobridge_p2p_profile(
            "nvidia_h200", "infiniband"))
    scale_up_profile, scale_up_digest = (
        simulation_runner._protobridge_p2p_profile("nvidia_h200", "nvlink"))

    assert scale_out_profile.link == "infiniband"
    assert scale_out_profile.source_kind == "development_only"
    assert scale_up_profile.link == "nvlink"
    assert scale_out_digest != scale_up_digest


def test_p2p_profile_resolution_fails_closed_without_registered_link():
    with pytest.raises(NotImplementedError):
        simulation_runner._protobridge_p2p_profile("metax_c500", "infiniband")


def test_h200_pipeline_p2p_protobridge_stays_on_scale_up_link():
    configs = simulation_runner._pipeline_p2p_configs(
        "prefill",
        2,
        hardware_id="nvidia_h200",
        backend="protobridge",
        tensor_parallel_size=1,
        ranks_per_node=8,
    )

    assert configs
    for config in configs:
        assert config.link == "nvlink"
        assert config.backend_config_sha256 == scale_up_digest()


def scale_up_digest():
    _profile, digest = simulation_runner._protobridge_p2p_profile(
        "nvidia_h200", "nvlink")
    return digest


def test_h200_launch_provenance_records_actual_proxy_source(tmp_path, monkeypatch):
    source = tmp_path / "launch_data" / "nvidia_a800"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text('{"synthetic": true}')
    monkeypatch.setattr(simulation_runner, "ROOT", tmp_path)
    provenance = simulation_runner._launch_data_provenance(("nvidia_h200",))
    assert provenance["nvidia_h200"]["path"] == str(source)
    (source / "manifest.json").unlink()
    with pytest.raises(RuntimeError, match="no selected files"):
        simulation_runner._launch_data_provenance(("nvidia_h200",))
    with pytest.raises(RuntimeError, match="does not exist"):
        simulation_runner._launch_data_provenance(("metax_c500",))


def _minimal_schema3_handoff_contract():
    request = {
        "request_id": "request-0",
        "target_output_tokens": 4,
        "prefill_emitted_tokens": 1,
        "decode_emitted_tokens": 3,
        "decode_raw_model_output_tokens": 3,
        "decode_raw_output_steps": 3,
        "decode_mtp_proposed_tokens": 3,
        "decode_mtp_verified_tokens": 3,
        "decode_mtp_accepted_tokens": 2,
        "decode_mtp_rejected_tokens": 1,
        "total_emitted_tokens": 4,
    }
    return {
        "schema_version": 3,
        "contract_id": "vllm_0_11_maximum_entropy_decode_fresh_mtp_pd_v1",
        "scheduler_pools": {"prefill": "prefill-0", "decode": "decode-0"},
        "prefill_mtp": "disabled",
        "decode_admission": {"computed_tokens": "full_prompt_length"},
        "kv_handoff_mode": "metadata_only_private_destination_kv",
        "kv_handoff_lanes": "one_request_scoped_transfer_per_prefill_rank",
        "decode_preemption_resume": (
            "local_prefix_recompute_after_handoff_kv_release"),
        "scheduler_reference": {"vllm_version": "0.11"},
        "decode_mtp": "maximum_entropy_fresh_state",
        "acceptance_assumption": {"request_stream_sha256": "0" * 64},
        "ingress_request_stream_sha256": "0" * 64,
        "decode_admission_contract_sha256": "1" * 64,
        "decode_arrival_time_binding": (
            "projected_transfer_completion_not_assumption_identity"),
        "validation_status": {"acceptance": "projected_explicit_assumption"},
        "requests": [request],
    }


def test_mtp_pd_handoff_aggregate_accepts_decode_preemption_resume_field():
    aggregate = simulation_runner._aggregate_mtp_pd_handoff(
        _minimal_schema3_handoff_contract())
    assert aggregate["decode_preemption_resume"] == (
        "local_prefix_recompute_after_handoff_kv_release")
    assert aggregate["aggregate_requests"]["request_count"] == 1


def test_mtp_pd_handoff_aggregate_rejects_unknown_fields():
    contract = _minimal_schema3_handoff_contract()
    contract["invented_field"] = "nope"
    with pytest.raises(ValueError, match="fields disagree"):
        simulation_runner._aggregate_mtp_pd_handoff(contract)


def test_versioned_h200_runtime_prepare_profile_is_explicit_only():
    with pytest.raises(ValueError, match="requires an explicit"):
        simulation_runner._host_runtime_prepare_profile_path("nvidia_h200", None)
    explicit = simulation_runner._host_runtime_prepare_profile_path(
        "nvidia_h200", "profile.json")
    assert str(explicit) == "profile.json"


def test_published_runtime_prepare_bindings_cannot_be_replaced():
    with pytest.raises(ValueError, match="only for hardware with versioned"):
        simulation_runner._host_runtime_prepare_profile_path(
            "nvidia_a800", "profile.json")
    assert simulation_runner._host_runtime_prepare_profile_path(
        "nvidia_a800", None).name == "nvidia_a800_eager.json"
