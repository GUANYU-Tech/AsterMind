"""Contract tests for the per-request Host request path mechanism."""

import json

import pytest

from src.serving.host_timing import (
    HostRequestPathProfile,
    load_host_request_path_profile,
)
from src.serving.metrics.serving_metrics import compute_serving_metrics
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.simulation import scheduler_visible_requests


_PROFILE_PATHS = {
    hardware_id: (
        "configs/host_timing/host_request_path_profiles/"
        f"{hardware_id}.json"
    )
    for hardware_id in ("nvidia_a800", "metax_c500", "ascend_910b")
}


def _profile(**overrides):
    values = dict(
        hardware_id="nvidia_a800",
        runtime_version="0.11.0",
        source_identity="test-source",
        host_cpu="test-cpu",
        api_entry_s=4.0e-4,
        tokenizer_async_call_s=2.2e-3,
        tokenizer_encode_intercept_s=3.0e-5,
        tokenizer_encode_per_token_s=1.8e-6,
        input_processor_intercept_s=5.0e-5,
        input_processor_per_token_s=8.0e-9,
        admission_enqueue_s=8.0e-6,
        engine_core_ipc_s=2.0e-5,
        first_token_delivery_s=3.0e-4,
        output_delivery_per_output_s=3.0e-5,
        profile_path="test",
        profile_sha256="0" * 64,
    )
    values.update(overrides)
    return HostRequestPathProfile(**values)


def test_admission_cost_is_affine_in_prompt_length():
    profile = _profile()
    short = profile.admission_service_time_s(1)
    long = profile.admission_service_time_s(1001)
    assert long - short == pytest.approx(1000 * (1.8e-6 + 8.0e-9))
    # Every fixed stage is charged exactly once whatever the prompt length.
    assert short == pytest.approx(
        4.0e-4 + 2.2e-3 + 3.0e-5 + 1.8e-6 + 5.0e-5 + 8.0e-9 + 8.0e-6 + 2.0e-5)


def _region_profile(**overrides):
    values = dict(
        tokenizer_async_call_s=None, tokenizer_encode_intercept_s=None,
        tokenizer_encode_per_token_s=None,
        tokenizer_render_region_intercept_s=2.0e-4,
        tokenizer_render_region_per_token_s=2.1e-6)
    values.update(overrides)
    return _profile(**values)


def test_tokenizer_entry_form_is_exclusive_and_fails_closed():
    # How the handler reaches the tokenizer is a per-runtime source fact: a
    # runtime whose source has the batching wrapper pays that call plus the
    # encode it then runs, while a runtime whose handler calls a renderer pays
    # that region, measured end to end.  Never both, and never neither.
    with pytest.raises(ValueError):
        _profile(tokenizer_render_region_intercept_s=2.0e-4,
                 tokenizer_render_region_per_token_s=2.1e-6)
    with pytest.raises(ValueError):
        _profile(tokenizer_async_call_s=None)
    region = _region_profile()
    assert region.admission_service_time_s(100) == pytest.approx(
        4.0e-4 + 2.0e-4 + 100 * 2.1e-6
        + 5.0e-5 + 100 * 8.0e-9 + 8.0e-6 + 2.0e-5)


def test_each_entry_form_needs_all_of_its_own_terms():
    # Half a form is never in domain: a wrapper runtime that dropped its encode
    # would silently stop charging work it does, and a region runtime with only
    # one of its two terms has no affine at all.
    with pytest.raises(ValueError):
        _profile(tokenizer_encode_intercept_s=None)
    with pytest.raises(ValueError):
        _profile(tokenizer_encode_per_token_s=None)
    with pytest.raises(ValueError):
        _region_profile(tokenizer_render_region_per_token_s=None)


def test_the_render_region_does_not_also_charge_an_encode():
    # The measured region already contains the encode, so charging one on top
    # would bill the same work twice.
    region = _region_profile()
    assert region.tokenizer_encode_intercept_s is None
    assert region.tokenizer_encode_per_token_s is None
    marginal = (region.admission_service_time_s(2)
                - region.admission_service_time_s(1))
    assert marginal == pytest.approx(2.1e-6 + 8.0e-9)


def test_admission_rejects_non_physical_prompt_lengths():
    profile = _profile()
    with pytest.raises(ValueError):
        profile.admission_service_time_s(0)
    with pytest.raises(TypeError):
        profile.admission_service_time_s(True)


def test_scheduler_visibility_is_delayed_and_canonical_arrival_is_kept():
    profile = _profile()
    requests = RequestStream([
        RequestSpec(0.0, "a", 1000, 4),
        RequestSpec(0.5, "b", 10, 4),
    ])
    visible = scheduler_visible_requests(requests, profile)
    by_id = {request.request_id: request for request in visible}
    for request in requests:
        expected = request.arrival_time_s + profile.admission_service_time_s(
            request.input_tokens)
        assert by_id[request.request_id].arrival_time_s == pytest.approx(expected)
    # A longer prompt costs more to admit, so the delay is not a flat constant.
    assert (by_id["a"].arrival_time_s - 0.0) > (by_id["b"].arrival_time_s - 0.5)
    # The canonical stream the metrics are measured against is untouched.
    assert [request.arrival_time_s for request in requests] == [0.0, 0.5]


def test_absent_profile_leaves_the_stream_identical():
    requests = RequestStream([RequestSpec(0.0, "a", 8, 2)])
    assert scheduler_visible_requests(requests, None) is requests


def test_first_token_delivery_rejects_invalid_values():
    requests = RequestStream([RequestSpec(0.0, "a", 8, 2)])
    for bad in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises((ValueError, TypeError)):
            compute_serving_metrics(requests, _EMPTY_TRACE, bad)


class _EmptyTrace:
    entries = ()


_EMPTY_TRACE = _EmptyTrace()


@pytest.mark.parametrize("hardware_id", sorted(_PROFILE_PATHS))
def test_published_profiles_load_and_stay_in_their_source_domain(hardware_id):
    profile = load_host_request_path_profile(_PROFILE_PATHS[hardware_id])
    assert profile.hardware_id == hardware_id
    # Every charged term is a positive measured service time.
    assert profile.admission_service_time_s(512) > 0.0
    assert profile.first_token_delivery_service_time_s(1) > 0.0
    # A step that carries more outputs walks the same delivery
    # path once per output, so the delivered token waits longer.
    assert (profile.first_token_delivery_service_time_s(8)
            > profile.first_token_delivery_service_time_s(1))
    payload = json.loads(open(_PROFILE_PATHS[hardware_id]).read())
    service = payload["service_time_s"]
    # The batching-tokenizer term is present only where the pinned runtime
    # source actually has that wrapper; the other entry form covers the rest.
    has_wrapper = service["tokenizer_async_call"] is not None
    assert has_wrapper == (
        service["tokenizer_render_region_intercept"] is None)
    # Each form carries its own terms and never the other form's.
    if has_wrapper:
        assert service["tokenizer_encode_intercept"] > 0.0
        assert service["tokenizer_encode_per_token"] > 0.0
    else:
        assert service["tokenizer_encode_intercept"] is None
        assert service["tokenizer_render_region_per_token"] > 0.0
    # The input processor runs once per request on every runtime.
    assert service["input_processor_intercept"] > 0.0
    assert service["input_processor_per_token"] > 0.0
    assert service["output_delivery_per_output"] > 0.0
    if hardware_id == "metax_c500":
        assert not has_wrapper, (
            "vLLM 0.24.0 has no batching tokenizer wrapper")
    else:
        assert has_wrapper


@pytest.mark.parametrize("hardware_id", sorted(_PROFILE_PATHS))
def test_published_profiles_reject_tampering(hardware_id, tmp_path):
    payload = json.loads(open(_PROFILE_PATHS[hardware_id]).read())
    payload["measurement"]["artifacts"][0]["sha256"] = "f" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_host_request_path_profile(tampered)


def test_loader_rejects_unknown_and_missing_fields(tmp_path):
    payload = json.loads(open(_PROFILE_PATHS["nvidia_a800"]).read())
    payload["unexpected"] = 1
    extra = tmp_path / "extra.json"
    extra.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_host_request_path_profile(extra)

    payload = json.loads(open(_PROFILE_PATHS["nvidia_a800"]).read())
    del payload["service_time_s"]["engine_core_ipc"]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_host_request_path_profile(missing)
