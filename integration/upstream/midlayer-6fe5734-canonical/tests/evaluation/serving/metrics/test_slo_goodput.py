import hashlib
import json

import pytest

from evaluation.serving.metrics.slo_goodput import (
    SLOContract,
    load_slo_contract,
    slo_goodput_manifest,
)
from src.serving.metrics.serving_metrics import RequestMetrics, ServingMetrics
from src.serving.request_gen.request_stream import RequestSpec, RequestStream


def _contract_payload(*, token_basis="output_tokens"):
    return {
        "schema_version": 1,
        "contract_id": "interactive-v1",
        "ttft_threshold_s": 0.2,
        "tpot_threshold_s": 0.05,
        "comparison": "less_than_or_equal",
        "threshold_composition": "ttft_and_tpot",
        "request_eligibility": "positive_output_tokens",
        "single_output_tpot": "zero_seconds",
        "token_basis": token_basis,
        "duration_basis": "simulation_origin_to_last_completion",
    }


def _contract(*, token_basis="output_tokens"):
    payload = _contract_payload(token_basis=token_basis)
    return SLOContract(
        schema_version=payload["schema_version"],
        contract_id=payload["contract_id"],
        ttft_threshold_s=payload["ttft_threshold_s"],
        tpot_threshold_s=payload["tpot_threshold_s"],
        comparison=payload["comparison"],
        threshold_composition=payload["threshold_composition"],
        request_eligibility=payload["request_eligibility"],
        single_output_tpot=payload["single_output_tpot"],
        token_basis=payload["token_basis"],
        duration_basis=payload["duration_basis"],
        source_path="/contracts/interactive-v1.json",
        source_sha256="0" * 64,
    )


def _requests_and_metrics():
    requests = RequestStream((
        RequestSpec(0.0, "pass", 4, 2),
        RequestSpec(0.0, "ttft-fail", 4, 1),
        RequestSpec(0.0, "zero", 4, 0),
    ))
    metrics = ServingMetrics(
        requests={
            "pass": RequestMetrics("pass", 0.1, 0.01, 0.3),
            "ttft-fail": RequestMetrics(
                "ttft-fail", 0.3, 0.0, 0.4),
            "zero": RequestMetrics("zero", None, None, 0.2),
        },
        request_throughput_per_s=1.5,
        output_token_throughput_per_s=1.5,
        token_throughput_per_s=7.5,
    )
    return requests, metrics


def test_load_slo_contract_preserves_exact_config_and_source_hash(tmp_path):
    path = tmp_path / "slo.json"
    raw = json.dumps(_contract_payload(), sort_keys=True).encode()
    path.write_bytes(raw)

    contract = load_slo_contract(path)

    assert contract.contract_id == "interactive-v1"
    assert contract.ttft_threshold_s == 0.2
    assert contract.tpot_threshold_s == 0.05
    assert contract.token_basis == "output_tokens"
    assert contract.source_path == str(path.resolve())
    assert contract.source_sha256 == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mutation", ("missing", "unexpected"))
def test_load_slo_contract_rejects_schema_drift(tmp_path, mutation):
    payload = _contract_payload()
    if mutation == "missing":
        del payload["token_basis"]
    elif mutation == "unexpected":
        payload["threshold_unit"] = "seconds"
    else:
        raise AssertionError(f"unknown mutation: {mutation}")
    path = tmp_path / "slo.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fields disagree"):
        load_slo_contract(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("ttft_threshold_s", True),
        ("tpot_threshold_s", -1.0),
        ("comparison", "strictly_less_than"),
        ("threshold_composition", "ttft_or_tpot"),
        ("request_eligibility", "all_requests"),
        ("single_output_tpot", "unavailable"),
        ("token_basis", "hidden_tokens"),
        ("duration_basis", "first_arrival_to_last_completion"),
    ],
)
def test_load_slo_contract_rejects_unsupported_semantics(
        tmp_path, field, value):
    payload = _contract_payload()
    payload[field] = value
    path = tmp_path / "slo.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        load_slo_contract(path)


def test_load_slo_contract_rejects_duplicate_fields(tmp_path):
    path = tmp_path / "slo.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1,'
        '"contract_id":"duplicate","ttft_threshold_s":1,'
        '"tpot_threshold_s":1,"comparison":"less_than_or_equal",'
        '"threshold_composition":"ttft_and_tpot",'
        '"request_eligibility":"positive_output_tokens",'
        '"single_output_tpot":"zero_seconds",'
        '"token_basis":"output_tokens",'
        '"duration_basis":"simulation_origin_to_last_completion"}'
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_slo_contract(path)


def test_slo_goodput_counts_requests_tokens_and_violation_kinds():
    requests, metrics = _requests_and_metrics()

    manifest = slo_goodput_manifest(
        requests, metrics, 2.0, _contract())

    assert manifest["availability"] == "available"
    assert manifest["contract"]["contract_id"] == "interactive-v1"
    assert manifest["population"] == {
        "total_request_count": 3,
        "eligible_request_count": 2,
        "excluded_request_count": 1,
        "slo_met_request_count": 1,
        "violated_request_count": 1,
        "eligible_token_count": 3,
        "slo_met_token_count": 2,
    }
    assert manifest["violation_counts"] == {
        "ttft_only": 1,
        "tpot_only": 0,
        "ttft_and_tpot": 0,
    }
    assert manifest["request_goodput_per_s"] == 0.5
    assert manifest["token_goodput_per_s"] == 1.0


def test_slo_goodput_honors_explicit_input_plus_output_token_basis():
    requests, metrics = _requests_and_metrics()

    manifest = slo_goodput_manifest(
        requests, metrics, 2.0,
        _contract(token_basis="input_plus_output_tokens"),
    )

    assert manifest["population"]["eligible_token_count"] == 11
    assert manifest["population"]["slo_met_token_count"] == 6
    assert manifest["token_goodput_per_s"] == 3.0


def test_slo_goodput_thresholds_are_inclusive_for_one_token_request():
    requests = RequestStream((RequestSpec(0.0, "one", 4, 1),))
    metrics = ServingMetrics(
        requests={
            "one": RequestMetrics("one", 0.2, 0.0, 0.2),
        },
        request_throughput_per_s=5.0,
        output_token_throughput_per_s=5.0,
        token_throughput_per_s=25.0,
    )
    contract = _contract()

    manifest = slo_goodput_manifest(requests, metrics, 0.2, contract)

    assert manifest["population"]["slo_met_request_count"] == 1
    assert manifest["population"]["slo_met_token_count"] == 1
    assert manifest["request_goodput_per_s"] == 5.0
    assert manifest["token_goodput_per_s"] == 5.0


def test_slo_goodput_is_explicitly_unavailable_without_contract():
    requests, metrics = _requests_and_metrics()

    manifest = slo_goodput_manifest(requests, metrics, 2.0, None)

    assert manifest == {
        "schema_version": 1,
        "availability": "unavailable_no_slo_contract",
        "contract": None,
        "population": {
            "total_request_count": 3,
            "eligible_request_count": 2,
            "excluded_request_count": 1,
            "slo_met_request_count": None,
            "violated_request_count": None,
            "eligible_token_count": None,
            "slo_met_token_count": None,
        },
        "violation_counts": None,
        "request_goodput_per_s": None,
        "token_goodput_per_s": None,
    }


def test_slo_goodput_rejects_half_available_positive_output_metrics():
    requests = RequestStream((RequestSpec(0.0, "request", 4, 1),))
    metrics = ServingMetrics(
        requests={
            "request": RequestMetrics("request", 0.1, None, 0.2),
        },
        request_throughput_per_s=1.0,
        output_token_throughput_per_s=1.0,
        token_throughput_per_s=5.0,
    )

    with pytest.raises(ValueError, match="TPOT"):
        slo_goodput_manifest(requests, metrics, 1.0, _contract())


def test_slo_goodput_rejects_nonzero_single_output_tpot():
    requests = RequestStream((RequestSpec(0.0, "request", 4, 1),))
    metrics = ServingMetrics(
        requests={
            "request": RequestMetrics("request", 0.1, 0.01, 0.2),
        },
        request_throughput_per_s=1.0,
        output_token_throughput_per_s=1.0,
        token_throughput_per_s=5.0,
    )

    with pytest.raises(ValueError, match="zero seconds"):
        slo_goodput_manifest(requests, metrics, 1.0, _contract())
