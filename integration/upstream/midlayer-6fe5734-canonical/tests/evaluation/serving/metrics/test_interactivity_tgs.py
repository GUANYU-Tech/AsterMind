import hashlib
import json

import pytest

from evaluation.serving.metrics.interactivity_tgs import (
    arrival_rate_sweep_analysis_manifest,
    load_interactivity_tgs_contract,
    sweep_metric_point_from_manifest,
)


def _contract_payload():
    return {
        "schema_version": 1,
        "contract_id": "paper-p99-tpot-output-tgs-v1",
        "interactivity_metric": "tpot_s",
        "interactivity_percentile": 99,
        "interactivity_unit": "seconds",
        "interactivity_aggregation": (
            "linear_percentile_over_eligible_requests"),
        "interactivity_eligibility": (
            "successful_requests_with_positive_output_tokens"),
        "tgs_metric": "output_token_throughput_per_s",
        "tgs_unit": "output_tokens_per_second",
        "tgs_aggregation": (
            "completed_tokens_over_simulation_origin_to_last_completion"),
        "peak_selection": (
            "maximum_tgs_then_lowest_arrival_rate_scale"),
        "saturation_rule": (
            "first_consecutive_marginal_tgs_gain_ratio_at_or_below_threshold"),
        "saturation_marginal_gain_ratio_threshold": 0.1,
        "saturation_minimum_consecutive_intervals": 2,
    }


def _load_contract(tmp_path, payload=None):
    if payload is None:
        payload = _contract_payload()
    path = tmp_path / "interactivity-tgs.json"
    raw = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(raw)
    return load_interactivity_tgs_contract(path), path, raw


def _metrics(interactivity_s, output_tgs, *, eligible_count=4):
    if interactivity_s is None:
        distribution = {
            "availability": "unavailable_no_eligible_requests",
            "eligible_request_count": 0,
            "excluded_request_count": 4,
            "quantiles": {
                "percentiles": (),
                "values_s": (),
            },
        }
    else:
        values = [interactivity_s] * 101
        distribution = {
            "availability": "available",
            "eligible_request_count": eligible_count,
            "excluded_request_count": 0,
            "quantiles": {
                "percentiles": tuple(range(101)),
                "values_s": tuple(values),
            },
        }
    return {
        "request_throughput_per_s": output_tgs / 4,
        "output_token_throughput_per_s": output_tgs,
        "token_throughput_per_s": output_tgs * 3,
        "latency_distributions": {
            "percentile_method": (
                "linear_interpolation_rank_n_minus_1_v1"),
            "tpot_s": distribution,
        },
    }


def test_load_contract_records_exact_source_and_rejects_inconsistent_units(
        tmp_path):
    contract, path, raw = _load_contract(tmp_path)

    assert contract.contract_id == "paper-p99-tpot-output-tgs-v1"
    assert contract.interactivity_metric == "tpot_s"
    assert contract.interactivity_percentile == 99
    assert contract.source_path == str(path.resolve())
    assert contract.source_sha256 == hashlib.sha256(raw).hexdigest()

    invalid = _contract_payload()
    invalid["tgs_unit"] = "input_plus_output_tokens_per_second"
    with pytest.raises(ValueError, match="tgs_unit disagrees"):
        _load_contract(tmp_path, invalid)

    invalid = _contract_payload()
    invalid["interactivity_metric"] = "unsupported_s"
    with pytest.raises(ValueError, match="'latency_s'"):
        _load_contract(tmp_path, invalid)


def test_sweep_analysis_builds_pareto_peak_and_confirmed_saturation(tmp_path):
    contract, _, _ = _load_contract(tmp_path)
    point_specs = (
        (0.5, 0.20, 90.0),
        (1.0, 0.10, 100.0),
        (2.0, 0.20, 180.0),
        (4.0, 0.50, 190.0),
        (8.0, 1.00, 191.0),
    )
    points = tuple(
        sweep_metric_point_from_manifest(scale, _metrics(tpot, tgs), contract)
        for scale, tpot, tgs in reversed(point_specs)
    )

    manifest = arrival_rate_sweep_analysis_manifest(points, contract)

    tradeoff = manifest["interactivity_tgs"]
    assert tradeoff["availability"] == "available"
    assert tuple(
        point["arrival_rate_scale"] for point in tradeoff["points"]
    ) == (0.5, 1.0, 2.0, 4.0, 8.0)
    # Scale 0.5 is dominated by scale 1; the other points expose actual
    # latency/throughput trade-offs.
    assert tuple(
        point["arrival_rate_scale"]
        for point in tradeoff["pareto_frontier"]
    ) == (1.0, 2.0, 4.0, 8.0)
    assert manifest["peak_selection"]["availability"] == (
        "available_observed_sampled_range")
    assert manifest["peak_selection"]["selection_scope"] == (
        "sampled_arrival_rate_scales_only")
    assert manifest["peak_selection"]["point"]["arrival_rate_scale"] == 8.0
    assert manifest["peak_selection"]["point"]["tgs_value"] == 191.0

    saturation = manifest["saturation_analysis"]
    assert saturation["availability"] == (
        "observed_sampled_elasticity_threshold")
    assert saturation["saturation_onset_point"][
        "arrival_rate_scale"] == 4.0
    assert saturation["saturation_confirmed_at_point"][
        "arrival_rate_scale"] == 8.0
    intervals = saturation["intervals"]
    assert intervals[-2]["at_or_below_threshold"] is True
    assert intervals[-1]["at_or_below_threshold"] is True


def test_sweep_analysis_reports_missing_contract_and_missing_population(
        tmp_path):
    unavailable = arrival_rate_sweep_analysis_manifest(None, None)
    assert unavailable["interactivity_tgs"] == {
        "availability": "unavailable_no_metric_contract",
        "contract": None,
        "points": (),
        "pareto_frontier": (),
    }
    assert unavailable["peak_selection"]["availability"] == (
        "unavailable_no_metric_contract")
    assert unavailable["saturation_analysis"]["availability"] == (
        "unavailable_no_metric_contract")

    contract, _, _ = _load_contract(tmp_path)
    points = (
        sweep_metric_point_from_manifest(1.0, _metrics(None, 0.0), contract),
        sweep_metric_point_from_manifest(2.0, _metrics(None, 0.0), contract),
    )
    manifest = arrival_rate_sweep_analysis_manifest(points, contract)
    assert manifest["interactivity_tgs"]["availability"] == (
        "unavailable_no_eligible_requests")
    assert manifest["interactivity_tgs"]["pareto_frontier"] == ()
    assert manifest["peak_selection"]["availability"] == (
        "unavailable_no_positive_tgs")
    assert manifest["peak_selection"]["point"] is None
    assert manifest["saturation_analysis"]["availability"] == (
        "unavailable_no_positive_tgs_baseline")


def test_sweep_point_rejects_missing_contract_percentile(tmp_path):
    contract, _, _ = _load_contract(tmp_path)
    metrics = _metrics(0.1, 100.0)
    metrics["latency_distributions"]["tpot_s"]["quantiles"] = {
        "percentiles": (50,),
        "values_s": (0.1,),
    }

    with pytest.raises(ValueError, match="does not contain contract percentile"):
        sweep_metric_point_from_manifest(1.0, metrics, contract)


def test_sweep_point_accepts_versioned_e2e_interactivity(tmp_path):
    payload = _contract_payload()
    payload["interactivity_metric"] = "latency_s"
    payload["interactivity_eligibility"] = "all_successful_requests"
    contract, _, _ = _load_contract(tmp_path, payload)
    metrics = _metrics(0.1, 100.0)
    metrics["latency_distributions"]["latency_s"] = {
        "availability": "available",
        "eligible_request_count": 4,
        "excluded_request_count": 0,
        "quantiles": {
            "percentiles": tuple(range(101)),
            "values_s": (2.5,) * 101,
        },
    }

    point = sweep_metric_point_from_manifest(1.0, metrics, contract)

    assert point.interactivity_value_s == 2.5
    assert point.interactivity_eligible_request_count == 4


def test_saturation_labels_throughput_regression_without_hiding_it(tmp_path):
    payload = _contract_payload()
    payload["saturation_minimum_consecutive_intervals"] = 1
    contract, _, _ = _load_contract(tmp_path, payload)
    points = (
        sweep_metric_point_from_manifest(
            1.0, _metrics(0.1, 100.0), contract),
        sweep_metric_point_from_manifest(
            2.0, _metrics(0.2, 90.0), contract),
    )

    saturation = arrival_rate_sweep_analysis_manifest(
        points, contract)["saturation_analysis"]

    assert saturation["availability"] == (
        "observed_sampled_elasticity_threshold")
    assert saturation["contains_throughput_regression"] is True
    assert saturation["intervals"][0]["classification"] == (
        "throughput_regression")
