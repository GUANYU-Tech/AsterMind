import math

import pytest

from evaluation.serving.metrics.latency_distribution import (
    latency_distribution_manifest,
    linear_percentile,
)
from src.serving.metrics.serving_metrics import RequestMetrics


def test_linear_percentile_uses_n_minus_one_rank_interpolation():
    values = [1.0, 5.0]

    assert linear_percentile(values, 0) == 1.0
    assert linear_percentile(values, 25) == 2.0
    assert linear_percentile(values, 50) == 3.0
    assert linear_percentile(values, 75) == 4.0
    assert linear_percentile(values, 100) == 5.0


def test_linear_percentile_preserves_identical_small_values_exactly():
    value = 3.04035205540734e-05

    quantiles = tuple(
        linear_percentile([value, value], percentile)
        for percentile in range(101)
    )

    assert quantiles == (value,) * 101


def test_linear_percentile_is_monotonic_for_nearly_equal_small_values():
    values = [
        3.0334914201306153e-05,
        3.0334914201306153e-05,
        3.0334914201306153e-05,
        3.0334914201306153e-05,
        3.0334914201306167e-05,
    ]

    quantiles = tuple(
        linear_percentile(values, percentile)
        for percentile in range(101)
    )

    assert all(
        lower <= upper
        for lower, upper in zip(quantiles, quantiles[1:])
    )
    assert quantiles[0] == values[0]
    assert quantiles[-1] == values[-1]


def test_linear_percentile_does_not_overflow_finite_extreme_values():
    assert linear_percentile([-1e308, 1e308], 50) == 0.0


@pytest.mark.parametrize(
    "values,percentile",
    [
        ([], 50),
        ([2.0, 1.0], 50),
        ([math.inf], 50),
        ([True], 50),
        ([1.0], -1),
        ([1.0], 101),
        ([1.0], True),
    ],
)
def test_linear_percentile_rejects_invalid_inputs(values, percentile):
    with pytest.raises(ValueError):
        linear_percentile(values, percentile)


def test_latency_distribution_reports_cdf_quantiles_and_exclusions():
    manifest = latency_distribution_manifest({
        "a": RequestMetrics("a", 1.0, 0.0, 2.0),
        "b": RequestMetrics("b", None, None, 3.0),
        "c": RequestMetrics("c", 5.0, 4.0, 6.0),
    })

    assert manifest["schema_version"] == 2
    assert manifest["quantile_representation"] == (
        "linear_interpolated_quantile_points_p0_p100")
    assert manifest["percentile_method"] == (
        "linear_interpolation_rank_n_minus_1_v1")
    assert manifest["cdf_representation"] == (
        "empirical_right_continuous_step_points")
    assert manifest["cdf_method"] == (
        "empirical_right_continuous_less_than_or_equal_v1")
    ttft = manifest["ttft_s"]
    assert ttft["availability"] == "available"
    assert ttft["eligible_request_count"] == 2
    assert ttft["excluded_request_count"] == 1
    assert ttft["quantiles"]["percentiles"] == tuple(range(101))
    assert len(ttft["quantiles"]["values_s"]) == 101
    assert ttft["quantiles"]["values_s"][0] == 1.0
    assert ttft["quantiles"]["values_s"][50] == 3.0
    assert ttft["quantiles"]["values_s"][100] == 5.0
    assert ttft["empirical_cdf"] == {
        "values_s": (1.0, 5.0),
        "cumulative_probabilities": (0.5, 1.0),
    }
    tpot = manifest["tpot_s"]
    assert tpot["eligible_request_count"] == 2
    assert tpot["excluded_request_count"] == 1
    assert tpot["quantiles"]["values_s"][0] == 0.0
    assert tpot["quantiles"]["values_s"][50] == 2.0
    assert tpot["quantiles"]["values_s"][100] == 4.0
    assert tpot["empirical_cdf"] == {
        "values_s": (0.0, 4.0),
        "cumulative_probabilities": (0.5, 1.0),
    }
    latency = manifest["latency_s"]
    assert latency["eligible_request_count"] == 3
    assert latency["excluded_request_count"] == 0
    assert latency["exclusion_reason"] == (
        "none_all_successful_requests_are_eligible")
    assert latency["quantiles"]["values_s"][0] == 2.0
    assert latency["quantiles"]["values_s"][50] == 3.0
    assert latency["quantiles"]["values_s"][100] == 6.0


def test_latency_distribution_keeps_zero_output_explicitly_unavailable():
    manifest = latency_distribution_manifest({
        "zero": RequestMetrics("zero", None, None, 1.0),
    })

    for metric_name in ("ttft_s", "tpot_s"):
        distribution = manifest[metric_name]
        assert distribution["availability"] == (
            "unavailable_no_eligible_requests")
        assert distribution["eligible_request_count"] == 0
        assert distribution["excluded_request_count"] == 1
        assert distribution["exclusion_reason"] == (
            "requests_without_output_tokens")
        assert distribution["quantiles"] == {
            "percentiles": (),
            "values_s": (),
        }
        assert distribution["empirical_cdf"] == {
            "values_s": (),
            "cumulative_probabilities": (),
        }
    latency = manifest["latency_s"]
    assert latency["availability"] == "available"
    assert latency["eligible_request_count"] == 1
    assert latency["excluded_request_count"] == 0
    assert latency["quantiles"]["values_s"] == (1.0,) * 101


def test_latency_distribution_empirical_cdf_coalesces_duplicate_values():
    manifest = latency_distribution_manifest({
        "a": RequestMetrics("a", 1.0, 2.0, 3.0),
        "b": RequestMetrics("b", 1.0, 4.0, 5.0),
        "c": RequestMetrics("c", 5.0, 6.0, 7.0),
    })

    assert manifest["ttft_s"]["empirical_cdf"] == {
        "values_s": (1.0, 5.0),
        "cumulative_probabilities": (2 / 3, 1.0),
    }


def test_latency_distribution_rejects_mismatched_request_identity():
    with pytest.raises(ValueError, match="disagree"):
        latency_distribution_manifest({
            "request": RequestMetrics("other", 1.0, 1.0, 2.0),
        })


def test_latency_distribution_rejects_half_available_output_latencies():
    with pytest.raises(ValueError, match="available or unavailable"):
        latency_distribution_manifest({
            "request": RequestMetrics("request", 1.0, None, 2.0),
        })
