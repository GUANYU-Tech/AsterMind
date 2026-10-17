import math

import pytest

from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.request_gen.synthetic_requests import (
    SyntheticRequestConfig, generate_synthetic_requests,
)


def test_synthetic_requests_are_deterministic_and_inject_prefix_hit_length():
    config = SyntheticRequestConfig(
        num_requests=3, request_rate=2.0, input_lengths=(8,),
        output_lengths=(4,), prefix_hit_fraction=0.5, seed=7)
    first = generate_synthetic_requests(config)
    second = generate_synthetic_requests(config)
    assert first == second
    assert all(request.input_tokens == 8 for request in first)
    assert all(request.prefix_hit_tokens == 4 for request in first)
    assert all(request.output_tokens == 4 for request in first)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_request_rejects_non_finite_arrival_time(value):
    with pytest.raises(ValueError, match="arrival time"):
        RequestSpec(value, "request", 1, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_tokens", True),
        ("output_tokens", 1.0),
        ("prefix_hit_tokens", "0"),
    ],
)
def test_request_rejects_non_integer_token_counts(field, value):
    values = {
        "input_tokens": 4,
        "output_tokens": 2,
        "prefix_hit_tokens": 0,
    }
    values[field] = value
    with pytest.raises(TypeError, match=field):
        RequestSpec(0.0, "request", **values)


def test_request_accepts_zero_output_length():
    request = RequestSpec(0.0, "request", 4, 0)

    assert request.output_tokens == 0


def test_request_rejects_negative_output_length():
    with pytest.raises(ValueError, match="output_tokens"):
        RequestSpec(0.0, "request", 4, -1)


@pytest.mark.parametrize("prefix_hit_tokens", [-1, 5])
def test_request_rejects_prefix_hit_outside_input(prefix_hit_tokens):
    with pytest.raises(ValueError, match="prefix_hit_tokens"):
        RequestSpec(0.0, "request", 4, 1, prefix_hit_tokens)


def test_synthetic_requests_preserve_zero_output_length():
    requests = generate_synthetic_requests(SyntheticRequestConfig(
        num_requests=1,
        request_rate=1.0,
        input_lengths=(4,),
        output_lengths=(0,),
    ))

    assert requests.requests[0].output_tokens == 0


def test_request_stream_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one request"):
        RequestStream(())


def test_request_stream_sorts_by_arrival_and_preserves_equal_arrival_order():
    stream = RequestStream((
        RequestSpec(1.0, "later", 1, 1),
        RequestSpec(0.0, "z", 1, 1),
        RequestSpec(0.0, "a", 1, 1),
    ))

    assert [request.request_id for request in stream] == ["z", "a", "later"]
