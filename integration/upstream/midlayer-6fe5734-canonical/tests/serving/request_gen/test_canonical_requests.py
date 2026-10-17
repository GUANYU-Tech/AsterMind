import csv

import pytest

from src.serving.request_gen import canonical_requests
from src.serving.request_gen import (
    CANONICAL_REQUEST_COLUMNS,
    load_canonical_requests,
)
from src.serving.request_gen.canonical_requests import (
    parse_canonical_requests_bytes,
)


def _write_requests(path, *, header=CANONICAL_REQUEST_COLUMNS):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerow((10, 8, 2, 3))
        writer.writerow((0, 4, 0, 0))
        writer.writerow((0, 6, 3, 2))


def test_canonical_requests_use_physical_row_ids_and_stable_arrival_order(tmp_path):
    path = tmp_path / "requests.csv"
    _write_requests(path)

    requests = load_canonical_requests(path)

    assert [request.request_id for request in requests] == [
        "request-000000000001",
        "request-000000000002",
        "request-000000000000",
    ]
    assert [request.arrival_time_s for request in requests] == [0.0, 0.0, 0.01]
    assert [
        (
            request.input_tokens,
            request.output_tokens,
            request.prefix_hit_tokens,
        )
        for request in requests
    ] == [(4, 0, 0), (6, 3, 2), (8, 2, 3)]


def test_canonical_requests_path_loader_reads_once_and_uses_bytes_parser(
        tmp_path, monkeypatch):
    path = tmp_path / "requests.csv"
    _write_requests(path)
    source_bytes = path.read_bytes()
    read_count = 0
    original_read_bytes = type(path).read_bytes
    production_parse = canonical_requests.parse_production_trace_bytes
    delegated_payloads = []

    def counted_read_bytes(candidate):
        nonlocal read_count
        if candidate == path:
            read_count += 1
        return original_read_bytes(candidate)

    monkeypatch.setattr(type(path), "read_bytes", counted_read_bytes)

    def record_production_parse(payload, config, *, source_path):
        delegated_payloads.append((payload, config, source_path))
        return production_parse(payload, config, source_path=source_path)

    monkeypatch.setattr(
        canonical_requests,
        "parse_production_trace_bytes",
        record_production_parse,
    )

    from_path = load_canonical_requests(path)
    from_bytes = parse_canonical_requests_bytes(
        source_bytes,
        source_path=path,
    )

    assert read_count == 1
    assert from_path == from_bytes
    assert len(delegated_payloads) == 2
    assert all(payload == source_bytes
               for payload, _, _ in delegated_payloads)
    assert all(config.source_format == "csv"
               for _, config, _ in delegated_payloads)


@pytest.mark.parametrize(
    "header",
    [
        CANONICAL_REQUEST_COLUMNS + ("extra",),
        CANONICAL_REQUEST_COLUMNS[:-1],
        tuple(reversed(CANONICAL_REQUEST_COLUMNS)),
    ],
)
def test_canonical_requests_reject_nonexact_header(tmp_path, header):
    path = tmp_path / "requests.csv"
    _write_requests(path, header=header)

    with pytest.raises(ValueError, match="header does not match"):
        load_canonical_requests(path)


@pytest.mark.parametrize(
    "source_bytes",
    (
        b"secret-request-0,8,2,3\n",
        b"secret-a,secret-b,secret-c,secret-d\n0,8,2,3\n",
    ),
)
def test_canonical_requests_header_errors_do_not_echo_supplied_values(source_bytes):
    with pytest.raises(ValueError, match="header does not match") as captured:
        parse_canonical_requests_bytes(
            source_bytes,
            source_path="sensitive-requests.csv",
        )
    assert "secret" not in str(captured.value)


@pytest.mark.parametrize(
    "row,error",
    [
        ((0, 0, 1, 0), "input_tokens must be positive"),
        ((0, 1, -1, 0), "output_tokens cannot be negative"),
        ((0, 1, 1, 2), "prefix_hit_tokens must be between zero"),
        (("nan", 1, 1, 0), "arrival_time must normalize"),
    ],
)
def test_canonical_requests_fail_loud_on_invalid_rows(tmp_path, row, error):
    path = tmp_path / "requests.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CANONICAL_REQUEST_COLUMNS)
        writer.writerow(row)

    with pytest.raises((TypeError, ValueError), match=error):
        load_canonical_requests(path)
