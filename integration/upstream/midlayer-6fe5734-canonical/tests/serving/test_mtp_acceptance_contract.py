"""Contract tests for the PD decode MTP admission identity hash."""

from src.serving.backend.mtp_acceptance_replay import (
    pd_decode_admission_contract_sha256,
)
from src.serving.request_gen.request_stream import RequestSpec, RequestStream


def _stream(*specs):
    return RequestStream(tuple(specs))


def test_pd_decode_admission_contract_is_transfer_completion_order_insensitive():
    ingress = _stream(
        RequestSpec(0.0, "request-000000000020", 900, 117),
        RequestSpec(0.0, "request-000000000021", 984, 122),
        RequestSpec(0.0, "request-000000000022", 844, 105),
    )
    # A decode instance admits in KV transfer completion order, which can
    # depart from ingress order when same-step handoffs carry different
    # byte counts; acceptance sampling is keyed by (step_id, request_id),
    # so order carries no acceptance semantics.
    completion_order = _stream(
        RequestSpec(0.0, "request-000000000022", 844, 105),
        RequestSpec(0.0, "request-000000000020", 900, 117),
        RequestSpec(0.0, "request-000000000021", 984, 122),
    )

    assert (pd_decode_admission_contract_sha256(ingress)
            == pd_decode_admission_contract_sha256(completion_order))


def test_pd_decode_admission_contract_binds_request_set_and_fields():
    base = _stream(
        RequestSpec(0.0, "request-000000000020", 900, 117),
        RequestSpec(0.0, "request-000000000021", 984, 122),
    )
    different_length = _stream(
        RequestSpec(0.0, "request-000000000020", 900, 117),
        RequestSpec(0.0, "request-000000000021", 985, 122),
    )
    different_id = _stream(
        RequestSpec(0.0, "request-000000000020", 900, 117),
        RequestSpec(0.0, "request-000000000023", 984, 122),
    )

    assert (pd_decode_admission_contract_sha256(base)
            != pd_decode_admission_contract_sha256(different_length))
    assert (pd_decode_admission_contract_sha256(base)
            != pd_decode_admission_contract_sha256(different_id))
