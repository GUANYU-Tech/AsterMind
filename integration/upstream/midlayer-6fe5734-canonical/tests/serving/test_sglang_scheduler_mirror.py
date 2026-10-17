import pytest

from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
)
from src.serving.backend.sglang_scheduler_mirror import (
    SGLangSchedulerMirrorBackend,
    SGLangSchedulerMirrorConfig,
)
from src.serving.request_gen.request_stream import RequestSpec
from src.serving.schedule.backend_protocol import BackendConfig


def _backend(capacity, chunk=32768, depth=0):
    assumption = (
        MTPAcceptanceAssumption("synthetic-source", depth, "0.5", 0, "0" * 64)
        if depth else None)
    return SGLangSchedulerMirrorBackend(
        BackendConfig("main", "colocated", chunk, 64, 1,
                      scheduler_pipeline_depth=1),
        SGLangSchedulerMirrorConfig(
            source_version="v0.5.19",
            max_total_num_tokens=capacity,
            max_total_num_tokens_source="synthetic fixture",
            chunked_prefill_size=chunk,
            max_prefill_tokens=chunk,
            max_running_requests=256,
            schedule_conservativeness=1.0,
            mtp_num_speculative_tokens=depth,
            mtp_acceptance_assumption=assumption,
            mtp_acceptance_source_id="synthetic-source" if depth else "disabled",
        ),
    )


def _drain(backend):
    steps = []
    while backend.has_work():
        step = backend.plan_step(0.0)
        backend.complete_step(step, 0.0)
        steps.append(step)
    return steps


def test_prefill_batches_are_never_mixed_and_chunk_first():
    backend = _backend(capacity=10**6)
    for index in range(3):
        backend.add_request(RequestSpec(0.0, f"r{index}", 20000, 4))
    steps = _drain(backend)
    first, second = steps[0].schedule.requests, steps[1].schedule.requests
    assert [(r.request_id, r.input_tokens) for r in first] == [
        ("r0", 20000), ("r1", 12768)]
    assert [(r.request_id, r.input_tokens) for r in second] == [
        ("r1", 7232), ("r2", 20000)]
    for step in steps:
        assert len({r.phase for r in step.schedule.requests}) == 1


def test_ignore_eos_admission_reserves_full_output():
    backend = _backend(capacity=320)
    for index in range(3):
        backend.add_request(RequestSpec(0.0, f"r{index}", 100, 50))
    steps = _drain(backend)
    assert [r.request_id for r in steps[0].schedule.requests] == ["r0", "r1"]
    first_finish = next(
        i for i, step in enumerate(steps) if step.finished_request_ids)
    r2_prefill = next(
        i for i, step in enumerate(steps)
        if any(r.request_id == "r2" for r in step.schedule.requests))
    assert r2_prefill == first_finish + 1


def test_eagle_decode_verifies_depth_plus_one_tokens():
    backend = _backend(capacity=10**6, depth=2)
    backend.add_request(RequestSpec(0.0, "r0", 64, 40))
    steps = _drain(backend)
    decode = [r for s in steps for r in s.schedule.requests
              if r.phase == "decode"]
    assert decode and all(r.output_tokens == 3 for r in decode)
    assert all(r.raw_model_output_tokens == r.mtp_accepted_tokens + 1
               for r in decode)
    produced = sum(s.generated_tokens["r0"] for s in steps)
    assert produced == 40


def test_request_larger_than_pool_fails_closed():
    backend = _backend(capacity=64)
    backend.add_request(RequestSpec(0.0, "r0", 100, 4))
    with pytest.raises(RuntimeError):
        backend.plan_step(0.0)
