from __future__ import annotations

import unittest

from astermind.workload.contract import CAUSAL_SESSION_REQUEST_KEYS, validate_exact_keys, validate_session_consistency


def _row(**updates):
    row = {
        "request_id": "request-0",
        "arrival_time_s": 10.0,
        "user_region_id": "region-0",
        "access_mode": "gateway_sgl",
        "ingress_id": "gateway-0",
        "gateway_id": "gateway-0",
        "model_family": "test",
        "prompt_tokens": 120,
        "new_user_tokens": 20,
        "context_tokens_before_turn": 100,
        "output_tokens": 8,
        "input_bytes": 480,
        "output_bytes": 32,
        "context_window_tokens": 1024,
        "slo_class": "interactive_short",
        "ttft_deadline_s": 3.0,
        "e2e_deadline_s": 30.0,
        "session_id": "session-0",
        "turn_index": 0,
        "session_turn_count": 1,
        "inter_turn_gap_s": None,
        "session_start_time_s": 10.0,
        "session_ttl_s": 60.0,
        "seed": 1,
        "arrival_time_mode": "absolute",
        "depends_on_request_id": None,
        "source_trace_time_s": 0.0,
        "source_api_time_s": 2.0,
    }
    row.update(updates)
    return row


class CausalContractTests(unittest.TestCase):
    def test_causal_contract_uses_one_token_field_set(self):
        row = _row()
        validate_exact_keys((row,), CAUSAL_SESSION_REQUEST_KEYS, "session_requests")
        validate_session_consistency([row])
        self.assertNotIn("kv_reusable_tokens", row)
        self.assertNotIn("prefill_tokens", row)
        self.assertNotIn("kv_block_size_tokens", row)

    def test_prompt_must_equal_reusable_context_plus_current_prefill(self):
        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            validate_session_consistency([_row(new_user_tokens=19)])

    def test_completion_relative_request_requires_nonnegative_think_time(self):
        root = _row(session_turn_count=2)
        child = _row(
            request_id="request-1",
            turn_index=1,
            session_turn_count=2,
            arrival_time_mode="after_dependency_completion",
            depends_on_request_id="request-0",
            inter_turn_gap_s=None,
        )
        with self.assertRaisesRegex(ValueError, "requires inter_turn_gap_s"):
            validate_session_consistency([root, child])

        child["inter_turn_gap_s"] = -0.1
        with self.assertRaisesRegex(ValueError, "negative inter_turn_gap_s"):
            validate_session_consistency([root, child])
