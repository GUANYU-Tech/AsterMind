import hashlib
import json
import math
import tempfile
import unittest
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

from orbitserve.orbit import propagate_sgp4_teme
from orbitserve.orbit.validation import (
    STK_STATE_CORPUS_CASE_IDS,
    STK_STATE_CORPUS_REQUIRED_COVERAGE,
    STK_STATE_CORPUS_SHA256,
    load_stk_state_corpus,
)


def _write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StkStateCorpusTests(unittest.TestCase):
    def test_default_corpus_has_pinned_identity_order_and_coverage(self):
        corpus = load_stk_state_corpus()

        self.assertEqual(corpus.source_sha256, STK_STATE_CORPUS_SHA256)
        self.assertEqual(corpus.case_ids, STK_STATE_CORPUS_CASE_IDS)
        self.assertEqual(corpus.coverage, STK_STATE_CORPUS_REQUIRED_COVERAGE)
        self.assertEqual(len(corpus.cases), 4)
        self.assertEqual(len({case.element_set_id for case in corpus.cases}), 4)

    def test_packaged_resource_matches_pinned_hash(self):
        resource = files("orbitserve.orbit").joinpath(
            "data",
            "stk_validation",
            "state_corpus.json",
        )
        raw = resource.read_bytes()

        self.assertEqual(hashlib.sha256(raw).hexdigest(), STK_STATE_CORPUS_SHA256)

    def test_canonical_bytes_are_repeatable_finite_utf8_lf_json(self):
        first = load_stk_state_corpus().canonical_bytes()
        second = load_stk_state_corpus().canonical_bytes()

        self.assertEqual(first, second)
        self.assertTrue(first.endswith(b"\n"))
        self.assertNotIn(b"\r\n", first)
        decoded = json.loads(first.decode("utf-8"))
        self.assertEqual(decoded["source_sha256"], STK_STATE_CORPUS_SHA256)
        self.assertNotIn("NaN", first.decode("utf-8"))

    def test_every_case_propagates_on_its_frozen_grid(self):
        corpus = load_stk_state_corpus()

        for case in corpus.cases:
            with self.subTest(case_id=case.case_id):
                result = propagate_sgp4_teme(
                    case.elements,
                    simulation_epoch_utc=case.simulation_epoch_utc,
                    sample_times_s=case.sample_times_s,
                )
                self.assertEqual(len(result.samples), len(case.sample_times_s))
                self.assertEqual(result.element_set_id, case.element_set_id)
                for sample in result.samples:
                    self.assertTrue(
                        all(math.isfinite(value) for value in sample.position_teme_km)
                    )
                    self.assertTrue(
                        all(math.isfinite(value) for value in sample.velocity_teme_km_s)
                    )

    def test_hash_mismatch_and_unqualified_custom_source_fail_closed(self):
        default_path = Path(
            files("orbitserve.orbit").joinpath(
                "data",
                "stk_validation",
                "state_corpus.json",
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "state_corpus.json"
            copied.write_bytes(default_path.read_bytes() + b" ")

            with self.assertRaisesRegex(ValueError, "requires expected_sha256"):
                load_stk_state_corpus(copied)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_stk_state_corpus(
                    copied,
                    expected_sha256=STK_STATE_CORPUS_SHA256,
                )

    def test_invalid_coverage_claim_and_case_order_fail_closed(self):
        default_path = Path(
            files("orbitserve.orbit").joinpath(
                "data",
                "stk_validation",
                "state_corpus.json",
            )
        )
        payload = json.loads(default_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "state_corpus.json"

            invalid_claim = deepcopy(payload)
            invalid_claim["cases"][3]["elements"]["satellites"][0][
                "mean_motion_rev_per_day"
            ] = 15.0
            invalid_hash = _write_json(temp_path, invalid_claim)
            with self.assertRaisesRegex(ValueError, "invalid coverage claims"):
                load_stk_state_corpus(
                    temp_path,
                    expected_sha256=invalid_hash,
                )

            reordered = deepcopy(payload)
            reordered["cases"].reverse()
            reordered_hash = _write_json(temp_path, reordered)
            with self.assertRaisesRegex(ValueError, "frozen IDs and order"):
                load_stk_state_corpus(
                    temp_path,
                    expected_sha256=reordered_hash,
                )

    def test_duplicate_and_nonfinite_json_values_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "state_corpus.json"

            duplicate = (
                b'{"schema_version":"orbitserve.orbit.stk-state-corpus",'
                b'"schema_version":"duplicate","corpus_id":"x","cases":[]}\n'
            )
            temp_path.write_bytes(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_stk_state_corpus(
                    temp_path,
                    expected_sha256=hashlib.sha256(duplicate).hexdigest(),
                )

            nonfinite = (
                b'{"schema_version":"orbitserve.orbit.stk-state-corpus",'
                b'"corpus_id":"stk-state-pilot-2026-07","cases":[NaN]}\n'
            )
            temp_path.write_bytes(nonfinite)
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                load_stk_state_corpus(
                    temp_path,
                    expected_sha256=hashlib.sha256(nonfinite).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
