import unittest
from types import SimpleNamespace

from orbitserve.runtime.setup.kv_capacity import derive_kv_cache_capacity_tokens


class KVCapacityDerivationTests(unittest.TestCase):
    def test_capacity_is_derived_from_architecture_storage_and_model_bytes(self):
        capacity = derive_kv_cache_capacity_tokens(
            {"architecture": {"storage_capacity_tb": 2.0}},
            SimpleNamespace(kv_cache_bytes_per_token=4000.0),
        )
        self.assertEqual(capacity, 500_000_000)

    def test_zero_storage_is_a_valid_zero_capacity(self):
        capacity = derive_kv_cache_capacity_tokens(
            {"architecture": {"storage_capacity_tb": 0.0}},
            SimpleNamespace(kv_cache_bytes_per_token=4000.0),
        )
        self.assertEqual(capacity, 0)

    def test_missing_storage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "storage_capacity_missing"):
            derive_kv_cache_capacity_tokens(
                {"architecture": {}},
                SimpleNamespace(kv_cache_bytes_per_token=4000.0),
            )

    def test_non_positive_bytes_per_token_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bytes_per_token_invalid"):
            derive_kv_cache_capacity_tokens(
                {"architecture": {"storage_capacity_tb": 1.0}},
                SimpleNamespace(kv_cache_bytes_per_token=0.0),
            )


if __name__ == "__main__":
    unittest.main()
