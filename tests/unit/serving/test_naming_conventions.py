import unittest

from orbitserve.serving import ServingSpec


class ServingNamingTests(unittest.TestCase):
    def test_aliases_are_normalized_to_canonical_model_name(self):
        spec = ServingSpec.from_dict({"model": {"name": "Llama-3.3-70B"}})
        self.assertEqual(spec.model.name, "llama3.3-70b")

    def test_model_structure_overrides_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "accepts only name/preset"):
            ServingSpec.from_dict({
                "model": {"name": "deepseek-v3", "hidden_size": 1}
            })


if __name__ == "__main__":
    unittest.main()
