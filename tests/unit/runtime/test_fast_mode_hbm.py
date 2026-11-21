"""Check the H200 HBM input used by the grouped Roofline predictor."""

from unittest import TestCase

from astermind.common.bootstrap import activate


class FastModeHbmTests(TestCase):
    def test_h200_read_rate_comes_from_the_native_profile(self):
        root = activate()
        from astermind.engine.compute.fast_mode.hardware import Hardware
        from src.kernel.hardware_spec import HWSpec

        spec = HWSpec(root / "configs/hardware/nvidia_h200.yaml")
        hardware = Hardware.from_hardware_spec(spec)
        expected = spec.dram_read_capacity_per_clk * spec.nominal_dram_freq_hz
        self.assertEqual(hardware.hbm_bytes, expected)
        self.assertAlmostEqual(hardware.hbm_bytes / 1e12, 4.621128079772083)
        self.assertEqual(hardware.roofline(0, expected), 1.)
        self.assertEqual(Hardware().hbm_bytes, expected)
        self.assertAlmostEqual(hardware.roofline(hardware.bf16_flops, 0), 1 / .70)
        self.assertAlmostEqual(hardware.roofline(hardware.fp8_flops, 0, fp8=True), 1 / .70)

    def test_portable_profiles_do_not_inherit_h200_tensor_rates(self):
        root = activate()
        from astermind.engine.compute.fast_mode.hardware import Hardware
        from src.kernel.hardware_spec import HWSpec

        for hardware_id in ("nvidia_a800", "ascend_910b", "metax_c500"):
            with self.subTest(hardware_id=hardware_id):
                spec = HWSpec(root / f"configs/hardware/{hardware_id}.yaml")
                hardware = Hardware.from_hardware_spec(spec)
                peak = spec.resource_by_role("compute_tensor_bf16").capacity
                self.assertEqual(hardware.bf16_flops, peak)
                self.assertEqual(hardware.hbm_bytes, spec.dram_read_capacity_per_clk * spec.nominal_dram_freq_hz)
                self.assertIsNone(hardware.fp8_flops)
                with self.assertRaisesRegex(ValueError, "no FP8 tensor profile"):
                    hardware.roofline(peak, 1024, fp8=True)
