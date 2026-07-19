import sys
import unittest
from pathlib import Path


from orbitserve.architecture.single_satellite_cost import estimate_single_satellite_cost
from orbitserve.architecture.single_satellite_cost.models import (
    CommunicationConfig,
    ComputeConfig,
    SatelliteDesignLimits,
    PowerConfig,
    SingleSatelliteConfig,
)


class FeasibilityTests(unittest.TestCase):
    def test_mass_and_power_limits_create_violations(self):
        req = SatelliteDesignLimits(
            payload_power_limit_w=100,
            satellite_mass_limit_kg=10,
        )
        config = SingleSatelliteConfig(
            compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000)
        )
        result = estimate_single_satellite_cost(req, config)
        self.assertFalse(result.feasible)
        self.assertTrue(any("payload peak power" in item for item in result.violations))
        self.assertTrue(any("satellite mass" in item for item in result.violations))

    def test_missing_purchase_cost_is_infeasible(self):
        req = SatelliteDesignLimits()
        result = estimate_single_satellite_cost(req, SingleSatelliteConfig())
        self.assertFalse(result.feasible)
        self.assertTrue(any("purchase_cost_per_unit_usd" in item for item in result.violations))

    def test_negative_parameter_is_infeasible(self):
        req = SatelliteDesignLimits(payload_power_limit_w=-1)
        result = estimate_single_satellite_cost(
            req,
            SingleSatelliteConfig(
                compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000)
            ),
        )
        self.assertFalse(result.feasible)
        self.assertTrue(any("design_limits.payload_power_limit_w" in item for item in result.violations))

    def test_positive_bandwidth_requires_comm_bandwidth_mass_and_power(self):
        req = SatelliteDesignLimits()
        config = SingleSatelliteConfig(
            compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000),
            communication=CommunicationConfig(
                ground_link_bandwidth_gbps=200,
                ground_link_mass_kg=0,
                ground_link_power_w=0,
                isl_terminal_count=1,
                isl_terminal_bandwidth_gbps=200,
                isl_terminal_mass_kg=0,
                isl_terminal_power_w=0,
            ),
        )
        result = estimate_single_satellite_cost(req, config)
        self.assertFalse(result.feasible)
        self.assertTrue(any("positive ground bandwidth" in item for item in result.violations))
        self.assertTrue(any("positive ISL bandwidth" in item for item in result.violations))

    def test_direct_link_cost_mode_requires_coefficients_for_positive_bandwidth(self):
        req = SatelliteDesignLimits()
        config = SingleSatelliteConfig(
            compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000),
            communication=CommunicationConfig(
                ground_link_bandwidth_gbps=100,
                ground_link_mass_kg=28,
                ground_link_power_w=100,
                isl_terminal_count=1,
                isl_terminal_bandwidth_gbps=100,
                isl_terminal_mass_kg=10,
                isl_terminal_power_w=100,
                use_direct_link_costs=True,
            ),
        )
        result = estimate_single_satellite_cost(req, config)
        self.assertFalse(result.feasible)
        self.assertTrue(any("ground_cost_per_gbps_usd" in item for item in result.violations))
        self.assertTrue(any("isl_cost_per_gbps_usd" in item for item in result.violations))

    def test_empty_design_limits_keep_explicit_accelerator_quantity(self):
        config = SingleSatelliteConfig(
            compute=ComputeConfig(
                profile_name="h200_sxm",
                accelerator_count=1,
                purchase_cost_per_unit_usd=50000,
            ),
            communication=CommunicationConfig(
                ground_link_bandwidth_gbps=0,
                ground_link_mass_kg=0,
                ground_link_power_w=0,
            ),
        )
        result = estimate_single_satellite_cost(SatelliteDesignLimits(), config)
        self.assertTrue(result.feasible)
        self.assertEqual(result.provenance["N_acc"], 1)


    def test_fraction_like_parameters_above_one_are_infeasible(self):
        config = SingleSatelliteConfig(
            compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000),
            
        )
        result = estimate_single_satellite_cost(SatelliteDesignLimits(), config, orbit_energy_condition={"sunlit_fraction": 2.0, "eclipse_duration_min": 35.0})
        self.assertFalse(result.feasible)
        self.assertTrue(any("sunlit_fraction" in item for item in result.violations))

    def test_pmad_efficiency_above_one_is_infeasible(self):
        config = SingleSatelliteConfig(
            compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000),
            power=PowerConfig(eta_pmad=2.0),
        )
        result = estimate_single_satellite_cost(SatelliteDesignLimits(), config)
        self.assertFalse(result.feasible)
        self.assertTrue(any("eta_pmad" in item for item in result.violations))


if __name__ == "__main__":
    unittest.main()
