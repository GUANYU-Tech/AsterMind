import unittest

from orbitserve.architecture.single_satellite_cost import estimate_single_satellite_cost
from orbitserve.architecture.single_satellite_cost.models import (
    ComputeConfig,
    CostConfig,
    SatelliteDesignLimits,
    PowerConfig,
    SingleSatelliteConfig,
    ThermalConfig,
)


def config_with(power=None, thermal=None, cost=None):
    return SingleSatelliteConfig(
        compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000),
        power=power or PowerConfig(),
        thermal=thermal or ThermalConfig(),
        cost=cost or CostConfig(),
    )


def orbit_condition(*, sunlit_fraction=0.64, eclipse_duration_min=35.0):
    return {
        "sunlit_fraction": sunlit_fraction,
        "eclipse_duration_min": eclipse_duration_min,
    }


class PowerThermalLaunchTests(unittest.TestCase):
    def setUp(self):
        self.req = SatelliteDesignLimits()

    def test_lower_dod_increases_battery_requirement(self):
        base = estimate_single_satellite_cost(
            self.req, config_with(power=PowerConfig(dod_max=0.30)),
            orbit_energy_condition=orbit_condition(),
        )
        conservative = estimate_single_satellite_cost(
            self.req, config_with(power=PowerConfig(dod_max=0.20)),
            orbit_energy_condition=orbit_condition(),
        )
        self.assertGreater(conservative.E_battery_req_wh, base.E_battery_req_wh)

    def test_higher_orbit_sunlit_fraction_reduces_solar_requirement(self):
        base = estimate_single_satellite_cost(
            self.req, config_with(), orbit_energy_condition=orbit_condition(sunlit_fraction=0.64)
        )
        high_sun = estimate_single_satellite_cost(
            self.req, config_with(), orbit_energy_condition=orbit_condition(sunlit_fraction=0.90)
        )
        self.assertLess(high_sun.P_solar_req_w, base.P_solar_req_w)

    def test_lower_charge_efficiency_increases_shadow_recharge_solar_requirement(self):
        ideal = estimate_single_satellite_cost(
            self.req, config_with(power=PowerConfig(battery_charge_efficiency=1.0)),
            orbit_energy_condition=orbit_condition(),
        )
        lossy = estimate_single_satellite_cost(
            self.req, config_with(power=PowerConfig(battery_charge_efficiency=0.5)),
            orbit_energy_condition=orbit_condition(),
        )
        self.assertGreater(lossy.P_solar_req_w, ideal.P_solar_req_w)

    def test_ideal_storage_piecewise_sizing_matches_orbit_average_energy_balance(self):
        power = PowerConfig(battery_charge_efficiency=1.0, battery_discharge_efficiency=1.0)
        condition = orbit_condition()
        result = estimate_single_satellite_cost(
            self.req, config_with(power=power), orbit_energy_condition=condition
        )
        expected = result.P_avg_w * (1.0 + power.power_margin_fraction) / (
            power.eta_pmad * condition["sunlit_fraction"] * power.pv_eol_fraction
        )
        self.assertAlmostEqual(result.P_solar_req_w, expected)

    def test_lower_discharge_efficiency_increases_battery_and_solar_requirements(self):
        ideal = estimate_single_satellite_cost(
            self.req, config_with(power=PowerConfig(battery_discharge_efficiency=1.0)),
            orbit_energy_condition=orbit_condition(),
        )
        lossy = estimate_single_satellite_cost(
            self.req, config_with(power=PowerConfig(battery_discharge_efficiency=0.5)),
            orbit_energy_condition=orbit_condition(),
        )
        self.assertGreater(lossy.E_battery_req_wh, ideal.E_battery_req_wh)
        self.assertGreater(lossy.P_solar_req_w, ideal.P_solar_req_w)

    def test_bad_thermal_denominator_marks_infeasible(self):
        bad = ThermalConfig(radiator_temperature_k=150, environment_heat_flux_w_m2=150)
        result = estimate_single_satellite_cost(
            self.req, config_with(thermal=bad), orbit_energy_condition=orbit_condition()
        )
        self.assertFalse(result.feasible)
        self.assertTrue(any("radiator heat rejection" in item for item in result.violations))

    def test_launch_cost_sweep_is_monotonic(self):
        low = estimate_single_satellite_cost(
            self.req, config_with(cost=CostConfig(launch_cost_per_kg_usd=500)),
            orbit_energy_condition=orbit_condition(),
        )
        high = estimate_single_satellite_cost(
            self.req, config_with(cost=CostConfig(launch_cost_per_kg_usd=5000)),
            orbit_energy_condition=orbit_condition(),
        )
        self.assertGreater(high.C_breakdown_usd["launch"], low.C_breakdown_usd["launch"])
        self.assertGreater(high.C_sat_usd, low.C_sat_usd)


if __name__ == "__main__":
    unittest.main()
