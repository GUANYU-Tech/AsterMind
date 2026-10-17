"""Test suite 2: diff_formula.py consistency — forward matches formula.py, gradients match numerical diff."""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kernel.hardware_spec import HWSpec
from src.kernel.validation.legacy_kernel_base import PredictionInput
from src.kernel.formula import PerformanceFormula
from src.dse.diff_formula import DiffPerformanceFormula


HW = HWSpec("configs/hardware/nvidia_a800.yaml")
FWD = PerformanceFormula(HW)
DIFF = DiffPerformanceFormula(HW)


def _inp(demand, **kwargs):
    defaults = dict(
        demand_per_resource=demand,
        eta=0.8,
        active_units=HW.num_compute_units,
        sm_freq_hz=HW.nominal_sm_freq_hz,
        dram_freq_hz=HW.nominal_dram_freq_hz,
        l1_bypass=False,
        duration_ns=1000.0,
        grid_size=HW.num_compute_units,
        kernel_name="test",
        is_non_tensor=False,
        pipeline_floor_ns=0.0,
    )
    defaults.update(kwargs)
    return PredictionInput(**defaults)


CASES = [
    {"dram": 1e9},
    {"l2_cache": 1e8, "dram": 5e8},
    {"compute_tensor": 1e12},
    {"dram": 1e10},
]


class TestDiffFormulaForwardConsistency:
    @pytest.mark.parametrize("demand", CASES)
    def test_forward_equals_formula(self, demand):
        inp = _inp(demand)
        pf = FWD.predict_kernel(inp)
        pd = DIFF.predict_kernel(inp)
        assert abs(pf.t_total_s - pd.t_total_s) < 1e-12, (
            f"forward mismatch: formula={pf.t_total_s:.6e} diff={pd.t_total_s:.6e}"
        )


class TestDiffFormulaGradients:
    def _numerical_grad(self, demand, role, eps=1.0):
        """Numerical gradient d(t_total)/d(demand[role]) via finite difference."""
        inp1 = _inp(dict(demand))
        p1 = DIFF.predict_kernel(inp1)

        d2 = dict(demand)
        d2[role] = d2.get(role, 0) + eps
        inp2 = _inp(d2)
        p2 = DIFF.predict_kernel(inp2)

        return (p2.t_total_s - p1.t_total_s) / eps

    @pytest.mark.parametrize("demand,role", [
        ({"dram": 1e9}, "dram"),
        ({"l2_cache": 1e8, "dram": 5e8}, "dram"),
        ({"compute_tensor": 1e12}, "compute_tensor"),
    ])
    def test_analytical_grad_matches_numerical(self, demand, role):
        inp = _inp(demand)
        pd = DIFF.predict_kernel(inp)

        analytical = pd.grad_demand.get(role, 0.0)
        numerical = self._numerical_grad(demand, role, eps=1.0)

        if abs(numerical) < 1e-20:
            # Both should be ~0
            assert abs(analytical) < 1e-15
        else:
            rel_err = abs(analytical - numerical) / abs(numerical)
            assert rel_err < 0.01, (
                f"Gradient mismatch for demand[{role}]: "
                f"analytical={analytical:.6e}, numerical={numerical:.6e}, "
                f"rel_err={rel_err:.3f}"
            )

    def test_l2_serial_gradient_fires_when_active(self):
        """When t_l2_serial > t_bottleneck, grad_demand['l2_cache'] must be nonzero."""
        inp = _inp(
            {"l2_cache": 5_000_000},
            is_non_tensor=True,
            grid_size=2,
            active_units=2,
            enable_l2_serial_latency=True,
        )
        pd = DIFF.predict_kernel(inp)
        p_fwd = FWD.predict_kernel(inp)
        inp_off = _inp(
            {"l2_cache": 5_000_000},
            is_non_tensor=True,
            grid_size=2,
            active_units=2,
            enable_l2_serial_latency=False,
        )
        p_off = FWD.predict_kernel(inp_off)
        if p_fwd.t_total_s > p_off.t_total_s + 1e-20:
            # additive fired; gradient should be nonzero
            assert abs(pd.grad_demand.get("l2_cache", 0.0)) > 0, (
                "l2_serial additive fired but grad_demand['l2_cache'] is 0"
            )


class TestL2SizeGradient:
    """P48: grad_l2_size = ∂t/∂(L2_size) via stack-distance hit curve."""

    def test_dram_bound_chain_rule_matches_fd(self):
        """DRAM-bound kernel with a known d_dram_d_l2size: analytic grad_l2_size
        must equal FD of t_total when L2 size moves (dram changes by slope·ΔC)."""
        dram_bytes = 200e6
        slope = -3.0  # d(dram_bytes)/d(L2_bytes)
        inp = _inp({"dram": dram_bytes}, is_non_tensor=False,
                   d_dram_d_l2size=slope)
        pd = DIFF.predict_kernel(inp)
        assert pd.bottleneck_group == "dram"
        analytic = pd.grad_l2_size
        assert analytic != 0.0

        dC = HW.l2_size_bytes * 0.01

        def t_with_dram(d):
            return FWD.predict_kernel(_inp({"dram": d}, is_non_tensor=False)).t_total_s
        fd = (t_with_dram(dram_bytes + slope * dC)
              - t_with_dram(dram_bytes - slope * dC)) / (2 * dC)
        assert abs(analytic - fd) / abs(fd) < 0.01

    def test_zero_slope_gives_zero_grad(self):
        """No L2-size sensitivity (d_dram_d_l2size=0) → grad_l2_size = 0."""
        inp = _inp({"dram": 200e6}, is_non_tensor=False, d_dram_d_l2size=0.0)
        pd = DIFF.predict_kernel(inp)
        assert pd.grad_l2_size == 0.0

    def test_l2_bound_kernel_no_l2_size_grad(self):
        """When L2-bound (not DRAM-bound), grad_l2_size stays 0 even with a
        nonzero slope — L2 demand (total accesses) is L2-size-independent."""
        inp = _inp({"l2_cache": 200e6}, is_non_tensor=True,
                   d_dram_d_l2size=-3.0)
        pd = DIFF.predict_kernel(inp)
        assert pd.grad_l2_size == 0.0
