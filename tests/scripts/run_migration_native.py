import os, sys, unittest
from importlib.abc import MetaPathFinder


class RejectFlowSim(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "_flowsim" or fullname.startswith("_flowsim."):
            raise AssertionError("FlowSim execution is excluded by the user")


sys.meta_path.insert(0, RejectFlowSim())
from pathlib import Path

root = Path(
    os.environ.get("ASTER_TEST_ROOT", Path(__file__).resolve().parents[2])
).resolve()
sys.path.insert(0, str(root))
if os.environ.get("PYTHONPATH"):
    sys.path[:0] = os.environ["PYTHONPATH"].split(os.pathsep)
os.chdir(root)
from tests.external.native_cells.test_native_cells import NativeCellIntegrationTests
from tests.external.native_cells.test_timeline_modes import NativeTimelineModesTests

# Preserve the exact inputs before temporary test directories are cleaned.
import json, shutil


def preserve_inputs(method):
    def run(self, config):
        destination = (
            Path(os.environ["ORBITSERVE_NATIVE_REPORT_DIR"])
            / "inputs"
            / self._testMethodName
        )
        destination.mkdir(parents=True, exist_ok=True)
        refs = {}
        if config.physical_dataset:
            source = Path(config.physical_dataset)
            shutil.copytree(source, destination / "physical", dirs_exist_ok=True)
            refs[str(source)] = "physical"
        for cell in config.cells:
            if cell.memory_policy_file:
                source = Path(cell.memory_policy_file)
                shutil.copyfile(source, destination / (cell.id + "-policy.json"))
                refs[str(source)] = cell.id + "-policy.json"
        (destination / "references.json").write_text(json.dumps(refs, indent=2))
        return method(self, config)

    return run


NativeCellIntegrationTests.run_scene = preserve_inputs(
    NativeCellIntegrationTests.run_scene
)
NativeTimelineModesTests.run_scene = preserve_inputs(NativeTimelineModesTests.run_scene)


class NativeSimGridChecks(NativeCellIntegrationTests):
    def config(self, name, **kwargs):
        return super().config(
            name,
            **{
                **kwargs,
                "backend": "simgrid",
                "timeline_mode": "simgrid",
                "communication_mode": "flowsim",
            },
        )


def suite():
    cases = [NativeCellIntegrationTests("test_native_tp2_move_and_prefix_reuse_ns3")]
    cases += [
        NativeTimelineModesTests(n)
        for n in [
            "test_simgrid_all_communication",
            "test_simgrid_satellite_surrogate",
            "test_simgrid_cell_surrogate",
            "test_simgrid_kv_migration_and_prefix_reuse",
        ]
    ]
    cases += [
        NativeSimGridChecks(n)
        for n in [
            "test_cross_satellite_tp2_pp2_waits_for_recovery",
            "test_native_migration_failure_rolls_back_allocator",
            "test_heterogeneous_cells_and_hardware_derived_hbm_capacity",
            "test_native_moe_tp2_ep2",
        ]
    ]
    return unittest.TestSuite(cases)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    raise SystemExit(not result.wasSuccessful() or bool(result.skipped))
