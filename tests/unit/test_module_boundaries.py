import ast
import unittest
from pathlib import Path

import orbitserve.architecture as architecture
import orbitserve.orbit as orbit
import orbitserve.runtime as runtime
import orbitserve.workload as workload


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CORE = ROOT / "src" / "orbitserve"


class CoreModuleBoundaryTests(unittest.TestCase):
    def test_public_contracts_are_exported(self):
        for module, names in {
            architecture: ("HardwareSpec", "ResolvedArchitecture", "SatelliteHardwareProfile", "resolve_architecture"),
            orbit: ("OrbitTraceManifest", "OrbitTraceBundle", "generate_orbit_trace", "load_orbit_trace"),
            workload: ("WorkloadBundleRef", "WorkloadGenerationResult", "materialize_workload_bundle"),
            runtime: ("NetworkSpec", "NetworkState", "GlobalTimeline", "CommunicationExecutor"),
        }.items():
            with self.subTest(module=module.__name__):
                for name in names:
                    self.assertTrue(hasattr(module, name), f"{module.__name__}.{name} is not exported")

    def test_core_domain_imports_follow_one_way_allowlist(self):
        allowed = {
            "architecture": {"common"},
            "orbit": {"common"},
            "workload": {"common"},
            "serving": {"architecture", "common", "workload"},
            "runtime": {"common", "orbit"},
            "common": set(),
        }
        observed = {name: set() for name in allowed}
        for path in CORE.rglob("*.py"):
            source = path.relative_to(CORE).parts[0]
            if source not in allowed:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                target = self._target_domain(path, node)
                if target in allowed and target != source:
                    observed[source].add(target)
        for source, targets in observed.items():
            self.assertLessEqual(targets, allowed[source], f"unexpected {source} imports: {targets - allowed[source]}")

    def test_cross_domain_imports_use_public_package_interfaces(self):
        domains = {"architecture", "orbit", "workload", "serving", "runtime", "common"}
        violations = []
        for path in CORE.rglob("*.py"):
            source = path.relative_to(CORE).parts[0]
            if source not in domains:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom):
                    continue
                target = self._target_domain(path, node)
                if target not in domains or target == source:
                    continue
                module = node.module or ""
                if module.startswith("orbitserve."):
                    tail = module.split(".")[1:]
                else:
                    tail = module.split(".") if module else []
                if len(tail) > 1:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} imports internal module {module}"
                    )
        self.assertEqual(violations, [], "\n".join(violations))

    def test_all_source_consumers_use_core_public_package_interfaces(self):
        domains = {"architecture", "orbit", "workload", "serving", "runtime", "common"}
        violations = []
        for path in SRC.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.ImportFrom) or not node.module:
                    continue
                parts = node.module.split(".")
                if parts[0] == "orbitserve" and len(parts) >= 3 and parts[1] in domains:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {node.module}")
                elif node.level >= 2 and len(parts) >= 2 and parts[0] in domains:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports {node.module}")
        self.assertEqual(violations, [], "\n".join(violations))

    @staticmethod
    def _target_domain(path, node):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("orbitserve."):
                    return alias.name.split(".")[1]
            return None
        if not isinstance(node, ast.ImportFrom):
            return None
        if node.module and node.module.startswith("orbitserve."):
            return node.module.split(".")[1]
        if node.level:
            relative = list(path.relative_to(CORE).parts[:-1])
            keep = max(0, len(relative) - (node.level - 1))
            parts = relative[:keep]
            if node.module:
                parts.extend(node.module.split("."))
            return parts[0] if parts else None
        return None

    def test_removed_cross_domain_fields_are_absent(self):
        hardware_fields = architecture.SatelliteHardwareProfile.__dataclass_fields__
        for field in ("isl_contention_factor", "max_concurrent_isl_links", "laser_terminal_utilization"):
            self.assertNotIn(field, hardware_fields)
        for removed in ("ConstellationDesign", "RuntimeTraceBundle", "SimulationConfig", "run_simulation"):
            self.assertFalse(hasattr(orbit, removed))

    def test_workload_contract_is_unversioned(self):
        self.assertEqual(workload.WORKLOAD_BUNDLE_SCHEMA, "orbitserve.workload")
        self.assertNotIn("v2", workload.WORKLOAD_BUNDLE_SCHEMA.lower())


if __name__ == "__main__":
    unittest.main()
