import ast
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import yaml

import src
from src.kernel.hardware_ids import CANONICAL_HARDWARE_IDS
from src.kernel.launch_config.gemm_launch_selection import (
    gemm_launch_schema_version,
)


def test_production_import_boundaries():
    forbidden = {
        "evaluation", "dataset_pipeline", "tools", "src.serve_trace",
        "src.serve_predict", "src.dse", "scripts", "archive",
    }
    for path in list(Path("src/kernel").rglob("*.py")) + list(Path("src/serving").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = ["" if node.module is None else node.module]
            else:
                modules = []
            assert not any(
                module == prefix or module.startswith(prefix + ".")
                for module in modules for prefix in forbidden
            ), (path, modules)


def test_dataset_pipeline_src_imports_are_explicit_contracts_or_launch_derivation():
    allowed = {
        "src.kernel.dataset_storage",
        "src.kernel.hardware_ids",
        "src.kernel.hardware_spec",
        "src.kernel.kernel_spec",
        "src.kernel.launch_config.ascend_910b.gemm_launch",
        "src.kernel.launch_config.flash_attn_workload",
        "src.kernel.launch_config.fused_moe_tiling",
        "src.kernel.launch_config.gemm_launch_selection",
        "src.kernel.launch_config.gemm_workload",
        "src.kernel.launch_config.select_launch_config",
        "src.kernel.launch_config.tiling_spec",
    }
    violations = []
    for path in Path("dataset_pipeline").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = ["" if node.module is None else node.module]
            else:
                modules = []
            for module in modules:
                if ((module == "src" or module.startswith("src."))
                        and module not in allowed):
                    violations.append(f"{path}:{node.lineno}:{module}")
    assert not violations, (
        "dataset_pipeline may import only explicit src contracts or canonical "
        "launch derivation modules:\n" + "\n".join(sorted(violations)))


def test_kernel_never_imports_serving():
    for path in Path("src/kernel").rglob("*.py"):
        assert "src.serving" not in path.read_text(), path


def test_hardware_launch_packages_do_not_import_each_other():
    packages = ("nvidia_a800", "metax_c500", "ascend_910b")
    for owner in packages:
        for path in (Path("src/kernel/launch_config") / owner).glob("*.py"):
            content = path.read_text()
            for other in set(packages) - {owner}:
                assert f"launch_config.{other}" not in content, path


def test_hardware_launch_packages_expose_the_same_selector_surface():
    packages = ("nvidia_a800", "metax_c500", "ascend_910b")
    compute_modules = {
        "gemm_launch": "select_gemm_launch",
        "flash_attn_launch": "select_flash_attn_launch",
        "fused_moe_launch": "select_fused_moe_launch",
        "vector_launch": "select_vector_launch",
    }
    compute_parameters = (
        "shape", "dtype", "variant", "attributes", "hardware")
    from src.kernel.launch_config.collective_selection import (
        CollectiveRuleSet,
    )

    for owner in packages:
        package_path = Path("src/kernel/launch_config") / owner
        # A package holds its compute selectors plus declarations; the
        # collective library is data (``COLLECTIVE_RULES``) for the one
        # shared engine, not a per-hardware selector module, and the
        # declaration tables live in their own modules so ``__init__``
        # stays a thin re-export.
        assert {path.stem for path in package_path.glob("*.py")} == {
            "__init__", *compute_modules,
            "collective_rules", "protobridge_inputs"}
        package = importlib.import_module(f"src.kernel.launch_config.{owner}")
        assert isinstance(package.COLLECTIVE_RULES, CollectiveRuleSet)
        assert package.COLLECTIVE_RULES.hardware_id == owner
        for module_name, selector_name in compute_modules.items():
            module = importlib.import_module(
                f"src.kernel.launch_config.{owner}.{module_name}")
            selector = getattr(module, selector_name)
            signature = inspect.signature(selector)
            assert tuple(signature.parameters) == compute_parameters
            assert signature.parameters["hardware"].default is inspect.Parameter.empty
        fused_moe_module = importlib.import_module(
            f"src.kernel.launch_config.{owner}.fused_moe_launch")
        recipe = fused_moe_module.select_fused_moe_recipe
        assert tuple(inspect.signature(recipe).parameters) == (
            "shape", "hardware")


def test_generic_predictors_and_serving_lowering_do_not_branch_on_hardware_ids():
    roots = (
        Path("src/kernel/predictors"),
        Path("src/serving/kernel_lowering"),
    )
    exceptions = {Path("src/kernel/predictors/pipeweave_adapter.py")}
    violations = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path in exceptions:
                continue
            content = path.read_text()
            for hardware_id in CANONICAL_HARDWARE_IDS:
                if hardware_id in content:
                    violations.append(f"{path}:{hardware_id}")
    assert not violations, (
        "hardware-specific dispatch belongs in HWSpec capabilities or launch "
        "packages:\n" + "\n".join(sorted(violations)))


def test_hardware_dataset_providers_do_not_import_each_other():
    providers = (
        "ascend_910b", "metax_c500", "nvidia_a800", "samsung_hbm_pim")
    root = Path("dataset_pipeline/kernel_dataset")
    for owner in providers:
        content = (root / f"{owner}.py").read_text()
        for other in set(providers) - {owner}:
            assert f"kernel_dataset.{other}" not in content, owner


def test_kernel_evaluation_and_serving_share_the_production_predict_api():
    from evaluation.kernel import evaluation_runner
    from src.kernel import kernel_base
    from src.serving.kernel_lowering import kernel_predict

    assert evaluation_runner.predict_production is kernel_base.predict
    assert kernel_predict.predict is kernel_base.predict


def test_every_src_module_imports_without_optional_frameworks():
    modules = sorted(
        module.name
        for module in pkgutil.walk_packages(src.__path__, src.__name__ + ".")
    )
    assert modules
    for module in modules:
        importlib.import_module(module)


def test_serving_evaluation_never_imports_kernel_evaluation():
    for path in Path("evaluation/serving").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = ["" if node.module is None else node.module]
            else:
                modules = []
            assert not any(
                module == "evaluation.kernel"
                or module.startswith("evaluation.kernel.")
                for module in modules
            ), (path, modules)


def test_hardware_identity_paths_and_active_metadata_are_canonical():
    hardware_config_ids = {
        path.stem for path in Path("configs/hardware").glob("*.yaml")
    }
    assert hardware_config_ids == CANONICAL_HARDWARE_IDS
    identity_directories = (
        Path("configs/hardware"),
        Path("configs/dataset"),
    )
    for directory in identity_directories:
        entries = [
            entry for entry in directory.iterdir()
            if not entry.name.startswith(".")
            and not (directory == Path("configs/dataset") and entry.is_dir())
        ]
        assert entries, directory
        for entry in entries:
            identity = entry.stem if entry.is_file() else entry.name
            assert identity in CANONICAL_HARDWARE_IDS, entry

    kernel_dataset_entries = sorted(
        Path("kernel_dataset").glob("*/*.json.zst"))
    assert kernel_dataset_entries
    for entry in kernel_dataset_entries:
        assert len(entry.parts) >= 3, entry
        assert entry.parts[1] in CANONICAL_HARDWARE_IDS, entry

    launch_identities = {
        entry.name
        for entry in Path("src/kernel/launch_config").iterdir()
        if entry.is_dir() and entry.name != "__pycache__"
    }
    assert launch_identities == {
        "nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b"}

    for path in Path("configs/hardware").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        assert payload["hardware_id"] == path.stem
    for path in Path("configs/dataset").glob("*.yaml"):
        payload = yaml.safe_load(path.read_text())
        assert payload["hardware_id"] == path.stem

    for path in Path("launch_data").glob(
            "*/gemm_launch_selection.manifest.json"):
        payload = json.loads(path.read_text())
        assert payload["schema_version"] == gemm_launch_schema_version(
            payload["hardware_id"]), path
        assert payload["artifact_type"] == "gemm_launch_selection", path
        assert payload["hardware_id"] == path.parent.name, path
        assert payload["software_revision"], path
        assert payload["evidence_sha256"], path
        assert payload["generated_by"], path
        assert payload["duration_consumed"] is False, path
        assert (path.parent / "gemm_launch_selection.csv.zst").is_file(), path
        current_generator = (
            Path("scripts/launch_data")
            / f"{payload['hardware_id']}_launch_generator.py"
        )
        assert current_generator.is_file(), path

    launch_generators = set(Path("scripts/launch_data").glob("*.py"))
    assert {path.name for path in launch_generators} == {
        "ascend_910b_launch_generator.py",
        "build_gemm_workload_candidates.py",
        "metax_c500_launch_generator.py",
        "nvidia_a800_launch_generator.py",
        "nvidia_h200_launch_generator.py",
    }

    dataset_pipeline_hardware_ids = {
        "ascend_910b",
        "metax_c500",
        "nvidia_a800",
        "nvidia_h200",
        "samsung_hbm_pim",
    }
    dataset_script_groups = {
        path.name
        for path in Path("scripts/kernel_dataset").iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert dataset_script_groups == dataset_pipeline_hardware_ids
    assert dataset_pipeline_hardware_ids == {
        path.parent.name
        for path in Path("scripts/kernel_dataset").glob("*/build_dataset.py")
    }
    for path in Path("scripts/kernel_dataset").glob("*/*.py"):
        if path.name != "build_dataset.py":
            assert path.name.endswith("_runner.py"), path

    configs = {
        path.stem: yaml.safe_load(path.read_text())
        for path in Path("configs/dataset").glob("*.yaml")
    }
    dataset_pipeline_ids = CANONICAL_HARDWARE_IDS - {"nvidia_h200"}
    # H200 has capture runners but no published canonical dataset yet, so it
    # has no dataset config.
    assert set(configs) == dataset_pipeline_ids
    for hardware_id in ("ucb_gemmini", "groq_lpu"):
        assert configs[hardware_id]["kernel"]["pipeline"] is None
        assert configs[hardware_id]["serving"]["suites"] == {}
    pim_workloads = configs["samsung_hbm_pim"]["kernel"]["pipeline"]["workloads"]
    assert set(pim_workloads) == {
        "gemm", "eager_attn", "unfused_moe", "vector"}
    metax_workloads = configs["metax_c500"]["kernel"]["pipeline"]["workloads"]
    assert set(metax_workloads) == {
        "gemm", "flash_attn", "fused_moe", "vector"}
    assert Path(
        "scripts/kernel_dataset/metax_c500/flash_attn_runner.py").is_file()
    assert Path(
        "scripts/kernel_dataset/metax_c500/fused_moe_runner.py").is_file()
    assert Path(
        "scripts/kernel_dataset/metax_c500/vector_runner.py").is_file()
    for hardware_id in ("ascend_910b", "nvidia_a800"):
        workloads = configs[hardware_id]["kernel"]["pipeline"]["workloads"]
        assert set(workloads) == {"gemm", "flash_attn", "fused_moe", "vector"}
        for kernel in workloads:
            runner = Path(
                "scripts/kernel_dataset") / hardware_id / f"{kernel}_runner.py"
            assert runner.name.endswith("_runner.py"), runner
            assert runner.is_file(), runner
