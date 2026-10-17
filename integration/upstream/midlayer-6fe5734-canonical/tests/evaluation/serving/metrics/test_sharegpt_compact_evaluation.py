import pytest

from evaluation.serving.metrics.sharegpt_compact_evaluation import (
    _known_not_implemented_launch_gap_reason,
)


@pytest.mark.parametrize(
    "message",
    (
        "CANN 8.5.1 public RoPE GatherV3 is unsupported: its normalized "
        "launch contract declares the two calls but lacks the selected "
        "GatherV3 program and typed tiling geometry",
        "CANN GatherV3 Robin multi-core source supports at most five equal "
        "X scans",
    ),
)
def test_obsolete_ascend_gather_v3_gaps_fail_closed(message):
    assert _known_not_implemented_launch_gap_reason(
        NotImplementedError(message), "ascend_910b") is None


from pathlib import Path  # noqa: E402

from evaluation.serving.metrics import sharegpt_compact_evaluation as compact  # noqa: E402
from src.serving.host_timing import (  # noqa: E402
    load_host_launch_profile,
    load_host_request_path_profile,
    load_host_runtime_prepare_profile,
    load_scheduler_host_profile,
)

_ROOT = Path(__file__).resolve().parents[4]


def test_official_matrix_excludes_h200_unless_a_version_is_selected():
    default_cells = [
        cell for _config, cells in compact.official_metrics_suite_runs(_ROOT)
        for cell in cells
    ]
    assert len(default_cells) == 45
    assert {cell.hardware.hardware_id for cell in default_cells} == {
        "nvidia_a800", "metax_c500", "ascend_910b"}


@pytest.mark.parametrize("version", ("0.30.0", "0.11.0"))
def test_selected_h200_version_adds_its_fifteen_cells(version):
    cells = [
        cell for _config, cells in compact.official_metrics_suite_runs(
            _ROOT, nvidia_h200_vllm_version=version)
        for cell in cells
    ]
    h200 = sorted(cell.cell_id for cell in cells
                  if cell.hardware.hardware_id == "nvidia_h200")
    assert len(cells) == 60
    assert len(h200) == 15
    assert "nvidia_h200/qwen3-30b-a3b/tp2_ep2_qps16" in h200
    assert "nvidia_h200/qwen3-8b/tp1_qps64" in h200


def test_unknown_h200_version_fails_closed():
    with pytest.raises(ValueError, match="unknown H200 vLLM version"):
        compact.official_metrics_suite_runs(_ROOT, nvidia_h200_vllm_version="0.24.0")


@pytest.mark.parametrize("version", ("0.30.0", "0.11.0"))
def test_h200_host_profiles_are_bound_by_case_runtime_version(version):
    paths = compact.case_host_profile_paths(_ROOT, "nvidia_h200", version)
    assert set(paths) == {
        "host_launch", "scheduler_host", "host_request_path", "host_runtime_prepare"}
    tag = {"0.30.0": "vllm030", "0.11.0": "vllm011"}[version]
    assert all(tag in path.name for path in paths.values())
    loaded = (
        load_host_launch_profile(paths["host_launch"]),
        load_scheduler_host_profile(paths["scheduler_host"]),
        load_host_request_path_profile(paths["host_request_path"]),
        load_host_runtime_prepare_profile(paths["host_runtime_prepare"]),
    )
    assert {profile.hardware_id for profile in loaded} == {"nvidia_h200"}
    assert loaded[1].vllm_version == version


def test_h200_host_profiles_reject_unmeasured_versions_and_hardware():
    with pytest.raises(ValueError, match="no nvidia_h200 Host profiles"):
        compact.case_host_profile_paths(_ROOT, "nvidia_h200", "0.24.0")
    with pytest.raises(ValueError, match="not versioned"):
        compact.case_host_profile_paths(_ROOT, "nvidia_a800", "0.11.0")


def test_default_profile_discovery_keeps_one_profile_per_published_hardware():
    launch = compact.resolve_host_launch_profile_paths(_ROOT, (), None)
    scheduler = compact.resolve_scheduler_host_profile_paths(_ROOT, (), None)
    for profiles in (launch, scheduler):
        assert set(profiles) == {"nvidia_a800", "metax_c500", "ascend_910b"}
