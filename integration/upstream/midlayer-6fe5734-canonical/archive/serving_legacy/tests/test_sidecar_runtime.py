import importlib
import json
import os
import sys
from pathlib import Path


def _clear_vllm_modules():
    for name in list(sys.modules):
        if name == "vllm" or name.startswith("vllm."):
            sys.modules.pop(name, None)


def _write_package(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.py"
    if not init.exists():
        init.write_text("")


def _make_fake_vllm_tree(root: Path):
    for rel in [
        "vllm",
        "vllm/v1",
        "vllm/v1/core",
        "vllm/v1/core/sched",
        "vllm/distributed",
        "vllm/distributed/kv_transfer",
        "vllm/distributed/kv_transfer/kv_connector",
        "vllm/distributed/kv_transfer/kv_connector/v1",
        "vllm/distributed/kv_transfer/kv_connector/v1/p2p",
    ]:
        _write_package(root / rel)

    (root / "vllm/v1/core/sched/scheduler.py").write_text(
        """
class Output:
    num_scheduled_tokens = {"req-0": 4}
    total_num_scheduled_tokens = 4
    finished_req_ids = {"req-0"}

class Scheduler:
    def schedule(self):
        return Output()
""".lstrip()
    )
    (root / "vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py").write_text(
        """
class P2pNcclConnector:
    def __init__(self):
        self.is_producer = True
        self.chunked_prefill = {"keep"}

    def build_connector_meta(self, scheduler_output):
        return list(scheduler_output.scheduled_cached_reqs.req_ids)
""".lstrip()
    )


def test_sidecar_finder_uses_find_spec_and_patches_scheduler(tmp_path, monkeypatch):
    _clear_vllm_modules()
    _make_fake_vllm_tree(tmp_path)
    sidecar_path = tmp_path / "scheduler.jsonl"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CUDA_SIDECAR_PATH", str(sidecar_path))

    import src.serve_trace._sidecar_patch as patch
    assert hasattr(patch._SidecarFinder, "find_spec")
    assert "find_module" not in patch._SidecarFinder.__dict__
    assert "load_module" not in patch._SidecarFinder.__dict__

    patch.install_sidecar_patch()
    mod = importlib.import_module("vllm.v1.core.sched.scheduler")
    assert getattr(mod.Scheduler.schedule, "_sidecar_patched", False)

    output = mod.Scheduler().schedule()
    assert output.total_num_scheduled_tokens == 4
    records = [json.loads(line) for line in sidecar_path.read_text().splitlines()]
    assert records[0]["scheduled"] == {"req-0": 4}


def test_sidecar_finder_patches_p2p_connector_and_restores_cached_arrays(tmp_path, monkeypatch):
    _clear_vllm_modules()
    _make_fake_vllm_tree(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("CUDA_SIDECAR_PATH", str(tmp_path / "scheduler.jsonl"))

    import src.serve_trace._sidecar_patch as patch
    patch.install_sidecar_patch()
    mod = importlib.import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector")
    assert getattr(mod.P2pNcclConnector, "_p2p_patched", False)

    class CachedReqs:
        req_ids = ["keep", "drop"]
        num_computed_tokens = [1, 2]
        new_block_ids = [[10], [20]]
        resumed_from_preemption = [False, True]

    class SchedulerOutput:
        scheduled_cached_reqs = CachedReqs()

    output = SchedulerOutput()
    result = mod.P2pNcclConnector().build_connector_meta(output)
    assert result == ["keep"]
    assert output.scheduled_cached_reqs.req_ids == ["keep", "drop"]
    assert output.scheduled_cached_reqs.num_computed_tokens == [1, 2]
    assert output.scheduled_cached_reqs.new_block_ids == [[10], [20]]
    assert output.scheduled_cached_reqs.resumed_from_preemption == [False, True]


def test_sidecar_pth_install_and_cleanup_owned_payload(tmp_path, monkeypatch):
    from src.serve_trace.sidecar_lifecycle import (
        SIDECAR_PTH_NAME,
        cleanup_sidecar_pth,
        install_sidecar_pth,
        sidecar_pth_payload,
    )
    monkeypatch.setenv("CUDA_SIDECAR_PATH", str(tmp_path / "scheduler.jsonl"))
    project_root = tmp_path / "project"
    project_root.mkdir()
    site_dir = tmp_path / "site"

    pth_path = install_sidecar_pth(project_root=project_root, site_dir=site_dir)
    assert pth_path == site_dir / SIDECAR_PTH_NAME
    assert pth_path.read_text() == sidecar_pth_payload(project_root)

    assert cleanup_sidecar_pth(pth_path, project_root=project_root) is True
    assert not pth_path.exists()


def test_sidecar_pth_cleanup_ignores_unowned_payload(tmp_path):
    from src.serve_trace.sidecar_lifecycle import cleanup_sidecar_pth
    project_root = tmp_path / "project"
    project_root.mkdir()
    pth_path = tmp_path / "midlayer_sidecar.pth"
    pth_path.write_text("/unrelated/path\n")

    assert cleanup_sidecar_pth(pth_path, project_root=project_root) is False
    assert pth_path.read_text() == "/unrelated/path\n"


def test_sidecar_pth_install_noops_without_sidecar_env(tmp_path, monkeypatch):
    from src.serve_trace.sidecar_lifecycle import install_sidecar_pth
    monkeypatch.delenv("CUDA_SIDECAR_PATH", raising=False)
    assert install_sidecar_pth(project_root=tmp_path, site_dir=tmp_path / "site") is None


def test_serving_installer_returns_owned_pth_in_user_site(tmp_path, monkeypatch):
    import scripts.serve_trace.trace_serving as serving
    from src.serve_trace.sidecar_lifecycle import cleanup_sidecar_pth

    monkeypatch.setenv("CUDA_SIDECAR_PATH", str(tmp_path / "scheduler.jsonl"))
    monkeypatch.setattr(
        "src.serve_trace.sidecar_lifecycle.site.getusersitepackages",
        lambda: str(tmp_path / "site"),
    )

    pth_path = serving._install_scheduler_sidecar()
    assert pth_path == tmp_path / "site" / "midlayer_sidecar.pth"
    assert pth_path.exists()
    assert serving._SIDECAR_PTH_PATH == pth_path
    assert cleanup_sidecar_pth(pth_path) is True


def test_disagg_sidecar_setup_records_cleanup_path(tmp_path, monkeypatch):
    import scripts.serve_trace._disagg_instance as disagg
    trace_base = str(tmp_path / "trace")
    pth_path = tmp_path / "site" / "midlayer_sidecar.pth"

    monkeypatch.setattr(
        "scripts.serve_trace.trace_serving._install_scheduler_sidecar",
        lambda: pth_path,
    )

    result = disagg._setup_scheduler_sidecar(trace_base)
    assert result == f"{trace_base}_scheduler.jsonl"
    assert disagg._SIDECAR_PTH_PATH == pth_path
