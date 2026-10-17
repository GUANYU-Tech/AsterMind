"""
Auto-installed scheduler sidecar patch for vLLM V1 EngineCore child processes.

Installs a post-import hook that patches Scheduler.schedule() when the
scheduler module is first imported. Works in both fork and spawn modes
since the .pth file ensures this module runs at Python startup.
"""
import importlib
import importlib.abc
import importlib.machinery
import os
import sys


_TARGET_MODULE = "vllm.v1.core.sched.scheduler"
_P2P_CONNECTOR_MODULE = "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector"


class _SidecarFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that patches Scheduler and P2pNcclConnector after import."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (_TARGET_MODULE, _P2P_CONNECTOR_MODULE):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _SidecarLoader(fullname, spec.loader, self)
        return spec


class _SidecarLoader(importlib.abc.Loader):
    """Loader wrapper that patches the target module after real execution."""

    def __init__(self, fullname, wrapped_loader, finder):
        self.fullname = fullname
        self.wrapped_loader = wrapped_loader
        self.finder = finder

    def create_module(self, spec):
        if hasattr(self.wrapped_loader, "create_module"):
            return self.wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.wrapped_loader.exec_module(module)
        if self.fullname == _TARGET_MODULE:
            _apply_patch(module.Scheduler)
        elif self.fullname == _P2P_CONNECTOR_MODULE:
            _apply_p2p_patch(module.P2pNcclConnector)
        _remove_finder_if_done(self.finder)


def _remove_finder_if_done(finder):
    if _TARGET_MODULE in sys.modules and _P2P_CONNECTOR_MODULE in sys.modules:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass


def _apply_patch(scheduler_cls):
    """Apply the sidecar patch to the Scheduler class."""
    if getattr(scheduler_cls.schedule, "_sidecar_patched", False):
        return

    import json
    import time as _time

    _orig_schedule = scheduler_cls.schedule
    _iter_counter = [0]

    def _patched_schedule(self):
        output = _orig_schedule(self)
        path = os.environ.get("CUDA_SIDECAR_PATH")
        if path:
            record = {
                "iter_id": _iter_counter[0],
                "ts_us": int(_time.time() * 1e6),
                "scheduled": output.num_scheduled_tokens,
                "total_tokens": output.total_num_scheduled_tokens,
                "finished": sorted(output.finished_req_ids),
            }
            try:
                with open(path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError:
                pass
            _iter_counter[0] += 1
        return output

    _patched_schedule._sidecar_patched = True
    scheduler_cls.schedule = _patched_schedule


def _apply_p2p_patch(connector_cls):
    """Patch P2pNcclConnector.build_connector_meta to skip producer cached
    requests that have already completed their KV send.

    Filters both req_ids AND companion arrays (num_computed_tokens,
    new_block_ids, resumed_from_preemption) to maintain index alignment.
    """
    if getattr(connector_cls, "_p2p_patched", False):
        return

    _orig_build_meta = connector_cls.build_connector_meta

    def _patched_build_meta(self, scheduler_output):
        if hasattr(self, 'is_producer') and self.is_producer:
            cached_reqs = scheduler_output.scheduled_cached_reqs
            if cached_reqs and hasattr(cached_reqs, 'req_ids') and cached_reqs.req_ids:
                keep_indices = [i for i, rid in enumerate(cached_reqs.req_ids)
                               if rid in self.chunked_prefill]
                if len(keep_indices) < len(cached_reqs.req_ids):
                    # Save originals
                    orig_ids = cached_reqs.req_ids
                    orig_computed = getattr(cached_reqs, 'num_computed_tokens', None)
                    orig_blocks = getattr(cached_reqs, 'new_block_ids', None)
                    orig_resumed = getattr(cached_reqs, 'resumed_from_preemption', None)

                    # Filter all arrays to keep only chunked-prefill requests
                    cached_reqs.req_ids = [orig_ids[i] for i in keep_indices]
                    if orig_computed is not None:
                        cached_reqs.num_computed_tokens = [orig_computed[i] for i in keep_indices]
                    if orig_blocks is not None:
                        cached_reqs.new_block_ids = [orig_blocks[i] for i in keep_indices]
                    if orig_resumed is not None:
                        cached_reqs.resumed_from_preemption = [orig_resumed[i] for i in keep_indices]

                    try:
                        return _orig_build_meta(self, scheduler_output)
                    finally:
                        # Restore originals
                        cached_reqs.req_ids = orig_ids
                        if orig_computed is not None:
                            cached_reqs.num_computed_tokens = orig_computed
                        if orig_blocks is not None:
                            cached_reqs.new_block_ids = orig_blocks
                        if orig_resumed is not None:
                            cached_reqs.resumed_from_preemption = orig_resumed
        return _orig_build_meta(self, scheduler_output)

    connector_cls.build_connector_meta = _patched_build_meta
    connector_cls._p2p_patched = True


def install_sidecar_patch():
    """Install the scheduler sidecar and P2P connector patches."""
    if not os.environ.get("CUDA_SIDECAR_PATH"):
        return

    target = "vllm.v1.core.sched.scheduler"
    p2p_target = "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector"

    # Patch already-imported modules directly
    if target in sys.modules:
        _apply_patch(sys.modules[target].Scheduler)
    if p2p_target in sys.modules:
        _apply_p2p_patch(sys.modules[p2p_target].P2pNcclConnector)

    # Install finder for not-yet-imported modules
    needs_finder = (target not in sys.modules) or (p2p_target not in sys.modules)
    if needs_finder:
        if not any(isinstance(f, _SidecarFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _SidecarFinder())


# Auto-install when this module is imported (triggered by .pth file)
install_sidecar_patch()
