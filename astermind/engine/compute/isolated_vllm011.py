"""Run the bundled 0.11 control plane in a separate, pinned Python environment."""
from dataclasses import replace
from multiprocessing import Pipe
from pathlib import Path
import subprocess
import tempfile
import weakref

from src.serving.backend import vllm_scheduler_process as native
from src.serving.backend.vllm import VLLMBackend, _validate_native_scheduler_scope


class IsolatedScheduler(native.VLLMSchedulerProcess):
    def __init__(self, python, config, scheduler):
        model = Path(scheduler.model).resolve()
        temporary = None
        if model.is_file():
            if model.suffix != ".json":
                raise ValueError("Expected a JSON model configuration")
            temporary = tempfile.TemporaryDirectory(prefix="vllm011-model-")
            directory = Path(temporary.name)
            (directory / "config.json").write_bytes(model.read_bytes())
            model = directory
        self.scheduler_config = replace(scheduler, model=str(model))
        if not (model / "config.json").is_file():
            raise ValueError("Missing scheduler model configuration")
        parent, child = Pipe()
        # Start with the worker SDK's own site-packages. The server and its other
        # Cells can continue using a different vLLM or Torch version concurrently.
        entry = ("import sys; sys.path[:0] = sys.argv[2:]; "
                 "from multiprocessing.connection import Connection; "
                 "from src.serving.backend.vllm_scheduler_process import _scheduler_worker; "
                 "pipe = Connection(int(sys.argv[1])); _scheduler_worker(pipe, *pipe.recv())")
        try:
            process = subprocess.Popen(
                [python, "-I", "-B", "-c", entry, str(child.fileno()),
                 str(Path(native.__file__).resolve().parents[3]), str(Path(__file__).resolve().parents[3])],
                pass_fds=(child.fileno(),))
        except BaseException:
            parent.close()
            if temporary is not None:
                temporary.cleanup()
            raise
        finally:
            child.close()
        self._connection = parent
        self._process = native._SchedulerPopenWorker(process)
        self._temporary_model_directory = temporary
        self._closed = False
        self._finalizer = weakref.finalize(self, native._finalize_worker, parent, self._process, temporary)
        try:
            parent.send((config, self.scheduler_config))
            self.runtime_info = self._parse_ready(self._receive(native._STARTUP_TIMEOUT_S))
        except BaseException:
            self._force_close()
            raise


def create_backend(python, config, scheduler):
    _validate_native_scheduler_scope(config, scheduler)
    worker = IsolatedScheduler(python, config, scheduler)
    try:
        return VLLMBackend(config, worker)
    except BaseException:
        worker.close()
        raise
