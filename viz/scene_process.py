"""One Linux worker and one private control channel per full-scene run."""
import json
import os
from multiprocessing.connection import Connection
from multiprocessing.reduction import recv_handle, send_handle
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading


class SceneProcess:
    def __init__(self, manifest, job):
        self.job = job
        self.output = Path(job['output'])
        self.control_lock = threading.Lock()
        self.lifecycle = threading.Lock()
        self.finished = threading.Event()
        self.cancelled = False
        parent, child = socket.socketpair()
        with parent, child, (self.output / 'execution.log').open('ab') as log:
            fd = child.fileno()
            self.process = subprocess.Popen(
                [sys.executable, '-B', str(Path(__file__).resolve()), str(manifest),
                 str(self.output), str(fd)],
                cwd=self.output, pass_fds=(fd,), start_new_session=True, stdout=log, stderr=subprocess.STDOUT,
            )
            self.control = Connection(parent.detach())

    @property
    def payload(self):
        # Registry consumers only need to know whether the run is active.
        return {'state': 'stopping' if self.cancelled else 'preparing'}

    def monitor(self):
        threading.Thread(target=self._wait, name=self.job['id'], daemon=True).start()

    def _wait(self):
        from live import LOCK, SESSIONS
        from tasks import write_status
        code = self.process.wait()
        try:
            with self.lifecycle:
                status = json.loads((self.output / 'status.json').read_text(encoding='utf-8'))
                if status['state'] == 'running':
                    if self.cancelled:
                        write_status(status, 'cancelled')
                    else:
                        write_status(status, 'failed', error=f'仿真进程意外退出 ({code})，请查看执行日志')
                elif status.get('replay_state') == 'preparing':
                    write_status(status, 'completed', replay_state='failed', replay_error=f'回放进程退出 ({code})')
        finally:
            with LOCK:
                SESSIONS.pop(self.job['id'], None)
            self.finished.set()
            with self.control_lock:
                self.control.close()

    def _call(self, method, *args):
        # Give each HTTP call its own reply. Waiting for a simulation command
        # must leave state polling and pause requests responsive.
        local, remote = socket.socketpair()
        try:
            with local, remote:
                with self.control_lock:
                    send_handle(self.control, remote.fileno(), self.process.pid)
                remote.close()
                with Connection(local.detach()) as call:
                    call.send_bytes(json.dumps([method, args]).encode())
                    reply = json.loads(call.recv_bytes())
        except (EOFError, OSError) as error:
            if not self.finished.wait(timeout=5):
                raise RuntimeError('暂时无法读取仿真进程，请刷新任务状态') from error
            if method == 'state':
                from live import get
                return get(self.job['id']).state(*args)
            raise ValueError('该任务已结束')
        if 'error' in reply:
            raise ValueError(reply['error'])
        return reply['result']

    def state(self, after=None):
        return self._call('state', after)

    def advance(self, until_s=None, mode=None):
        return self._call('advance', until_s, mode)

    def command(self, command):
        return self._call('command', command)

    def pause(self):
        return self._call('pause')

    def stop(self):
        with self.lifecycle:
            if self.process.poll() is not None:
                state = json.loads((self.output / 'status.json').read_text(encoding='utf-8'))['state']
                return {'id': self.job['id'], 'state': state}
            self.cancelled = True
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return {'id': self.job['id'], 'state': 'stopping'}


def answer(fd, methods):
    with Connection(fd) as call:
        try:
            method, args = json.loads(call.recv_bytes())
            try:
                reply = {'result': methods[method](*args)}
            except Exception as error:
                reply = {'error': str(error)}
            call.send_bytes(json.dumps(reply, ensure_ascii=False).encode())
        except (EOFError, OSError):
            pass  # The requester disconnected.


def serve_control(control, session):
    methods = {name: getattr(session, name) for name in ('state', 'advance', 'command', 'pause')}
    try:
        while True:
            fd = recv_handle(control)
            threading.Thread(target=answer, args=(fd, methods), daemon=True).start()
    except (EOFError, OSError, RuntimeError):
        # The application exited. Its worker group owns only this task.
        os.killpg(os.getpid(), signal.SIGTERM)


def main(manifest, output, fd):
    os.set_inheritable(fd, False)
    from astermind.builder.config import SimulationConfig
    from native_scene_live import DeviceLiveSession, run_session
    job = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    config = SimulationConfig.model_validate_json(manifest.read_text(encoding='utf-8'))
    session = DeviceLiveSession(job['id'], output, config, job['automatic'])
    control = Connection(fd)
    threading.Thread(target=serve_control, args=(control, session), daemon=True).start()
    run_session(session, job)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
