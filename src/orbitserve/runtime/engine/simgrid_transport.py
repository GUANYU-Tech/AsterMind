"""Causal subprocess adapter for optional native SimGrid bandwidth sharing."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from ..setup.radio_capacity import capacity_pools, hop_pools, directional_profile

BYTE_SCALE = 1_000_000


class SimGridUnavailable(RuntimeError):
    """The explicitly selected native backend could not be started."""


def require_simgrid():
    if importlib.util.find_spec("simgrid") is None:
        raise SimGridUnavailable(
            "SimGrid mode requires the optional dependency. Install OrbitServe[simgrid] "
            "in an environment supported by SimGrid; queue mode remains independent."
        )


class SimGridTransport:
    """One native engine with independently shared forward/reverse resources."""

    worker_module = 'orbitserve.runtime.engine.simgrid_worker'
    worker_name = 'SimGrid'
    error_type = SimGridUnavailable
    check_available = staticmethod(require_simgrid)

    def __init__(self, nodes, edges, profiles, *, start_s=0.0, shared_pools=None, link_pools=None):
        self.check_available()
        if isinstance(start_s, bool) or not isinstance(start_s, (int, float)) or not math.isfinite(start_s) or start_s < 0:
            raise ValueError('native transport start_s must be finite and nonnegative')
        if not isinstance(nodes, dict):
            from ..setup.network_spec import NetworkNodeSpec
            nodes = {n: NetworkNodeSpec(n, 'compute') for n in nodes}
        edges = tuple(edges)
        # Validate unpublished capacities before launching a worker subprocess.
        for edge in edges:
            hop_pools(nodes, edge['endpoint_a'], edge['endpoint_b'])
            hop_pools(nodes, edge['endpoint_b'], edge['endpoint_a'])
        self.now = 0.0
        self.nodes, self.edges, self.profiles = nodes, tuple(edges), profiles
        self.pool_overrides, self.link_pool_overrides = shared_pools, link_pools
        self.configured_capacities = {}
        self.active = {}
        self.resources = {}
        self.closed = False
        # Resolve/validate all input-derived values before owning a subprocess.
        links = []
        for i, edge in enumerate(edges):
            for direction, (source, target) in enumerate(
                ((edge['endpoint_a'], edge['endpoint_b']), (edge['endpoint_b'], edge['endpoint_a']))
            ):
                key = f'link-{i}-{direction}'
                self.resources[edge['link_id'], source, target] = key
                links.append({
                    'key': key, 'source': source, 'target': target,
                    'bandwidth_bps': directional_profile(nodes, profiles, edge, source).bandwidth_bps,
                    'resource_gate': edge.get('kind') == 'device_gate',
                    'pools': list(link_pools.get((edge['link_id'], source, target), ())
                                  if link_pools is not None else hop_pools(nodes, source, target)),
                })
        pools = capacity_pools(nodes) if shared_pools is None else dict(shared_pools)
        self.responses = queue.Queue()
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        # Official bindings may ship libsimgrid next to their extension module.
        # Configure only this child loader, without importing native global state
        # into the parent or requiring workstation-specific library paths.
        location = str(Path(importlib.util.find_spec("simgrid").origin).parent) if self.worker_module.endswith('simgrid_worker') else ''
        loader_variable = (
            "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        )
        environment[loader_variable] = os.pathsep.join(
            value for value in (location, environment.get(loader_variable, "")) if value
        )
        self.errors = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-B", "-m", self.worker_module],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors,
                text=True, encoding="utf-8", bufsize=1, env=environment,
            )
        except BaseException:
            self.errors.close()
            raise

        def read_responses():
            try:
                for line in self.process.stdout:
                    self.responses.put(line)
            except (OSError, UnicodeError) as error:
                self.reader_error = str(error)
            finally:
                self.responses.put(None)

        self.reader_error = None
        self.reader = threading.Thread(target=read_responses, daemon=True)
        try:
            self.reader.start()
            ready = self._request({"nodes": sorted(nodes), "links": links, 'pools': pools, 'start_s': start_s})
            if not ready.get("ready"):
                raise self.error_type(f"{self.worker_name} worker did not initialize")
            self.validate_ready(ready)
            self.provenance = {
                "backend": "simgrid",
                "simgrid_version": ready["version"],
                "simgrid_network_model": "CM02",
                "crosstraffic": False,
                "sharing_scope": "each_directed_physical_hop",
                "propagation_accounting": "orbitserve_after_serialization",
                "completion_source": "native_simgrid_activity",
                "payload_quantization_bytes": 1 / BYTE_SCALE,
                "capacity_units": "public_bits_per_second_native_bytes_per_second",
                'aggregate_sharing': 'native_shared_satellite_rf_resources',
            }
        except BaseException:
            self.close(force=True)
            raise

    def _request(self, payload):
        if self.closed and payload.get('op') != 'close':
            raise self.error_type(f'{self.worker_name} transport is closed')
        encoded = json.dumps(payload, allow_nan=False) + '\n'
        try:
            self.process.stdin.write(encoded)
            self.process.stdin.flush()
            line = self.responses.get(timeout=60)
            if line is None:
                self.errors.seek(0)
                diagnostic = self.errors.read()
                if self.reader_error:
                    diagnostic += f'\nWorker output could not be decoded: {self.reader_error}'
                if len(diagnostic) > 12000:
                    diagnostic = diagnostic[:6000] + '\n... stderr truncated ...\n' + diagnostic[-6000:]
                returncode = self.process.poll()
                self.close(force=True)
                raise self.error_type(
                    f"{self.worker_name} worker exited (returncode {returncode}):\n{diagnostic}"
                )
            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError('worker response must be an object')
            if "error" in result:
                raise self.error_type(result["error"])
            time_s = result['time_s']
            if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < self.now:
                raise ValueError('worker time must be finite and monotonic')
            self.now = float(time_s)
            return result
        except (OSError, queue.Empty, ValueError, TypeError, KeyError) as error:
            # A timeout/malformed reply loses request-response alignment. Reusing
            # that worker can assign a late response to an unrelated operation.
            self.close(force=True)
            raise self.error_type(
                f"{self.worker_name} worker protocol failed: {error}"
            ) from error

    def validate_ready(self, ready):
        """Backend-specific handshake checks, inside startup cleanup ownership."""

    def resource(self, leg):
        return self.resources[
            leg["link_id"], leg["source_node_id"], leg["target_node_id"]
        ]

    def configure(self, capacities):
        if any(not math.isfinite(v) or v < 0 for v in capacities.values()):
            raise ValueError("SimGrid capacities must be finite nonnegative bits/s")
        if capacities:
            self._request({"op": "configure", "capacities": capacities})
            self.configured_capacities.update(capacities)

    def register_edge(self, edge):
        """SimGrid seals platforms: rebuild at this causal boundary, retaining work."""
        remaining = self._request({'op': 'remaining'})['remaining']
        replacement = type(self)(self.nodes, (*self.edges, edge), self.profiles, start_s=self.now,
                                 shared_pools=self.pool_overrides, link_pools=self.link_pool_overrides)
        try:
            replacement.configure(self.configured_capacities)
            for identity, row in self.active.items():
                replacement.start(identity, row['leg'], remaining[identity])
            # Zero capacity is represented by suspended activities, so apply it
            # again after restoring activities at exactly the same native time.
            replacement.configure(self.configured_capacities)
        except BaseException:
            replacement.close(force=True)
            raise
        self.close()
        self.__dict__.update(replacement.__dict__)

    def start(self, identity, leg, payload_bytes):
        if identity in self.active:
            raise ValueError('native transport requires a unique active transfer id')
        if isinstance(payload_bytes, bool) or not math.isfinite(payload_bytes) or payload_bytes <= 0:
            raise ValueError("SimGrid payload_bytes must be finite and positive")
        if payload_bytes >= 2**64 / BYTE_SCALE:
            raise ValueError("SimGrid payload exceeds its native scaled size limit")
        amount = max(1, round(payload_bytes * BYTE_SCALE))
        if amount >= 2**64:
            raise ValueError("SimGrid payload exceeds its native scaled size limit")
        key = self.resource(leg)
        self._request(
            {
                "op": "start",
                "id": identity,
                "key": key,
                "source": leg["source_node_id"],
                "target": leg["target_node_id"],
                "bytes": amount,
            }
        )
        self.active[identity] = {"key": key, "leg": leg}

    def advance_until(self, time_s):
        if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < self.now:
            raise ValueError('native transport time must be finite and monotonic')
        response = self._request({"op": "advance", "until_s": time_s})
        for identity in response["completed"]:
            self.active.pop(identity, None)
        return response

    def cancel(self, identities):
        identities = [i for i in identities if i in self.active]
        if identities:
            self._request({"op": "cancel", "ids": identities})
            for identity in identities:
                self.active.pop(identity, None)

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        try:
            if not force and self.process.poll() is None:
                self._request({"op": "close"})
                self.process.wait(timeout=10)
        except (RuntimeError, OSError, subprocess.TimeoutExpired):
            pass
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=10)
            self.process.stdin.close()
            if self.reader.ident is not None:
                self.reader.join(timeout=2)
            self.process.stdout.close()
            self.errors.close()
