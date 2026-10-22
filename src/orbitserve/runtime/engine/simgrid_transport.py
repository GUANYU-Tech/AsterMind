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
    check_available = staticmethod(require_simgrid)

    def __init__(self, nodes, edges, profiles):
        self.check_available()
        if not isinstance(nodes, dict):
            from ..setup.network_spec import NetworkNodeSpec
            nodes = {n: NetworkNodeSpec(n, 'compute') for n in nodes}
        # Validate unpublished capacities before launching a worker subprocess.
        for edge in edges:
            hop_pools(nodes, edge['endpoint_a'], edge['endpoint_b'])
            hop_pools(nodes, edge['endpoint_b'], edge['endpoint_a'])
        self.now = 0.0
        self.active = {}
        self.resources = {}
        self.closed = False
        self.responses = queue.Queue()
        self.errors = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")
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
        self.process = subprocess.Popen(
            [sys.executable, "-B", "-m", self.worker_module],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )

        def read_responses():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        self.reader = threading.Thread(target=read_responses, daemon=True)
        self.reader.start()
        links = []
        for i, edge in enumerate(edges):
            for direction, (source, target) in enumerate(
                (
                    (edge["endpoint_a"], edge["endpoint_b"]),
                    (edge["endpoint_b"], edge["endpoint_a"]),
                )
            ):
                key = f"link-{i}-{direction}"
                self.resources[edge["link_id"], source, target] = key
                links.append(
                    {
                        "key": key,
                        "source": source,
                        "target": target,
                        "bandwidth_bps": profiles[
                            edge.get("profile_id", edge["kind"])
                        ].bandwidth_bps,
                        'pools': list(hop_pools(nodes, source, target)),
                    }
                )
                links[-1]['bandwidth_bps'] = directional_profile(nodes, profiles, edge, source).bandwidth_bps
        try:
            ready = self._request({"nodes": sorted(nodes), "links": links, 'pools': capacity_pools(nodes)})
            if not ready.get("ready"):
                raise SimGridUnavailable("SimGrid worker did not initialize")
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
        try:
            self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
            self.process.stdin.flush()
            line = self.responses.get(timeout=60)
            if line is None:
                self.errors.seek(0)
                raise SimGridUnavailable(
                    "SimGrid worker exited: " + self.errors.read()[-2500:]
                )
            result = json.loads(line)
            if "error" in result:
                raise SimGridUnavailable(result["error"])
            self.now = float(result["time_s"])
            return result
        except (BrokenPipeError, queue.Empty, json.JSONDecodeError) as error:
            raise SimGridUnavailable(
                f"SimGrid worker protocol failed: {error}"
            ) from error

    def resource(self, leg):
        return self.resources[
            leg["link_id"], leg["source_node_id"], leg["target_node_id"]
        ]

    def configure(self, capacities):
        if any(not math.isfinite(v) or v < 0 for v in capacities.values()):
            raise ValueError("SimGrid capacities must be finite nonnegative bits/s")
        if capacities:
            self._request({"op": "configure", "capacities": capacities})

    def start(self, identity, leg, payload_bytes):
        if not math.isfinite(payload_bytes) or payload_bytes <= 0:
            raise ValueError("SimGrid payload_bytes must be finite and positive")
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
            self.reader.join(timeout=2)
            self.process.stdout.close()
            self.errors.close()
