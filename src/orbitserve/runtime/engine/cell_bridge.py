"""Host the Merge_ML_Version Cell contracts on the dynamic Runtime network.

No native model is imported here. A Cell factory supplies its scheduler and
kernel executor; all callbacks, collectives and KV copies use the host clock.
"""

from __future__ import annotations

import math
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from types import SimpleNamespace

from ..communication import (
    CommunicationCompleted,
    CommunicationFailed,
    PhysicalRoute,
    RouteHop,
)
from ..state.network import NetworkUnavailable
from ..timeline import GlobalTimeline
from .astermind_bridge import AsterMindRuntime
from .cell_events import CellRequest


@dataclass(frozen=True)
class CellTransfer:
    id: str
    source: str
    target: str
    size_bytes: float
    ready_time_ns: float
    origin: str = "midlayer"
    operation: str = "transfer"
    correlation_id: str | None = None

    def model_dump(self):
        return dict(id=self.id, source=self.source, target=self.target,
                    size_bytes=self.size_bytes, ready_time_ns=self.ready_time_ns,
                    origin=self.origin, operation=self.operation,
                    correlation_id=self.correlation_id)


@dataclass(frozen=True)
class CellTransferEvent:
    kind: str
    id: str
    source: str
    target: str
    size_bytes: float
    origin: str
    operation: str
    correlation_id: str | None
    path: tuple[str, ...]
    completion_time_ns: float | None = None
    failure_time_ns: float | None = None
    reason: str | None = None

    def model_dump(self):
        # Contract fields are immutable scalar values and a tuple of node IDs.
        # Avoid recursively deepcopying every field for each completion listener.
        result = dict(kind=self.kind, id=self.id, source=self.source, target=self.target,
                      size_bytes=self.size_bytes, origin=self.origin,
                      operation=self.operation, correlation_id=self.correlation_id,
                      path=self.path, completion_time_ns=self.completion_time_ns,
                      failure_time_ns=self.failure_time_ns, reason=self.reason)
        if self.kind == "completed":
            result.pop("failure_time_ns")
            result.pop("reason")
        else:
            result.pop("completion_time_ns")
            result["remaining_bytes"] = self.size_bytes
        return result


class RuntimeCellNetwork:
    """Callback-driven transfers on the host's existing directed resources.

    ``next_event_time_ns/advance_to`` intentionally are not implemented: the
    host advances physical transfers and publishes terminal events. Use
    bind_batch_executor() for the colleague's NetworkBatchExecutor.
    """

    def __init__(self, host, *, endpoints=None, cell_members=None):
        self.host = host
        self.endpoints = dict(endpoints or {})
        if any(node not in host.net.nodes for node in self.endpoints.values()):
            raise ValueError("Cell endpoint maps to an unknown physical node")
        members = dict(cell_members or {})
        self.topology = SimpleNamespace(
            devices={
                key: SimpleNamespace(kind="gpu", satellite=node)
                for key, node in self.endpoints.items()
            },
            satellite_to_cell=members,
        )
        self._listeners = []
        self.boundary_listeners = []
        self._jobs = {}
        self.events = []
        self.closed = False

    @property
    def now_ns(self):
        return self.host.now * 1e9

    def resolve(self, endpoint):
        node = self.endpoints.get(str(endpoint), str(endpoint))
        if node not in self.host.net.nodes:
            raise ValueError(f"unknown physical endpoint: {endpoint}")
        return node

    def resolve_pair(self, source, target):
        return self.resolve(source), self.resolve(target)

    def bind_endpoint(self, endpoint, node, cell_id):
        """Bind a native GPU ID to its declared host physical member."""
        if self.topology.satellite_to_cell.get(node) != cell_id:
            raise ValueError("native endpoint does not belong to this physical Cell")
        if endpoint in self.host.net.nodes and endpoint != node:
            raise ValueError("native endpoint shadows another physical node")
        if endpoint in self.endpoints and self.endpoints[endpoint] != node:
            raise ValueError(
                "native endpoint already belongs to another physical member"
            )
        self.endpoints[endpoint] = node
        self.topology.devices[endpoint] = SimpleNamespace(kind="gpu", satellite=node)

    def subscribe(self, callback):
        self._listeners.append(callback)

    def candidate_path(self, source, target):
        """Validate structure only; do not assume a future contact is active."""
        source, target = self.resolve_pair(source, target)
        adjacency = {}
        for edge in self.host.net.edges:
            a, b = edge["endpoint_a"], edge["endpoint_b"]
            allowed = getattr(self.host.net, "allows_hop", lambda source, target: True)
            if allowed(a, b):
                adjacency.setdefault(a, []).append(b)
            if allowed(b, a):
                adjacency.setdefault(b, []).append(a)
        queue = deque([(source, (source,))])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in sorted(adjacency.get(node, ())):
                if neighbor not in visited and self.host.net.route_node_allowed(
                    neighbor, source, target
                ):
                    visited.add(neighbor)
                    queue.append((neighbor, (*path, neighbor)))
        raise ValueError(f"no candidate path: {source}->{target}")

    def submit(self, request):
        if self.closed:
            raise RuntimeError("Cell network is closed")
        if (
            not isinstance(request.id, str)
            or not request.id
            or request.id in self._jobs
        ):
            raise ValueError("Cell transfer requires a unique nonempty ID")
        if (
            isinstance(request.size_bytes, bool)
            or not math.isfinite(request.size_bytes)
            or request.size_bytes < 0
        ):
            raise ValueError("Cell transfer size must be finite and nonnegative")
        if (
            not math.isfinite(request.ready_time_ns)
            or request.ready_time_ns < self.now_ns - 1e-6
        ):
            raise ValueError("Cell transfer cannot be submitted in the past")
        source, target = self.resolve_pair(request.source, request.target)
        identity = f"__cell_transfer__/{request.id}"
        if identity in self.host.jobs or identity in self.host.service_jobs:
            raise ValueError(f"reserved Runtime request ID: {identity}")
        job = {
            "request": {"request_id": identity},
            "done": False,
            "path": [],
            "ledger": [],
            "external_service": True,
        }
        self._jobs[request.id] = job
        self.host.service_jobs[identity] = job

        def terminal(reason=None):
            job["done"] = True
            path = [source]
            path.extend(
                leg["target_node_id"]
                for leg in job["path"]
                if leg.get("arrival_s") is not None
                and leg["arrival_s"] <= self.host.now
            )
            event = CellTransferEvent(
                "failed" if reason else "completed",
                request.id,
                request.source,
                request.target,
                request.size_bytes,
                request.origin,
                request.operation,
                request.correlation_id,
                tuple(self.map_path(path)),
                completion_time_ns=None if reason else self.now_ns,
                failure_time_ns=self.now_ns if reason else None,
                reason=reason,
            )
            self.events.append(event.model_dump())
            self.host.log.append(
                {
                    **event.model_dump(),
                    "time_s": self.host.now,
                    "terminal_kind": event.kind,
                    "kind": "cell_transfer_terminal",
                }
            )
            for listener in tuple(self._listeners):
                listener(event)

        job["external_failure_callback"] = terminal

        def start():
            if job["done"]:
                return
            try:
                self.host.send(
                    job,
                    source,
                    target,
                    float(request.size_bytes),
                    None,
                    request.operation,
                    "external_flow_done",
                    context={"callback": terminal, "cell_transfer_id": request.id},
                )
            except NetworkUnavailable as error:
                if (
                    self.host.net.spec.failure_policy != "retry_next_sample"
                    or not self.host.net.spec.max_transfer_retries
                ):
                    self.host.fail(job, str(error))
                    return
                try:
                    self.candidate_path(request.source, request.target)
                except ValueError:
                    self.host.fail(job, str(error))
                    return
                self.host.waiting_retries[identity] = {
                    "request_id": identity,
                    "leg": {"transfer_id": identity},
                    "resume_node": source,
                    "origin": source,
                    "target": target,
                    "size": float(request.size_bytes),
                    "kinds": None,
                    "stage": request.operation,
                    "completion": "external_flow_done",
                    "attempt": 1,
                    "context": {
                        "callback": terminal,
                        "cell_transfer_id": request.id,
                        "attempt": 1,
                    },
                }
                self.host.log.append(
                    {
                        "time_s": self.host.now,
                        "kind": "route_wait",
                        "request_id": identity,
                        "source_node_id": source,
                        "target_node_id": target,
                    }
                )
            except ValueError as error:
                self.host.fail(job, str(error))

        self.host.push(
            max(self.host.now, request.ready_time_ns / 1e9),
            33,
            "external_callback",
            payload=start,
        )
        return request.id

    def map_path(self, path):
        return path

    def close(self):
        self.closed = True
        for job in tuple(self._jobs.values()):
            if not job["done"]:
                self.host.fail(job, "cell_network_closed")

    def cancel(self, identifiers, reason="cell_transfer_cancelled"):
        for identifier in identifiers:
            job = self._jobs.get(identifier)
            if job is not None and not job["done"]:
                self.host.fail(job, reason)


class CellTransferExecutor:
    """The submit_transport seam used by per-rank KV migration."""

    def __init__(self, loop, network):
        self.loop, self.network = loop, network
        self._callbacks = {}
        self.boundary_listeners = network.boundary_listeners
        self.transfer_events = []
        network.subscribe(self._deliver)

    def _deliver(self, event):
        pair = self._callbacks.pop(event.id, None)
        if pair is None:
            return
        self.transfer_events.append(event.model_dump())
        complete, failure = pair
        if event.kind == "failed":
            if failure is None:
                raise RuntimeError(f"Cell communication failed: {event.reason}")
            failure(event)
        else:
            complete()

    def submit_transport(
        self,
        identifier,
        source,
        target,
        size_bytes,
        operation,
        correlation_id,
        on_complete,
        on_failure=None,
    ):
        if identifier in self._callbacks:
            raise ValueError("duplicate active Cell transfer")
        self._callbacks[identifier] = on_complete, on_failure
        try:
            self.network.submit(
                CellTransfer(
                    identifier,
                    source,
                    target,
                    size_bytes,
                    self.network.now_ns,
                    operation=operation,
                    correlation_id=correlation_id,
                )
            )
        except BaseException:
            self._callbacks.pop(identifier, None)
            raise

    def notify_batch_boundary(self):
        for notify in tuple(self.boundary_listeners):
            notify()


def bind_batch_executor(executor_class, loop, network, placements):
    """Reuse the colleague's native graph executor with host-driven completion.

    Pass integration.network_batch_executor.NetworkBatchExecutor after loading
    its explicit external Mid-Layer dependency. Its graph lowering is retained;
    only its independent network-pump methods are replaced.
    """

    class HostedExecutor(executor_class):
        def __init__(self):
            super().__init__(loop, network, placements)
            self.boundary_listeners = network.boundary_listeners
            self._transport_failures = {}
            self.input_distributions = []
            network.subscribe(self._deliver)

        def _arm_network(self):
            # The Runtime event queue already drives every physical hop.
            return None

        def _deliver(self, event):
            callback = self._callbacks.pop(event.id, None)
            if callback is None:
                return
            self.transfer_events.append(event.model_dump())
            failure = self._transport_failures.pop(event.id, None)
            if event.kind == "failed":
                if failure is None:
                    raise RuntimeError(
                        f"native Cell communication failed: {event.reason}"
                    )
                failure(event)
            else:
                callback()

        def submit_transport(
            self,
            identifier,
            source,
            target,
            size_bytes,
            operation,
            correlation_id,
            on_complete,
            on_failure=None,
        ):
            if identifier in self._callbacks:
                raise ValueError("duplicate active Cell transfer")
            network.submit(
                CellTransfer(
                    identifier,
                    source,
                    target,
                    size_bytes,
                    network.now_ns,
                    operation=operation,
                    correlation_id=correlation_id,
                )
            )
            self._callbacks[identifier] = on_complete
            if on_failure is not None:
                self._transport_failures[identifier] = on_failure

        def start_batch(
            self, cell_id, step_id, components, execution, on_complete, **kwargs
        ):
            def finished(at_s):
                on_complete(at_s)
                for notify in tuple(self.boundary_listeners):
                    notify()

            def start():
                return super(HostedExecutor, self).start_batch(
                    cell_id, step_id, components, execution, finished, **kwargs
                )

            pool_name, pool = next(iter(execution.deployment.worker_pools.items()))
            participants = pool.physical_ranks_for_pipeline_stage(0)
            if len(participants) > 1 and not getattr(
                execution, "input_distribution_in_graph", True
            ):
                count = execution.last_step_token_count
                if type(count) is not int or count <= 0:
                    raise ValueError(
                        "native input distribution requires scheduled token count"
                    )
                record = {
                    "cell_id": cell_id,
                    "step_id": step_id,
                    "ready_s": loop.now_s,
                    "ready_ns": math.ceil(loop.now_ns),
                    "phases": [],
                }
                self.input_distributions.append(record)

                def distributed():
                    record["completed_s"] = loop.now_s
                    record["completed_ns"] = math.ceil(loop.now_ns)
                    start()

                self._phase(
                    cell_id,
                    pool_name,
                    "input_token_ids",
                    f"{cell_id}/{step_id}/input",
                    [(participants[0], rank, count * 8) for rank in participants[1:]],
                    distributed,
                    record,
                )
                return None
            return start()

    return HostedExecutor()


def native_cell_factory(
    config,
    *,
    cell_class,
    executor_class,
    placements,
    history_store=None,
    execution_wrapper=None,
):
    """Bind the colleague's unchanged Cell and execution classes explicitly.

    Use NetworkMidlayerCell, NetworkBatchExecutor and NativeStepExecution from
    Merge_ML_Version. Their external Mid-Layer/vLLM dependencies must already be
    loaded in the caller's isolated environment. No path discovery or import
    rewriting takes place here.
    """
    placements = tuple(placements)

    def factory(*, cell_id, loop, network, on_output, on_complete):
        for placement in placements:
            if placement.cell_id != cell_id:
                raise ValueError(
                    "native rank placement must use the selected Cell identity"
                )
            network.bind_endpoint(
                placement.gpu_device_id, placement.satellite_id, cell_id
            )
        executor = bind_batch_executor(executor_class, loop, network, placements)
        cell = cell_class.create(
            cell_id,
            loop,
            config,
            on_complete,
            on_output=on_output,
            history_store=history_store,
            batch_executor=executor,
        )
        try:
            if execution_wrapper is not None:
                cell.execution = execution_wrapper(cell.execution)
            return cell
        except BaseException:
            cell.close()
            raise

    return factory


class HostedCommunicationBackend:
    """CommunicationExecutor backend that shares the host's actual transport."""

    def __init__(self, name, network):
        self.name, self.network = name, network
        self.executor = CellTransferExecutor(GlobalTimeline(host=network.host), network)

    def submit(self, transfer, route, on_result):
        started = self.network.host.now

        def completed():
            actual_route = None
            if route is not None:
                legs = self.network._jobs[transfer.transfer_id]["path"]
                hops = []
                for leg in legs:
                    if (
                        leg.get("arrival_s") is None
                        or leg["arrival_s"] > self.network.host.now
                    ):
                        continue
                    segments = leg["segments"]
                    capacity = (
                        segments[0].get(
                            "link_capacity_bps", segments[0]["bandwidth_bps"]
                        )
                        if segments
                        else 0.0
                    )
                    hops.append(
                        RouteHop(
                            leg["link_id"],
                            leg["source_node_id"],
                            leg["target_node_id"],
                            capacity,
                            leg["propagation_delay_s"],
                            tuple(leg["shared_pools"]),
                            {"segments": tuple(dict(row) for row in segments)},
                        )
                    )
                actual_route = PhysicalRoute(
                    transfer.source_satellite_id,
                    transfer.target_satellite_id,
                    tuple(hops),
                    self.network.host.net.revision,
                )
            on_result(
                CommunicationCompleted(
                    transfer.transfer_id,
                    started,
                    self.network.host.now,
                    self.name,
                    actual_route,
                )
            )

        self.executor.submit_transport(
            transfer.transfer_id,
            transfer.source_endpoint or transfer.source_satellite_id,
            transfer.target_endpoint or transfer.target_satellite_id,
            transfer.size_bytes,
            transfer.operation,
            None,
            completed,
            lambda event: on_result(
                CommunicationFailed(
                    transfer.transfer_id, self.network.host.now, event.reason
                )
            ),
        )


@dataclass(frozen=True)
class CellDeployment:
    """Factory(**cell_id, loop, network, on_output, on_complete) and rank map.

    The factory must bind its native batch executor via bind_batch_executor()
    so that native communication predictions are replaced by physical transfers.
    """

    factory: object
    rank_mapping: dict


class CellRuntime(AsterMindRuntime):
    """Adapt the colleague's CellRequest/output/completion protocol to Runtime.

    Linear dependent turns stay on their selected Cell. Native Cells own their
    real block pools and prefix decisions; OrbitServe retains its metric cache.
    Energy remains the existing per-token admission estimate. Zero-output,
    cancellation and failed active compute Cells are outside this bridge.
    """

    serving_rate_model = "native_cell_external_shared_network"

    def __init__(self, cells, *, record_packets=False):
        super().__init__(cells, record_packets=record_packets)
        self._session_nodes = {}
        self.network = None
        self.migrations = None

    def initialize(self, engine, requests):
        if self.engine is not None:
            raise RuntimeError("create a new CellRuntime for each simulation")
        if engine.spec.strategy.shadow_dvfs_enabled:
            raise NotImplementedError(
                "native Cell execution does not model shadow DVFS"
            )
        if any(r["output_tokens"] < 1 for r in requests):
            raise NotImplementedError("native Cells require at least one output token")
        by_id = {r["request_id"]: r for r in requests}
        histories = set()
        for request in requests:
            session = request.get("session_id")
            parent = request.get("depends_on_request_id")
            if request.get("context_tokens_before_turn", 0) and parent is None:
                raise ValueError(
                    "native history reuse requires an explicit parent request"
                )
            if parent is not None and by_id[parent].get("session_id") != session:
                raise ValueError(
                    "native dependent turns must belong to the same session"
                )
            if session is not None:
                key = session, parent
                if key in histories:
                    raise ValueError(
                        "native Cells require a declared linear session history"
                    )
                histories.add(key)
        self.engine = engine
        self.timeline = GlobalTimeline(host=engine)
        self.jobs = {}
        endpoints, owners = {}, {}
        for node, deployment in self.cells.items():
            members = set(map(str, deployment.rank_mapping.values()))
            if (
                node not in engine.eligible
                or members != set(engine.members(node))
                or len(members) != len(deployment.rank_mapping)
            ):
                raise ValueError(
                    f"Cell mapping must cover its physical members: {node}"
                )
            if set(owners).intersection(members):
                raise ValueError("Cells cannot share physical members")
            endpoints.update({member: member for member in members})
            owners.update({member: node for member in members})
        self.network = RuntimeCellNetwork(
            engine, endpoints=endpoints, cell_members=owners
        )
        self.transfer_executor = CellTransferExecutor(self.timeline, self.network)
        from .cell_migration import NetworkKVMigration

        self.migrations = NetworkKVMigration(
            self.timeline,
            self.transfer_executor,
            {node: cell.rank_mapping for node, cell in self.cells.items()},
        )
        for node, deployment in self.cells.items():
            cell = deployment.factory(
                cell_id=node,
                loop=self.timeline,
                network=self.network,
                on_output=self._cell_output,
                on_complete=self._cell_complete,
            )
            self.sessions[node] = cell
            if cell.cell_id != node:
                raise ValueError("Cell factory returned an unexpected Cell identity")
            if hasattr(cell, "set_failure_callback"):
                cell.set_failure_callback(self._cell_failure)

    def accepts(self, node, request):
        return (
            node in self.sessions
            and self._session_nodes.get(request.get("session_id"), node) == node
        )

    def selected(self, job):
        session = job["request"].get("session_id")
        if session is not None:
            self._session_nodes.setdefault(session, job["node"])

    def candidate_load(self, node):
        if node not in self.sessions:
            return 1.0
        pending = self.sessions[node].snapshot()["pending_request_count"]
        return pending / (pending + 1)

    def submit(self, engine, job):
        self._admit_job(engine, job)
        request = job["request"]
        parent = request.get("depends_on_request_id")
        self.sessions[job["node"]].submit_ready_request(
            CellRequest(
                request["request_id"],
                engine.now,
                request["prompt_tokens"],
                request["output_tokens"],
                session_id=request.get("session_id"),
                context_id=request["request_id"]
                if request.get("session_id") is not None
                else None,
                parent_context_id=parent
                if request.get("session_id") is not None
                else None,
            )
        )

    def _cell_output(self, output):
        job = self.jobs[output.request_id]
        if output.cell_id != job["node"] or output.produced_time_s != self.engine.now:
            raise ValueError("Cell output must match its selected Cell and host time")
        generated = job.get("native_generated_tokens", 0)
        if (
            output.token_start != generated
            or generated + output.token_count > job["request"]["output_tokens"]
        ):
            raise ValueError("native Cell emitted an invalid token interval")
        job["native_generated_tokens"] = generated + output.token_count
        self._tokens(output.request_id, output.token_count, output.produced_time_s)

    def _cell_complete(self, result):
        if result.completion_time_s != self.engine.now:
            raise ValueError(
                "native Cell completion must be delivered at the host time"
            )
        job = self.jobs[result.request_id]
        if (
            result.cell_id != job["node"]
            or result.generated_tokens != job["request"]["output_tokens"]
            or job.get("native_generated_tokens", 0) != result.generated_tokens
        ):
            raise ValueError("native Cell completion emitted the wrong output tokens")
        if job["done"] or not job.get("astermind_first_token"):
            raise RuntimeError("invalid native Cell completion")
        if (
            not job["compute_ready_s"]
            <= result.first_execution_s
            <= result.completion_time_s
        ):
            raise ValueError("native Cell execution must follow actual input readiness")
        if any(
            not math.isfinite(value) or value < 0
            for value in (result.prefill_active_s, result.decode_active_s)
        ):
            raise ValueError(
                "native Cell active service times must be finite and nonnegative"
            )
        job.update(
            prefill_start_s=result.first_execution_s,
            prefill_service_s=result.prefill_active_s,
            decode_service_s=result.decode_active_s,
            decode_done_s=self.engine.now,
        )
        self.engine.log.append(
            {
                "time_s": self.engine.now,
                "kind": "cell_compute_complete",
                **asdict(result),
            }
        )
        self.engine.on_compute_complete(job, None)

    def _cell_failure(self, failure):
        raise RuntimeError(f"native Cell request failed: {failure}")

    def migrate(self, spec, source, target, history, *, activate_target=True):
        if self.migrations is None:
            raise RuntimeError("initialize the Cell runtime before migrating KV")

        def completed(record):
            session = history.get("session_id")
            if activate_target and session is not None:
                self._session_nodes[session] = target

        return self.migrations.start(
            spec,
            self.sessions[source],
            self.sessions[target],
            history,
            on_complete=completed,
        )

    def close(self):
        with ExitStack() as cleanup:
            for cell in self.sessions.values():
                cleanup.callback(cell.close)
            if self.migrations is not None:
                cleanup.callback(self.migrations.close)
            if self.network is not None:
                self.network.close()
