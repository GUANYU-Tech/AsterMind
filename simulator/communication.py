"""Scoped ProtoBridge operations on the same device/physical Runtime clock.

Surrogate calls reserve the participating satellite fabrics exclusively. Active
FlowSim transfers drain before a conflicting reservation; new transfers wait.
This deliberately coarse contention model retains resource occupancy without
pretending to reproduce the flow-level sharing skipped by this mode.
"""

from collections import deque
import bisect
from dataclasses import replace
import hashlib
import math
from pathlib import Path

from orbitserve.runtime.engine.cell_bridge import CellTransferEvent
from orbitserve.runtime.clock import stamp
from orbitserve.runtime.timeline import nanoseconds


class ScopedProtoBridge:
    def __init__(self, network, mode):
        if mode not in {"protobridge_satellite", "protobridge_cell"}:
            raise ValueError("unknown ProtoBridge scope")
        from src.kernel.predictors.protobridge import whitebox_surrogate_v2 as model

        self.model, self.network, self.mode = model, network, mode
        self.host, self.topology = network.host, network.topology
        self.source_sha256 = hashlib.sha256(
            Path(model.__file__).read_bytes()
        ).hexdigest()
        self.pending, self.active, self.records = [], {}, []
        self.flow_owners, self.identities = {}, set()
        self.requests = {}
        self.serial, self.draining = 0, False
        self._path_cache, self._path_epoch = {}, None
        network.subscribe(self._flow_done)

    def _owner(self, device):
        return self.topology.devices[device].satellite

    def _eligible(self, devices):
        owners = {self._owner(d) for d in devices}
        cells = {self.topology.satellite_to_cell[o] for o in owners}
        return len(cells) == 1 and (self.mode == "protobridge_cell" or len(owners) == 1)

    def _scope(self, devices):
        owners = {self._owner(d) for d in devices}
        if self.mode == "protobridge_cell":
            cell = self.topology.satellite_to_cell[next(iter(owners))]
            owners = {
                o for o, c in self.topology.satellite_to_cell.items() if c == cell
            }
        return owners

    def _flow_done(self, event):
        self.flow_owners.pop(event.id, None)
        self._drain()

    def submit(self, request, send):
        if self.network.closed:
            raise RuntimeError("Cell network is closed")
        if (
            not isinstance(request.id, str)
            or not request.id
            or request.id in self.identities
        ):
            raise ValueError("communication requires a unique nonempty ID")
        if (
            isinstance(request.size_bytes, bool)
            or request.size_bytes < 0
            or not math.isfinite(request.size_bytes)
        ):
            raise ValueError("communication bytes must be finite and nonnegative")
        ready = self.network.ready_time(request)
        self.network.resolve_pair(request.source, request.target)
        self.identities.add(request.id)
        self.requests[request.id] = request
        if ready > self.host.now:
            self.host.timeline.schedule(
                ready,
                33,
                "protobridge:ready",
                lambda: self._submit_ready(request, send)
                if not self.network.closed and request.id in self.requests
                else None,
            )
        else:
            self._submit_ready(request, send)
        return request.id

    def _submit_ready(self, request, send):
        devices = (request.source, request.target)
        internal = (
            request.origin == "multi_gpu" or request.operation == "input_token_ids"
        )
        if internal and request.size_bytes and self._eligible(devices):

            def completed(row):
                self.requests.pop(request.id, None)
                event = CellTransferEvent(
                    "completed",
                    request.id,
                    request.source,
                    request.target,
                    request.size_bytes,
                    request.origin,
                    request.operation,
                    request.correlation_id,
                    (),
                    completion_time_ns=nanoseconds(self.host.now),
                    time_ticks=str(self.host.now.ticks),
                )
                self.network.events.append(
                    {**event.model_dump(), "backend": "protobridge"}
                )
                for listener in tuple(self.network._listeners):
                    listener(event)

            self._enqueue(
                devices,
                "sendrecv",
                request.size_bytes,
                request.id,
                completed,
                transfer_id=request.id,
            )
            return

        def start():
            if request.id not in self.requests:
                return True
            # Read the real route at admission. Cell endpoints reserve their
            # fabrics for the complete transfer, including propagation/retries.
            owners = {self._owner(d) for d in devices}
            try:
                a, b = self.network.resolve_pair(*devices)
                _, legs = self.host.net.transfer(
                    a, b, self.host.now, request.size_bytes
                )
                owners.update(
                    self.host.net.owner[n]
                    for leg in legs
                    for n in (leg["source_node_id"], leg["target_node_id"])
                )
            except (ValueError, LookupError):
                pass  # The real transport retains its own unavailable-route policy.
            if any(owners & entry["owners"] for entry in self.active.values()):
                return False
            self.flow_owners[request.id] = owners
            if hasattr(request, "model_copy"):
                accepted = request.model_copy(
                    update={"ready_time_ns": math.ceil(self.network.now_ns)}
                )
            else:
                accepted = replace(request, ready_time_ns=nanoseconds(self.host.now), ready_time_ticks=str(self.host.now.ticks))
            self.requests.pop(request.id)
            send(accepted)
            return True

        self.pending.append(
            {
                "owners": {self._owner(d) for d in devices},
                "flow": start,
                "transfer_id": request.id,
            }
        )
        self._drain()

    def collective(self, cell_id, transfer, devices, record, on_complete):
        if not self._eligible(devices):
            return False
        if any(
            self.topology.satellite_to_cell[self._owner(d)] != cell_id for d in devices
        ):
            raise ValueError("collective escapes its declared Cell")
        operation = {
            "all_reduce": "allreduce",
            "all_gather": "allgather",
            "send_recv": "sendrecv",
        }.get(transfer.operation)
        # Ragged collectives use the executor's existing phase lowering; local
        # point-to-point pieces still use ProtoBridge at the configured scope.
        if operation is None:
            return False
        if transfer.bytes == 0:
            return False
        if operation != "sendrecv" and transfer.algorithm != "ring":
            raise ValueError(
                "this physical ProtoBridge adapter requires ring collectives"
            )
        if operation == "sendrecv":
            if transfer.bytes_semantics != "payload" or len(devices) != 2:
                raise ValueError("ProtoBridge PP needs two endpoints and payload bytes")
        elif transfer.bytes_semantics != "per_rank_input":
            raise ValueError("ProtoBridge collective needs per-rank input bytes")
        if operation == "allreduce" and transfer.reduction != "sum":
            raise ValueError("ProtoBridge ring supports sum reduction")
        amount = (
            transfer.bytes * len(devices)
            if operation == "allgather"
            else transfer.bytes
        )
        record["communication_backend"] = "protobridge"
        self._enqueue(
            devices,
            operation,
            amount,
            record["call_id"],
            lambda row: (record.update(surrogate=row), on_complete()),
        )
        return True

    def _enqueue(self, devices, operation, amount, identity, done, transfer_id=None):
        if amount <= 0 or not math.isfinite(amount):
            raise ValueError("ProtoBridge bytes must be finite and positive")
        self.serial += 1
        row = dict(
            id=f"surrogate-{self.serial}",
            correlation_id=identity,
            devices=list(devices),
            operation=operation,
            data_bytes=amount,
            mode=self.mode,
            source_sha256=self.source_sha256,
            ready_s=self.host.now,
            start_s=None,
            end_s=None,
            attempts=[],
            resource_policy="exclusive_participating_satellite_fabrics",
            parameter_source="current_device_routes_and_physical_link_budgets",
        )
        self.records.append(row)
        self.pending.append(
            dict(
                id=row["id"],
                owners=self._scope(devices),
                devices=devices,
                row=row,
                done=done,
                retries=0,
                transfer_id=transfer_id,
            )
        )
        self._drain()

    def cancel(self, identifiers, reason):
        for identifier in identifiers:
            request = self.requests.pop(identifier, None)
            if request is None:
                continue  # Already admitted FlowSim transfers use parent cancellation.
            for entry in tuple(self.pending):
                if entry.get("transfer_id") == identifier:
                    self.pending.remove(entry)
            for key, entry in tuple(self.active.items()):
                if entry.get("transfer_id") == identifier:
                    entry["row"].update(cancelled_s=self.host.now, failure=reason)
                    self.active.pop(key)
            event = CellTransferEvent(
                "failed",
                identifier,
                request.source,
                request.target,
                request.size_bytes,
                request.origin,
                request.operation,
                request.correlation_id,
                (),
                failure_time_ns=self.network.now_ns,
                reason=reason,
            )
            self.network.events.append(event.model_dump())
            for listener in tuple(self.network._listeners):
                listener(event)
        if not self.network.closed:
            self._drain()

    def _next_change(self):
        now = self.host.now
        return min(
            (time for time in self.host.net.environment_times if time > now),
            default=None,
        )

    def network_changed(self):
        # Manual controls and fractional fault boundaries may fall between
        # physical samples. Invalidate stale completions at that exact time.
        self._path_epoch = None
        for entry in tuple(self.active.values()):
            check = entry.get("check")
            if check is not None:
                self.host.timeline.schedule(
                    self.host.now, 35, "protobridge:changed", check
                )

    def _path_parameters(self, source, target, owners):
        net = self.host.net
        epoch = (
            net.revision,
            bisect.bisect_right(net.environment_times, self.host.now),
        )
        if epoch != self._path_epoch:
            self._path_cache.clear()
            self._path_epoch = epoch
        key = source, target, frozenset(owners)
        if key in self._path_cache:
            return self._path_cache[key]
        a, b = self.network.resolve_pair(source, target)
        adjacency = {}
        for edge in net.edges:
            if (
                net.owner[edge["endpoint_a"]] not in owners
                or net.owner[edge["endpoint_b"]] not in owners
            ):
                continue
            if net.link_at(edge["link_id"], self.host.now)["state"] != "UP":
                continue
            for x, y in (
                (edge["endpoint_a"], edge["endpoint_b"]),
                (edge["endpoint_b"], edge["endpoint_a"]),
            ):
                if (
                    net.owner[x] not in owners
                    or net.owner[y] not in owners
                    or not net.allows_hop(x, y)
                ):
                    continue
                attributes = net.directional_attributes(edge, x, y, self.host.now)
                rate = attributes["bandwidth_bps"]
                if rate > 0:
                    adjacency.setdefault(x, []).append((y, edge, rate))
        queue, seen = deque([(a, ())]), {a}
        while queue:
            node, hops = queue.popleft()
            if node == b:
                rates = [h[2] for h in hops]
                latency = sum(
                    net.link_at(h[1]["link_id"], self.host.now)["propagation_delay_s"]
                    for h in hops
                )
                value = min(rates), latency, [h[1]["link_id"] for h in hops]
                self._path_cache[key] = value
                return value
            for neighbor, edge, rate in adjacency.get(node, ()):
                if neighbor not in seen and net.route_node_allowed(neighbor, a, b):
                    seen.add(neighbor)
                    queue.append((neighbor, (*hops, (neighbor, edge, rate))))
        raise ValueError(
            "ProtoBridge has no available route inside its configured scope"
        )

    def _predict(self, entry):
        devices, row = entry["devices"], entry["row"]
        pairs = (
            [(devices[0], devices[1])]
            if row["operation"] == "sendrecv"
            else list(zip(devices, (*devices[1:], devices[0])))
        )
        paths = [self._path_parameters(a, b, entry["owners"]) for a, b in pairs]
        # Shared links are not multiplied into independent ring capacities.
        multiplicity = {}
        for _, _, path in paths:
            for link in path:
                multiplicity[link] = multiplicity.get(link, 0) + 1
        bandwidth = min(p[0] for p in paths) / max(multiplicity.values()) / 8e6
        delay_us = max(p[1] for p in paths) * 1e6
        parameters = dict(
            packet_bw_bytes_per_us=bandwidth,
            ring_bw_bytes_per_us=bandwidth,
            tree_bw_per_level_bytes_per_us=bandwidth,
            nvls_bw_bytes_per_us=bandwidth,
            fullmesh_bw_bytes_per_us=bandwidth,
            link_latency_us=delay_us,
            credit_bdp_ring_packets=0,
            credit_bdp_tree_packets=0,
            startup_us=self.host.protobridge_startup_us,
            nvls_startup_us=self.host.protobridge_startup_us,
            wire_efficiency=1.0,
        )
        predictor = self.model.TheoryDerivedSurrogate(**parameters)
        duration = (
            predictor.predict(
                data_bytes=row["data_bytes"],
                num_gpus=len(devices),
                algo="ring",
                credits=1,
                collective=row["operation"],
                competing_bw=0,
                include_credit=False,
                include_fec=False,
                include_retransmission=False,
                include_contention=False,
            )
            / 1e6
        )
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("ProtoBridge returned an invalid duration")
        return duration, parameters, paths

    def _drain(self):
        if self.draining or self.network.closed:
            return
        self.draining = True
        try:
            blocked = set()
            for entry in tuple(self.pending):
                if entry["owners"] & blocked:
                    continue
                if "flow" in entry:
                    if entry["flow"]():
                        self.pending.remove(entry)
                    else:
                        blocked.update(entry["owners"])
                    continue
                busy = any(
                    entry["owners"] & owners for owners in self.flow_owners.values()
                ) or any(
                    entry["owners"] & active["owners"]
                    for active in self.active.values()
                )
                if busy:
                    blocked.update(entry["owners"])
                    continue
                self.pending.remove(entry)
                self.active[entry["id"]] = entry
                self._start(entry)
        finally:
            self.draining = False

    def _start(self, entry):
        generation = entry.get("generation", 0) + 1
        entry["generation"] = generation

        def valid():
            return (
                self.active.get(entry["id"]) is entry
                and entry["generation"] == generation
            )

        def retry():
            if valid():
                self._start(entry)

        entry["check"] = retry
        try:
            duration, parameters, paths = self._predict(entry)
        except ValueError:
            change = self._next_change()
            if (
                change is None
                or self.host.net.spec.failure_policy != "retry_next_sample"
            ):
                raise
            entry["retries"] += 1
            if entry["retries"] > self.host.net.spec.max_transfer_retries:
                raise ValueError("ProtoBridge route recovery retry limit exceeded")
            self.host.timeline.schedule(change, 35, "protobridge:retry", retry)
            return
        row = entry["row"]
        if row["start_s"] is None:
            row["start_s"] = self.host.now
        attempt = dict(
            start_s=self.host.now,
            duration_s=duration,
            parameters=parameters,
            paths=paths,
        )
        stamp(attempt)
        stamp(row)
        row["attempts"].append(attempt)
        end = self.host.now + duration

        def finish():
            if not valid():
                return
            try:
                current = self._predict(entry)[1:]
            except ValueError:
                current = None
            if current != (parameters, paths):
                attempt["invalidated_s"] = self.host.now
                self._start(entry)
                return
            row["end_s"] = self.host.now
            stamp(row)
            self.active.pop(entry["id"])
            self.host.log.append(
                dict(
                    kind="surrogate_complete",
                    time_s=self.host.now,
                    surrogate_id=entry["id"],
                    communication_mode=self.mode,
                    correlation_id=row["correlation_id"],
                    operation=row["operation"],
                    data_bytes=row["data_bytes"],
                    devices=row["devices"],
                )
            )
            entry["done"](row)
            self._drain()

        def check():
            if not valid():
                return
            try:
                current = self._predict(entry)[1:]
            except ValueError:
                current = None
            if current != (parameters, paths):
                attempt["invalidated_s"] = self.host.now
                self._start(
                    entry
                )  # Restart against the changed physical state; never complete early.
            # The existing scheduled boundary/completion remains valid when
            # this notification does not change any participating route.

        entry["check"] = check

        def boundary():
            check()
            if valid():
                schedule()

        def schedule():
            change = self._next_change()
            at = min(end, change) if change is not None else end
            self.host.timeline.schedule(
                at,
                35,
                "protobridge:complete" if at == end else "protobridge:boundary",
                finish if at == end else boundary,
            )

        schedule()
