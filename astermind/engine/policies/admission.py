"""Input, predecessor and KV admission gates on the existing Engine state."""
from bisect import bisect_right

from astermind.engine.clock import Time
from astermind.engine.timeline import nanoseconds
from .scheduling import NoReachableCell, PER_TURN_POLICIES, case_study_policy


def legacy_cell_history(store, request):
    """Match the archived Case Study's immediate parent at the selected Cell."""
    session, parent = request.session_id, request.parent_context_id
    if session is None:
        return session, parent
    tip = store.tips.get(session)
    if ((tip is not None and tip != parent)
            or any(owner == session for owner, _ in store.pending)):
        # Give an out-of-order local turn a fresh namespace so an older
        # resident prefix cannot be adopted by the native Scheduler.
        return f"{session}|cold:{request.id}", None
    return session, tip


def admit_request(self, request):
    from astermind.engine.cell_events import CellRequest

    if self.state[request.id]["status"] == "failed":
        return
    policy = getattr(self.config, "cell_selection", "least_loaded")
    case_study = case_study_policy(policy)
    per_turn = case_study and policy in PER_TURN_POLICIES
    try:
        cell_config = self._select(request)
    except (ValueError, RuntimeError) as error:
        if case_study and isinstance(error, NoReachableCell):
            times = self.host.net.environment_times
            index = bisect_right(times, self.loop.now_s)
            if index < len(times) and times[index] < self.host.end:
                self.loop.schedule(Time(times[index]), 1, "request:retry-cell-selection",
                                   lambda: admit_request(self, request))
                return
        self._request_failed(request.id, 0, "selection_failed",
                             dict(stage="cell_selection", message=str(error)))
        return
    self._update_request(request.id, "cell_selected", status="input", cell=cell_config.id)
    self._record_request("request_arrival", request.id)
    if request.session_id is not None and not per_turn:
        self._session_cells[request.session_id] = cell_config.id
    self.decisions.append(
        {
            "request": request.id,
            "cell": cell_config.id,
            "time_ns": request.arrival_ns,
            "policy": policy if case_study else "least_assigned_then_isolated_ingress_then_id",
        }
    )

    attempt = 0
    internal_id = request.id
    required = tuple(m for m in self.config.migrations if m.before_request == request.id)
    parent = self.config.predecessor(request)
    readiness = {"input": False, "kv": not required, "admitted": False,
                 "parent": parent is None or 'compute_complete_ns' in self.state[parent]}
    pending_kv = {m.id for m in required}
    def kv_ready(migration_id, error=None):
        if not self._attempt_current(request.id, attempt):return
        if error is not None:
            self._request_failed(request.id,attempt,'kv_failed',error)
            return
        pending_kv.discard(migration_id)
        readiness['kv'] = not pending_kv
        if readiness['kv']:
            self._update_request(request.id, 'kv_ready',
                kv_ready_ns=nanoseconds(self.loop.now_s), kv_ready_ticks=str(self.loop.now_s.ticks))
        ready()
    def ready():
        if not readiness['input'] or not readiness['kv'] or not readiness['parent'] or readiness['admitted']:
            return
        readiness['admitted'] = True
        if not self._attempt_current(request.id,attempt):
            return
        if cell_config.id in self._failed_cells or getattr(self.cells[cell_config.id],'_closed',False):
            self._request_failed(request.id,attempt,'cell_unavailable_at_admission',
                dict(cell_id=cell_config.id,reason='target Cell stopped before input arrived'))
            return
        self._update_request(request.id, "input_delivered", status="computing",
                             cell_ready_ns=nanoseconds(self.loop.now_s),
                             cell_ready_ticks=str(self.loop.now_s.ticks))
        self._record_request('cell_input_ready', request.id)
        # The shared history describes content. Only the selected Scheduler
        # can find resident pages, including earlier turns at this Cell.
        cell = self.cells[cell_config.id]
        session, context_parent = request.session_id, request.parent_context_id
        if getattr(self, "_legacy_cell_local_history", False):
            session, context_parent = legacy_cell_history(cell.history_store, request)
        cell_request = CellRequest(internal_id, self.loop.now_s, request.input_tokens, request.output_tokens,
            prefix_hit_tokens=0 if per_turn else request.prefix_hit_tokens,
            session_id=session, context_id=request.context_id, parent_context_id=context_parent)
        try:
            cell.submit_ready_request(cell_request)
        except (ValueError,RuntimeError) as error:
            self._request_failed(request.id,attempt,'cell_admission_failed',
                dict(cell_id=cell_config.id,error_type=type(error).__name__,reason=str(error)))
    def parent_ready(error=None):
        if not self._attempt_current(request.id, attempt):return
        if error is not None:
            self._request_failed(request.id,attempt,'parent_failed',error)
            return
        readiness['parent'] = True
        self._update_request(request.id, 'parent_ready',
            parent_ready_ns=self.state[parent]['compute_complete_ns'],
            parent_ready_ticks=self.state[parent]['compute_complete_ticks'])
        ready()
    if parent is not None and readiness['parent']:
        parent_ready()
    if parent is not None and not readiness['parent']:
        if self.state[parent]['status'] == 'failed':
            parent_ready(self.state[parent]['failure'])
        else:
            self._parent_waiters.setdefault(parent, []).append(parent_ready)
    for migration in required:
        record = self.migrations.records.get(migration.id)
        if record and record['state'] == 'completed':
            kv_ready(migration.id)
        elif record and record['state'] in ('failed', 'capacity_blocked', 'source_release_failed'):
            kv_ready(migration.id, record)
        else:
            self._kv_waiters.setdefault(migration.id, []).append(kv_ready)
    def input_ready():
        readiness['input'] = True
        self._update_request(request.id, 'input_ready',
            input_ready_ns=nanoseconds(self.loop.now_s), input_ready_ticks=str(self.loop.now_s.ticks))
        ready()
    if not self._attempt_current(request.id, attempt):
        return
    self._submit(request.user_device, cell_config.ingress_device, request.input_bytes,
                 "request_input", request.id, input_ready)

