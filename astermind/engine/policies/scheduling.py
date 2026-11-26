"""Request scheduling across Cells; native batch scheduling stays in compute."""

from math import dist

PER_TURN_POLICIES = frozenset(("nearest_per_turn", "least_loaded_per_turn"))
CASE_STUDY_POLICIES = PER_TURN_POLICIES | {"session_fixed"}


class NoReachableCell(RuntimeError):
    pass


def case_study_policy(policy):
    return isinstance(policy, str) and policy in CASE_STUDY_POLICIES


def _geometric_distance(self, source, target, path):
    net = self.network.host.net
    a, b = self.network.resolve_pair(source, target)
    first, last = net.position(a, self.loop.now_s), net.position(b, self.loop.now_s)
    return dist(first, last) if first is not None and last is not None else path["routing_distance_km"]


def select_cell(self, request):
    if getattr(self.config, "fast_orbital_route_frames", None) is not None and case_study_policy(self.config.cell_selection):
        return select_orbital_cell(self, request, self.config.cell_selection)
    candidates = []
    policy = getattr(self.config, 'cell_selection', 'least_loaded')
    if policy == "session_fixed" and request.cell is not None:
        bound = self._session_cells.get(request.session_id)
        if bound is not None and bound != request.cell:
            raise ValueError("explicit Cell conflicts with the session's existing Cell")
    if isinstance(policy, str):
        order = {
            "least_loaded": ("load", "distance"),
            "nearest": ("distance", "load"),
            "estimated_arrival": ("load", "arrival"),
            "nearest_per_turn": ("distance",),
            "least_loaded_per_turn": ("load", "distance"),
            "session_fixed": ("distance",),
        }[policy]
    else:
        order = tuple(metric for metric in (policy.primary, policy.secondary) if metric)
    context = None
    case_study = case_study_policy(policy)
    for c in self.config.cells:
        if request.cell is not None and request.cell != c.id:
            continue
        if (
            request.cell is None
            and (not isinstance(policy, str) or policy not in PER_TURN_POLICIES)
            and request.session_id in self._session_cells
            and self._session_cells[request.session_id] != c.id
        ):
            continue
        if request.input_tokens + request.output_tokens > c.max_model_len:
            continue
        try:
            path = self.network.route(request.user_device, c.ingress_device)
            if 'arrival' in order:
                if context is None:
                    context = self.network.estimation_context()
                path = self.network.estimate_transfer(
                    request.user_device, c.ingress_device, request.input_bytes, context=context)
        except ValueError:
            if case_study:
                continue
            if (
                self.topology.scenario.link_policy is None
                and not self.topology.scenario.dynamic_external
            ):
                continue
            try:
                self.network.candidate_path(request.user_device, c.ingress_device)
            except ValueError:
                continue
            # Acquisition is handled by the communication runtime. A
            # structural candidate is eligible to wait, never ready early.
            path = {"routing_distance_km": float("inf"), "estimated_transfer_ns": float("inf")}
        assigned = (
            self._case_study_load[c.id] if case_study else
            sum(s.get("cell") == c.id and s["status"] in ("input", "computing")
                for s in self.state.values())
        ) if "load" in order else 0
        # Explicit deterministic first policy. Reservation/load stays at
        # whole requests; native vLLM decides actual admission and KV.
        distance = (_geometric_distance(self, request.user_device, c.ingress_device, path)
                    if case_study else path["routing_distance_km"])
        metrics = {"distance": distance, "load": assigned,
                   "arrival": path.get("estimated_transfer_ns")}
        candidates.append((tuple(metrics[key] for key in order), c.id))
    if not candidates:
        error = NoReachableCell if case_study else RuntimeError
        raise error(f"No reachable Cell for {request.id}")
    return self._cell_configs[min(candidates)[1]]


def select_orbital_cell(engine, request, policy):
    from astermind.engine.comm.frame_surrogate import NoRouteError

    bound = engine._session_cells.get(request.session_id) if policy == "session_fixed" else None
    if bound is not None and request.cell not in (None, bound):
        raise ValueError("explicit Cell conflicts with the session's existing Cell")
    target = bound or request.cell
    candidates = []
    for cell in engine.config.cells:
        if target is not None and cell.id != target:
            continue
        if request.input_tokens + request.output_tokens > cell.max_model_len:
            continue
        distance = 0. if target is not None else _geometric_distance(
            engine, request.user_device, cell.ingress_device, {"routing_distance_km": float("inf")})
        load = engine._case_study_load[cell.id] if policy == "least_loaded_per_turn" else 0
        candidates.append((load, distance, cell.id))
    for _, _, cell_id in sorted(candidates):
        cell = engine._cell_configs[cell_id]
        try:
            engine.network.estimate_transfer(request.user_device, cell.ingress_device, request.input_bytes)
        except NoRouteError:
            continue
        return cell
    raise NoReachableCell(f"No reachable Cell for {request.id}")

