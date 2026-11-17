"""Request scheduling across Cells; native batch scheduling stays in compute."""

def select_cell(self, request):
    candidates = []
    policy = getattr(self.config, 'cell_selection', 'least_loaded')
    context = None
    for c in self.config.cells:
        if request.cell is not None and request.cell != c.id:
            continue
        if (
            request.cell is None
            and request.session_id in self._session_cells
            and self._session_cells[request.session_id] != c.id
        ):
            continue
        if request.input_tokens + request.output_tokens > c.max_model_len:
            continue
        try:
            path = self.network.route(request.user_device, c.ingress_device)
            if policy == 'estimated_arrival':
                if context is None:
                    context = self.network.estimation_context()
                path = self.network.estimate_transfer(
                    request.user_device, c.ingress_device, request.input_bytes, context=context)
        except ValueError:
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
        assigned = sum(
            s.get("cell") == c.id and s["status"] in ("input", "computing")
            for s in self.state.values()
        )
        # Explicit deterministic first policy. Reservation/load stays at
        # whole requests; native vLLM decides actual admission and KV.
        distance = path['routing_distance_km']
        if policy == 'nearest':
            score = (distance, assigned, c.id)
        elif policy == 'estimated_arrival':
            score = (assigned, path['estimated_transfer_ns'], c.id)
        else:
            score = (assigned, distance, c.id)
        candidates.append(score)
    if not candidates:
        raise RuntimeError(f"No reachable Cell for {request.id}")
    return self._cell_configs[min(candidates)[2]]

