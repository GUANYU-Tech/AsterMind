"""Request scheduling across Cells; native batch scheduling stays in compute."""

def select_cell(self, request):
    candidates = []
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
            path = self.network.route(
                request.user_device, c.ingress_device, request.input_bytes
            )
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
            path = {"isolated_estimate_ns": float("inf")}
        assigned = sum(
            s.get("cell") == c.id and s["status"] in ("input", "computing")
            for s in self.state.values()
        )
        # Explicit deterministic first policy. Reservation/load stays at
        # whole requests; native vLLM decides actual admission and KV.
        candidates.append((assigned, path["isolated_estimate_ns"], c.id))
    if not candidates:
        raise RuntimeError(f"No reachable Cell for {request.id}")
    return self._cell_configs[min(candidates)[2]]

